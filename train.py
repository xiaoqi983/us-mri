#!/usr/bin/env python
import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.utils.data import DataLoader

from dataset import RemindSliceDataset, discover_pairs
from loss import DDICLoss
from model import DualDDPMCorrelationModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DDIC dual-DDPM for intraoperative USpreimri -> intraop 2DAXT2BLADE synthesis.")
    parser.add_argument("--data-root", type=str, required=True, help="Preprocessed root containing case folders or manifest.")
    parser.add_argument("--save-dir", type=str, default="./checkpoints")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--samples-per-volume", type=int, default=64)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lambda-corr", type=float, default=0.1)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--base-channels", type=int, default=64)
    parser.add_argument("--num-res-blocks", type=int, default=2)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--resume", type=str, default=None)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def split_pairs(pairs: Sequence[Dict[str, str]], val_ratio: float, seed: int) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    pairs = list(pairs)
    rng = random.Random(seed)
    rng.shuffle(pairs)
    if len(pairs) == 1:
        return pairs, pairs
    val_count = max(1, int(len(pairs) * val_ratio))
    train_pairs = pairs[val_count:]
    val_pairs = pairs[:val_count]
    if not train_pairs:
        train_pairs = val_pairs
    return train_pairs, val_pairs


def build_dataloaders(args: argparse.Namespace) -> Tuple[DataLoader, DataLoader]:
    pairs = discover_pairs(Path(args.data_root).resolve())
    train_pairs, val_pairs = split_pairs(pairs, args.val_ratio, args.seed)

    train_dataset = RemindSliceDataset(
        preprocessed_root=args.data_root,
        image_size=args.image_size,
        samples_per_volume=args.samples_per_volume,
        augment=True,
        pairs=train_pairs,
        seed=args.seed,
    )
    val_dataset = RemindSliceDataset(
        preprocessed_root=args.data_root,
        image_size=args.image_size,
        samples_per_volume=max(8, args.samples_per_volume // 2),
        augment=False,
        pairs=val_pairs,
        seed=args.seed,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    return train_loader, val_loader


def move_batch_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    result = {
        "us": batch["us"].to(device, non_blocking=True),
        "mr": batch["mr"].to(device, non_blocking=True),
        "overlap_mask": batch["overlap_mask"].to(device, non_blocking=True),
    }
    if "preop_mr" in batch:
        result["preop_mr"] = batch["preop_mr"].to(device, non_blocking=True)
    return result


def reduce_metrics(running: Dict[str, float], losses: Dict[str, torch.Tensor], count: int) -> None:
    for key, value in losses.items():
        running[key] = running.get(key, 0.0) + float(value.detach().item()) * count


def finalize_metrics(running: Dict[str, float], num_samples: int) -> Dict[str, float]:
    return {key: value / max(num_samples, 1) for key, value in running.items()}


def save_checkpoint(
    save_path: Path,
    epoch: int,
    model: DualDDPMCorrelationModel,
    optimizer: AdamW,
    scaler: GradScaler,
    args: argparse.Namespace,
    best_val: float,
) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scaler_state": scaler.state_dict(),
            "best_val_loss": best_val,
            "model_config": {
                "timesteps": args.timesteps,
                "base_channels": args.base_channels,
                "channel_mults": [1, 2, 4, 8],
                "num_res_blocks": args.num_res_blocks,
            },
            "train_config": vars(args),
        },
        save_path,
    )


def load_checkpoint(
    ckpt_path: Path,
    model: DualDDPMCorrelationModel,
    optimizer: AdamW,
    scaler: GradScaler,
    device: torch.device,
) -> Tuple[int, float]:
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    scaler.load_state_dict(checkpoint["scaler_state"])
    start_epoch = int(checkpoint["epoch"]) + 1
    best_val = float(checkpoint.get("best_val_loss", float("inf")))
    return start_epoch, best_val


def run_epoch(
    model: DualDDPMCorrelationModel,
    criterion: DDICLoss,
    data_loader: DataLoader,
    optimizer: AdamW,
    scaler: GradScaler,
    device: torch.device,
    grad_clip: float,
    train: bool,
) -> Dict[str, float]:
    model.train(train)
    criterion.train(train)
    running: Dict[str, float] = {}
    num_samples = 0

    for batch in data_loader:
        batch = move_batch_to_device(batch, device)
        us = batch["us"]
        mr = batch["mr"]
        preop_mr = batch.get("preop_mr")
        batch_size = us.size(0)

        if train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train):
            with autocast(enabled=device.type == "cuda"):
                outputs = model.forward_train(us, mr, preop_mr=preop_mr)
                losses = criterion(
                    us_pred_noise=outputs["us"]["pred_noise"],
                    us_true_noise=outputs["us"]["noise"],
                    mr_pred_noise=outputs["mr"]["pred_noise"],
                    mr_true_noise=outputs["mr"]["noise"],
                    x0_hat_mr=outputs["mr"]["x0_hat"],
                    us_condition=us,
                    overlap_mask=batch["overlap_mask"],
                )

            if train:
                scaler.scale(losses["total_loss"]).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()

        reduce_metrics(running, losses, batch_size)
        num_samples += batch_size

    return finalize_metrics(running, num_samples)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = Path(args.save_dir).resolve()
    save_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader = build_dataloaders(args)

    model = DualDDPMCorrelationModel(
        timesteps=args.timesteps,
        base_channels=args.base_channels,
        channel_mults=(1, 2, 4, 8),
        num_res_blocks=args.num_res_blocks,
    ).to(device)
    criterion = DDICLoss(lambda_corr=args.lambda_corr).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = GradScaler(enabled=device.type == "cuda")

    start_epoch = 1
    best_val = float("inf")
    if args.resume:
        start_epoch, best_val = load_checkpoint(Path(args.resume).resolve(), model, optimizer, scaler, device)

    with open(save_dir / "train_config.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    for epoch in range(start_epoch, args.epochs + 1):
        train_metrics = run_epoch(
            model=model,
            criterion=criterion,
            data_loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            grad_clip=args.grad_clip,
            train=True,
        )
        val_metrics = run_epoch(
            model=model,
            criterion=criterion,
            data_loader=val_loader,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            grad_clip=args.grad_clip,
            train=False,
        )

        log_line = (
            f"Epoch {epoch:03d} | "
            f"train_total={train_metrics['total_loss']:.6f} "
            f"train_mr={train_metrics['mr_denoise_loss']:.6f} "
            f"train_corr={train_metrics['corr_loss']:.6f} | "
            f"val_total={val_metrics['total_loss']:.6f} "
            f"val_mr={val_metrics['mr_denoise_loss']:.6f} "
            f"val_corr={val_metrics['corr_loss']:.6f}"
        )
        print(log_line)

        latest_path = save_dir / "latest.pt"
        save_checkpoint(latest_path, epoch, model, optimizer, scaler, args, best_val)

        if val_metrics["total_loss"] < best_val:
            best_val = val_metrics["total_loss"]
            save_checkpoint(save_dir / "best.pt", epoch, model, optimizer, scaler, args, best_val)

        if epoch % args.save_every == 0:
            save_checkpoint(save_dir / f"epoch_{epoch:03d}.pt", epoch, model, optimizer, scaler, args, best_val)


if __name__ == "__main__":
    main()
