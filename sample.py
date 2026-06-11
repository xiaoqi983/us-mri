#!/usr/bin/env python
import argparse
import json
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch

from dataset import resize_tensor
from model import DualDDPMCorrelationModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample T2w MRI from 2.5D ioUS condition using trained DDIC.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained checkpoint.")
    parser.add_argument("--output-dir", type=str, default="./samples")
    parser.add_argument("--preprocessed-root", type=str, default=None, help="Root directory containing case folders.")
    parser.add_argument("--subject-id", type=str, default=None, help="Case identifier under preprocessed root.")
    parser.add_argument("--us-path", type=str, default=None, help="Direct path to preprocessed us.npy.")
    parser.add_argument("--mr-path", type=str, default=None, help="Optional path to preprocessed mr.npy for reference saving.")
    parser.add_argument("--slice-index", type=int, required=True, help="Center slice index i used for [i-1, i, i+1].")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--sampling", type=str, default="ddim", choices=["ddpm", "ddim"])
    parser.add_argument("--ddim-steps", type=int, default=100)
    parser.add_argument("--eta", type=float, default=0.0)
    parser.add_argument("--device", type=str, default=None, help="cuda / cpu. Defaults to auto.")
    return parser.parse_args()


def resolve_case_paths(args: argparse.Namespace) -> Tuple[Path, Optional[Path], str]:
    if args.us_path is not None:
        us_path = Path(args.us_path).resolve()
        mr_path = Path(args.mr_path).resolve() if args.mr_path is not None else None
        subject_id = args.subject_id or us_path.parent.name
        return us_path, mr_path, subject_id

    if args.preprocessed_root is None or args.subject_id is None:
        raise ValueError("Either provide --us-path or provide both --preprocessed-root and --subject-id.")

    case_dir = Path(args.preprocessed_root).resolve() / args.subject_id
    us_path = case_dir / "us.npy"
    mr_path = case_dir / "mr.npy"
    if not us_path.exists():
        raise FileNotFoundError(f"Missing US volume: {us_path}")
    if not mr_path.exists():
        mr_path = None
    return us_path, mr_path, args.subject_id


def load_us_triplet(us_path: Path, slice_index: int, image_size: int) -> Tuple[torch.Tensor, int]:
    us_volume = np.load(us_path).astype(np.float32)
    if us_volume.ndim != 3:
        raise ValueError(f"Expected 3D US volume, got {us_volume.ndim}D")
    if us_volume.shape[0] < 3:
        raise ValueError("US volume has too few slices for 2.5D sampling.")

    slice_index = max(1, min(slice_index, us_volume.shape[0] - 2))
    us_slice = torch.from_numpy(us_volume[slice_index - 1 : slice_index + 2]).float()
    us_slice = resize_tensor(us_slice, image_size)
    return us_slice.unsqueeze(0), slice_index


def load_mr_reference(mr_path: Optional[Path], slice_index: int, image_size: int) -> Optional[np.ndarray]:
    if mr_path is None or not mr_path.exists():
        return None
    mr_volume = np.load(mr_path).astype(np.float32)
    if mr_volume.ndim != 3:
        return None
    slice_index = max(0, min(slice_index, mr_volume.shape[0] - 1))
    mr_slice = torch.from_numpy(mr_volume[slice_index : slice_index + 1]).float()
    mr_slice = resize_tensor(mr_slice, image_size)
    return mr_slice.squeeze(0).cpu().numpy()


def build_model_from_checkpoint(checkpoint: dict, device: torch.device) -> DualDDPMCorrelationModel:
    model_config = checkpoint.get("model_config", {})
    model = DualDDPMCorrelationModel(
        timesteps=int(model_config.get("timesteps", 1000)),
        base_channels=int(model_config.get("base_channels", 64)),
        channel_mults=tuple(model_config.get("channel_mults", [1, 2, 4, 8])),
        num_res_blocks=int(model_config.get("num_res_blocks", 2)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model


def main() -> None:
    args = parse_args()
    device_name = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    us_path, mr_path, subject_id = resolve_case_paths(args)
    us_input, slice_index = load_us_triplet(us_path, args.slice_index, args.image_size)
    us_input = us_input.to(device)

    checkpoint = torch.load(Path(args.checkpoint).resolve(), map_location=device)
    model = build_model_from_checkpoint(checkpoint, device)

    with torch.no_grad():
        generated = model.sample(
            us=us_input,
            sampling=args.sampling,
            ddim_steps=args.ddim_steps,
            eta=args.eta,
        )

    generated_np = generated.squeeze(0).squeeze(0).cpu().numpy().astype(np.float32)
    us_np = us_input.squeeze(0).cpu().numpy().astype(np.float32)
    mr_ref = load_mr_reference(mr_path, slice_index, args.image_size)

    prefix = f"{subject_id}_slice_{slice_index:04d}"
    np.save(output_dir / f"{prefix}_generated_mr.npy", generated_np)
    np.save(output_dir / f"{prefix}_us_input.npy", us_np)
    if mr_ref is not None:
        np.save(output_dir / f"{prefix}_reference_mr.npy", mr_ref.astype(np.float32))

    with open(output_dir / f"{prefix}_meta.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "subject_id": subject_id,
                "slice_index": slice_index,
                "sampling": args.sampling,
                "ddim_steps": args.ddim_steps,
                "eta": args.eta,
                "checkpoint": str(Path(args.checkpoint).resolve()),
                "us_path": str(us_path),
                "mr_path": str(mr_path) if mr_path is not None else None,
            },
            f,
            indent=2,
        )

    print(f"Saved generated MRI to {output_dir / f'{prefix}_generated_mr.npy'}")


if __name__ == "__main__":
    main()
