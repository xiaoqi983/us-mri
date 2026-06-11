import csv
import math
import random
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


def load_pairs_from_manifest(manifest_path: Path) -> List[Dict[str, str]]:
    pairs: List[Dict[str, str]] = []
    with open(manifest_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pairs.append(
                {
                    "subject_id": row["subject_id"],
                    "us_path": row["us_path"],
                    "mr_path": row["mr_path"],
                    "mask_path": row.get("mask_path", ""),
                    "meta_path": row.get("meta_path", ""),
                }
            )
    return pairs


def discover_pairs(preprocessed_root: Path) -> List[Dict[str, str]]:
    manifest_path = preprocessed_root / "preprocessed_pairs.csv"
    if manifest_path.exists():
        return load_pairs_from_manifest(manifest_path)

    pairs: List[Dict[str, str]] = []
    for case_dir in sorted(preprocessed_root.iterdir()):
        if not case_dir.is_dir():
            continue
        us_path = case_dir / "us.npy"
        mr_path = case_dir / "mr.npy"
        if us_path.exists() and mr_path.exists():
            pairs.append(
                {
                    "subject_id": case_dir.name,
                    "us_path": str(us_path),
                    "mr_path": str(mr_path),
                    "mask_path": str(case_dir / "overlap_mask.npy") if (case_dir / "overlap_mask.npy").exists() else "",
                    "meta_path": str(case_dir / "meta.json") if (case_dir / "meta.json").exists() else "",
                }
            )
    if not pairs:
        raise RuntimeError(f"No preprocessed case pairs found under {preprocessed_root}")
    return pairs


def resize_tensor(image: torch.Tensor, size: int = 256) -> torch.Tensor:
    return F.interpolate(
        image.unsqueeze(0),
        size=(size, size),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)


def apply_rotation(us_slice: torch.Tensor, mr_slice: torch.Tensor, angle_deg: float) -> Tuple[torch.Tensor, torch.Tensor]:
    theta = math.radians(angle_deg)
    affine = torch.tensor(
        [
            [math.cos(theta), -math.sin(theta), 0.0],
            [math.sin(theta), math.cos(theta), 0.0],
        ],
        dtype=us_slice.dtype,
        device=us_slice.device,
    ).unsqueeze(0)

    grid = F.affine_grid(affine, size=(1, us_slice.size(0), us_slice.size(1), us_slice.size(2)), align_corners=False)
    us_rotated = F.grid_sample(us_slice.unsqueeze(0), grid, mode="bilinear", padding_mode="border", align_corners=False)

    mr_grid = F.affine_grid(affine, size=(1, mr_slice.size(0), mr_slice.size(1), mr_slice.size(2)), align_corners=False)
    mr_rotated = F.grid_sample(mr_slice.unsqueeze(0), mr_grid, mode="bilinear", padding_mode="border", align_corners=False)
    return us_rotated.squeeze(0), mr_rotated.squeeze(0)


class RemindSliceDataset(Dataset):
    def __init__(
        self,
        preprocessed_root: str,
        image_size: int = 256,
        samples_per_volume: int = 64,
        augment: bool = True,
        pairs: Optional[Sequence[Dict[str, str]]] = None,
        seed: int = 42,
        min_mask_ratio: float = 0.01,
    ) -> None:
        self.preprocessed_root = Path(preprocessed_root).resolve()
        self.image_size = image_size
        self.samples_per_volume = samples_per_volume
        self.augment = augment
        self.rng = random.Random(seed)
        self.min_mask_ratio = min_mask_ratio
        self.cases = list(pairs) if pairs is not None else discover_pairs(self.preprocessed_root)

        self.us_volumes: List[np.ndarray] = []
        self.mr_volumes: List[np.ndarray] = []
        self.mask_volumes: List[np.ndarray] = []
        self.valid_indices: List[List[int]] = []
        for case in self.cases:
            us = np.load(case["us_path"]).astype(np.float32)
            mr = np.load(case["mr_path"]).astype(np.float32)
            mask_path = case.get("mask_path", "")
            meta_path = case.get("meta_path", "")
            overlap_mask = np.load(mask_path).astype(np.float32) if mask_path and Path(mask_path).exists() else np.ones_like(us, dtype=np.float32)
            if us.shape != mr.shape:
                raise ValueError(f"US/MR shape mismatch for {case['subject_id']}: {us.shape} vs {mr.shape}")
            if overlap_mask.shape != us.shape:
                raise ValueError(f"Mask shape mismatch for {case['subject_id']}: {overlap_mask.shape} vs {us.shape}")
            if us.ndim != 3:
                raise ValueError(f"Expected 3D volume for {case['subject_id']}, got {us.ndim}D")
            if us.shape[0] < 3:
                raise ValueError(f"Volume {case['subject_id']} has too few slices for 2.5D sampling.")
            self.us_volumes.append(us)
            self.mr_volumes.append(mr)
            self.mask_volumes.append(overlap_mask)

            valid_indices = []
            if meta_path and Path(meta_path).exists():
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                valid_indices = [int(i) for i in meta.get("valid_slice_indices", []) if 0 < int(i) < us.shape[0] - 1]
            if not valid_indices:
                for idx in range(1, us.shape[0] - 1):
                    if float(overlap_mask[idx].mean()) >= self.min_mask_ratio:
                        valid_indices.append(idx)
            if not valid_indices:
                valid_indices = list(range(1, us.shape[0] - 1))
            self.valid_indices.append(valid_indices)

        self.num_cases = len(self.cases)
        self.total_samples = self.num_cases * self.samples_per_volume

    def __len__(self) -> int:
        return self.total_samples

    def _sample_case_and_index(self, index: int) -> Tuple[int, int]:
        case_idx = index % self.num_cases
        slice_idx = self.rng.choice(self.valid_indices[case_idx])
        return case_idx, slice_idx

    def _augment(
        self,
        us_slice: torch.Tensor,
        mr_slice: torch.Tensor,
        overlap_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self.augment:
            return us_slice, mr_slice, overlap_mask

        if self.rng.random() < 0.5:
            us_slice = torch.flip(us_slice, dims=[2])
            mr_slice = torch.flip(mr_slice, dims=[2])
            overlap_mask = torch.flip(overlap_mask, dims=[2])

        if self.rng.random() < 0.2:
            angle = self.rng.uniform(-15.0, 15.0)
            us_slice, mr_slice = apply_rotation(us_slice, mr_slice, angle)
            overlap_mask, _ = apply_rotation(overlap_mask, overlap_mask, angle)
            overlap_mask = (overlap_mask > 0.5).float()

        return us_slice, mr_slice, overlap_mask

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        case_idx, slice_idx = self._sample_case_and_index(index)
        us_volume = self.us_volumes[case_idx]
        mr_volume = self.mr_volumes[case_idx]
        mask_volume = self.mask_volumes[case_idx]

        us_slice = torch.from_numpy(us_volume[slice_idx - 1 : slice_idx + 2]).float()
        mr_slice = torch.from_numpy(mr_volume[slice_idx : slice_idx + 1]).float()
        overlap_mask = torch.from_numpy(mask_volume[slice_idx : slice_idx + 1]).float()

        us_slice = resize_tensor(us_slice, self.image_size)
        mr_slice = resize_tensor(mr_slice, self.image_size)
        overlap_mask = resize_tensor(overlap_mask, self.image_size)
        overlap_mask = (overlap_mask > 0.5).float()
        us_slice, mr_slice, overlap_mask = self._augment(us_slice, mr_slice, overlap_mask)

        return {
            "subject_id": self.cases[case_idx]["subject_id"],
            "slice_index": torch.tensor(slice_idx, dtype=torch.long),
            "us": us_slice.contiguous(),
            "mr": mr_slice.contiguous(),
            "overlap_mask": overlap_mask.contiguous(),
        }
