#!/usr/bin/env python
import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import SimpleITK as sitk


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess ReMIND for intraoperative ultrasound to intraoperative MRI synthesis."
    )
    parser.add_argument("--dataset-root", type=str, required=True, help="ReMIND dataset root directory.")
    parser.add_argument("--metadata-csv", type=str, default=None, help="Path to metadata.csv.")
    parser.add_argument("--output-root", type=str, required=True, help="Directory to save preprocessed .npy volumes.")
    parser.add_argument("--subjects", type=str, nargs="*", default=None, help="Optional subject IDs.")
    parser.add_argument("--study-description", type=str, default="Intraop", help="Study description to match.")
    parser.add_argument("--us-series", type=str, default="USpreimri", help="Preferred ioUS series description.")
    parser.add_argument("--mr-series", type=str, default="2DAXT2BLADE", help="Preferred intraoperative MRI series.")
    parser.add_argument("--preop-mr-series", type=str, default="3DAXT1postcontrast", help="Preoperative MRI series for structural prior.")
    parser.add_argument(
        "--spacing",
        type=float,
        nargs=3,
        default=None,
        metavar=("SX", "SY", "SZ"),
        help="Optional output spacing in MRI reference space. Defaults to MRI spacing.",
    )
    parser.add_argument("--us-lower-percentile", type=float, default=1.0)
    parser.add_argument("--us-upper-percentile", type=float, default=99.0)
    parser.add_argument("--min-overlap-ratio", type=float, default=0.05, help="Minimum overlap ratio for valid slices.")
    parser.add_argument("--min-valid-slices", type=int, default=8, help="Discard cases with fewer valid overlap slices.")
    parser.add_argument("--crop-padding", type=int, default=8, help="Padding added around the cropped ROI.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing preprocessed cases.")
    return parser.parse_args()


def resolve_dataset_path(dataset_root: Path, relative_path: str) -> Path:
    cleaned = relative_path.replace(".\\", "").replace("./", "")
    direct = (dataset_root / cleaned).resolve()
    if direct.exists():
        return direct

    relative = Path(cleaned)
    if len(relative.parts) >= 2 and relative.parts[0].lower() == "remind":
        subject = relative.parts[1]
        remapped = dataset_root / subject / Path(*relative.parts[1:])
        if remapped.exists():
            return remapped.resolve()

    return direct


def read_dicom_or_volume(path: Path) -> sitk.Image:
    if path.is_dir():
        reader = sitk.ImageSeriesReader()
        series_ids = reader.GetGDCMSeriesIDs(str(path))
        if series_ids:
            file_names = reader.GetGDCMSeriesFileNames(str(path), series_ids[0])
            reader.SetFileNames(file_names)
            return reader.Execute()

        dicom_files = sorted(str(p) for p in path.iterdir() if p.is_file())
        if not dicom_files:
            raise FileNotFoundError(f"No files found under {path}")
        reader.SetFileNames(dicom_files)
        return reader.Execute()
    return sitk.ReadImage(str(path))


def cast_for_registration(image: sitk.Image) -> sitk.Image:
    if image.GetPixelID() not in (sitk.sitkFloat32, sitk.sitkFloat64):
        return sitk.Cast(image, sitk.sitkFloat32)
    return image


def rigid_register(fixed: sitk.Image, moving: sitk.Image) -> sitk.Transform:
    fixed = cast_for_registration(fixed)
    moving = cast_for_registration(moving)
    initial_transform = sitk.CenteredTransformInitializer(
        fixed,
        moving,
        sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY,
    )

    registration = sitk.ImageRegistrationMethod()
    registration.SetMetricAsMattesMutualInformation(50)
    registration.SetMetricSamplingStrategy(registration.RANDOM)
    registration.SetMetricSamplingPercentage(0.1, seed=42)
    registration.SetInterpolator(sitk.sitkLinear)
    registration.SetOptimizerAsRegularStepGradientDescent(
        learningRate=1.0,
        minStep=1e-4,
        numberOfIterations=200,
        relaxationFactor=0.5,
        gradientMagnitudeTolerance=1e-8,
    )
    registration.SetOptimizerScalesFromPhysicalShift()
    registration.SetShrinkFactorsPerLevel([4, 2, 1])
    registration.SetSmoothingSigmasPerLevel([2, 1, 0])
    registration.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    registration.SetInitialTransform(initial_transform, inPlace=False)
    return registration.Execute(fixed, moving)


def make_reference_from_spacing(image: sitk.Image, out_spacing: Sequence[float]) -> sitk.Image:
    in_spacing = np.array(image.GetSpacing(), dtype=np.float64)
    in_size = np.array(image.GetSize(), dtype=np.int32)
    out_spacing = np.array(out_spacing, dtype=np.float64)
    out_size = np.maximum(np.round(in_size * (in_spacing / out_spacing)).astype(np.int32), 1)

    reference = sitk.Image([int(v) for v in out_size], image.GetPixelIDValue())
    reference.SetOrigin(image.GetOrigin())
    reference.SetDirection(image.GetDirection())
    reference.SetSpacing([float(v) for v in out_spacing])
    return reference


def resample_image(
    image: sitk.Image,
    reference: sitk.Image,
    transform: Optional[sitk.Transform] = None,
    interpolator: int = sitk.sitkBSpline,
    default_value: float = 0.0,
) -> sitk.Image:
    if transform is None:
        transform = sitk.Transform(3, sitk.sitkIdentity)
    return sitk.Resample(image, reference, transform, interpolator, default_value, image.GetPixelID())


def compute_us_mask(volume: np.ndarray) -> np.ndarray:
    max_value = float(volume.max())
    if max_value <= 0:
        return np.zeros_like(volume, dtype=bool)
    return volume > (0.01 * max_value)


def compute_mr_mask(volume: np.ndarray) -> np.ndarray:
    return volume != 0


def compute_crop_bbox(
    us_mask: np.ndarray, mr_mask: np.ndarray, padding: int
) -> Tuple[slice, slice, slice]:
    foreground = us_mask | mr_mask
    if not np.any(foreground):
        shape = foreground.shape
        return (slice(0, shape[0]), slice(0, shape[1]), slice(0, shape[2]))

    coords = np.argwhere(foreground)
    z0, y0, x0 = coords.min(axis=0)
    z1, y1, x1 = coords.max(axis=0) + 1
    z0 = max(0, z0 - padding)
    y0 = max(0, y0 - padding)
    x0 = max(0, x0 - padding)
    z1 = min(foreground.shape[0], z1 + padding)
    y1 = min(foreground.shape[1], y1 + padding)
    x1 = min(foreground.shape[2], x1 + padding)
    return (slice(z0, z1), slice(y0, y1), slice(x0, x1))


def normalize_us(volume: np.ndarray, lower_percentile: float, upper_percentile: float) -> np.ndarray:
    nonzero = volume[volume > 0]
    if nonzero.size == 0:
        return np.zeros_like(volume, dtype=np.float32)
    lo = np.percentile(nonzero, lower_percentile)
    hi = np.percentile(nonzero, upper_percentile)
    if hi <= lo:
        hi = lo + 1e-6
    volume = np.clip(volume, lo, hi)
    volume = 2.0 * (volume - lo) / (hi - lo) - 1.0
    return volume.astype(np.float32)


def normalize_mr(volume: np.ndarray) -> np.ndarray:
    mask = volume != 0
    if not np.any(mask):
        return np.zeros_like(volume, dtype=np.float32)
    values = volume[mask]
    mean = values.mean()
    std = values.std()
    if std < 1e-6:
        std = 1.0
    z = (volume - mean) / std
    z = np.clip(z, -3.0, 3.0) / 3.0
    z[~mask] = -1.0
    return z.astype(np.float32)


def compute_overlap_mask(us_mask: np.ndarray, mr_mask: np.ndarray) -> np.ndarray:
    return (us_mask & mr_mask).astype(np.uint8)


def compute_valid_slice_indices(
    overlap_mask: np.ndarray,
    mr_mask: np.ndarray,
    min_overlap_ratio: float,
) -> List[int]:
    valid_indices: List[int] = []
    for index in range(1, overlap_mask.shape[0] - 1):
        mr_pixels = int(mr_mask[index].sum())
        if mr_pixels == 0:
            continue
        overlap_ratio = float(overlap_mask[index].sum()) / float(mr_pixels)
        if overlap_ratio >= min_overlap_ratio:
            valid_indices.append(index)
    return valid_indices


def save_case(
    output_root: Path,
    subject_id: str,
    us_volume: np.ndarray,
    mr_volume: np.ndarray,
    overlap_mask: np.ndarray,
    extra_meta: Dict[str, object],
) -> None:
    case_dir = output_root / subject_id
    case_dir.mkdir(parents=True, exist_ok=True)
    np.save(case_dir / "us.npy", us_volume.astype(np.float32))
    np.save(case_dir / "mr.npy", mr_volume.astype(np.float32))
    np.save(case_dir / "overlap_mask.npy", overlap_mask.astype(np.uint8))
    with open(case_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(extra_meta, f, indent=2)


def collect_cases_from_metadata(
    dataset_root: Path,
    metadata_csv: Path,
    subject_filter: Optional[Sequence[str]],
    study_description: str,
    us_series: str,
    mr_series: str,
    preop_mr_series: str,
) -> List[Dict[str, str]]:
    grouped: Dict[str, Dict[str, Dict[str, str]]] = {}
    with open(metadata_csv, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            subject_id = row["Subject ID"]
            if subject_filter and subject_id not in subject_filter:
                continue

            study = row["Study Description"].strip().lower()
            modality = row["Modality"].strip().upper()
            desc = row["Series Description"].strip()
            path = resolve_dataset_path(dataset_root, row["File Location"])
            if not path.exists():
                continue

            bucket = grouped.setdefault(subject_id, {})

            if study_description.lower() in study:
                if modality == "US" and desc == us_series:
                    bucket["us"] = {"description": desc, "path": str(path)}
                elif modality == "MR" and desc == mr_series:
                    bucket["mr"] = {"description": desc, "path": str(path)}

            if study.lower() == "preop":
                if modality == "MR" and desc == preop_mr_series:
                    bucket["preop_mr"] = {"description": desc, "path": str(path)}

    cases: List[Dict[str, str]] = []
    for subject_id in sorted(grouped):
        bucket = grouped[subject_id]
        if "us" not in bucket or "mr" not in bucket:
            continue
        case = {
            "subject_id": subject_id,
            "us_path": bucket["us"]["path"],
            "mr_path": bucket["mr"]["path"],
            "us_description": bucket["us"]["description"],
            "mr_description": bucket["mr"]["description"],
        }
        if "preop_mr" in bucket:
            case["preop_mr_path"] = bucket["preop_mr"]["path"]
            case["preop_mr_description"] = bucket["preop_mr"]["description"]
        cases.append(case)
    return cases


def preprocess_case(
    subject_id: str,
    us_path: Path,
    mr_path: Path,
    output_root: Path,
    out_spacing: Optional[Sequence[float]],
    lower_percentile: float,
    upper_percentile: float,
    min_overlap_ratio: float,
    min_valid_slices: int,
    crop_padding: int,
    overwrite: bool,
    descriptions: Optional[Dict[str, str]] = None,
    preop_mr_path: Optional[Path] = None,
) -> bool:
    case_dir = output_root / subject_id
    if case_dir.exists() and not overwrite:
        print(f"[Skip] {subject_id}: output exists at {case_dir}")
        return True

    print(f"[Load] {subject_id}")
    us_image = read_dicom_or_volume(us_path)
    mr_image = read_dicom_or_volume(mr_path)
    preop_mr_image = None
    if preop_mr_path is not None:
        try:
            preop_mr_image = read_dicom_or_volume(preop_mr_path)
            print(f"[Load] {subject_id}: preop MR loaded from {preop_mr_path}")
        except Exception as e:
            print(f"[Warn] {subject_id}: failed to load preop MR: {e}")
            preop_mr_image = None

    if us_image.GetDimension() != 3 or mr_image.GetDimension() != 3:
        print(f"[Skip] {subject_id}: only 3D volumes are supported.")
        return False

    print(f"[Register] {subject_id}")
    us_to_mr = rigid_register(mr_image, us_image)

    preop_to_mr = None
    if preop_mr_image is not None:
        if preop_mr_image.GetDimension() == 3:
            try:
                preop_to_mr = rigid_register(mr_image, preop_mr_image)
                print(f"[Register] {subject_id}: preop MR -> intraop MR done")
            except Exception as e:
                print(f"[Warn] {subject_id}: preop MR registration failed: {e}")
                preop_to_mr = None
        else:
            print(f"[Warn] {subject_id}: preop MR is not 3D, skipping")
            preop_mr_image = None

    spacing = out_spacing if out_spacing is not None else mr_image.GetSpacing()
    reference = make_reference_from_spacing(mr_image, spacing)

    print(f"[Resample] {subject_id}")
    mr_resampled = resample_image(mr_image, reference, interpolator=sitk.sitkBSpline)
    us_resampled = resample_image(us_image, reference, transform=us_to_mr, interpolator=sitk.sitkBSpline)

    preop_mr_resampled = None
    if preop_mr_image is not None and preop_to_mr is not None:
        preop_mr_resampled = resample_image(
            preop_mr_image, reference, transform=preop_to_mr, interpolator=sitk.sitkBSpline
        )

    us_array = sitk.GetArrayFromImage(us_resampled).astype(np.float32)
    mr_array = sitk.GetArrayFromImage(mr_resampled).astype(np.float32)
    preop_mr_array = None
    if preop_mr_resampled is not None:
        preop_mr_array = sitk.GetArrayFromImage(preop_mr_resampled).astype(np.float32)

    us_mask = compute_us_mask(us_array)
    mr_mask = compute_mr_mask(mr_array)
    overlap_mask = compute_overlap_mask(us_mask, mr_mask)

    print(f"[Crop] {subject_id}")
    bbox = compute_crop_bbox(us_mask, mr_mask, crop_padding)
    us_cropped = us_array[bbox]
    mr_cropped = mr_array[bbox]
    overlap_cropped = overlap_mask[bbox]
    mr_mask_cropped = mr_mask[bbox]
    preop_mr_cropped = preop_mr_array[bbox] if preop_mr_array is not None else None

    valid_slice_indices = compute_valid_slice_indices(overlap_cropped, mr_mask_cropped, min_overlap_ratio)
    if len(valid_slice_indices) < min_valid_slices:
        print(f"[Skip] {subject_id}: only {len(valid_slice_indices)} valid overlap slices.")
        return False

    print(f"[Normalize] {subject_id}")
    us_norm = normalize_us(us_cropped, lower_percentile, upper_percentile)
    mr_norm = normalize_mr(mr_cropped)
    preop_mr_norm = normalize_mr(preop_mr_cropped) if preop_mr_cropped is not None else None

    meta = {
        "subject_id": subject_id,
        "us_path": str(us_path),
        "mr_path": str(mr_path),
        "preop_mr_path": str(preop_mr_path) if preop_mr_path is not None else None,
        "us_shape_zyx": list(us_norm.shape),
        "mr_shape_zyx": list(mr_norm.shape),
        "spacing_xyz": [float(v) for v in spacing],
        "us_percentile_clip": [lower_percentile, upper_percentile],
        "mr_zscore_clip_sigma": 3.0,
        "min_overlap_ratio": min_overlap_ratio,
        "valid_slice_indices": valid_slice_indices,
        "valid_slice_count": len(valid_slice_indices),
        "has_preop_mr": preop_mr_norm is not None,
    }
    if descriptions:
        meta.update(descriptions)

    save_case(output_root, subject_id, us_norm, mr_norm, overlap_cropped, meta)

    if preop_mr_norm is not None:
        np.save(case_dir / "preop_mr.npy", preop_mr_norm.astype(np.float32))
        print(f"[Save] {subject_id}: preop_mr.npy saved (shape={preop_mr_norm.shape})")

    print(f"[Done] {subject_id} -> {case_dir}")
    return True


def write_manifest(output_root: Path, subject_ids: Sequence[str]) -> None:
    manifest_path = output_root / "preprocessed_pairs.csv"
    with open(manifest_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["subject_id", "us_path", "mr_path", "mask_path", "meta_path", "preop_mr_path"])
        for subject_id in subject_ids:
            case_dir = output_root / subject_id
            preop_mr_path = str(case_dir / "preop_mr.npy") if (case_dir / "preop_mr.npy").exists() else ""
            writer.writerow(
                [
                    subject_id,
                    str(case_dir / "us.npy"),
                    str(case_dir / "mr.npy"),
                    str(case_dir / "overlap_mask.npy"),
                    str(case_dir / "meta.json"),
                    preop_mr_path,
                ]
            )


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root).resolve()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    metadata_csv = Path(args.metadata_csv).resolve() if args.metadata_csv else dataset_root / "metadata.csv"

    cases = collect_cases_from_metadata(
        dataset_root=dataset_root,
        metadata_csv=metadata_csv,
        subject_filter=args.subjects,
        study_description=args.study_description,
        us_series=args.us_series,
        mr_series=args.mr_series,
        preop_mr_series=args.preop_mr_series,
    )
    if not cases:
        raise RuntimeError("No valid intraoperative US/MR pairs were found for the requested series.")

    processed_subjects: List[str] = []
    for case in cases:
        preop_mr_path = Path(case["preop_mr_path"]) if "preop_mr_path" in case else None
        saved = preprocess_case(
            subject_id=case["subject_id"],
            us_path=Path(case["us_path"]),
            mr_path=Path(case["mr_path"]),
            output_root=output_root,
            out_spacing=args.spacing,
            lower_percentile=args.us_lower_percentile,
            upper_percentile=args.us_upper_percentile,
            min_overlap_ratio=args.min_overlap_ratio,
            min_valid_slices=args.min_valid_slices,
            crop_padding=args.crop_padding,
            overwrite=args.overwrite,
            descriptions={
                "study_description": args.study_description,
                "us_description": case["us_description"],
                "mr_description": case["mr_description"],
                "preop_mr_description": case.get("preop_mr_description", ""),
            },
            preop_mr_path=preop_mr_path,
        )
        if saved:
            processed_subjects.append(case["subject_id"])

    if not processed_subjects:
        raise RuntimeError("No cases survived overlap-based filtering.")

    write_manifest(output_root, processed_subjects)
    print(f"Saved manifest to {output_root / 'preprocessed_pairs.csv'}")


if __name__ == "__main__":
    main()
