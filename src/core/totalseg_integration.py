"""
TotalSegmentator integration helpers with safe fallback behavior.
"""

from dataclasses import dataclass, field
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
from typing import Callable, List, Optional

import SimpleITK as sitk


SPINE_ROI_SUBSET = (
    "sacrum",
    "vertebrae_S1",
    "vertebrae_L5",
    "vertebrae_L4",
    "vertebrae_L3",
    "vertebrae_L2",
    "vertebrae_L1",
    "vertebrae_T12",
    "vertebrae_T11",
    "vertebrae_T10",
    "vertebrae_T9",
    "vertebrae_T8",
    "vertebrae_T7",
    "vertebrae_T6",
    "vertebrae_T5",
    "vertebrae_T4",
    "vertebrae_T3",
    "vertebrae_T2",
    "vertebrae_T1",
    "vertebrae_C7",
    "vertebrae_C6",
    "vertebrae_C5",
    "vertebrae_C4",
    "vertebrae_C3",
    "vertebrae_C2",
    "vertebrae_C1",
)


@dataclass
class SegmentationRunResult:
    """Result payload for auto-segmentation runs."""
    success: bool
    method: str  # "totalsegmentator" or "threshold_fallback"
    mask_path: str
    message: str
    geometry_warnings: List[str] = field(default_factory=list)


def is_totalsegmentator_available() -> bool:
    """Return True when TotalSegmentator package is importable."""
    return importlib.util.find_spec("totalsegmentator") is not None


def preferred_segmentation_device() -> str:
    """Prefer a CUDA GPU and otherwise use the reliable CPU path."""
    try:
        import torch

        if torch.cuda.is_available():
            return "gpu"
    except Exception:
        pass
    return "cpu"


def _emit_progress(
    progress_callback: Optional[Callable[[str], None]],
    message: str
) -> None:
    """Emit progress update if callback exists."""
    if progress_callback is not None:
        progress_callback(message)


def run_totalsegmentator(
    image: sitk.Image,
    work_dir: str,
    task: str = "total",
    device: str = "cpu",
    roi_subset: Optional[List[str]] = None,
    fast: bool = False,
    force_split: bool = False,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> str:
    """
    Run TotalSegmentator on a SimpleITK volume and return mask path.

    Output is a multilabel NIfTI mask file.
    """
    _ensure_torch_shm_executable()

    output_root = Path(work_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    input_path = output_root / "input_volume.nii.gz"
    output_path = output_root / "totalseg_multilabel.nii.gz"

    _emit_progress(progress_callback, "Preparing NIfTI input for TotalSegmentator...")
    sitk.WriteImage(image, str(input_path))

    _emit_progress(
        progress_callback,
        (
            f"Running TotalSegmentator task '{task}'. "
            "The first run may download model weights..."
        ),
    )

    if getattr(sys, "frozen", False):
        _run_frozen_totalsegmentator(
            input_path=input_path,
            output_path=output_path,
            task=task,
            device=device,
            roi_subset=roi_subset,
            fast=fast,
            force_split=force_split,
        )
        if not output_path.exists():
            raise RuntimeError(
                "TotalSegmentator finished but no output mask was created."
            )
        return str(output_path)

    module_spec = importlib.util.find_spec(
        "totalsegmentator.bin.TotalSegmentator"
    )
    if module_spec is None or module_spec.origin is None:
        raise RuntimeError("TotalSegmentator command module was not found.")

    command = [
        sys.executable,
        module_spec.origin,
        "-i",
        str(input_path),
        "-o",
        str(output_path),
        "-ta",
        task,
        "-ml",
        "-d",
        device,
        "-q",
        "-ns",
        "1",
    ]
    if roi_subset:
        command.extend(["-rs", *roi_subset])
    if fast:
        command.append("-f")
    if force_split:
        command.append("-fs")

    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(
            f"TotalSegmentator execution failed: {detail or 'unknown error'}"
        )

    if not output_path.exists():
        raise RuntimeError("TotalSegmentator finished but no output mask was created.")

    return str(output_path)


def _run_frozen_totalsegmentator(
    *,
    input_path: Path,
    output_path: Path,
    task: str,
    device: str,
    roi_subset: Optional[List[str]],
    fast: bool,
    force_split: bool,
) -> None:
    """Run the bundled Python API because frozen apps cannot spawn `-m`."""
    try:
        from totalsegmentator.python_api import totalsegmentator
    except ImportError as exc:
        raise RuntimeError(
            "The packaged TotalSegmentator runtime could not be loaded."
        ) from exc

    totalsegmentator(
        str(input_path),
        str(output_path),
        task=task,
        ml=True,
        device=device,
        roi_subset=roi_subset,
        fast=fast,
        force_split=force_split,
        quiet=True,
        nr_thr_saving=1,
    )


def _ensure_torch_shm_executable() -> None:
    """
    Ensure torch shared-memory manager binary is executable on macOS/Linux.

    Some installations can leave this file without execute permission,
    which breaks multiprocessing inference with "Permission denied".
    """
    try:
        import torch
    except Exception:
        return

    shm_path = Path(torch.__file__).resolve().parent / "bin" / "torch_shm_manager"
    if not shm_path.exists():
        return
    if os.access(shm_path, os.X_OK):
        return

    try:
        mode = shm_path.stat().st_mode
        shm_path.chmod(mode | 0o111)
    except Exception:
        # Best effort only; fallback path still protects runtime.
        return


def run_threshold_fallback(
    image: sitk.Image,
    work_dir: str,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> str:
    """
    Run a simple HU threshold fallback segmentation.

    This is not AI segmentation; it provides a safe non-blocking backup mask.
    """
    output_root = Path(work_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    output_path = output_root / "threshold_fallback_mask.nii.gz"

    _emit_progress(progress_callback, "Running threshold fallback segmentation...")
    # Conservative bone-like threshold for CT HU
    mask = sitk.BinaryThreshold(
        image,
        lowerThreshold=180,
        upperThreshold=6000,
        insideValue=1,
        outsideValue=0,
    )
    mask = sitk.Cast(mask, sitk.sitkUInt8)
    sitk.WriteImage(mask, str(output_path))
    return str(output_path)


def _geometry_close(a: float, b: float, tolerance: float = 1e-4) -> bool:
    return abs(float(a) - float(b)) <= tolerance


def check_segmentation_geometry(
    reference_image: sitk.Image,
    mask_image: sitk.Image,
    tolerance: float = 1e-4,
) -> List[str]:
    """
    Compare geometry metadata between reference image and segmentation mask.

    Returns list of warning messages when mismatch is detected.
    """
    warnings: List[str] = []

    ref_size = tuple(int(v) for v in reference_image.GetSize())
    mask_size = tuple(int(v) for v in mask_image.GetSize())
    if ref_size != mask_size:
        warnings.append(f"size mismatch: ref={ref_size}, mask={mask_size}")

    ref_spacing = tuple(float(v) for v in reference_image.GetSpacing())
    mask_spacing = tuple(float(v) for v in mask_image.GetSpacing())
    for axis, (ref_v, mask_v) in enumerate(zip(ref_spacing, mask_spacing)):
        if not _geometry_close(ref_v, mask_v, tolerance=tolerance):
            warnings.append(
                f"spacing mismatch at axis {axis}: ref={ref_v}, mask={mask_v}"
            )

    ref_origin = tuple(float(v) for v in reference_image.GetOrigin())
    mask_origin = tuple(float(v) for v in mask_image.GetOrigin())
    for axis, (ref_v, mask_v) in enumerate(zip(ref_origin, mask_origin)):
        if not _geometry_close(ref_v, mask_v, tolerance=tolerance):
            warnings.append(
                f"origin mismatch at axis {axis}: ref={ref_v}, mask={mask_v}"
            )

    ref_direction = tuple(float(v) for v in reference_image.GetDirection())
    mask_direction = tuple(float(v) for v in mask_image.GetDirection())
    if len(ref_direction) != len(mask_direction):
        warnings.append(
            f"direction length mismatch: ref={len(ref_direction)}, "
            f"mask={len(mask_direction)}"
        )
    else:
        for index, (ref_v, mask_v) in enumerate(zip(ref_direction, mask_direction)):
            if not _geometry_close(ref_v, mask_v, tolerance=tolerance):
                warnings.append(
                    f"direction mismatch at index {index}: ref={ref_v}, mask={mask_v}"
                )

    return warnings


def validate_segmentation_output(
    reference_image: sitk.Image,
    mask_path: str,
) -> List[str]:
    """
    Load segmentation mask and return geometry warnings relative to input image.
    """
    mask_image = sitk.ReadImage(mask_path)
    return check_segmentation_geometry(reference_image, mask_image)


def _build_result(
    image: sitk.Image,
    method: str,
    mask_path: str,
    message: str,
) -> SegmentationRunResult:
    geometry_warnings = validate_segmentation_output(image, mask_path)
    if geometry_warnings:
        summary = "; ".join(geometry_warnings[:2])
        if len(geometry_warnings) > 2:
            summary += f" (+{len(geometry_warnings) - 2} more)"
        message = f"{message} Geometry check warning: {summary}"

    return SegmentationRunResult(
        success=True,
        method=method,
        mask_path=mask_path,
        message=message,
        geometry_warnings=geometry_warnings,
    )


def run_segmentation_with_fallback(
    image: sitk.Image,
    work_dir: str,
    task: str = "total",
    device: str = "cpu",
    roi_subset: Optional[List[str]] = None,
    fast: bool = False,
    force_split: bool = False,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> SegmentationRunResult:
    """
    Run TotalSegmentator with fallback strategy.

    Behavior:
    1) If TotalSegmentator is available and succeeds -> use it.
    2) Otherwise fallback to threshold mask.
    """
    if not is_totalsegmentator_available():
        mask_path = run_threshold_fallback(
            image=image,
            work_dir=work_dir,
            progress_callback=progress_callback,
        )
        return _build_result(
            image=image,
            method="threshold_fallback",
            mask_path=mask_path,
            message=(
                "TotalSegmentator is not installed. "
                "Threshold fallback mask was generated instead."
            ),
        )

    attempt_devices = [device]
    if str(device).startswith("gpu"):
        attempt_devices.append("cpu")
    errors = []

    for attempt_index, attempt_device in enumerate(attempt_devices):
        try:
            mask_path = run_totalsegmentator(
                image=image,
                work_dir=work_dir,
                task=task,
                device=attempt_device,
                roi_subset=roi_subset,
                fast=fast,
                force_split=(
                    force_split
                    or (attempt_device == "cpu" and not fast)
                ),
                progress_callback=progress_callback,
            )
            message = "TotalSegmentator segmentation completed."
            if attempt_index > 0:
                message = (
                    "TotalSegmentator segmentation completed after CPU retry."
                )
            return _build_result(
                image=image,
                method="totalsegmentator",
                mask_path=mask_path,
                message=message,
            )
        except BaseException as exc:
            if isinstance(exc, KeyboardInterrupt):
                raise
            errors.append(f"{attempt_device}: {exc}")
            if attempt_index + 1 < len(attempt_devices):
                _emit_progress(
                    progress_callback,
                    "GPU segmentation failed; retrying on CPU...",
                )

    mask_path = run_threshold_fallback(
        image=image,
        work_dir=work_dir,
        progress_callback=progress_callback,
    )
    return _build_result(
        image=image,
        method="threshold_fallback",
        mask_path=mask_path,
        message=(
            "TotalSegmentator failed; fallback mask was generated. "
            f"Reason: {'; '.join(errors)}"
        ),
    )
