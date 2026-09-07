"""
Volume scale assessment helpers for DICOM/CT workloads.
"""

from dataclasses import dataclass
from math import prod
from typing import Tuple

# Practical workload buckets for this project (spine CT + PoC segmentation).
MEDIUM_VOLUME_MIN_VOXELS = 80_000_000
LARGE_VOLUME_MIN_VOXELS = 150_000_000
XL_VOLUME_MIN_VOXELS = 250_000_000

# Reference "large" dataset size used in team discussions.
LARGE_REFERENCE_SIZE = (512, 512, 700)


@dataclass(frozen=True)
class VolumeScaleAssessment:
    """Summary of volume size category and threshold checks."""

    size: Tuple[int, int, int]
    voxel_count: int
    tier: str  # "small" | "medium" | "large" | "xl"
    meets_large_criteria: bool


def estimate_voxel_count(size: Tuple[int, int, int]) -> int:
    """Return voxel count from image size tuple."""
    if len(size) != 3:
        raise ValueError("size must have exactly 3 dimensions")
    dims = tuple(int(v) for v in size)
    if any(v <= 0 for v in dims):
        raise ValueError("size values must be positive")
    return int(prod(dims))


def assess_volume_scale(size: Tuple[int, int, int]) -> VolumeScaleAssessment:
    """
    Classify volume size into small/medium/large/xl bucket.
    """
    dims = tuple(int(v) for v in size)
    voxel_count = estimate_voxel_count(dims)

    if voxel_count >= XL_VOLUME_MIN_VOXELS:
        tier = "xl"
    elif voxel_count >= LARGE_VOLUME_MIN_VOXELS:
        tier = "large"
    elif voxel_count >= MEDIUM_VOLUME_MIN_VOXELS:
        tier = "medium"
    else:
        tier = "small"

    return VolumeScaleAssessment(
        size=dims,
        voxel_count=voxel_count,
        tier=tier,
        meets_large_criteria=voxel_count >= LARGE_VOLUME_MIN_VOXELS,
    )


def format_scale_summary(assessment: VolumeScaleAssessment) -> str:
    """Create one-line textual summary for logs/reports."""
    x, y, z = assessment.size
    return (
        f"size={x}x{y}x{z}, voxels={assessment.voxel_count:,}, "
        f"tier={assessment.tier}, large={assessment.meets_large_criteria}"
    )
