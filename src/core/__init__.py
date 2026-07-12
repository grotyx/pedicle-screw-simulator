"""
Core module - DICOM loading and volume management
"""

from .coordinate_system import CoordinateSystem
from .volume_scale import (
    assess_volume_scale,
    estimate_voxel_count,
    format_scale_summary,
    VolumeScaleAssessment,
)

__all__ = [
    "DicomLoader",
    "VolumeManager",
    "CoordinateSystem",
    "VolumeScaleAssessment",
    "assess_volume_scale",
    "estimate_voxel_count",
    "format_scale_summary",
]


def __getattr__(name):
    """Lazily import heavy modules so lightweight imports still work."""
    if name == "DicomLoader":
        from .dicom_loader import DicomLoader
        return DicomLoader
    if name == "VolumeManager":
        from .volume_manager import VolumeManager
        return VolumeManager
    if name == "CoordinateSystem":
        return CoordinateSystem
    if name in {"VolumeScaleAssessment", "assess_volume_scale", "estimate_voxel_count", "format_scale_summary"}:
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
