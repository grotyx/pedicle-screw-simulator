"""
Utilities module - Helper functions and constants
"""

from .constants import *

__all__ = ["sitk_to_vtk", "lps_to_ras_transform"]


def __getattr__(name):
    """Lazily import VTK-dependent helpers."""
    if name in {"sitk_to_vtk", "lps_to_ras_transform"}:
        from . import vtk_helpers
        return getattr(vtk_helpers, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
