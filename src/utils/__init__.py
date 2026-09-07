"""
Utilities module - Helper functions and constants
"""

from .constants import *  # noqa: F403 - intentional re-export of all constants

# sitk_to_vtk / lps_to_ras_transform are provided lazily by __getattr__ below
# (VTK-dependent helpers), not by the star import above; ruff cannot see
# that binding so it reports them as possibly star-import-derived.
__all__ = ["sitk_to_vtk", "lps_to_ras_transform"]  # noqa: F405


def __getattr__(name):
    """Lazily import VTK-dependent helpers."""
    if name in {"sitk_to_vtk", "lps_to_ras_transform"}:
        from . import vtk_helpers
        return getattr(vtk_helpers, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
