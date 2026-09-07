"""
Pedicle Screw Fixation Simulator
================================

A VTK-based medical imaging application for pedicle screw placement planning.

Main Features:
- DICOM CT volume loading
- MPR (Multiplanar Reconstruction) views
- CPU volume rendering
- Synchronized MPR-3D visualization
- Screw placement simulation
"""

import sys
from pathlib import Path


def _read_version() -> str:
    """Read the single-sourced version from the repository-root VERSION file.

    In a frozen (PyInstaller) build, the file is bundled next to the
    executable and extracted under ``sys._MEIPASS``; otherwise it lives two
    directories above this file (the repository root). If the file cannot be
    read (missing, unreadable, wrong permissions, ...), fall back to a
    placeholder rather than letting ``import src`` fail outright.
    """
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        version_path = Path(sys._MEIPASS) / "VERSION"
    else:
        version_path = Path(__file__).resolve().parent.parent / "VERSION"
    try:
        return version_path.read_text(encoding="utf-8").strip()
    except OSError:
        return "0.0.0+unknown"


__title__ = "Pedicle Screw Simulator"
__version__ = _read_version()
__author__ = "Sang-Min Park, MD, Ph.D."
__department__ = "Spine Center and Department of Orthopaedic Surgery"
__organization__ = "Seoul National University Bundang Hospital"
__academic_affiliation__ = "Seoul National University College of Medicine"
__email__ = "psmini@snu.ac.kr"
__website__ = "https://sangmin.me"
__repository__ = "https://github.com/grotyx/pedicle-screw-simulator"
__license__ = "MIT"
__copyright__ = "Copyright (c) 2026 Sang-Min Park, MD, Ph.D."
