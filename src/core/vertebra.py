"""
Vertebra data models for auto pedicle screw placement.

Provides dataclasses representing individual vertebrae extracted from
TotalSegmentator segmentation masks and the results of pedicle
morphology analysis.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class Vertebra:
    """Single vertebra extracted from a segmentation mask.

    Attributes:
        label: TotalSegmentator label id (25=sacrum .. 43=T1).
        name: Human-readable name, e.g. "L1", "T12", "sacrum".
        centroid_lps: Centroid in LPS world coordinates, shape (3,).
        bounding_box: (min_corner, max_corner) in LPS world coordinates.
        volume_mm3: Physical volume in cubic millimetres.
        mask_indices: (N, 3) array of *ijk* voxel indices belonging to
            this vertebra.
    """

    label: int
    name: str
    centroid_lps: np.ndarray
    bounding_box: Tuple[np.ndarray, np.ndarray]
    volume_mm3: float
    mask_indices: np.ndarray

    def __post_init__(self) -> None:
        self.centroid_lps = np.asarray(self.centroid_lps, dtype=np.float64)
        if self.centroid_lps.shape != (3,):
            raise ValueError(
                f"centroid_lps must have shape (3,), got {self.centroid_lps.shape}"
            )

        bb_min = np.asarray(self.bounding_box[0], dtype=np.float64)
        bb_max = np.asarray(self.bounding_box[1], dtype=np.float64)
        if bb_min.shape != (3,) or bb_max.shape != (3,):
            raise ValueError("bounding_box corners must each have shape (3,)")
        self.bounding_box = (bb_min, bb_max)

        self.mask_indices = np.asarray(self.mask_indices)
        if self.mask_indices.ndim != 2 or self.mask_indices.shape[1] != 3:
            raise ValueError(
                f"mask_indices must have shape (N, 3), got {self.mask_indices.shape}"
            )


@dataclass
class PedicleAnalysisResult:
    """Result of pedicle morphology analysis for one vertebra.

    All spatial quantities are expressed in the **LPS** coordinate system
    to stay consistent with SimpleITK / DICOM conventions used
    throughout the application.
    """

    vertebra: Vertebra

    # Pedicle isthmus centres (LPS world coordinates)
    left_pedicle_center: Optional[np.ndarray] = None
    right_pedicle_center: Optional[np.ndarray] = None

    # Pedicle axis unit vectors (LPS)
    left_pedicle_axis: Optional[np.ndarray] = None
    right_pedicle_axis: Optional[np.ndarray] = None

    # Minimum transverse pedicle width (mm)
    left_pedicle_width: float = 0.0
    right_pedicle_width: float = 0.0

    # Pedicle craniocaudal height at the isthmus (mm)
    left_pedicle_height: float = 0.0
    right_pedicle_height: float = 0.0

    # Which detection path produced the pedicle data ("subregion_label",
    # "subregion_label+coronal_isthmus", "coronal_isthmus" or
    # "axial_components"; empty when none succeeded).
    method: str = ""

    # Anterior vertebral body centre (LPS)
    vertebral_body_center: Optional[np.ndarray] = None

    # Superior vertebral-body endplate plane normal (LPS, +Z oriented)
    upper_endplate_normal: Optional[np.ndarray] = None

    # Overall analysis status
    success: bool = False
    warnings: List[str] = field(default_factory=list)
