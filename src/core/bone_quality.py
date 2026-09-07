"""Bone-quality (HU) metrics for a planned screw trajectory.

Three complementary measurements are reported, each with its own literature
threshold:

* **Trajectory HU** — mean/min HU of the bone the screw itself engages, the
  quantity most directly tied to pull-out strength.
* **Pedicle HU** — the same sampling restricted to the isthmus neighbourhood,
  where purchase actually matters.
* **Vertebral body HU** — a trabecular ROI at the body centre, the opportunistic
  osteoporosis screen that radiology reports use.

Thresholds live in :mod:`src.utils.constants`; see the citations there.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np
import SimpleITK as sitk

from ..utils.constants import (TRAJECTORY_BODY_HU_RATIO_THRESHOLD,
                               TRAJECTORY_HU_LOOSENING_THRESHOLD,
                               VERTEBRAL_HU_LOW_BMD_THRESHOLD,
                               VERTEBRAL_HU_OSTEOPOROSIS_THRESHOLD)
from .screw_grading import ScrewGrader

Point3 = Sequence[float]

#: Radius around the isthmus centre within which a sample counts as "pedicle".
PEDICLE_ROI_RADIUS_MM = 10.0
#: Semi-axes (x, y, z in mm) of the default trabecular ROI at the body centre.
BODY_ROI_RADII_MM = (8.0, 8.0, 6.0)
#: Fewer voxels than this and the ROI mean is too noisy to report.
MIN_ROI_VOXELS = 20


@dataclass
class BoneQualityMetrics:
    """HU metrics for one trajectory. Every value is ``None`` when unmeasurable."""

    trajectory_mean_hu: Optional[float] = None
    trajectory_min_hu: Optional[float] = None
    pedicle_mean_hu: Optional[float] = None
    body_mean_hu: Optional[float] = None
    trajectory_body_ratio: Optional[float] = None
    warnings: List[str] = field(default_factory=list)


def vertebral_body_hu(
    ct: sitk.Image,
    mask: sitk.Image,
    label: int,
    body_center_lps: Point3,
    radii_mm: Tuple[float, float, float] = BODY_ROI_RADII_MM,
) -> Optional[float]:
    """Mean HU of an ellipsoidal trabecular ROI at ``body_center_lps``.

    The ROI is intersected with ``label`` so cortex-adjacent background, discs
    and neighbouring vertebrae cannot skew the mean. Returns ``None`` when
    fewer than :data:`MIN_ROI_VOXELS` voxels survive.

    ``ct`` and ``mask`` must share a grid with an identity direction, which is
    what :class:`~src.core.screw_grading.ScrewGrader` already enforces.
    """
    ct_arr = sitk.GetArrayFromImage(ct)
    mask_arr = sitk.GetArrayFromImage(mask)
    origin = np.asarray(mask.GetOrigin(), dtype=np.float64)
    spacing = np.asarray(mask.GetSpacing(), dtype=np.float64)
    centre_idx = (np.asarray(body_center_lps, dtype=np.float64) - origin) / spacing  # x, y, z
    radii_idx = np.asarray(radii_mm, dtype=np.float64) / spacing                     # x, y, z

    # Work inside the ellipsoid's index-space bounding box: no voxel outside it
    # can satisfy the inequality, and a whole-volume grid would cost gigabytes
    # on a full-resolution CT.
    extent = np.asarray(mask_arr.shape[::-1], dtype=np.int64)                        # x, y, z
    lo = np.clip(np.floor(centre_idx - radii_idx).astype(np.int64), 0, extent)
    hi = np.clip(np.ceil(centre_idx + radii_idx).astype(np.int64) + 1, 0, extent)
    if np.any(hi <= lo):
        return None

    axes = [np.arange(lo[a], hi[a], dtype=np.float64) for a in range(3)]             # x, y, z
    normalised = [((axes[a] - centre_idx[a]) / radii_idx[a]) ** 2 for a in range(3)]
    ellipsoid = (
        normalised[2][:, None, None] + normalised[1][None, :, None] + normalised[0][None, None, :]
    ) <= 1.0                                                                          # (z, y, x)

    box = (slice(lo[2], hi[2]), slice(lo[1], hi[1]), slice(lo[0], hi[0]))
    roi = ellipsoid & (mask_arr[box] == label)
    if int(roi.sum()) < MIN_ROI_VOXELS:
        return None
    return float(ct_arr[box][roi].mean())


def assess_bone_quality(
    grader: ScrewGrader,
    entry: Point3,
    target: Point3,
    diameter_mm: float,
    label: int,
    body_center_lps: Optional[Point3] = None,
    isthmus_center_lps: Optional[Point3] = None,
    trajectory_threshold: float = TRAJECTORY_HU_LOOSENING_THRESHOLD,
) -> BoneQualityMetrics:
    """Measure trajectory, pedicle and vertebral-body HU and flag weak bone.

    ``body_center_lps`` and ``isthmus_center_lps`` come from the pedicle
    analyser; each metric that depends on one is simply omitted when it is not
    supplied, and every metric is omitted when the grader has no CT.
    """
    points = grader.cylinder_points(entry, target, diameter_mm)
    hu = grader.hu_at_points(points)
    valid = ~np.isnan(hu)
    warnings: List[str] = []

    traj_mean = float(hu[valid].mean()) if valid.any() else None
    traj_min = float(hu[valid].min()) if valid.any() else None

    pedicle_mean = None
    if isthmus_center_lps is not None and valid.any():
        centre = np.asarray(isthmus_center_lps, dtype=np.float64)
        near = np.linalg.norm(points - centre, axis=1) <= PEDICLE_ROI_RADIUS_MM
        near &= valid
        if near.any():
            pedicle_mean = float(hu[near].mean())

    body_mean = None
    if body_center_lps is not None and grader.ct_image() is not None:
        body_mean = vertebral_body_hu(
            grader.ct_image(), grader.mask_image(), label, body_center_lps
        )

    ratio = None
    if traj_mean is not None and body_mean is not None and body_mean != 0.0:
        ratio = traj_mean / body_mean

    if traj_mean is not None and traj_mean < trajectory_threshold:
        warnings.append(
            f"Trajectory HU {traj_mean:.0f} below {trajectory_threshold:.0f} HU — loosening risk "
            "(consider larger diameter, augmentation, or CBT)"
        )
    if body_mean is not None:
        if body_mean < VERTEBRAL_HU_OSTEOPOROSIS_THRESHOLD:
            warnings.append(
                f"Vertebral body HU {body_mean:.0f} suggests osteoporosis "
                f"(<{VERTEBRAL_HU_OSTEOPOROSIS_THRESHOLD:.0f} HU)"
            )
        elif body_mean < VERTEBRAL_HU_LOW_BMD_THRESHOLD:
            warnings.append(
                f"Vertebral body HU {body_mean:.0f} suggests low bone density "
                f"(<{VERTEBRAL_HU_LOW_BMD_THRESHOLD:.0f} HU)"
            )
    if ratio is not None and ratio < TRAJECTORY_BODY_HU_RATIO_THRESHOLD:
        warnings.append(
            f"Trajectory/body HU ratio {ratio:.2f} below {TRAJECTORY_BODY_HU_RATIO_THRESHOLD:.1f}"
        )

    return BoneQualityMetrics(traj_mean, traj_min, pedicle_mean, body_mean, ratio, warnings)
