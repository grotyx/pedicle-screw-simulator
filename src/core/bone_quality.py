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

from ..utils.constants import (
    TRAJECTORY_BODY_HU_RATIO_THRESHOLD,
    TRAJECTORY_HU_LOOSENING_THRESHOLD,
    VERTEBRAL_HU_LOW_BMD_THRESHOLD,
    VERTEBRAL_HU_OSTEOPOROSIS_THRESHOLD,
)
from .screw_grading import ScrewGrader

Point3 = Sequence[float]

#: Radius around the isthmus centre within which a sample counts as "pedicle".
PEDICLE_ROI_RADIUS_MM = 10.0
#: Semi-axes (x, y, z in mm) of the default trabecular ROI at the body centre.
BODY_ROI_RADII_MM = (8.0, 8.0, 6.0)
#: Reference vertebral volume (mm3) the default ROI was sized for. Smaller
#: levels scale the ROI down by the cube root of the volume ratio so the
#: ellipsoid stays inside small bodies instead of spilling into cortex.
BODY_ROI_REFERENCE_VOLUME_MM3 = 30000.0
#: Clamp on the volume-derived linear scale: never enlarge past the default
#: (which would overflow) and never shrink below half (which would starve).
BODY_ROI_MIN_SCALE = 0.5
BODY_ROI_MAX_SCALE = 1.0
#: Fewer voxels than this and the ROI mean is too noisy to report.
MIN_ROI_VOXELS = 100


@dataclass
class BoneQualityMetrics:
    """HU metrics for one trajectory. Every value is ``None`` when unmeasurable."""

    trajectory_mean_hu: Optional[float] = None
    trajectory_min_hu: Optional[float] = None
    pedicle_mean_hu: Optional[float] = None
    body_mean_hu: Optional[float] = None
    trajectory_body_ratio: Optional[float] = None
    warnings: List[str] = field(default_factory=list)


def body_roi_radii_mm(
    volume_mm3: Optional[float] = None,
) -> Tuple[float, float, float]:
    """Trabecular ROI semi-axes scaled to the level's size.

    Linear scale is the cube root of ``volume_mm3`` over
    :data:`BODY_ROI_REFERENCE_VOLUME_MM3`, clamped to
    ``[BODY_ROI_MIN_SCALE, BODY_ROI_MAX_SCALE]``. ``None`` (or a
    non-positive volume) reproduces :data:`BODY_ROI_RADII_MM` exactly, so
    callers without a volume keep the old behavior.
    """
    if volume_mm3 is None or not np.isfinite(volume_mm3) or volume_mm3 <= 0.0:
        return BODY_ROI_RADII_MM
    scale = (float(volume_mm3) / BODY_ROI_REFERENCE_VOLUME_MM3) ** (1.0 / 3.0)
    scale = min(BODY_ROI_MAX_SCALE, max(BODY_ROI_MIN_SCALE, scale))
    return tuple(float(r) * scale for r in BODY_ROI_RADII_MM)


def vertebral_body_hu(
    ct: Optional[sitk.Image],
    mask: Optional[sitk.Image],
    label: int,
    body_center_lps: Point3,
    radii_mm: Optional[Tuple[float, float, float]] = None,
    volume_mm3: Optional[float] = None,
    ct_array: Optional[np.ndarray] = None,
    mask_array: Optional[np.ndarray] = None,
    origin_lps: Optional[Point3] = None,
    spacing_mm: Optional[Point3] = None,
    min_voxels: int = MIN_ROI_VOXELS,
) -> Optional[float]:
    """Mean HU of an ellipsoidal trabecular ROI at ``body_center_lps``.

    The ROI is intersected with ``label`` so cortex-adjacent background, discs
    and neighbouring vertebrae cannot skew the mean. Returns ``None`` when
    fewer than ``min_voxels`` voxels survive.

    ``radii_mm`` overrides the ROI semi-axes; when omitted they come from
    :func:`body_roi_radii_mm`, which shrinks the default for small
    ``volume_mm3`` and reproduces it when the volume is unknown.

    ``ct``/``mask`` keep the old SimpleITK signature, but callers that already
    hold the arrays (e.g. :class:`~src.core.screw_grading.ScrewGrader`) may
    pass ``ct_array``/``mask_array`` plus ``origin_lps``/``spacing_mm`` to skip
    the per-call ``GetArrayFromImage`` copies. Either pair suffices: arrays win
    when both are given. ``ct`` and ``mask`` must share a grid with an identity
    direction, which is what the grader already enforces.
    """
    if radii_mm is None:
        radii_mm = body_roi_radii_mm(volume_mm3)
    if ct_array is None:
        if ct is None:
            return None
        ct_array = sitk.GetArrayFromImage(ct)
    if mask_array is None:
        if mask is None:
            return None
        mask_array = sitk.GetArrayFromImage(mask)
    if origin_lps is None:
        if mask is None:
            return None
        origin = np.asarray(mask.GetOrigin(), dtype=np.float64)
    else:
        origin = np.asarray(origin_lps, dtype=np.float64)
    if spacing_mm is None:
        if mask is None:
            return None
        spacing = np.asarray(mask.GetSpacing(), dtype=np.float64)
    else:
        spacing = np.asarray(spacing_mm, dtype=np.float64)
    if np.any(spacing <= 0.0):
        return None
    centre_idx = (np.asarray(body_center_lps, dtype=np.float64) - origin) / spacing  # x, y, z
    radii_idx = np.asarray(radii_mm, dtype=np.float64) / spacing                     # x, y, z
    if np.any(~np.isfinite(radii_idx)) or np.any(radii_idx <= 0.0):
        return None

    # Work inside the ellipsoid's index-space bounding box: no voxel outside it
    # can satisfy the inequality, and a whole-volume grid would cost gigabytes
    # on a full-resolution CT.
    extent = np.asarray(mask_array.shape[::-1], dtype=np.int64)                      # x, y, z
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
    roi = ellipsoid & (mask_array[box] == label)
    if int(roi.sum()) < min_voxels:
        return None
    return float(ct_array[box][roi].mean())


def assess_bone_quality(
    grader: ScrewGrader,
    entry: Point3,
    target: Point3,
    diameter_mm: float,
    label: int,
    body_center_lps: Optional[Point3] = None,
    isthmus_center_lps: Optional[Point3] = None,
    trajectory_threshold: float = TRAJECTORY_HU_LOOSENING_THRESHOLD,
    volume_mm3: Optional[float] = None,
    radii_mm: Optional[Tuple[float, float, float]] = None,
) -> BoneQualityMetrics:
    """Measure trajectory, pedicle and vertebral-body HU and flag weak bone.

    ``body_center_lps`` and ``isthmus_center_lps`` come from the pedicle
    analyser; each metric that depends on one is simply omitted when it is not
    supplied, and every metric is omitted when the grader has no CT.
    ``volume_mm3``/``radii_mm`` forward to :func:`vertebral_body_hu`; both
    default to the legacy fixed ROI so existing callers are unchanged.

    The body ROI reuses the grader's cached arrays instead of copying the
    whole CT and mask again per screw.
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
            None,
            None,
            label,
            body_center_lps,
            radii_mm=radii_mm,
            volume_mm3=volume_mm3,
            ct_array=grader._ct_array,
            mask_array=grader._mask_array,
            origin_lps=grader._origin,
            spacing_mm=grader._spacing,
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
