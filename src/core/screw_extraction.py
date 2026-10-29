"""Extract implanted screws from a post-operative CT.

Metal is thresholded.  On real CTs blooming makes the rods and tulips thicker
than the shafts, so the thick metal (deeper than a radius, grown back by it)
is subtracted at growing radii and the freed shafts are fitted.  When that
finds nothing (screws thicker than their rod, as in a clean phantom), the
metal is instead eroded by growing radii until each screw breaks away and is
grown back.  All distances are millimetres, so the anisotropic voxel grid of
an axial series does not bias them.  A piece is kept if its shape fits a
pedicle screw: elongated, 20-70 mm long, 3-10 mm thick, thin tip, front to
back; pieces on one axis are joined.

A kept piece is measured in millimetres:

* **Axis** -- principal axis (PCA) of the shaft voxels.
* **Diameter** -- the voxels of a solid cylinder of radius ``r`` fill its
  cross-section evenly, so the 90th percentile of their distance from the axis
  is ``r * sqrt(0.9)``.  The percentile ignores the few voxels a thread crest or
  a touching neighbour adds, which a maximum would not.
* **Direction** -- a pedicle screw runs mainly front to back; a piece whose
  axis is mostly craniocaudal or transverse is a rod or cross-link.
* **Head** -- the end with the larger cross-section (the tulip or the head
  sphere).  When neither end is clearly wider, the dorsal end (larger LPS
  ``y``) is taken as the head, because pedicle screws are inserted from the
  back; such a screw is reported with lower confidence.
* **Entry** -- where the shaft leaves the head, so a tulip does not count
  towards the screw length; **target** -- the tip.  A shaft freed by
  subtraction lost its head, so its entry is the dorsal end, walked back over
  the neck the subtraction took.

Nothing here is clinically validated: real CT has streak and blooming
artefacts that a synthetic phantom does not, and the result must be checked
against the images.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import List, Optional, Tuple

import numpy as np
import SimpleITK as sitk
from scipy import ndimage

Point3 = Tuple[float, float, float]

#: Lowest HU counted as metal.  Cortical bone stays well below it; titanium and
#: cobalt-chrome implants read far above it on clinical scanners.
METAL_HU_THRESHOLD = 2000.0
#: Erosion radii tried in turn, millimetres.  Fine steps near the end because a
#: rod (5.5-6 mm) and a screw shaft (4.5-7.5 mm) differ by well under a
#: millimetre in radius, and the screw only comes loose in that window.
EROSION_RADII_MM = (0.0, 1.0, 1.5, 2.0, 2.5, 2.75, 3.0, 3.25, 3.5)
#: First pass, for real CTs where metal blooming makes the rod and tulips
#: thicker than the screw shafts (so erosion removes the shafts first): metal
#: deeper than each radius is grown back by that radius and subtracted, which
#: leaves the thinner shafts as separate pieces.  Ascending, so each shaft is
#: taken at the smallest radius that frees it.
THICK_RADII_MM = (3.0, 3.5, 4.0, 4.5, 5.0)
MIN_LENGTH_MM = 20.0
MAX_LENGTH_MM = 70.0
MIN_DIAMETER_MM = 3.0
MAX_DIAMETER_MM = 10.0
#: A piece outside this volume is not examined further (a clip; a whole rod
#: construct that has not yet come apart).
MIN_VOLUME_MM3 = 200.0
MAX_VOLUME_MM3 = 15000.0
#: sqrt(largest / second PCA eigenvalue).  A 6.5 x 45 mm cylinder scores ~8.
MIN_ELONGATION = 3.0
#: Smallest |A-P (LPS y) component| of a screw axis.  A pedicle screw with
#: 45 deg convergence and 30 deg craniocaudal tilt still scores 0.61; a rod
#: (craniocaudal) or cross-link (transverse) scores near 0.  On real post-op
#: CTs headless rod pieces otherwise passed every other check.
MIN_AP_COMPONENT = 0.5
#: A slab this many times wider than the shaft belongs to the head.
HEAD_AREA_RATIO = 1.5

_IDENTITY_DIRECTION = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
_SLAB_MM = 1.0


@dataclass
class ExtractedScrew:
    entry_lps: Point3
    target_lps: Point3
    diameter_mm: float
    length_mm: float
    voxel_count: int
    confidence: float
    reason: str


@dataclass
class ScrewExtraction:
    screws: List[ExtractedScrew] = field(default_factory=list)
    reason: str = ""


def extract_screws(
    ct_image: sitk.Image, threshold_hu: float = METAL_HU_THRESHOLD
) -> ScrewExtraction:
    """Every screw in ``ct_image`` (HU, identity direction, LPS)."""
    if not np.allclose(ct_image.GetDirection(), _IDENTITY_DIRECTION, atol=1e-6):
        raise ValueError(
            "ct_image must have an identity (LPS axis-aligned) direction matrix"
        )
    metal = sitk.GetArrayFromImage(ct_image) >= threshold_hu   # (z, y, x)
    if not metal.any():
        return ScrewExtraction(reason=f"No voxels at or above {threshold_hu:.0f} HU")

    sx, sy, sz = ct_image.GetSpacing()
    sampling = (sz, sy, sx)
    # Crop to the metal's bounding box: everything below runs on that alone.
    spans = [np.flatnonzero(metal.any(axis=other)) for other in ((1, 2), (0, 2), (0, 1))]
    lo = np.array([span[0] for span in spans])
    hi = np.array([span[-1] + 1 for span in spans])
    metal = metal[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
    origin = np.asarray(ct_image.GetOrigin()) + lo[::-1] * np.array([sx, sy, sz])
    spacing_xyz = np.array([sx, sy, sz])

    depth = ndimage.distance_transform_edt(np.pad(metal, 1), sampling=sampling)[
        1:-1, 1:-1, 1:-1
    ]
    remaining = metal.copy()
    screws: List[ExtractedScrew] = []
    objects = 0
    for radius in THICK_RADII_MM:
        core = remaining & (depth > radius)
        if not core.any():
            break
        reach = ndimage.distance_transform_edt(~core, sampling=sampling)
        shafts = remaining & (reach > radius + max(sampling))
        labels, _count = ndimage.label(shafts, structure=np.ones((3, 3, 3)))
        for index, box in enumerate(ndimage.find_objects(labels), start=1):
            if box is None:
                continue
            component = labels[box] == index
            screw = _fit_screw(component, box, origin, spacing_xyz)
            if screw is None:
                continue
            screws.append(
                _restore_neck(
                    screw, metal, depth, origin, spacing_xyz, radius + max(sampling)
                )
            )
            remaining[box][component] = False

    # Erosion only when subtraction found nothing: after it, erosion splits
    # the leftover metal into screw-length fragments.
    for radius in EROSION_RADII_MM if not screws else ():
        core = remaining & (depth > radius)
        if not core.any():
            break
        if radius > 0:
            # Grow the eroded pieces back, but only into metal not yet claimed.
            reach = ndimage.distance_transform_edt(~core, sampling=sampling)
            pieces = remaining & (reach <= radius + max(sampling))
        else:
            pieces = remaining
        labels, count = ndimage.label(pieces, structure=np.ones((3, 3, 3)))
        if radius == 0:
            objects = count
        for index, box in enumerate(ndimage.find_objects(labels), start=1):
            if box is None:
                continue
            component = labels[box] == index
            screw = _fit_screw(component, box, origin, spacing_xyz)
            if screw is None:
                continue
            screws.append(screw)
            remaining[box][component] = False

    screws = _merge_collinear(screws)
    if not screws:
        return ScrewExtraction(
            reason=(
                f"{objects} metal object(s) found, none shaped like a screw "
                f"({MIN_LENGTH_MM:.0f}-{MAX_LENGTH_MM:.0f} mm long, "
                f"{MIN_DIAMETER_MM:.0f}-{MAX_DIAMETER_MM:.0f} mm thick)"
            )
        )
    return ScrewExtraction(screws=screws, reason=f"{len(screws)} screw(s) found")



def _restore_neck(
    screw, metal, depth, origin, spacing_xyz, removed_mm: float
) -> ExtractedScrew:
    """Orient a shaft freed by subtraction and give back its neck.

    The head went with the thick metal, so the wide-end rule saw nothing: a
    pedicle screw enters from the back (larger LPS y).  Subtraction also took
    the shaft within ``removed_mm`` of the head; the entry walks back along
    the axis, at most that far, through metal no deeper than the shaft.
    """
    entry = np.asarray(screw.entry_lps, dtype=np.float64)
    target = np.asarray(screw.target_lps, dtype=np.float64)
    if entry[1] < target[1]:
        entry, target = target, entry
    axis = (entry - target) / np.linalg.norm(entry - target)
    step = 0.25
    limit = screw.diameter_mm / 2.0 + float(spacing_xyz.max())
    for _ in range(int(removed_mm / step)):
        candidate = entry + axis * step
        index = np.round((candidate - origin) / spacing_xyz).astype(int)[::-1]
        if np.any(index < 0) or np.any(index >= metal.shape):
            break
        if not metal[tuple(index)] or depth[tuple(index)] > limit:
            break
        entry = candidate
    return replace(
        screw,
        entry_lps=tuple(float(v) for v in entry),
        target_lps=tuple(float(v) for v in target),
        length_mm=float(np.linalg.norm(entry - target)),
        confidence=0.8,
        reason="shaft freed from the rod and tulip; entry at the dorsal end",
    )


#: Two pieces this close to one line are one screw split in two (coarse CT).
MERGE_ANGLE_DEG = 10.0
MERGE_OFFSET_MM = 3.0


def _merge_collinear(screws: List[ExtractedScrew]) -> List[ExtractedScrew]:
    """Join pieces that lie on one axis; the longer piece keeps its fit."""
    merged: List[ExtractedScrew] = []
    for screw in sorted(screws, key=lambda s: -s.length_mm):
        for k, kept in enumerate(merged):
            a = np.asarray(kept.entry_lps)
            axis = np.asarray(kept.target_lps) - a
            axis /= np.linalg.norm(axis)
            other = np.asarray(screw.target_lps) - np.asarray(screw.entry_lps)
            cos = abs(float(np.dot(axis, other))) / float(np.linalg.norm(other))
            if cos < np.cos(np.radians(MERGE_ANGLE_DEG)):
                continue
            ends = [np.asarray(p) for p in (screw.entry_lps, screw.target_lps)]
            offsets = [np.linalg.norm((p - a) - np.dot(p - a, axis) * axis) for p in ends]
            if max(offsets) > MERGE_OFFSET_MM:
                continue
            # Span both pieces along the kept axis, entry at the dorsal end.
            ts = [0.0, float(np.linalg.norm(np.asarray(kept.target_lps) - a))]
            ts += [float(np.dot(p - a, axis)) for p in ends]
            lo, hi = a + axis * min(ts), a + axis * max(ts)
            entry, target = (lo, hi) if lo[1] >= hi[1] else (hi, lo)
            merged[k] = replace(
                kept,
                entry_lps=tuple(float(v) for v in entry),
                target_lps=tuple(float(v) for v in target),
                length_mm=float(max(ts) - min(ts)),
                voxel_count=kept.voxel_count + screw.voxel_count,
                reason=f"{kept.reason}; joined with a second piece on the same axis",
            )
            break
        else:
            merged.append(screw)
    return merged


def _fit_screw(
    component: np.ndarray, box, origin: np.ndarray, spacing_xyz: np.ndarray
) -> Optional[ExtractedScrew]:
    """Measure one connected piece of metal; ``None`` if it is not a screw."""
    voxel_mm3 = float(np.prod(spacing_xyz))
    count = int(component.sum())
    if not MIN_VOLUME_MM3 <= count * voxel_mm3 <= MAX_VOLUME_MM3:
        return None
    zyx = np.argwhere(component) + np.array([s.start for s in box])
    points = origin + zyx[:, ::-1] * spacing_xyz          # (N, 3) LPS mm

    centre, axis = _principal_axis(points)
    if axis is None:
        return None
    t = (points - centre) @ axis
    areas, edges = _slab_areas(t, voxel_mm3)
    if len(areas) < 4:
        return None
    n = len(areas)
    shaft_area = float(np.median(areas[n // 4: max(n // 4 + 1, 3 * n // 4)]))
    end = max(3, int(round(0.15 * n)))
    low_end, high_end = float(areas[:end].mean()), float(areas[-end:].mean())

    if max(low_end, high_end) >= 1.3 * min(low_end, high_end):
        head_low = low_end > high_end
        confidence, how = 0.9, "head at the wider end"
    else:
        # Neither end is clearly wider: take the dorsal (posterior, +y) end.
        y_low = centre[1] + axis[1] * t.min()
        y_high = centre[1] + axis[1] * t.max()
        head_low = y_low > y_high
        confidence, how = 0.6, (
            "no wider head found, head assumed at the dorsal end; "
            "may be a rod or connector piece"
        )
    tip_area = high_end if head_low else low_end
    if tip_area > HEAD_AREA_RATIO * shaft_area:
        return None                                    # wide at both ends: rod, connector

    # Trim the head: everything up to the innermost wide slab in the head half.
    # (Not "while wide": the outermost slab of a head is often a thin sliver.)
    wide = areas > HEAD_AREA_RATIO * shaft_area
    from_head = wide if head_low else wide[::-1]
    wide_in_half = np.flatnonzero(from_head[: n // 2])
    head_slabs = int(wide_in_half[-1]) + 1 if wide_in_half.size else 0
    if head_low:
        shaft = t >= edges[head_slabs]
    else:
        shaft = t <= edges[n - head_slabs]
    if shaft.sum() < 10:
        return None
    head_point = centre + axis * (t.min() if head_low else t.max())

    # Refit on the shaft alone: a tulip off to one side would tilt the axis.
    centre, axis = _principal_axis(points[shaft])
    # On the shaft axis: a tulip with a rod stub can tilt the whole piece.
    if axis is None or abs(axis[1]) < MIN_AP_COMPONENT:
        return None                                    # rod or cross-link
    t = (points[shaft] - centre) @ axis
    radial = np.linalg.norm(points[shaft] - centre - t[:, None] * axis, axis=1)
    if np.dot(head_point - centre, axis) < 0:
        entry_t, tip_t = t.min(), t.max()
    else:
        entry_t, tip_t = t.max(), t.min()
    length = abs(tip_t - entry_t)
    middle = np.abs(t - (entry_t + tip_t) / 2.0) <= 0.3 * length
    if not middle.any():
        return None
    diameter = 2.0 * float(np.percentile(radial[middle], 90)) / np.sqrt(0.9)
    if not MIN_LENGTH_MM <= length <= MAX_LENGTH_MM:
        return None
    if not MIN_DIAMETER_MM <= diameter <= MAX_DIAMETER_MM:
        return None
    entry = centre + axis * entry_t
    target = centre + axis * tip_t
    return ExtractedScrew(
        entry_lps=tuple(float(v) for v in entry),
        target_lps=tuple(float(v) for v in target),
        diameter_mm=diameter,
        length_mm=float(length),
        voxel_count=count,
        confidence=confidence,
        reason=how,
    )


def _principal_axis(points: np.ndarray):
    """Centroid and unit principal axis, or ``(centroid, None)`` if not elongated."""
    centre = points.mean(axis=0)
    eigvals, eigvecs = np.linalg.eigh(np.cov((points - centre).T))
    if eigvals[1] <= 0 or np.sqrt(eigvals[2] / eigvals[1]) < MIN_ELONGATION:
        return centre, None
    return centre, eigvecs[:, 2]


def _slab_areas(t: np.ndarray, voxel_mm3: float):
    """Cross-section area (mm^2) of each 1 mm slab along the axis, and the slab edges."""
    edges = np.arange(t.min(), t.max() + _SLAB_MM, _SLAB_MM)
    counts, edges = np.histogram(t, bins=edges)
    return counts * voxel_mm3 / _SLAB_MM, edges
