"""Plan-vs-plan comparison metrics for pedicle screw plans.

Compares a *predicted* plan (automatic planner output, trainee plan, ...)
against a *reference* plan (expert plan, ground truth) and reports the
deviation measures used in the pedicle-screw planning literature:

* **Head/tip MAD** - Euclidean entry- and tip-point deviations, and the 3D
  angle between the screw axes (Scherer 2022).
* **Angle deltas** - convergence (axial) and craniocaudal (sagittal)
  differences, plus screw-volume Dice (Wang 2024).
* **Pedicle-centre offset** - perpendicular distance from the predicted screw
  axis to the reference pedicle centre, when the reference plan supplies one
  (Massalimova 2025).
* **Bland-Altman** - bias and 95% limits of agreement for diameter and length
  (Herkner 2026).

The module is deliberately Qt-free (numpy + stdlib only) so it can be reused
from scripts, notebooks and CI as well as from the desktop app.

All coordinates are LPS millimetres, matching :class:`src.models.screw.Screw`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..models.screw import Screw
from ..utils.planning_io import deserialize_plan

Point3 = Sequence[float]

#: Default voxel edge length (mm) for the Dice overlap rasterisation.
DEFAULT_VOXEL_MM = 0.5

#: Bland-Altman limits of agreement multiplier (95% interval).
LOA_Z = 1.96

#: Upper bound on the rasterisation grid; the voxel size is coarsened rather
#: than allocating an unbounded array for pathologically long screws.
MAX_DICE_VOXELS = 8_000_000

_EPS = 1e-12


@dataclass
class ScrewComparison:
    """Per-screw deviation of a predicted screw from its reference screw."""

    level: str
    side: str
    head_mad_mm: float
    tip_mad_mm: float
    axis_angle_deg: float
    convergence_delta_deg: float
    craniocaudal_delta_deg: float
    diameter_delta_mm: float
    length_delta_mm: float
    pedicle_center_offset_mm: Optional[float]
    dice: float


@dataclass
class ComparisonSummary:
    """Cohort-level summary over a list of :class:`ScrewComparison`."""

    n_matched: int
    n_unmatched_pred: int
    n_unmatched_ref: int
    head_mad_mean: float
    head_mad_sd: float
    tip_mad_mean: float
    tip_mad_sd: float
    axis_angle_mean: float
    axis_angle_sd: float
    diameter_bias: float
    diameter_loa: Tuple[float, float]
    length_bias: float
    length_loa: Tuple[float, float]
    dice_mean: float


# --- geometry helpers ----------------------------------------------------


def _as_vector(point: Point3) -> np.ndarray:
    return np.asarray(point, dtype=float).reshape(3)


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= _EPS:
        return np.zeros(3, dtype=float)
    return vector / norm


def point_to_axis_distance(
    point: Point3, entry: Point3, target: Point3
) -> Optional[float]:
    """Perpendicular distance from ``point`` to the infinite screw axis.

    Returns ``None`` when the screw is degenerate (entry == target) and no
    axis direction can be defined.
    """
    origin = _as_vector(entry)
    direction = _unit(_as_vector(target) - origin)
    if not direction.any():
        return None
    offset = _as_vector(point) - origin
    perpendicular = offset - float(np.dot(offset, direction)) * direction
    return float(np.linalg.norm(perpendicular))


def axis_angle_deg(direction_a: Point3, direction_b: Point3) -> float:
    """Unsigned angle in degrees between two screw axes (0-180)."""
    unit_a = _unit(_as_vector(direction_a))
    unit_b = _unit(_as_vector(direction_b))
    if not unit_a.any() or not unit_b.any():
        return 0.0
    cosine = float(np.clip(np.dot(unit_a, unit_b), -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def _cylinder_mask(
    grid: Tuple[np.ndarray, np.ndarray, np.ndarray],
    entry: np.ndarray,
    target: np.ndarray,
    radius: float,
) -> np.ndarray:
    """Boolean mask of voxel centres inside a capped cylinder."""
    axis = target - entry
    length = float(np.linalg.norm(axis))
    if length <= _EPS or radius <= 0.0:
        return np.zeros(grid[0].shape, dtype=bool)
    direction = axis / length

    dx = grid[0] - entry[0]
    dy = grid[1] - entry[1]
    dz = grid[2] - entry[2]
    # Projection onto the axis; inside the caps when 0 <= t <= length.
    t = dx * direction[0] + dy * direction[1] + dz * direction[2]
    # Perpendicular distance squared = |p - e|^2 - t^2.
    perp_sq = dx * dx + dy * dy + dz * dz - t * t
    np.maximum(perp_sq, 0.0, out=perp_sq)
    return (t >= 0.0) & (t <= length) & (perp_sq <= radius * radius)


def cylinder_dice(
    entry_a: Point3,
    target_a: Point3,
    diameter_a: float,
    entry_b: Point3,
    target_b: Point3,
    diameter_b: float,
    voxel_mm: float = DEFAULT_VOXEL_MM,
) -> float:
    """Dice overlap of two screw cylinders, rasterised at ``voxel_mm``.

    Both cylinders are voxelised on one common grid spanning their combined
    bounding boxes; a voxel centre counts as inside when its axial parameter
    lies within the screw length and its perpendicular distance is within the
    radius.  Returns ``2|A and B| / (|A| + |B|)``, or 0.0 when either screw
    has no volume.
    """
    if voxel_mm <= 0.0:
        raise ValueError("voxel_mm must be positive")

    entry_a_v, target_a_v = _as_vector(entry_a), _as_vector(target_a)
    entry_b_v, target_b_v = _as_vector(entry_b), _as_vector(target_b)
    radius_a = float(diameter_a) / 2.0
    radius_b = float(diameter_b) / 2.0

    length_a = float(np.linalg.norm(target_a_v - entry_a_v))
    length_b = float(np.linalg.norm(target_b_v - entry_b_v))
    if radius_a <= 0.0 or radius_b <= 0.0 or length_a <= _EPS or length_b <= _EPS:
        return 0.0

    lower = np.minimum(
        np.minimum(entry_a_v, target_a_v) - radius_a,
        np.minimum(entry_b_v, target_b_v) - radius_b,
    )
    upper = np.maximum(
        np.maximum(entry_a_v, target_a_v) + radius_a,
        np.maximum(entry_b_v, target_b_v) + radius_b,
    )

    spacing = _grid_spacing(lower, upper, voxel_mm)
    axes = [
        np.arange(lower[i] + spacing / 2.0, upper[i] + spacing / 2.0, spacing)
        for i in range(3)
    ]
    if any(axis.size == 0 for axis in axes):
        return 0.0
    grid = np.meshgrid(axes[0], axes[1], axes[2], indexing="ij")

    mask_a = _cylinder_mask(grid, entry_a_v, target_a_v, radius_a)
    mask_b = _cylinder_mask(grid, entry_b_v, target_b_v, radius_b)

    count_a = int(mask_a.sum())
    count_b = int(mask_b.sum())
    if count_a + count_b == 0:
        return 0.0
    intersection = int(np.count_nonzero(mask_a & mask_b))
    return 2.0 * intersection / float(count_a + count_b)


def _grid_spacing(lower: np.ndarray, upper: np.ndarray, voxel_mm: float) -> float:
    """Voxel size for the Dice grid, coarsened if the grid would be huge."""
    extents = np.maximum(upper - lower, voxel_mm)
    counts = np.ceil(extents / voxel_mm)
    total = float(np.prod(counts))
    if total <= MAX_DICE_VOXELS:
        return voxel_mm
    return voxel_mm * (total / MAX_DICE_VOXELS) ** (1.0 / 3.0)


# --- per-screw comparison ------------------------------------------------


def compare_screw(
    pred: Screw,
    ref: Screw,
    ref_pedicle_center: Optional[Point3] = None,
    voxel_mm: float = DEFAULT_VOXEL_MM,
) -> ScrewComparison:
    """Compare one predicted screw against its reference screw."""
    entry_pred = _as_vector(pred.entry_point)
    entry_ref = _as_vector(ref.entry_point)
    target_pred = _as_vector(pred.target_point)
    target_ref = _as_vector(ref.target_point)

    offset: Optional[float] = None
    if ref_pedicle_center is not None:
        offset = point_to_axis_distance(
            ref_pedicle_center, pred.entry_point, pred.target_point
        )

    return ScrewComparison(
        level=pred.vertebra_level or ref.vertebra_level,
        side=pred.side or ref.side,
        head_mad_mm=float(np.linalg.norm(entry_pred - entry_ref)),
        tip_mad_mm=float(np.linalg.norm(target_pred - target_ref)),
        axis_angle_deg=axis_angle_deg(pred.trajectory, ref.trajectory),
        convergence_delta_deg=float(pred.medial_angle - ref.medial_angle),
        craniocaudal_delta_deg=float(pred.insertion_angle - ref.insertion_angle),
        diameter_delta_mm=float(pred.diameter - ref.diameter),
        length_delta_mm=float(pred.length - ref.length),
        pedicle_center_offset_mm=offset,
        dice=cylinder_dice(
            pred.entry_point,
            pred.target_point,
            pred.diameter,
            ref.entry_point,
            ref.target_point,
            ref.diameter,
            voxel_mm=voxel_mm,
        ),
    )


# --- matching ------------------------------------------------------------


def _match_key(screw: Screw) -> Tuple[str, str]:
    return (screw.vertebra_level, screw.side)


def match_screws(
    pred: List[Screw], ref: List[Screw]
) -> Tuple[List[Tuple[Screw, Screw]], List[Screw], List[Screw]]:
    """Pair predicted with reference screws by ``(vertebra_level, side)``.

    When several screws share a key (revision constructs, mislabelled sides),
    the candidates are paired greedily by nearest entry point.  Returns
    ``(pairs, unmatched_pred, unmatched_ref)`` with pairs in predicted-list
    order and the unmatched lists in their original order.
    """
    ref_by_key: Dict[Tuple[str, str], List[int]] = {}
    for index, screw in enumerate(ref):
        ref_by_key.setdefault(_match_key(screw), []).append(index)

    matched_pred: Dict[int, int] = {}
    matched_ref: set[int] = set()

    pred_by_key: Dict[Tuple[str, str], List[int]] = {}
    for index, screw in enumerate(pred):
        pred_by_key.setdefault(_match_key(screw), []).append(index)

    for key, pred_indices in pred_by_key.items():
        ref_indices = ref_by_key.get(key, [])
        if not ref_indices:
            continue
        # Greedy nearest-entry assignment: shortest candidate pair first.
        candidates = []
        for p_index in pred_indices:
            entry_p = _as_vector(pred[p_index].entry_point)
            for r_index in ref_indices:
                distance = float(
                    np.linalg.norm(entry_p - _as_vector(ref[r_index].entry_point))
                )
                candidates.append((distance, p_index, r_index))
        candidates.sort(key=lambda item: (item[0], item[1], item[2]))
        for _distance, p_index, r_index in candidates:
            if p_index in matched_pred or r_index in matched_ref:
                continue
            matched_pred[p_index] = r_index
            matched_ref.add(r_index)

    pairs = [
        (pred[p_index], ref[matched_pred[p_index]])
        for p_index in range(len(pred))
        if p_index in matched_pred
    ]
    unmatched_pred = [
        screw for index, screw in enumerate(pred) if index not in matched_pred
    ]
    unmatched_ref = [
        screw for index, screw in enumerate(ref) if index not in matched_ref
    ]
    return pairs, unmatched_pred, unmatched_ref


# --- summary -------------------------------------------------------------


def _mean_sd(values: Sequence[float]) -> Tuple[float, float]:
    """Mean and sample SD; SD is 0.0 for fewer than two values."""
    if not values:
        return 0.0, 0.0
    array = np.asarray(values, dtype=float)
    mean = float(array.mean())
    sd = float(array.std(ddof=1)) if array.size > 1 else 0.0
    return mean, sd


def _bland_altman(deltas: Sequence[float]) -> Tuple[float, Tuple[float, float]]:
    """Bland-Altman bias and 95% limits of agreement (bias +- 1.96 SD)."""
    bias, sd = _mean_sd(deltas)
    return bias, (bias - LOA_Z * sd, bias + LOA_Z * sd)


def summarize(
    comparisons: List[ScrewComparison],
    unmatched_pred: int,
    unmatched_ref: int,
) -> ComparisonSummary:
    """Aggregate per-screw comparisons into cohort statistics."""
    head_mean, head_sd = _mean_sd([c.head_mad_mm for c in comparisons])
    tip_mean, tip_sd = _mean_sd([c.tip_mad_mm for c in comparisons])
    angle_mean, angle_sd = _mean_sd([c.axis_angle_deg for c in comparisons])
    diameter_bias, diameter_loa = _bland_altman(
        [c.diameter_delta_mm for c in comparisons]
    )
    length_bias, length_loa = _bland_altman([c.length_delta_mm for c in comparisons])
    dice_mean, _ = _mean_sd([c.dice for c in comparisons])

    return ComparisonSummary(
        n_matched=len(comparisons),
        n_unmatched_pred=int(unmatched_pred),
        n_unmatched_ref=int(unmatched_ref),
        head_mad_mean=head_mean,
        head_mad_sd=head_sd,
        tip_mad_mean=tip_mean,
        tip_mad_sd=tip_sd,
        axis_angle_mean=angle_mean,
        axis_angle_sd=angle_sd,
        diameter_bias=diameter_bias,
        diameter_loa=diameter_loa,
        length_bias=length_bias,
        length_loa=length_loa,
        dice_mean=dice_mean,
    )


# --- plan-level entry point ---------------------------------------------


def _pedicle_centers(payload: Dict[str, Any]) -> List[Optional[Tuple[float, float, float]]]:
    """Optional ``pedicle_center`` per raw screw item, in payload order.

    ``deserialize_plan`` does not model this key, so it is read straight from
    the raw payload and matched to the deserialised screws by index.
    """
    items = payload.get("screws", []) if isinstance(payload, dict) else []
    centers: List[Optional[Tuple[float, float, float]]] = []
    for item in items if isinstance(items, list) else []:
        center = item.get("pedicle_center") if isinstance(item, dict) else None
        if isinstance(center, (list, tuple)) and len(center) == 3:
            try:
                centers.append(tuple(float(value) for value in center))
                continue
            except (TypeError, ValueError):
                pass
        centers.append(None)
    return centers


def compare_plans(
    pred_payload: Dict[str, Any],
    ref_payload: Dict[str, Any],
    voxel_mm: float = DEFAULT_VOXEL_MM,
) -> Tuple[List[ScrewComparison], ComparisonSummary]:
    """Compare two plan JSON payloads (any schema version).

    Reference screws may carry an optional ``"pedicle_center": [x, y, z]``
    entry, which feeds :attr:`ScrewComparison.pedicle_center_offset_mm`.
    """
    pred_screws: List[Screw] = deserialize_plan(pred_payload)["screws"]
    ref_screws: List[Screw] = deserialize_plan(ref_payload)["screws"]

    centers = _pedicle_centers(ref_payload)
    center_by_screw = {
        id(screw): center
        for screw, center in zip(ref_screws, centers, strict=True)
    }

    pairs, unmatched_pred, unmatched_ref = match_screws(pred_screws, ref_screws)
    comparisons = [
        compare_screw(
            pred,
            ref,
            ref_pedicle_center=center_by_screw.get(id(ref)),
            voxel_mm=voxel_mm,
        )
        for pred, ref in pairs
    ]
    return comparisons, summarize(comparisons, len(unmatched_pred), len(unmatched_ref))
