"""Candidate-based multi-objective pedicle screw trajectory optimisation.

The optimiser enumerates a dense grid of straight trajectories for one pedicle
(entry offsets x convergence x craniocaudal angulation x catalogue length),
grades all of them in a single vectorised :meth:`ScrewGrader.evaluate_batch`
pass, drops the infeasible ones and ranks the survivors with a weighted sum of
normalised objectives.

All geometry is in the DICOM **LPS** frame used across the app: ``+X`` left,
``+Y`` posterior, ``+Z`` superior.  Pedicle axes coming out of
:class:`~src.core.vertebra.PedicleAnalysisResult` are posterior-oriented, so the
insertion direction is their negation.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import SimpleITK as sitk

from .planner_config import PlannerConfig
from .screw_grading import BatchResult, ScrewGrader
from .vertebra import PedicleAnalysisResult

logger = logging.getLogger(__name__)

#: Wall distance at which the safety objective saturates (mm).
SAFETY_CAP_MM = 3.0

#: HU window the density objective is normalised over.
DENSITY_LOW_HU = 100.0
DENSITY_HIGH_HU = 600.0

#: Deviation from the endplate plane that drives the endplate objective to 0.
ENDPLATE_TOLERANCE_DEG = 15.0

#: Length of the distal segment the anterior-margin check is measured over (mm).
TIP_SEGMENT_MM = 4.0

#: How far the seated entry may sit anterior of the posterior cortex before the
#: trajectory counts as unreachable.  A pedicle is continuous with the lamina, so
#: a few millimetres of countersinking is normal; more than this means a drill
#: would have to cross air (or another structure) to reach the corridor.
MAX_ENTRY_SHORTFALL_MM = 3.0

#: How many catalogue steps below the recommended diameter the search may go
#: before giving up.  Bounds the worst-case runtime of :func:`optimize_screw`.
MAX_DIAMETER_STEPS = 2

#: Head-misalignment (mm) at which the rod objective costs a full unit of score.
ROD_TOLERANCE_MM = 3.0

#: Fraction of a screw's own best score the construct re-ranking may trade away.
CONSTRUCT_SCORE_TOLERANCE = 0.10

#: Cap on the greedy coordinate-descent sweeps over the construct.
_CONSTRUCT_MAX_PASSES = 5

#: Step and reach of the 1-D scan that seats an entry point on the posterior
#: surface of the screw corridor (mm).
_ANCHOR_STEP_MM = 0.5
_ANCHOR_MAX_MM = 40.0

@dataclass(frozen=True)
class OptimizerWeights:
    """Relative importance of each normalised objective.

    Every component is clipped to ``[0, 1]`` before weighting, so a weight is
    directly comparable to the others.  ``rod`` is unused by the single-screw
    scoring; it is consumed by :func:`optimize_construct`.
    """

    safety: float = 1.0      # min wall distance (capped at 3 mm, normalised)
    density: float = 0.5     # trajectory mean HU, normalised (100..600 HU -> 0..1)
    length: float = 0.2      # length / max catalogue length
    endplate: float = 0.3    # 1 - |angle to the endplate plane| / 15 deg (parallel = 1)
    centering: float = 0.3   # 1 - distance(line, isthmus centre) / (pedicle_width / 2)
    rod: float = 0.3         # used by the construct optimiser only


#: Shared immutable default, used as the ``optimize_screw`` weights default.
DEFAULT_WEIGHTS = OptimizerWeights()


@dataclass
class Candidate:
    """One scored, feasible trajectory."""

    entry: np.ndarray
    target: np.ndarray
    length: float
    diameter: float
    breach_mm: float
    min_wall_mm: float
    mean_hu: float
    convergence_deg: float
    craniocaudal_deg: float
    score: float
    #: How far anterior of the posterior cortex the entry had to be seated, in
    #: mm along the trajectory; 0 when the entry sits on the cortex itself.
    surface_shortfall_mm: float = 0.0
    components: Dict[str, float] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)


# --------------------------------------------------------------------- helpers
def _rotate(v: np.ndarray, axis: np.ndarray, deg: float) -> np.ndarray:
    """Rodrigues rotation of ``v`` about ``axis`` by ``deg`` degrees."""
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    a = math.radians(float(deg))
    return (
        v * math.cos(a)
        + np.cross(axis, v) * math.sin(a)
        + axis * float(np.dot(axis, v)) * (1.0 - math.cos(a))
    )


#: Guard for the "grader has no CT" warning, which is a property of the loaded
#: study rather than of any one candidate and would otherwise repeat per scoring
#: pass.
_missing_ct_warned = False


def _warn_missing_ct() -> None:
    global _missing_ct_warned
    if not _missing_ct_warned:
        _missing_ct_warned = True
        logger.warning(
            "Grader has no CT: trajectory HU is unavailable, scoring every candidate "
            "with a density component of 0"
        )


def _unit(v: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(v))
    if norm <= 1e-12:
        return np.zeros(3, dtype=np.float64)
    return np.asarray(v, dtype=np.float64) / norm


def _trajectory_angles(
    deltas: np.ndarray, side: str
) -> Tuple[np.ndarray, np.ndarray]:
    """Vectorised ``screw_geometry`` convergence and craniocaudal angles (degrees).

    Closed form over ``(C, 3)`` entry->target deltas, matching
    :func:`~src.core.screw_geometry.convergence_angle_deg` and
    :func:`~src.core.screw_geometry.craniocaudal_angle_deg` exactly, including
    their degenerate cases.  Positive convergence is medial for ``side``;
    positive craniocaudal points the tip superiorly.
    """
    dx, dy, dz = deltas[:, 0], deltas[:, 1], deltas[:, 2]
    horizontal = np.hypot(dx, dy)
    magnitude = np.degrees(np.arctan2(np.abs(dx), np.abs(dy)))
    medial = dx <= 0.0 if side == "left" else dx >= 0.0
    convergence = np.where(horizontal <= 1e-9, 0.0, np.where(medial, magnitude, -magnitude))
    craniocaudal = np.where(
        (horizontal <= 1e-9) & (np.abs(dz) <= 1e-9), 0.0, np.degrees(np.arctan2(dz, horizontal))
    )
    return convergence, craniocaudal


def _side_data(
    analysis: PedicleAnalysisResult, side: str
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], float]:
    if side not in ("left", "right"):
        raise ValueError(f"side must be 'left' or 'right', got {side!r}")
    if side == "left":
        return (
            analysis.left_pedicle_center,
            analysis.left_pedicle_axis,
            float(analysis.left_pedicle_width),
        )
    return (
        analysis.right_pedicle_center,
        analysis.right_pedicle_axis,
        float(analysis.right_pedicle_width),
    )


def make_planner(grader: ScrewGrader, config: PlannerConfig):
    """An :class:`AutoScrewPlanner` over the grader's own volumes.

    Only its entry-point search and diameter rules are reused.
    :class:`AutoScrewPlanner` still requires a CT to construct (it sizes and
    caches its own array), so a grader built without one gets a zero-filled
    stand-in; every HU the optimiser reports comes from the grader, never from
    this image.
    """
    from .auto_screw_planner import AutoScrewPlanner  # local import: avoids a cycle

    mask = grader.mask_image()
    ct = grader.ct_image()
    if ct is None:
        ct = sitk.Image(mask.GetSize(), sitk.sitkInt16)
        ct.CopyInformation(mask)
    return AutoScrewPlanner(ct, mask, grader=grader, config=config)


def _corridor_radius(width_mm: float, config: PlannerConfig) -> float:
    """Half the widest screw the measured pedicle can hold, per the config rules."""
    available = min(
        config.pedicle_fill_ratio * width_mm,
        width_mm - 2.0 * config.wall_clearance_mm,
    )
    smallest = min(config.implant_diameters_mm)
    return max(available, smallest) / 2.0


def _seat_entries(
    grader: ScrewGrader,
    seeds: np.ndarray,
    directions: np.ndarray,
    limits: np.ndarray,
    label: int,
    radius_mm: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Back each seed out to the posterior surface of the screw corridor.

    Marching posteriorly from inside the pedicle, the last point whose
    inside-distance still exceeds ``radius_mm`` is the most posterior start from
    which the screw cross-section is guaranteed to be contained -- the corridor
    surface.  ``limits`` (mm, per seed) caps the march at the posterior cortex
    reported by :meth:`AutoScrewPlanner._find_entry_point`, so on a wide corridor
    the scan simply stops at the cortex.

    Returns ``(entries, travel_mm)``; ``limits - travel_mm`` is how far short of
    the cortex each entry had to stop.
    """
    # Never scan past the furthest cortex any seed reported: the march exists to
    # find a surface at most that far back, so a fixed 40 mm sweep would be
    # mostly wasted lookups.
    reach_mm = float(np.clip(limits.max(initial=0.0), 0.0, _ANCHOR_MAX_MM))
    steps = np.arange(0.0, reach_mm + 1e-9, _ANCHOR_STEP_MM)
    points = seeds[:, None, :] - directions[:, None, :] * steps[None, :, None]
    d_out, d_in = grader.distances_at_points(points.reshape(-1, 3), label)
    ok = ((d_out <= 0.0) & (d_in >= radius_mm)).reshape(seeds.shape[0], steps.size)
    ok &= steps[None, :] <= limits[:, None]
    # Leading run of valid steps; its last member is the corridor surface. A seed
    # whose own cross-section does not fit stays put (run length 0).
    reach = np.logical_and.accumulate(ok, axis=1).sum(axis=1)
    travel = np.where(reach > 0, steps[np.maximum(reach - 1, 0)], 0.0)
    return seeds - directions * travel[:, None], travel


# ------------------------------------------------------------------ generation
def generate_candidates(
    grader: ScrewGrader,
    analysis: PedicleAnalysisResult,
    side: str,
    config: PlannerConfig,
    entry_grid_mm: Sequence[float] = (-2.0, -1.0, 0.0, 1.0, 2.0),
    convergence_step_deg: float = 2.5,
    craniocaudal_range_deg: Tuple[float, float] = (-10.0, 10.0),
    craniocaudal_step_deg: float = 5.0,
    corridor_radius_mm: Optional[float] = None,
    planner=None,
    diagnostics: Optional[Dict[str, np.ndarray]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Enumerate straight trajectories for one pedicle.

    Returns ``(entries (C, 3), targets (C, 3), lengths (C,))`` covering every
    entry offset x direction x catalogue length whose tip still lands inside the
    vertebra.  Directions rotate the anterior-pointing pedicle axis about ``Z``
    by the convergence angle (positive = medial for the given side) and about the
    lateral axis by the craniocaudal angle (positive = cranial).  Entry points
    are seated on the posterior surface of the screw corridor along their own
    direction, so each direction gets the most posterior start that can still
    hold the screw.

    Feasibility (containment, wall clearance, anterior margin, reachability) is
    *not* checked here -- :func:`score_candidates` does that from the batch
    grading.  Pass a dict as ``diagnostics`` to receive per-candidate arrays
    aligned with the return value; it currently carries
    ``"surface_shortfall_mm"``, how far anterior of the posterior cortex each
    entry had to be seated.
    """
    empty = (np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0))
    center, axis, width = _side_data(analysis, side)
    if center is None or axis is None:
        return empty
    lengths_catalogue = np.asarray(config.implant_lengths_mm, dtype=np.float64)
    if lengths_catalogue.size == 0:
        return empty

    center = np.asarray(center, dtype=np.float64)
    posterior_axis = _unit(axis)
    if posterior_axis[1] < 0.0:
        posterior_axis = -posterior_axis
    base = -posterior_axis                                  # insertion direction
    z_axis = np.array([0.0, 0.0, 1.0])
    lateral_axis = _unit(np.cross(z_axis, base))
    if not lateral_axis.any():                              # axis parallel to Z
        lateral_axis = np.array([1.0, 0.0, 0.0])
    # Rotating about ``+lateral_axis`` tips the trajectory caudally, so the
    # craniocaudal sweep uses its negation to keep "positive = cranial".
    craniocaudal_axis = -lateral_axis
    medial_sign = -1.0 if side == "left" else 1.0

    directions = np.array(
        [
            _unit(_rotate(_rotate(base, z_axis, medial_sign * conv), craniocaudal_axis, cc))
            for conv in np.arange(
                config.min_convergence_deg, config.max_convergence_deg + 1e-9, convergence_step_deg
            )
            for cc in np.arange(
                craniocaudal_range_deg[0], craniocaudal_range_deg[1] + 1e-9, craniocaudal_step_deg
            )
        ]
    )
    if directions.size == 0:
        return empty

    offsets = np.atleast_1d(np.asarray(entry_grid_mm, dtype=np.float64))
    label = int(analysis.vertebra.label)
    radius = (
        float(corridor_radius_mm)
        if corridor_radius_mm is not None
        else _corridor_radius(width, config)
    )

    planner = planner if planner is not None else make_planner(grader, config)
    surface = planner._find_entry_point(center, posterior_axis, label)
    surface_reach = (
        float(np.dot(np.asarray(surface, dtype=np.float64) - center, posterior_axis))
        if surface is not None
        else _ANCHOR_MAX_MM
    )
    surface_reach = float(np.clip(surface_reach, 0.0, _ANCHOR_MAX_MM))

    # One ray per direction, seated from the isthmus centre where the corridor
    # is widest; the tangential grid then spreads entry points over that surface.
    anchors, travel = _seat_entries(
        grader,
        np.repeat(center[None, :], directions.shape[0], axis=0),
        directions,
        np.full(directions.shape[0], surface_reach),
        label,
        radius,
    )
    shortfall_per_direction = np.maximum(surface_reach - travel, 0.0)

    entries = []
    entry_directions = []
    shortfall = []
    for anchor, direction, short in zip(anchors, directions, shortfall_per_direction, strict=True):
        # ``lateral_axis`` is deliberately *not* rotated with the direction: it is
        # a fixed patient-frame axis, so the "a" offsets are a stable mediolateral
        # slide across the entry surface for every candidate angle.  Only the
        # second tangent follows the direction, keeping the pair perpendicular.
        tangent = _unit(np.cross(direction, lateral_axis))
        for a in offsets:
            for b in offsets:
                entries.append(anchor + a * lateral_axis + b * tangent)
                entry_directions.append(direction)
                shortfall.append(short)
    entry_count = len(entries)
    per_entry = lengths_catalogue.size
    entries = np.repeat(np.asarray(entries, dtype=np.float64), per_entry, axis=0)
    entry_directions = np.repeat(
        np.asarray(entry_directions, dtype=np.float64), per_entry, axis=0
    )
    shortfall = np.repeat(np.asarray(shortfall, dtype=np.float64), per_entry)
    lengths = np.tile(lengths_catalogue, entry_count)
    targets = entries + entry_directions * lengths[:, None]

    # Cheap prune: neither end outside the vertebra can ever be feasible.
    entry_out, _ = grader.distances_at_points(entries, label)
    tip_out, _ = grader.distances_at_points(targets, label)
    keep = (entry_out <= 0.0) & (tip_out <= 0.0)
    if diagnostics is not None:
        diagnostics["surface_shortfall_mm"] = shortfall[keep]
    return entries[keep], targets[keep], lengths[keep]


# --------------------------------------------------------------------- scoring
def score_candidates(
    batch: BatchResult,
    entries: np.ndarray,
    targets: np.ndarray,
    lengths: np.ndarray,
    diameter: float,
    analysis: PedicleAnalysisResult,
    side: str,
    config: PlannerConfig,
    weights: OptimizerWeights,
    tip_batch: Optional[BatchResult] = None,
    surface_shortfall_mm: Optional[np.ndarray] = None,
) -> List[Candidate]:
    """Filter graded candidates to the feasible ones and rank them.

    ``tip_batch`` is the grading of each candidate's distal
    :data:`TIP_SEGMENT_MM`; when given it enforces the anterior safety margin.
    ``surface_shortfall_mm`` (from ``generate_candidates(..., diagnostics=...)``)
    rejects entries buried more than :data:`MAX_ENTRY_SHORTFALL_MM` inside the
    posterior cortex.

    A grader with no CT reports ``NaN`` HU for *every* candidate; that is a
    missing measurement, not an unrankable trajectory, so the density objective
    drops to 0 instead of failing the whole batch.  Individual ``NaN`` values in
    an otherwise-sampled batch still mean the candidate could not be measured
    and stay infeasible.
    """
    entries = np.asarray(entries, dtype=np.float64).reshape(-1, 3)
    targets = np.asarray(targets, dtype=np.float64).reshape(-1, 3)
    lengths = np.asarray(lengths, dtype=np.float64).reshape(-1)
    count = entries.shape[0]
    if count == 0:
        return []

    center, _axis, width = _side_data(analysis, side)
    max_length = max(config.implant_lengths_mm)
    normal = (
        _unit(analysis.upper_endplate_normal)
        if analysis.upper_endplate_normal is not None
        else None
    )
    half_width = max(float(width) / 2.0, 1.0)

    deltas = targets - entries
    norms = np.linalg.norm(deltas, axis=1)
    directions = np.zeros_like(deltas)
    movable = norms > 1e-12
    directions[movable] = deltas[movable] / norms[movable, None]

    convergence, craniocaudal = _trajectory_angles(deltas, side)

    sampled_hu = np.isfinite(batch.mean_hu)
    without_ct = not sampled_hu.any()
    if without_ct:
        _warn_missing_ct()

    feasible = (
        (batch.breach_mm <= 0.0)
        & (batch.min_wall_mm >= config.wall_clearance_mm)
        & (sampled_hu | without_ct)
        & movable
        & (convergence >= config.min_convergence_deg - 1e-6)
        & (convergence <= config.max_convergence_deg + 1e-6)
    )
    if tip_batch is not None:
        tip_margin = max(config.anterior_margin_mm - config.wall_clearance_mm, 0.0)
        feasible &= (tip_batch.breach_mm <= 0.0) & (tip_batch.min_wall_mm >= tip_margin)
    if surface_shortfall_mm is not None:
        shortfall = np.asarray(surface_shortfall_mm, dtype=np.float64).reshape(-1)
        feasible &= shortfall <= MAX_ENTRY_SHORTFALL_MM + 1e-9
    else:
        shortfall = np.zeros(count)
    if not feasible.any():
        return []

    safety = np.clip(np.minimum(batch.min_wall_mm, SAFETY_CAP_MM) / SAFETY_CAP_MM, 0.0, 1.0)
    density = np.clip(
        (np.where(sampled_hu, batch.mean_hu, DENSITY_LOW_HU) - DENSITY_LOW_HU)
        / (DENSITY_HIGH_HU - DENSITY_LOW_HU),
        0.0,
        1.0,
    )
    length_score = np.clip(lengths / max_length, 0.0, 1.0)
    if normal is None:
        endplate = np.ones(count)
    else:
        tilt = np.degrees(np.arcsin(np.clip(directions @ normal, -1.0, 1.0)))
        endplate = np.clip(1.0 - np.abs(tilt) / ENDPLATE_TOLERANCE_DEG, 0.0, 1.0)
    if center is None:
        centering = np.ones(count)
    else:
        offset = np.asarray(center, dtype=np.float64)[None, :] - entries
        perpendicular = offset - directions * np.sum(offset * directions, axis=1)[:, None]
        centering = np.clip(
            1.0 - np.linalg.norm(perpendicular, axis=1) / half_width, 0.0, 1.0
        )

    total = (
        weights.safety * safety
        + weights.density * density
        + weights.length * length_score
        + weights.endplate * endplate
        + weights.centering * centering
    )

    candidates = [
        Candidate(
            entry=entries[i].copy(),
            target=targets[i].copy(),
            length=float(lengths[i]),
            diameter=float(diameter),
            breach_mm=float(batch.breach_mm[i]),
            min_wall_mm=float(batch.min_wall_mm[i]),
            mean_hu=float(batch.mean_hu[i]),
            convergence_deg=float(convergence[i]),
            craniocaudal_deg=float(craniocaudal[i]),
            score=float(total[i]),
            surface_shortfall_mm=float(shortfall[i]),
            components={
                "safety": float(safety[i]),
                "density": float(density[i]),
                "length": float(length_score[i]),
                "endplate": float(endplate[i]),
                "centering": float(centering[i]),
            },
        )
        for i in np.flatnonzero(feasible)
    ]
    candidates.sort(key=lambda c: c.score, reverse=True)
    return candidates


# ---------------------------------------------------------------------- driver
def optimize_screw(
    grader: ScrewGrader,
    analysis: PedicleAnalysisResult,
    side: str,
    config: PlannerConfig,
    weights: OptimizerWeights = DEFAULT_WEIGHTS,
    top_k: int = 10,
) -> List[Candidate]:
    """Best ``top_k`` trajectories for one pedicle, highest score first.

    The diameter starts at the level/pedicle recommendation and steps down the
    catalogue until some candidate is feasible, mirroring
    :meth:`AutoScrewPlanner.plan_screw`, but at most
    :data:`MAX_DIAMETER_STEPS` steps: past that the screw is too small to be a
    sensible answer and the extra passes only cost runtime.  Returns ``[]`` when
    no diameter in that window admits a reachable, contained screw.
    """
    center, _axis, width = _side_data(analysis, side)
    if center is None or not analysis.success:
        return []

    planner = make_planner(grader, config)
    recommended = planner._compute_diameter(width, analysis.vertebra.name)
    if recommended is None:
        return []
    label = int(analysis.vertebra.label)

    catalogue = sorted(
        (d for d in config.implant_diameters_mm if d <= recommended + 1e-9), reverse=True
    )[: MAX_DIAMETER_STEPS + 1]
    for diameter in catalogue:
        diagnostics: Dict[str, np.ndarray] = {}
        entries, targets, lengths = generate_candidates(
            grader,
            analysis,
            side,
            config,
            corridor_radius_mm=diameter / 2.0,
            planner=planner,
            diagnostics=diagnostics,
        )
        if entries.shape[0] == 0:
            continue
        batch = grader.evaluate_batch(entries, targets, diameter, label)
        directions = targets - entries
        norms = np.linalg.norm(directions, axis=1)
        norms[norms <= 1e-12] = 1.0
        directions /= norms[:, None]
        tip_batch = grader.evaluate_batch(
            targets - directions * TIP_SEGMENT_MM, targets, diameter, label
        )
        ranked = score_candidates(
            batch, entries, targets, lengths, diameter, analysis, side, config, weights,
            tip_batch=tip_batch,
            surface_shortfall_mm=diagnostics.get("surface_shortfall_mm"),
        )
        if ranked:
            if diameter < recommended:
                warning = (
                    f"Diameter reduced from {recommended:.1f} to {diameter:.1f} mm "
                    "for cortical containment"
                )
                for candidate in ranked:
                    candidate.warnings.append(warning)
            return ranked[:top_k]
    return []


# ------------------------------------------------------------ construct level
def rod_misalignment_mm(head_points: np.ndarray) -> float:
    """RMS distance of the screw heads on one side from their best-fit 3-D line.

    The rod is a smooth curve threaded through the heads of one side, so how far
    the heads sit off a common line is a first-order proxy for how much the rod
    has to be bent (and how much the surgeon has to reduce).  Fewer than three
    heads always lie on a line, so they score a perfect 0.
    """
    heads = np.asarray(head_points, dtype=np.float64).reshape(-1, 3)
    if heads.shape[0] < 3:
        return 0.0
    centred = heads - heads.mean(axis=0)
    # First right-singular vector = principal axis = direction of the best-fit line.
    direction = np.linalg.svd(centred, full_matrices=False)[2][0]
    residuals = centred - np.outer(centred @ direction, direction)
    return float(np.sqrt(np.mean(np.sum(residuals**2, axis=1))))


def optimize_construct(
    per_screw_candidates: Dict[Tuple[str, str], List[Candidate]],
    weights: OptimizerWeights = DEFAULT_WEIGHTS,
) -> Dict[Tuple[str, str], Candidate]:
    """Re-rank per-screw candidates so the heads of each side line up.

    Keys are ``(level_name, side)``.  Starting from every screw's own best
    trajectory, greedy coordinate descent sweeps the screws (at most
    :data:`_CONSTRUCT_MAX_PASSES` times, stopping early once a pass changes
    nothing) and swaps in the candidate minimising

    ``-score + weights.rod * rod_misalignment_mm(side heads) / ROD_TOLERANCE_MM``

    Only candidates whose own score stays within
    :data:`CONSTRUCT_SCORE_TOLERANCE` of that screw's best are eligible, so rod
    alignment can never buy a materially worse screw.  Levels are only ever
    compared against the same side's heads; the other side's term is constant
    for that screw and cannot change the choice.
    """
    chosen: Dict[Tuple[str, str], Candidate] = {}
    eligible: Dict[Tuple[str, str], List[Candidate]] = {}
    for key, candidates in per_screw_candidates.items():
        if not candidates:
            continue
        best = max(candidates, key=lambda c: c.score)
        floor = best.score - CONSTRUCT_SCORE_TOLERANCE * abs(best.score)
        chosen[key] = best
        eligible[key] = [c for c in candidates if c.score >= floor - 1e-12]

    def side_misalignment(side: str, key: Tuple[str, str], candidate: Candidate) -> float:
        heads = [
            (candidate if k == key else c).entry
            for k, c in chosen.items()
            if k[1] == side
        ]
        return rod_misalignment_mm(np.asarray(heads, dtype=np.float64)) if heads else 0.0

    for _ in range(_CONSTRUCT_MAX_PASSES):
        changed = False
        for key in list(chosen):
            side = key[1]
            best_candidate = chosen[key]
            best_cost = -best_candidate.score + weights.rod * side_misalignment(
                side, key, best_candidate
            ) / ROD_TOLERANCE_MM
            for candidate in eligible[key]:
                cost = -candidate.score + weights.rod * side_misalignment(
                    side, key, candidate
                ) / ROD_TOLERANCE_MM
                if cost < best_cost - 1e-12:
                    best_cost, best_candidate = cost, candidate
            if best_candidate is not chosen[key]:
                chosen[key] = best_candidate
                changed = True
        if not changed:
            break
    return chosen
