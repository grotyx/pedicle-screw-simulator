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
import weakref
from dataclasses import dataclass, field, replace
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import SimpleITK as sitk

from ..utils.constants import VERTEBRAL_HU_OSTEOPOROSIS_THRESHOLD
from .bone_quality import vertebral_body_hu
from .planner_config import PlannerConfig
from .screw_geometry import endplate_slope_deg
from .screw_grading import ENTRY_ZONE_MM, BatchResult, ScrewGrader
from .vertebra import PedicleAnalysisResult, aiming_endplate_normal

logger = logging.getLogger(__name__)

#: Wall distance at which the safety objective saturates (mm).
SAFETY_CAP_MM = 3.0

#: HU window the density objective is normalised over.
DENSITY_LOW_HU = 100.0
DENSITY_HIGH_HU = 600.0

#: Deviation from the endplate plane that drives the endplate objective to 0.
ENDPLATE_TOLERANCE_DEG = 15.0

#: How far the craniocaudal sweep may be shifted off the pedicle axis to follow
#: the endplate (degrees).  A thoracolumbar endplate is within about 20 degrees
#: of the pedicle axis; a larger offset means the plane fit is wrong, and
#: chasing it would sweep the whole grid out of the vertebra.
MAX_ENDPLATE_RECENTRE_DEG = 20.0

#: How far the convergence sweep may be shifted off the pedicle axis so that
#: its rotation *offsets* land on the configured *absolute* window (degrees).
#: :func:`generate_candidates` sweeps ``conv`` as a rotation of the pedicle
#: axis, while :func:`score_candidates` filters the *measured* convergence
#: against ``[min_convergence_deg, max_convergence_deg]``; on an axis that
#: converges strongly on its own the two disagree by exactly that angle and the
#: entire grid can land outside the window.  A pedicle axis more than 60
#: degrees off the sagittal plane is a mis-fit rather than an anatomy, and
#: recentring further would aim the grid across the vertebral body instead of
#: down the corridor, so the shift is clamped here the way
#: :data:`MAX_ENDPLATE_RECENTRE_DEG` clamps the endplate one.
MAX_CONVERGENCE_RECENTRE_DEG = 60.0

#: Prefix of the note left on every candidate when the endplate band had to be
#: dropped.  Callers match on the prefix; the text carries the tolerance.
ENDPLATE_BAND_RELAXED_PREFIX = "Endplate band relaxed"


def endplate_band_relaxed_warning(tolerance_deg: float) -> str:
    """The single wording for a dropped endplate band."""
    return (
        f"{ENDPLATE_BAND_RELAXED_PREFIX}: no trajectory within "
        f"±{tolerance_deg:g}° of the upper endplate"
    )


#: Length of the distal segment the anterior-margin check is measured over (mm).
TIP_SEGMENT_MM = 4.0

#: Relief subtracted from ``anterior_margin_mm`` before the distal
#: :data:`TIP_SEGMENT_MM` is tested (mm).  The tip test was calibrated while the
#: wall clearance defaulted to 1.0 mm and the margin was written as
#: ``anterior_margin_mm - wall_clearance_mm``, i.e. an effective 3.0 mm.  W7
#: dropped the clearance default to 0.0, which silently tightened the tip test
#: to the full 4.0 mm for every side, narrow or not, and pushed sides into the
#: legacy fallback.  Naming the relief restores the calibrated 3.0 mm and keeps
#: a future clearance change from moving the anterior margin with it.
TIP_MARGIN_RELIEF_MM = 1.0

#: How far the seated entry may sit anterior of the posterior cortex before the
#: trajectory counts as unreachable.  A pedicle is continuous with the lamina, so
#: a few millimetres of countersinking is normal; more than this means a drill
#: would have to cross air (or another structure) to reach the corridor.
#:
#: The shortfall measures *burial*, not exposure: the entry sits inside bone,
#: anterior of the cortex the posterior ray-cast reached, so a larger value can
#: never admit a screw hanging in air.  Containment is a separate, unrelaxed
#: test -- the grader still requires ``breach_mm <= 0`` and the configured wall
#: clearance over the whole shaft -- so the only thing this bound buys is a
#: shallower countersink.  Part of what it measures is an artefact besides:
#: ``surface_reach`` is cast along the *pedicle axis* while ``travel`` runs
#: along each candidate's own direction, so an obliquely angled candidate on a
#: tilted axis reports a shortfall from the geometry of the two rays alone.  At
#: 3 mm that made this the sole binder on tilted levels whose trajectories were
#: otherwise contained; 6 mm still rejects an entry buried in the lamina with
#: no drillable bone behind it (the arch phantom reports 12 mm and up).
MAX_ENTRY_SHORTFALL_MM = 6.0

#: How many catalogue steps below the recommended diameter the search may go
#: before giving up.  Bounds the worst-case runtime of :func:`optimize_screw`.
MAX_DIAMETER_STEPS = 2

#: Tangential entry offsets (mm) the candidate sweep spreads over the entry
#: surface.  A narrow side needs a wider one: the whole point of the policy is
#: that the corridor can be slid laterally until the medial wall survives, and
#: +/- 2 mm is not always far enough to get there.
DEFAULT_ENTRY_GRID_MM: Tuple[float, ...] = (-2.0, -1.0, 0.0, 1.0, 2.0)
NARROW_ENTRY_GRID_MM: Tuple[float, ...] = (-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0)

#: Head-misalignment (mm) at which the rod objective costs a full unit of score.
ROD_TOLERANCE_MM = 3.0

#: Same-side convergence disagreement (deg RMS) at which the construct's
#: convergence objective costs a full unit of score, mirroring
#: :data:`ROD_TOLERANCE_MM` for the rod line.
CONVERGENCE_TOLERANCE_DEG = 5.0

#: Width of the convergence bins the candidate pool is required to cover, so
#: the construct descent always has a trajectory at a *different* angle to move
#: to and is not trapped in the 10 near-identical bests of one bin.
CONVERGENCE_BIN_DEG = 2.5

#: Labels whose screws are left out of the convergence agreement: a sacral
#: screw's angle is anatomically different from the lumbar levels above it, so
#: including it would drag every other level's angle toward it.  Values are the
#: TotalSegmentator labels in
#: :data:`src.core.pedicle_analyzer.VERTEBRA_LABELS` (25 sacrum, 26 S1).
_SACRAL_LABELS = frozenset({25, 26})

#: Fraction of a screw's own best score the construct re-ranking may trade away.
CONSTRUCT_SCORE_TOLERANCE = 0.10

#: Cap on the greedy coordinate-descent sweeps over the construct.
_CONSTRUCT_MAX_PASSES = 5

#: Step and reach of the 1-D scan that seats an entry point on the posterior
#: surface of the screw corridor (mm).
_ANCHOR_STEP_MM = 0.5
_ANCHOR_MAX_MM = 40.0

#: How far behind a seated head the drill's approach must be free of the same
#: vertebra's bone (mm).  A head is seated on the first surface its trajectory
#: meets on the way out, so a head under a lamina with an air pocket between
#: them lands on the pedicle's own surface, facing the pocket: a real drill
#: would have to go through the lamina first and then cross the pocket.  Such a
#: head is unreachable and is discarded.  Fifteen millimetres reaches past any
#: lamina covering an entry, and stops well short of the distant processes a
#: lateral-dorsal approach never meets.
DORSAL_APPROACH_CLEAR_MM = 15.0

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

#: Body-mean HU below which the optimiser treats the level as osteoporotic and
#: re-weights density (see :func:`adaptive_weights_for_bone`).  The threshold
#: is the literature screen in :mod:`src.utils.constants`, re-exported here so
#: :func:`optimize_screw` can consume it without importing the whole constants
#: module at every call site.
OSTEOPOROSIS_BODY_HU = VERTEBRAL_HU_OSTEOPOROSIS_THRESHOLD

#: Density-weight multiplier applied in osteoporotic bone.
OSTEOPOROSIS_DENSITY_BOOST = 1.5

#: Cap on the boosted density weight: the density objective may grow past
#: safety in soft bone, but it must stay on the same clipping scale so a safe
#: corridor still outranks a dense breach.
OSTEOPOROSIS_DENSITY_CAP = 1.0

#: Extra clearance two screws in one construct must keep between each other
#: (mm), on top of the sum of their radii.  A construct whose shafts touch can
#: still read as two contained screws, so the pairing is rejected even though
#: each screw alone is feasible.
INTER_SCREW_CLEARANCE_MM = 1.0


def adaptive_weights_for_bone(
    body_mean_hu: Optional[float],
    weights: OptimizerWeights = DEFAULT_WEIGHTS,
) -> OptimizerWeights:
    """Boost the density objective when the vertebral body is osteoporotic.

    ``body_mean_hu`` is the trabecular ROI mean at the level's body centre
    (:func:`~src.core.bone_quality.vertebral_body_hu`); a value below
    :data:`OSTEOPOROSIS_BODY_HU` scales ``weights.density`` by
    :data:`OSTEOPOROSIS_DENSITY_BOOST`, capped at
    :data:`OSTEOPOROSIS_DENSITY_CAP`.  Missing (``None``/``NaN``) or
    at/above-threshold values return ``weights`` unchanged, so normodense bone
    keeps today's ranking exactly.
    """
    if body_mean_hu is None:
        return weights
    body = float(body_mean_hu)
    if not np.isfinite(body) or body >= OSTEOPOROSIS_BODY_HU:
        return weights
    return replace(
        weights, density=min(weights.density * OSTEOPOROSIS_DENSITY_BOOST,
                             OSTEOPOROSIS_DENSITY_CAP)
    )


def body_mean_hu_for_analysis(
    grader: ScrewGrader,
    analysis: PedicleAnalysisResult,
) -> Optional[float]:
    """Trabecular body-centre mean HU for one analysed vertebra, or ``None``.

    Reuses the grader's cached arrays (no per-call volume copies) and the
    level's own volume for ROI sizing; returns ``None`` exactly when there is
    no CT, no body centre, or too few ROI voxels to report.
    """
    body_center = getattr(analysis, "vertebral_body_center", None)
    if body_center is None or grader.ct_image() is None:
        return None
    volume = getattr(getattr(analysis, "vertebra", None), "volume_mm3", None)
    return vertebral_body_hu(
        None,
        None,
        int(analysis.vertebra.label),
        body_center,
        volume_mm3=float(volume) if volume else None,
        ct_array=grader._ct_array,
        mask_array=grader._mask_array,
        origin_lps=grader._origin,
        spacing_mm=grader._spacing,
    )


def _segment_segment_distance(
    p1: np.ndarray, q1: np.ndarray, p2: np.ndarray, q2: np.ndarray
) -> float:
    """Closest distance (mm) between segments ``p1q1`` and ``p2q2`` (LPS).

    Standard parameterisation over the two unit-interval parameters; parallel
    and degenerate (zero-length) segments fall back to point-to-segment form
    rather than dividing by a vanishing denominator.
    """
    p1 = np.asarray(p1, dtype=np.float64).reshape(3)
    q1 = np.asarray(q1, dtype=np.float64).reshape(3)
    p2 = np.asarray(p2, dtype=np.float64).reshape(3)
    q2 = np.asarray(q2, dtype=np.float64).reshape(3)
    d1 = q1 - p1
    d2 = q2 - p2
    r = p1 - p2
    a = float(d1 @ d1)
    e = float(d2 @ d2)

    def point_segment(p: np.ndarray, a0: np.ndarray, b0: np.ndarray) -> float:
        ab = b0 - a0
        denom = float(ab @ ab)
        t = float((p - a0) @ ab / denom) if denom > 1e-12 else 0.0
        return float(np.linalg.norm(p - (a0 + np.clip(t, 0.0, 1.0) * ab)))

    if a <= 1e-12:
        return point_segment(p1, p2, q2)
    if e <= 1e-12:
        return point_segment(p2, p1, q1)
    f = float(d2 @ r)
    c = float(d1 @ r)
    b = float(d1 @ d2)
    denom = a * e - b * b
    s = 0.0 if denom <= 1e-12 else min(max((b * f - c * e) / denom, 0.0), 1.0)
    t = (b * s + f) / e if e > 1e-12 else 0.0
    if t < 0.0:
        t, s = 0.0, min(max(-c / a, 0.0), 1.0) if a > 1e-12 else 0.0
    elif t > 1.0:
        t, s = 1.0, min(max((b - c) / a, 0.0), 1.0) if a > 1e-12 else 0.0
    return float(np.linalg.norm(d1 * s - d2 * t + r))


def screws_collide(a: Candidate, b: Candidate,
                   clearance_mm: float = INTER_SCREW_CLEARANCE_MM) -> bool:
    """Whether two construct screws violate each other's corridor.

    True when tip-to-tip *or* shaft-to-shaft (segment-segment over the
    ``(entry, target)`` pairs) distance falls below the sum of the screws'
    radii plus ``clearance_mm``.
    """
    entry_a = np.asarray(a.entry, dtype=np.float64)
    target_a = np.asarray(a.target, dtype=np.float64)
    entry_b = np.asarray(b.entry, dtype=np.float64)
    target_b = np.asarray(b.target, dtype=np.float64)
    limit = float(a.diameter + b.diameter) / 2.0 + float(clearance_mm)
    if float(np.linalg.norm(target_a - target_b)) < limit:
        return True
    return _segment_segment_distance(entry_a, target_a, entry_b, target_b) < limit


def _adjacent_levels(level_a: Optional[int], level_b: Optional[int]) -> bool:
    """Whether two TotalSegmentator labels are the same level or neighbours."""
    if level_a is None or level_b is None:
        return True
    try:
        return abs(int(level_a) - int(level_b)) <= 1
    except (TypeError, ValueError):
        return True


@dataclass
class RunProgress:
    """Per-``(level, side)`` narration and cancellation for one planning run.

    Shared by every planning back-end -- the optimiser, the legacy planner and
    the CBT planner -- so a run reads the same wherever it is planned and stops
    at the same granularity: between screws, never part-way through grading one.
    It lives here rather than in ``auto_screw_planner`` because
    :mod:`~src.core.cbt_planner` cannot import that module at module scope
    (``auto_screw_planner`` imports *it*), and this module is the one both
    already depend on.
    """

    #: How many ``(level, side)`` pairs the run will visit, for the "i/n" count.
    total: int
    progress: Optional[Callable[[str], None]] = None
    cancel: Optional[Callable[[], bool]] = None
    done: int = 0
    #: Set once :meth:`start` has refused an iteration.
    cancelled: bool = False

    def start(self, level: str, side: str) -> bool:
        """Announce one ``(level, side)``; ``False`` means stop the run here."""
        if self.cancel is not None and self.cancel():
            self.cancelled = True
            return False
        self.done += 1
        if self.progress is not None:
            self.progress(f"Planning {level} {side} ({self.done}/{self.total})…")
        return True


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
    #: Directional split of ``breach_mm`` from the grader, and the wall left on
    #: the medial side.  All four equal the undirected numbers for a candidate
    #: graded without a side.
    medial_breach_mm: float = 0.0
    lateral_breach_mm: float = 0.0
    craniocaudal_breach_mm: float = 0.0
    medial_wall_mm: float = 0.0


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


#: Graders already warned about a missing CT.  The warning is a property of the
#: loaded study, not of any one candidate, so it is deduplicated -- but per
#: grader rather than per process, so a second study loaded into the same
#: session is still warned about.  Weak references keep a closed study's grader
#: collectable.
_missing_ct_warned: "weakref.WeakSet[ScrewGrader]" = weakref.WeakSet()


def _warn_missing_ct(grader: Optional[ScrewGrader] = None) -> None:
    """Warn once per ``grader`` that trajectory HU is unavailable.

    A caller with no grader to key on (a direct ``score_candidates`` call) is
    warned every time rather than silenced by another study's run.
    """
    if grader is not None:
        if grader in _missing_ct_warned:
            return
        _missing_ct_warned.add(grader)
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


def seat_on_cortex(
    grader: ScrewGrader,
    seeds: np.ndarray,
    directions: np.ndarray,
    label: int,
    reach_mm: float = _ANCHOR_MAX_MM,
) -> Tuple[np.ndarray, np.ndarray]:
    """Walk each seed back along its own trajectory to the dorsal cortex.

    The head of a pedicle screw belongs on the bone surface the drill enters,
    measured along the screw's *own* axis.  :func:`_seat_entries` stops where
    the whole screw cross-section still fits, which near the thin posterior
    elements is well short of that surface -- on the sample study 5 to 21 mm
    short, so the screws came out at roughly half the length the vertebra
    held and their heads were hidden inside the lamina instead of sitting at
    the facet.

    Marches posteriorly from each seed while the *centreline* stays inside
    ``label`` and returns the last such point, together with how far each seed
    moved.  A seed that is not itself inside the label does not move; the
    caller's containment prune then discards it.  The cross-section is not
    tested here: the cortex the head sits on straddles the surface by
    construction, and the grader's :data:`~src.core.screw_grading.ENTRY_ZONE_MM`
    is what keeps that from reading as a breach while still grading every
    millimetre past it.
    """
    seeds = np.asarray(seeds, dtype=np.float64).reshape(-1, 3)
    directions = np.asarray(directions, dtype=np.float64).reshape(-1, 3)
    if seeds.shape[0] == 0:
        return seeds.copy(), np.zeros(0)
    steps = np.arange(0.0, float(reach_mm) + 1e-9, _ANCHOR_STEP_MM)
    points = seeds[:, None, :] - directions[:, None, :] * steps[None, :, None]
    d_out, _ = grader.distances_at_points(points.reshape(-1, 3), label)
    inside = (d_out <= 0.0).reshape(seeds.shape[0], steps.size)
    run = np.logical_and.accumulate(inside, axis=1).sum(axis=1)
    travel = np.where(run > 0, steps[np.maximum(run - 1, 0)], 0.0)
    return seeds - directions * travel[:, None], travel


def anterior_margin_clear(
    grader: ScrewGrader,
    tips: np.ndarray,
    directions: np.ndarray,
    label: int,
    margin_mm: float,
) -> np.ndarray:
    """Whether each tip keeps ``margin_mm`` of bone ahead of it along its axis.

    This is the anterior margin as a surgeon states it -- the tip stops that far
    short of the anterior cortex *along the screw's path* -- and it is what the
    length is chosen against.  The old test demanded the same clearance of the
    distal cylinder's whole surface in every direction, which in the rounded
    front of a vertebral body is set by the side walls long before the anterior
    cortex: on the sample study it capped screws 5 to 17 mm short of a tip that
    still had 5 to 9 mm of bone in front of it.  The distal cylinder is still
    required to be contained, and to keep the configured wall clearance, like
    every other millimetre of the shaft (see :func:`score_candidates`).
    """
    tips = np.asarray(tips, dtype=np.float64).reshape(-1, 3)
    directions = np.asarray(directions, dtype=np.float64).reshape(-1, 3)
    if tips.shape[0] == 0 or margin_mm <= 0.0:
        return np.ones(tips.shape[0], dtype=bool)
    steps = np.arange(_ANCHOR_STEP_MM, float(margin_mm) + 1e-9, _ANCHOR_STEP_MM)
    points = tips[:, None, :] + directions[:, None, :] * steps[None, :, None]
    d_out, _ = grader.distances_at_points(points.reshape(-1, 3), label)
    return (d_out.reshape(tips.shape[0], steps.size) <= 0.0).all(axis=1)


def dorsal_approach_clear(
    grader: ScrewGrader,
    heads: np.ndarray,
    directions: np.ndarray,
    label: int,
    reach_mm: float = DORSAL_APPROACH_CLEAR_MM,
) -> np.ndarray:
    """Whether the approach behind each head is clear of the same vertebra.

    Samples the ray from just behind each head back along ``-direction`` for
    ``reach_mm`` and reports ``False`` where it meets ``label`` again -- bone
    the drill would have to pass through before reaching this head, with the
    gap between them left as a breach the shaft never gets graded for.  See
    :data:`DORSAL_APPROACH_CLEAR_MM`.
    """
    heads = np.asarray(heads, dtype=np.float64).reshape(-1, 3)
    directions = np.asarray(directions, dtype=np.float64).reshape(-1, 3)
    if heads.shape[0] == 0:
        return np.zeros(0, dtype=bool)
    steps = np.arange(_ANCHOR_STEP_MM, float(reach_mm) + 1e-9, _ANCHOR_STEP_MM)
    points = heads[:, None, :] - directions[:, None, :] * steps[None, :, None]
    d_out, _ = grader.distances_at_points(points.reshape(-1, 3), label)
    inside = (d_out <= 0.0).reshape(heads.shape[0], steps.size)
    # The first step behind the head is outside by construction (the head is the
    # last inside point); bone *after* that first gap is what blocks the drill.
    left_bone = np.logical_or.accumulate(~inside, axis=1)
    return ~(left_bone & inside).any(axis=1)


def keep_longest_per_trajectory(candidates: List[Candidate]) -> List[Candidate]:
    """The longest feasible screw for each entry and direction, best first.

    The catalogue is swept for every trajectory, and the score's density term
    rewards staying in dense pedicle bone, which a longer screw leaves for the
    softer body -- so, weighed freely, a short screw could outscore a long one
    on the same trajectory.  The surgeon's rule is simpler: take the longest
    screw whose tip still keeps the anterior margin.  Every candidate here has
    already passed the hard feasibility tests (containment, walls, the tip
    margin), so keeping only the longest per trajectory applies that rule and
    leaves the score to choose *between* trajectories.
    """
    best: Dict[Tuple, Candidate] = {}
    for candidate in candidates:
        direction = _unit(np.asarray(candidate.target) - np.asarray(candidate.entry))
        key = (
            tuple(np.round(np.asarray(candidate.entry, dtype=np.float64), 3)),
            tuple(np.round(direction, 4)),
            float(candidate.diameter),
        )
        kept = best.get(key)
        if kept is None or candidate.length > kept.length:
            best[key] = candidate
    return sorted(best.values(), key=lambda c: c.score, reverse=True)


# ------------------------------------------------------------------ generation
def generate_candidates(
    grader: ScrewGrader,
    analysis: PedicleAnalysisResult,
    side: str,
    config: PlannerConfig,
    entry_grid_mm: Sequence[float] = DEFAULT_ENTRY_GRID_MM,
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

    When ``config.endplate_parallel`` is on and the analysis carries an
    upper-endplate normal, the craniocaudal window is centred on the
    endplate-parallel direction rather than on the pedicle axis, so
    ``craniocaudal_range_deg`` reads as "how far either side of the endplate".

    The convergence window is recentred the same way, but unconditionally: the
    swept ``conv`` is a rotation *offset* from the axis while
    ``score_candidates`` tests the *measured* angle, so the sweep is shifted by
    the axis's own convergence (clamped to
    :data:`MAX_CONVERGENCE_RECENTRE_DEG`) to make the two agree.  On an axis
    that is already anteroposterior the shift is zero and the grid is unchanged.
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
    # The (-10, 10) window is a window around the *endplate* direction, not
    # around the pedicle axis: an endplate-parallel trajectory that the hard
    # band in ``score_candidates`` demands has to be in the grid to be found.
    # ``cc`` rotates about ``craniocaudal_axis``, which is perpendicular to the
    # base direction's horizontal projection, so ``cc`` is an elevation offset
    # from the base direction (exactly at zero convergence, compressed by
    # cos(convergence) elsewhere -- close enough to populate the band, and the
    # band itself is applied to the measured angle, not to this offset).
    craniocaudal_center = 0.0
    slope_deg = (
        endplate_slope_deg(aiming_endplate_normal(analysis))
        if config.endplate_parallel
        else None
    )
    if slope_deg is not None:
        base_elevation = math.degrees(
            math.atan2(float(base[2]), float(math.hypot(base[0], base[1])))
        )
        craniocaudal_center = float(
            np.clip(
                slope_deg - base_elevation,
                -MAX_ENDPLATE_RECENTRE_DEG,
                MAX_ENDPLATE_RECENTRE_DEG,
            )
        )
    # ``conv`` below *rotates* ``base``, so it is an offset from the pedicle
    # axis's own convergence, while ``score_candidates`` filters the *measured*
    # absolute angle against the same window.  Shift the offsets back by the
    # axis's own convergence -- the recentring ``craniocaudal_center`` already
    # does for the endplate -- so the swept angles are the configured window as
    # measured, instead of that window plus whatever the axis brings with it.
    base_convergence = float(_trajectory_angles(base[None, :], side)[0][0])
    convergence_center = float(
        np.clip(
            -base_convergence,
            -MAX_CONVERGENCE_RECENTRE_DEG,
            MAX_CONVERGENCE_RECENTRE_DEG,
        )
    )
    medial_sign = -1.0 if side == "left" else 1.0

    directions = np.array(
        [
            _unit(_rotate(_rotate(base, z_axis, medial_sign * conv), craniocaudal_axis, cc))
            for conv in np.arange(
                convergence_center + config.min_convergence_deg,
                convergence_center + config.max_convergence_deg + 1e-9,
                convergence_step_deg,
            )
            for cc in np.arange(
                craniocaudal_center + craniocaudal_range_deg[0],
                craniocaudal_center + craniocaudal_range_deg[1] + 1e-9,
                craniocaudal_step_deg,
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
    # The anchors above sit where the whole cross-section first fits, which is
    # the right place to spread the tangential grid but the wrong place for a
    # head.  Each entry now climbs back along its own direction to the dorsal
    # cortex, which is where the drill starts: the screw gains the length the
    # buried head was hiding, and the head lands at the facet instead of inside
    # the lamina.  ``shortfall`` measured burial against a cortex found along
    # the *pedicle axis*; a head on its own trajectory's cortex has none left.
    entries, climb = seat_on_cortex(
        grader,
        np.asarray(entries, dtype=np.float64),
        np.asarray(entry_directions, dtype=np.float64),
        label,
    )
    # A head with the same vertebra's bone behind it sits in a pocket the drill
    # cannot reach without crossing that bone first -- discard it here, where
    # the old burial bound used to reject the same geometry by proxy.
    reachable = dorsal_approach_clear(
        grader, entries, np.asarray(entry_directions, dtype=np.float64), label
    )
    entries = entries[reachable]
    climb = climb[reachable]
    entry_directions = [d for d, ok in zip(entry_directions, reachable, strict=True) if ok]
    if entries.shape[0] == 0:
        if diagnostics is not None:
            diagnostics["surface_shortfall_mm"] = np.zeros(0)
            diagnostics["cortical_climb_mm"] = np.zeros(0)
            diagnostics["occluded_entries"] = int((~reachable).sum())
        return empty
    shortfall = np.zeros(len(entries), dtype=np.float64)

    entry_count = len(entries)
    lengths_catalogue = np.sort(np.asarray(lengths_catalogue, dtype=np.float64))
    shortfall_arr = np.asarray(shortfall, dtype=np.float64)
    climb_arr = np.asarray(climb, dtype=np.float64)
    entry_arr = np.asarray(entries, dtype=np.float64)
    dir_arr = np.asarray(entry_directions, dtype=np.float64)

    # Cheap prunes, cheapest first: ``distances_at_points`` is one map lookup
    # per point, so reject by reach (one centreline ray-cast per direction)
    # and by endpoint before any batch grading.  Reach is measured once per
    # direction along the screw's own centreline; a catalogue length fits
    # that direction only if its tip still lands inside the label.  (The
    # anterior margin is *not* part of this test: it is measured from the
    # tip along the axis by ``anterior_margin_clear`` and by the tip batch,
    # and folding it in here emptied corridors the scorer still solves --
    # including the narrow policy's only survivors.)  This expands fitting
    # lengths per direction instead of tiling all 7 over every trajectory,
    # so short corridors grade far fewer candidates while a corridor that
    # holds the longest screw keeps today's exact candidate set.
    reach_cap = float(lengths_catalogue.max())
    probe_steps = np.arange(_ANCHOR_STEP_MM, reach_cap + 1e-9, _ANCHOR_STEP_MM)
    probe = entry_arr[:, None, :] + dir_arr[:, None, :] * probe_steps[None, :, None]
    probe_out, _ = grader.distances_at_points(probe.reshape(-1, 3), label)
    inside = (probe_out.reshape(entry_arr.shape[0], probe_steps.size) <= 0.0)
    exit_idx = np.argmax(~inside, axis=1)
    all_inside = inside.all(axis=1)
    reach_mm = np.where(
        all_inside, reach_cap, np.maximum(probe_steps[exit_idx] - _ANCHOR_STEP_MM, 0.0)
    )
    fits = lengths_catalogue[None, :] <= reach_mm[:, None] + 1e-9
    # A corridor that holds the longest screw keeps every length, so the
    # default phantom's candidate set is unchanged.
    fits[reach_mm >= reach_cap - 1e-9] = True
    if not bool(fits.any()):
        if diagnostics is not None:
            diagnostics["surface_shortfall_mm"] = np.zeros(0)
            diagnostics["cortical_climb_mm"] = np.zeros(0)
            diagnostics["occluded_entries"] = int((~reachable).sum())
        return empty
    sel_entry = np.repeat(np.arange(entry_count), lengths_catalogue.size)[fits.ravel()]
    sel_len = np.tile(np.arange(lengths_catalogue.size), entry_count)[fits.ravel()]
    entries = entry_arr[sel_entry]
    entry_directions = dir_arr[sel_entry]
    shortfall = shortfall_arr[np.repeat(np.arange(entry_count), lengths_catalogue.size)][fits.ravel()]
    climb = climb_arr[np.repeat(np.arange(entry_count), lengths_catalogue.size)][fits.ravel()]
    lengths = lengths_catalogue[sel_len]
    targets = entries + entry_directions * lengths[:, None]

    # Cheap prune: neither end outside the vertebra can ever be feasible.
    entry_out, _ = grader.distances_at_points(entries, label)
    tip_out, _ = grader.distances_at_points(targets, label)
    keep = (entry_out <= 0.0) & (tip_out <= 0.0)
    if diagnostics is not None:
        diagnostics["surface_shortfall_mm"] = shortfall[keep]
        diagnostics["cortical_climb_mm"] = climb[keep]
        diagnostics["occluded_entries"] = int((~reachable).sum())
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
    grader: Optional[ScrewGrader] = None,
    narrow: bool = False,
    anterior_clear: Optional[np.ndarray] = None,
) -> List[Candidate]:
    """Filter graded candidates to the feasible ones and rank them.

    ``tip_batch`` is the grading of each candidate's distal
    :data:`TIP_SEGMENT_MM`.  Given ``anterior_clear`` as well (from
    :func:`anterior_margin_clear`), the anterior margin is that along-axis test
    and the distal cylinder only has to be contained with the configured wall
    clearance, like the rest of the shaft.  Given ``tip_batch`` alone, the
    older all-around test applies: the distal cylinder's surface must keep the
    anterior margin (less :data:`TIP_MARGIN_RELIEF_MM`) in every direction.
    ``surface_shortfall_mm`` (from ``generate_candidates(..., diagnostics=...)``)
    rejects entries buried more than :data:`MAX_ENTRY_SHORTFALL_MM` inside the
    posterior cortex.

    A grader with no CT reports ``NaN`` HU for *every* candidate; that is a
    missing measurement, not an unrankable trajectory, so the density objective
    drops to 0 instead of failing the whole batch.  Individual ``NaN`` values in
    an otherwise-sampled batch still mean the candidate could not be measured
    and stay infeasible.  ``grader``, when given, deduplicates that missing-CT
    warning to one per study instead of one per scoring pass.

    ``narrow`` swaps the feasibility rule and the objective for a pedicle the
    planner flagged: containment is required medially and craniocaudally only,
    a lateral breach up to ``config.narrow_lateral_breach_mm`` is accepted, the
    safety term measures the *medial* wall rather than the thinnest one, and a
    new ``lateral`` term (sharing the safety weight) pays for keeping that
    breach small.  Centering is dropped -- on a narrow side it would pull the
    corridor back toward the canal, which is the one direction it must not go.
    Narrow and normal scores are never compared with each other (every
    comparison is within one screw's own candidate list), so the two objectives
    do not have to be on the same scale.

    With ``config.endplate_parallel`` on and a fitted endplate plane, a
    candidate is feasible only while its endplate angle stays inside
    ``config.endplate_tolerance_deg``.  If that empties the set the band is
    dropped for this call and every returned candidate carries
    :func:`endplate_band_relaxed_warning`.  With the option off the soft
    ``endplate`` objective is pinned to 1.0, which is rank-neutral.
    """
    entries = np.asarray(entries, dtype=np.float64).reshape(-1, 3)
    targets = np.asarray(targets, dtype=np.float64).reshape(-1, 3)
    lengths = np.asarray(lengths, dtype=np.float64).reshape(-1)
    count = entries.shape[0]
    if count == 0:
        return []

    center, _axis, width = _side_data(analysis, side)
    max_length = max(config.implant_lengths_mm)
    aiming_normal = aiming_endplate_normal(analysis)
    normal = _unit(aiming_normal) if aiming_normal is not None else None
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
        _warn_missing_ct(grader)

    if narrow:
        feasible = (
            (batch.medial_breach_mm <= 0.0)
            & (batch.craniocaudal_breach_mm <= 0.0)
            & (batch.lateral_breach_mm <= config.narrow_lateral_breach_mm + 1e-9)
            & (sampled_hu | without_ct)
            & movable
            & (convergence >= config.min_convergence_deg - 1e-6)
            & (convergence <= config.max_convergence_deg + 1e-6)
        )
    else:
        feasible = (
            (batch.breach_mm <= 0.0)
            & (batch.min_wall_mm >= config.wall_clearance_mm)
            & (sampled_hu | without_ct)
            & movable
            & (convergence >= config.min_convergence_deg - 1e-6)
            & (convergence <= config.max_convergence_deg + 1e-6)
        )
    if tip_batch is not None:
        if anterior_clear is None:
            tip_margin = max(config.anterior_margin_mm - TIP_MARGIN_RELIEF_MM, 0.0)
        else:
            tip_margin = config.wall_clearance_mm
        feasible &= (tip_batch.breach_mm <= 0.0) & (tip_batch.min_wall_mm >= tip_margin)
    if anterior_clear is not None:
        feasible &= np.asarray(anterior_clear, dtype=bool).reshape(-1)
    if surface_shortfall_mm is not None:
        shortfall = np.asarray(surface_shortfall_mm, dtype=np.float64).reshape(-1)
        feasible &= shortfall <= MAX_ENTRY_SHORTFALL_MM + 1e-9
    else:
        shortfall = np.zeros(count)

    # Hard endplate band.  A band that admits nothing is dropped for this side
    # with a note rather than costing it a screw: an off-parallel screw the
    # surgeon can see is better than a missing one they have to explain.
    band_relaxed = False
    endplate_slope = (
        endplate_slope_deg(aiming_endplate_normal(analysis))
        if config.endplate_parallel
        else None
    )
    if endplate_slope is not None:
        within_band = feasible & (
            np.abs(craniocaudal - endplate_slope)
            <= config.endplate_tolerance_deg + 1e-9
        )
        if within_band.any():
            feasible = within_band
        else:
            band_relaxed = True
    if not feasible.any():
        return []

    safety_wall = batch.medial_wall_mm if narrow else batch.min_wall_mm
    safety = np.clip(np.minimum(safety_wall, SAFETY_CAP_MM) / SAFETY_CAP_MM, 0.0, 1.0)
    density = np.clip(
        (np.where(sampled_hu, batch.mean_hu, DENSITY_LOW_HU) - DENSITY_LOW_HU)
        / (DENSITY_HIGH_HU - DENSITY_LOW_HU),
        0.0,
        1.0,
    )
    length_score = np.clip(lengths / max_length, 0.0, 1.0)
    if normal is None or not config.endplate_parallel:
        # Neutral, not absent: a constant 1.0 adds ``weights.endplate`` to every
        # score, which leaves the ranking exactly as if the weight were zero,
        # while keeping the component present in every candidate's breakdown.
        endplate = np.ones(count)
    else:
        tilt = np.degrees(np.arcsin(np.clip(directions @ normal, -1.0, 1.0)))
        endplate = np.clip(1.0 - np.abs(tilt) / ENDPLATE_TOLERANCE_DEG, 0.0, 1.0)

    # Insertion order is the order the weighted sum is accumulated in, and it
    # matches the pre-split expression term for term, so a normal pedicle keeps
    # its exact score.
    component_arrays: Dict[str, np.ndarray] = {
        "safety": safety,
        "density": density,
        "length": length_score,
        "endplate": endplate,
    }
    if narrow:
        cap = max(float(config.narrow_lateral_breach_mm), 1e-9)
        component_arrays["lateral"] = np.clip(
            1.0 - batch.lateral_breach_mm / cap, 0.0, 1.0
        )
    elif center is None:
        component_arrays["centering"] = np.ones(count)
    else:
        offset = np.asarray(center, dtype=np.float64)[None, :] - entries
        perpendicular = offset - directions * np.sum(offset * directions, axis=1)[:, None]
        component_arrays["centering"] = np.clip(
            1.0 - np.linalg.norm(perpendicular, axis=1) / half_width, 0.0, 1.0
        )

    # The lateral cap and the medial wall are two halves of one trade-off, so
    # they share the safety weight: raising "safety" tightens both.
    component_weights = {
        "safety": weights.safety,
        "density": weights.density,
        "length": weights.length,
        "endplate": weights.endplate,
        "lateral": weights.safety,
        "centering": weights.centering,
    }
    total = sum(
        component_weights[name] * values for name, values in component_arrays.items()
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
            medial_breach_mm=float(batch.medial_breach_mm[i]),
            lateral_breach_mm=float(batch.lateral_breach_mm[i]),
            craniocaudal_breach_mm=float(batch.craniocaudal_breach_mm[i]),
            medial_wall_mm=float(batch.medial_wall_mm[i]),
            components={
                name: float(values[i]) for name, values in component_arrays.items()
            },
        )
        for i in np.flatnonzero(feasible)
    ]
    if band_relaxed:
        warning = endplate_band_relaxed_warning(config.endplate_tolerance_deg)
        for candidate in candidates:
            candidate.warnings.append(warning)
    candidates.sort(key=lambda c: c.score, reverse=True)
    return candidates


# ---------------------------------------------------------------------- driver
def _cover_convergence_bins(ranked: List[Candidate], top_k: int) -> List[Candidate]:
    """The best ``top_k`` candidates, but never all from one convergence bin.

    ``ranked`` is score-descending, so the first candidate seen in a bin is that
    bin's best.  Taking those representatives first guarantees the construct
    stage is handed a trajectory at a genuinely different angle for every angle
    that is feasible at all -- otherwise a pedicle whose 40 best candidates sit
    within half a degree of each other can never join a harmonised construct,
    however cheap the alternative would be.  The remaining slots go to the next
    best overall, and the result is re-sorted by score so callers keep reading
    ``[0]`` as "this screw's own best".
    """
    if top_k <= 0 or len(ranked) <= top_k:
        return list(ranked)
    representatives: Dict[int, Candidate] = {}
    for candidate in ranked:
        representatives.setdefault(
            int(math.floor(candidate.convergence_deg / CONVERGENCE_BIN_DEG)), candidate
        )
    # dict preserves insertion order, which here is score order.
    selection = list(representatives.values())[:top_k]
    picked = {id(c) for c in selection}
    for candidate in ranked:
        if len(selection) >= top_k:
            break
        if id(candidate) not in picked:
            selection.append(candidate)
            picked.add(id(candidate))
    selection.sort(key=lambda c: c.score, reverse=True)
    return selection


def optimize_screw(
    grader: ScrewGrader,
    analysis: PedicleAnalysisResult,
    side: str,
    config: PlannerConfig,
    weights: OptimizerWeights = DEFAULT_WEIGHTS,
    top_k: int = 10,
    planner=None,
    narrow: bool = False,
) -> List[Candidate]:
    """Best ``top_k`` trajectories for one pedicle, highest score first.

    The diameter starts at the level/pedicle recommendation and steps down the
    catalogue until some candidate is feasible, mirroring
    :meth:`AutoScrewPlanner.plan_screw`, but at most
    :data:`MAX_DIAMETER_STEPS` steps: past that the screw is too small to be a
    sensible answer and the extra passes only cost runtime.  Returns ``[]`` when
    no diameter in that window admits a reachable, contained screw.

    ``weights`` are adapted to the level's bone before scoring (see
    :func:`adaptive_weights_for_bone`): an osteoporotic body centre boosts the
    density objective so the ranking prefers denser purchase over extra
    length, while normodense levels keep today's ranking exactly.

    ``planner`` reuses a caller's :class:`AutoScrewPlanner` instead of building
    one per pedicle, which copies the whole CT and mask each time.  Only its
    entry-point search and diameter rules are used, so a substitute is
    equivalent exactly when it holds this ``grader``'s mask and this ``config``;
    :func:`make_planner` builds that planner when none is given.

    ``narrow`` plans the pedicle under the narrow policy: only
    ``planner.MIN_SCREW_DIAMETER``, the wider :data:`NARROW_ENTRY_GRID_MM` so a
    lateral shift is reachable, and the relaxed, medial-first feasibility of
    :func:`score_candidates`.  The diameter is the minimum rather than the
    level's recommendation because a side is also flagged narrow when its width
    could not be trusted at all -- including an implausibly *wide* measurement,
    whose recommendation would otherwise put a full-size screw through the
    relaxed feasibility rule.  This matches :meth:`AutoScrewPlanner._plan_screw`,
    which hard-codes the same minimum for a narrow side, so the two back-ends
    agree.  A narrow candidate therefore carries no "diameter reduced ... for
    cortical containment" warning: the minimum is the policy, not a step-down.
    """
    center, _axis, width = _side_data(analysis, side)
    if center is None or not analysis.success:
        return []

    planner = planner if planner is not None else make_planner(grader, config)
    recommended = planner._compute_diameter(
        planner._effective_width(analysis, side), analysis.vertebra.name
    )
    label = int(analysis.vertebra.label)
    weights = adaptive_weights_for_bone(body_mean_hu_for_analysis(grader, analysis),
                                        weights)

    if narrow:
        catalogue = [float(planner.MIN_SCREW_DIAMETER)]
        entry_grid = NARROW_ENTRY_GRID_MM
    else:
        catalogue = sorted(
            (d for d in config.implant_diameters_mm if d <= recommended + 1e-9),
            reverse=True,
        )[: MAX_DIAMETER_STEPS + 1]
        entry_grid = DEFAULT_ENTRY_GRID_MM
    for diameter in catalogue:
        diagnostics: Dict[str, np.ndarray] = {}
        entries, targets, lengths = generate_candidates(
            grader,
            analysis,
            side,
            config,
            entry_grid_mm=entry_grid,
            corridor_radius_mm=diameter / 2.0,
            planner=planner,
            diagnostics=diagnostics,
        )
        if entries.shape[0] == 0:
            continue
        # Graded from the cortex the head sits on, not from the head: see
        # ENTRY_ZONE_MM.  The tip test grades the survivors only: a candidate
        # the shaft already rejected cannot come back, so re-grading the whole
        # sweep only doubles the second batch's cost.  The tip segment starts
        # deep inside the body, so it takes no entry zone.
        batch = grader.evaluate_batch(
            entries, targets, diameter, label, side=side,
            entry_zone_mm=ENTRY_ZONE_MM,
        )
        directions = targets - entries
        norms = np.linalg.norm(directions, axis=1)
        norms[norms <= 1e-12] = 1.0
        directions /= norms[:, None]
        shaft_feasible = (
            (batch.medial_breach_mm <= 0.0)
            & (batch.craniocaudal_breach_mm <= 0.0)
            & (batch.lateral_breach_mm <= config.narrow_lateral_breach_mm + 1e-9)
            if narrow
            else (batch.breach_mm <= 0.0)
            & (batch.min_wall_mm >= config.wall_clearance_mm)
        )
        shaft_feasible &= (np.isfinite(batch.mean_hu)) | (grader.ct_image() is None)
        tip_rows = np.flatnonzero(shaft_feasible)
        if tip_rows.size == entries.shape[0]:
            # No cut: the old code graded every row, so keep one full call
            # and byte-identical numbers instead of scattering them.
            tip_batch = grader.evaluate_batch(
                targets - directions * TIP_SEGMENT_MM, targets, diameter, label
            )
        elif tip_rows.size:
            tip_full = np.full(entries.shape[0], np.inf, dtype=np.float64)
            tip_wall = np.full(entries.shape[0], -np.inf, dtype=np.float64)
            tip_batch = grader.evaluate_batch(
                targets[tip_rows] - directions[tip_rows] * TIP_SEGMENT_MM,
                targets[tip_rows], diameter, label,
            )
            tip_full[tip_rows] = tip_batch.breach_mm
            tip_wall[tip_rows] = tip_batch.min_wall_mm
            tip_batch = BatchResult(
                breach_mm=tip_full,
                min_wall_mm=tip_wall,
                mean_hu=np.full(entries.shape[0], np.nan),
                min_hu=np.full(entries.shape[0], np.nan),
            )
        else:
            tip_batch = BatchResult(
                breach_mm=np.full(entries.shape[0], np.inf),
                min_wall_mm=np.full(entries.shape[0], -np.inf),
                mean_hu=np.full(entries.shape[0], np.nan),
                min_hu=np.full(entries.shape[0], np.nan),
            )
        ranked = score_candidates(
            batch, entries, targets, lengths, diameter, analysis, side, config, weights,
            tip_batch=tip_batch,
            surface_shortfall_mm=diagnostics.get("surface_shortfall_mm"),
            grader=grader,
            narrow=narrow,
            anterior_clear=anterior_margin_clear(
                grader, targets, directions, label, config.anterior_margin_mm
            ),
        )
        ranked = keep_longest_per_trajectory(ranked)
        if ranked:
            if not narrow and diameter < recommended:
                warning = (
                    f"Diameter reduced from {recommended:.1f} to {diameter:.1f} mm "
                    "for cortical containment"
                )
                for candidate in ranked:
                    candidate.warnings.append(warning)
            return _cover_convergence_bins(ranked, top_k)
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


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    """Lower weighted median, averaging the two middles on an exact tie.

    With uniform weights this reproduces :func:`numpy.median` for both odd and
    even counts, so the neighbour weighting degrades cleanly to a plain median
    when no level ordering is known.
    """
    order = np.argsort(values, kind="stable")
    ordered = values[order]
    ordered_weights = weights[order]
    total = float(ordered_weights.sum())
    if total <= 0.0:
        return float(np.median(values))
    cumulative = np.cumsum(ordered_weights)
    half = total / 2.0
    index = min(int(np.searchsorted(cumulative, half - 1e-12, side="left")), ordered.size - 1)
    if index + 1 < ordered.size and abs(float(cumulative[index]) - half) <= 1e-9:
        return float((ordered[index] + ordered[index + 1]) / 2.0)
    return float(ordered[index])


def convergence_deviations_deg(
    angles: Sequence[float],
    levels: Optional[Sequence[Optional[int]]] = None,
    exclude_s1: bool = True,
) -> List[Optional[float]]:
    """How far each screw's convergence sits from its neighbours' agreement.

    ``levels`` are TotalSegmentator vertebra labels (see
    :data:`src.core.pedicle_analyzer.VERTEBRA_LABELS`), one per angle, and may
    be ``None`` where the level is unknown.  Adjacent-level agreement matters
    more to a surgeon than global agreement -- a lumbar construct legitimately
    converges more at L5 than at L1 -- so each screw is compared against a
    median in which the entries one level away (and itself) count double.  With
    no levels every weight is 1 and this is the plain median of the side.

    Positions excluded from the term (sacral screws when ``exclude_s1``) come
    back as ``None`` rather than 0.0, so a caller can tell "agrees perfectly"
    from "was not asked to agree".
    """
    values = [float(a) for a in angles]
    count = len(values)
    resolved: List[Optional[int]] = (
        list(levels) if levels is not None else [None] * count
    )
    if len(resolved) != count:
        raise ValueError(
            f"levels must have one entry per angle, got {len(resolved)} for {count}"
        )
    included = [
        i
        for i in range(count)
        if not (exclude_s1 and resolved[i] is not None and resolved[i] in _SACRAL_LABELS)
    ]
    deviations: List[Optional[float]] = [None] * count
    if len(included) < 2:
        for i in included:
            deviations[i] = 0.0
        return deviations

    included_values = np.array([values[i] for i in included], dtype=np.float64)
    for i in included:
        weights = np.array(
            [
                2.0
                if (
                    resolved[i] is not None
                    and resolved[j] is not None
                    and abs(int(resolved[i]) - int(resolved[j])) <= 1
                )
                else 1.0
                for j in included
            ],
            dtype=np.float64,
        )
        deviations[i] = values[i] - _weighted_median(included_values, weights)
    return deviations


def convergence_spread_deg(
    angles: Sequence[float],
    levels: Optional[Sequence[Optional[int]]] = None,
    exclude_s1: bool = True,
) -> float:
    """RMS convergence disagreement of one side, in degrees.

    0.0 when fewer than two screws take part, which is the honest answer: one
    screw always agrees with itself, and reporting anything else would let the
    construct term push a single-level fusion around for nothing.
    """
    deviations = [
        d for d in convergence_deviations_deg(angles, levels, exclude_s1) if d is not None
    ]
    if len(deviations) < 2:
        return 0.0
    return float(np.sqrt(np.mean(np.square(np.asarray(deviations, dtype=np.float64)))))


def optimize_construct(
    per_screw_candidates: Dict[Tuple[str, str], List[Candidate]],
    weights: OptimizerWeights = DEFAULT_WEIGHTS,
    levels: Optional[Mapping[Tuple[str, str], int]] = None,
) -> Dict[Tuple[str, str], Candidate]:
    """Re-rank per-screw candidates so each side agrees with itself.

    Keys are ``(level_name, side)``; ``levels`` maps those keys to
    TotalSegmentator vertebra labels so the convergence term knows which levels
    are neighbours and which screw is sacral.  Without it every screw is an
    unknown level: the convergence median is then the plain side median and no
    screw is excluded.

    Starting from every screw's own best trajectory, greedy coordinate descent
    sweeps the screws (at most :data:`_CONSTRUCT_MAX_PASSES` times, stopping
    early once a pass changes nothing) and swaps in the candidate minimising

    ``-score
      + weights.rod * rod_misalignment_mm(side heads) / ROD_TOLERANCE_MM
      + weights.rod * convergence_spread_deg(side angles) / CONVERGENCE_TOLERANCE_DEG``

    One weight governs both terms because they are one surgical property: a
    construct whose heads line up but whose angles fight each other is not
    harmonised.  Only candidates whose own score stays within
    :data:`CONSTRUCT_SCORE_TOLERANCE` of that screw's best are eligible, so
    neither alignment term can ever buy a materially worse screw.  Levels are
    only ever compared against the same side's screws; the other side's terms
    are constant for that screw and cannot change the choice.

    Two screws on adjacent levels (same level or neighbours by their
    TotalSegmentator labels, every pair when the labels are unknown) must also
    keep clear of each other: any pairing whose tip-to-tip *or* shaft-to-shaft
    distance (segment-segment over the ``(entry, target)`` pairs) falls below
    the sum of the screws' radii plus :data:`INTER_SCREW_CLEARANCE_MM` is
    rejected, even when each screw alone is feasible.  The descent only ever
    moves to a collision-free candidate, starting from the per-screw bests, so
    a screw with no clean alternative keeps its best rather than vanishing.
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

    def alignment_cost(side: str, key: Tuple[str, str], candidate: Candidate) -> float:
        heads: List[np.ndarray] = []
        angles: List[float] = []
        side_levels: List[Optional[int]] = []
        for other, current in chosen.items():
            if other[1] != side:
                continue
            pick = candidate if other == key else current
            heads.append(pick.entry)
            angles.append(pick.convergence_deg)
            side_levels.append(None if levels is None else levels.get(other))
        rod = rod_misalignment_mm(np.asarray(heads, dtype=np.float64)) if heads else 0.0
        spread = convergence_spread_deg(angles, side_levels)
        return weights.rod * (
            rod / ROD_TOLERANCE_MM + spread / CONVERGENCE_TOLERANCE_DEG
        )

    def collides(key: Tuple[str, str], candidate: Candidate) -> bool:
        for other, current in chosen.items():
            if other == key:
                continue
            level_a = None if levels is None else levels.get(key)
            level_b = None if levels is None else levels.get(other)
            if not _adjacent_levels(level_a, level_b):
                continue
            if screws_collide(candidate, current):
                return True
        return False

    for _ in range(_CONSTRUCT_MAX_PASSES):
        changed = False
        for key in list(chosen):
            side = key[1]
            best_candidate = chosen[key]
            if collides(key, best_candidate):
                # A colliding start is infeasible, not a cost to beat: take
                # the best clean alternative even if it scores lower.
                alts = [c for c in eligible[key] if not collides(key, c)]
                if alts:
                    best_candidate = min(
                        alts,
                        key=lambda c: (
                            -c.score + alignment_cost(side, key, c), -c.score,
                        ),
                    )
                    chosen[key] = best_candidate
                    changed = True
                continue
            best_cost = -best_candidate.score + alignment_cost(side, key, best_candidate)
            for candidate in eligible[key]:
                if collides(key, candidate):
                    continue
                cost = -candidate.score + alignment_cost(side, key, candidate)
                if cost < best_cost - 1e-12:
                    best_cost, best_candidate = cost, candidate
            if best_candidate is not chosen[key]:
                chosen[key] = best_candidate
                changed = True
        if not changed:
            break
    return chosen
