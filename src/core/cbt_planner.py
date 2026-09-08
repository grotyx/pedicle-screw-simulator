"""Cortical bone trajectory (CBT / mCBT) screw planning.

A CBT screw starts inferomedial to the pedicle isthmus, on the pars/lamina
junction, and runs *cranially and laterally* into the vertebral body — the
mirror image of a traditional convergent pedicle screw.  It is shorter and
narrower and buys its pull-out strength from the cortical bone it engages along
the way rather than from filling the pedicle, which is what makes it attractive
in osteoporotic bone.

Geometry is in the DICOM **LPS** frame used across the app: ``+X`` left, ``+Y``
posterior, ``+Z`` superior.  So "cranial" is ``+Z``, "lateral" is ``+X`` on the
patient's left and ``-X`` on the right, and the insertion direction starts from
the negated posterior entry normal, ``-Y``.

Starting angles and the implant catalogue come from
:data:`~src.utils.constants.CBT_DEFAULTS`; every planned screw carries
:data:`~src.utils.constants.CBT_CONTRAINDICATION_NOTE` as a warning because the
technique is ruled out by findings this planner cannot see.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, List, Optional, Tuple

import numpy as np
import SimpleITK as sitk

from ..utils.constants import CBT_CONTRAINDICATION_NOTE, CBT_DEFAULTS
from .planner_config import PlannerConfig
from .screw_grading import ScrewGrader
from .trajectory_optimizer import (
    DENSITY_HIGH_HU,
    DENSITY_LOW_HU,
    SAFETY_CAP_MM,
    TIP_SEGMENT_MM,
    RunProgress,
)
from .vertebra import PedicleAnalysisResult

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .auto_screw_planner import PlannedScrew

logger = logging.getLogger(__name__)

#: How far inferior and medial of the isthmus corner the entry seed sits (mm).
#: The corner is the last pedicle voxel; the pars/lamina mass a CBT screw is
#: started from lies just below and inside it.
ENTRY_INFERIOR_MM = 3.0
ENTRY_MEDIAL_MM = 2.0

#: How far back inside the posterior cortex the entry is seated (mm).  The
#: actual back-off is at least one voxel of the mask's ``y`` spacing, so that a
#: coarse volume cannot leave the entry in the outermost bone voxel.
ENTRY_BACKOFF_MM = 1.0

#: Step and reach of the posterior ray-cast that finds the entry surface (mm).
_CAST_STEP_MM = 0.5
_CAST_MAX_MM = 40.0

#: Step of the angle sweep either side of the CBT defaults (degrees).
ANGLE_STEP_DEG = 2.5

#: The safety cap and HU window are imported from the trajectory optimiser, not
#: restated, so the two families' component scores cannot drift apart.  Only the
#: weights are local: CBT scores density at 1.0 rather than the optimiser's 0.5
#: because cortical purchase *is* the technique.
SAFETY_WEIGHT = 1.0
DENSITY_WEIGHT = 1.0

#: Float slack on the wall-clearance test.  A CBT entry is seated one voxel
#: inside the posterior cortex by design, so its widest cylinder samples land in
#: the outermost bone layer and measure the clearance exactly, never comfortably.
_CLEARANCE_EPS = 1e-6


def _corner_for_side(analysis: PedicleAnalysisResult, side: str) -> Optional[np.ndarray]:
    if side not in ("left", "right"):
        raise ValueError(f"side must be 'left' or 'right', got {side!r}")
    corner = (
        analysis.left_pedicle_inferior_medial_lps
        if side == "left"
        else analysis.right_pedicle_inferior_medial_lps
    )
    return None if corner is None else np.asarray(corner, dtype=np.float64)


def _side_geometry(
    analysis: PedicleAnalysisResult, side: str
) -> Tuple[Optional[np.ndarray], float]:
    """``(isthmus centre, pedicle width)`` for one side."""
    if side == "left":
        return analysis.left_pedicle_center, float(analysis.left_pedicle_width)
    return analysis.right_pedicle_center, float(analysis.right_pedicle_width)


def _make_planner(grader: ScrewGrader, config: PlannerConfig):
    """An :class:`AutoScrewPlanner` over the grader's own volumes.

    Only its shared grading tail (``_finalise_screw``) is reused, so that a CBT
    screw carries exactly the same metrics, bone-quality assessment and facet
    classification as a traditional one.  :class:`AutoScrewPlanner` requires a CT
    to construct, so a grader built without one gets a zero-filled stand-in;
    every HU reported here comes from the grader, never from that image.
    """
    from .auto_screw_planner import AutoScrewPlanner  # local import: avoids a cycle

    mask = grader.mask_image()
    ct = grader.ct_image()
    if ct is None:
        ct = sitk.Image(mask.GetSize(), sitk.sitkInt16)
        ct.CopyInformation(mask)
    return AutoScrewPlanner(ct, mask, grader=grader, config=config)


def cbt_entry_point(
    grader: ScrewGrader,
    analysis: PedicleAnalysisResult,
    side: str,
    label: int,
) -> Optional[np.ndarray]:
    """The CBT starting point on the pars/lamina, in LPS.

    Starts from the isthmus slice's inferior-medial corner
    (:attr:`PedicleAnalysisResult.left_pedicle_inferior_medial_lps` and its
    right-hand twin), steps :data:`ENTRY_INFERIOR_MM` inferior and
    :data:`ENTRY_MEDIAL_MM` medial to clear the pedicle itself, then casts
    posteriorly (``+Y``) to the last ``label`` voxel on that ray — the pars /
    lamina junction — and backs off :data:`ENTRY_BACKOFF_MM` (or one voxel,
    whichever is deeper) inside the bone.

    Returns ``None`` when the analysis has no isthmus corner for this side (the
    axial fallback never produces one) or when the ray finds no bone behind the
    seed, which is what an absent lamina or a pars defect looks like here.
    """
    return _cbt_entry_point(grader, analysis, side, label)[0]


def _cbt_entry_point(
    grader: ScrewGrader,
    analysis: PedicleAnalysisResult,
    side: str,
    label: int,
) -> Tuple[Optional[np.ndarray], str]:
    """:func:`cbt_entry_point` plus the reason it gave up, for the caller to report."""
    corner = _corner_for_side(analysis, side)
    if corner is None:
        return None, "no inferior-medial isthmus corner in the pedicle analysis"

    medial_sign = -1.0 if side == "left" else 1.0
    seed = corner + np.array(
        [medial_sign * ENTRY_MEDIAL_MM, 0.0, -ENTRY_INFERIOR_MM], dtype=np.float64
    )

    steps = np.arange(0.0, _CAST_MAX_MM + 1e-9, _CAST_STEP_MM)
    points = seed[None, :] + np.array([0.0, 1.0, 0.0])[None, :] * steps[:, None]
    d_out, _d_in = grader.distances_at_points(points, int(label))
    inside = np.flatnonzero(d_out <= 0.0)
    if inside.size == 0:
        return None, "no bone posterior to the isthmus corner"

    # ``inside[-1]`` is the last sample still in bone, i.e. the outermost bone
    # voxel on the ray.  Backing off by less than one voxel would leave the entry
    # in that same voxel, where the cylinder's widest samples immediately breach;
    # on a volume with sub-millimetre spacing the nominal back-off is deeper
    # still, so take whichever is larger.
    spacing_y = float(grader.mask_image().GetSpacing()[1])
    surface = seed + np.array([0.0, float(steps[inside[-1]]), 0.0])
    entry = surface - np.array([0.0, max(ENTRY_BACKOFF_MM, spacing_y), 0.0])
    entry_out, _ = grader.distances_at_points(entry.reshape(1, 3), int(label))
    if entry_out[0] > 0.0:
        # A shell thinner than the back-off; there is nothing to start a screw in.
        return None, "posterior cortex thinner than the entry back-off"
    return entry, ""


def cbt_directions(side: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The swept insertion directions, with their cranial and lateral angles.

    Returns ``(directions (N, 3), cranial_deg (N,), lateral_deg (N,))``.  Each
    direction starts from the negated posterior entry normal (``-Y``), turns
    ``lateral_deg`` away from the midline for ``side`` and rises ``cranial_deg``
    above the axial plane, sweeping
    :data:`~src.utils.constants.CBT_DEFAULTS`\\ ``["angle_search_deg"]`` either
    side of the consensus angles in :data:`ANGLE_STEP_DEG` steps.
    """
    search = float(CBT_DEFAULTS["angle_search_deg"])
    cranial_mid = float(CBT_DEFAULTS["cranial_angle_deg"])
    lateral_mid = float(CBT_DEFAULTS["lateral_angle_deg"])
    cranials = np.arange(cranial_mid - search, cranial_mid + search + 1e-9, ANGLE_STEP_DEG)
    laterals = np.arange(lateral_mid - search, lateral_mid + search + 1e-9, ANGLE_STEP_DEG)
    lateral_sign = 1.0 if side == "left" else -1.0

    grid_cranial, grid_lateral = (a.reshape(-1) for a in np.meshgrid(cranials, laterals, indexing="ij"))
    cr = np.radians(grid_cranial)
    la = np.radians(grid_lateral)
    directions = np.stack(
        [
            lateral_sign * np.sin(la) * np.cos(cr),
            -np.cos(la) * np.cos(cr),
            np.sin(cr),
        ],
        axis=1,
    )
    return directions, grid_cranial, grid_lateral


def plan_cbt_screw(
    grader: ScrewGrader,
    analysis: PedicleAnalysisResult,
    side: str,
    label: int,
    config: PlannerConfig,
    planner=None,
) -> Optional["PlannedScrew"]:
    """Plan one cortical bone trajectory screw, or ``None`` if none is feasible.

    Every combination of swept direction (:func:`cbt_directions`) and CBT
    catalogue diameter and length is graded in one
    :meth:`ScrewGrader.evaluate_batch` pass per diameter.  A candidate is
    feasible when it is fully contained (breach 0), keeps
    ``config.wall_clearance_mm`` of cortical wall, and clears
    ``config.anterior_margin_mm`` at the tip — the last measured over its distal
    :data:`~src.core.trajectory_optimizer.TIP_SEGMENT_MM` exactly as the
    optimiser measures it.  Feasible candidates are ranked by
    ``safety + density`` — the optimiser's two bone-contact objectives, with
    density at full weight because cortical purchase is the whole point of CBT.
    Ties go to the longer screw, then the wider one.

    The returned screw is finalised through the same tail as every other planned
    screw, so its grade, HU statistics, bone-quality metrics and facet
    classification are directly comparable with a traditional trajectory.  It is
    marked ``metrics["trajectory_type"] == "cbt"`` and carries
    :data:`~src.utils.constants.CBT_CONTRAINDICATION_NOTE`.

    ``planner`` reuses a caller's :class:`AutoScrewPlanner` (which caches full
    CT and mask arrays) instead of building one per screw.
    """
    return _plan_cbt_screw(grader, analysis, side, label, config, planner=planner)[0]


def _plan_cbt_screw(
    grader: ScrewGrader,
    analysis: PedicleAnalysisResult,
    side: str,
    label: int,
    config: PlannerConfig,
    planner=None,
) -> Tuple[Optional["PlannedScrew"], str]:
    """:func:`plan_cbt_screw` plus the reason it gave up, for the caller to report.

    A dropped side is a construct the surgeon did not ask for, so the reason has
    to travel back out to the UI rather than only into the log.
    """
    entry, reason = _cbt_entry_point(grader, analysis, side, label)
    if entry is None:
        logger.info("No %s CBT entry for %s: %s", side, analysis.vertebra.name, reason)
        return None, reason

    pedicle_center, pedicle_width = _side_geometry(analysis, side)
    body_center = analysis.vertebral_body_center
    if pedicle_center is None or body_center is None:
        reason = "missing isthmus or body centre"
        logger.info("No %s CBT screw: %s", side, reason)
        return None, reason

    directions, cranial_deg, _lateral_deg = cbt_directions(side)
    lengths = np.asarray(CBT_DEFAULTS["lengths_mm"], dtype=np.float64)
    n_dir, n_len = directions.shape[0], lengths.shape[0]

    # One row per (direction, length): the entry is fixed, so the whole sweep is
    # a single outer product.
    dirs = np.repeat(directions, n_len, axis=0)
    cranials = np.repeat(cranial_deg, n_len)
    span = np.tile(lengths, n_dir)
    entries = np.repeat(entry[None, :], dirs.shape[0], axis=0)
    targets = entries + dirs * span[:, None]

    # Ranked by (score, length, diameter): ties on the objective go to the
    # longer screw, then the wider one.
    # The anterior safety margin is a property of the *tip*, not of the whole
    # screw: a trajectory can keep a millimetre of wall along its shaft and still
    # end a millimetre behind the anterior cortex.  Measured over the distal
    # TIP_SEGMENT_MM exactly as the optimiser measures it, so a CBT screw and a
    # traditional one are held to the same anterior rule.
    tip_margin = max(config.anterior_margin_mm - config.wall_clearance_mm, 0.0)
    tip_entries = targets - dirs * TIP_SEGMENT_MM

    best_key: Optional[Tuple[float, float, float]] = None
    best_pick: Optional[Tuple[int, float, float, float]] = None
    for diameter in sorted(CBT_DEFAULTS["diameter_mm"], reverse=True):
        batch = grader.evaluate_batch(entries, targets, float(diameter), int(label))
        sampled = np.isfinite(batch.mean_hu)
        without_ct = not sampled.any()
        tip_batch = grader.evaluate_batch(
            tip_entries, targets, float(diameter), int(label)
        )
        feasible = (
            (batch.breach_mm <= 0.0)
            & (batch.min_wall_mm >= config.wall_clearance_mm - _CLEARANCE_EPS)
            & (sampled | without_ct)
            & (tip_batch.breach_mm <= 0.0)
            & (tip_batch.min_wall_mm >= tip_margin)
        )
        if not feasible.any():
            continue

        safety = np.clip(np.minimum(batch.min_wall_mm, SAFETY_CAP_MM) / SAFETY_CAP_MM, 0.0, 1.0)
        density = np.clip(
            (np.where(sampled, batch.mean_hu, DENSITY_LOW_HU) - DENSITY_LOW_HU)
            / (DENSITY_HIGH_HU - DENSITY_LOW_HU),
            0.0,
            1.0,
        )
        score = SAFETY_WEIGHT * safety + DENSITY_WEIGHT * density
        for i in np.flatnonzero(feasible):
            key = (float(score[i]), float(span[i]), float(diameter))
            if best_key is None or key > best_key:
                best_key = key
                best_pick = (int(i), float(diameter), float(safety[i]), float(density[i]))
    if best_key is None or best_pick is None:
        reason = "no contained trajectory with the required wall clearance"
        logger.info(
            "No feasible CBT trajectory for %s %s: %s",
            analysis.vertebra.name, side, reason,
        )
        return None, reason

    score_value, length_value, _ = best_key
    index, diameter, safety_value, density_value = best_pick
    target = targets[index]

    planner = planner if planner is not None else _make_planner(grader, config)
    planned = planner._finalise_screw(
        analysis.vertebra,
        side,
        entry,
        target,
        diameter,
        pedicle_center,
        body_center,
        [CBT_CONTRAINDICATION_NOTE],
        pedicle_width,
    )
    # ``span`` is the catalogue length by construction; recomputing it from the
    # end points would only add float noise to a nominal implant size.
    planned.length_mm = length_value
    planned.metrics["trajectory_type"] = "cbt"
    planned.metrics["cbt_cranial_angle_deg"] = float(cranials[index])
    planned.metrics["score"] = score_value
    planned.metrics["score_components"] = {
        "safety": safety_value,
        "density": density_value,
    }
    return planned, ""


def plan_cbt_screws(
    grader: ScrewGrader,
    analyses: List[PedicleAnalysisResult],
    sides: List[str],
    config: PlannerConfig,
    skip_labels: Tuple[int, ...] = (),
    planner=None,
    progress: Optional[RunProgress] = None,
    skipped: Optional[List[Tuple[str, str, str]]] = None,
) -> List["PlannedScrew"]:
    """Plan a CBT screw per requested side of every analysed vertebra.

    Sides that cannot be solved are dropped rather than falling back to a
    traditional trajectory: the two families place their heads in different
    places, and silently mixing them would break the construct the surgeon asked
    for.  Dropping is the policy; hiding it is not, so every dropped side is
    appended to ``skipped`` as ``(vertebra name, side, reason)`` for the caller
    to put in front of the surgeon.  Vertebrae in ``skip_labels`` are not
    recorded: excluding the sacrum is a request, not a failure.

    ``progress`` narrates and, when its ``cancel`` fires, stops the run between
    sides; the screws planned so far are returned.
    """
    planner = planner if planner is not None else _make_planner(grader, config)
    results: List["PlannedScrew"] = []
    for analysis in analyses:
        vertebra = analysis.vertebra
        if vertebra.label in skip_labels:
            continue
        for side in sides:
            if progress is not None and not progress.start(vertebra.name, side):
                return results
            if not analysis.success:
                _record_skip(skipped, vertebra.name, side, "pedicle analysis failed")
                continue
            planned, reason = _plan_cbt_screw(
                grader, analysis, side, vertebra.label, config, planner=planner
            )
            if planned is None:
                _record_skip(skipped, vertebra.name, side, reason)
            else:
                results.append(planned)
    return results


def _record_skip(
    skipped: Optional[List[Tuple[str, str, str]]],
    name: str,
    side: str,
    reason: str,
) -> None:
    if skipped is not None:
        skipped.append((name, side, reason or "no feasible trajectory"))
