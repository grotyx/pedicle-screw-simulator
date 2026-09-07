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
from .vertebra import PedicleAnalysisResult

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .auto_screw_planner import PlannedScrew

logger = logging.getLogger(__name__)

#: How far inferior and medial of the isthmus corner the entry seed sits (mm).
#: The corner is the last pedicle voxel; the pars/lamina mass a CBT screw is
#: started from lies just below and inside it.
ENTRY_INFERIOR_MM = 3.0
ENTRY_MEDIAL_MM = 2.0

#: How far back inside the posterior cortex the entry is seated (mm).
ENTRY_BACKOFF_MM = 1.0

#: Step and reach of the posterior ray-cast that finds the entry surface (mm).
_CAST_STEP_MM = 0.5
_CAST_MAX_MM = 40.0

#: Step of the angle sweep either side of the CBT defaults (degrees).
ANGLE_STEP_DEG = 2.5

#: Wall distance at which the safety objective saturates (mm), and the HU window
#: the density objective is normalised over.  Mirrors the trajectory optimiser so
#: the two families' component scores stay comparable, but CBT weights density at
#: 1.0 rather than 0.5: cortical purchase *is* the technique.
SAFETY_CAP_MM = 3.0
DENSITY_LOW_HU = 100.0
DENSITY_HIGH_HU = 600.0
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
    lamina junction — and backs off :data:`ENTRY_BACKOFF_MM` inside the bone.

    Returns ``None`` when the analysis has no isthmus corner for this side (the
    axial fallback never produces one) or when the ray finds no bone behind the
    seed, which is what an absent lamina or a pars defect looks like here.
    """
    corner = _corner_for_side(analysis, side)
    if corner is None:
        return None

    medial_sign = -1.0 if side == "left" else 1.0
    seed = corner + np.array(
        [medial_sign * ENTRY_MEDIAL_MM, 0.0, -ENTRY_INFERIOR_MM], dtype=np.float64
    )

    steps = np.arange(0.0, _CAST_MAX_MM + 1e-9, _CAST_STEP_MM)
    points = seed[None, :] + np.array([0.0, 1.0, 0.0])[None, :] * steps[:, None]
    d_out, _d_in = grader.distances_at_points(points, int(label))
    inside = np.flatnonzero(d_out <= 0.0)
    if inside.size == 0:
        logger.info("No %s CBT entry: no bone posterior to the isthmus corner", side)
        return None

    surface = seed + np.array([0.0, float(steps[inside[-1]]), 0.0])
    entry = surface - np.array([0.0, ENTRY_BACKOFF_MM, 0.0])
    entry_out, _ = grader.distances_at_points(entry.reshape(1, 3), int(label))
    if entry_out[0] > 0.0:
        # A shell thinner than the back-off; there is nothing to start a screw in.
        logger.info("No %s CBT entry: posterior cortex thinner than the back-off", side)
        return None
    return entry


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
    feasible when it is fully contained (breach 0) and keeps
    ``config.wall_clearance_mm`` of cortical wall, and is ranked by
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
    entry = cbt_entry_point(grader, analysis, side, label)
    if entry is None:
        return None

    pedicle_center, pedicle_width = _side_geometry(analysis, side)
    body_center = analysis.vertebral_body_center
    if pedicle_center is None or body_center is None:
        logger.info("No %s CBT screw: missing isthmus or body centre", side)
        return None

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
    best_key: Optional[Tuple[float, float, float]] = None
    best_pick: Optional[Tuple[int, float, float, float]] = None
    for diameter in sorted(CBT_DEFAULTS["diameter_mm"], reverse=True):
        batch = grader.evaluate_batch(entries, targets, float(diameter), int(label))
        sampled = np.isfinite(batch.mean_hu)
        without_ct = not sampled.any()
        feasible = (
            (batch.breach_mm <= 0.0)
            & (batch.min_wall_mm >= config.wall_clearance_mm - _CLEARANCE_EPS)
            & (sampled | without_ct)
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
        logger.info(
            "No feasible CBT trajectory for %s %s", analysis.vertebra.name, side
        )
        return None

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
    return planned


def plan_cbt_screws(
    grader: ScrewGrader,
    analyses: List[PedicleAnalysisResult],
    sides: List[str],
    config: PlannerConfig,
    skip_labels: Tuple[int, ...] = (),
    planner=None,
) -> List["PlannedScrew"]:
    """Plan a CBT screw per requested side of every analysed vertebra.

    Sides that cannot be solved are dropped rather than falling back to a
    traditional trajectory: the two families place their heads in different
    places, and silently mixing them would break the construct the surgeon asked
    for.
    """
    planner = planner if planner is not None else _make_planner(grader, config)
    results: List["PlannedScrew"] = []
    for analysis in analyses:
        vertebra = analysis.vertebra
        if vertebra.label in skip_labels or not analysis.success:
            continue
        for side in sides:
            planned = plan_cbt_screw(
                grader, analysis, side, vertebra.label, config, planner=planner
            )
            if planned is not None:
                results.append(planned)
    return results
