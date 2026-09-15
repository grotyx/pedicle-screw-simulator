"""
Automatic pedicle screw trajectory planning.

Uses pedicle analysis results to compute optimal entry/target points
for pedicle screws with safety validation.  All spatial calculations
operate in the DICOM **LPS** coordinate system:

    L (+X)  --  Patient's Left
    P (+Y)  --  Patient's Posterior
    S (+Z)  --  Patient's Superior

Key directions:
    Posterior  =  +Y
    Anterior   =  -Y
    Left       =  +X
    Right      =  -X
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import SimpleITK as sitk

from .bone_quality import assess_bone_quality
from .breach_classification import facet_violation_grade, heary_direction, medial_breach_warning
from .cbt_planner import plan_cbt_screws
from .construct_alignment import stamp_alignment
from .pedicle_analyzer import (
    ENDPLATE_FIT_RMSE_WARNING_MM,
    ENDPLATE_REFERENCE_NEIGHBOURS,
    ENDPLATE_REFERENCE_NONE,
    ENDPLATE_REFERENCE_OWN,
    VERTEBRA_LABELS,
    endplate_fit_warning,
    endplate_neighbour_reference_warning,
    endplate_no_reference_warning,
    resolve_endplate_references,
)
from .planner_config import PlannerConfig
from .screw_geometry import convergence_angle_deg, craniocaudal_angle_deg, endplate_angle_deg
from .screw_grading import ENTRY_ZONE_MM, ScrewGrader, resample_mask_to_ct
from .trajectory_optimizer import (
    COLLIDING_SCREWS_SKIP_REASON,
    Candidate,
    RunProgress,
    optimize_construct,
    optimize_screw,
)
from .vertebra import PedicleAnalysisResult, Vertebra, aiming_endplate_normal

logger = logging.getLogger(__name__)

# Sacrum label -- not suitable for pedicle screw planning.
_SACRUM_LABEL = 25

#: Warning attached to a screw the optimiser could not solve.
OPTIMIZER_FALLBACK_WARNING = (
    "Optimizer found no feasible trajectory; legacy planner used"
)

#: Warning attached to a screw planned from a width the analyser flagged.
WIDTH_UNCERTAIN_SCREW_WARNING = "Pedicle width uncertain – verify diameter"

#: Warning attached to a legacy screw placed at grade B with the
#: :attr:`~src.core.planner_config.PlannerConfig.accept_grade_b` opt-in.
#: Planner-owned: it describes how the screw was *chosen*, so like the narrow
#: note it is deliberately absent from
#: :data:`src.tools.screw_tool._DERIVED_WARNING_PREFIXES` and survives a
#: re-grade verbatim.
GRADE_B_ACCEPTED_WARNING = "Grade B accepted (< 2 mm breach) — verify on CT"

#: Grade ladder the legacy no-worse guard ranks by.  The medial-breach key's
#: (medial, total) millimetres cannot see a grade change inside the same
#: numbers -- but with accept_grade_b on, "B, 1.5 mm then A, 1.8 mm" is a
#: worse screw at a bigger breach the key alone would bless.  The guard
#: compares grades first and only consults the key within the same grade.
_GRADE_ORDER = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}

#: Prefix of the note a narrow pedicle carries.  Planner-owned: it describes how
#: the screw was *chosen*, not how it grades, so it is deliberately absent from
#: :data:`src.tools.screw_tool._DERIVED_WARNING_PREFIXES` and survives a
#: re-grade verbatim.
NARROW_PEDICLE_WARNING_PREFIX = "Narrow pedicle"


def narrow_pedicle_warning(pedicle_width_mm: float, diameter_mm: float) -> str:
    """Why this level got the smallest screw, and what the surgeon has to decide.

    The percentage is the honest number the fill-ratio rule refused: a 4.0 mm
    screw in a 4.5 mm pedicle fills 89 % of it, which is either a mismeasured
    pedicle or an accepted in-out-in trajectory -- and only the surgeon can say
    which.
    """
    width = float(pedicle_width_mm)
    fill = 100.0 * float(diameter_mm) / width if width > 0.0 else 100.0
    return (
        f"{NARROW_PEDICLE_WARNING_PREFIX} ({width:.1f} mm): {float(diameter_mm):.1f} mm "
        f"screw is {fill:.0f} % of the width — verify the measurement or accept a "
        "lateral (in-out-in) breach"
    )


def _endplate_warnings(analysis: PedicleAnalysisResult, endplate_parallel: bool) -> List[str]:
    """The endplate-fit warnings for one analysis, shared by both planning modes.

    With ``endplate_parallel`` off the trajectory is deliberately horizontal,
    so only today's rough-fit warning on the *own* measurement still applies
    -- the reference resolution decided nothing this screw is aimed by, and
    saying otherwise would misdescribe a setting the surgeon chose.  With it
    on, the wording follows ``analysis.endplate_reference``:

    * ``own`` -- :func:`~src.core.pedicle_analyzer.endplate_fit_warning` when
      the RMSE exceeds :data:`ENDPLATE_FIT_RMSE_WARNING_MM`, as before.
    * ``neighbours`` -- :func:`endplate_neighbour_reference_warning`.
    * ``none`` with no own normal at all -- today's exact wording for a
      missing endplate.
    * ``none`` with a rough own normal -- :func:`endplate_no_reference_warning`.
    """
    rmse = analysis.endplate_fit_rmse_mm
    if not endplate_parallel:
        if rmse is not None and rmse > ENDPLATE_FIT_RMSE_WARNING_MM:
            return [endplate_fit_warning(rmse)]
        return []

    reference = analysis.endplate_reference
    if reference == ENDPLATE_REFERENCE_OWN or reference == "":
        # "" is the unresolved case: behave exactly as before resolution
        # existed, since some callers (a lone analysis not yet passed through
        # resolve_endplate_references) may still reach here that way.
        if rmse is not None and rmse > ENDPLATE_FIT_RMSE_WARNING_MM:
            return [endplate_fit_warning(rmse)]
        return []
    if reference == ENDPLATE_REFERENCE_NEIGHBOURS:
        return [
            endplate_neighbour_reference_warning(
                rmse, analysis.endplate_reference_levels
            )
        ]
    assert reference == ENDPLATE_REFERENCE_NONE  # only remaining possibility
    if analysis.upper_endplate_normal is None:
        return ["Upper endplate unavailable; used horizontal sagittal trajectory"]
    return [endplate_no_reference_warning(rmse)]


def construct_summary(screws: Sequence[PlannedScrew]) -> str:
    """How well the finished construct agrees with itself, as the status line words it.

    Reads the metrics the construct stage stamped rather than re-measuring, so
    the number the surgeon reads is the one the optimiser actually minimised.
    Returns ``""`` when no screw carries them -- legacy planning never runs the
    construct stage, and reporting "rod fit 0.0 mm" for a construct that was
    never harmonised would be a lie in the safest-looking direction.
    """
    rod: Dict[str, float] = {}
    spread: Dict[str, float] = {}
    for screw in screws:
        for bucket, key in ((rod, "rod_misalignment_mm"), (spread, "convergence_spread_deg")):
            value = screw.metrics.get(key)
            if value is None:
                continue
            try:
                bucket[screw.side] = float(value)
            except (TypeError, ValueError):
                logger.warning("Ignoring unreadable construct metric %s=%r", key, value)

    sides = (("left", "L"), ("right", "R"))
    parts: List[str] = []
    rod_text = ", ".join(f"{rod[s]:.1f} mm ({i})" for s, i in sides if s in rod)
    if rod_text:
        parts.append(f"rod fit {rod_text}")
    spread_text = ", ".join(f"{spread[s]:.1f}° ({i})" for s, i in sides if s in spread)
    if spread_text:
        parts.append(f"convergence spread {spread_text}")
    if not parts:
        return ""
    return "Construct: " + " · ".join(parts)


#: How many ranked trajectories per pedicle the construct stage may choose from.
#: Wide enough that the pool spans several convergence bins (see
#: :func:`~src.core.trajectory_optimizer._cover_convergence_bins`): a construct
#: cannot harmonise angles it was never offered.
_CONSTRUCT_TOP_K = 40

#: Level name -> TotalSegmentator label, so a construct assembled from
#: :class:`PlannedScrew` (which carries the name, not the label) still knows
#: which levels are neighbours and which one is sacral.
_LEVEL_LABELS: Dict[str, int] = {name: label for label, name in VERTEBRA_LABELS.items()}


@dataclass
class PlannedScrew:
    """Result of automatic screw trajectory planning."""

    vertebra_name: str          # "L4", "T12", etc.
    side: str                   # "left" or "right"
    entry_lps: np.ndarray       # Entry point in LPS (3,)
    target_lps: np.ndarray      # Target point in LPS (3,)
    length_mm: float
    diameter_mm: float          # Recommended diameter based on pedicle width
    convergence_angle: float    # Medial angulation (degrees)
    craniocaudal_angle: float   # Sagittal angulation (degrees)
    mean_bone_density: float    # Average HU along trajectory
    min_bone_density: float     # Minimum HU along trajectory
    gertzbein_grade: str        # "A", "B", "C", "D", "E"
    confidence: float           # 0.0 -- 1.0
    breach_mm: float = 0.0      # Maximum cortical breach depth (mm)
    min_wall_mm: float = 0.0    # Thinnest cortical wall clearance (mm)
    warnings: List[str] = field(default_factory=list)
    #: Clinical metric bundle; see :attr:`src.models.screw.Screw.metrics`.
    metrics: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.entry_lps = np.asarray(self.entry_lps, dtype=np.float64)
        self.target_lps = np.asarray(self.target_lps, dtype=np.float64)


class AutoScrewPlanner:
    """Plan optimal pedicle screw trajectories from pedicle analysis results.

    Algorithm
    ---------
    1. Entry point: posterior surface along pedicle axis from isthmus centre.
    2. Target point: anterior vertebral body with safety margin.
    3. Trajectory optimization: medial convergence, craniocaudal alignment.
    4. Safety validation: HU sampling along path, breach detection.
    """

    # -- Screw sizing rules --------------------------------------------------
    MIN_SCREW_DIAMETER: float = 4.0    # mm
    MAX_SCREW_DIAMETER: float = 7.5    # mm
    DIAMETER_WIDE_HEADROOM: float = 1.0  # mm before one-step upsize

    MAX_BONE_CORRIDOR_SCAN: float = 75.0  # mm, maximum anterior ray-cast distance

    # -- HU thresholds -------------------------------------------------------
    BONE_HU_MIN: float = 200          # Below this = outside bone
    CORTICAL_HU: float = 800          # Cortical bone threshold

    # -- Sampling parameters -------------------------------------------------
    TRAJECTORY_SAMPLE_STEP: float = 0.5   # mm between samples
    ENTRY_BACKOFF: float = 1.0            # mm back inside bone from surface
    TARGET_LATERAL_OFFSETS: Tuple[float, ...] = (
        0.0,
        3.0,
        6.0,
        9.0,
        12.0,
        15.0,
        18.0,
    )

    #: Lateral entry shifts (mm) a narrow side is tried at before the medial
    #: wall is given up on.  The legacy path cannot step the diameter down --
    #: it is already on the smallest implant -- so sliding the corridor away
    #: from the canal is the only protection left, and it is the same move the
    #: optimiser makes with :data:`NARROW_ENTRY_GRID_MM`.  A lateral wall
    #: breach is recoverable; a medial one is the nerve root and the canal.
    NARROW_LATERAL_SHIFTS_MM: Tuple[float, ...] = (0.0, 1.0, 2.0, 3.0)

    @property
    def MIN_SCREW_LENGTH(self) -> float:
        return self.config.implant_lengths_mm[0]

    @property
    def MAX_SCREW_LENGTH(self) -> float:
        return self.config.implant_lengths_mm[-1]

    @property
    def ANTERIOR_SAFETY_MARGIN(self) -> float:
        return self.config.anterior_margin_mm

    @property
    def MAX_CONVERGENCE_ANGLE(self) -> float:
        return self.config.max_convergence_deg

    def __init__(
        self,
        ct_image: sitk.Image,
        mask_image: sitk.Image,
        grader: Optional[ScrewGrader] = None,
        config: Optional[PlannerConfig] = None,
    ) -> None:
        """
        Args:
            ct_image:  Original CT volume (for HU sampling).
            mask_image:  TotalSegmentator segmentation mask.
            grader:  Optional pre-built grader (defaults to one over this pair).
            config:  Optional sizing/safety rules (defaults to :class:`PlannerConfig`).
        """
        self.config = config or PlannerConfig()
        self.config.validate()
        # The grader samples the CT by mask index, so the two must share a grid.
        # Size equality is not enough: a mask saved as NIfTI comes back with a
        # float32 origin, and a mask segmented on a resampled copy of the study
        # can match in size while sitting half a voxel off.
        if not ScrewGrader.grids_match(ct_image, mask_image):
            mask_image = resample_mask_to_ct(mask_image, ct_image)
        self._ct = ct_image
        self._mask = mask_image
        self._ct_array: np.ndarray = sitk.GetArrayFromImage(ct_image)   # (z, y, x)
        self._mask_array: np.ndarray = sitk.GetArrayFromImage(mask_image)
        self._grader = grader or ScrewGrader(mask_image, ct_image)
        #: Whether the last :meth:`plan_all` stopped early on its ``cancel``.
        self.last_run_cancelled: bool = False
        #: ``(vertebra name, side, reason)`` per side the last :meth:`plan_all`
        #: could not place a screw on -- filled in by every path (legacy,
        #: optimiser and CBT).  A side excluded by policy rather than by failure
        #: (the sacrum) is not listed: not planning it is what was asked for.
        self.skipped_sides: List[Tuple[str, str, str]] = []

    # =====================================================================
    # Public API
    # =====================================================================

    def plan_screw(
        self,
        analysis: PedicleAnalysisResult,
        side: str,
    ) -> Optional[PlannedScrew]:
        """Plan a single pedicle screw for one side of one vertebra.

        Returns ``None`` if planning fails (e.g. no pedicle detected,
        sacrum, no reachable entry point).
        """
        return self._plan_screw(analysis, side)[0]

    def _plan_screw(
        self,
        analysis: PedicleAnalysisResult,
        side: str,
    ) -> Tuple[Optional[PlannedScrew], Optional[str]]:
        """:meth:`plan_screw` plus why it gave up, for the caller to report.

        The second element is a short reason phrase when the side *failed*, and
        ``None`` when it was excluded on purpose (the sacrum) -- the difference
        between a construct the surgeon is missing and one they did not ask for.
        """
        if side not in ("left", "right"):
            raise ValueError(f"side must be 'left' or 'right', got {side!r}")

        vertebra = analysis.vertebra

        # Sacrum is not suitable for pedicle screw placement.
        if vertebra.label == _SACRUM_LABEL:
            logger.info("Skipping sacrum (label %d) for screw planning", vertebra.label)
            return None, None

        if not analysis.success:
            logger.info("Skipping %s: pedicle analysis unsuccessful", vertebra.name)
            return None, "pedicle analysis unsuccessful"

        # A caller that hands _plan_screw a single analysis directly (rather
        # than through plan_all, which resolves the whole batch up front)
        # still needs a reference before it can aim or warn -- resolved alone,
        # it can only ever land on its own fit or none, never neighbours.
        if analysis.endplate_reference == "":
            resolve_endplate_references([analysis])

        # Retrieve side-specific pedicle data.
        pedicle_center, pedicle_axis, pedicle_width = self._get_side_data(analysis, side)
        if pedicle_center is None or pedicle_axis is None:
            logger.info("No %s pedicle data for %s", side, vertebra.name)
            return None, f"no {side} pedicle measured"

        body_center = analysis.vertebral_body_center
        if body_center is None:
            logger.info("No vertebral body centre for %s", vertebra.name)
            return None, "no vertebral body centre"

        warnings: List[str] = []
        # ``endplate_normal`` is what the trajectory is *aimed* along -- None
        # when the user switched the endplate-parallel option off (a
        # deliberate horizontal trajectory) or when neither this level's own
        # fit nor a neighbour's is trustworthy.  ``aiming_endplate_normal``
        # resolves against the reference resolve_endplate_references chose,
        # which is also what the screw is *measured* against below.
        endplate_normal = (
            aiming_endplate_normal(analysis) if self.config.endplate_parallel else None
        )
        warnings.extend(_endplate_warnings(analysis, self.config.endplate_parallel))

        # 1. Determine optimal screw diameter.  A width is never a reason to
        #    drop the side: a narrow pedicle takes the smallest implant.
        uncertain = self._is_width_uncertain(analysis, side)
        narrow = self._is_narrow_side(analysis, side)
        if uncertain:
            # The number is not a finding about the patient, so the warning may
            # not present it as one.
            warnings.append(WIDTH_UNCERTAIN_SCREW_WARNING)
        diameter = (
            self.MIN_SCREW_DIAMETER
            if narrow
            else self._compute_diameter(
                self._effective_width(analysis, side), vertebra.name
            )
        )

        # 2. Orient pedicle axis so it points posteriorly (+Y in LPS).
        #    Validate that PCA axis is roughly AP-directed (>30% Y component).
        #    Fall back to body_center→pedicle_center direction if PCA is unreliable.
        oriented_axis = self._orient_axis_posterior(pedicle_axis)
        ap_component = abs(oriented_axis[1])
        if ap_component < 0.3:
            # PCA axis not aligned with AP direction — use anatomical fallback
            fallback_dir = pedicle_center - body_center
            fb_norm = np.linalg.norm(fallback_dir)
            if fb_norm > 1e-9:
                oriented_axis = self._orient_axis_posterior(fallback_dir / fb_norm)
                warnings.append("PCA axis unreliable; used anatomical fallback direction")

        # 3. Find entry point on posterior bone surface.
        entry = self._find_entry_point(
            pedicle_center, oriented_axis, vertebra.label,
        )
        if entry is None:
            logger.info("Entry point search failed for %s %s", vertebra.name, side)
            return None, f"no {side} entry point on the posterior surface"

        # 4. Find target point in anterior vertebral body.
        target = self._find_best_target(
            entry,
            oriented_axis,
            body_center,
            vertebra.label,
            diameter,
            endplate_normal,
        )
        if target is None:
            logger.info("Target point search failed for %s %s", vertebra.name, side)
            return None, f"no {side} target point in the vertebral body"

        # 4b. A narrow side is already on the smallest implant, so the
        #     diameter step-down below cannot help it.  Slide the entry
        #     laterally instead and keep whichever trajectory leaves the least
        #     screw in the canal.  The side is never dropped for width: if even
        #     the best shift breaches medially it is placed and warned about.
        if narrow:
            entry, target, lateral_shift = self._least_medial_entry(
                side,
                entry,
                target,
                oriented_axis,
                pedicle_center,
                body_center,
                vertebra.label,
                diameter,
                endplate_normal,
            )
            if lateral_shift > 0.0:
                warnings.append(
                    f"Entry moved {lateral_shift:.0f} mm laterally to protect "
                    f"the medial wall"
                )

        # 5. Enforce length constraints.
        length = float(np.linalg.norm(target - entry))
        if length < self.MIN_SCREW_LENGTH:
            logger.info(
                "Rejecting short trajectory for %s %s: %.1f mm",
                vertebra.name,
                side,
                length,
            )
            return None, (
                f"{side} corridor is {length:.1f} mm, shorter than the "
                f"{self.MIN_SCREW_LENGTH:.0f} mm minimum implant"
            )

        if length > self.MAX_SCREW_LENGTH:
            direction = target - entry
            direction /= np.linalg.norm(direction)
            target = entry + direction * self.MAX_SCREW_LENGTH
            length = self.MAX_SCREW_LENGTH

        initial_diameter = diameter
        grade, breach_dist = self._evaluate_gertzbein_grade(
            entry,
            target,
            diameter,
            vertebra.label,
        )
        # A narrow side has nothing to step down to -- it is already on the
        # smallest implant -- and stepping down would only shrink the screw
        # without buying containment, so it is graded and placed as it stands.
        accepted = {"A", "B"} if self.config.accept_grade_b else {"A"}
        while not narrow and grade not in accepted:
            smaller = self._next_smaller_diameter(diameter)
            if smaller is None:
                logger.info(
                    "No contained screw diameter for %s %s",
                    vertebra.name,
                    side,
                )
                return None, f"no contained {side} screw diameter in the catalogue"
            diameter = smaller
            candidate = self._find_best_target(
                entry,
                oriented_axis,
                body_center,
                vertebra.label,
                diameter,
                endplate_normal,
            )
            if candidate is None:
                return None, f"no {side} target point at the reduced diameter"
            target = candidate
            length = float(np.linalg.norm(target - entry))
            grade, breach_dist = self._evaluate_gertzbein_grade(
                entry,
                target,
                diameter,
                vertebra.label,
            )
        if diameter < initial_diameter:
            warnings.append(
                f"Diameter reduced from {initial_diameter:.1f} to "
                f"{diameter:.1f} mm for cortical containment"
            )
        if not narrow and grade == "B" and self.config.accept_grade_b:
            # The opt-in above is a choice, not a measurement: the grade
            # itself is recorded by _finalise_screw, but the choice to stop
            # stepping down at B has to travel with the screw so a re-grade
            # keeps it -- and so the record never reads as a silently placed
            # grade-A screw.
            warnings.append(GRADE_B_ACCEPTED_WARNING)

        # 6. Head on the dorsal cortex, tip at the anterior margin -- the same
        #    rule the optimiser applies, so a fallback screw is not the short,
        #    buried one the optimiser was changed to stop producing. Applied
        #    only when the result is no less safe than the validated screw
        #    above.
        entry, target = self._seat_head_and_extend(
            entry, target, diameter, vertebra.label, side
        )

        planned = self._finalise_screw(
            vertebra,
            side,
            entry,
            target,
            diameter,
            pedicle_center,
            body_center,
            warnings,
            pedicle_width,
            upper_endplate_normal=aiming_endplate_normal(analysis),
            narrow=narrow,
            width_uncertain=uncertain,
            endplate_reference=analysis.endplate_reference,
            endplate_reference_levels=analysis.endplate_reference_levels,
        )
        return planned, None

    def _longest_length_from(
        self,
        head: np.ndarray,
        direction: np.ndarray,
        vertebra_label: int,
    ) -> Optional[float]:
        """The longest catalogue length from ``head`` that keeps the anterior margin.

        Walks forward along ``direction`` to where the centreline leaves the
        vertebra, takes off :attr:`PlannerConfig.anterior_margin_mm`, and returns
        the longest :attr:`PlannerConfig.implant_lengths_mm` entry that still
        fits -- or ``None`` when not even the shortest does.  Measured along the
        screw's own axis: the old target search measured toward the vertebral
        body centre instead, which is not the direction the screw goes.
        """
        step = self.TRAJECTORY_SAMPLE_STEP
        steps = np.arange(step, self.MAX_BONE_CORRIDOR_SCAN + 1e-9, step)
        points = head[None, :] + direction[None, :] * steps[:, None]
        d_out, _ = self._grader.distances_at_points(points, int(vertebra_label))
        outside = np.nonzero(d_out > 0.0)[0]
        reach = float(steps[outside[0] - 1]) if outside.size and outside[0] > 0 else (
            0.0 if outside.size else float(steps[-1])
        )
        available = reach - float(self.config.anterior_margin_mm)
        fitting = [
            float(length)
            for length in self.config.implant_lengths_mm
            if float(length) <= available + 1e-9
        ]
        return max(fitting) if fitting else None

    def _seat_head_and_extend(
        self,
        entry: np.ndarray,
        target: np.ndarray,
        diameter: float,
        vertebra_label: int,
        side: str,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Move the head out to the dorsal cortex and the tip up to the margin.

        The legacy entry is found along the *pedicle axis* and backed 1 mm into
        bone, and the chosen trajectory then leaves along a different
        direction, so measured along the screw itself the head was buried and
        the tip stopped short of the anterior margin.

        The trajectory itself -- the (entry, target) pair handed in -- was
        already chosen and validated by the legacy search: containment, the
        diameter step-down, or the narrow policy's lateral slide.  That
        unchanged, unextended screw is therefore the safety baseline, never
        the thing being improved on.  Re-seating the head and extending the
        tip only makes the screw longer along the *same* line, and
        ``_longest_length_from`` checks only that the centreline stays in
        bone -- so an extension can walk the full-diameter shaft into a wall
        the 0-radius centreline never touches, and a head re-seated onto the
        first surface ``seat_on_cortex`` finds can land behind an air pocket
        (a lamina beyond a gap under the facet) that the drill can never
        actually reach.

        So each candidate head is tried in order of preference -- the
        re-seated head first (only when :func:`dorsal_approach_clear` confirms
        the approach behind it is clear of the vertebra), then the original
        head -- each with its tip extended to :meth:`_longest_length_from`'s
        longest fitting catalogue length, and the first one graded no worse
        than the baseline (``grade`` first -- a B-for-A swap is a worse
        screw even at a smaller breach -- then lexicographic
        ``(medial_breach_mm, breach_mm)``, no tolerance) is returned.  If
        neither is as safe as the baseline, the validated screw is returned
        unchanged: a legacy screw is never made less safe in order to make it
        longer or move its head.
        """
        from .trajectory_optimizer import dorsal_approach_clear, seat_on_cortex

        entry = np.asarray(entry, dtype=np.float64)
        target = np.asarray(target, dtype=np.float64)
        axis = target - entry
        norm = float(np.linalg.norm(axis))
        if norm <= 1e-9:
            return entry, target
        direction = axis / norm
        label = int(vertebra_label)

        base_grade, _ = self._evaluate_gertzbein_grade(
            entry, target, diameter, label
        )
        base_key = self._medial_breach_key(entry, target, diameter, label, side)

        heads, _travel = seat_on_cortex(
            self._grader, entry[None, :], direction[None, :], label
        )
        reseated_clear = bool(
            dorsal_approach_clear(self._grader, heads, direction[None, :], label)[0]
        )

        candidate_heads = []
        if reseated_clear:
            candidate_heads.append(heads[0])
        candidate_heads.append(entry)

        for head in candidate_heads:
            length = self._longest_length_from(head, direction, label)
            if length is None:
                continue
            tip = head + direction * length
            tip_grade, _ = self._evaluate_gertzbein_grade(
                head, tip, diameter, label
            )
            if _GRADE_ORDER.get(tip_grade, 99) > _GRADE_ORDER.get(base_grade, 99):
                continue
            if self._medial_breach_key(head, tip, diameter, label, side) <= base_key:
                return head, tip

        return entry, target

    def _least_medial_entry(
        self,
        side: str,
        entry: np.ndarray,
        target: np.ndarray,
        oriented_axis: np.ndarray,
        pedicle_center: np.ndarray,
        body_center: np.ndarray,
        vertebra_label: int,
        diameter: float,
        endplate_normal: Optional[np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """The narrow policy's lateral slide: ``(entry, target, shift_mm)``.

        Grades the trajectory at each of :attr:`NARROW_LATERAL_SHIFTS_MM` and
        keeps the one with the smallest ``medial_breach_mm``, breaking ties on
        the smallest total breach so a shift is only taken when it buys
        something.  The unshifted entry is graded first and wins every tie,
        which makes this a no-op on a side whose axis already clears the canal.

        Each shift moves the *pedicle centre*, not the already-seated entry,
        and re-seats it on the posterior cortex with :meth:`_find_entry_point`.
        Translating the seated entry directly would slide it along a fixed
        plane, which buries it under bone on a sloped lamina; re-seating keeps
        every candidate on the surface.  All shifts are evaluated -- there is
        no early exit -- since the cortex is not assumed convex, so a nearer
        shift finding no entry does not mean a farther one won't.
        """
        best_key = self._medial_breach_key(entry, target, diameter, vertebra_label, side)
        best = (entry, target, 0.0)
        lateral = self._lateral_direction(oriented_axis, side)
        if lateral is None or best_key == (0.0, 0.0):
            return best

        for shift in self.NARROW_LATERAL_SHIFTS_MM:
            if shift <= 0.0:
                continue
            shifted_center = pedicle_center + lateral * float(shift)
            shifted_entry = self._find_entry_point(
                shifted_center, oriented_axis, vertebra_label
            )
            if shifted_entry is None:
                # No posterior cortex under this shift; try the next one
                # rather than assuming farther shifts fare no better.
                continue
            shifted_target = self._find_best_target(
                shifted_entry,
                oriented_axis,
                body_center,
                vertebra_label,
                diameter,
                endplate_normal,
            )
            if shifted_target is None:
                continue
            key = self._medial_breach_key(
                shifted_entry, shifted_target, diameter, vertebra_label, side
            )
            if key < best_key:
                best_key = key
                best = (shifted_entry, shifted_target, float(shift))
        return best

    def _medial_breach_key(
        self,
        entry: np.ndarray,
        target: np.ndarray,
        diameter: float,
        vertebra_label: int,
        side: str,
    ) -> Tuple[float, float]:
        """``(medial_breach_mm, breach_mm)``, the narrow slide's ranking key.

        Graded with ``side=`` so the medial half is measured toward the canal
        rather than as an undirected minimum.  An ungradeable trajectory is
        ranked as the worst possible rather than as a clean one.
        """
        result = self._grader.grade(
            entry, target, diameter, label=int(vertebra_label), side=side,
            entry_zone_mm=ENTRY_ZONE_MM,
        )
        if result is None:
            margin = float(self._grader.crop_margin_mm)
            return (margin, margin)
        return (float(result.medial_breach_mm), float(result.breach_mm))

    @staticmethod
    def _lateral_direction(
        oriented_axis: np.ndarray,
        side: str,
    ) -> Optional[np.ndarray]:
        """Unit vector perpendicular to the pedicle axis in the axial plane.

        Oriented away from the midline -- ``+X`` for a left pedicle, ``-X`` for
        a right one, LPS -- so a positive step along it is always the
        in-out-in direction.  ``None`` when no such direction exists: a purely
        craniocaudal axis has no axial projection, and an axis that already
        runs mediolaterally has an axial perpendicular with no ``X`` component
        at all, which is not a lateral slide by any reading.
        """
        axial = np.array(
            [float(oriented_axis[0]), float(oriented_axis[1]), 0.0], dtype=np.float64
        )
        norm = float(np.linalg.norm(axial))
        if norm <= 1e-9:
            return None
        lateral = np.cross(np.array([0.0, 0.0, 1.0]), axial / norm)
        norm = float(np.linalg.norm(lateral))
        if norm <= 1e-9:
            return None
        lateral = lateral / norm
        if abs(float(lateral[0])) <= 1e-9:
            return None
        outward = 1.0 if side == "left" else -1.0
        if outward * float(lateral[0]) < 0.0:
            lateral = -lateral
        return lateral

    def _finalise_screw(
        self,
        vertebra: Vertebra,
        side: str,
        entry: np.ndarray,
        target: np.ndarray,
        diameter: float,
        pedicle_center: np.ndarray,
        body_center: np.ndarray,
        extra_warnings: Optional[List[str]] = None,
        pedicle_width: float = 0.0,
        upper_endplate_normal: Optional[np.ndarray] = None,
        narrow: bool = False,
        width_uncertain: bool = False,
        endplate_reference: str = "",
        endplate_reference_levels: Sequence[str] = (),
        entry_zone_mm: float = ENTRY_ZONE_MM,
    ) -> PlannedScrew:
        """Grade an accepted trajectory and wrap it in a :class:`PlannedScrew`.

        Shared tail of both planning modes, so an optimiser-chosen trajectory
        carries exactly the same grade, HU statistics, bone-quality metrics,
        facet/Heary classification, angles and warnings as a legacy one.
        ``extra_warnings`` are the messages the caller accumulated while
        constructing the trajectory; they are copied, never mutated.

        ``entry_zone_mm`` is the cortical entry length the grade excuses (see
        :data:`~src.core.screw_grading.ENTRY_ZONE_MM`): the traditional
        default for a head seated on the dorsal cortex.  A CBT head sits
        inferomedial on the pars/lamina, where the whole shaft is purchase,
        so :mod:`.cbt_planner` passes 0.  The value travels out in
        ``metrics["entry_zone_mm"]`` beside ``narrow``/``width_uncertain``,
        which are likewise forwarded untouched, so a reader can tell which
        rule graded the screw.

        ``upper_endplate_normal`` is the normal the trajectory was actually
        aimed and should be measured against -- the caller resolves that via
        :func:`~src.core.vertebra.aiming_endplate_normal` before calling this.
        It is passed even when ``config.endplate_parallel`` is off: the
        endplate angle is a measurement of the trajectory that was chosen, not
        a record of the setting that chose it, so the inspector shows it
        either way.  It is ``None`` exactly when ``endplate_reference`` is
        ``"none"`` (or empty/unresolved with no own fit), in which case
        ``endplate_angle_deg`` is deliberately absent from the metrics below.

        ``narrow`` and ``width_uncertain`` both mean "this side was planned
        under the narrow policy", and both mark it for review, but only
        ``narrow`` licenses quoting the width as a finding about the patient
        (see :meth:`_is_width_uncertain`).

        ``endplate_reference`` and ``endplate_reference_levels`` are recorded
        on the metrics verbatim so the plan table, CSV and JSON round-trip
        agree with the sagittal view about what the screw was measured
        against.
        """
        warnings: List[str] = list(extra_warnings or [])
        length = float(np.linalg.norm(target - entry))

        # 6. Grade the accepted trajectory once: containment plus HU statistics.
        #    From the cortex the head sits on, like every whole-screw grade, so
        #    this and the screw tool's re-grade of the same screw agree.
        result = self._grader.grade(
            entry, target, diameter, label=vertebra.label, side=side,
            entry_zone_mm=entry_zone_mm,
        )
        if result is None:
            grade = "E"
            breach_dist = float(self._grader.crop_margin_mm)
            min_wall = 0.0
            mean_hu = min_hu = 0.0
            medial_breach = lateral_breach = craniocaudal_breach = breach_dist
            medial_wall = 0.0
        else:
            grade = result.grade
            breach_dist = result.breach_mm
            min_wall = result.min_wall_mm
            mean_hu = result.mean_hu if result.mean_hu is not None else 0.0
            min_hu = result.min_hu if result.min_hu is not None else 0.0
            medial_breach = float(result.medial_breach_mm)
            lateral_breach = float(result.lateral_breach_mm)
            craniocaudal_breach = float(result.craniocaudal_breach_mm)
            medial_wall = float(result.medial_wall_mm)

        # 7. Clinical metrics: bone quality, breach direction, facet violation.
        quality = assess_bone_quality(
            self._grader,
            entry,
            target,
            diameter,
            vertebra.label,
            body_center_lps=body_center,
            isthmus_center_lps=pedicle_center,
            trajectory_threshold=self.config.trajectory_hu_threshold,
            volume_mm3=float(vertebra.volume_mm3)
            if vertebra.volume_mm3 > 0.0
            else None,
        )
        facet_grade, facet_text = facet_violation_grade(
            self._grader, entry, target, diameter, vertebra.label
        )
        if result is not None and result.breach_point_lps is not None:
            from .breach_classification import heary_directions

            heary = heary_direction(
                result.breach_point_lps, result.breach_centre_lps, side
            )
            secondary = heary_directions(
                result.breach_point_lps, result.breach_centre_lps, side
            )
            heary_secondary = (
                secondary[1]
                if len(secondary) > 1 and secondary[1] != heary
                else None
            )
        else:
            heary = "none"
            heary_secondary = None
        metrics: Dict[str, Any] = {
            "trajectory_mean_hu": quality.trajectory_mean_hu,
            "trajectory_min_hu": quality.trajectory_min_hu,
            "pedicle_mean_hu": quality.pedicle_mean_hu,
            "body_mean_hu": quality.body_mean_hu,
            "trajectory_body_ratio": quality.trajectory_body_ratio,
            "min_wall_mm": min_wall,
            "medial_breach_mm": medial_breach,
            "lateral_breach_mm": lateral_breach,
            "craniocaudal_breach_mm": craniocaudal_breach,
            "medial_wall_mm": medial_wall,
            "pedicle_width_mm": float(pedicle_width),
            "narrow_pedicle": bool(narrow),
            "width_uncertain": bool(width_uncertain),
            "heary_direction": heary,
            "heary_secondary": heary_secondary,
            "facet_grade": facet_grade,
            "facet_text": facet_text,
            "entry_zone_mm": float(entry_zone_mm),
            # Overwritten by :mod:`.cbt_planner`, which shares this tail.
            "trajectory_type": "traditional",
        }
        endplate_angle = endplate_angle_deg(entry, target, upper_endplate_normal)
        if endplate_angle is not None:
            # Absent, not 0.0, when the endplate could not be fitted: a missing
            # measurement must never read as a perfectly parallel screw.
            metrics["endplate_angle_deg"] = float(endplate_angle)
        if endplate_reference:
            metrics["endplate_reference"] = endplate_reference
        if endplate_reference == ENDPLATE_REFERENCE_NEIGHBOURS:
            metrics["endplate_reference_levels"] = ", ".join(endplate_reference_levels)
        if width_uncertain:
            # First in the list, for the same reason the narrow note is: it
            # explains why this screw looks the way it does.  It replaces the
            # narrow note rather than joining it -- a width the analyser
            # rejected may not be quoted back as a millimetre finding, even
            # when it happens to fall below the narrow threshold.
            if WIDTH_UNCERTAIN_SCREW_WARNING in warnings:
                warnings.remove(WIDTH_UNCERTAIN_SCREW_WARNING)
            warnings.insert(0, WIDTH_UNCERTAIN_SCREW_WARNING)
        elif narrow:
            # First in the list: it is the reason this screw looks the way it
            # does, and the cockpit reads the block top down.
            warnings.insert(0, narrow_pedicle_warning(pedicle_width, diameter))
        warnings.extend(quality.warnings)
        if facet_grade >= 2:
            warnings.append(f"Facet violation grade {facet_grade}: {facet_text}")

        # 8. Calculate angles.
        convergence_angle = self._compute_convergence_angle(entry, target, side)
        craniocaudal_angle = self._compute_craniocaudal_angle(entry, target)

        if convergence_angle > 30.0:
            warnings.append(
                f"High convergence angle {convergence_angle:.1f}° — verify on CT"
            )

        # 9. Calculate confidence score.
        confidence = self._calculate_confidence(grade, mean_hu, pedicle_width, diameter)

        if breach_dist > 0:
            warnings.append(f"Breach distance {breach_dist:.1f} mm (grade {grade})")

        if medial_breach > 0.0:
            warnings.append(medial_breach_warning(medial_breach))

        if 0 < min_wall < self.config.wall_clearance_mm:
            warnings.append(
                f"Cortical clearance {min_wall:.1f} mm below "
                f"{self.config.wall_clearance_mm:.0f} mm"
            )

        return PlannedScrew(
            vertebra_name=vertebra.name,
            side=side,
            entry_lps=entry,
            target_lps=target,
            length_mm=length,
            diameter_mm=diameter,
            convergence_angle=convergence_angle,
            craniocaudal_angle=craniocaudal_angle,
            mean_bone_density=mean_hu,
            min_bone_density=min_hu,
            gertzbein_grade=grade,
            confidence=confidence,
            breach_mm=breach_dist,
            min_wall_mm=min_wall,
            warnings=warnings,
            metrics=metrics,
        )

    def plan_all(
        self,
        analyses: List[PedicleAnalysisResult],
        sides: str = "both",
        progress: Optional[Callable[[str], None]] = None,
        cancel: Optional[Callable[[], bool]] = None,
        endplate_context: Sequence[PedicleAnalysisResult] = (),
    ) -> List[PlannedScrew]:
        """Plan screws for all analyzed vertebrae.

        Parameters
        ----------
        analyses:
            List of pedicle analysis results.
        sides:
            ``"both"`` (default), ``"left"``, or ``"right"``.
        progress:
            Called with ``"Planning L4 left (3/10)…"`` as each ``(level, side)``
            starts.  Grading one pedicle takes seconds, so a multi-level
            construct is otherwise a long silence.
        cancel:
            Polled at the top of each ``(level, side)``.  When it returns
            ``True`` the run stops there and returns the screws planned so far,
            with :attr:`last_run_cancelled` set.  Never interrupts a pedicle
            part-way: a half-graded trajectory is not a screw.
        endplate_context:
            Analyses of levels the surgeon did not select for planning, but
            close enough to donate a trusted endplate normal to a rough
            selected level (see
            :func:`~src.core.pedicle_analyzer.endplate_context_labels`).
            Resolved alongside ``analyses`` and never planned themselves.

        ``config.trajectory == "cbt"`` replaces the trajectory family for both
        back-ends: a cortical bone trajectory has its own entry landmark, angle
        window and implant catalogue, so neither the optimiser's convergent
        candidate grid nor the legacy planner's medialising search applies.

        Every path leaves :attr:`skipped_sides` holding one
        ``(level, side, reason)`` entry per side that ended up with no screw, so
        a half-planned level is reported rather than silently short.
        """
        if sides not in ("both", "left", "right"):
            raise ValueError(f"sides must be 'both', 'left', or 'right', got {sides!r}")

        # Resolved before anything else sees these analyses, so the legacy
        # planner, the optimiser and the CBT branch all aim and measure
        # against the same reference -- the context levels never get a screw,
        # but they do get to donate their fit to a selected rough neighbour.
        resolve_endplate_references(list(analyses) + list(endplate_context))

        side_list = ["left", "right"] if sides == "both" else [sides]
        self.last_run_cancelled = False
        self.skipped_sides = []

        if self.config.trajectory == "cbt":
            planned_levels = sum(
                1 for a in analyses if a.vertebra.label != _SACRUM_LABEL
            )
            reporter = RunProgress(
                total=planned_levels * len(side_list),
                progress=progress,
                cancel=cancel,
            )
            skipped: List[Tuple[str, str, str]] = []
            screws = plan_cbt_screws(
                self._grader,
                analyses,
                side_list,
                self.config,
                skip_labels=(_SACRUM_LABEL,),
                planner=self,
                progress=reporter,
                skipped=skipped,
            )
            self.skipped_sides = skipped
            self.last_run_cancelled = reporter.cancelled
            self._stamp_construct_alignment(screws)
            return screws

        reporter = RunProgress(
            total=len(analyses) * len(side_list),
            progress=progress,
            cancel=cancel,
        )
        if self.config.mode == "optimizer":
            results = self._plan_all_optimized(analyses, side_list, reporter)
        else:
            results = []
            for analysis in analyses:
                for side in side_list:
                    if not reporter.start(analysis.vertebra.name, side):
                        break
                    planned, reason = self._plan_screw(analysis, side)
                    if planned is not None:
                        results.append(planned)
                    else:
                        self._record_skip(analysis.vertebra.name, side, reason)
                if reporter.cancelled:
                    break

        self.last_run_cancelled = reporter.cancelled
        return results

    # =====================================================================
    # Optimiser-backed planning
    # =====================================================================

    def _plan_all_optimized(
        self,
        analyses: List[PedicleAnalysisResult],
        side_list: List[str],
        reporter: RunProgress,
    ) -> List[PlannedScrew]:
        """Plan every screw through the candidate optimiser.

        Each pedicle contributes its best :data:`_CONSTRUCT_TOP_K` trajectories;
        :func:`~src.core.trajectory_optimizer.optimize_construct` then picks one
        per screw so that each side's heads line up *and* its convergence angles
        agree with their neighbours.  A pedicle the optimiser cannot solve falls
        back to :meth:`plan_screw`; one neither can place is appended to
        :attr:`skipped_sides` with the legacy planner's reason.

        A construct choice that still collides is itself demoted to the legacy
        fallback: the fallback plans independently of the construct, so it is
        the best clean alternative available.  When the fallback also
        collides, the construct screw is kept with its collision warning and
        the side is additionally recorded in :attr:`skipped_sides` with
        :data:`~src.core.trajectory_optimizer.COLLIDING_SCREWS_SKIP_REASON`,
        so a colliding pair is warned *and* reported rather than silently
        kept.  When the fallback cannot plan at all, its own reason is
        recorded instead.  Demotion runs in visit order, so a demoted screw
        is checked against the screws already placed, not against a
        construct choice that may itself be demoted later.

        ``reporter`` narrates and cancels the candidate search — the pass that
        actually costs the time.  A cancelled run assembles a construct from the
        pedicles it reached, and never plans one it never announced.
        """
        per_screw: Dict[Tuple[str, str], List[Candidate]] = {}
        keys: Dict[Tuple[int, str], Tuple[str, str]] = {}
        construct_levels: Dict[Tuple[str, str], int] = {}
        visited: List[Tuple[PedicleAnalysisResult, str]] = []

        # `optimize_screw` otherwise builds itself an AutoScrewPlanner per
        # pedicle, each copying the whole CT and mask.  Ours is interchangeable
        # with that one exactly when the grader holds our own volumes -- which
        # it does not when the caller supplied a grader over a different pair.
        reusable = (
            self
            if self._grader.mask_image() is self._mask
            and self._grader.ct_image() is self._ct
            else None
        )

        for analysis in analyses:
            vertebra = analysis.vertebra
            for side in side_list:
                if not reporter.start(vertebra.name, side):
                    break
                visited.append((analysis, side))
                if vertebra.label == _SACRUM_LABEL or not analysis.success:
                    continue    # the legacy fallback rejects these anyway
                key = self._construct_key(vertebra, side, per_screw)
                try:
                    candidates = optimize_screw(
                        self._grader,
                        analysis,
                        side,
                        self.config,
                        self.config.weights,
                        top_k=_CONSTRUCT_TOP_K,
                        planner=reusable,
                        narrow=self._is_narrow_side(analysis, side),
                    )
                except Exception:   # pragma: no cover - defensive
                    logger.exception(
                        "Optimizer failed for %s %s; falling back to the legacy planner",
                        vertebra.name,
                        side,
                    )
                    continue
                if candidates:
                    per_screw[key] = candidates
                    keys[(vertebra.label, side)] = key
                    construct_levels[key] = int(vertebra.label)
            if reporter.cancelled:
                break

        chosen = (
            optimize_construct(
                per_screw, self.config.weights, levels=construct_levels,
                warn_on_collision=True,
            )
            if per_screw
            else {}
        )

        results: List[PlannedScrew] = []
        for analysis, side in visited:
            key = keys.get((analysis.vertebra.label, side))
            candidate = chosen.get(key) if key is not None else None
            if candidate is not None and self._candidate_collides(
                candidate, chosen, key, construct_levels
            ):
                # The construct could not separate this pair: demote to the
                # legacy fallback, which plans independently of the
                # construct.  The fallback's own warnings travel with it, so
                # the demotion is visible.
                fallback, fallback_reason = self._legacy_fallback(analysis, side)
                if fallback is not None and not self._screw_collides(
                    fallback, results, analysis.vertebra.label, side
                ):
                    results.append(fallback)
                    continue
                if fallback is None:
                    self._record_skip(analysis.vertebra.name, side, fallback_reason)
                    continue
                # Even the fallback collides: keep the construct screw with
                # its warning, and record the side as well so the pair is
                # reported, not silently kept.
                self._record_skip(
                    analysis.vertebra.name, side, COLLIDING_SCREWS_SKIP_REASON
                )
            planned, reason = (
                self._screw_from_candidate(analysis, side, candidate)
                if candidate is not None
                else self._legacy_fallback(analysis, side)
            )
            if planned is not None:
                results.append(planned)
            else:
                self._record_skip(analysis.vertebra.name, side, reason)

        self._stamp_construct_alignment(results)
        return results

    @staticmethod
    def _candidate_collides(
        candidate: Candidate,
        chosen: Dict[Tuple[str, str], Candidate],
        key: Tuple[str, str],
        levels: Dict[Tuple[str, str], int],
    ) -> bool:
        """Whether ``candidate`` collides with any other chosen construct screw."""
        from .trajectory_optimizer import _adjacent_levels, screws_collide

        for other, current in chosen.items():
            if other == key:
                continue
            if not _adjacent_levels(levels.get(key), levels.get(other)):
                continue
            if screws_collide(candidate, current):
                return True
        return False

    def _screw_collides(
        self,
        screw: PlannedScrew,
        results: List[PlannedScrew],
        label: int,
        side: str,
    ) -> bool:
        """Whether ``screw`` collides with an already-planned adjacent screw."""
        from .trajectory_optimizer import Candidate as _Candidate
        from .trajectory_optimizer import _adjacent_levels, screws_collide

        candidate = _Candidate(
            entry=np.asarray(screw.entry_lps, dtype=np.float64),
            target=np.asarray(screw.target_lps, dtype=np.float64),
            length=float(screw.length_mm),
            diameter=float(screw.diameter_mm),
            breach_mm=0.0,
            min_wall_mm=0.0,
            mean_hu=0.0,
            convergence_deg=float(screw.convergence_angle),
            craniocaudal_deg=float(screw.craniocaudal_angle),
            score=0.0,
        )
        for planned in results:
            if planned.side != side:
                continue
            other_label = _LEVEL_LABELS.get(planned.vertebra_name)
            if not _adjacent_levels(label, other_label):
                continue
            other = _Candidate(
                entry=np.asarray(planned.entry_lps, dtype=np.float64),
                target=np.asarray(planned.target_lps, dtype=np.float64),
                length=float(planned.length_mm),
                diameter=float(planned.diameter_mm),
                breach_mm=0.0,
                min_wall_mm=0.0,
                mean_hu=0.0,
                convergence_deg=float(planned.convergence_angle),
                craniocaudal_deg=float(planned.craniocaudal_angle),
                score=0.0,
            )
            if screws_collide(candidate, other):
                return True
        return False

    @staticmethod
    def _construct_key(
        vertebra: Vertebra,
        side: str,
        taken: Dict[Tuple[str, str], List[Candidate]],
    ) -> Tuple[str, str]:
        """A ``(level, side)`` construct key, disambiguated on name collisions."""
        key = (vertebra.name, side)
        if key in taken:
            key = (f"{vertebra.name}#{vertebra.label}", side)
        return key

    def _screw_from_candidate(
        self,
        analysis: PedicleAnalysisResult,
        side: str,
        candidate: Candidate,
    ) -> Tuple[Optional[PlannedScrew], Optional[str]]:
        """Convert a chosen :class:`Candidate` into a fully graded screw.

        Returns ``(screw, reason)`` like :meth:`_plan_screw`, so the caller can
        report a side that ends up with no screw at all.
        """
        vertebra = analysis.vertebra
        pedicle_center, _axis, pedicle_width = self._get_side_data(analysis, side)
        body_center = analysis.vertebral_body_center
        if pedicle_center is None or body_center is None:
            return self._legacy_fallback(analysis, side)

        warnings: List[str] = list(
            _endplate_warnings(analysis, self.config.endplate_parallel)
        )
        uncertain = self._is_width_uncertain(analysis, side)
        if uncertain:
            warnings.append(WIDTH_UNCERTAIN_SCREW_WARNING)
        narrow = self._is_narrow_side(analysis, side)
        recommended = self._compute_diameter(
            self._effective_width(analysis, side), vertebra.name
        )
        # A narrow side is planned at MIN_SCREW_DIAMETER by policy, not stepped
        # down for containment, so it must not claim it was.
        if not narrow and candidate.diameter < recommended - 1e-9:
            warnings.append(
                f"Diameter reduced from {recommended:.1f} to "
                f"{candidate.diameter:.1f} mm for cortical containment"
            )

        planned = self._finalise_screw(
            vertebra,
            side,
            np.asarray(candidate.entry, dtype=np.float64),
            np.asarray(candidate.target, dtype=np.float64),
            float(candidate.diameter),
            pedicle_center,
            body_center,
            warnings,
            pedicle_width,
            upper_endplate_normal=aiming_endplate_normal(analysis),
            narrow=narrow,
            width_uncertain=uncertain,
            endplate_reference=analysis.endplate_reference,
            endplate_reference_levels=analysis.endplate_reference_levels,
        )
        for message in candidate.warnings:
            # The optimiser explains how a trajectory was chosen; the planner
            # cannot re-derive those notes, and dropping them loses the only
            # record that a constraint was relaxed.  Worded identically to the
            # ones rebuilt above, so a repeat is a duplicate, not a new note.
            if message not in planned.warnings:
                planned.warnings.append(message)
        planned.metrics["score"] = float(candidate.score)
        planned.metrics["score_components"] = dict(candidate.components)
        return planned, None

    def _legacy_fallback(
        self,
        analysis: PedicleAnalysisResult,
        side: str,
    ) -> Tuple[Optional[PlannedScrew], Optional[str]]:
        """Plan one screw the legacy way and flag it as an optimiser fallback."""
        planned, reason = self._plan_screw(analysis, side)
        if planned is not None:
            planned.warnings.append(OPTIMIZER_FALLBACK_WARNING)
        return planned, reason

    def _record_skip(self, name: str, side: str, reason: Optional[str]) -> None:
        """Note a side that ended up with no screw, unless it was excluded by policy.

        ``reason`` is ``None`` for a deliberate exclusion (the sacrum), which is
        not a dropped side and must not be put in front of the surgeon as one.
        """
        if reason:
            self.skipped_sides.append((name, side, reason))

    @staticmethod
    def _stamp_construct_alignment(screws: List[PlannedScrew]) -> None:
        """Record each side's rod fit and convergence agreement on its screws.

        Shares its implementation with the re-stamp the screw tool runs after
        every edit (see :mod:`src.core.construct_alignment`): these numbers
        describe the set, so a dragged screw and a planned one have to be
        measured the same way or the cockpit would contradict the plan.
        """
        stamp_alignment(
            screws,
            side_of=lambda s: s.side,
            entry_of=lambda s: s.entry_lps,
            convergence_of=lambda s: s.convergence_angle,
            level_of=lambda s: _LEVEL_LABELS.get(s.vertebra_name),
            metrics_of=lambda s: s.metrics,
        )

    # =====================================================================
    # Entry / target point finding
    # =====================================================================

    def _find_entry_point(
        self,
        pedicle_center: np.ndarray,
        pedicle_axis_posterior: np.ndarray,
        vertebra_label: int,
    ) -> Optional[np.ndarray]:
        """Find entry point on posterior bone surface.

        Ray-cast from pedicle centre in BOTH directions along the
        pedicle axis to find the posterior bone-air boundary.
        Then back up ``ENTRY_BACKOFF`` mm inside bone for a safe entry.
        """
        step = self.TRAJECTORY_SAMPLE_STEP
        max_distance = 50.0  # mm -- safety limit
        n_steps = int(max_distance / step)

        # Try both posterior (+axis) and anterior (-axis) directions;
        # keep the one in the posterior direction (higher Y in LPS).
        best_entry = None
        best_y = -1e9

        for direction_sign in [1.0, -1.0]:
            direction = pedicle_axis_posterior * direction_sign
            last_bone_point = None
            in_bone = False

            for i in range(n_steps):
                distance = i * step
                point = pedicle_center + direction * distance
                if self._lps_to_index(point) is None:
                    break  # outside volume
                if self._is_point_inside_mask(point, vertebra_label):
                    last_bone_point = point.copy()
                    in_bone = True
                elif in_bone:
                    # Transition from bone to air — found the surface.
                    backoff_point = last_bone_point - direction * self.ENTRY_BACKOFF
                    if backoff_point[1] > best_y:
                        best_y = backoff_point[1]
                        best_entry = backoff_point
                    break
                # If not yet in bone, keep searching (skip air gaps)

        if best_entry is not None:
            return best_entry
        return None

    def _find_target_point(
        self,
        entry: np.ndarray,
        pedicle_axis_posterior: np.ndarray,
        body_center: np.ndarray,
        vertebra_label: int,
    ) -> Optional[np.ndarray]:
        """Find target point in anterior vertebral body.

        Follow the trajectory from entry toward the body centre
        (anteriorly).  Stop when exiting the vertebra mask or reaching
        the anterior cortex, then apply the anterior safety margin.
        """
        # Direction from entry to body centre.
        direction = body_center - entry
        dir_norm = np.linalg.norm(direction)
        if dir_norm < 1e-9:
            # Fallback: anterior direction is -Y in LPS.
            direction = np.array([0.0, -1.0, 0.0])
        else:
            direction = direction / dir_norm

        step = self.TRAJECTORY_SAMPLE_STEP
        max_distance = self.MAX_BONE_CORRIDOR_SCAN
        n_steps = int(max_distance / step)

        last_inside_point = entry.copy()

        for i in range(1, n_steps):
            distance = i * step
            point = entry + direction * distance
            idx = self._lps_to_index(point)
            if idx is None:
                # Outside volume bounds.
                break

            # Check if still inside the vertebra mask.
            z_idx, y_idx, x_idx = idx[2], idx[1], idx[0]
            if not self._is_in_bounds(x_idx, y_idx, z_idx):
                break

            mask_val = self._mask_array[z_idx, y_idx, x_idx]
            if mask_val != vertebra_label:
                # Exited the vertebra.
                break

            last_inside_point = point.copy()

        # Apply anterior safety margin by pulling back from the boundary.
        total_distance = float(np.linalg.norm(last_inside_point - entry))
        safe_distance = total_distance - self.ANTERIOR_SAFETY_MARGIN
        standard_length = self._select_standard_length(safe_distance)
        if standard_length is None:
            return None
        target = entry + direction * standard_length

        return target

    def _find_best_target(
        self,
        entry: np.ndarray,
        pedicle_axis_posterior: np.ndarray,
        body_center: np.ndarray,
        vertebra_label: int,
        diameter: float,
        upper_endplate_normal: Optional[np.ndarray] = None,
    ) -> Optional[np.ndarray]:
        """Search an ipsilateral body grid and retain the safest trajectory."""
        best_target: Optional[np.ndarray] = None
        best_score: Optional[Tuple[float, float, float]] = None
        side_sign = 1.0 if entry[0] >= body_center[0] else -1.0

        target_z = self._endplate_aligned_target_z(
            entry,
            float(body_center[1]),
            upper_endplate_normal,
        )
        for lateral_offset in self.TARGET_LATERAL_OFFSETS:
            candidate_anchor = np.array(
                [
                    body_center[0] + side_sign * lateral_offset,
                    body_center[1],
                    target_z,
                ],
                dtype=np.float64,
            )
            candidate = self._find_target_point(
                entry,
                pedicle_axis_posterior,
                candidate_anchor,
                vertebra_label,
            )
            if candidate is None:
                continue
            length = float(np.linalg.norm(candidate - entry))
            if not self.MIN_SCREW_LENGTH <= length <= self.MAX_SCREW_LENGTH:
                continue
            if side_sign * (candidate[0] - body_center[0]) < 0.0:
                continue
            convergence = self._compute_convergence_angle(
                entry,
                candidate,
                "left" if side_sign > 0.0 else "right",
            )
            if convergence > self.MAX_CONVERGENCE_ANGLE or convergence < self.config.min_convergence_deg:
                # Reject over-converging and laterally diverging candidates.
                continue
            _, breach_distance = self._evaluate_gertzbein_grade(
                entry,
                candidate,
                diameter,
                vertebra_label,
            )
            mean_hu, _, _ = self._sample_hu_along_trajectory(
                entry,
                candidate,
                diameter,
                label=vertebra_label,
            )
            score = (breach_distance, -mean_hu, -length)
            if best_score is None or score < best_score:
                best_score = score
                best_target = candidate

        return best_target

    # =====================================================================
    # HU sampling
    # =====================================================================

    def _sample_hu_along_trajectory(
        self,
        entry: np.ndarray,
        target: np.ndarray,
        diameter: float,
        label: Optional[int] = None,
    ) -> Tuple[float, float, List[float]]:
        """Sample HU statistics along the screw trajectory via the grader.

        ``label`` is auto-detected from the trajectory only when the caller
        does not already know which vertebra is being instrumented.

        Returns
        -------
        mean_hu : float
        min_hu : float
        all_samples : list of float
            Always empty; retained for signature compatibility.
        """
        if label is None:
            label = self._grader.detect_label(entry, target)
        result = self._grader.grade(
            entry, target, diameter, label=label, entry_zone_mm=ENTRY_ZONE_MM
        )
        if result is None or result.mean_hu is None:
            return 0.0, 0.0, []
        return result.mean_hu, result.min_hu, []

    # =====================================================================
    # Gertzbein-Robbins evaluation
    # =====================================================================

    def _evaluate_gertzbein_grade(
        self,
        entry: np.ndarray,
        target: np.ndarray,
        diameter: float,
        vertebra_label: int,
    ) -> Tuple[str, float]:
        """Evaluate the Gertzbein-Robbins grade via :class:`ScrewGrader`.

        Returns
        -------
        grade : str
            "A" through "E".
        breach_distance_mm : float
            Maximum distance of the screw envelope outside the vertebra mask.
        """
        result = self._grader.grade(
            entry, target, diameter, label=int(vertebra_label),
            entry_zone_mm=ENTRY_ZONE_MM,
        )
        if result is None:
            return "E", float(self._grader.crop_margin_mm)
        return result.grade, result.breach_mm

    # =====================================================================
    # Angle calculations
    # =====================================================================

    def _compute_convergence_angle(
        self,
        entry: np.ndarray,
        target: np.ndarray,
        side: str,
    ) -> float:
        """Signed medial convergence angle (positive = toward the midline)."""
        return convergence_angle_deg(entry, target, side)

    def _compute_craniocaudal_angle(
        self,
        entry: np.ndarray,
        target: np.ndarray,
    ) -> float:
        """Elevation of the trajectory above the axial plane, positive cranial.

        This is the module-wide convention shared with ``Screw.insertion_angle``
        and the inspector.  It differs from a sagittal-projection angle for
        converging screws: the lateral component counts toward the horizontal
        run, so a converging screw's elevation is smaller than its projection
        onto the YZ plane.
        """
        return craniocaudal_angle_deg(entry, target)

    # =====================================================================
    # Confidence scoring
    # =====================================================================

    def _calculate_confidence(
        self,
        grade: str,
        mean_hu: float,
        pedicle_width: float,
        diameter: float,
    ) -> float:
        """Calculate confidence score in range [0.0, 1.0].

        Factors:
        - Gertzbein grade (dominant): A=0.6, B=0.4, C=0.2, D=0.1, E=0.0
        - Mean HU bone density contribution: 0.0 -- 0.2
        - Pedicle width margin: 0.0 -- 0.2
        """
        grade_scores = {"A": 0.60, "B": 0.40, "C": 0.20, "D": 0.10, "E": 0.0}
        base = grade_scores.get(grade, 0.0)

        # HU contribution: 200--800 maps to 0.0--0.2.
        hu_range = self.CORTICAL_HU - self.BONE_HU_MIN
        hu_frac = max(0.0, min(1.0, (mean_hu - self.BONE_HU_MIN) / hu_range)) if hu_range > 0 else 0.0
        hu_score = hu_frac * 0.20

        # Width margin: ratio of (pedicle_width - diameter) / diameter.
        if diameter > 0 and pedicle_width > 0:
            margin_ratio = (pedicle_width - diameter) / diameter
            margin_score = max(0.0, min(1.0, margin_ratio / 0.5)) * 0.20
        else:
            margin_score = 0.0

        confidence = min(1.0, base + hu_score + margin_score)
        return round(confidence, 3)

    # =====================================================================
    # Coordinate / sampling helpers
    # =====================================================================

    def _lps_to_index(self, point_lps: np.ndarray) -> Optional[Tuple[int, int, int]]:
        """Convert LPS world coordinate to (x, y, z) image index.

        Returns ``None`` if the point is outside the image volume.
        """
        try:
            idx = self._ct.TransformPhysicalPointToIndex(
                [float(point_lps[0]), float(point_lps[1]), float(point_lps[2])]
            )
        except RuntimeError:
            return None

        size = self._ct.GetSize()  # (x, y, z)
        if not (0 <= idx[0] < size[0] and 0 <= idx[1] < size[1] and 0 <= idx[2] < size[2]):
            return None

        return (int(idx[0]), int(idx[1]), int(idx[2]))

    def _get_hu_at_lps(self, point_lps: np.ndarray) -> Optional[float]:
        """Get HU value at an LPS coordinate.

        Returns ``None`` if the point is outside the volume.
        """
        idx = self._lps_to_index(point_lps)
        if idx is None:
            return None

        x, y, z = idx
        return float(self._ct_array[z, y, x])

    def _is_in_bounds(self, x: int, y: int, z: int) -> bool:
        """Check whether array index (x, y, z) is within the mask volume."""
        shape = self._mask_array.shape  # (Z, Y, X)
        return 0 <= z < shape[0] and 0 <= y < shape[1] and 0 <= x < shape[2]

    def _is_point_inside_mask(
        self,
        point_lps: np.ndarray,
        vertebra_label: int,
    ) -> bool:
        """Check if an LPS point is inside the specified vertebra mask."""
        idx = self._lps_to_index(point_lps)
        if idx is None:
            return False

        x, y, z = idx
        if not self._is_in_bounds(x, y, z):
            return False

        return int(self._mask_array[z, y, x]) == vertebra_label

    # =====================================================================
    # Private helpers
    # =====================================================================

    def _select_standard_length(self, safe_length: float) -> Optional[float]:
        """Select the longest catalogue implant that fits the safe corridor."""
        candidates = [
            length for length in self.config.implant_lengths_mm if length <= safe_length + 1e-9
        ]
        if not candidates:
            return None
        return float(max(candidates))

    @staticmethod
    def _endplate_aligned_target_z(
        entry: np.ndarray,
        target_y: float,
        upper_endplate_normal: Optional[np.ndarray],
    ) -> float:
        """Return target Z parallel to the endplate in the sagittal YZ plane."""
        if upper_endplate_normal is None:
            return float(entry[2])
        normal = np.asarray(upper_endplate_normal, dtype=np.float64)
        if normal.shape != (3,) or abs(float(normal[2])) <= 1e-9:
            return float(entry[2])
        sagittal_slope = -float(normal[1]) / float(normal[2])
        return float(entry[2]) + sagittal_slope * (
            float(target_y) - float(entry[1])
        )

    @staticmethod
    def _get_side_data(
        analysis: PedicleAnalysisResult,
        side: str,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], float]:
        """Extract (pedicle_center, pedicle_axis, pedicle_width) for a side."""
        if side == "left":
            return (
                analysis.left_pedicle_center,
                analysis.left_pedicle_axis,
                analysis.left_pedicle_width,
            )
        else:
            return (
                analysis.right_pedicle_center,
                analysis.right_pedicle_axis,
                analysis.right_pedicle_width,
            )

    @staticmethod
    def _side_width_lower_bound(analysis: PedicleAnalysisResult, side: str) -> float:
        """The analyser's conservative second opinion for this side's width.

        ``0.0`` means the path never produced one -- hand-built analyses in
        tests, or a stand-in without the attribute -- so the caller falls back
        to the headline width rather than sizing from "no measurement".
        """
        bound = float(getattr(analysis, f"{side}_width_lower_bound_mm", 0.0) or 0.0)
        return bound if bound > 0.0 else 0.0

    def _effective_width(
        self,
        analysis: PedicleAnalysisResult,
        side: str,
    ) -> float:
        """The width the planner sizes this side from.

        The headline width is the larger of the analyser's two estimates
        (bounding-box extent vs. inscribed diameter); when they disagree by
        more than :attr:`PlannerConfig.width_bound_disagreement_mm` the
        measurement is cross-section dependent and the conservative bound is
        the honest input to the fill-ratio rule.  A zero bound is "no second
        opinion", not a 0 mm pedicle.
        """
        _center, _axis, width = self._get_side_data(analysis, side)
        width = float(width)
        bound = self._side_width_lower_bound(analysis, side)
        if bound > 0.0 and width - bound > self.config.width_bound_disagreement_mm:
            return bound
        return width

    @staticmethod
    def _is_width_uncertain(
        analysis: PedicleAnalysisResult,
        side: str,
    ) -> bool:
        """Whether the analyser refused to stand behind this side's width.

        Distinct from narrowness on purpose.  The plausibility gate flags a
        width *outside* its level's band in either direction, so a 24 mm L3 is
        flagged too -- and calling that "narrow" in the warning, the plan
        table and the cockpit produced the self-contradicting note "Narrow
        pedicle (24.0 mm): 4.0 mm screw is 17 % of the width".  The policy is
        the same for both (smallest screw, medial wall guarded, marked for
        review); only the words differ, and they have to be true.
        """
        return analysis.width_flags.get(side) == "implausible"

    def _is_narrow_side(
        self,
        analysis: PedicleAnalysisResult,
        side: str,
    ) -> bool:
        """Whether this side must be planned under the narrow-pedicle policy.

        Either the measurement is below :attr:`PlannerConfig.narrow_pedicle_mm`,
        or the analyser could not trust it at all -- in which case the smallest
        screw and the medial-wall guard are the safe assumption, not the level's
        diameter preset.  :meth:`_is_width_uncertain` separates the second case
        for everything the surgeon reads.

        "The measurement" is the conservative one: a side whose headline width
        clears the threshold but whose lower bound disagrees with it by more
        than :attr:`PlannerConfig.width_bound_disagreement_mm` is narrow, the
        same disagreement that makes :meth:`_effective_width` size from the
        bound.
        """
        _center, _axis, width = self._get_side_data(analysis, side)
        return (
            float(width) < self.config.narrow_pedicle_mm
            or self._effective_width(analysis, side) < self.config.narrow_pedicle_mm
            or self._is_width_uncertain(analysis, side)
        )

    def _compute_diameter(
        self,
        pedicle_width: float,
        vertebra_name: str,
    ) -> float:
        """Combine a level preset with the measured safe pedicle capacity.

        The corridor is limited both by the pedicle fill ratio and by a cortical
        clearance on each side.  A pedicle too small for even the narrowest
        implant still gets :attr:`MIN_SCREW_DIAMETER`: the policy is to place the
        smallest screw and mark the level, never to leave the side bare, so this
        never returns ``None`` and no caller has to handle "no diameter".
        """
        available = min(
            self.config.pedicle_fill_ratio * pedicle_width,
            pedicle_width - 2.0 * self.config.wall_clearance_mm,
        )
        catalogue = [
            d for d in self.config.implant_diameters_mm if d <= available + 1e-9
        ]
        if available < self.MIN_SCREW_DIAMETER or not catalogue:
            return self.MIN_SCREW_DIAMETER
        available = min(self.MAX_SCREW_DIAMETER, max(catalogue))
        preferred, automatic_maximum = self._diameter_profile_for_level(
            vertebra_name
        )
        recommendation = preferred
        if available >= preferred + self.DIAMETER_WIDE_HEADROOM:
            recommendation = min(preferred + 0.5, automatic_maximum)
        return min(available, recommendation)

    def _next_smaller_diameter(self, diameter: float) -> Optional[float]:
        """The largest catalogue diameter strictly below ``diameter``, or ``None``.

        The step-down walks ``config.implant_diameters_mm`` rather than a fixed
        0.5 mm decrement: a custom catalogue may be coarser or finer than the
        default, and a decrement off it lands on a size that cannot be ordered.
        A catalogue entry below :attr:`MIN_SCREW_DIAMETER` is still not offered.
        """
        smaller = [
            float(d)
            for d in self.config.implant_diameters_mm
            if d < diameter - 1e-9 and d >= self.MIN_SCREW_DIAMETER
        ]
        return max(smaller) if smaller else None

    @staticmethod
    def _diameter_profile_for_level(vertebra_name: str) -> Tuple[float, float]:
        level = str(vertebra_name).strip().upper()
        if level == "S1" or level in {"L3", "L4", "L5"}:
            return 6.5, 7.0
        if level in {"L1", "L2"}:
            return 6.0, 6.5
        if level.startswith("T") and level[1:].isdigit():
            return 5.5, 6.0
        return 6.0, 6.5

    @staticmethod
    def _orient_axis_posterior(pedicle_axis: np.ndarray) -> np.ndarray:
        """Ensure the pedicle axis points in the posterior (+Y) direction.

        If the axis Y-component is negative (points anteriorly), negate it.
        """
        if pedicle_axis[1] < 0:
            return -pedicle_axis
        return pedicle_axis.copy()
