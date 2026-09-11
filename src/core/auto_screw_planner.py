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
from .pedicle_analyzer import (
    ENDPLATE_FIT_RMSE_WARNING_MM,
    VERTEBRA_LABELS,
    endplate_fit_warning,
)
from .planner_config import PlannerConfig
from .screw_geometry import convergence_angle_deg, craniocaudal_angle_deg, endplate_angle_deg
from .screw_grading import ScrewGrader, resample_mask_to_ct
from .trajectory_optimizer import (
    Candidate,
    RunProgress,
    convergence_deviations_deg,
    convergence_spread_deg,
    optimize_construct,
    optimize_screw,
    rod_misalignment_mm,
)
from .vertebra import PedicleAnalysisResult, Vertebra

logger = logging.getLogger(__name__)

# Sacrum label -- not suitable for pedicle screw planning.
_SACRUM_LABEL = 25

#: Warning attached to a screw the optimiser could not solve.
OPTIMIZER_FALLBACK_WARNING = (
    "Optimizer found no feasible trajectory; legacy planner used"
)

#: Warning attached to a screw planned from a width the analyser flagged.
WIDTH_UNCERTAIN_SCREW_WARNING = "Pedicle width uncertain – verify diameter"

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
        # when the user switched the endplate-parallel option off, which is a
        # deliberate horizontal trajectory and therefore not worth a warning.
        # ``analysis.upper_endplate_normal`` stays the thing the screw is
        # *measured* against below.
        endplate_normal = (
            analysis.upper_endplate_normal if self.config.endplate_parallel else None
        )
        if self.config.endplate_parallel and analysis.upper_endplate_normal is None:
            warnings.append(
                "Upper endplate unavailable; used horizontal sagittal trajectory"
            )
        if (
            analysis.endplate_fit_rmse_mm is not None
            and analysis.endplate_fit_rmse_mm > ENDPLATE_FIT_RMSE_WARNING_MM
        ):
            warnings.append(endplate_fit_warning(analysis.endplate_fit_rmse_mm))

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
            else self._compute_diameter(pedicle_width, vertebra.name)
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
        while not narrow and grade not in {"A", "B"}:
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
            upper_endplate_normal=analysis.upper_endplate_normal,
            narrow=narrow,
            width_uncertain=uncertain,
        )
        return planned, None

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
            entry, target, diameter, label=int(vertebra_label), side=side
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
    ) -> PlannedScrew:
        """Grade an accepted trajectory and wrap it in a :class:`PlannedScrew`.

        Shared tail of both planning modes, so an optimiser-chosen trajectory
        carries exactly the same grade, HU statistics, bone-quality metrics,
        facet/Heary classification, angles and warnings as a legacy one.
        ``extra_warnings`` are the messages the caller accumulated while
        constructing the trajectory; they are copied, never mutated.

        ``upper_endplate_normal`` is the analysis's plane normal.  It is passed
        even when ``config.endplate_parallel`` is off: the endplate angle is a
        measurement of the trajectory that was chosen, not a record of the
        setting that chose it, so the inspector shows it either way.

        ``narrow`` and ``width_uncertain`` both mean "this side was planned
        under the narrow policy", and both mark it for review, but only
        ``narrow`` licenses quoting the width as a finding about the patient
        (see :meth:`_is_width_uncertain`).
        """
        warnings: List[str] = list(extra_warnings or [])
        length = float(np.linalg.norm(target - entry))

        # 6. Grade the accepted trajectory once: containment plus HU statistics.
        result = self._grader.grade(
            entry, target, diameter, label=vertebra.label, side=side
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
        )
        facet_grade, facet_text = facet_violation_grade(
            self._grader, entry, target, diameter, vertebra.label
        )
        if result is not None and result.breach_point_lps is not None:
            heary = heary_direction(
                result.breach_point_lps, result.breach_centre_lps, side
            )
        else:
            heary = "none"
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
            "facet_grade": facet_grade,
            "facet_text": facet_text,
            # Overwritten by :mod:`.cbt_planner`, which shares this tail.
            "trajectory_type": "traditional",
        }
        endplate_angle = endplate_angle_deg(entry, target, upper_endplate_normal)
        if endplate_angle is not None:
            # Absent, not 0.0, when the endplate could not be fitted: a missing
            # measurement must never read as a perfectly parallel screw.
            metrics["endplate_angle_deg"] = float(endplate_angle)
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
            self._stamp_rod_misalignment(screws)
            self._stamp_convergence_alignment(screws)
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
            optimize_construct(per_screw, self.config.weights, levels=construct_levels)
            if per_screw
            else {}
        )

        results: List[PlannedScrew] = []
        for analysis, side in visited:
            key = keys.get((analysis.vertebra.label, side))
            candidate = chosen.get(key) if key is not None else None
            planned, reason = (
                self._screw_from_candidate(analysis, side, candidate)
                if candidate is not None
                else self._legacy_fallback(analysis, side)
            )
            if planned is not None:
                results.append(planned)
            else:
                self._record_skip(analysis.vertebra.name, side, reason)

        self._stamp_rod_misalignment(results)
        self._stamp_convergence_alignment(results)
        return results

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

        warnings: List[str] = []
        if self.config.endplate_parallel and analysis.upper_endplate_normal is None:
            warnings.append(
                "Upper endplate unavailable; used horizontal sagittal trajectory"
            )
        if (
            analysis.endplate_fit_rmse_mm is not None
            and analysis.endplate_fit_rmse_mm > ENDPLATE_FIT_RMSE_WARNING_MM
        ):
            warnings.append(endplate_fit_warning(analysis.endplate_fit_rmse_mm))
        uncertain = self._is_width_uncertain(analysis, side)
        if uncertain:
            warnings.append(WIDTH_UNCERTAIN_SCREW_WARNING)
        narrow = self._is_narrow_side(analysis, side)
        recommended = self._compute_diameter(pedicle_width, vertebra.name)
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
            upper_endplate_normal=analysis.upper_endplate_normal,
            narrow=narrow,
            width_uncertain=uncertain,
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
    def _stamp_rod_misalignment(screws: List[PlannedScrew]) -> None:
        """Record each side's head-to-rod-line RMS deviation on its screws."""
        for side in ("left", "right"):
            on_side = [s for s in screws if s.side == side]
            if not on_side:
                continue
            deviation = rod_misalignment_mm(
                np.asarray([s.entry_lps for s in on_side], dtype=np.float64)
            )
            for screw in on_side:
                screw.metrics["rod_misalignment_mm"] = deviation

    @staticmethod
    def _stamp_convergence_alignment(screws: List[PlannedScrew]) -> None:
        """Record how far each screw's convergence sits from its side's agreement.

        Each side is measured on its own: the two rods are bent independently
        and a left-side outlier says nothing about the right.  A screw whose
        level is excluded from the term (S1) gets the side's spread but no
        deviation of its own, because it was never asked to agree.
        """
        for side in ("left", "right"):
            on_side = [s for s in screws if s.side == side]
            if not on_side:
                continue
            angles = [float(s.convergence_angle) for s in on_side]
            levels = [_LEVEL_LABELS.get(s.vertebra_name) for s in on_side]
            spread = convergence_spread_deg(angles, levels)
            deviations = convergence_deviations_deg(angles, levels)
            for screw, deviation in zip(on_side, deviations, strict=True):
                screw.metrics["convergence_spread_deg"] = spread
                if deviation is not None:
                    screw.metrics["convergence_deviation_deg"] = float(deviation)

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
        result = self._grader.grade(entry, target, diameter, label=label)
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
        result = self._grader.grade(entry, target, diameter, label=int(vertebra_label))
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
        """
        _center, _axis, width = self._get_side_data(analysis, side)
        return (
            float(width) < self.config.narrow_pedicle_mm
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
