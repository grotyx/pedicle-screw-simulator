"""
Screw Tool - Interactive screw placement tool

Implements two placement modes:
1. Entry-Target Mode: Click entry point, then click target point
2. Entry-Angle Mode: Click entry point, specify angles and length

Based on 3D Slicer Pedicle Screw Simulator algorithms.
"""

import math
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Mapping, Optional, Tuple

from src.models.screw import Screw

from ..core.screw_geometry import endplate_angle_deg
from ..utils.constants import (
    DEFAULT_SCREW_DIAMETER,
    DEFAULT_SCREW_LENGTH,
    GRADE_A_DESCRIPTION,
    GRADE_B_DESCRIPTION,
    GRADE_C_DESCRIPTION,
    GRADE_D_DESCRIPTION,
    GRADE_E_DESCRIPTION,
    MAX_SCREW_DIAMETER,
    MAX_SCREW_LENGTH,
    MIN_SCREW_DIAMETER,
    MIN_SCREW_LENGTH,
    TRAJECTORY_BODY_HU_RATIO_THRESHOLD,
)

#: Warning prefixes the grader owns: they are regenerated on every evaluation so
#: a re-graded screw can never keep a note that contradicts its current grade.
#: The texts must match :mod:`src.core.bone_quality`,
#: :mod:`src.core.breach_classification` (including its medial-breach note)
#: and :mod:`src.core.auto_screw_planner` verbatim, or a planner-authored note
#: will survive an edit that disproves it.  Everything else a screw carries — the
#: CBT contraindication note, the optimiser fallback note, the diameter
#: step-down note — describes how the screw was *chosen*, not how it grades,
#: and is left alone.
#:
#: ``"Vertebral body HU"`` is deliberately **not** here.  It describes the
#: vertebra, not the trajectory, and this tool cannot re-measure it (see
#: :data:`_PEDICLE_ANALYSIS_METRIC_KEYS`), so ``body_mean_hu`` is preserved
#: verbatim across a re-grade — the note that explains that number has to be
#: preserved with it, or the inspector shows an osteoporotic body HU with
#: nothing saying so.
_DERIVED_WARNING_PREFIXES = (
    "Not graded",
    "Breach distance",
    "Cortical clearance",
    "Trajectory HU",
    "Trajectory/body HU ratio",
    "Facet violation grade",
    "High convergence angle",
    "Medial breach",
)

#: Metrics that need the pedicle analysis (isthmus and vertebral-body centres)
#: that only the auto planner has.  ``_compute_metrics`` returns ``None`` for
#: them here, which means "not measured", not "measured as nothing", so a
#: re-grade keeps whatever the planner recorded instead of blanking a column
#: the inspector and the CSV export both read.
_PEDICLE_ANALYSIS_METRIC_KEYS = (
    "pedicle_mean_hu",
    "body_mean_hu",
    "trajectory_body_ratio",
    # Needs the level's upper-endplate plane, which only a pedicle analysis has.
    # Deliberately *not* in _GRADER_MEASURED_METRIC_KEYS: a mid-drag frame
    # outside the mask must not delete a planner measurement this tool may have
    # no way to reproduce (see _clear_grading).
    "endplate_angle_deg",
    # The isthmus width of the level the screw is *now* in, and the verdict
    # drawn from it.  Both need the analysis registry, and both are read by
    # something the surgeon looks at rather than reads -- the plan table's
    # Pedicle column, the cockpit's Pedicle row and the colour of the screw in
    # 3D and on the MPR planes -- so a screw dragged from a narrow level to a
    # wide one has to stop being red.  Also kept out of
    # _GRADER_MEASURED_METRIC_KEYS, for the same reason as the endplate angle.
    "pedicle_width_mm",
    "narrow_pedicle",
    "width_uncertain",
)

#: The metrics this tool measures from the trajectory in front of it, and so
#: the only ones an ungradable screw has to give up.  ``pedicle_mean_hu`` and
#: ``body_mean_hu`` are excluded for the same reason the merge protects them:
#: they came from a pedicle analysis this tool cannot repeat, so dropping them
#: would be permanent.  ``trajectory_body_ratio`` *is* dropped, because it is
#: derived from the ``trajectory_mean_hu`` that is going away; the merge
#: rebuilds it from the preserved body HU on the next gradable evaluation.
_GRADER_MEASURED_METRIC_KEYS = (
    "trajectory_mean_hu",
    "trajectory_min_hu",
    "trajectory_body_ratio",
    "min_wall_mm",
    "heary_direction",
    "facet_grade",
    "facet_text",
    "medial_breach_mm",
    "lateral_breach_mm",
    "craniocaudal_breach_mm",
    "medial_wall_mm",
)

#: Mirrors the planner's ``High convergence angle`` threshold
#: (:meth:`src.core.auto_screw_planner.AutoScrewPlanner._finalise_screw`), so a
#: manual and an auto screw are flagged at the same convergence.
_HIGH_CONVERGENCE_ANGLE_DEG = 30.0

#: Thresholds an untouched tool judges a screw against.  They mirror
#: :class:`~src.core.planner_config.PlannerConfig`'s own defaults rather than
#: importing it: constructing a ``PlannerConfig`` pulls in the optimiser (for
#: its default weights), which this module has no other reason to load.
#: ``tests/test_screw_tool.py`` asserts the two stay equal.  Whoever owns the
#: active config -- the planning controller after a run, the main window
#: whenever the Planning-parameters panel changes -- pushes the real values in
#: through :meth:`ScrewTool.set_wall_clearance_mm` and
#: :meth:`ScrewTool.set_narrow_pedicle_mm`, so a manual edit and an auto screw
#: are never judged against different numbers.
DEFAULT_WALL_CLEARANCE_MM = 0.0
DEFAULT_NARROW_PEDICLE_MM = 5.0

if TYPE_CHECKING:
    from ..core.screw_grading import GradeResult, ScrewGrader
    from ..core.volume_manager import VolumeManager


class ScrewTool:
    """
    Interactive tool for placing pedicle screws.

    Usage:
        tool = ScrewTool(volume_manager)
        tool.set_mode("entry_target")  # or "entry_angle"
        tool.on_click(x, y, z)  # Process click events
    """

    def __init__(self, volume_manager: "VolumeManager"):
        """
        Initialize screw tool.

        Args:
            volume_manager: Shared VolumeManager instance
        """
        self.volume_manager = volume_manager

        # Tool state
        self._mode = "entry_target"  # or "entry_angle"
        self._state = "waiting_entry"  # or "waiting_target"
        self._pending_entry: Optional[Tuple[float, float, float]] = None

        # Screw parameters
        self._default_length = DEFAULT_SCREW_LENGTH
        self._default_diameter = DEFAULT_SCREW_DIAMETER
        self._default_insertion_angle = 15.0  # degrees
        self._default_medial_angle = 10.0  # degrees

        # Placed screws
        self._screws: List[Screw] = []

        # Segmentation-based grader (None until a mask is available)
        self._grader: Optional["ScrewGrader"] = None

        # Pedicle analyses keyed by mask label, handed over by the planning
        # controller after a run.  The tool reads only ``upper_endplate_normal``
        # from them, so any object with that attribute works.
        self._analysis_by_level: Dict[int, Any] = {}

        # Thresholds the active PlannerConfig owns, pushed in by whoever holds
        # it.  Without them a manual edit and an auto screw would be judged
        # against different numbers -- the drag itself would then appear to
        # have changed the screw.
        self._wall_clearance_mm = float(DEFAULT_WALL_CLEARANCE_MM)
        self._narrow_pedicle_mm = float(DEFAULT_NARROW_PEDICLE_MM)

        # Callbacks
        self._on_screw_placed: Optional[Callable[[Screw], None]] = None
        self._on_state_changed: Optional[Callable[[str], None]] = None

    def set_mode(self, mode: str):
        """
        Set placement mode.

        Args:
            mode: "entry_target" or "entry_angle"
        """
        if mode not in ("entry_target", "entry_angle"):
            raise ValueError(f"Invalid mode: {mode}")

        self._mode = mode
        self._reset_state()

    def set_grader(self, grader: Optional["ScrewGrader"]) -> None:
        """Attach (or detach) the segmentation-based grader."""
        self._grader = grader

    @property
    def grader(self) -> Optional["ScrewGrader"]:
        return self._grader

    def set_analysis_by_level(self, analyses: Optional[Mapping[int, Any]]) -> None:
        """Register the pedicle analyses behind the current plan, by mask label.

        A manually placed screw has no analysis of its own, so without this the
        tool cannot re-measure anything that needs one.  ``None`` clears it (a
        new study's analyses do not describe the old study's vertebrae).
        """
        self._analysis_by_level = dict(analyses or {})

    def set_wall_clearance_mm(self, value: float) -> None:
        """Adopt the active plan's cortical clearance for the manual note.

        The planner only writes ``"Cortical clearance ..."`` when a screw's
        wall falls below :attr:`PlannerConfig.wall_clearance_mm`; at the
        shipped default of 0 mm it never does.  Reading a fixed constant here
        instead made the *first drag* of an auto screw invent a note the plan
        never carried -- a warning that appeared because the user nudged the
        screw, not because anything about it changed.
        """
        self._wall_clearance_mm = max(0.0, float(value))

    def set_narrow_pedicle_mm(self, value: float) -> None:
        """Adopt the active plan's narrow-pedicle threshold.

        Used when a re-grade re-reads the width of the level the screw is now
        in; the verdict has to be the one the planner would have reached for
        that width, or a drag between levels could recolour a screw by a rule
        the plan was never made under.
        """
        self._narrow_pedicle_mm = float(value)

    def _analysis_for(self, label: int) -> Optional[Any]:
        """The analysis for one mask label, from this registry or the grader's."""
        analysis = self._analysis_by_level.get(int(label))
        if analysis is not None:
            return analysis
        # The grader may carry an analysis cache instead; prefer this tool's own
        # registry, then fall back to whatever the grader offers.
        lookup = getattr(self._grader, "analysis_for", None)
        return lookup(int(label)) if callable(lookup) else None

    def _endplate_angle(self, screw: Screw, label: int) -> Optional[float]:
        """Signed angle to the level's upper endplate, or None if unknown."""
        analysis = self._analysis_for(label)
        normal = getattr(analysis, "upper_endplate_normal", None)
        if normal is None:
            return None
        return endplate_angle_deg(screw.entry_point, screw.target_point, normal)

    def _pedicle_width(
        self, screw: Screw, label: int
    ) -> Tuple[Optional[float], Optional[bool], Optional[bool]]:
        """This side's isthmus width at ``label``, its narrow verdict and its trust.

        ``(None, None, None)`` -- "not measured" -- whenever the level is not
        in the registry or the screw has no side, so :meth:`_merge_metrics`
        keeps whatever the planner recorded.  A drag *between* levels is the
        case that matters: without this the screw would keep the width of the
        vertebra it left, and stay red for a pedicle it is no longer in.

        The verdict mirrors
        :meth:`src.core.auto_screw_planner.AutoScrewPlanner._is_narrow_side`
        exactly, threshold included, or an edited screw and a planned one would
        disagree about the same pedicle.  So do the *preconditions*: the
        planner only ever reaches that verdict for a side it actually detected,
        and ``PedicleAnalysisResult`` defaults an undetected side's width to
        ``0.0``.  Reading that default back was the bug -- a manual screw on
        the side the analyser missed came out red, "0.0 mm", and carrying a
        warning that a 4.0 mm screw filled 100 % of the pedicle, none of which
        had ever been measured.  An analysis that failed, a side with no
        isthmus centre and a non-positive width are all "not measured".

        The registry is documented as accepting any object with the attributes
        this tool reads, so each check is skipped on an object that does not
        model it at all -- absent means "this stand-in has nothing to say about
        detection", while a present ``None`` (which the real
        :class:`~src.core.vertebra.PedicleAnalysisResult` always has for an
        undetected side) means "not found".
        """
        side = self._screw_side(screw)
        analysis = self._analysis_for(label)
        if side is None or analysis is None:
            return None, None, None
        if not getattr(analysis, "success", True):
            return None, None, None
        center_attr = f"{side}_pedicle_center"
        if hasattr(analysis, center_attr) and getattr(analysis, center_attr) is None:
            return None, None, None
        width = getattr(analysis, f"{side}_pedicle_width", None)
        if not isinstance(width, (int, float)) or isinstance(width, bool):
            return None, None, None
        width = float(width)
        if width <= 0.0:
            return None, None, None
        flags = getattr(analysis, "width_flags", None)
        uncertain = bool(
            isinstance(flags, Mapping) and flags.get(side) == "implausible"
        )
        return width, bool(width < self._narrow_pedicle_mm or uncertain), uncertain

    def set_screw_parameters(
        self,
        length: Optional[float] = None,
        diameter: Optional[float] = None,
        insertion_angle: Optional[float] = None,
        medial_angle: Optional[float] = None
    ):
        """Set default screw parameters."""
        if length is not None:
            self._default_length = max(MIN_SCREW_LENGTH,
                                       min(MAX_SCREW_LENGTH, length))
        if diameter is not None:
            self._default_diameter = max(MIN_SCREW_DIAMETER,
                                        min(MAX_SCREW_DIAMETER, diameter))
        if insertion_angle is not None:
            self._default_insertion_angle = insertion_angle
        if medial_angle is not None:
            self._default_medial_angle = medial_angle

    def set_callbacks(
        self,
        on_screw_placed: Optional[Callable[[Screw], None]] = None,
        on_state_changed: Optional[Callable[[str], None]] = None
    ):
        """Set callback functions."""
        self._on_screw_placed = on_screw_placed
        self._on_state_changed = on_state_changed

    def on_click(
        self,
        x: float,
        y: float,
        z: float,
        plane: Optional[str] = None
    ) -> Optional[Screw]:
        """
        Process a click event.

        Args:
            x, y, z: World coordinates of click
            plane: Source plane ('axial', 'sagittal', 'coronal', or None for 3D)

        Returns:
            Screw object if placement completed, None otherwise
        """
        if self._mode == "entry_target":
            return self._handle_entry_target_click(x, y, z, plane)
        else:
            return self._handle_entry_angle_click(x, y, z, plane)

    def _handle_entry_target_click(
        self,
        x: float, y: float, z: float,
        plane: Optional[str]
    ) -> Optional[Screw]:
        """Handle click in entry-target mode."""
        if self._state == "waiting_entry":
            # First click: set entry point
            self._pending_entry = (x, y, z)
            self._state = "waiting_target"
            self._notify_state_changed()
            return None

        else:  # waiting_target
            # Second click: set target and create screw
            entry = self._pending_entry
            target = (x, y, z)

            screw = self._create_screw(entry, target)

            self._reset_state()
            return screw

    def _handle_entry_angle_click(
        self,
        x: float, y: float, z: float,
        plane: Optional[str]
    ) -> Optional[Screw]:
        """Handle click in entry-angle mode."""
        if self._state != "waiting_entry":
            return None

        # Calculate target from entry point and angles
        entry = (x, y, z)
        target = self._calculate_target_from_angles(
            entry,
            self._default_length,
            self._default_insertion_angle,
            self._default_medial_angle
        )

        screw = self._create_screw(entry, target)
        return screw

    def _create_screw(
        self,
        entry: Tuple[float, float, float],
        target: Tuple[float, float, float]
    ) -> Screw:
        """Create a new screw and add to list."""
        screw = Screw(
            entry_point=entry,
            target_point=target,
            diameter=self._default_diameter
        )

        # Evaluate placement
        self._evaluate_screw(screw)

        # Add to list
        self._screws.append(screw)
        self._restamp_construct_alignment()

        # Notify callback
        if self._on_screw_placed:
            self._on_screw_placed(screw)

        return screw

    def _calculate_target_from_angles(
        self,
        entry: Tuple[float, float, float],
        length: float,
        insertion_angle: float,
        medial_angle: float
    ) -> Tuple[float, float, float]:
        """
        Calculate target point from entry point and angles, in LPS millimetres.

        Entry-angle mode is a legacy convenience mode: it assumes the volume
        midline sits near x = 0, so the entry point's sign decides which side
        "medial" points towards.

        Args:
            entry: Entry point coordinates (LPS mm)
            length: Screw length
            insertion_angle: Craniocaudal angle, cranial positive (degrees)
            medial_angle: Axial convergence towards the midline (degrees)

        Returns:
            Target point coordinates
        """
        ins_rad = math.radians(insertion_angle)   # cranial positive
        med_rad = math.radians(medial_angle)      # medial positive; entry assumed left of midline when x > 0
        lateral_sign = 1.0 if entry[0] >= 0.0 else -1.0
        dx = -lateral_sign * math.sin(med_rad) * math.cos(ins_rad)
        dy = -math.cos(med_rad) * math.cos(ins_rad)
        dz = math.sin(ins_rad)

        return (
            entry[0] + length * dx,
            entry[1] + length * dy,
            entry[2] + length * dz,
        )

    def _evaluate_screw(self, screw: Screw):
        """Grade with the segmentation mask; mark N/A when no mask exists.

        Every grader-derived field is rewritten from scratch — the measured
        metrics and every warning in :data:`_DERIVED_WARNING_PREFIXES` — so
        that re-grading an edited or restored screw can never leave a stale
        value (or a note contradicting the grade now displayed) behind.

        What the grader cannot re-measure is *merged*, not discarded.  This
        runs on every drag, every diameter change and every ``regrade_all``
        (which ``PlanController.load`` and the segmentation controller both
        trigger), so replacing the bundle wholesale used to erase the
        optimiser's ``score``/``score_components``, ``rod_misalignment_mm`` and
        the ``trajectory_type`` that is the only marker of a CBT screw — from
        every screw in the plan at once.
        """
        screw.warnings = [
            w for w in screw.warnings
            if not w.startswith(_DERIVED_WARNING_PREFIXES)
        ]
        if self._grader is None:
            self._clear_grading(screw)
            screw.warnings.append("Not graded: run segmentation first")
            return
        from ..core.screw_grading import ENTRY_ZONE_MM

        # From the cortex the head sits on, exactly as the planner grades the
        # screw it proposes -- otherwise the first drag of a planned screw
        # would re-grade it by a stricter rule and the grade would drop for a
        # screw that had not moved.
        result = self._grader.grade(
            screw.entry_point,
            screw.target_point,
            screw.diameter,
            side=self._screw_side(screw),
            entry_zone_mm=ENTRY_ZONE_MM,
        )
        if result is None:
            self._clear_grading(screw)
            screw.warnings.append(
                "Not graded: trajectory does not pass through a segmented vertebra"
            )
            return
        screw.grade = result.grade
        screw.breach_distance = result.breach_mm
        screw.mean_hu = result.mean_hu
        screw.min_hu = result.min_hu
        measured, derived_warnings = self._compute_metrics(screw, result)
        screw.metrics = self._merge_metrics(screw.metrics, measured)
        self._refresh_narrow_pedicle_warning(screw, measured)
        if not screw.vertebra_level:
            from ..core.pedicle_analyzer import VERTEBRA_LABELS
            screw.vertebra_level = VERTEBRA_LABELS.get(result.label, "")

        screw.warnings.extend(derived_warnings)
        screw.warnings.extend(self._ratio_warnings(screw.metrics, measured))
        if result.breach_mm > 0:
            screw.warnings.append(
                f"Breach distance {result.breach_mm:.1f} mm (grade {result.grade})"
            )
        # The active plan's clearance, not a fixed constant: the planner only
        # writes this note below its own `wall_clearance_mm`, so at the shipped
        # default of 0 mm there is no threshold to fall below and no note to
        # write.  Judging a dragged screw by a stricter number than the one
        # that planned it made the drag itself look like the problem.
        if 0 < result.min_wall_mm < self._wall_clearance_mm:
            screw.warnings.append(
                f"Cortical clearance {result.min_wall_mm:.1f} mm below "
                f"{self._wall_clearance_mm:.1f} mm"
            )

    @staticmethod
    def _refresh_narrow_pedicle_warning(
        screw: Screw, measured: Dict[str, Any]
    ) -> None:
        """Re-state the planner's narrow-pedicle note for the width just read.

        The note is planner-owned, so it is deliberately absent from
        :data:`_DERIVED_WARNING_PREFIXES` -- stripping it there would delete it
        on every re-grade of a screw whose level this tool cannot re-measure,
        and the tool could never put it back.  It is rewritten here instead,
        and only when :meth:`_pedicle_width` actually produced a verdict: the
        same condition under which ``pedicle_width_mm`` / ``narrow_pedicle``
        are refreshed, so the note and the number the cockpit shows can never
        disagree.  Without a verdict (no analysis for the level the screw is
        now in) both are left exactly as the planner wrote them.

        The diameter is the screw's current one, so a step-down that ends the
        narrowness retracts the note along with the flag.

        The "width uncertain" note is rewritten by the same rule and is
        mutually exclusive with the narrow one, exactly as in
        :meth:`~src.core.auto_screw_planner.AutoScrewPlanner._finalise_screw`:
        a width the analyser rejected may not be quoted back as millimetres.
        A screw dragged out of a flagged level has to lose the note with it.
        """
        from ..core.auto_screw_planner import (
            NARROW_PEDICLE_WARNING_PREFIX,
            WIDTH_UNCERTAIN_SCREW_WARNING,
            narrow_pedicle_warning,
        )

        width = measured.get("pedicle_width_mm")
        narrow = measured.get("narrow_pedicle")
        if width is None or narrow is None:
            return
        uncertain = bool(measured.get("width_uncertain"))
        screw.warnings = [
            warning
            for warning in screw.warnings
            if not warning.startswith(NARROW_PEDICLE_WARNING_PREFIX)
            and warning != WIDTH_UNCERTAIN_SCREW_WARNING
        ]
        if uncertain:
            screw.warnings.append(WIDTH_UNCERTAIN_SCREW_WARNING)
        elif narrow:
            screw.warnings.append(
                narrow_pedicle_warning(float(width), screw.diameter)
            )

    @staticmethod
    def _clear_grading(screw: Screw) -> None:
        """Drop what this trajectory disproves, keep what the plan still knows.

        ``ScrewEditController._apply_points`` re-evaluates on every frame of a
        drag, so a screw briefly pulled outside the mask and back passes
        through here mid-gesture.  Emptying the bundle would make that single
        frame permanent: the optimiser's ``score``/``score_components``, the
        construct's ``rod_misalignment_mm``, the ``trajectory_type`` marking a
        CBT screw and the pedicle-analysis HU would all be gone by the time the
        screw was back inside the vertebra, with nothing left to restore them.
        Only the metrics measured from the trajectory itself are cleared.
        """
        screw.grade = "N/A"
        screw.breach_distance = 0.0
        screw.mean_hu = None
        screw.min_hu = None
        metrics = screw.metrics if isinstance(screw.metrics, dict) else {}
        screw.metrics = {
            key: value
            for key, value in metrics.items()
            if key not in _GRADER_MEASURED_METRIC_KEYS
        }

    def _compute_metrics(
        self, screw: Screw, result: "GradeResult"
    ) -> Tuple[Dict[str, Any], List[str]]:
        """Bone-quality, breach-direction and facet metrics, and their warnings.

        A manually placed screw has no pedicle analysis behind it, so the
        isthmus and vertebral-body centres are unknown and the metrics that
        depend on them come back ``None``; :meth:`_merge_metrics` decides what
        that means for a screw that once had them.

        The warnings are returned alongside rather than pushed onto the screw
        so that both halves of one measurement are produced together — the
        clinical notes a reviewer reads next to a grade have to describe the
        same trajectory the grade does.

        The medial/lateral/craniocaudal split needs to know which side of the
        spine the screw is on.  A screw with no side leaves all four ``None`` —
        "not measured", which is what :meth:`_merge_metrics` writes through, so
        an auto screw that loses its side never keeps a stale canal number.

        ``endplate_angle_deg`` needs the level's endplate plane, which arrives
        through :meth:`set_analysis_by_level`; without it the value is ``None``
        ("not measured") and :meth:`_merge_metrics` keeps whatever the planner
        recorded.  ``pedicle_width_mm`` / ``narrow_pedicle`` come from the same
        registry and behave the same way, but for a louder reason: they decide
        the colour of the screw, so a drag that moves it to another level has
        to move the width with it.
        """
        from ..core.bone_quality import assess_bone_quality
        from ..core.breach_classification import facet_violation_grade, medial_breach_warning

        quality = assess_bone_quality(
            self._grader,
            screw.entry_point,
            screw.target_point,
            screw.diameter,
            result.label,
        )
        facet_grade, facet_text = facet_violation_grade(
            self._grader, screw.entry_point, screw.target_point, screw.diameter, result.label
        )

        side = self._screw_side(screw)
        directional: Dict[str, Any] = {
            "medial_breach_mm": None,
            "lateral_breach_mm": None,
            "craniocaudal_breach_mm": None,
            "medial_wall_mm": None,
        }
        if side is not None:
            directional = {
                "medial_breach_mm": float(result.medial_breach_mm),
                "lateral_breach_mm": float(result.lateral_breach_mm),
                "craniocaudal_breach_mm": float(result.craniocaudal_breach_mm),
                "medial_wall_mm": float(result.medial_wall_mm),
            }

        pedicle_width, narrow, width_uncertain = self._pedicle_width(
            screw, result.label
        )
        metrics = {
            "trajectory_mean_hu": quality.trajectory_mean_hu,
            "trajectory_min_hu": quality.trajectory_min_hu,
            "pedicle_mean_hu": quality.pedicle_mean_hu,
            "body_mean_hu": quality.body_mean_hu,
            "trajectory_body_ratio": quality.trajectory_body_ratio,
            "min_wall_mm": result.min_wall_mm,
            "endplate_angle_deg": self._endplate_angle(screw, result.label),
            "pedicle_width_mm": pedicle_width,
            "narrow_pedicle": narrow,
            "width_uncertain": width_uncertain,
            **directional,
            "heary_direction": self._heary_label(screw.side, result),
            "facet_grade": facet_grade,
            "facet_text": facet_text,
        }
        warnings = list(quality.warnings)
        if facet_grade >= 2:
            warnings.append(f"Facet violation grade {facet_grade}: {facet_text}")
        # ``Screw.medial_angle`` is the same signed convergence the planner
        # computes, so the same threshold flags the same trajectories.
        if screw.medial_angle > _HIGH_CONVERGENCE_ANGLE_DEG:
            warnings.append(
                f"High convergence angle {screw.medial_angle:.1f}° — verify on CT"
            )
        medial_breach = directional["medial_breach_mm"]
        if medial_breach is not None and medial_breach > 0.0:
            warnings.append(medial_breach_warning(medial_breach))
        return metrics, warnings

    @staticmethod
    def _ratio_warnings(merged: Dict[str, Any], measured: Dict[str, Any]) -> List[str]:
        """Re-raise the trajectory/body HU ratio note for a merged bundle.

        ``assess_bone_quality`` only emits this note when it measured the ratio
        itself, which needs a vertebral-body centre this tool never has — so
        for an edited auto screw it never fires.  The ratio is nonetheless
        rebuilt by :meth:`_merge_metrics` from the new trajectory HU against
        the preserved body HU, and the note is stripped as derived.  Without
        this the warning would be deleted on every drag and every plan load
        while the number it describes stayed on screen.  The text must match
        :mod:`src.core.bone_quality` verbatim.
        """
        if measured.get("trajectory_body_ratio") is not None:
            return []                    # bone_quality already spoke for itself
        ratio = merged.get("trajectory_body_ratio")
        if not isinstance(ratio, (int, float)) or isinstance(ratio, bool):
            return []
        if float(ratio) >= TRAJECTORY_BODY_HU_RATIO_THRESHOLD:
            return []
        return [
            f"Trajectory/body HU ratio {float(ratio):.2f} below "
            f"{TRAJECTORY_BODY_HU_RATIO_THRESHOLD:.1f}"
        ]

    @staticmethod
    def _merge_metrics(existing: Any, measured: Dict[str, Any]) -> Dict[str, Any]:
        """Overlay freshly measured metrics onto the bundle a screw already has.

        Keys the grader does not own are carried through untouched: the
        optimiser's ``score`` and ``score_components``, the construct-level
        ``rod_misalignment_mm``, and ``trajectory_type`` /
        ``cbt_cranial_angle_deg``.

        The :data:`_PEDICLE_ANALYSIS_METRIC_KEYS` need centres (or an endplate
        plane) this tool may not have, so a ``None`` from that cause is not
        allowed to overwrite a planner measurement.  ``trajectory_body_ratio``
        is the exception among them: it is purely derived, so it is recomputed
        from the new trajectory HU against the preserved body HU rather than
        left behind describing the old trajectory.
        """
        merged: Dict[str, Any] = dict(existing) if isinstance(existing, dict) else {}
        for key, value in measured.items():
            if (
                value is None
                and key in _PEDICLE_ANALYSIS_METRIC_KEYS
                and merged.get(key) is not None
            ):
                continue
            merged[key] = value

        if measured.get("trajectory_body_ratio") is None:
            trajectory_hu = merged.get("trajectory_mean_hu")
            body_hu = merged.get("body_mean_hu")
            if (
                isinstance(trajectory_hu, (int, float))
                and not isinstance(trajectory_hu, bool)
                and isinstance(body_hu, (int, float))
                and not isinstance(body_hu, bool)
                and float(body_hu) != 0.0
            ):
                merged["trajectory_body_ratio"] = float(trajectory_hu) / float(body_hu)
        return merged

    @staticmethod
    def _heary_label(side: str, result: "GradeResult") -> str:
        """Breach direction, degraded to ``"mediolateral"`` when the side is unknown.

        :func:`~src.core.breach_classification.heary_direction` needs to know
        which side of the midline the screw sits on to tell medial from lateral,
        and a manually placed screw often has no ``side`` yet. Rather than let
        the empty string be read as "right" — which silently flips the label a
        surgeon reads in the inspector and in the exported CSV — an x-dominant
        breach is reported side-agnostically. The other four directions do not
        depend on the side and are passed through unchanged.
        """
        from ..core.breach_classification import heary_direction

        if result.breach_point_lps is None:
            return "none"
        known_side = str(side).lower()
        if known_side in ("left", "right"):
            return heary_direction(
                result.breach_point_lps, result.breach_centre_lps, known_side
            )
        label = heary_direction(
            result.breach_point_lps, result.breach_centre_lps, "left"
        )
        return "mediolateral" if label in ("medial", "lateral") else label

    @staticmethod
    def _screw_side(screw: Screw) -> Optional[str]:
        """The screw's side as the grader wants it, or ``None`` when unknown.

        A manually placed screw often has no side yet, and the medial/lateral
        split is undefined without one -- the grader would then report the whole
        breach as medial, which would put a canal warning on a screw that left
        the vertebra laterally.  ``None`` keeps the split unmeasured instead.
        """
        side = str(getattr(screw, "side", "") or "").strip().lower()
        return side if side in ("left", "right") else None

    def _reset_state(self):
        """Reset tool state."""
        self._state = "waiting_entry"
        self._pending_entry = None
        self._notify_state_changed()

    def _notify_state_changed(self):
        """Notify callback of state change."""
        if self._on_state_changed:
            self._on_state_changed(self._state)

    def cancel(self):
        """Cancel current operation."""
        self._reset_state()

    def get_screws(self) -> List[Screw]:
        """Get all placed screws."""
        return self._screws.copy()

    def add_screw(self, screw: Screw):
        """Append an existing screw (used for plan restore)."""
        self._screws.append(screw)
        self._restamp_construct_alignment()

    def regrade_all(self) -> List[Screw]:
        """Re-evaluate every stored screw against the current grader."""
        for screw in self._screws:
            self._evaluate_screw(screw)
        self._restamp_construct_alignment()
        return self._screws.copy()

    def _restamp_construct_alignment(self) -> None:
        """Re-measure the construct metrics for the screws that carry them.

        ``rod_misalignment_mm`` and the two convergence numbers describe the
        *set*, not the screw: move one head and all three change for every
        screw on that rod.  The planner stamped them once at the end of its
        run and nothing refreshed them, so the first drag of an auto screw
        left the cockpit's Alignment row quoting the rod offset of a
        trajectory the user had already replaced.

        Only the screws that already carry them are updated, and the rod line
        is refitted over those alone -- see
        :func:`src.core.construct_alignment.restamp_carriers` for why a
        hand-placed screw must not acquire an alignment it was never part of.
        """
        from ..core.construct_alignment import restamp_carriers
        from ..core.pedicle_analyzer import VERTEBRA_LABELS

        labels = {name: label for label, name in VERTEBRA_LABELS.items()}
        restamp_carriers(
            self._screws,
            side_of=lambda s: s.side,
            entry_of=lambda s: s.entry_point,
            # ``Screw.medial_angle`` is the same signed convergence the planner
            # records, so the two agree about the same trajectory.
            convergence_of=lambda s: s.medial_angle,
            level_of=lambda s: labels.get(s.vertebra_level),
            metrics_of=lambda s: s.metrics,
        )

    def replace_screw(
        self,
        index: int,
        entry_point: Tuple[float, float, float],
        target_point: Tuple[float, float, float],
    ) -> Screw:
        """Replace one screw geometry and recalculate derived safety data."""
        if index < 0 or index >= len(self._screws):
            raise IndexError(f"Screw index out of range: {index}")

        original = self._screws[index]
        updated = Screw(
            entry_point=tuple(float(value) for value in entry_point),
            target_point=tuple(float(value) for value in target_point),
            diameter=original.diameter,
            vertebra_level=original.vertebra_level,
            side=original.side,
            source=original.source,
            # _evaluate_screw strips and regenerates every grader-derived note,
            # and merges over the metrics it can re-measure; carrying both
            # across keeps what only the planner knows (score, trajectory_type,
            # rod alignment) attached to the screw the user just dragged.
            warnings=list(original.warnings),
            metrics=self._carried_metrics(original),
        )
        self._evaluate_screw(updated)
        self._screws[index] = updated
        self._restamp_construct_alignment()
        return updated

    @staticmethod
    def _carried_metrics(original: Screw) -> Dict[str, Any]:
        """A private copy of a screw's metric bundle for its replacement."""
        return dict(original.metrics) if isinstance(original.metrics, dict) else {}

    def update_screw_diameter(self, index: int, diameter: float) -> Screw:
        """Resize one screw and recalculate its diameter-dependent safety data."""
        if index < 0 or index >= len(self._screws):
            raise IndexError(f"Screw index out of range: {index}")
        diameter = float(diameter)
        if not MIN_SCREW_DIAMETER <= diameter <= MAX_SCREW_DIAMETER:
            raise ValueError(
                f"Screw diameter must be between {MIN_SCREW_DIAMETER:.1f} "
                f"and {MAX_SCREW_DIAMETER:.1f} mm"
            )

        original = self._screws[index]
        updated = Screw(
            entry_point=original.entry_point,
            target_point=original.target_point,
            diameter=diameter,
            vertebra_level=original.vertebra_level,
            side=original.side,
            source=original.source,
            # _evaluate_screw strips and regenerates every grader-derived note.
            warnings=list(original.warnings),
            metrics=self._carried_metrics(original),
        )
        self._evaluate_screw(updated)
        self._screws[index] = updated
        return updated

    def set_screws(self, screws: List[Screw]):
        """Replace screw list with existing screws (used for plan restore)."""
        self._screws = screws.copy()
        self._reset_state()
        self._restamp_construct_alignment()

    def remove_screw(self, index: int):
        """Remove a screw by index."""
        if 0 <= index < len(self._screws):
            del self._screws[index]
            # One fewer head on that rod: the line the others are measured
            # against has moved.
            self._restamp_construct_alignment()

    def clear_screws(self):
        """Remove all screws."""
        self._screws.clear()

    def get_state(self) -> str:
        """Get current placement state."""
        return self._state

    @staticmethod
    def get_grade_description(grade: str) -> str:
        """Get description for a Gertzbein-Robbins grade."""
        descriptions = {
            "A": GRADE_A_DESCRIPTION,
            "B": GRADE_B_DESCRIPTION,
            "C": GRADE_C_DESCRIPTION,
            "D": GRADE_D_DESCRIPTION,
            "E": GRADE_E_DESCRIPTION,
            "N/A": "Not graded — run segmentation first",
        }
        return descriptions.get(grade, "Unknown grade")
