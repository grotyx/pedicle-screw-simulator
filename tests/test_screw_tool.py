"""Re-grading an already-planned screw must refresh it without amnesia.

Every path that edits a placed screw — dragging it (``replace_screw``),
changing its diameter, or re-running segmentation (``regrade_all``) — routes
through :meth:`~src.tools.screw_tool.ScrewTool._evaluate_screw`.  These tests
pin the two halves of that contract: the metric bundle is *merged* so nothing
only the planner can know is lost, and every warning the grader owns is
regenerated so a re-graded screw can never display a note that contradicts the
trajectory it now has.
"""

import math

import numpy as np
import pytest
import SimpleITK as sitk

from src.core.screw_grading import ScrewGrader
from src.models.screw import Screw
from src.tools.screw_tool import ScrewTool
from src.utils.constants import CBT_CONTRAINDICATION_NOTE


class _VolumeManager:
    """Minimal stand-in: ``ScrewTool`` only asks whether an image is loaded."""

    def get_vtk_image(self):
        return object()

    def get_voxel_value(self, x, y, z):
        return 0.0


def _split_density_tool():
    """A cube vertebra whose left half is osteoporotic and right half dense.

    Spacing is 1 mm with a zero origin, so LPS millimetres equal array indices
    (arrays are ``(z, y, x)``; LPS is +x left, +y posterior, +z superior).  The
    label spans 20-39 mm on every axis; inside it, x < 30 mm is 80 HU (below
    the 123 HU loosening threshold) and x >= 30 mm is 400 HU.  Moving a screw
    across x = 30 therefore flips whether a "Trajectory HU" warning applies.
    """
    arr = np.zeros((60, 60, 60), dtype=np.uint8)
    arr[20:40, 20:40, 20:40] = 28
    hu = np.full((60, 60, 60), -50, dtype=np.int16)
    hu[20:40, 20:40, 20:30] = 80
    hu[20:40, 20:40, 30:40] = 400
    mask = sitk.GetImageFromArray(arr)
    ct = sitk.GetImageFromArray(hu)
    tool = ScrewTool(_VolumeManager())
    tool.set_grader(ScrewGrader(mask, ct))
    return tool


def _planner_screw(entry, target, **overrides):
    """An auto-planned CBT screw as the planner and CBT planner leave it."""
    metrics = {
        "trajectory_mean_hu": 95.0,
        "trajectory_min_hu": 60.0,
        "pedicle_mean_hu": 210.0,
        "body_mean_hu": 150.0,
        "trajectory_body_ratio": 0.633,
        "min_wall_mm": 1.2,
        "heary_direction": "none",
        "facet_grade": 0,
        "facet_text": "no facet contact",
        "trajectory_type": "cbt",
        "cbt_cranial_angle_deg": 24.0,
        "score": 1.9,
        "score_components": {"safety": 0.9, "density": 0.4},
        "rod_misalignment_mm": 2.5,
    }
    metrics.update(overrides.pop("metrics", {}))
    screw = Screw(
        entry_point=entry,
        target_point=target,
        diameter=6.5,
        side="left",
        vertebra_level="L4",
        source="auto",
        metrics=metrics,
        warnings=[
            "Trajectory HU 95 below 123 HU — loosening risk "
            "(consider larger diameter, augmentation, or CBT)",
            CBT_CONTRAINDICATION_NOTE,
            "Diameter reduced from 7.5 to 6.5 mm to fit the pedicle",
        ],
        **overrides,
    )
    return screw


def test_regrade_keeps_every_planner_authored_metric():
    """Nudging an auto screw must not blank what only the planner can know."""
    tool = _split_density_tool()
    tool.add_screw(_planner_screw((34.0, 38.0, 30.0), (34.0, 22.0, 30.0)))

    tool.regrade_all()

    metrics = tool.get_screws()[0].metrics
    assert metrics["score"] == pytest.approx(1.9)
    assert metrics["score_components"] == {"safety": 0.9, "density": 0.4}
    assert metrics["rod_misalignment_mm"] == pytest.approx(2.5)
    assert metrics["trajectory_type"] == "cbt"
    assert metrics["cbt_cranial_angle_deg"] == pytest.approx(24.0)
    # The pedicle analysis behind these is gone, so they are kept, not blanked.
    assert metrics["pedicle_mean_hu"] == pytest.approx(210.0)
    assert metrics["body_mean_hu"] == pytest.approx(150.0)
    # ... but the grader-owned values really are re-measured.
    assert metrics["trajectory_mean_hu"] == pytest.approx(400.0)
    # The ratio is purely derived: it follows the new trajectory HU rather than
    # staying behind describing the old one.
    assert metrics["trajectory_body_ratio"] == pytest.approx(400.0 / 150.0)


def test_regrade_drops_a_hu_warning_the_new_trajectory_disproves():
    """A screw moved into dense bone must lose the stale loosening note."""
    tool = _split_density_tool()
    tool.add_screw(_planner_screw((34.0, 38.0, 30.0), (34.0, 22.0, 30.0)))

    tool.regrade_all()

    warnings = tool.get_screws()[0].warnings
    assert not any(w.startswith("Trajectory HU") for w in warnings)
    # Notes the grader does not own survive untouched.
    assert CBT_CONTRAINDICATION_NOTE in warnings
    assert "Diameter reduced from 7.5 to 6.5 mm to fit the pedicle" in warnings


def test_regrade_raises_a_hu_warning_for_a_newly_osteoporotic_trajectory():
    """A screw moved into weak bone must gain the loosening note."""
    tool = _split_density_tool()
    screw = _planner_screw(
        (34.0, 38.0, 30.0),
        (34.0, 22.0, 30.0),
        metrics={"trajectory_mean_hu": 400.0},
    )
    screw.warnings = [CBT_CONTRAINDICATION_NOTE]
    tool.add_screw(screw)

    moved = tool.replace_screw(
        0, entry_point=(24.0, 38.0, 30.0), target_point=(24.0, 22.0, 30.0)
    )

    assert moved.metrics["trajectory_mean_hu"] == pytest.approx(80.0)
    assert any(w.startswith("Trajectory HU 80 below 123 HU") for w in moved.warnings)
    assert moved.metrics["trajectory_type"] == "cbt"
    assert CBT_CONTRAINDICATION_NOTE in moved.warnings


def test_regrade_regenerates_facet_and_convergence_warnings():
    """Facet and convergence notes follow the current trajectory, not the old one."""
    tool = _split_density_tool()
    screw = _planner_screw((34.0, 38.0, 30.0), (34.0, 22.0, 30.0))
    screw.warnings = [
        "Facet violation grade 3: superior facet violated",
        "High convergence angle 44.0° — verify on CT",
        "Trajectory/body HU ratio 0.63 below 1.0",
        CBT_CONTRAINDICATION_NOTE,
    ]
    tool.add_screw(screw)

    tool.regrade_all()

    warnings = tool.get_screws()[0].warnings
    # This straight, facet-free trajectory disproves all three planner notes.
    assert not any(w.startswith("Facet violation grade") for w in warnings)
    assert not any(w.startswith("High convergence angle") for w in warnings)
    assert not any(w.startswith("Trajectory/body HU ratio") for w in warnings)
    assert CBT_CONTRAINDICATION_NOTE in warnings


def test_edit_keeps_the_body_hu_warning_while_replacing_the_trajectory_one():
    """The body HU note describes the vertebra, not the trajectory.

    ``body_mean_hu`` needs a body centre this tool never has, so it survives an
    edit verbatim; the warning that explains that number has to survive with
    it, or the inspector shows an osteoporotic body HU and says nothing about
    it.  The trajectory HU note *is* re-measurable and must be replaced.
    """
    body_note = "Vertebral body HU 120 suggests osteoporosis (<132 HU)"
    tool = _split_density_tool()
    screw = _planner_screw(
        (34.0, 38.0, 30.0),
        (34.0, 22.0, 30.0),
        metrics={"trajectory_mean_hu": 400.0, "body_mean_hu": 120.0},
    )
    screw.warnings = [
        "Trajectory HU 400 below 123 HU — loosening risk "
        "(consider larger diameter, augmentation, or CBT)",
        body_note,
    ]
    tool.add_screw(screw)

    moved = tool.replace_screw(
        0, entry_point=(24.0, 38.0, 30.0), target_point=(24.0, 22.0, 30.0)
    )

    assert body_note in moved.warnings
    assert moved.metrics["body_mean_hu"] == pytest.approx(120.0)
    # The stale trajectory note is replaced by one measured on the new path.
    assert "Trajectory HU 400 below 123 HU — loosening risk " \
           "(consider larger diameter, augmentation, or CBT)" not in moved.warnings
    assert any(w.startswith("Trajectory HU 80 below 123 HU") for w in moved.warnings)


def test_regrade_adds_a_convergence_warning_for_a_steeply_medialised_screw():
    tool = _split_density_tool()
    screw = _planner_screw((34.0, 35.0, 30.0), (24.0, 25.0, 30.0))
    screw.warnings = []
    tool.add_screw(screw)

    tool.regrade_all()

    warnings = tool.get_screws()[0].warnings
    assert any(w.startswith("High convergence angle 45.0°") for w in warnings)


def test_diameter_change_keeps_planner_metrics():
    tool = _split_density_tool()
    tool.add_screw(_planner_screw((34.0, 38.0, 30.0), (34.0, 22.0, 30.0)))

    updated = tool.update_screw_diameter(0, 5.5)

    assert updated.metrics["score_components"] == {"safety": 0.9, "density": 0.4}
    assert updated.metrics["trajectory_type"] == "cbt"
    assert updated.metrics["body_mean_hu"] == pytest.approx(150.0)


def test_ungradable_screw_drops_only_what_it_measured():
    """A trajectory that misses every vertebra gives up its measurements only.

    ``ScrewEditController._apply_points`` re-evaluates on every drag frame, so
    emptying the bundle here would make one out-of-mask frame permanent.
    """
    tool = _split_density_tool()
    tool.add_screw(_planner_screw(
        (2.0, 2.0, 2.0), (2.0, 10.0, 2.0),
        metrics={
            "narrow_pedicle": True,
            "pedicle_width_mm": 4.5,
            "medial_breach_mm": 1.0,
            "lateral_breach_mm": 0.0,
            "craniocaudal_breach_mm": 0.0,
            "medial_wall_mm": 2.0,
        },
    ))

    tool.regrade_all()

    screw = tool.get_screws()[0]
    assert screw.grade == "N/A"
    assert screw.mean_hu is None and screw.min_hu is None
    assert any(w.startswith("Not graded") for w in screw.warnings)
    # Measured off the (missing) trajectory: gone.
    for key in ("trajectory_mean_hu", "trajectory_min_hu", "trajectory_body_ratio",
                "min_wall_mm", "heary_direction", "facet_grade", "facet_text",
                "medial_breach_mm", "lateral_breach_mm", "craniocaudal_breach_mm",
                "medial_wall_mm"):
        assert key not in screw.metrics
    # Everything the plan still knows: kept.
    assert screw.metrics["score"] == pytest.approx(1.9)
    assert screw.metrics["score_components"] == {"safety": 0.9, "density": 0.4}
    assert screw.metrics["rod_misalignment_mm"] == pytest.approx(2.5)
    assert screw.metrics["trajectory_type"] == "cbt"
    assert screw.metrics["cbt_cranial_angle_deg"] == pytest.approx(24.0)
    assert screw.metrics["pedicle_mean_hu"] == pytest.approx(210.0)
    assert screw.metrics["body_mean_hu"] == pytest.approx(150.0)
    assert screw.metrics["narrow_pedicle"] is True
    assert screw.metrics["pedicle_width_mm"] == pytest.approx(4.5)


def test_a_drag_through_open_space_and_back_restores_the_full_bundle():
    """One ungradable frame mid-gesture must not cost the plan anything."""
    tool = _split_density_tool()
    tool.add_screw(_planner_screw((34.0, 38.0, 30.0), (34.0, 22.0, 30.0)))

    # Frame 1: the user has dragged the screw clear of every labelled vertebra.
    outside = tool.replace_screw(
        0, entry_point=(2.0, 2.0, 2.0), target_point=(2.0, 10.0, 2.0)
    )
    assert outside.grade == "N/A"

    # Frame 2: released back inside the vertebra.
    inside = tool.replace_screw(
        0, entry_point=(34.0, 38.0, 30.0), target_point=(34.0, 22.0, 30.0)
    )

    assert inside.grade == "A"
    assert inside.metrics["score"] == pytest.approx(1.9)
    assert inside.metrics["score_components"] == {"safety": 0.9, "density": 0.4}
    assert inside.metrics["rod_misalignment_mm"] == pytest.approx(2.5)
    assert inside.metrics["trajectory_type"] == "cbt"
    assert inside.metrics["cbt_cranial_angle_deg"] == pytest.approx(24.0)
    assert inside.metrics["body_mean_hu"] == pytest.approx(150.0)
    assert inside.metrics["pedicle_mean_hu"] == pytest.approx(210.0)
    # The measurements come back too, rebuilt from the trajectory it now has.
    assert inside.metrics["trajectory_mean_hu"] == pytest.approx(400.0)
    assert inside.metrics["trajectory_body_ratio"] == pytest.approx(400.0 / 150.0)


def test_a_low_ratio_warning_is_regenerated_not_silently_dropped():
    """The ratio note is stripped as derived, so it has to be re-raised.

    ``assess_bone_quality`` cannot produce it here (no vertebral-body centre),
    but ``_merge_metrics`` does recompute the ratio itself — so without an
    explicit regeneration every drag and every plan load would delete the
    warning while leaving the number it describes on screen.
    """
    tool = _split_density_tool()
    screw = _planner_screw(
        (34.0, 38.0, 30.0),
        (34.0, 22.0, 30.0),
        metrics={"trajectory_mean_hu": 400.0, "trajectory_body_ratio": 400.0 / 150.0},
    )
    screw.warnings = [CBT_CONTRAINDICATION_NOTE]
    tool.add_screw(screw)

    # Dragged into the osteoporotic half: 80 / 150 = 0.53, below the 1.0 ratio.
    moved = tool.replace_screw(
        0, entry_point=(24.0, 38.0, 30.0), target_point=(24.0, 22.0, 30.0)
    )

    assert moved.metrics["trajectory_body_ratio"] == pytest.approx(80.0 / 150.0)
    assert "Trajectory/body HU ratio 0.53 below 1.0" in moved.warnings
    assert CBT_CONTRAINDICATION_NOTE in moved.warnings
    # Exactly one copy, no matter how many times the screw is re-evaluated.
    tool.regrade_all()
    regraded = tool.get_screws()[0]
    assert regraded.warnings.count("Trajectory/body HU ratio 0.53 below 1.0") == 1

    # Dragged back into dense bone: 2.67, so the note goes away again.
    back = tool.replace_screw(
        0, entry_point=(34.0, 38.0, 30.0), target_point=(34.0, 22.0, 30.0)
    )
    assert not any(w.startswith("Trajectory/body HU ratio") for w in back.warnings)


def test_a_medial_breach_warning_is_regenerated_not_silently_dropped():
    """The canal note is stripped as derived, so it has to be re-raised.

    Mirrors ``test_a_low_ratio_warning_is_regenerated_not_silently_dropped``:
    without an explicit regeneration, repeated re-grading would either drop the
    note while the breach stayed on screen, or pile up a duplicate copy per call.
    """
    tool = _split_density_tool()
    screw = Screw(
        entry_point=(39.0, 35.0, 30.0),
        target_point=(39.0, 25.0, 30.0),
        diameter=6.0,
        side="right",              # +X is toward the midline for a right pedicle
    )
    tool.add_screw(screw)

    tool.regrade_all()
    tool.regrade_all()
    tool.regrade_all()

    regraded = tool.get_screws()[0]
    assert regraded.warnings.count("Medial breach 3.0 mm — canal side") == 1

    # Dragged off the canal side: the note has nothing left to describe.
    moved = tool.replace_screw(
        0, entry_point=(30.0, 35.0, 30.0), target_point=(30.0, 25.0, 30.0)
    )
    assert sum(1 for w in moved.warnings if w.startswith("Medial breach")) == 0


def test_medial_breach_mm_tracks_the_trajectory_across_a_drag():
    """The number itself must follow the trajectory, not just its sign."""
    tool = _split_density_tool()
    tool.add_screw(Screw(
        entry_point=(37.0, 35.0, 30.0),
        target_point=(37.0, 25.0, 30.0),
        diameter=6.0,
        side="right",
    ))

    tool.regrade_all()
    first = tool.get_screws()[0].metrics["medial_breach_mm"]

    deeper = tool.replace_screw(
        0, entry_point=(38.0, 35.0, 30.0), target_point=(38.0, 25.0, 30.0)
    )

    assert first == pytest.approx(1.0)
    assert deeper.metrics["medial_breach_mm"] == pytest.approx(2.0)
    assert deeper.metrics["medial_breach_mm"] != pytest.approx(first)


class TestDirectionalRegrade:
    """A re-graded screw keeps the planner's flag and refreshes the measurement."""

    def _grader(self):
        import numpy as np
        import SimpleITK as sitk

        from src.core.screw_grading import ScrewGrader

        arr = np.zeros((60, 60, 60), dtype=np.uint8)
        arr[20:40, 20:40, 20:40] = 28
        mask = sitk.GetImageFromArray(arr)
        ct = sitk.GetImageFromArray(np.where(arr > 0, 350, -50).astype(np.int16))
        ct.CopyInformation(mask)
        return ScrewGrader(mask, ct)

    def test_regrade_preserves_the_planner_flag_and_refreshes_the_breach(self):
        from src.models.screw import Screw
        from src.tools.screw_tool import ScrewTool

        tool = ScrewTool(volume_manager=None)
        tool.set_grader(self._grader())
        screw = Screw(
            entry_point=(39.0, 35.0, 30.0),
            target_point=(39.0, 25.0, 30.0),
            diameter=6.0,
            side="left",
            metrics={"narrow_pedicle": True, "pedicle_width_mm": 4.5, "score": 1.25},
        )

        tool.add_screw(screw)
        tool.regrade_all()

        # Planner-owned, untouched by the grader.
        assert screw.metrics["narrow_pedicle"] is True
        assert screw.metrics["pedicle_width_mm"] == 4.5
        assert screw.metrics["score"] == 1.25
        # Grader-measured, rewritten from the trajectory in front of it.
        assert screw.metrics["medial_breach_mm"] == 0.0
        assert screw.metrics["lateral_breach_mm"] > 0.0
        assert screw.metrics["medial_wall_mm"] > 0.0

    def test_a_medial_breach_raises_a_canal_warning_that_a_move_clears(self):
        from src.models.screw import Screw
        from src.tools.screw_tool import ScrewTool

        tool = ScrewTool(volume_manager=None)
        tool.set_grader(self._grader())
        screw = Screw(
            entry_point=(39.0, 35.0, 30.0),
            target_point=(39.0, 25.0, 30.0),
            diameter=6.0,
            side="right",              # +X is toward the midline for a right pedicle
        )

        tool.add_screw(screw)
        tool.regrade_all()
        assert "Medial breach 3.0 mm — canal side" in screw.warnings

        screw.entry_point = (30.0, 35.0, 30.0)
        screw.target_point = (30.0, 25.0, 30.0)
        tool.regrade_all()
        assert not any(w.startswith("Medial breach") for w in screw.warnings)

    def test_a_screw_with_no_side_reports_no_direction(self):
        """Without a side the split is meaningless; it must not be invented."""
        from src.models.screw import Screw
        from src.tools.screw_tool import ScrewTool

        tool = ScrewTool(volume_manager=None)
        tool.set_grader(self._grader())
        screw = Screw(
            entry_point=(39.0, 35.0, 30.0),
            target_point=(39.0, 25.0, 30.0),
            diameter=6.0,
        )

        tool.add_screw(screw)
        tool.regrade_all()

        assert screw.breach_distance > 0.0
        assert screw.metrics["medial_breach_mm"] is None
        assert screw.metrics["lateral_breach_mm"] is None
        assert not any(w.startswith("Medial breach") for w in screw.warnings)


class _Analysis:
    """Duck-typed stand-in: ScrewTool only reads upper_endplate_normal."""

    def __init__(self, normal):
        self.upper_endplate_normal = normal


_TEN_DEGREE_ENDPLATE = (0.0, math.sin(math.radians(10.0)), math.cos(math.radians(10.0)))


def test_regrade_measures_the_endplate_angle_when_the_level_is_registered():
    tool = _split_density_tool()
    tool.set_analysis_by_level({28: _Analysis(_TEN_DEGREE_ENDPLATE)})
    screw = Screw(
        entry_point=(35.0, 38.0, 30.0),
        target_point=(35.0, 22.0, 30.0),   # horizontal: 10 degrees caudal of the endplate
        diameter=5.0,
        vertebra_level="L3",
        side="left",
    )

    tool.add_screw(screw)
    tool.regrade_all()

    assert screw.metrics["endplate_angle_deg"] == pytest.approx(-10.0, abs=0.5)


def test_regrade_leaves_a_planner_endplate_angle_alone_without_a_registry():
    tool = _split_density_tool()
    screw = Screw(
        entry_point=(35.0, 38.0, 30.0),
        target_point=(35.0, 22.0, 30.0),
        diameter=5.0,
        vertebra_level="L3",
        side="left",
        metrics={"endplate_angle_deg": 2.4},
    )

    tool.add_screw(screw)
    tool.regrade_all()

    assert screw.metrics["endplate_angle_deg"] == pytest.approx(2.4)


def test_regrade_refreshes_a_stale_endplate_angle_when_the_level_is_registered():
    tool = _split_density_tool()
    tool.set_analysis_by_level({28: _Analysis(_TEN_DEGREE_ENDPLATE)})
    screw = Screw(
        entry_point=(35.0, 38.0, 30.0),
        target_point=(35.0, 22.0, 30.0),
        diameter=5.0,
        vertebra_level="L3",
        side="left",
        metrics={"endplate_angle_deg": 99.0},
    )

    tool.add_screw(screw)
    tool.regrade_all()

    assert screw.metrics["endplate_angle_deg"] == pytest.approx(-10.0, abs=0.5)


def test_an_ungradable_frame_does_not_erase_the_endplate_angle():
    """A screw dragged briefly outside the mask must come back with its metric."""
    tool = _split_density_tool()
    # The same clear-of-every-label trajectory the drag test above uses; a
    # far-off-volume point is clamped back into the mask by the grader.
    screw = Screw(
        entry_point=(2.0, 2.0, 2.0),
        target_point=(2.0, 10.0, 2.0),
        diameter=5.0,
        vertebra_level="L3",
        side="left",
        metrics={"endplate_angle_deg": 2.4},
    )

    tool.add_screw(screw)
    tool.regrade_all()

    assert screw.grade == "N/A"
    assert screw.metrics["endplate_angle_deg"] == pytest.approx(2.4)


# --------------------------------------------------------------------------
# The thresholds a manual edit is judged against follow the planner's config
# --------------------------------------------------------------------------


def _thin_wall_tool():
    """A 0.2 mm-spaced cube, so a sub-millimetre cortical wall is measurable.

    At 1 mm spacing the distance map can only resolve whole millimetres, and
    the clearance note is about fractions of one.  The label spans 4.0-19.8 mm
    on every axis.
    """
    arr = np.zeros((120, 120, 120), dtype=np.uint8)
    arr[20:100, 20:100, 20:100] = 28
    mask = sitk.GetImageFromArray(arr)
    mask.SetSpacing((0.2, 0.2, 0.2))
    ct = sitk.GetImageFromArray(np.where(arr > 0, 350, -50).astype(np.int16))
    ct.CopyInformation(mask)
    tool = ScrewTool(_VolumeManager())
    tool.set_grader(ScrewGrader(mask, ct))
    return tool


def _thin_wall_screw():
    """A contained screw whose cylinder leaves exactly 0.4 mm of cortex."""
    return Screw(
        entry_point=(16.5, 14.0, 12.0),
        target_point=(16.5, 8.0, 12.0),
        diameter=6.0,
        vertebra_level="L3",
        side="left",
    )


def test_the_tool_thresholds_default_to_the_planner_config_defaults():
    """A default-configured run and an untouched tool must judge alike."""
    from src.core.planner_config import PlannerConfig
    from src.tools import screw_tool as screw_tool_module

    config = PlannerConfig()

    assert screw_tool_module.DEFAULT_WALL_CLEARANCE_MM == config.wall_clearance_mm
    assert screw_tool_module.DEFAULT_NARROW_PEDICLE_MM == config.narrow_pedicle_mm


def test_a_zero_clearance_adds_no_cortical_note_on_the_first_drag():
    """The planner would not have written one, so neither may a regrade."""
    tool = _thin_wall_tool()
    screw = _thin_wall_screw()

    tool.add_screw(screw)
    tool.regrade_all()

    assert screw.metrics["min_wall_mm"] == pytest.approx(0.4)
    assert not any(w.startswith("Cortical clearance") for w in screw.warnings)


def test_the_configured_clearance_is_the_one_the_note_reports():
    tool = _thin_wall_tool()
    tool.set_wall_clearance_mm(1.0)
    screw = _thin_wall_screw()

    tool.add_screw(screw)
    tool.regrade_all()

    assert "Cortical clearance 0.4 mm below 1.0 mm" in screw.warnings


def test_lowering_the_clearance_retracts_the_note_it_authored():
    tool = _thin_wall_tool()
    tool.set_wall_clearance_mm(1.0)
    screw = _thin_wall_screw()
    tool.add_screw(screw)
    tool.regrade_all()
    assert any(w.startswith("Cortical clearance") for w in screw.warnings)

    tool.set_wall_clearance_mm(0.0)
    tool.regrade_all()

    assert not any(w.startswith("Cortical clearance") for w in screw.warnings)


# --------------------------------------------------------------------------
# The pedicle width follows the level the screw is actually in
# --------------------------------------------------------------------------


class _WidthAnalysis:
    """Duck-typed stand-in carrying the two width fields the tool re-reads."""

    def __init__(self, left, right, flags=None, normal=None):
        self.left_pedicle_width = left
        self.right_pedicle_width = right
        self.width_flags = dict(flags or {})
        self.upper_endplate_normal = normal


def _two_level_tool():
    """A cube split into an inferior label 28 and a superior label 29.

    Array indices are ``(z, y, x)`` at 1 mm spacing with a zero origin, so a
    screw at z = 25 mm is in label 28 and one at z = 35 mm is in label 29 --
    the drag that moves a screw from one level to the next.
    """
    arr = np.zeros((60, 60, 60), dtype=np.uint8)
    arr[20:30, 20:40, 20:40] = 28
    arr[30:40, 20:40, 20:40] = 29
    mask = sitk.GetImageFromArray(arr)
    ct = sitk.GetImageFromArray(np.where(arr > 0, 350, -50).astype(np.int16))
    ct.CopyInformation(mask)
    tool = ScrewTool(_VolumeManager())
    tool.set_grader(ScrewGrader(mask, ct))
    return tool


def _narrow_l2_screw(z):
    return Screw(
        entry_point=(30.0, 38.0, z),
        target_point=(30.0, 22.0, z),
        diameter=5.0,
        vertebra_level="L2",
        side="left",
        metrics={"narrow_pedicle": True, "pedicle_width_mm": 4.5, "score": 1.25},
    )


def test_dragging_a_narrow_screw_into_a_wide_level_re_measures_the_width():
    tool = _two_level_tool()
    tool.set_analysis_by_level(
        {28: _WidthAnalysis(4.5, 4.5), 29: _WidthAnalysis(9.2, 9.2)}
    )
    screw = _narrow_l2_screw(25.0)
    tool.add_screw(screw)
    tool.regrade_all()
    assert screw.metrics["narrow_pedicle"] is True

    moved = tool.replace_screw(
        0, entry_point=(30.0, 38.0, 35.0), target_point=(30.0, 22.0, 35.0)
    )

    assert moved.metrics["narrow_pedicle"] is False
    assert moved.metrics["pedicle_width_mm"] == pytest.approx(9.2)
    # Still the optimiser's, still untouched by the grader.
    assert moved.metrics["score"] == 1.25


def test_without_a_registry_the_planner_width_survives_the_drag():
    """Never blank a number the tool has no way to re-measure."""
    tool = _two_level_tool()
    screw = _narrow_l2_screw(25.0)
    tool.add_screw(screw)

    moved = tool.replace_screw(
        0, entry_point=(30.0, 38.0, 35.0), target_point=(30.0, 22.0, 35.0)
    )

    assert moved.metrics["narrow_pedicle"] is True
    assert moved.metrics["pedicle_width_mm"] == pytest.approx(4.5)


def test_the_re_measured_width_uses_the_screws_own_side():
    tool = _two_level_tool()
    tool.set_analysis_by_level({28: _WidthAnalysis(4.5, 9.2)})
    screw = _narrow_l2_screw(25.0)
    screw.side = "right"

    tool.add_screw(screw)
    tool.regrade_all()

    assert screw.metrics["pedicle_width_mm"] == pytest.approx(9.2)
    assert screw.metrics["narrow_pedicle"] is False


def test_an_implausible_width_still_counts_as_narrow():
    tool = _two_level_tool()
    tool.set_analysis_by_level(
        {28: _WidthAnalysis(21.0, 21.0, flags={"left": "implausible"})}
    )
    screw = _narrow_l2_screw(25.0)

    tool.add_screw(screw)
    tool.regrade_all()

    assert screw.metrics["pedicle_width_mm"] == pytest.approx(21.0)
    assert screw.metrics["narrow_pedicle"] is True


def test_the_narrow_threshold_follows_the_configured_one():
    tool = _two_level_tool()
    tool.set_analysis_by_level({28: _WidthAnalysis(6.0, 6.0)})
    screw = _narrow_l2_screw(25.0)
    tool.add_screw(screw)
    tool.regrade_all()
    assert screw.metrics["narrow_pedicle"] is False

    tool.set_narrow_pedicle_mm(7.0)
    tool.regrade_all()

    assert screw.metrics["narrow_pedicle"] is True


def test_a_screw_with_no_side_keeps_the_planner_width():
    """The width is per side; without one there is nothing to re-read."""
    tool = _two_level_tool()
    tool.set_analysis_by_level({28: _WidthAnalysis(9.2, 9.2)})
    screw = _narrow_l2_screw(25.0)
    screw.side = ""

    tool.add_screw(screw)
    tool.regrade_all()

    assert screw.metrics["narrow_pedicle"] is True
    assert screw.metrics["pedicle_width_mm"] == pytest.approx(4.5)


def test_an_ungradable_frame_does_not_erase_the_pedicle_width():
    """Mid-drag, outside every label: the colour must not flicker to normal."""
    tool = _two_level_tool()
    tool.set_analysis_by_level({28: _WidthAnalysis(4.5, 4.5)})
    screw = _narrow_l2_screw(25.0)

    tool.add_screw(screw)
    screw.entry_point = (2.0, 2.0, 2.0)
    screw.target_point = (2.0, 10.0, 2.0)
    tool.regrade_all()

    assert screw.grade == "N/A"
    assert screw.metrics["narrow_pedicle"] is True
    assert screw.metrics["pedicle_width_mm"] == pytest.approx(4.5)
