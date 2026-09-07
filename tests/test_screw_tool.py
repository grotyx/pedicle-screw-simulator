"""Re-grading an already-planned screw must refresh it without amnesia.

Every path that edits a placed screw — dragging it (``replace_screw``),
changing its diameter, or re-running segmentation (``regrade_all``) — routes
through :meth:`~src.tools.screw_tool.ScrewTool._evaluate_screw`.  These tests
pin the two halves of that contract: the metric bundle is *merged* so nothing
only the planner can know is lost, and every warning the grader owns is
regenerated so a re-graded screw can never display a note that contradicts the
trajectory it now has.
"""

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
        "Vertebral body HU 120 suggests osteoporosis (<132 HU)",
        CBT_CONTRAINDICATION_NOTE,
    ]
    tool.add_screw(screw)

    tool.regrade_all()

    warnings = tool.get_screws()[0].warnings
    # This straight, facet-free trajectory disproves all four planner notes.
    assert not any(w.startswith("Facet violation grade") for w in warnings)
    assert not any(w.startswith("High convergence angle") for w in warnings)
    assert not any(w.startswith("Trajectory/body HU ratio") for w in warnings)
    assert not any(w.startswith("Vertebral body HU") for w in warnings)
    assert CBT_CONTRAINDICATION_NOTE in warnings


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


def test_ungradable_screw_still_drops_every_metric():
    """A trajectory that misses every vertebra keeps nothing at all."""
    tool = _split_density_tool()
    tool.add_screw(_planner_screw((2.0, 2.0, 2.0), (2.0, 10.0, 2.0)))

    tool.regrade_all()

    screw = tool.get_screws()[0]
    assert screw.grade == "N/A"
    assert screw.metrics == {}
    assert any(w.startswith("Not graded") for w in screw.warnings)
