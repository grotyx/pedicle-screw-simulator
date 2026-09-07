"""
Tests for interactive tools (ScrewTool, MeasurementTool).
"""

import pytest

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.models.screw import Screw
from src.models.measurement import Measurement
from src.tools.screw_tool import ScrewTool
from src.tools.measurement_tool import MeasurementTool


class FakeVolumeManager:
    """Simple volume manager test double."""

    def __init__(self, hu_value=1000.0, has_image=True):
        self.hu_value = hu_value
        self.has_image = has_image

    def get_vtk_image(self):
        if self.has_image:
            return object()
        return None

    def get_voxel_value(self, x, y, z):
        return self.hu_value


class TestScrewTool:
    """Tests for screw placement behavior."""

    def test_entry_target_mode_places_screw_after_two_clicks(self):
        tool = ScrewTool(FakeVolumeManager())

        result = tool.on_click(0.0, 0.0, 0.0, plane="axial")
        assert result is None
        assert tool.get_state() == "waiting_target"

        result = tool.on_click(0.0, 0.0, 40.0, plane="axial")
        assert result is not None
        assert pytest.approx(result.length, rel=1e-6) == 40.0
        assert len(tool.get_screws()) == 1
        assert tool.get_state() == "waiting_entry"
        assert result.diameter == pytest.approx(6.5)

    def test_entry_angle_mode_places_screw_immediately(self):
        tool = ScrewTool(FakeVolumeManager())
        tool.set_mode("entry_angle")
        tool.set_screw_parameters(length=45.0, diameter=6.0)

        result = tool.on_click(10.0, 20.0, 30.0)

        assert result is not None
        assert pytest.approx(result.length, rel=1e-6) == 45.0
        assert len(tool.get_screws()) == 1

    def test_without_grader_grade_is_not_available(self):
        tool = ScrewTool(FakeVolumeManager(hu_value=50.0))
        tool.on_click(0.0, 0.0, 0.0, plane="axial")
        screw = tool.on_click(0.0, 0.0, 30.0, plane="axial")
        assert screw.grade == "N/A"
        assert screw.breach_distance == 0.0
        assert any("segmentation" in w.lower() for w in screw.warnings)

    def test_with_grader_uses_mask_containment(self):
        import numpy as np
        import SimpleITK as sitk
        from src.core.screw_grading import ScrewGrader
        arr = np.zeros((60, 60, 60), dtype=np.uint8)
        arr[20:40, 20:40, 20:40] = 28
        mask = sitk.GetImageFromArray(arr)
        ct = sitk.GetImageFromArray(np.where(arr > 0, 80, -50).astype(np.int16))  # osteoporotic HU
        tool = ScrewTool(FakeVolumeManager())
        tool.set_grader(ScrewGrader(mask, ct))
        tool.on_click(30.0, 38.0, 30.0, plane="axial")
        screw = tool.on_click(30.0, 22.0, 30.0, plane="axial")
        assert screw.grade == "A"          # low HU must not be called a breach
        assert screw.mean_hu == pytest.approx(80.0)

    def test_regrade_all_grades_screws_placed_before_segmentation(self):
        import numpy as np
        import SimpleITK as sitk
        from src.core.screw_grading import ScrewGrader

        tool = ScrewTool(FakeVolumeManager())
        tool.on_click(30.0, 38.0, 30.0, plane="axial")
        screw = tool.on_click(30.0, 22.0, 30.0, plane="axial")
        assert screw.grade == "N/A"
        assert any(w.startswith("Not graded") for w in screw.warnings)

        arr = np.zeros((60, 60, 60), dtype=np.uint8)
        arr[20:40, 20:40, 20:40] = 28
        mask = sitk.GetImageFromArray(arr)
        ct = sitk.GetImageFromArray(np.where(arr > 0, 80, -50).astype(np.int16))
        tool.set_grader(ScrewGrader(mask, ct))

        tool.regrade_all()

        regraded = tool.get_screws()[0]
        assert regraded.grade == "A"
        assert not any(w.startswith("Not graded") for w in regraded.warnings)
        assert regraded.mean_hu == pytest.approx(80.0)

    def test_grade_description_covers_not_available(self):
        assert (
            ScrewTool.get_grade_description("N/A")
            == "Not graded — run segmentation first"
        )

    def test_evaluate_strips_every_not_graded_warning_variant(self):
        tool = ScrewTool(FakeVolumeManager())
        screw = Screw(
            entry_point=(0.0, 0.0, 0.0),
            target_point=(0.0, 0.0, 30.0),
            diameter=6.0,
            warnings=[
                "Not graded: run segmentation first",
                "Not graded: trajectory does not pass through a segmented vertebra",
                "planner note",
            ],
        )
        tool._evaluate_screw(screw)
        assert screw.warnings.count("Not graded: run segmentation first") == 1
        assert "planner note" in screw.warnings
        assert (
            "Not graded: trajectory does not pass through a segmented vertebra"
            not in screw.warnings
        )

    def test_replace_screw_keeps_metadata(self):
        tool = ScrewTool(FakeVolumeManager())
        original = Screw(entry_point=(0.0, 0.0, 0.0), target_point=(0.0, 0.0, 40.0),
                         diameter=6.5, vertebra_level="L3", side="left",
                         source="auto", warnings=["planner note"])
        tool.add_screw(original)
        updated = tool.replace_screw(0, entry_point=(10.0, 0.0, 0.0), target_point=(10.0, 0.0, 30.0))
        assert updated.source == "auto"
        assert "planner note" in updated.warnings

    def test_add_existing_screw(self):
        tool = ScrewTool(FakeVolumeManager())
        screw = Screw(
            entry_point=(0.0, 0.0, 0.0),
            target_point=(0.0, 0.0, 30.0),
            diameter=6.0,
        )

        tool.add_screw(screw)
        screws = tool.get_screws()

        assert len(screws) == 1
        assert screws[0].diameter == 6.0

    def test_replace_screw_recomputes_geometry_and_preserves_identity(self):
        tool = ScrewTool(FakeVolumeManager())
        original = Screw(
            entry_point=(0.0, 0.0, 0.0),
            target_point=(0.0, 0.0, 40.0),
            diameter=6.5,
            vertebra_level="L3",
            side="left",
        )
        tool.add_screw(original)

        updated = tool.replace_screw(
            0,
            entry_point=(10.0, 0.0, 0.0),
            target_point=(10.0, 0.0, 30.0),
        )

        assert updated is tool.get_screws()[0]
        assert updated is not original
        assert updated.entry_point == (10.0, 0.0, 0.0)
        assert updated.target_point == (10.0, 0.0, 30.0)
        assert updated.length == pytest.approx(30.0)
        assert updated.trajectory == pytest.approx((0.0, 0.0, 1.0))
        assert updated.diameter == 6.5
        assert updated.vertebra_level == "L3"
        assert updated.side == "left"
        assert updated.grade == "N/A"

    def test_update_screw_diameter_rebuilds_safety_data_and_limits_maximum(self):
        tool = ScrewTool(FakeVolumeManager())
        tool.add_screw(
            Screw(
                entry_point=(0.0, 0.0, 0.0),
                target_point=(0.0, 0.0, 40.0),
                diameter=6.5,
                vertebra_level="L3",
                side="left",
            )
        )

        updated = tool.update_screw_diameter(0, 7.5)

        assert updated is tool.get_screws()[0]
        assert updated.diameter == pytest.approx(7.5)
        assert updated.vertebra_level == "L3"
        assert updated.side == "left"
        assert updated.grade == "N/A"
        with pytest.raises(ValueError, match="4.0.*7.5"):
            tool.update_screw_diameter(0, 8.0)


class TestMeasurementTool:
    """Tests for measurement behavior."""

    def test_distance_mode_completes_after_two_points(self):
        tool = MeasurementTool()
        tool.set_mode("distance")

        assert tool.on_click(0.0, 0.0, 0.0) is None
        measurement = tool.on_click(3.0, 4.0, 0.0)

        assert measurement is not None
        assert pytest.approx(measurement.distance, rel=1e-6) == 5.0
        assert measurement.label == "5.00 mm"
        assert len(tool.get_measurements()) == 1

    def test_path_mode_waits_for_double_click(self):
        tool = MeasurementTool()
        tool.set_mode("path")

        assert tool.on_click(0.0, 0.0, 0.0) is None
        assert tool.on_click(3.0, 4.0, 0.0) is None
        measurement = tool.on_click(6.0, 8.0, 0.0, double_click=True)

        assert measurement is not None
        assert pytest.approx(measurement.distance, rel=1e-6) == 10.0
        assert measurement.mode == "path"
        assert measurement.label == "10.0 mm"

    def test_finish_pending_path(self):
        tool = MeasurementTool()
        tool.set_mode("path")

        tool.on_click(0.0, 0.0, 0.0)
        tool.on_click(3.0, 4.0, 0.0)
        measurement = tool.finish_pending()

        assert measurement is not None
        assert measurement.mode == "path"
        assert pytest.approx(measurement.distance, rel=1e-6) == 5.0

    def test_angle_mode_returns_angle_value(self):
        tool = MeasurementTool()
        tool.set_mode("angle")

        assert tool.get_mode() == "angle"
        assert tool.on_click(1.0, 0.0, 0.0) is None
        assert tool.on_click(0.0, 0.0, 0.0) is None
        measurement = tool.on_click(0.0, 1.0, 0.0)

        assert measurement is not None
        assert measurement.mode == "angle"
        assert measurement.angle is not None
        assert pytest.approx(measurement.angle, rel=1e-6) == 90.0
        assert measurement.label == "90.0°"

    def test_remove_measurement_by_index(self):
        tool = MeasurementTool()
        tool.set_mode("distance")

        tool.on_click(0.0, 0.0, 0.0)
        tool.on_click(3.0, 4.0, 0.0)
        tool.on_click(0.0, 0.0, 0.0)
        tool.on_click(0.0, 0.0, 5.0)

        measurements = tool.get_measurements()
        assert len(measurements) == 2
        assert measurements[0].label == "5.00 mm"
        assert measurements[1].label == "5.00 mm"

        tool.remove_measurement(0)
        measurements = tool.get_measurements()
        assert len(measurements) == 1

    def test_add_existing_measurement(self):
        tool = MeasurementTool()
        measurement = Measurement(
            points=[(0.0, 0.0, 0.0), (3.0, 4.0, 0.0)],
            distance=5.0,
            mode="distance",
            label="5.00 mm",
        )

        tool.add_measurement(measurement)
        measurements = tool.get_measurements()

        assert len(measurements) == 1
        assert measurements[0].label == "5.00 mm"

    def test_replace_measurement_keeps_list_position(self):
        tool = MeasurementTool()
        original = Measurement(
            points=[(0.0, 0.0, 0.0), (3.0, 4.0, 0.0)],
            distance=5.0,
            mode="distance",
            label="5.00 mm",
        )
        replacement = Measurement(
            points=[(0.0, 0.0, 0.0), (0.0, 10.0, 0.0)],
            distance=10.0,
            mode="distance",
            label="10.0 mm",
        )
        tool.add_measurement(original)

        tool.replace_measurement(0, replacement)

        assert tool.get_measurements() == [replacement]
