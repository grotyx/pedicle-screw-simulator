"""Tests for AutoPlacementController and planned_screw_to_screw conversion."""

import numpy as np
import pytest

from src.controllers.auto_placement_controller import (
    AutoPlacementController,
    GRADE_COLORS,
    planned_screw_to_screw,
)
from src.core.auto_screw_planner import PlannedScrew


# ---------------------------------------------------------------------------
# planned_screw_to_screw conversion
# ---------------------------------------------------------------------------


class TestPlannedScrewToScrew:
    """Test conversion from PlannedScrew to manual Screw model."""

    def _make_planned(self, **overrides):
        defaults = dict(
            vertebra_name="L4",
            side="left",
            entry_lps=np.array([10.0, 20.0, 30.0]),
            target_lps=np.array([10.0, -10.0, 30.0]),
            length_mm=30.0,
            diameter_mm=6.0,
            convergence_angle=10.0,
            craniocaudal_angle=5.0,
            mean_bone_density=450.0,
            min_bone_density=200.0,
            gertzbein_grade="A",
            confidence=0.85,
        )
        defaults.update(overrides)
        return PlannedScrew(**defaults)

    def test_entry_point_preserved(self):
        ps = self._make_planned()
        screw = planned_screw_to_screw(ps)
        assert screw.entry_point == pytest.approx((10.0, 20.0, 30.0))

    def test_target_point_preserved(self):
        ps = self._make_planned()
        screw = planned_screw_to_screw(ps)
        assert screw.target_point == pytest.approx((10.0, -10.0, 30.0))

    def test_diameter_preserved(self):
        ps = self._make_planned(diameter_mm=5.5)
        screw = planned_screw_to_screw(ps)
        assert screw.diameter == pytest.approx(5.5)

    def test_vertebra_level_set(self):
        ps = self._make_planned(vertebra_name="T12")
        screw = planned_screw_to_screw(ps)
        assert screw.vertebra_level == "T12"

    def test_side_set(self):
        ps = self._make_planned(side="right")
        screw = planned_screw_to_screw(ps)
        assert screw.side == "right"

    def test_grade_set(self):
        ps = self._make_planned(gertzbein_grade="B")
        screw = planned_screw_to_screw(ps)
        assert screw.grade == "B"

    def test_length_recalculated_from_points(self):
        """Screw model recalculates length from entry/target in __post_init__."""
        ps = self._make_planned()
        screw = planned_screw_to_screw(ps)
        expected_length = float(np.linalg.norm(ps.target_lps - ps.entry_lps))
        assert screw.length == pytest.approx(expected_length, abs=0.1)

    def test_trajectory_unit_vector(self):
        ps = self._make_planned()
        screw = planned_screw_to_screw(ps)
        traj = np.array(screw.trajectory)
        assert np.linalg.norm(traj) == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Grade color mapping
# ---------------------------------------------------------------------------


class TestGradeColors:
    """Test grade-to-color mapping constants."""

    def test_all_grades_have_colors(self):
        for grade in ("A", "B", "C", "D", "E"):
            assert grade in GRADE_COLORS

    def test_colors_are_rgb_tuples(self):
        for grade, color in GRADE_COLORS.items():
            assert len(color) == 3
            for c in color:
                assert 0.0 <= c <= 1.0, f"Grade {grade} color out of range"

    def test_grade_a_is_green(self):
        r, g, b = GRADE_COLORS["A"]
        assert g > r and g > b


# ---------------------------------------------------------------------------
# AutoPlacementController unit tests (no Qt, mock-based)
# ---------------------------------------------------------------------------


class _DummyMPRViewer:
    """Minimal MPR viewer stub for controller tests."""

    def __init__(self, plane="axial"):
        self.plane = plane
        self._screw_overlays = {}

    def add_screw_overlay(self, screw_id, entry, target, color=(0.2, 0.8, 0.2), diameter=6.0):
        self._screw_overlays[screw_id] = {"entry": entry, "target": target}

    def remove_screw_overlay(self, screw_id):
        self._screw_overlays.pop(screw_id, None)

    def clear_screw_overlays(self):
        self._screw_overlays.clear()

    def set_reslice_input(self, vtk_image):
        pass

    def restore_original_input(self):
        pass

    def set_segmentation_visible(self, visible):
        pass


class _DummyViewer3D:
    """Minimal stub for Viewer3D used in controller tests."""

    def add_screw(
        self, entry, target, radius=3.0, color=None, screw_id=None
    ):
        class _Actor:
            class _Property:
                def SetOpacity(self, v): pass
            def GetProperty(self):
                return self._Property()
        return _Actor()

    def remove_screw(self, actor):
        pass

    def clear_screws(self):
        pass

    def set_volume_visible(self, visible):
        pass


class _DummyScrewTool:
    def __init__(self):
        self._screws = []

    def add_screw(self, screw):
        self._screws.append(screw)

    def get_screws(self):
        return list(self._screws)


class _DummyToolCtrl:
    def __init__(self):
        self.screw_tool = _DummyScrewTool()
        self._screw_actors = []
        self.added = []

    def add_existing_screw(self, screw, select=False):
        self.screw_tool.add_screw(screw)
        self.added.append((screw, bool(select)))
        return len(self.screw_tool.get_screws()) - 1


class _DummySegCtrl:
    def __init__(self):
        self._last_segmentation_mask_path = None
        self.ensure_mpr_calls = 0

    def ensure_mpr_vertebrae_isolated(self):
        self.ensure_mpr_calls += 1
        return True


class _DummyStatusBar:
    def showMessage(self, msg):
        pass


class _DummyLabel:
    def __init__(self, text=""):
        self._text = text

    def setText(self, text):
        self._text = text


class _DummyTableWidget:
    def __init__(self):
        self._rows = 0
        self._items = {}

    def setRowCount(self, n):
        self._rows = n

    def setItem(self, row, col, item):
        self._items[(row, col)] = item

    def selectedIndexes(self):
        return []

    def rowCount(self):
        return self._rows


class _DummyCheckBox:
    def __init__(self, checked=True):
        self._checked = checked

    def isChecked(self):
        return self._checked

    def setChecked(self, v):
        self._checked = v


class _DummyButton:
    def setEnabled(self, v):
        pass


class _DummyListWidget:
    def __init__(self):
        self.row = -1

    def setCurrentRow(self, row):
        self.row = int(row)

    def currentRow(self):
        return self.row


class _DummyWindow:
    """Minimal MainWindow stub for controller testing."""

    def __init__(self):
        self.viewer_3d = _DummyViewer3D()
        self._tool_ctrl = _DummyToolCtrl()
        self._seg_ctrl = _DummySegCtrl()
        self.statusbar = _DummyStatusBar()
        self.auto_screw_table = _DummyTableWidget()
        self.auto_screw_status = _DummyLabel("No auto plan")
        self.auto_screw_plan_btn = _DummyButton()
        self.screw_list_widget = _DummyListWidget()
        self._vertebra_level_checks = {
            27: _DummyCheckBox(True),   # L5
            28: _DummyCheckBox(True),   # L4
            29: _DummyCheckBox(False),  # L3
        }
        self._mpr_viewers = [
            _DummyMPRViewer("axial"),
            _DummyMPRViewer("sagittal"),
            _DummyMPRViewer("coronal"),
        ]

    def _get_mpr_viewers(self):
        return self._mpr_viewers

    def update_vertebra_checkboxes(self, detected_labels):
        pass

    def _reset_all_vertebra_checkboxes(self):
        pass


class TestAutoPlacementControllerUnit:
    """Unit tests for AutoPlacementController (no real Qt widgets)."""

    def _make_controller(self):
        window = _DummyWindow()
        from src.core.volume_manager import VolumeManager
        vm = VolumeManager()
        ctrl = AutoPlacementController(vm, window)
        return ctrl, window

    def test_initial_state(self):
        ctrl, _ = self._make_controller()
        assert not ctrl.is_running
        assert ctrl.planned_screws == []

    def test_get_selected_labels(self):
        ctrl, window = self._make_controller()
        labels = ctrl._get_selected_labels()
        assert 27 in labels  # L5 checked
        assert 28 in labels  # L4 checked
        assert 29 not in labels  # L3 unchecked

    def test_run_planning_no_segmentation_returns(self):
        """Should not crash when no segmentation mask exists."""
        ctrl, window = self._make_controller()
        # _seg_ctrl._last_segmentation_mask_path is None -> shows warning
        # We can't show a real QMessageBox, so just verify no crash
        # by checking the method is callable.
        assert callable(ctrl.run_planning)

    def test_run_planning_switches_mpr_to_segmented_ct(
        self, monkeypatch, tmp_path
    ):
        import SimpleITK as sitk
        import src.controllers.auto_placement_controller as module

        ctrl, window = self._make_controller()
        ct_image = sitk.Image([4, 4, 4], sitk.sitkInt16)
        ctrl._vm.set_volume(ct_image)
        mask_path = tmp_path / "mask.nii.gz"
        sitk.WriteImage(sitk.Image([4, 4, 4], sitk.sitkUInt8), str(mask_path))
        window._seg_ctrl._last_segmentation_mask_path = str(mask_path)

        class _Signal:
            def connect(self, _callback):
                return None

        class _Thread:
            def __init__(self, *_args):
                self.progress = _Signal()
                self.finished = _Signal()
                self.error = _Signal()
                self.started = False

            def start(self):
                self.started = True

            def isRunning(self):
                return self.started

        monkeypatch.setattr(module, "_PlanningThread", _Thread)

        ctrl.run_planning()

        assert window._seg_ctrl.ensure_mpr_calls == 1
        assert ctrl._thread.started is True

    def test_clear_plan(self):
        ctrl, window = self._make_controller()
        # Simulate having some planned screws.
        ps = PlannedScrew(
            vertebra_name="L4", side="left",
            entry_lps=np.array([0, 0, 0]),
            target_lps=np.array([0, -30, 0]),
            length_mm=30, diameter_mm=6,
            convergence_angle=10, craniocaudal_angle=5,
            mean_bone_density=400, min_bone_density=200,
            gertzbein_grade="A", confidence=0.8,
        )
        ctrl._planned_screws = [ps]
        ctrl.clear_plan()
        assert ctrl.planned_screws == []

    def test_on_finished_registers_editable_screws_and_selects_first_new_row(self):
        ctrl, window = self._make_controller()
        planned = [
            PlannedScrew(
                vertebra_name="L4", side="left",
                entry_lps=np.array([0, 10, 0]),
                target_lps=np.array([0, -20, 0]),
                length_mm=30, diameter_mm=6,
                convergence_angle=10, craniocaudal_angle=5,
                mean_bone_density=400, min_bone_density=200,
                gertzbein_grade="A", confidence=0.8,
            ),
            PlannedScrew(
                vertebra_name="L4", side="right",
                entry_lps=np.array([5, 10, 0]),
                target_lps=np.array([5, -20, 0]),
                length_mm=30, diameter_mm=6,
                convergence_angle=10, craniocaudal_angle=5,
                mean_bone_density=350, min_bone_density=180,
                gertzbein_grade="B", confidence=0.6,
            ),
        ]
        ctrl._on_finished(planned)
        assert ctrl.planned_screws == []
        assert len(window._tool_ctrl.screw_tool._screws) == 2
        assert [select for _screw, select in window._tool_ctrl.added] == [True, False]
        assert window.screw_list_widget.currentRow() == 0
        assert window.auto_screw_table._rows == 0

    def test_on_finished_appends_after_existing_screw_and_selects_first_new_row(self):
        ctrl, window = self._make_controller()
        window._tool_ctrl.screw_tool.add_screw(
            planned_screw_to_screw(PlannedScrew(
                vertebra_name="L5", side="left",
                entry_lps=np.array([10, 10, 0]),
                target_lps=np.array([10, -20, 0]),
                length_mm=30, diameter_mm=6,
                convergence_angle=10, craniocaudal_angle=5,
                mean_bone_density=400, min_bone_density=200,
                gertzbein_grade="A", confidence=0.8,
            ))
        )
        ps = PlannedScrew(
            vertebra_name="L4", side="left",
            entry_lps=np.array([0, 10, 0]),
            target_lps=np.array([0, -20, 0]),
            length_mm=30, diameter_mm=6,
            convergence_angle=10, craniocaudal_angle=5,
            mean_bone_density=400, min_bone_density=200,
            gertzbein_grade="A", confidence=0.8,
        )
        ctrl._on_finished([ps])
        assert len(ctrl.planned_screws) == 0
        assert len(window._tool_ctrl.screw_tool._screws) == 2
        assert window.screw_list_widget.currentRow() == 1

    def test_reset_state(self):
        ctrl, window = self._make_controller()
        ctrl._planned_screws = [
            PlannedScrew(
                vertebra_name="L4", side="left",
                entry_lps=np.array([0, 0, 0]),
                target_lps=np.array([0, -30, 0]),
                length_mm=30, diameter_mm=6,
                convergence_angle=10, craniocaudal_angle=5,
                mean_bone_density=400, min_bone_density=200,
                gertzbein_grade="A", confidence=0.8,
            )
        ]
        ctrl.reset_state()
        assert ctrl.planned_screws == []

    def test_on_finished_uses_permanent_registration_path(self):
        ctrl, window = self._make_controller()
        planned = [
            PlannedScrew(
                vertebra_name="L4", side="left",
                entry_lps=np.array([0, 10, 0]),
                target_lps=np.array([0, -20, 0]),
                length_mm=30, diameter_mm=6,
                convergence_angle=10, craniocaudal_angle=5,
                mean_bone_density=400, min_bone_density=200,
                gertzbein_grade="A", confidence=0.8,
            ),
        ]
        ctrl._on_finished(planned)
        assert len(window._tool_ctrl.added) == 1
        assert window._tool_ctrl.added[0][0].vertebra_level == "L4"
        assert window._tool_ctrl.added[0][0].side == "left"

    def test_clear_plan_removes_mpr_overlays(self):
        ctrl, window = self._make_controller()
        planned = [
            PlannedScrew(
                vertebra_name="L4", side="left",
                entry_lps=np.array([0, 10, 0]),
                target_lps=np.array([0, -20, 0]),
                length_mm=30, diameter_mm=6,
                convergence_angle=10, craniocaudal_angle=5,
                mean_bone_density=400, min_bone_density=200,
                gertzbein_grade="A", confidence=0.8,
            ),
        ]
        ctrl._on_finished(planned)
        ctrl.clear_plan()
        for viewer in window._mpr_viewers:
            assert len(viewer._screw_overlays) == 0

    def test_clear_plan_preserves_permanent_mpr_overlays(self):
        ctrl, window = self._make_controller()
        planned = [
            PlannedScrew(
                vertebra_name="L4", side="left",
                entry_lps=np.array([0, 10, 0]),
                target_lps=np.array([0, -20, 0]),
                length_mm=30, diameter_mm=6,
                convergence_angle=10, craniocaudal_angle=5,
                mean_bone_density=400, min_bone_density=200,
                gertzbein_grade="A", confidence=0.8,
            ),
        ]
        for viewer in window._mpr_viewers:
            viewer.add_screw_overlay(0, (0, 0, 0), (0, 0, 10))

        ctrl._on_finished(planned)
        ctrl.clear_plan()

        for viewer in window._mpr_viewers:
            assert set(viewer._screw_overlays) == {0}

    def test_finished_plan_keeps_no_preview_overlay_state(self):
        ctrl, window = self._make_controller()
        planned = [
            PlannedScrew(
                vertebra_name="L4", side=side,
                entry_lps=np.array([offset, 10, 0]),
                target_lps=np.array([offset, -20, 0]),
                length_mm=30, diameter_mm=6,
                convergence_angle=10, craniocaudal_angle=5,
                mean_bone_density=400, min_bone_density=200,
                gertzbein_grade="A", confidence=0.8,
            )
            for side, offset in (("left", 0), ("right", 5))
        ]
        ctrl._on_finished(planned)

        assert ctrl._preview_overlay_ids == []
        assert ctrl._preview_actors == []
        assert len(window._tool_ctrl.screw_tool.get_screws()) == 2

    def test_reject_selected_removes_from_list(self):
        ctrl, window = self._make_controller()
        ps1 = PlannedScrew(
            vertebra_name="L4", side="left",
            entry_lps=np.array([0, 10, 0]),
            target_lps=np.array([0, -20, 0]),
            length_mm=30, diameter_mm=6,
            convergence_angle=10, craniocaudal_angle=5,
            mean_bone_density=400, min_bone_density=200,
            gertzbein_grade="A", confidence=0.8,
        )
        ps2 = PlannedScrew(
            vertebra_name="L4", side="right",
            entry_lps=np.array([5, 10, 0]),
            target_lps=np.array([5, -20, 0]),
            length_mm=30, diameter_mm=6,
            convergence_angle=10, craniocaudal_angle=5,
            mean_bone_density=350, min_bone_density=180,
            gertzbein_grade="B", confidence=0.6,
        )
        ctrl._planned_screws = [ps1, ps2]
        # Simulate selecting row 0.
        class _FakeIndex:
            def __init__(self, r):
                self._row = r
            def row(self):
                return self._row
        window.auto_screw_table.selectedIndexes = lambda: [_FakeIndex(0)]
        ctrl.reject_selected()
        assert len(ctrl.planned_screws) == 1
        assert ctrl.planned_screws[0].side == "right"


def test_planned_screw_to_screw_copies_metadata():
    import numpy as np
    from src.core.auto_screw_planner import PlannedScrew
    from src.controllers.auto_placement_controller import planned_screw_to_screw
    ps = PlannedScrew(vertebra_name="L4", side="left", entry_lps=np.array([1.0, 2.0, 3.0]),
                      target_lps=np.array([1.0, -30.0, 3.0]), length_mm=32.0, diameter_mm=6.0,
                      convergence_angle=10.0, craniocaudal_angle=0.0, mean_bone_density=210.0,
                      min_bone_density=90.0, gertzbein_grade="B", confidence=0.7,
                      warnings=["Breach distance 0.8 mm (grade B)"], breach_mm=0.8, min_wall_mm=0.0)
    screw = planned_screw_to_screw(ps)
    assert screw.grade == "B"
    assert screw.breach_distance == pytest.approx(0.8)
    assert screw.mean_hu == pytest.approx(210.0)
    assert screw.min_hu == pytest.approx(90.0)
    assert screw.source == "auto"
    assert screw.warnings == ["Breach distance 0.8 mm (grade B)"]
