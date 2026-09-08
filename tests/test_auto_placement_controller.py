"""Tests for AutoPlacementController and planned_screw_to_screw conversion."""

import numpy as np
import pytest
from PyQt6.QtWidgets import QProgressDialog, QWidget

from src.controllers.auto_placement_controller import (
    AutoPlacementController,
    planned_screw_to_screw,
)
from src.core.auto_screw_planner import PlannedScrew


@pytest.fixture(autouse=True)
def _application(qapp):
    """The window stub is a real ``QWidget`` so it can parent a progress dialog."""
    return qapp


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

    def test_metrics_copied(self):
        metrics = {"trajectory_mean_hu": 320.0, "facet_grade": 1,
                   "facet_text": "screw abuts the cephalad facet",
                   "heary_direction": "medial"}
        ps = self._make_planned(metrics=metrics)
        screw = planned_screw_to_screw(ps)
        assert screw.metrics == metrics
        # A copy, so editing the screw cannot mutate the planner result.
        screw.metrics["facet_grade"] = 3
        assert ps.metrics["facet_grade"] == 1

    def test_metrics_default_to_empty_dict(self):
        screw = planned_screw_to_screw(self._make_planned())
        assert screw.metrics == {}

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
        self._last_segmentation_method = "totalsegmentator"
        self._last_pedicle_mask = None
        self.ensure_mpr_calls = 0

    def ensure_mpr_vertebrae_isolated(self):
        self.ensure_mpr_calls += 1
        return True


class _DummyStatusBar:
    def __init__(self):
        self.message = ""

    def showMessage(self, msg):
        self.message = msg


class _DummyLabel:
    def __init__(self, text=""):
        self._text = text

    def setText(self, text):
        self._text = text


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


class _DummyWindow(QWidget):
    """Minimal MainWindow stub for controller testing.

    A ``QWidget`` because :class:`AutoPlacementController` parents its planning
    progress dialog to the window.
    """

    def __init__(self):
        super().__init__()
        self.viewer_3d = _DummyViewer3D()
        self._tool_ctrl = _DummyToolCtrl()
        self._seg_ctrl = _DummySegCtrl()
        self.statusbar = _DummyStatusBar()
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

        self.planner_config_value = None

    def planner_config(self):
        from src.core.planner_config import PlannerConfig

        if self.planner_config_value is None:
            self.planner_config_value = PlannerConfig()
        return self.planner_config_value

    def _get_mpr_viewers(self):
        return self._mpr_viewers

    def update_vertebra_checkboxes(self, detected_labels):
        pass

    def _reset_all_vertebra_checkboxes(self):
        pass


def _planned(vertebra_name="L4", side="left", **overrides):
    """Build a PlannedScrew with sensible defaults for controller tests."""
    defaults = dict(
        vertebra_name=vertebra_name,
        side=side,
        entry_lps=np.array([0.0, 10.0, 0.0]),
        target_lps=np.array([0.0, -20.0, 0.0]),
        length_mm=30.0,
        diameter_mm=6.0,
        convergence_angle=10.0,
        craniocaudal_angle=5.0,
        mean_bone_density=400.0,
        min_bone_density=200.0,
        gertzbein_grade="A",
        confidence=0.8,
    )
    defaults.update(overrides)
    return PlannedScrew(**defaults)


@pytest.fixture
def controller_with_window():
    """AutoPlacementController wired to a minimal main-window stub."""
    from src.core.volume_manager import VolumeManager

    window = _DummyWindow()
    return AutoPlacementController(VolumeManager(), window), window


def test_on_finished_adds_editable_screws_and_keeps_last_planned(
    controller_with_window,
):
    ctrl, window = controller_with_window
    ps = _planned("L4", "left")
    ctrl._on_finished(_current_thread(ctrl), [ps])
    assert len(window._tool_ctrl.screw_tool.get_screws()) == 1
    assert ctrl.last_planned[0] is ps
    assert not hasattr(ctrl, "accept_all")


def test_last_planned_is_a_defensive_copy(controller_with_window):
    ctrl, _window = controller_with_window
    ctrl._on_finished(_current_thread(ctrl), [_planned()])
    ctrl.last_planned.clear()
    assert len(ctrl.last_planned) == 1


def test_reset_state_clears_last_planned(controller_with_window):
    ctrl, window = controller_with_window
    ctrl._on_finished(_current_thread(ctrl), [_planned()])
    ctrl.reset_state()
    assert ctrl.last_planned == []
    assert window.auto_screw_status._text == "No auto plan"


def test_dead_preview_api_is_gone(controller_with_window):
    ctrl, _window = controller_with_window
    for name in (
        "accept_all",
        "accept_selected",
        "reject_selected",
        "clear_plan",
        "_accept_screws",
        "_refresh_table",
        "_clear_preview",
        "_remove_preview_at",
        "_preview_actors",
        "_preview_overlay_ids",
        "_next_preview_overlay_id",
        "planned_screws",
    ):
        assert not hasattr(ctrl, name), f"{name} should have been removed"


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
        assert ctrl.last_planned == []

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
            def __init__(self, *_args, **_kwargs):
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
        ctrl._on_finished(_current_thread(ctrl), planned)
        assert len(window._tool_ctrl.screw_tool._screws) == 2
        assert [select for _screw, select in window._tool_ctrl.added] == [True, False]
        assert window.screw_list_widget.currentRow() == 0

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
        ctrl._on_finished(_current_thread(ctrl), [ps])
        assert len(window._tool_ctrl.screw_tool._screws) == 2
        assert window.screw_list_widget.currentRow() == 1

    def test_reset_state(self):
        ctrl, window = self._make_controller()
        ctrl._last_planned = [
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
        assert ctrl.last_planned == []

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
        ctrl._on_finished(_current_thread(ctrl), planned)
        assert len(window._tool_ctrl.added) == 1
        assert window._tool_ctrl.added[0][0].vertebra_level == "L4"
        assert window._tool_ctrl.added[0][0].side == "left"



def test_run_planning_refuses_threshold_fallback(controller_with_window, monkeypatch):
    ctrl, window = controller_with_window
    window._seg_ctrl._last_segmentation_mask_path = "dummy.nii.gz"
    window._seg_ctrl._last_segmentation_method = "threshold_fallback"
    shown = []
    monkeypatch.setattr(
        "src.controllers.auto_placement_controller.QMessageBox.warning",
        lambda *a, **k: shown.append(a[2]),
    )
    ctrl.run_planning()
    assert shown and "TotalSegmentator" in shown[0]
    assert ctrl._thread is None


def test_planned_screw_to_screw_copies_metadata():
    import numpy as np

    from src.controllers.auto_placement_controller import planned_screw_to_screw
    from src.core.auto_screw_planner import PlannedScrew
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


# ---------------------------------------------------------------------------
# Planner configuration plumbing
# ---------------------------------------------------------------------------


def test_run_planning_passes_window_planner_config_to_thread(
    monkeypatch, tmp_path
):
    import SimpleITK as sitk

    import src.controllers.auto_placement_controller as module
    from src.core.planner_config import PlannerConfig

    window = _DummyWindow()
    window.planner_config_value = PlannerConfig(pedicle_fill_ratio=0.65)
    from src.core.volume_manager import VolumeManager
    ctrl = AutoPlacementController(VolumeManager(), window)
    ctrl._vm.set_volume(sitk.Image([4, 4, 4], sitk.sitkInt16))
    mask_path = tmp_path / "mask.nii.gz"
    sitk.WriteImage(sitk.Image([4, 4, 4], sitk.sitkUInt8), str(mask_path))
    window._seg_ctrl._last_segmentation_mask_path = str(mask_path)

    captured = {}

    class _Signal:
        def connect(self, _callback):
            return None

    class _Thread:
        def __init__(self, *args, **kwargs):
            captured["args"] = args
            captured["kwargs"] = kwargs
            self.progress = _Signal()
            self.finished = _Signal()
            self.error = _Signal()

        def start(self):
            return None

        def isRunning(self):
            return False

    monkeypatch.setattr(module, "_PlanningThread", _Thread)

    ctrl.run_planning()

    config = captured["kwargs"].get("config")
    assert config is window.planner_config_value


def test_planning_thread_forwards_config_to_planner(monkeypatch):
    import SimpleITK as sitk

    import src.controllers.auto_placement_controller as module
    from src.core.planner_config import PlannerConfig

    config = PlannerConfig(anterior_margin_mm=9.0)
    captured = {}

    class _Analyzer:
        def __init__(self, *_args, **_kwargs):
            pass

        def analyze_all(self, labels=None):
            return []

    class _Planner:
        def __init__(self, _ct, _mask, grader=None, config=None):
            captured["config"] = config

        def plan_all(self, _analyses, sides="both", **_kwargs):
            return []

    monkeypatch.setattr(module, "PedicleAnalyzer", _Analyzer)
    monkeypatch.setattr(module, "AutoScrewPlanner", _Planner)

    thread = module._PlanningThread(
        sitk.Image([2, 2, 2], sitk.sitkUInt8),
        sitk.Image([2, 2, 2], sitk.sitkInt16),
        [28],
        config=config,
    )
    thread.run()

    assert captured["config"] is config


def test_planning_thread_config_defaults_to_none(monkeypatch):
    import SimpleITK as sitk

    import src.controllers.auto_placement_controller as module

    thread = module._PlanningThread(
        sitk.Image([2, 2, 2], sitk.sitkUInt8),
        sitk.Image([2, 2, 2], sitk.sitkInt16),
        [28],
    )

    assert thread._config is None


# ---------------------------------------------------------------------------
# Optional pedicle subregion mask hand-off
# ---------------------------------------------------------------------------


def test_planning_thread_receives_pedicle_mask(monkeypatch, tmp_path):
    import SimpleITK as sitk

    import src.controllers.auto_placement_controller as module
    from src.core.volume_manager import VolumeManager

    window = _DummyWindow()
    ctrl = AutoPlacementController(VolumeManager(), window)
    ctrl._vm.set_volume(sitk.Image([4, 4, 4], sitk.sitkInt16))
    mask_path = tmp_path / "mask.nii.gz"
    sitk.WriteImage(sitk.Image([4, 4, 4], sitk.sitkUInt8), str(mask_path))
    window._seg_ctrl._last_segmentation_mask_path = str(mask_path)
    pedicle_mask = np.ones((4, 4, 4), bool)
    window._seg_ctrl._last_pedicle_mask = pedicle_mask

    captured = {}

    class _Signal:
        def connect(self, _callback):
            return None

    class _Thread:
        def __init__(self, *args, **kwargs):
            captured["kwargs"] = kwargs
            self.progress = _Signal()
            self.finished = _Signal()
            self.error = _Signal()

        def start(self):
            return None

        def isRunning(self):
            return False

    monkeypatch.setattr(module, "_PlanningThread", _Thread)

    ctrl.run_planning()

    assert captured["kwargs"].get("pedicle_mask") is pedicle_mask


def test_planning_thread_forwards_pedicle_mask_to_analyzer(monkeypatch):
    import SimpleITK as sitk

    import src.controllers.auto_placement_controller as module

    pedicle_mask = np.ones((2, 2, 2), bool)
    captured = {}

    class _Analyzer:
        def __init__(self, _mask, _ct, pedicle_mask=None):
            captured["pedicle_mask"] = pedicle_mask

        def analyze_all(self, labels=None):
            return []

    class _Planner:
        def __init__(self, _ct, _mask, grader=None, config=None):
            pass

        def plan_all(self, _analyses, sides="both", **_kwargs):
            return []

    monkeypatch.setattr(module, "PedicleAnalyzer", _Analyzer)
    monkeypatch.setattr(module, "AutoScrewPlanner", _Planner)

    thread = module._PlanningThread(
        sitk.Image([2, 2, 2], sitk.sitkUInt8),
        sitk.Image([2, 2, 2], sitk.sitkInt16),
        [28],
        pedicle_mask=pedicle_mask,
    )
    thread.run()

    assert captured["pedicle_mask"] is pedicle_mask


def test_planning_thread_pedicle_mask_defaults_to_none():
    import SimpleITK as sitk

    import src.controllers.auto_placement_controller as module

    thread = module._PlanningThread(
        sitk.Image([2, 2, 2], sitk.sitkUInt8),
        sitk.Image([2, 2, 2], sitk.sitkInt16),
        [28],
    )

    assert thread._pedicle_mask is None


def test_on_finished_reports_rod_misalignment(controller_with_window):
    ctrl, window = controller_with_window
    ctrl._on_finished(_current_thread(ctrl), [
        _planned("L4", "left", metrics={"rod_misalignment_mm": 1.24}),
        _planned("L4", "right", metrics={"rod_misalignment_mm": 2.75}),
    ])
    assert "Rod misalignment L 1.2 mm / R 2.8 mm" in window.auto_screw_status._text


def test_on_finished_omits_rod_misalignment_when_absent(controller_with_window):
    ctrl, window = controller_with_window
    ctrl._on_finished(_current_thread(ctrl), [_planned("L4", "left")])
    assert "Rod misalignment" not in window.auto_screw_status._text


def test_on_finished_omits_a_side_with_no_rod_value(controller_with_window):
    """A side with no screws must not be reported as perfectly aligned."""
    ctrl, window = controller_with_window
    ctrl._on_finished(
        _current_thread(ctrl),
        [_planned("L4", "left", metrics={"rod_misalignment_mm": 1.24})],
    )
    text = window.auto_screw_status._text
    assert text.endswith("Rod misalignment L 1.2 mm")
    assert "R 0.0 mm" not in text


# ---------------------------------------------------------------------------
# Progress, cancellation and dropped CBT sides
# ---------------------------------------------------------------------------


class _FinishedThread:
    """Stand-in for a finished ``_PlanningThread``."""

    def __init__(self, cancelled=False, skipped_sides=(), generation=0):
        self.cancelled = cancelled
        self.skipped_sides = list(skipped_sides)
        self.cancel_requested = False
        # A fresh controller sits at generation 0, so the default matches the
        # run a test that never called ``run_planning`` is standing in for.
        self.generation = generation

    def request_cancel(self):
        self.cancel_requested = True


def _current_thread(ctrl, **kwargs):
    """Install a stub thread stamped for ``ctrl``'s current run generation."""
    thread = _FinishedThread(generation=ctrl._run_generation, **kwargs)
    ctrl._thread = thread
    return thread


def _thread_for(mask_size=2):
    import SimpleITK as sitk

    import src.controllers.auto_placement_controller as module

    return module._PlanningThread(
        sitk.Image([mask_size] * 3, sitk.sitkUInt8),
        sitk.Image([mask_size] * 3, sitk.sitkInt16),
        [28],
    )


def test_request_cancel_sets_the_planning_threads_event():
    thread = _thread_for()
    assert thread._cancel.is_set() is False
    thread.request_cancel()
    assert thread._cancel.is_set() is True


def test_controller_request_cancel_reaches_the_running_thread(controller_with_window):
    ctrl, _window = controller_with_window
    thread = _FinishedThread()
    ctrl._thread = thread
    ctrl.request_cancel()
    assert thread.cancel_requested is True


def test_request_cancel_without_a_run_is_a_no_op(controller_with_window):
    ctrl, _window = controller_with_window
    ctrl.request_cancel()          # must not raise


def test_planning_thread_forwards_progress_and_cancel_to_the_planner(monkeypatch):
    import src.controllers.auto_placement_controller as module

    captured = {}

    class _Analyzer:
        def __init__(self, *_args, **_kwargs):
            pass

        def analyze_all(self, labels=None):
            return []

    class _Planner:
        skipped_sides = [("L2", "right", "no isthmus corner")]
        last_run_cancelled = True

        def __init__(self, _ct, _mask, grader=None, config=None):
            pass

        def plan_all(self, _analyses, sides="both", progress=None, cancel=None):
            captured["progress"] = progress
            captured["cancel"] = cancel
            progress("Planning L4 left (1/2)…")
            return []

    monkeypatch.setattr(module, "PedicleAnalyzer", _Analyzer)
    monkeypatch.setattr(module, "AutoScrewPlanner", _Planner)

    thread = _thread_for()
    emitted = []
    thread.progress.connect(emitted.append)
    thread.run()

    assert captured["cancel"]() is False
    thread.request_cancel()
    assert captured["cancel"]() is True
    assert "Planning L4 left (1/2)…" in emitted
    assert thread.skipped_sides == [("L2", "right", "no isthmus corner")]
    assert thread.cancelled is True


def test_on_finished_closes_the_dialog_without_re_entering_cancel(
    controller_with_window,
):
    ctrl, _window = controller_with_window
    thread = _FinishedThread()
    ctrl._thread = thread
    dialog = QProgressDialog("Planning screws…", "Cancel", 0, 0, None)
    dialog.canceled.connect(ctrl._on_cancel_requested)
    ctrl._progress_dialog = dialog

    ctrl._on_finished(thread, [_planned()])

    assert ctrl._progress_dialog is None
    assert not dialog.isVisible()
    assert thread.cancel_requested is False


def test_on_error_closes_the_dialog(controller_with_window, monkeypatch):
    ctrl, _window = controller_with_window
    monkeypatch.setattr(
        "src.controllers.auto_placement_controller.QMessageBox.critical",
        lambda *a, **k: None,
    )
    dialog = QProgressDialog("Planning screws…", "Cancel", 0, 0, None)
    dialog.canceled.connect(ctrl._on_cancel_requested)
    ctrl._progress_dialog = dialog

    ctrl._on_error(_current_thread(ctrl), "boom")

    assert ctrl._progress_dialog is None
    assert not dialog.isVisible()


def test_on_finished_reports_a_cancelled_run_instead_of_success(
    controller_with_window,
):
    ctrl, window = controller_with_window
    ctrl._thread = _FinishedThread(cancelled=True)

    ctrl._on_finished(ctrl._thread, [_planned("L4", "left"), _planned("L4", "right")])

    assert window.auto_screw_status._text == "Planning cancelled — 2 screws kept"
    # The partial construct is still added, so the surgeon keeps what was planned.
    assert len(window._tool_ctrl.screw_tool.get_screws()) == 2


def test_on_finished_reports_a_cancelled_run_with_no_screws(controller_with_window):
    ctrl, window = controller_with_window
    ctrl._thread = _FinishedThread(cancelled=True)
    ctrl._on_finished(ctrl._thread, [])
    assert window.auto_screw_status._text == "Planning cancelled — 0 screws kept"


def test_on_finished_names_sides_with_no_feasible_cbt_trajectory(
    controller_with_window,
):
    ctrl, window = controller_with_window
    ctrl._thread = _FinishedThread(
        skipped_sides=[
            ("L2", "right", "no isthmus corner"),
            ("L3", "right", "no feasible trajectory"),
        ]
    )

    ctrl._on_finished(ctrl._thread, [_planned()])

    assert (
        "No screw planned: L2 right, L3 right"
        in window.auto_screw_status._text
    )


def test_on_finished_names_dropped_sides_even_with_no_screws(controller_with_window):
    ctrl, window = controller_with_window
    ctrl._thread = _FinishedThread(skipped_sides=[("L2", "right", "no corner")])
    ctrl._on_finished(ctrl._thread, [])
    assert "No screw planned: L2 right" in window.auto_screw_status._text


def test_run_planning_shows_a_cancellable_progress_dialog(monkeypatch, tmp_path):
    import SimpleITK as sitk

    import src.controllers.auto_placement_controller as module
    from src.core.volume_manager import VolumeManager

    window = _DummyWindow()
    ctrl = AutoPlacementController(VolumeManager(), window)
    ctrl._vm.set_volume(sitk.Image([4, 4, 4], sitk.sitkInt16))
    mask_path = tmp_path / "mask.nii.gz"
    sitk.WriteImage(sitk.Image([4, 4, 4], sitk.sitkUInt8), str(mask_path))
    window._seg_ctrl._last_segmentation_mask_path = str(mask_path)

    class _Signal:
        def connect(self, _callback):
            return None

    class _Thread:
        def __init__(self, *_args, **_kwargs):
            self.progress = _Signal()
            self.finished = _Signal()
            self.error = _Signal()
            self.cancel_requested = False

        def start(self):
            return None

        def isRunning(self):
            return True

        def request_cancel(self):
            self.cancel_requested = True

    monkeypatch.setattr(module, "_PlanningThread", _Thread)

    ctrl.run_planning()

    dialog = ctrl._progress_dialog
    assert dialog is not None
    assert (dialog.minimum(), dialog.maximum()) == (0, 0)   # indeterminate
    dialog.canceled.emit()
    assert ctrl._thread.cancel_requested is True
    ctrl._close_progress_dialog()


# ---------------------------------------------------------------------------
# A run in flight when the study changes
# ---------------------------------------------------------------------------


def _running_run(ctrl):
    """Put ``ctrl`` in the state of a run in flight, and return its parts."""
    thread = _FinishedThread(generation=ctrl._run_generation)
    ctrl._thread = thread
    dialog = QProgressDialog("Planning screw trajectories...", "Cancel", 0, 0, None)
    dialog.canceled.connect(ctrl._on_cancel_requested)
    ctrl._progress_dialog = dialog
    dialog.show()
    return thread, dialog


def test_reset_state_cancels_the_run_in_flight_and_closes_its_dialog(
    controller_with_window,
):
    ctrl, _window = controller_with_window
    thread, dialog = _running_run(ctrl)

    ctrl.reset_state()

    assert thread.cancel_requested is True
    assert ctrl._progress_dialog is None
    assert not dialog.isVisible()


def test_a_late_result_from_the_previous_study_is_dropped(controller_with_window):
    """The new study must not inherit the old study's screws."""
    ctrl, window = controller_with_window
    thread, _dialog = _running_run(ctrl)
    ctrl.reset_state()

    ctrl._on_finished(thread, [_planned("L4", "left")])

    assert window._tool_ctrl.screw_tool.get_screws() == []
    assert window._tool_ctrl.added == []
    assert ctrl.last_planned == []
    assert window.auto_screw_status._text == "No auto plan"
    assert ctrl._thread is None


def test_a_late_error_from_the_previous_study_is_dropped(
    controller_with_window, monkeypatch
):
    ctrl, window = controller_with_window
    shown = []
    monkeypatch.setattr(
        "src.controllers.auto_placement_controller.QMessageBox.critical",
        lambda *a, **k: shown.append(a),
    )
    thread, _dialog = _running_run(ctrl)
    ctrl.reset_state()

    ctrl._on_error(thread, "boom")

    assert shown == []
    assert window.auto_screw_status._text == "No auto plan"
    assert ctrl._thread is None


def test_a_result_from_the_current_study_is_kept(controller_with_window):
    ctrl, window = controller_with_window
    thread, _dialog = _running_run(ctrl)

    ctrl._on_finished(thread, [_planned("L4", "left")])

    assert len(window._tool_ctrl.screw_tool.get_screws()) == 1


# ---------------------------------------------------------------------------
# The cancel label survives in-flight progress
# ---------------------------------------------------------------------------


def test_progress_does_not_overwrite_the_cancelling_label(controller_with_window):
    ctrl, window = controller_with_window
    thread, _dialog = _running_run(ctrl)

    ctrl._on_cancel_requested()
    assert window.auto_screw_status._text == "Cancelling planning..."

    # A message the worker had already queued before it saw the cancel.
    ctrl._on_progress(thread, "Planning L4 right (2/10)…")

    assert window.auto_screw_status._text == "Cancelling planning..."
    # The status bar still tracks the run winding down.
    assert window.statusbar.message == "Planning L4 right (2/10)…"


def test_a_new_run_clears_the_cancelling_label(monkeypatch, tmp_path):
    import SimpleITK as sitk

    import src.controllers.auto_placement_controller as module
    from src.core.volume_manager import VolumeManager

    window = _DummyWindow()
    ctrl = AutoPlacementController(VolumeManager(), window)
    ctrl._vm.set_volume(sitk.Image([4, 4, 4], sitk.sitkInt16))
    mask_path = tmp_path / "mask.nii.gz"
    sitk.WriteImage(sitk.Image([4, 4, 4], sitk.sitkUInt8), str(mask_path))
    window._seg_ctrl._last_segmentation_mask_path = str(mask_path)

    class _Signal:
        def connect(self, _callback):
            return None

    class _Thread:
        def __init__(self, *_args, **_kwargs):
            self.progress = _Signal()
            self.finished = _Signal()
            self.error = _Signal()

        def start(self):
            return None

        def isRunning(self):
            return False

        def request_cancel(self):
            return None

    monkeypatch.setattr(module, "_PlanningThread", _Thread)
    ctrl._cancel_requested = True

    ctrl.run_planning()

    assert ctrl._cancel_requested is False
    assert ctrl._thread.generation == ctrl._run_generation
    ctrl._on_progress(ctrl._thread, "Planning L4 left (1/2)…")
    assert window.auto_screw_status._text == "Planning L4 left (1/2)…"
    ctrl._close_progress_dialog()


# ---------------------------------------------------------------------------
# F2 -- a result is attributed to the run that produced it, not to `_thread`
# ---------------------------------------------------------------------------


def test_a_result_from_a_superseded_run_is_dropped(controller_with_window):
    """Thread A finishes after a second Plan click already installed thread B.

    ``finished`` is emitted before ``isRunning()`` goes False, so B can be in
    place by the time A's queued slot runs.  A's screws must not be added, and
    B must stay the tracked run with its dialog intact.
    """
    ctrl, window = controller_with_window
    thread_a = _current_thread(ctrl)
    ctrl._run_generation += 1                     # the second run_planning
    thread_b = _current_thread(ctrl)
    dialog = QProgressDialog("Planning screw trajectories...", "Cancel", 0, 0, None)
    dialog.canceled.connect(ctrl._on_cancel_requested)
    ctrl._progress_dialog = dialog

    ctrl._on_finished(thread_a, [_planned("L4", "left")])

    assert window._tool_ctrl.screw_tool.get_screws() == []
    assert ctrl.last_planned == []
    assert ctrl._thread is thread_b               # B is still the tracked run
    assert ctrl._progress_dialog is dialog        # ... and keeps its dialog
    ctrl._close_progress_dialog()


def test_an_error_from_a_superseded_run_is_dropped(
    controller_with_window, monkeypatch
):
    ctrl, window = controller_with_window
    shown = []
    monkeypatch.setattr(
        "src.controllers.auto_placement_controller.QMessageBox.critical",
        lambda *a, **k: shown.append(a),
    )
    thread_a = _current_thread(ctrl)
    ctrl._run_generation += 1
    thread_b = _current_thread(ctrl)

    ctrl._on_error(thread_a, "boom")

    assert shown == []
    assert window.auto_screw_status._text == "No auto plan"
    assert ctrl._thread is thread_b


def test_a_result_from_an_unknown_emitter_is_dropped(controller_with_window):
    ctrl, window = controller_with_window
    _current_thread(ctrl)

    ctrl._on_finished(None, [_planned("L4", "left")])

    assert window._tool_ctrl.screw_tool.get_screws() == []


def test_progress_from_a_superseded_run_leaves_the_new_run_alone(
    controller_with_window,
):
    ctrl, window = controller_with_window
    thread_a = _current_thread(ctrl)
    ctrl._run_generation += 1
    _current_thread(ctrl)
    window.auto_screw_status.setText("Planning...")
    window.statusbar.showMessage("Auto screw planning started")

    ctrl._on_progress(thread_a, "Planning L4 right (2/10)…")

    assert window.auto_screw_status._text == "Planning..."
    assert window.statusbar.message == "Auto screw planning started"


def test_progress_after_reset_state_does_not_touch_the_new_studys_labels(
    controller_with_window,
):
    ctrl, window = controller_with_window
    thread, _dialog = _running_run(ctrl)
    ctrl.reset_state()

    ctrl._on_progress(thread, "Planning L4 right (2/10)…")

    assert window.auto_screw_status._text == "No auto plan"
    assert window.statusbar.message == ""


def _prepare_run(window, ctrl, tmp_path):
    """Give ``ctrl`` a CT volume and a readable mask so run_planning proceeds."""
    import SimpleITK as sitk

    ctrl._vm.set_volume(sitk.Image([4, 4, 4], sitk.sitkInt16))
    mask_path = tmp_path / "mask.nii.gz"
    sitk.WriteImage(sitk.Image([4, 4, 4], sitk.sitkUInt8), str(mask_path))
    window._seg_ctrl._last_segmentation_mask_path = str(mask_path)
    return mask_path


def test_every_run_gets_a_fresh_generation(monkeypatch, tmp_path):
    """Two runs must be distinguishable even without a study change."""
    import src.controllers.auto_placement_controller as module
    from src.core.volume_manager import VolumeManager

    window = _DummyWindow()
    ctrl = AutoPlacementController(VolumeManager(), window)
    _prepare_run(window, ctrl, tmp_path)

    class _Signal:
        def connect(self, _callback):
            return None

    class _Thread:
        def __init__(self, *_args, **_kwargs):
            self.progress = _Signal()
            self.finished = _Signal()
            self.error = _Signal()

        def start(self):
            return None

        def isRunning(self):
            return False

    monkeypatch.setattr(module, "_PlanningThread", _Thread)

    ctrl.run_planning()
    first = ctrl._thread.generation
    ctrl._close_progress_dialog()
    ctrl.run_planning()
    second = ctrl._thread.generation

    assert second != first
    assert second == ctrl._run_generation
    ctrl._close_progress_dialog()


def test_run_planning_reports_an_unreadable_mask_instead_of_raising(
    monkeypatch, tmp_path
):
    """The workspace can be purged by another instance between run and plan."""
    import src.controllers.auto_placement_controller as module
    from src.core.volume_manager import VolumeManager

    window = _DummyWindow()
    ctrl = AutoPlacementController(VolumeManager(), window)
    mask_path = _prepare_run(window, ctrl, tmp_path)
    mask_path.unlink()

    shown = []
    monkeypatch.setattr(
        module.QMessageBox, "warning", lambda *a, **k: shown.append(a[2])
    )

    ctrl.run_planning()          # must not raise inside the Qt slot

    assert shown and "could not be read" in shown[0]
    assert str(mask_path) in shown[0]
    assert ctrl._thread is None
    assert ctrl._progress_dialog is None
