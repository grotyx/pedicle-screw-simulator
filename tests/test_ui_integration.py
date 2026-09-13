"""
UI integration tests for MainWindow workflow.
"""

import json
import os
import sys
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("pytestqt")
sitk = pytest.importorskip("SimpleITK")

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from PyQt6 import sip
from PyQt6.QtCore import QCoreApplication, QEvent, QObject, Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QApplication,
    QMessageBox,
    QProgressDialog,
    QPushButton,
    QWidget,
)

import src.controllers.plan_controller as plan_controller_module
import src.controllers.segmentation_controller as seg_controller_module
import src.ui.main_window as main_window_module
from src.core.totalseg_integration import SegmentationRunResult
from src.models.screw import Screw


class DummyMPRViewer(QWidget):
    """Lightweight MPR test double for UI workflow tests."""

    slice_changed = pyqtSignal(str, float)
    crosshair_moved = pyqtSignal(str, float, float, float)
    header_double_clicked = pyqtSignal(str)

    def __init__(self, plane, volume_manager, parent=None):
        super().__init__(parent)
        self.plane = plane
        self.volume_manager = volume_manager
        self.visible = True
        self.seg_label_value = 0
        self.seg_mask = None
        self.measurements = {}
        self.custom_axes = None
        self.custom_title = None
        self.custom_readout = None
        self.custom_scroll_handler = None
        self.custom_rotate_handler = None
        self.review_screw_id = None
        self.screw_overlays = {}
        self.fit_count = 0
        self.screw_interaction_callbacks = {}
        self.measurement_interaction_callbacks = {}
        self.selected_measurement_id = None
        self.screw_interaction_cancelled = 0
        self.last_slice_position = None
        self.orientation_refresh_count = 0

    def set_window_level(self, _window, _level):
        return

    def set_segmentation_mask(self, mask_image, label_value=0, **_kwargs):
        self.seg_mask = mask_image
        self.seg_label_value = int(label_value)

    def set_segmentation_label(self, label_value):
        self.seg_label_value = int(label_value)

    def set_segmentation_visible(self, visible):
        self.visible = bool(visible)

    def clear_segmentation_mask(self):
        self.seg_mask = None
        self.seg_label_value = 0

    def add_measurement(self, measurement_id, points, label):
        self.measurements[measurement_id] = {"points": points, "label": label}

    def remove_measurement(self, measurement_id):
        self.measurements.pop(measurement_id, None)

    def clear_measurements(self):
        self.measurements = {}

    def set_slice_position(self, position):
        self.last_slice_position = float(position)
        self.volume_manager.set_slice_position(self.plane, float(position))

    def cleanup(self):
        return

    # Stubs for reslice input swap (vertebral isolation)
    def set_reslice_input(self, vtk_image):
        return

    def restore_original_input(self):
        return

    # Stubs for screw projection overlays
    def add_screw_overlay(self, screw_id, entry, target, color=(0.2, 0.8, 0.2), diameter=6.0):
        self.screw_overlays[screw_id] = {
            "entry": tuple(entry),
            "target": tuple(target),
            "diameter": float(diameter),
        }

    def remove_screw_overlay(self, screw_id):
        self.screw_overlays.pop(screw_id, None)

    def clear_screw_overlays(self):
        self.screw_overlays = {}

    def set_custom_reslice_axes(self, axes, title):
        self.custom_axes = axes
        self.custom_title = title

    def set_custom_readout(self, text):
        self.custom_readout = text

    def set_custom_scroll_handler(self, handler):
        self.custom_scroll_handler = handler

    def set_custom_rotate_handler(self, handler):
        self.custom_rotate_handler = handler

    def clear_custom_reslice_axes(self):
        self.custom_axes = None
        self.custom_title = None
        self.custom_readout = None

    def set_review_screw(self, screw_id):
        self.review_screw_id = screw_id

    def set_screw_interaction_callbacks(self, **callbacks):
        self.screw_interaction_callbacks = callbacks

    def set_measurement_interaction_callbacks(self, **callbacks):
        self.measurement_interaction_callbacks = callbacks

    def set_selected_measurement(self, measurement_id):
        self.selected_measurement_id = measurement_id

    def cancel_screw_interaction(self):
        self.screw_interaction_cancelled += 1

    def fit_to_view(self):
        self.fit_count += 1

    def refresh_orientation_markers(self, render=False):
        self.orientation_refresh_count += 1

    # Stubs for _coordinated_initial_render
    _render_guard_active = False
    _settled = False
    _extra_render_count = 0

    def _deferred_initial_render(self):
        return


class DummyViewer3D(QWidget):
    """Lightweight 3D viewer test double for UI workflow tests."""

    header_double_clicked = pyqtSignal(str)
    isolation_requested = pyqtSignal(bool)

    def __init__(self, volume_manager, parent=None):
        super().__init__(parent)
        self.visible = True
        self.seg_label_value = 0
        self.seg_mask = None
        self.screws = []
        self.measurements = {}
        self.zoom_factors = []
        self.fit_count = 0
        self.vertebral_mesh_labels = None
        self.volume_visible = True
        self.screw_interaction_cancelled = 0
        # Screw MPR in 3D: the last planes shown, or None once cleared.
        self.screw_mpr_planes = None
        self.screw_mpr_clear_count = 0
        self.screw_mpr_label = None
        self.screw_mpr_window_level = None
        # (isolated, available) as last reported by set_isolation_state.
        self.isolation_state = (False, False)

    def set_isolation_state(self, isolated: bool, available: bool) -> None:
        self.isolation_state = (bool(isolated), bool(available))

    def show_screw_mpr(
        self,
        oblique_axial,
        oblique_sagittal,
        cross_section,
        *,
        vertebra_label=None,
        window_level=None,
    ):
        self.screw_mpr_planes = (oblique_axial, oblique_sagittal, cross_section)
        self.screw_mpr_label = vertebra_label
        self.screw_mpr_window_level = window_level

    def clear_screw_mpr(self):
        self.screw_mpr_planes = None
        self.screw_mpr_clear_count += 1

    def add_screw(
        self, entry_point, target_point, radius=3.0, color=None, screw_id=None
    ):
        class _Property:
            def SetOpacity(self, v): pass
            def GetOpacity(self): return 1.0
        class _Actor:
            def __init__(self, e, t, r, c, sid):
                self.entry_point = e
                self.target_point = t
                self.radius = r
                self.color = c
                self.screw_id = sid
                self._prop = _Property()
            def GetProperty(self):
                return self._prop
        actor = _Actor(
            tuple(entry_point), tuple(target_point), float(radius), color, screw_id
        )
        self.screws.append(actor)
        return actor

    def set_screw_interaction_callbacks(self, **callbacks):
        self.screw_interaction_callbacks = callbacks

    def set_selected_screw(self, screw_id):
        self.selected_screw_id = screw_id

    def cancel_screw_interaction(self):
        self.screw_interaction_cancelled += 1

    def remove_screw(self, actor):
        if actor in self.screws:
            self.screws.remove(actor)

    def clear_screws(self):
        self.screws = []

    def add_measurement(self, measurement_id, points, label):
        self.measurements[measurement_id] = {"points": points, "label": label}

    def remove_measurement(self, measurement_id):
        self.measurements.pop(measurement_id, None)

    def clear_measurements(self):
        self.measurements = {}

    def set_segmentation_mask(self, mask_image, label_value=0, **_kwargs):
        self.seg_mask = mask_image
        self.seg_label_value = int(label_value)

    def set_segmentation_label(self, label_value):
        self.seg_label_value = int(label_value)

    def set_segmentation_visible(self, visible):
        self.visible = bool(visible)

    # Stubs for _coordinated_initial_render
    _render_guard_active = False
    _extra_render_count = 0

    def _deferred_render_phase1(self):
        return

    def clear_segmentation_mask(self):
        self.seg_mask = None
        self.seg_label_value = 0

    def set_vertebral_mesh(self, _mask_image, labels=None):
        self.vertebral_mesh_labels = list(labels) if labels is not None else None

    def clear_vertebral_mesh(self):
        return

    def set_volume_visible(self, visible):
        self.volume_visible = bool(visible)

    def set_bone_opacity(self, _opacity):
        return

    def _update_plane_positions(self):
        return

    def zoom_camera(self, factor):
        self.zoom_factors.append(float(factor))

    def fit_to_view(self):
        self.fit_count += 1

    def cleanup(self):
        return


class _ProgressStub:
    """Simple close-only progress stub for private load callback tests."""

    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


def _create_test_image():
    image = sitk.Image([16, 16, 12], sitk.sitkInt16)
    image = image + 250
    image.SetSpacing((1.0, 1.0, 1.2))
    image.SetOrigin((0.5, 1.0, 2.0))
    return image


def _write_mask(image, path):
    mask = sitk.Cast(image > 0, sitk.sitkUInt8)
    mask.CopyInformation(image)
    sitk.WriteImage(mask, str(path))


def _write_vertebra_mask(image, path, label=28):
    """A mask carrying one real vertebra label (28 = L4), for the isolation
    tests: a plain thresholded mask (see _write_mask) has no label
    TotalSegmentator would ever emit, so ``_on_finished`` never treats it as
    having detected vertebrae and never auto-isolates on it.
    """
    mask = sitk.Cast(image > 0, sitk.sitkUInt8) * int(label)
    mask.CopyInformation(image)
    sitk.WriteImage(mask, str(path))


@pytest.fixture
def isolated_qsettings(tmp_path, monkeypatch):
    """Redirect MainWindow's QSettings into temp INI files.

    Returns the factory the window uses, so tests can read back the very
    same store without touching the developer's real settings.
    """
    from PyQt6.QtCore import QSettings

    directory = tmp_path / "qsettings"
    directory.mkdir(parents=True, exist_ok=True)

    def factory(organization="Default", application="App", *_args, **_kwargs):
        path = directory / f"{organization}-{application}.ini"
        return QSettings(str(path), QSettings.Format.IniFormat)

    monkeypatch.setattr(main_window_module, "QSettings", factory)
    return factory


@pytest.fixture
def ui_main_window(monkeypatch, qtbot, isolated_qsettings):
    """Build MainWindow with lightweight viewer stubs."""
    monkeypatch.setattr(main_window_module, "MPRViewer", DummyMPRViewer)
    monkeypatch.setattr(main_window_module, "Viewer3D", DummyViewer3D)
    QApplication.instance().setProperty("themeName", "soft_light")

    window = main_window_module.MainWindow()
    qtbot.addWidget(window)
    return window


def _view_grid_position(window, widget):
    index = window._view_layout.indexOf(widget)
    assert index >= 0
    return window._view_layout.getItemPosition(index)


def test_vworks_planning_layout_is_default(ui_main_window):
    window = ui_main_window

    assert window._view_layout_mode == "planning"
    assert _view_grid_position(window, window.axial_viewer) == (0, 0, 1, 1)
    assert _view_grid_position(window, window.sagittal_viewer) == (1, 0, 1, 1)
    assert _view_grid_position(window, window.coronal_viewer) == (2, 0, 1, 1)
    assert _view_grid_position(window, window.viewer_3d) == (0, 1, 3, 1)


def test_default_window_size_is_screen_aware_and_compact(ui_main_window):
    window = ui_main_window

    assert window.minimumWidth() == 1280
    assert window.minimumHeight() == 760
    assert window._calculate_initial_window_size(1920, 1080) == (1680, 1050)
    assert window._calculate_initial_window_size(1440, 900) == (1353, 900)


def test_workspace_and_theme_selectors_are_compact(ui_main_window):
    window = ui_main_window

    assert window.workspace_mode_combo.maximumWidth() <= 105
    assert window.theme_combo.maximumWidth() <= 115
    assert window.workspace_mode_combo.maximumHeight() <= 26
    assert window.theme_combo.maximumHeight() <= 26


def test_help_menu_exposes_about_credit(ui_main_window):
    help_action = next(
        action
        for action in ui_main_window.menuBar().actions()
        if action.text() == "Help"
    )
    about_actions = [action.text() for action in help_action.menu().actions()]

    assert "About Pedicle Screw Simulator" in about_actions


def test_default_control_panel_expands_only_main_workflow(ui_main_window):
    window = ui_main_window

    assert window.segmentation_group.is_collapsed is False
    assert window.planning_group.is_collapsed is False
    assert window.selected_screw_group.property("role") == "review"
    assert all(
        group.is_collapsed
        for group in window.secondary_control_groups
    )
    assert not hasattr(window, "auto_accept_all_btn")
    assert not hasattr(window, "auto_accept_sel_btn")
    assert not hasattr(window, "auto_reject_btn")
    assert not hasattr(window, "auto_screw_table")
    assert window.screw_list_widget.minimumHeight() >= 110
    assert window.screw_list_widget.maximumHeight() >= 110
    assert window.remove_screw_btn.text() == "Delete Screw"


def test_layout_can_switch_to_mpr_focus_and_back(ui_main_window):
    window = ui_main_window

    window.set_view_layout("mpr_focus")

    assert window._view_layout_mode == "mpr_focus"
    assert _view_grid_position(window, window.axial_viewer) == (0, 0, 1, 1)
    assert _view_grid_position(window, window.sagittal_viewer) == (0, 1, 1, 1)
    assert _view_grid_position(window, window.coronal_viewer) == (1, 0, 1, 1)
    assert _view_grid_position(window, window.viewer_3d) == (1, 1, 1, 1)

    window.set_view_layout("planning")

    assert _view_grid_position(window, window.viewer_3d) == (0, 1, 3, 1)
    assert window.layout_combo.currentData() == "planning"


def test_toolbar_exposes_mpr_fit_and_3d_zoom_controls(ui_main_window):
    window = ui_main_window

    window._fit_mpr_action.trigger()
    window._zoom_in_3d_action.trigger()
    window._zoom_out_3d_action.trigger()
    window._fit_3d_action.trigger()

    assert all(viewer.fit_count >= 1 for viewer in window._get_mpr_viewers())
    assert window.viewer_3d.zoom_factors == [1.2, 1.0 / 1.2]
    assert window.viewer_3d.fit_count == 1


def test_toolbar_uses_one_icon_tool_palette_without_legacy_modes(ui_main_window):
    window = ui_main_window

    actions = window._tool_group.actions()
    assert [action.text() for action in actions] == [
        "Select",
        "Add Screw",
        "Distance",
        "Angle",
    ]
    assert all(not action.icon().isNull() for action in actions)
    assert window._select_tool_action.isChecked()
    assert not {"Navigate", "Screw Tool", "Measure"}.intersection(
        action.text() for action in actions
    )


def test_add_screw_palette_tool_places_and_selects_screw_from_two_mpr_clicks(
    ui_main_window,
):
    window = ui_main_window
    image = _create_test_image()
    window._on_dicom_loaded(
        image=image,
        metadata={"series_id": "SERIES-ADD-SCREW", "num_slices": 12},
        progress=_ProgressStub(),
    )

    window._add_screw_tool_action.trigger()
    window.axial_viewer.crosshair_moved.emit("axial", 2.0, 3.0, 4.0)
    assert "tip" in window.statusbar.currentMessage().lower()
    window.axial_viewer.crosshair_moved.emit("axial", 2.0, 3.0, 34.0)

    assert window._tool_ctrl.active_tool == "screw"
    assert len(window.screw_tool.get_screws()) == 1
    assert window.screw_list_widget.currentRow() == 0
    assert window.viewer_3d.screws[0].radius == pytest.approx(3.25)
    for viewer in window._get_mpr_viewers():
        assert viewer.screw_overlays[0]["entry"] == (2.0, 3.0, 4.0)
        assert viewer.screw_overlays[0]["target"] == (2.0, 3.0, 34.0)


def test_distance_and_angle_palette_tools_complete_measurements(ui_main_window):
    window = ui_main_window
    image = _create_test_image()
    window._on_dicom_loaded(
        image=image,
        metadata={"series_id": "SERIES-MEASURE", "num_slices": 12},
        progress=_ProgressStub(),
    )

    window._distance_tool_action.trigger()
    window.axial_viewer.crosshair_moved.emit("axial", 0.0, 0.0, 0.0)
    window.axial_viewer.crosshair_moved.emit("axial", 3.0, 4.0, 0.0)

    assert window.measurement_list_widget.count() == 1
    assert "5.00 mm" in window.measurement_list_widget.item(0).text()

    window._angle_tool_action.trigger()
    window.axial_viewer.crosshair_moved.emit("axial", 1.0, 0.0, 0.0)
    window.axial_viewer.crosshair_moved.emit("axial", 0.0, 0.0, 0.0)
    window.axial_viewer.crosshair_moved.emit("axial", 0.0, 1.0, 0.0)

    assert window.measurement_list_widget.count() == 2
    assert "90.0°" in window.measurement_list_widget.item(1).text()


def test_measurement_can_jump_to_cut_be_remeasured_and_deleted(ui_main_window):
    window = ui_main_window
    image = _create_test_image()
    window._on_dicom_loaded(
        image=image,
        metadata={"series_id": "SERIES-MEASURE-EDIT", "num_slices": 12},
        progress=_ProgressStub(),
    )

    window._distance_tool_action.trigger()
    window.axial_viewer.crosshair_moved.emit("axial", 0.0, 0.0, 0.0)
    window.axial_viewer.crosshair_moved.emit("axial", 3.0, 4.0, 0.0)
    window.measurement_list_widget.setCurrentRow(0)

    assert window.show_measurement_btn.text() == "Show Cut"
    assert window.edit_measurement_btn.text() == "Edit"
    assert window.remove_measurement_btn.text() == "Delete"

    window.axial_viewer.set_slice_position(5.0)
    window.show_measurement_btn.click()

    assert window.axial_viewer.last_slice_position == pytest.approx(0.0)

    window.edit_measurement_btn.click()
    window.axial_viewer.crosshair_moved.emit("axial", 0.0, 0.0, 0.0)
    window.axial_viewer.crosshair_moved.emit("axial", 0.0, 10.0, 0.0)

    assert window.measurement_list_widget.count() == 1
    assert "10.0 mm" in window.measurement_list_widget.item(0).text()
    assert len(window.measurement_tool.get_measurements()) == 1
    assert window.axial_viewer.measurements[1]["label"] == "10.0 mm"

    window.measurement_list_widget.setCurrentRow(0)
    window.remove_measurement_btn.click()

    assert window.measurement_list_widget.count() == 0
    assert window.axial_viewer.measurements == {}
    assert window.viewer_3d.measurements == {}


def test_measurement_point_drag_updates_value_without_opening_edit_mode(
    ui_main_window,
):
    window = ui_main_window
    image = _create_test_image()
    window._on_dicom_loaded(
        image=image,
        metadata={"series_id": "SERIES-MEASURE-DRAG", "num_slices": 12},
        progress=_ProgressStub(),
    )
    window._distance_tool_action.trigger()
    window.axial_viewer.crosshair_moved.emit("axial", 0.0, 0.0, 0.0)
    window.axial_viewer.crosshair_moved.emit("axial", 3.0, 4.0, 0.0)

    window._tool_ctrl.update_measurement_point(
        measurement_id=1,
        point_index=1,
        world_point=(0.0, 10.0, 0.0),
    )

    assert window.measurement_list_widget.count() == 1
    assert "10.0 mm" in window.measurement_list_widget.item(0).text()
    assert window.measurement_tool.get_measurements()[0].distance == pytest.approx(10.0)
    assert window.axial_viewer.measurements[1]["label"] == "10.0 mm"
    assert window.viewer_3d.measurements[1]["label"] == "10.0 mm"


def test_delete_key_removes_the_active_mpr_measurement(ui_main_window):
    window = ui_main_window
    image = _create_test_image()
    window._on_dicom_loaded(
        image=image,
        metadata={"series_id": "SERIES-MEASURE-DELETE", "num_slices": 12},
        progress=_ProgressStub(),
    )
    window._distance_tool_action.trigger()
    window.axial_viewer.crosshair_moved.emit("axial", 0.0, 0.0, 0.0)
    window.axial_viewer.crosshair_moved.emit("axial", 3.0, 4.0, 0.0)
    window._tool_ctrl.select_measurement_from_view(1)

    window._delete_screw_action.trigger()

    assert window.measurement_list_widget.count() == 0
    assert window.measurement_tool.get_measurements() == []


def test_selected_vertebrae_can_be_shown_alone_in_3d(ui_main_window):
    """The CT volume's 3D visibility now follows the isolation toggle, not
    the level checkboxes: in Full CT mode the CT stays on screen whatever
    levels are checked, and only an isolated session hides it."""
    window = ui_main_window
    window._seg_ctrl._last_vtk_mask = object()
    window._seg_ctrl._last_segmentation_mask_path = "mask.nii.gz"
    window.update_vertebra_level_checks([28, 29, 30])

    window._vertebra_level_checks[29].setChecked(False)

    assert window.viewer_3d.vertebral_mesh_labels == [28, 30]
    assert window.viewer_3d.volume_visible is True

    window._toggle_vertebra_level_checks(True)

    assert window.viewer_3d.vertebral_mesh_labels == [28, 29, 30]

    window._seg_ctrl._vertebrae_isolated = True
    window._on_vertebra_level_selection_changed()

    assert window.viewer_3d.volume_visible is False


def test_segmented_vertebrae_use_shared_live_checkboxes(ui_main_window):
    window = ui_main_window
    window._seg_ctrl._last_vtk_mask = object()
    window._seg_ctrl._last_segmentation_mask_path = "mask.nii.gz"

    window.update_vertebra_level_checks([28, 29, 30])

    assert sorted(window._vertebra_level_checks) == [28, 29, 30]
    assert all(
        checkbox.isChecked()
        for checkbox in window._vertebra_level_checks.values()
    )
    assert window.auto_screw_plan_btn.isEnabled() is True

    window._vertebra_level_checks[29].setChecked(False)

    assert window.viewer_3d.vertebral_mesh_labels == [28, 30]
    assert window._auto_placement_ctrl._get_selected_labels() == [28, 30]
    assert window._seg_ctrl._vertebrae_isolated is False
    assert window.vertebra_isolate_btn.text() == "Isolate Vertebrae"


@pytest.mark.parametrize(
    ("control_name", "initial", "typed", "expected"),
    [
        ("selected_screw_diameter", 5.5, "65", 6.5),
        ("diameter_spin", 6.5, "55", 5.5),
        ("selected_screw_diameter", 6.5, "7", 7.0),
    ],
)
def test_diameter_typing_replaces_and_interprets_shorthand(
    ui_main_window,
    qtbot,
    control_name,
    initial,
    typed,
    expected,
):
    spinbox = getattr(ui_main_window, control_name)
    spinbox.setEnabled(True)
    spinbox.setValue(initial)
    ui_main_window.show()

    qtbot.mouseClick(spinbox, Qt.MouseButton.LeftButton)
    qtbot.keyClicks(spinbox, typed)

    assert spinbox.value() == pytest.approx(expected)


def test_segmentation_card_shows_only_auto_and_isolate_controls(ui_main_window):
    window = ui_main_window

    assert window.seg_run_btn.isHidden() is False
    assert window.vertebra_isolate_btn.isHidden() is False
    assert window.seg_advanced_panel.isHidden() is True
    assert window.vertebra_restore_btn.isHidden() is True


def test_smooth_vertebral_mesh_does_not_overlap_raw_3d_segmentation(
    ui_main_window,
):
    assert ui_main_window.seg_show_2d_check.isChecked() is True
    assert ui_main_window.seg_show_3d_check.isChecked() is False


def test_planning_uses_visible_vertebra_checkboxes_without_dropdown(
    ui_main_window,
):
    window = ui_main_window
    window._seg_ctrl._last_segmentation_mask_path = "mask.nii.gz"

    window.update_vertebra_level_checks([27, 28, 29, 30])

    assert not hasattr(window, "vertebra_selector_button")
    assert not hasattr(window, "_auto_vertebra_menu")

    window._vertebra_level_checks[27].setChecked(False)

    assert window._auto_placement_ctrl._get_selected_labels() == [28, 29, 30]


def test_clearing_vertebra_levels_disables_planning(ui_main_window):
    window = ui_main_window
    window._seg_ctrl._last_segmentation_mask_path = "mask.nii.gz"
    window.update_vertebra_level_checks([28, 29, 30])

    window.clear_vertebra_display_options()

    assert window._vertebra_level_checks == {}
    assert window.auto_screw_plan_btn.isEnabled() is False


def test_segmentation_result_hides_method_and_mask_details(
    ui_main_window, tmp_path
):
    window = ui_main_window
    image = _create_test_image()
    window._on_dicom_loaded(
        image=image,
        metadata={"series_id": "SERIES-SIMPLE-SEG", "num_slices": 12},
        progress=_ProgressStub(),
    )
    mask_path = tmp_path / "simple_mask.nii.gz"
    _write_mask(image, mask_path)

    window._on_segmentation_finished(
        SegmentationRunResult(
            success=True,
            method="totalsegmentator",
            mask_path=str(mask_path),
            message="ok",
            geometry_warnings=[],
        )
    )

    status = window.seg_status_label.text()
    assert status.startswith("Segmentation ready")
    assert "Method:" not in status
    assert "Mask:" not in status
    # _write_mask carries no vertebra label, so there is nothing to isolate.
    assert window.vertebra_isolate_btn.isEnabled() is False


def test_threshold_fallback_disables_auto_planning(
    ui_main_window, monkeypatch, tmp_path
):
    window = ui_main_window
    image = _create_test_image()
    window._on_dicom_loaded(
        image=image,
        metadata={"series_id": "SERIES-FALLBACK-SEG", "num_slices": 12},
        progress=_ProgressStub(),
    )
    mask_path = tmp_path / "fallback_mask.nii.gz"
    _write_mask(image, mask_path)

    monkeypatch.setattr(
        seg_controller_module.QMessageBox,
        "warning",
        lambda *a, **k: None,
    )

    window._on_segmentation_finished(
        SegmentationRunResult(
            success=True,
            method="threshold_fallback",
            mask_path=str(mask_path),
            message="Threshold fallback used.",
            geometry_warnings=[],
        )
    )

    assert window.auto_screw_plan_btn.isEnabled() is False
    assert (
        window.seg_status_label.text()
        == "Threshold mask ready · planning unavailable"
    )


def test_selected_screw_enters_and_exits_screw_axis_mpr(ui_main_window):
    window = ui_main_window
    assert window.screw_axis_mpr_btn.text() == "Screw MPR"
    assert window.standard_mpr_btn.text() == "Std MPR"
    image = _create_test_image()
    window._on_dicom_loaded(
        image=image,
        metadata={"series_id": "SERIES-SCREW-MPR", "num_slices": 12},
        progress=_ProgressStub(),
    )
    screw = Screw(
        entry_point=(2.0, 3.0, 4.0),
        target_point=(8.0, 9.0, 24.0),
        diameter=6.0,
    )
    window._tool_ctrl.screw_tool.add_screw(screw)
    window._tool_ctrl._add_screw_to_list(screw)
    window.screw_list_widget.setCurrentRow(0)

    assert window.screw_axis_mpr_btn.isEnabled() is True

    window.screw_axis_mpr_btn.click()

    assert window._screw_mpr_ctrl.is_active is True
    assert window.axial_viewer.custom_title == "Oblique Axial · Screw #1"
    assert window.sagittal_viewer.custom_title == "Oblique Sagittal · Screw #1"
    assert window.coronal_viewer.custom_title == "Cross-section · 50%"
    assert window.screw_axis_position_slider.isEnabled() is True
    assert all(
        viewer.review_screw_id == 0
        for viewer in window._get_mpr_viewers()
    )

    window.standard_mpr_btn.click()

    assert window._screw_mpr_ctrl.is_active is False
    assert window.axial_viewer.custom_title is None
    assert all(
        viewer.review_screw_id is None
        for viewer in window._get_mpr_viewers()
    )


def test_entering_screw_mpr_through_the_button_opens_review(ui_main_window):
    """Entering Screw MPR through the real button (not by poking ``_active``
    directly) must land the step panel on Review, the way selecting a screw
    already does."""
    window = ui_main_window
    image = _create_test_image()
    window._on_dicom_loaded(
        image=image,
        metadata={"series_id": "SERIES-MPR-REVIEW", "num_slices": 12},
        progress=_ProgressStub(),
    )
    screw = Screw(
        entry_point=(2.0, 3.0, 4.0),
        target_point=(8.0, 9.0, 24.0),
        diameter=6.0,
    )
    window._tool_ctrl.screw_tool.add_screw(screw)
    window._tool_ctrl._add_screw_to_list(screw)
    window.screw_list_widget.setCurrentRow(0)

    window.show_step("Plan")
    assert window.step_panel.current_step == "Plan"

    window.screw_axis_mpr_btn.click()

    assert window._screw_mpr_ctrl.is_active is True
    assert window.step_panel.current_step == "Review"

    window.standard_mpr_btn.click()

    assert window._screw_mpr_ctrl.is_active is False


def test_planning_cockpit_workspace_and_guided_scaffold(ui_main_window):
    window = ui_main_window

    assert window.workspace_mode_combo.currentData() == "planning"
    guided_index = window.workspace_mode_combo.findData("guided")
    assert guided_index >= 0
    assert window.workspace_mode_combo.model().item(guided_index).isEnabled() is False
    assert list(window.step_section_order) == ["Study", "Segment", "Plan", "Review"]


def test_selected_screw_inspector_updates_from_list_selection(ui_main_window):
    window = ui_main_window
    screw = Screw(
        entry_point=(0.0, 0.0, 0.0),
        target_point=(0.0, -45.0, 8.0),
        diameter=6.5,
        vertebra_level="L3",
        side="left",
        grade="B",
        breach_distance=1.4,
    )
    window._tool_ctrl.screw_tool.add_screw(screw)
    window._tool_ctrl._add_screw_to_list(screw)

    window.screw_list_widget.setCurrentRow(0)

    assert window.selected_screw_title.text() == "L3 Left"
    assert window.selected_screw_diameter.value() == pytest.approx(6.5)
    assert window.selected_screw_diameter.maximum() == pytest.approx(7.5)
    assert window.selected_screw_length.text() == f"{screw.length:.1f} mm"
    assert window.selected_screw_grade.text() == "Grade B"
    assert "1.4 mm" in window.selected_screw_warning.text()


def test_selected_screw_diameter_control_updates_model_and_linked_views(
    ui_main_window,
):
    window = ui_main_window
    image = _create_test_image()
    window._on_dicom_loaded(
        image=image,
        metadata={"series_id": "SERIES-DIAMETER", "num_slices": 12},
        progress=_ProgressStub(),
    )
    screw = Screw(
        entry_point=(2.0, 3.0, 4.0),
        target_point=(2.0, 3.0, 30.0),
        diameter=6.5,
        vertebra_level="L3",
        side="left",
    )
    window._tool_ctrl.add_existing_screw(screw, select=True)

    window.selected_screw_diameter.setValue(7.5)

    updated = window.screw_tool.get_screws()[0]
    assert updated.diameter == pytest.approx(7.5)
    assert window.viewer_3d.screws[0].radius == pytest.approx(3.75)
    for viewer in window._get_mpr_viewers():
        assert viewer.screw_overlays[0]["diameter"] == pytest.approx(7.5)
    assert "7.5" in window.screw_list_widget.rowText(0)


def test_manual_screw_diameter_defaults_to_common_size_and_caps_at_7_5(
    ui_main_window,
):
    window = ui_main_window

    assert window.diameter_spin.value() == pytest.approx(6.5)
    assert window.diameter_spin.maximum() == pytest.approx(7.5)
    assert window.diameter_spin.singleStep() == pytest.approx(0.5)


def test_screw_review_card_exposes_direct_multi_screw_navigation(ui_main_window):
    window = ui_main_window
    screws = [
        Screw(
            entry_point=(0.0, 0.0, float(index)),
            target_point=(0.0, -40.0, float(index + 5)),
            diameter=6.0 + index,
            vertebra_level=f"L{index + 3}",
            side="left" if index == 0 else "right",
        )
        for index in range(2)
    ]
    for screw in screws:
        window._tool_ctrl.screw_tool.add_screw(screw)
        window._tool_ctrl._add_screw_to_list(screw)

    window.screw_list_widget.setCurrentRow(0)

    assert window.selected_screw_group.property("role") == "review"
    assert window.selected_screw_counter.text() == "Screw 1 of 2"
    assert window.screw_previous_btn.isEnabled() is False
    assert window.screw_next_btn.isEnabled() is True
    assert (
        window.screw_list_widget.verticalScrollBarPolicy()
        == Qt.ScrollBarPolicy.ScrollBarAlwaysOn
    )

    window.screw_next_btn.click()

    assert window.screw_list_widget.currentRow() == 1
    assert window.selected_screw_counter.text() == "Screw 2 of 2"
    assert window.selected_screw_title.text() == "L4 Right"
    assert window.screw_previous_btn.isEnabled() is True
    assert window.screw_next_btn.isEnabled() is False


def test_cockpit_actions_have_semantic_roles(ui_main_window):
    window = ui_main_window

    assert window.seg_run_btn.property("role") == "primary"
    assert window.auto_screw_plan_btn.property("role") == "primary"
    assert window.screw_axis_mpr_btn.property("role") == "primary"
    assert window.standard_mpr_btn.property("role") == "secondary"
    assert window.remove_screw_btn.property("role") == "danger"
    assert window._btn_clear_screws.property("role") == "danger"


def test_theme_selector_exposes_c_default_and_dark_alternatives(ui_main_window):
    window = ui_main_window

    assert window.theme_combo.currentData() == "soft_light"
    assert [
        window.theme_combo.itemData(index)
        for index in range(window.theme_combo.count())
    ] == ["soft_light", "graphite_blue", "graphite_mint"]


def test_theme_selection_applies_immediately_and_persists(ui_main_window):
    window = ui_main_window

    class _Settings:
        def __init__(self):
            self.values = {}

        def setValue(self, key, value):
            self.values[key] = value

        def sync(self):
            return

    settings = _Settings()
    window._settings = settings

    window.theme_combo.setCurrentIndex(
        window.theme_combo.findData("graphite_mint")
    )

    assert window._theme_name == "graphite_mint"
    assert "#10110F" in QApplication.instance().styleSheet()
    assert settings.values["appearance/theme"] == "graphite_mint"


def test_loaded_plan_rebuilds_screw_overlays_in_all_mpr_views(ui_main_window):
    window = ui_main_window
    screw = Screw(
        entry_point=(2.0, 3.0, 4.0),
        target_point=(8.0, 9.0, 24.0),
        diameter=6.0,
    )

    window._plan_ctrl._apply_loaded_plan([screw], [], [])

    for viewer in window._get_mpr_viewers():
        assert viewer.screw_overlays[0]["entry"] == screw.entry_point
        assert viewer.screw_overlays[0]["target"] == screw.target_point


def test_loaded_plan_selects_the_first_screw_and_opens_review(ui_main_window):
    """Loading a plan must match automatic planning: land on the first screw
    so the Review header, inspector and 3D highlight show it instead of
    "No screws" and "No screw selected"."""
    window = ui_main_window
    window.show_step("Plan")
    first = Screw(entry_point=(2.0, 3.0, 4.0), target_point=(8.0, 9.0, 24.0), diameter=6.0)
    second = Screw(entry_point=(12.0, 13.0, 14.0), target_point=(18.0, 19.0, 34.0), diameter=6.0)

    window._plan_ctrl._apply_loaded_plan([first, second], [], [])

    assert window.screw_list_widget.currentRow() == 0
    assert window.selected_screw_counter.text() == "Screw 1 of 2"
    assert window.selected_screw_title.text() == "Screw #1"
    assert window.step_panel.current_step == "Review"


def test_loaded_plan_is_regraded_when_a_grader_is_attached(ui_main_window):
    """Plans saved by the old HU heuristic carry stale grades; loading a plan
    while a segmentation-based grader is attached must re-grade every screw
    rather than trusting the persisted (possibly stale) values."""
    import numpy as np

    from src.core.screw_grading import ScrewGrader
    from src.utils.planning_io import screw_from_dict

    window = ui_main_window

    screw = screw_from_dict(
        {
            "entry_point": [30.0, 38.0, 30.0],
            "target_point": [30.0, 22.0, 30.0],
            "diameter": 6.5,
            "grade": "A",
        }
    )
    assert screw.grade == "A"
    assert screw.mean_hu is None

    arr = np.zeros((60, 60, 60), dtype=np.uint8)
    arr[20:40, 20:40, 20:40] = 28
    mask = sitk.GetImageFromArray(arr)
    ct = sitk.GetImageFromArray(np.where(arr > 0, 80, -50).astype(np.int16))
    window._tool_ctrl.screw_tool.set_grader(ScrewGrader(mask, ct))

    window._plan_ctrl._apply_loaded_plan([screw], [], [])

    regraded = window._tool_ctrl.screw_tool.get_screws()[0]
    assert regraded.grade == "A"
    assert regraded.mean_hu == pytest.approx(80.0)


def test_loaded_plan_shows_na_when_no_grader_is_attached(ui_main_window):
    """Without a grader there is nothing behind a stored grade.

    A plan saved by the removed HU heuristic would otherwise keep displaying
    its "A" as a current measurement with no segmentation loaded at all.
    """
    from src.utils.planning_io import screw_from_dict

    window = ui_main_window
    assert window._tool_ctrl.screw_tool.grader is None

    screw = screw_from_dict(
        {
            "entry_point": [2.0, 3.0, 4.0],
            "target_point": [8.0, 9.0, 24.0],
            "diameter": 6.0,
            "grade": "A",
        }
    )

    window._plan_ctrl._apply_loaded_plan([screw], [], [])

    loaded = window._tool_ctrl.screw_tool.get_screws()[0]
    assert loaded.grade == "N/A"


def test_refresh_screw_replaces_same_overlay_id_and_list_row(ui_main_window):
    window = ui_main_window
    image = _create_test_image()
    window._on_dicom_loaded(
        image=image,
        metadata={"series_id": "SERIES-EDIT-SYNC", "num_slices": 12},
        progress=_ProgressStub(),
    )
    screw = Screw(
        entry_point=(2.0, 3.0, 4.0),
        target_point=(2.0, 3.0, 10.0),
        diameter=6.0,
    )
    window.screw_tool.add_screw(screw)
    actor = window.viewer_3d.add_screw(
        screw.entry_point,
        screw.target_point,
        radius=screw.diameter / 2.0,
    )
    window._tool_ctrl._screw_actors.append(actor)
    window._tool_ctrl._add_screw_to_list(screw)
    window._tool_ctrl._add_screw_to_mpr(0, screw)

    updated = window.screw_tool.replace_screw(
        0,
        entry_point=(4.0, 5.0, 6.0),
        target_point=(7.0, 8.0, 12.0),
    )
    window._tool_ctrl.refresh_screw(0, updated)

    for viewer in window._get_mpr_viewers():
        assert viewer.screw_overlays[0]["entry"] == (4.0, 5.0, 6.0)
        assert viewer.screw_overlays[0]["target"] == (7.0, 8.0, 12.0)
    assert "7.3" in window.screw_list_widget.rowText(0)
    assert window.viewer_3d.screws[0].entry_point == (4.0, 5.0, 6.0)


def test_screw_list_row_shows_level_side_and_geometry(ui_main_window):
    window = ui_main_window
    screw = Screw(
        entry_point=(1.0, 2.0, 3.0),
        target_point=(1.0, 2.0, 43.0),
        diameter=6.5,
        vertebra_level="L4",
        side="left",
        grade="A",
    )

    window._tool_ctrl.add_existing_screw(screw, select=True)

    text = window.screw_list_widget.rowText(0)
    assert "L4" in text
    assert "Left" in text
    assert "6.5" in text
    assert "40.0" in text
    assert window.screw_list_widget.currentRow() == 0


def test_delete_shortcut_removes_selected_screw(ui_main_window):
    window = ui_main_window
    screw = Screw((1.0, 2.0, 3.0), (1.0, 2.0, 33.0))
    window._tool_ctrl.add_existing_screw(screw, select=True)

    window._delete_screw_action.trigger()

    assert window.screw_tool.get_screws() == []
    assert window.screw_list_widget.count() == 0


def test_deleting_first_screw_reindexes_remaining_visual_and_selects_it(
    ui_main_window,
):
    window = ui_main_window
    window._tool_ctrl.add_existing_screw(
        Screw((1.0, 2.0, 3.0), (1.0, 2.0, 33.0)), select=True
    )
    remaining = Screw(
        (4.0, 5.0, 6.0),
        (4.0, 5.0, 36.0),
        vertebra_level="L3",
        side="right",
    )
    window._tool_ctrl.add_existing_screw(remaining)
    window.screw_list_widget.setCurrentRow(0)

    window._tool_ctrl.remove_selected_screw()

    assert window.screw_tool.get_screws() == [remaining]
    assert window.viewer_3d.screws[0].screw_id == 0
    assert window.screw_list_widget.currentRow() == 0


def test_3d_direct_drag_callbacks_update_selected_screw(ui_main_window):
    window = ui_main_window
    image = _create_test_image()
    window._on_dicom_loaded(
        image=image,
        metadata={"series_id": "SERIES-3D-DRAG", "num_slices": 12},
        progress=_ProgressStub(),
    )
    screw = Screw((2.0, 3.0, 4.0), (2.0, 3.0, 10.0))
    window._tool_ctrl.add_existing_screw(screw, select=True)
    callbacks = window.viewer_3d.screw_interaction_callbacks

    assert callbacks["on_begin"](0, "entry", screw.entry_point) is True
    assert callbacks["on_drag"]((4.0, 5.0, 6.0), "3D") is True
    callbacks["on_end"]()

    updated = window.screw_tool.get_screws()[0]
    assert updated.entry_point == (4.0, 5.0, 6.0)
    assert updated.target_point == (2.0, 3.0, 10.0)
    assert window.screw_list_widget.currentRow() == 0


def test_mpr_drag_keeps_ct_fixed_then_realigns_when_edit_finishes(
    ui_main_window,
):
    window = ui_main_window
    image = _create_test_image()
    window._on_dicom_loaded(
        image=image,
        metadata={"series_id": "SERIES-EDIT-MPR", "num_slices": 12},
        progress=_ProgressStub(),
    )
    screw = Screw(
        entry_point=(2.0, 3.0, 4.0),
        target_point=(8.0, 9.0, 12.0),
        diameter=6.0,
    )
    window.screw_tool.add_screw(screw)
    actor = window.viewer_3d.add_screw(
        screw.entry_point,
        screw.target_point,
        radius=screw.diameter / 2.0,
    )
    window._tool_ctrl._screw_actors.append(actor)
    window._tool_ctrl._add_screw_to_list(screw)
    window._tool_ctrl._add_screw_to_mpr(0, screw)
    window.screw_list_widget.setCurrentRow(0)

    assert window.screw_edit_entry_btn.isEnabled() is True
    window.screw_axis_mpr_btn.click()
    previous_axes = window.axial_viewer.custom_axes
    callbacks = window.axial_viewer.screw_interaction_callbacks
    assert callbacks["on_begin"](0, "entry", screw.entry_point) is True

    callbacks["on_drag"]((4.0, 5.0, 6.0), "Axial MPR")

    updated = window.screw_tool.get_screws()[0]
    assert updated.entry_point == (4.0, 5.0, 6.0)
    assert updated.target_point == (8.0, 9.0, 12.0)
    assert window._screw_edit_ctrl.mode == "entry"
    assert window.axial_viewer.custom_axes is previous_axes

    callbacks["on_end"]()

    assert window._screw_edit_ctrl.mode == "idle"
    assert window.axial_viewer.custom_axes is not previous_axes
    for viewer in window._get_mpr_viewers():
        assert viewer.screw_overlays[0]["entry"] == (4.0, 5.0, 6.0)
    assert window.viewer_3d.screws[0].entry_point == (4.0, 5.0, 6.0)
    assert window.screw_axis_position_slider.value() == 50


def test_escape_action_cancels_pending_screw_edit(ui_main_window):
    window = ui_main_window
    image = _create_test_image()
    window._on_dicom_loaded(
        image=image,
        metadata={"series_id": "SERIES-EDIT-ESC", "num_slices": 12},
        progress=_ProgressStub(),
    )
    screw = Screw((2.0, 3.0, 4.0), (8.0, 9.0, 12.0))
    window.screw_tool.add_screw(screw)
    window._tool_ctrl._add_screw_to_list(screw)
    window.screw_list_widget.setCurrentRow(0)
    window.screw_edit_move_btn.click()

    window._cancel_screw_edit_action.trigger()

    assert window._screw_edit_ctrl.mode == "idle"
    assert window.screw_edit_move_btn.isChecked() is False
    assert all(
        viewer.screw_interaction_cancelled == 1
        for viewer in window._get_mpr_viewers()
    )
    assert window.viewer_3d.screw_interaction_cancelled == 1


def test_ui_workflow_load_segmentation_toggle_and_save_plan(
    ui_main_window, monkeypatch, tmp_path
):
    window = ui_main_window
    image = _create_test_image()
    progress = _ProgressStub()

    window._on_dicom_loaded(
        image=image,
        metadata={
            "series_id": "SERIES-TEST-001",
            "patient_name": "UnitTest",
            "study_date": "20260207",
            "modality": "CT",
            "size": str(image.GetSize()),
            "spacing": str(image.GetSpacing()),
            "num_slices": image.GetSize()[2],
        },
        progress=progress,
    )

    assert progress.closed is True
    assert window.volume_manager.get_vtk_image() is not None
    assert "UnitTest" in window.info_label.text()

    mask_path = tmp_path / "mask.nii.gz"
    _write_mask(image, mask_path)
    window._on_segmentation_finished(
        SegmentationRunResult(
            success=True,
            method="totalsegmentator",
            mask_path=str(mask_path),
            message="ok",
            geometry_warnings=[],
        )
    )

    assert window._last_segmentation_mask_path == str(mask_path)

    window.seg_show_2d_check.setChecked(False)
    window.seg_show_3d_check.setChecked(False)
    window._update_segmentation_visibility()

    for viewer in window._get_mpr_viewers():
        assert viewer.visible is False
    assert window.viewer_3d.visible is False

    window.screw_tool.add_screw(
        Screw(
            entry_point=(1.0, 2.0, 3.0),
            target_point=(1.0, 2.0, 25.0),
            diameter=6.0,
        )
    )

    output_path = tmp_path / "plan.json"
    monkeypatch.setattr(
        plan_controller_module.QFileDialog,
        "getSaveFileName",
        staticmethod(lambda *_args, **_kwargs: (str(output_path), "JSON Files (*.json)")),
    )
    window._save_plan_dialog()

    assert output_path.exists()
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["series_id"] == "SERIES-TEST-001"
    assert len(payload["screws"]) == 1


def test_ui_shows_geometry_warning_dialog(ui_main_window, monkeypatch, tmp_path):
    window = ui_main_window
    image = _create_test_image()
    progress = _ProgressStub()
    window._on_dicom_loaded(
        image=image,
        metadata={"series_id": "SERIES-TEST-002", "num_slices": image.GetSize()[2]},
        progress=progress,
    )

    mask_path = tmp_path / "mask_warn.nii.gz"
    _write_mask(image, mask_path)

    warning_calls = []

    def _capture_warning(_parent, title, text):
        warning_calls.append((title, text))
        return QMessageBox.StandardButton.Ok

    monkeypatch.setattr(seg_controller_module.QMessageBox, "warning", _capture_warning)

    window._on_segmentation_finished(
        SegmentationRunResult(
            success=True,
            method="totalsegmentator",
            mask_path=str(mask_path),
            message="ok",
            geometry_warnings=["spacing mismatch at axis 1: ref=1.0, mask=1.2"],
        )
    )

    assert any(title == "Geometry Warning" for title, _ in warning_calls)


def test_inspector_shows_hu_source_and_warnings(ui_main_window):
    window = ui_main_window
    screw = Screw(
        entry_point=(0, 30, 0), target_point=(0, -10, 0), diameter=6.0,
        vertebra_level="L4", side="left", grade="B", breach_distance=0.8,
        mean_hu=210.0, min_hu=95.0,
        warnings=["Breach distance 0.8 mm (grade B)"], source="auto",
    )
    index = window._tool_ctrl.add_existing_screw(screw, select=True)
    window.update_selected_screw_inspector(index, screw, False)

    assert "210" in window.selected_screw_hu.text()
    assert "95" in window.selected_screw_hu.text()
    assert "Auto" in window.selected_screw_source.text()
    assert "0.8 mm" in window.selected_screw_warning.text()
    assert "Breach distance 0.8 mm (grade B)" in window.selected_screw_warning.text()


def test_inspector_reports_manual_source_and_missing_hu(ui_main_window):
    window = ui_main_window
    screw = Screw(
        entry_point=(0, 30, 0), target_point=(0, -10, 0), diameter=6.0,
        vertebra_level="L4", side="right", grade="N/A",
    )
    index = window._tool_ctrl.add_existing_screw(screw, select=True)
    window.update_selected_screw_inspector(index, screw, False)

    assert "no CT sample" in window.selected_screw_hu.text()
    assert window.selected_screw_source.text() == "Manual"
    assert "run segmentation" in window.selected_screw_warning.text().lower()


def test_inspector_resets_hu_and_source_without_selection(ui_main_window):
    window = ui_main_window
    window.update_selected_screw_inspector(-1, None, False)

    assert window.selected_screw_hu.text() == "--"
    assert window.selected_screw_source.text() == "--"


def test_segmentation_regrades_screws_placed_before_segmentation(
    ui_main_window, tmp_path
):
    window = ui_main_window
    image = _create_test_image()
    window._on_dicom_loaded(
        image=image,
        metadata={"series_id": "SERIES-REGRADE", "num_slices": image.GetSize()[2]},
        progress=_ProgressStub(),
    )

    screw = Screw(
        entry_point=(4.0, 4.0, 4.0),
        target_point=(4.0, 12.0, 4.0),
        diameter=6.0,
        grade="N/A",
        warnings=["Not graded: run segmentation first"],
    )
    window._tool_ctrl.add_existing_screw(screw, select=True)

    mask_path = tmp_path / "mask_regrade.nii.gz"
    _write_mask(image, mask_path)
    window._on_segmentation_finished(
        SegmentationRunResult(
            success=True,
            method="totalsegmentator",
            mask_path=str(mask_path),
            message="ok",
        )
    )

    regraded = window._tool_ctrl.screw_tool.get_screws()[0]
    assert "Not graded: run segmentation first" not in regraded.warnings


def test_reset_workspace_purges_segmentation_temp_dirs(ui_main_window, tmp_path):
    from src.core.totalseg_integration import SegmentationWorkspace

    window = ui_main_window
    ctrl = window._seg_ctrl
    ctrl.workspace = SegmentationWorkspace(root=str(tmp_path))
    created = ctrl.workspace.create()
    assert Path(created).exists()

    window.reset_workspace()

    assert not Path(created).exists()


def test_close_event_purges_segmentation_temp_dirs(ui_main_window, tmp_path):
    from PyQt6.QtGui import QCloseEvent

    from src.core.totalseg_integration import SegmentationWorkspace

    window = ui_main_window
    ctrl = window._seg_ctrl
    ctrl.workspace = SegmentationWorkspace(root=str(tmp_path))
    created = ctrl.workspace.create()
    assert Path(created).exists()

    event = QCloseEvent()
    window.closeEvent(event)

    assert not Path(created).exists()
    assert event.isAccepted()


def test_cancel_request_asks_thread_to_terminate(ui_main_window):
    window = ui_main_window
    ctrl = window._seg_ctrl

    class _StubThread:
        def __init__(self):
            self.cancel_calls = 0

        def request_cancel(self):
            self.cancel_calls += 1

        def isRunning(self):
            return False

    stub = _StubThread()
    dialog = QProgressDialog("running", "Cancel", 0, 0, window)
    ctrl._segmentation_thread = stub
    ctrl._segmentation_progress = dialog
    try:
        ctrl._on_cancel_requested()
    finally:
        ctrl._segmentation_thread = None
        ctrl._segmentation_progress = None
        dialog.deleteLater()

    assert stub.cancel_calls == 1
    assert window.seg_status_label.text() == "Cancelling segmentation..."


def test_cancel_request_without_thread_is_noop(ui_main_window):
    window = ui_main_window
    ctrl = window._seg_ctrl
    ctrl._segmentation_thread = None
    window.seg_status_label.setText("Ready for automatic segmentation")

    ctrl._on_cancel_requested()

    assert window.seg_status_label.text() == "Ready for automatic segmentation"


def test_cancel_request_after_dialog_closed_is_noop(ui_main_window):
    """A late canceled() with no live dialog must not cancel anything."""
    window = ui_main_window
    ctrl = window._seg_ctrl

    class _StubThread:
        def __init__(self):
            self.cancel_calls = 0

        def request_cancel(self):
            self.cancel_calls += 1

        def isRunning(self):
            return False

    stub = _StubThread()
    ctrl._segmentation_thread = stub
    ctrl._segmentation_progress = None
    try:
        ctrl._on_cancel_requested()
    finally:
        ctrl._segmentation_thread = None

    assert stub.cancel_calls == 0


def test_on_cancelled_resets_ui_and_purges_workspace(ui_main_window, tmp_path):
    from src.core.totalseg_integration import SegmentationWorkspace

    window = ui_main_window
    ctrl = window._seg_ctrl
    ctrl.workspace = SegmentationWorkspace(root=str(tmp_path))
    created = ctrl.workspace.create()
    ctrl._active_work_dir = created
    ctrl._segmentation_thread = object()
    window.seg_run_btn.setEnabled(False)

    ctrl._on_cancelled()

    assert not Path(created).exists()
    assert ctrl._active_work_dir is None
    assert window.seg_run_btn.isEnabled() is True
    assert ctrl._segmentation_thread is None
    assert window.seg_status_label.text() == "Segmentation cancelled"


def test_segmentation_thread_emits_cancelled_signal(qtbot, tmp_path, monkeypatch):
    from src.core.totalseg_integration import SegmentationCancelled

    def _raise_cancel(**_kwargs):
        raise SegmentationCancelled("cancelled")

    monkeypatch.setattr(
        seg_controller_module,
        "run_segmentation_with_fallback",
        _raise_cancel,
    )

    thread = seg_controller_module.AutoSegmentationThread(
        sitk_image=sitk.Image([2, 2, 2], sitk.sitkInt16),
        task="total",
        device="cpu",
        work_dir=str(tmp_path),
    )
    errors = []
    thread.error.connect(errors.append)

    with qtbot.waitSignal(thread.cancelled, timeout=5000):
        thread.start()
    thread.wait(5000)

    assert errors == []


def test_segmentation_thread_request_cancel_marks_holder(tmp_path):
    thread = seg_controller_module.AutoSegmentationThread(
        sitk_image=sitk.Image([2, 2, 2], sitk.sitkInt16),
        task="total",
        device="cpu",
        work_dir=str(tmp_path),
    )

    thread.request_cancel()

    assert thread._holder.cancelled is True


class _FakeSegThread(QObject):
    """Non-running stand-in for AutoSegmentationThread in dialog tests."""

    finished = pyqtSignal(object)
    error = pyqtSignal(str)
    progress = pyqtSignal(str)
    cancelled = pyqtSignal()

    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs
        self.started = False

    def isRunning(self):
        return False

    def start(self):
        self.started = True

    def request_cancel(self):
        self.cancelled.emit()


def _prepare_segmentation_run(window, monkeypatch, tmp_path, frozen=False):
    from src.core.totalseg_integration import SegmentationWorkspace

    ctrl = window._seg_ctrl
    ctrl.workspace = SegmentationWorkspace(root=str(tmp_path))
    image = sitk.Image([4, 4, 4], sitk.sitkInt16)
    monkeypatch.setattr(ctrl._vm, "get_vtk_image", lambda: object())
    monkeypatch.setattr(ctrl._vm, "get_sitk_image", lambda: image)
    monkeypatch.setattr(
        seg_controller_module, "AutoSegmentationThread", _FakeSegThread
    )
    monkeypatch.setattr(
        seg_controller_module.sys, "frozen", frozen, raising=False
    )
    return ctrl


def test_progress_dialog_exposes_cancel_button(ui_main_window, monkeypatch, tmp_path):
    window = ui_main_window
    ctrl = _prepare_segmentation_run(window, monkeypatch, tmp_path)

    ctrl.run()

    dialog = ctrl._segmentation_progress
    assert dialog is not None
    cancel_button = dialog.findChild(QPushButton)
    assert cancel_button is not None
    assert cancel_button.text() == "Cancel"

    # Cancelling routes through the thread and resets the UI.
    cancel_button.click()
    assert window.seg_status_label.text() == "Segmentation cancelled"
    assert window.seg_run_btn.isEnabled() is True
    assert ctrl._segmentation_thread is None


def test_frozen_build_hides_cancel_button(ui_main_window, monkeypatch, tmp_path):
    window = ui_main_window
    ctrl = _prepare_segmentation_run(window, monkeypatch, tmp_path, frozen=True)

    ctrl.run()

    dialog = ctrl._segmentation_progress
    assert dialog is not None
    assert dialog.findChild(QPushButton) is None
    assert (
        window.seg_status_label.toolTip()
        == "Cancellation is not available in the packaged build"
    )
    ctrl._segmentation_thread = None
    dialog.close()
    ctrl._segmentation_progress = None


class _FakeSegmentationThread:
    """Stand-in for a still-running AutoSegmentationThread.

    Mimics real QThread semantics closely enough for this test: it reports
    itself as running until `request_cancel()`/`wait()` bring it down, so it
    never leaves the controller's `is_running` permanently True (which would
    otherwise make `MainWindow.closeEvent` pop a blocking modal warning
    during qtbot's window teardown).
    """

    def __init__(self):
        self.cancel_called = False
        self.wait_called_with = None
        self._running = True

    def isRunning(self):
        return self._running

    def request_cancel(self):
        self.cancel_called = True
        self._running = False

    def wait(self, timeout_ms):
        self.wait_called_with = timeout_ms
        self._running = False


def test_reset_state_cancels_and_waits_for_a_running_segmentation_thread(
    ui_main_window, monkeypatch
):
    monkeypatch.setattr(
        seg_controller_module.QMessageBox, "warning", lambda *a, **k: None
    )
    monkeypatch.setattr(
        seg_controller_module.QMessageBox, "question", lambda *a, **k: None
    )
    monkeypatch.setattr(
        main_window_module.QMessageBox, "warning", lambda *a, **k: None
    )
    monkeypatch.setattr(
        main_window_module.QMessageBox, "question", lambda *a, **k: None
    )
    window = ui_main_window
    ctrl = window._seg_ctrl
    fake_thread = _FakeSegmentationThread()
    ctrl._segmentation_thread = fake_thread

    ctrl.reset_state()

    assert fake_thread.cancel_called is True
    assert fake_thread.wait_called_with == 5000
    assert ctrl.is_running is False

    ctrl._segmentation_thread = None


def test_cancelling_a_run_keeps_the_previous_runs_mask_dir(ui_main_window, tmp_path):
    """Cancelling run N must not delete run N-1's mask directory."""
    from src.core.totalseg_integration import SegmentationWorkspace

    window = ui_main_window
    ctrl = window._seg_ctrl
    ctrl.workspace = SegmentationWorkspace(root=str(tmp_path))

    finished_dir = ctrl.workspace.create()
    ctrl._last_segmentation_mask_path = str(
        Path(finished_dir) / "totalseg_multilabel.nii.gz"
    )
    Path(ctrl._last_segmentation_mask_path).write_bytes(b"mask")

    cancelled_dir = ctrl.workspace.create()
    ctrl._active_work_dir = cancelled_dir
    ctrl._segmentation_thread = object()

    ctrl._on_cancelled()

    assert not Path(cancelled_dir).exists()
    assert Path(finished_dir).exists()
    assert Path(ctrl._last_segmentation_mask_path).exists()


def test_normal_completion_does_not_request_cancel(ui_main_window, monkeypatch, tmp_path):
    """Closing the progress dialog emits canceled(); it must not cancel the run."""
    window = ui_main_window
    ctrl = _prepare_segmentation_run(window, monkeypatch, tmp_path)

    monkeypatch.setattr(
        seg_controller_module.QMessageBox, "critical", lambda *a, **k: None
    )
    ctrl.run()
    thread = ctrl._segmentation_thread
    cancel_calls = []
    thread.request_cancel = lambda: cancel_calls.append(1)

    ctrl._on_error("boom")

    assert cancel_calls == []
    assert ctrl._segmentation_progress is None


def test_normal_finish_does_not_request_cancel(ui_main_window, monkeypatch, tmp_path):
    window = ui_main_window
    ctrl = _prepare_segmentation_run(window, monkeypatch, tmp_path)

    monkeypatch.setattr(
        seg_controller_module.QMessageBox, "warning", lambda *a, **k: None
    )
    ctrl.run()
    thread = ctrl._segmentation_thread
    cancel_calls = []
    thread.request_cancel = lambda: cancel_calls.append(1)

    image = sitk.Image([4, 4, 4], sitk.sitkInt16)
    mask_path = tmp_path / "finish_mask.nii.gz"
    _write_mask(image, mask_path)
    ctrl._on_finished(
        SegmentationRunResult(
            success=True,
            method="totalsegmentator",
            mask_path=str(mask_path),
            message="ok",
        )
    )

    assert cancel_calls == []
    assert ctrl._segmentation_progress is None


# ---------------------------------------------------------------------------
# Planning parameter panel + inspector metric rows
# ---------------------------------------------------------------------------


def _planner_settings(isolated_qsettings):
    settings = isolated_qsettings("SNUBH", "PedicleScrewSimulator")
    settings.beginGroup("planner")
    return settings


def test_planner_config_from_panel_and_persistence(ui_main_window, isolated_qsettings):
    window = ui_main_window
    window.plan_fill_ratio_spin.setValue(0.7)
    window.plan_anterior_margin_spin.setValue(5.0)
    window.plan_wall_clearance_spin.setValue(1.5)
    window.plan_max_convergence_spin.setValue(40.0)
    window.plan_hu_threshold_spin.setValue(120.0)

    cfg = window.planner_config()

    assert cfg.pedicle_fill_ratio == pytest.approx(0.7)
    assert cfg.anterior_margin_mm == pytest.approx(5.0)
    assert cfg.wall_clearance_mm == pytest.approx(1.5)
    assert cfg.max_convergence_deg == pytest.approx(40.0)
    assert cfg.trajectory_hu_threshold == pytest.approx(120.0)

    window.save_planner_settings()
    settings = _planner_settings(isolated_qsettings)
    assert float(settings.value("pedicle_fill_ratio")) == pytest.approx(0.7)
    assert float(settings.value("anterior_margin_mm")) == pytest.approx(5.0)
    assert float(settings.value("trajectory_hu_threshold")) == pytest.approx(120.0)


def test_planner_spin_boxes_use_specified_ranges(ui_main_window):
    window = ui_main_window

    assert window.plan_fill_ratio_spin.minimum() == pytest.approx(0.50)
    assert window.plan_fill_ratio_spin.maximum() == pytest.approx(1.00)
    assert window.plan_fill_ratio_spin.singleStep() == pytest.approx(0.05)
    assert (window.plan_wall_clearance_spin.minimum(),
            window.plan_wall_clearance_spin.maximum()) == (0.0, 3.0)
    assert (window.plan_anterior_margin_spin.minimum(),
            window.plan_anterior_margin_spin.maximum()) == (0.0, 15.0)
    assert (window.plan_max_convergence_spin.minimum(),
            window.plan_max_convergence_spin.maximum()) == (5.0, 60.0)
    assert (window.plan_hu_threshold_spin.minimum(),
            window.plan_hu_threshold_spin.maximum()) == (50.0, 300.0)


def test_narrow_pedicle_spin_boxes_reach_the_planner_config(ui_main_window):
    window = ui_main_window

    assert (window.plan_narrow_pedicle_spin.minimum(),
            window.plan_narrow_pedicle_spin.maximum()) == (3.0, 8.0)
    assert window.plan_narrow_pedicle_spin.singleStep() == pytest.approx(0.5)
    assert (window.plan_narrow_lateral_spin.minimum(),
            window.plan_narrow_lateral_spin.maximum()) == (0.0, 6.0)
    assert window.plan_narrow_lateral_spin.singleStep() == pytest.approx(0.5)

    window.plan_narrow_pedicle_spin.setValue(6.0)
    window.plan_narrow_lateral_spin.setValue(4.0)
    cfg = window.planner_config()

    assert cfg.narrow_pedicle_mm == pytest.approx(6.0)
    assert cfg.narrow_lateral_breach_mm == pytest.approx(4.0)


def test_narrow_settings_persist_into_a_new_window(ui_main_window, isolated_qsettings):
    window = ui_main_window
    window.plan_narrow_pedicle_spin.setValue(6.5)
    window.plan_narrow_lateral_spin.setValue(3.0)

    settings = _planner_settings(isolated_qsettings)
    assert float(settings.value("narrow_pedicle_mm")) == pytest.approx(6.5)
    assert float(settings.value("narrow_lateral_breach_mm")) == pytest.approx(3.0)

    reopened = main_window_module.MainWindow()
    try:
        assert reopened.plan_narrow_pedicle_spin.value() == pytest.approx(6.5)
        assert reopened.plan_narrow_lateral_spin.value() == pytest.approx(3.0)
    finally:
        reopened.close()
        reopened.deleteLater()


def test_the_wall_clearance_default_is_zero_but_a_stored_value_is_kept(
    ui_main_window, isolated_qsettings
):
    """No migration: a surgeon who raised the clearance keeps it."""
    from src.core.planner_config import PlannerConfig

    window = ui_main_window
    window.plan_wall_clearance_spin.setValue(1.5)

    reopened = main_window_module.MainWindow()
    try:
        assert reopened.plan_wall_clearance_spin.value() == pytest.approx(1.5)
    finally:
        reopened.close()
        reopened.deleteLater()

    window.plan_reset_defaults_btn.click()
    assert window.plan_wall_clearance_spin.value() == pytest.approx(
        PlannerConfig().wall_clearance_mm
    )
    assert PlannerConfig().wall_clearance_mm == 0.0


def test_planner_settings_persist_into_a_new_window(ui_main_window):
    window = ui_main_window
    window.plan_fill_ratio_spin.setValue(0.65)
    window.plan_anterior_margin_spin.setValue(7.0)

    reopened = main_window_module.MainWindow()
    try:
        assert reopened.plan_fill_ratio_spin.value() == pytest.approx(0.65)
        assert reopened.plan_anterior_margin_spin.value() == pytest.approx(7.0)
        assert reopened.planner_config().pedicle_fill_ratio == pytest.approx(0.65)
    finally:
        reopened.close()
        reopened.deleteLater()


def test_planner_reset_defaults_restores_and_persists_defaults(
    ui_main_window, isolated_qsettings
):
    from src.core.planner_config import PlannerConfig

    window = ui_main_window
    defaults = PlannerConfig()
    window.plan_fill_ratio_spin.setValue(0.9)
    window.plan_anterior_margin_spin.setValue(12.0)

    window.plan_reset_defaults_btn.click()

    assert window.plan_fill_ratio_spin.value() == pytest.approx(
        defaults.pedicle_fill_ratio
    )
    assert window.plan_anterior_margin_spin.value() == pytest.approx(
        defaults.anterior_margin_mm
    )
    settings = _planner_settings(isolated_qsettings)
    assert float(settings.value("pedicle_fill_ratio")) == pytest.approx(
        defaults.pedicle_fill_ratio
    )


def test_corrupt_planner_settings_fall_back_to_defaults(
    ui_main_window, isolated_qsettings
):
    from src.core.planner_config import PlannerConfig

    settings = _planner_settings(isolated_qsettings)
    settings.setValue("pedicle_fill_ratio", "not-a-number")
    settings.setValue("anterior_margin_mm", 999.0)
    settings.sync()

    reopened = main_window_module.MainWindow()
    try:
        defaults = PlannerConfig()
        assert reopened.plan_fill_ratio_spin.value() == pytest.approx(
            defaults.pedicle_fill_ratio
        )
        assert reopened.plan_anterior_margin_spin.value() == pytest.approx(
            defaults.anterior_margin_mm
        )
    finally:
        reopened.close()
        reopened.deleteLater()


def test_optimizer_controls_feed_planner_config(ui_main_window):
    window = ui_main_window

    assert window.plan_mode_combo.currentData() == "optimizer"
    assert window.plan_trajectory_combo.currentData() == "traditional"
    assert (window.plan_weight_safety.value(),
            window.plan_weight_density.value(),
            window.plan_weight_rod.value()) == (100, 50, 30)
    for slider in (window.plan_weight_safety, window.plan_weight_density,
                   window.plan_weight_rod):
        assert (slider.minimum(), slider.maximum()) == (0, 300)

    window.plan_mode_combo.setCurrentIndex(
        window.plan_mode_combo.findData("legacy")
    )
    window.plan_trajectory_combo.setCurrentIndex(
        window.plan_trajectory_combo.findData("cbt")
    )
    window.plan_weight_safety.setValue(250)
    window.plan_weight_density.setValue(75)
    window.plan_weight_rod.setValue(0)

    cfg = window.planner_config()
    assert cfg.mode == "legacy"
    assert cfg.trajectory == "cbt"
    assert cfg.weights.safety == pytest.approx(2.5)
    assert cfg.weights.density == pytest.approx(0.75)
    assert cfg.weights.rod == pytest.approx(0.0)


def test_rod_weight_slider_is_labelled_construct_alignment(ui_main_window):
    """The slider governs the rod line *and* convergence agreement now."""
    window = ui_main_window

    caption = window._planner_weight_captions["plan_weight_rod"]

    assert caption.text() == "Construct alignment"
    assert "convergence" in window.plan_weight_rod.toolTip().lower()


def test_optimizer_controls_persist_into_a_new_window(ui_main_window, isolated_qsettings):
    window = ui_main_window
    window.plan_mode_combo.setCurrentIndex(window.plan_mode_combo.findData("legacy"))
    window.plan_trajectory_combo.setCurrentIndex(
        window.plan_trajectory_combo.findData("cbt")
    )
    window.plan_weight_safety.setValue(220)
    window.plan_weight_rod.setValue(5)

    settings = _planner_settings(isolated_qsettings)
    assert settings.value("mode") == "legacy"
    assert settings.value("trajectory") == "cbt"
    assert float(settings.value("weights/safety")) == pytest.approx(2.2)

    reopened = main_window_module.MainWindow()
    try:
        assert reopened.plan_mode_combo.currentData() == "legacy"
        assert reopened.plan_trajectory_combo.currentData() == "cbt"
        assert reopened.plan_weight_safety.value() == 220
        assert reopened.plan_weight_rod.value() == 5
        assert reopened.planner_config().weights.safety == pytest.approx(2.2)
    finally:
        reopened.close()
        reopened.deleteLater()


def test_planner_reset_defaults_restores_optimizer_controls(ui_main_window):
    window = ui_main_window
    window.plan_mode_combo.setCurrentIndex(window.plan_mode_combo.findData("legacy"))
    window.plan_weight_rod.setValue(300)

    window.plan_reset_defaults_btn.click()

    assert window.plan_mode_combo.currentData() == "optimizer"
    assert window.plan_weight_rod.value() == 30


def test_inspector_shows_metric_rows(ui_main_window):
    window = ui_main_window
    screw = Screw(
        entry_point=(0, 30, 0), target_point=(0, -10, 0), grade="B",
        breach_distance=0.5,
        metrics={
            "body_mean_hu": 128.0,
            "min_wall_mm": 0.0,
            "facet_grade": 1,
            "facet_text": "screw abuts the cephalad facet",
            "heary_direction": "medial",
        },
    )
    index = window._tool_ctrl.add_existing_screw(screw, select=True)
    window.update_selected_screw_inspector(index, screw, False)

    assert "128" in window.selected_screw_body_hu.text()
    assert "0.0" in window.selected_screw_wall.text()
    assert "1" in window.selected_screw_facet.text()
    assert "medial" in window.selected_screw_heary.text()


def test_inspector_metric_rows_default_to_dashes(ui_main_window):
    window = ui_main_window
    screw = Screw(entry_point=(0, 30, 0), target_point=(0, -10, 0))
    index = window._tool_ctrl.add_existing_screw(screw, select=True)
    window.update_selected_screw_inspector(index, screw, False)

    assert window.selected_screw_body_hu.text() == "--"
    assert window.selected_screw_wall.text() == "--"
    assert window.selected_screw_facet.text() == "--"
    assert window.selected_screw_heary.text() == "--"
    assert window.selected_screw_pedicle.text() == "--"

    window.update_selected_screw_inspector(-1, None, False)

    assert window.selected_screw_body_hu.text() == "--"
    assert window.selected_screw_wall.text() == "--"
    assert window.selected_screw_facet.text() == "--"
    assert window.selected_screw_heary.text() == "--"
    assert window.selected_screw_pedicle.text() == "--"


# ---------------------------------------------------------------------------
# Optional pedicle subregion model controls
# ---------------------------------------------------------------------------


def _segmentation_settings(isolated_qsettings):
    settings = isolated_qsettings("SNUBH", "PedicleScrewSimulator")
    settings.beginGroup("segmentation")
    return settings


def test_subregion_controls_exist_and_default_off(ui_main_window):
    window = ui_main_window

    assert window.seg_use_subregion_check.text() == "Use pedicle subregion model"
    assert window.seg_use_subregion_check.isChecked() is False
    assert window.seg_subregion_dir_edit.text() == ""
    assert (
        window.seg_subregion_dir_edit.placeholderText()
        == "Model directory (dataset.json)"
    )
    assert window.seg_subregion_browse_btn is not None


def test_subregion_checkbox_toggle_persists(ui_main_window, isolated_qsettings):
    window = ui_main_window

    window.seg_use_subregion_check.setChecked(True)
    window.seg_subregion_dir_edit.setText(r"C:\models\Dataset501")

    settings = _segmentation_settings(isolated_qsettings)
    assert str(settings.value("use_subregion_model")).lower() in ("true", "1")
    assert settings.value("subregion_model_dir") == r"C:\models\Dataset501"


def test_subregion_settings_persist_into_a_new_window(ui_main_window, tmp_path):
    window = ui_main_window
    model_dir = str(tmp_path / "Dataset501")
    window.seg_use_subregion_check.setChecked(True)
    window.seg_subregion_dir_edit.setText(model_dir)

    reopened = main_window_module.MainWindow()
    try:
        assert reopened.seg_use_subregion_check.isChecked() is True
        assert reopened.seg_subregion_dir_edit.text() == model_dir
    finally:
        reopened.close()
        reopened.deleteLater()


def test_unresolved_subregion_model_shows_in_status(
    ui_main_window, monkeypatch, tmp_path
):
    """Asking for the pedicle model but not finding one must be visible."""
    window = ui_main_window
    monkeypatch.delenv("PSS_SUBREGION_MODEL_DIR", raising=False)
    ctrl = _prepare_segmentation_run(window, monkeypatch, tmp_path)
    monkeypatch.setattr(
        seg_controller_module.QMessageBox, "warning", lambda *a, **k: None
    )

    missing_dir = tmp_path / "no_such_model"
    missing_dir.mkdir()
    window.seg_use_subregion_check.setChecked(True)
    window.seg_subregion_dir_edit.setText(str(missing_dir))

    ctrl.run()
    assert ctrl._pending_subregion_message != ""

    image = sitk.Image([4, 4, 4], sitk.sitkInt16)
    mask_path = tmp_path / "unresolved_mask.nii.gz"
    _write_mask(image, mask_path)
    ctrl._on_finished(
        SegmentationRunResult(
            success=True,
            method="totalsegmentator",
            mask_path=str(mask_path),
            message="ok",
        )
    )

    status = window.seg_status_label.text()
    assert "pedicle model unavailable" in status
    assert str(missing_dir) in status


def test_empty_pedicle_mask_is_reported_as_no_voxels(
    ui_main_window, monkeypatch, tmp_path
):
    window = ui_main_window
    ctrl = _prepare_segmentation_run(window, monkeypatch, tmp_path)
    monkeypatch.setattr(
        seg_controller_module.QMessageBox, "warning", lambda *a, **k: None
    )

    image = sitk.Image([4, 4, 4], sitk.sitkInt16)
    mask_path = tmp_path / "empty_pedicle_mask.nii.gz"
    _write_mask(image, mask_path)
    subregion_path = tmp_path / "empty_subregion.nii.gz"
    subregion = sitk.Cast(image * 0, sitk.sitkUInt8)
    subregion.CopyInformation(image)
    sitk.WriteImage(subregion, str(subregion_path))

    ctrl._on_finished(
        SegmentationRunResult(
            success=True,
            method="totalsegmentator",
            mask_path=str(mask_path),
            message="ok",
            subregion_mask_path=str(subregion_path),
            subregion_labels={"pedicle": 2},
        )
    )

    assert ctrl._last_pedicle_mask is not None
    assert not ctrl._last_pedicle_mask.any()
    assert "pedicle model ran but found no pedicle voxels" in (
        window.seg_status_label.text()
    )


def test_dragging_a_narrow_screw_into_a_wide_level_repaints_it(ui_main_window):
    """The one visual cue for a narrow pedicle has to follow the drag.

    End to end through the path an edit really takes: the tool re-grades, reads
    the width of the level the screw is now in, and the tool controller redraws
    the 3D actor from the refreshed bundle.
    """
    import numpy as np

    from src.core.screw_grading import ScrewGrader
    from src.utils.constants import COLOR_SCREW, COLOR_SCREW_BREACH

    class _Analysis:
        """Duck-typed pedicle analysis: only the widths are read here."""

        def __init__(self, width):
            self.left_pedicle_width = width
            self.right_pedicle_width = width
            self.width_flags = {}
            self.upper_endplate_normal = None

    window = ui_main_window
    # (z, y, x) at 1 mm with a zero origin: label 28 below z = 30 mm, 29 above.
    arr = np.zeros((60, 60, 60), dtype=np.uint8)
    arr[20:30, 20:40, 20:40] = 28
    arr[30:40, 20:40, 20:40] = 29
    mask = sitk.GetImageFromArray(arr)
    ct = sitk.GetImageFromArray(np.where(arr > 0, 350, -50).astype(np.int16))
    ct.CopyInformation(mask)

    tool = window._tool_ctrl.screw_tool
    tool.set_grader(ScrewGrader(mask, ct))
    tool.set_analysis_by_level({28: _Analysis(4.5), 29: _Analysis(9.2)})

    narrow = Screw(
        entry_point=(30.0, 38.0, 25.0),
        target_point=(30.0, 22.0, 25.0),
        diameter=5.0,
        side="left",
        vertebra_level="L2",
        metrics={"narrow_pedicle": True, "pedicle_width_mm": 4.5},
    )
    index = window._tool_ctrl.add_existing_screw(narrow, select=True)
    assert window.viewer_3d.screws[0].color == pytest.approx(
        tuple(value / 255.0 for value in COLOR_SCREW_BREACH)
    )

    moved = tool.replace_screw(
        index, entry_point=(30.0, 38.0, 35.0), target_point=(30.0, 22.0, 35.0)
    )
    window._tool_ctrl.refresh_screw(index, moved)

    assert moved.metrics["pedicle_width_mm"] == pytest.approx(9.2)
    assert window.viewer_3d.screws[0].color == pytest.approx(
        tuple(value / 255.0 for value in COLOR_SCREW)
    )


def test_screw_mpr_opens_the_3d_view_at_the_cross_section(ui_main_window):
    """The 3D view follows Screw MPR instead of drawing planes no pane shows."""
    window = ui_main_window
    image = _create_test_image()
    window._on_dicom_loaded(
        image=image,
        metadata={"series_id": "SERIES-SCREW-MPR-3D", "num_slices": 12},
        progress=_ProgressStub(),
    )
    screw = Screw(entry_point=(2.0, 3.0, 4.0), target_point=(8.0, 9.0, 24.0), diameter=6.0)
    window._tool_ctrl.screw_tool.add_screw(screw)
    window._tool_ctrl._add_screw_to_list(screw)
    window.screw_list_widget.setCurrentRow(0)

    window.screw_axis_mpr_btn.click()

    planes = window.viewer_3d.screw_mpr_planes
    assert planes is not None
    cross_section = planes[2]
    centre = [cross_section.GetElement(row, 3) for row in range(3)]
    # Position starts at 50 %: the cut is at the screw's midpoint.
    assert centre == pytest.approx([5.0, 6.0, 14.0])

    window._screw_mpr_ctrl.set_position(100)

    centre = [window.viewer_3d.screw_mpr_planes[2].GetElement(row, 3) for row in range(3)]
    assert centre == pytest.approx([8.0, 9.0, 24.0])     # follows Position to the tip

    window.standard_mpr_btn.click()

    assert window.viewer_3d.screw_mpr_planes is None
    assert window.viewer_3d.screw_mpr_clear_count == 1


# ---------------------------------------------------------------------------
# Workflow bar: MainWindow drives WorkflowBar.set_states() from whatever the
# study, segmentation and planning state currently is, and must redraw it
# every time one of those changes -- including the deferred-delete teardown
# path pytest-qt exercises between fixtures (see the regression test at the
# bottom of this section).
# ---------------------------------------------------------------------------


def _load_study(window, series_id):
    window._on_dicom_loaded(
        image=_create_test_image(),
        metadata={"series_id": series_id, "num_slices": 12},
        progress=_ProgressStub(),
    )


def _finish_segmentation(window, tmp_path, monkeypatch, method="totalsegmentator"):
    mask_path = tmp_path / f"{method}_mask.nii.gz"
    _write_mask(_create_test_image(), mask_path)
    if method == "threshold_fallback":
        monkeypatch.setattr(
            seg_controller_module.QMessageBox, "warning", lambda *a, **k: None
        )
    window._on_segmentation_finished(
        SegmentationRunResult(
            success=True,
            method=method,
            mask_path=str(mask_path),
            message="ok",
            geometry_warnings=[],
        )
    )


def _add_screw(window, source):
    screw = Screw(
        entry_point=(2.0, 3.0, 4.0),
        target_point=(8.0, 9.0, 24.0),
        diameter=6.0,
        source=source,
    )
    window._tool_ctrl.screw_tool.add_screw(screw)
    window._tool_ctrl._add_screw_to_list(screw)


def test_workflow_bar_sits_above_the_views_and_starts_at_study(ui_main_window):
    window = ui_main_window
    bar = window.workflow_bar
    b = bar.buttons

    layout = bar.parentWidget().layout()
    assert layout.indexOf(bar) == 0

    assert all(button.isEnabled() for button in b)
    assert b[0].property("role") == "primary"
    assert b[0].text() == "①  Study"
    assert b[1].toolTip() == "Open a DICOM series first"


def test_workflow_bar_advances_to_segment_after_a_study_loads(ui_main_window):
    window = ui_main_window
    _load_study(window, "WF-STUDY-1")
    b = window.workflow_bar.buttons

    assert b[0].text() == "✓ Study"
    assert b[0].property("role") == "secondary"
    assert b[1].property("role") == "primary"
    assert b[1].toolTip() == "Run TotalSegmentator on the loaded study"
    assert b[2].toolTip() == "Run segmentation first"
    assert window.step_panel.current_step == "Segment"


def test_workflow_bar_finishes_segment_for_a_totalsegmentator_mask(
    ui_main_window, tmp_path, monkeypatch
):
    window = ui_main_window
    _load_study(window, "WF-STUDY-2")
    _finish_segmentation(window, tmp_path, monkeypatch)
    b = window.workflow_bar.buttons

    assert b[1].text() == "✓ Segment"
    assert b[2].property("role") == "primary"
    assert b[2].toolTip() == "Select vertebral levels in the Plan step"


def test_workflow_bar_keeps_segment_open_after_a_threshold_fallback(
    ui_main_window, tmp_path, monkeypatch
):
    window = ui_main_window
    _load_study(window, "WF-STUDY-3")
    _finish_segmentation(window, tmp_path, monkeypatch, method="threshold_fallback")
    b = window.workflow_bar.buttons

    assert b[1].text() == "②  Segment"
    assert b[1].property("role") == "primary"


def test_workflow_bar_enables_plan_once_levels_are_selected(
    ui_main_window, tmp_path, monkeypatch
):
    window = ui_main_window
    _load_study(window, "WF-STUDY-4")
    _finish_segmentation(window, tmp_path, monkeypatch)
    window.update_vertebra_level_checks([28, 29, 30])
    b = window.workflow_bar.buttons

    assert window.auto_screw_plan_btn.isEnabled() is True
    assert b[2].toolTip() == "Plan screws for the selected vertebral levels"

    window._toggle_vertebra_level_checks(False)

    assert b[2].toolTip() == "Select vertebral levels in the Plan step"


def test_workflow_bar_finishes_plan_for_auto_screws_but_not_manual_ones(
    ui_main_window, tmp_path, monkeypatch
):
    window = ui_main_window
    _load_study(window, "WF-STUDY-5")
    _finish_segmentation(window, tmp_path, monkeypatch)
    b = window.workflow_bar.buttons

    _add_screw(window, "manual")
    assert b[2].text() == "③  Plan"

    _add_screw(window, "auto")
    assert b[2].text() == "✓ Plan"
    assert b[3].property("role") == "primary"

    window.screw_list_widget.setCurrentRow(1)
    window._tool_ctrl.remove_selected_screw()

    assert b[2].text() == "③  Plan"
    assert b[2].property("role") == "primary"


def test_workflow_bar_resets_when_a_new_study_is_loaded(
    ui_main_window, tmp_path, monkeypatch
):
    """Regression for the stale-bar bug: set_volume() notifies before
    reset_workspace() clears the previous study's mask and screws, so the
    refresh that matters is the one reset_workspace() triggers itself once
    everything is actually cleared -- not one hung on the volume-loaded
    notification, which would still see the old state.
    """
    window = ui_main_window
    _load_study(window, "WF-STUDY-6")
    _finish_segmentation(window, tmp_path, monkeypatch)
    _add_screw(window, "auto")
    b = window.workflow_bar.buttons
    assert b[2].text() == "✓ Plan"

    _load_study(window, "WF-STUDY-SECOND")

    assert b[0].text() == "✓ Study"
    assert b[1].text() == "②  Segment"
    assert b[1].property("role") == "primary"
    assert b[2].text() == "③  Plan"


def test_workflow_bar_reopens_segment_when_the_segmentation_is_cleared(
    ui_main_window, tmp_path, monkeypatch
):
    window = ui_main_window
    _load_study(window, "WF-STUDY-7")
    _finish_segmentation(window, tmp_path, monkeypatch)

    window._seg_ctrl.clear_overlay()
    b = window.workflow_bar.buttons

    assert b[1].text() == "②  Segment"
    assert b[1].property("role") == "primary"


def test_workflow_bar_steps_navigate_to_their_pages(ui_main_window, tmp_path, monkeypatch):
    """Every step now navigates -- it never runs Open DICOM / Segment / Plan
    directly, since those actions live on the pages themselves."""
    window = ui_main_window
    b = window.workflow_bar.buttons

    b[0].click()
    assert window.step_panel.current_step == "Study"

    b[1].click()
    assert window.step_panel.current_step == "Segment"

    b[2].click()
    assert window.step_panel.current_step == "Plan"

    b[3].click()
    assert window.step_panel.current_step == "Review"


@pytest.mark.parametrize("close_first", [True, False])
def test_destroying_a_window_with_screw_rows_raises_nothing(
    monkeypatch, qtbot, isolated_qsettings, close_first
):
    """Regression for the teardown crash: a lambda connected to the screw
    table's row signals had no Qt receiver, so it could still fire -- from a
    DeferredDelete queued by an earlier test's window -- after this
    window's buttons were already gone, raising on the deleted C++ object.
    A bound method dies with the window instead.

    The window is built and torn down by hand rather than through
    qtbot.addWidget/the ui_main_window fixture: this test owns the exact
    teardown sequence (close, then deleteLater, then deliver the deferred
    delete) that reproduced the bug, and qtbot's own fixture teardown would
    add a second, redundant deleteLater on top of it.
    """
    monkeypatch.setattr(main_window_module, "MPRViewer", DummyMPRViewer)
    monkeypatch.setattr(main_window_module, "Viewer3D", DummyViewer3D)
    QApplication.instance().setProperty("themeName", "soft_light")

    window = main_window_module.MainWindow()
    # No _on_dicom_loaded here: it schedules a QTimer.singleShot on the
    # plain-Python DicomController, which would fire after this window is
    # gone if it survived to the timer's timeout.
    _add_screw(window, "auto")
    _add_screw(window, "manual")
    button = window.seg_run_btn

    with qtbot.captureExceptions() as exceptions:
        if close_first:
            window.close()
        window.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete.value)
        QApplication.processEvents()

    assert exceptions == []
    assert sip.isdeleted(button)


# ---------------------------------------------------------------------------
# Automatic vertebra isolation and the 3D header toggle.
# ---------------------------------------------------------------------------


def test_totalsegmentator_result_isolates_vertebrae_automatically(
    ui_main_window, tmp_path, monkeypatch
):
    window = ui_main_window
    _load_study(window, "ISO-STUDY-1")
    mask_path = tmp_path / "totalsegmentator_mask.nii.gz"
    _write_vertebra_mask(_create_test_image(), mask_path)
    window._on_segmentation_finished(
        SegmentationRunResult(
            success=True,
            method="totalsegmentator",
            mask_path=str(mask_path),
            message="ok",
            geometry_warnings=[],
        )
    )

    assert window._seg_ctrl._vertebrae_isolated is True
    assert window.viewer_3d.volume_visible is False
    assert window.viewer_3d.isolation_state == (True, True)
    assert window.vertebra_isolate_btn.text() == "Restore Full Volume"
    assert window.step_panel.current_step == "Plan"


def test_threshold_fallback_does_not_isolate_and_disables_the_toggle(
    ui_main_window, tmp_path, monkeypatch
):
    window = ui_main_window
    _load_study(window, "ISO-STUDY-2")
    monkeypatch.setattr(
        seg_controller_module.QMessageBox, "warning", lambda *a, **k: None
    )
    mask_path = tmp_path / "threshold_fallback_mask.nii.gz"
    _write_mask(_create_test_image(), mask_path)
    window._on_segmentation_finished(
        SegmentationRunResult(
            success=True,
            method="threshold_fallback",
            mask_path=str(mask_path),
            message="fallback",
            geometry_warnings=[],
        )
    )

    assert window._seg_ctrl._vertebrae_isolated is False
    assert window.viewer_3d.isolation_state == (False, False)
    assert window.vertebra_isolate_btn.isEnabled() is False


def test_3d_header_toggle_switches_between_vertebrae_and_full_ct(
    ui_main_window, tmp_path, monkeypatch
):
    window = ui_main_window
    _load_study(window, "ISO-STUDY-3")
    mask_path = tmp_path / "totalsegmentator_mask.nii.gz"
    _write_vertebra_mask(_create_test_image(), mask_path)
    window._on_segmentation_finished(
        SegmentationRunResult(
            success=True,
            method="totalsegmentator",
            mask_path=str(mask_path),
            message="ok",
            geometry_warnings=[],
        )
    )
    assert window._seg_ctrl._vertebrae_isolated is True

    window.viewer_3d.isolation_requested.emit(False)
    assert window._seg_ctrl._vertebrae_isolated is False
    assert window.viewer_3d.volume_visible is True

    window.viewer_3d.isolation_requested.emit(True)
    assert window._seg_ctrl._vertebrae_isolated is True
    assert window.viewer_3d.volume_visible is False


def test_panel_button_and_header_toggle_stay_in_step(
    ui_main_window, tmp_path, monkeypatch
):
    window = ui_main_window
    _load_study(window, "ISO-STUDY-4")
    mask_path = tmp_path / "totalsegmentator_mask.nii.gz"
    _write_vertebra_mask(_create_test_image(), mask_path)
    window._on_segmentation_finished(
        SegmentationRunResult(
            success=True,
            method="totalsegmentator",
            mask_path=str(mask_path),
            message="ok",
            geometry_warnings=[],
        )
    )

    window.vertebra_isolate_btn.click()
    assert window._seg_ctrl._vertebrae_isolated is False
    assert window.viewer_3d.isolation_state == (False, True)
    assert window.vertebra_isolate_btn.text() == "Isolate Vertebrae"

    window.vertebra_isolate_btn.click()
    assert window._seg_ctrl._vertebrae_isolated is True
    assert window.viewer_3d.isolation_state == (True, True)
    assert window.vertebra_isolate_btn.text() == "Restore Full Volume"


def test_loading_a_study_shows_the_segment_step(ui_main_window):
    window = ui_main_window
    assert window.step_panel.current_step == "Study"

    _load_study(window, "ISO-STUDY-5")

    assert window.step_panel.current_step == "Segment"


def test_study_page_open_button_opens_a_folder(monkeypatch, qtbot, isolated_qsettings):
    """open_dicom_btn is wired directly to ``self._dicom_ctrl.open_folder``
    (no lambda), so DicomController.open_folder has to be patched on the
    class before construction -- the connection captures that exact
    bound-method object, and an instance-attribute patch afterwards would
    not reach it."""
    from src.controllers.dicom_controller import DicomController

    calls = []
    monkeypatch.setattr(
        DicomController, "open_folder", lambda self: calls.append("open")
    )
    monkeypatch.setattr(main_window_module, "MPRViewer", DummyMPRViewer)
    monkeypatch.setattr(main_window_module, "Viewer3D", DummyViewer3D)
    QApplication.instance().setProperty("themeName", "soft_light")
    window = main_window_module.MainWindow()
    qtbot.addWidget(window)

    window.open_dicom_btn.click()

    assert calls == ["open"]


def test_window_level_sliders_refresh_the_3d_slice(monkeypatch, qtbot, isolated_qsettings):
    """Screw MPR's on_window_level_changed must hear every window/level
    change too, not just ViewController's -- and since the slider's
    ``.connect(self._screw_mpr_ctrl.on_window_level_changed)`` captures that
    exact bound-method object, the controller's class method has to be
    patched before construction for a test double to see the calls."""
    from src.controllers.screw_mpr_controller import ScrewMPRController

    calls = []
    monkeypatch.setattr(
        ScrewMPRController,
        "on_window_level_changed",
        lambda self, *a: calls.append(a),
    )
    monkeypatch.setattr(main_window_module, "MPRViewer", DummyMPRViewer)
    monkeypatch.setattr(main_window_module, "Viewer3D", DummyViewer3D)
    QApplication.instance().setProperty("themeName", "soft_light")
    window = main_window_module.MainWindow()
    qtbot.addWidget(window)

    window.window_slider.setValue(window.window_slider.value() + 10)
    window.level_slider.setValue(window.level_slider.value() + 10)

    assert len(calls) == 2


def test_screw_counter_refreshes_when_rows_are_added_and_removed(ui_main_window):
    """A planning run selects the first screw while the table has one row,
    then appends the rest without reselecting -- the counter must follow the
    table's row count, not just the last selection change."""
    window = ui_main_window
    first = Screw((0.0, 0.0, 0.0), (0.0, 0.0, 30.0))
    window._tool_ctrl.add_existing_screw(first, select=True)
    window._tool_ctrl.add_existing_screw(Screw((5.0, 5.0, 5.0), (5.0, 5.0, 35.0)))
    window._tool_ctrl.add_existing_screw(Screw((9.0, 9.0, 9.0), (9.0, 9.0, 39.0)))

    assert window.selected_screw_counter.text() == "Screw 1 of 3"

    while window.screw_list_widget.count() > 0:
        window.screw_list_widget.setCurrentRow(0)
        window._tool_ctrl.remove_selected_screw()

    assert window.selected_screw_counter.text() == "No screws"


def test_screw_counter_shows_the_count_when_rows_exist_but_none_is_selected(
    ui_main_window,
):
    """Rows added without a selection (as a loaded plan used to leave them)
    must not read "No screws" -- the count is right there in the table."""
    window = ui_main_window
    first = Screw((0.0, 0.0, 0.0), (0.0, 0.0, 30.0))
    second = Screw((5.0, 5.0, 5.0), (5.0, 5.0, 35.0))
    window._tool_ctrl.screw_tool.add_screw(first)
    window._tool_ctrl._add_screw_to_list(first)
    window._tool_ctrl.screw_tool.add_screw(second)
    window._tool_ctrl._add_screw_to_list(second)

    assert window.screw_list_widget.currentRow() == -1
    assert window.selected_screw_counter.text() == "2 screws"


def test_segmentation_advanced_options_toggle_shows_and_hides_panel(
    ui_main_window,
):
    window = ui_main_window

    assert "Advanced options" in window.seg_advanced_toggle.text()
    assert window.seg_advanced_toggle.isCheckable()
    assert window.seg_advanced_panel.isHidden()

    window.seg_advanced_toggle.setChecked(True)
    assert not window.seg_advanced_panel.isHidden()

    window.seg_advanced_toggle.setChecked(False)
    assert window.seg_advanced_panel.isHidden()


def test_isolation_hides_the_3d_overlay_and_full_ct_honours_show_3d(
    ui_main_window, tmp_path, monkeypatch
):
    """Returning to Full CT must not turn the 3D overlay on against "Show 3D".

    Viewer3D.set_volume_visible used to switch the overlay with the volume, so
    restoring Full CT re-showed it while the checkbox read unchecked.  Now the
    volume and the overlay are separate: isolation hides the overlay itself,
    and restoring hands it back to the checkbox.
    """
    window = ui_main_window
    _load_study(window, "ISO-STUDY-OVERLAY")
    mask_path = tmp_path / "totalsegmentator_mask.nii.gz"
    _write_vertebra_mask(_create_test_image(), mask_path)
    window._on_segmentation_finished(
        SegmentationRunResult(
            success=True,
            method="totalsegmentator",
            mask_path=str(mask_path),
            message="ok",
            geometry_warnings=[],
        )
    )
    assert window._seg_ctrl._vertebrae_isolated is True
    assert window.viewer_3d.visible is False          # the mesh replaces it

    window.seg_show_3d_check.setChecked(False)
    window.viewer_3d.isolation_requested.emit(False)  # Full CT
    assert window._seg_ctrl._vertebrae_isolated is False
    assert window.viewer_3d.volume_visible is True
    assert window.viewer_3d.visible is False          # checkbox still wins

    window.seg_show_3d_check.setChecked(True)
    assert window.viewer_3d.visible is True
