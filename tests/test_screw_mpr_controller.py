"""Tests for explicit screw-aligned MPR mode."""

from types import SimpleNamespace

import pytest

from src.controllers.screw_mpr_controller import (
    SCREW_MPR_CONTROLS_HELP,
    ScrewMPRController,
)
from src.models.screw import Screw


class _Viewer:
    def __init__(self):
        self.axes = None
        self.title = None
        self.readout = None
        self.clear_count = 0
        self.review_screw_id = None

    def set_custom_reslice_axes(self, axes, title):
        self.axes = axes
        self.title = title

    def set_custom_readout(self, text):
        self.readout = text

    def clear_custom_reslice_axes(self):
        self.axes = None
        self.title = None
        self.readout = None
        self.clear_count += 1

    def set_review_screw(self, screw_id):
        self.review_screw_id = screw_id


class _Button:
    def __init__(self):
        self.enabled = False

    def setEnabled(self, enabled):
        self.enabled = bool(enabled)


class _Slider:
    def __init__(self):
        self.enabled = False
        self.current_value = 50
        self.signals_blocked = False

    def setEnabled(self, enabled):
        self.enabled = bool(enabled)

    def setValue(self, value):
        self.current_value = int(value)

    def blockSignals(self, blocked):
        previous = self.signals_blocked
        self.signals_blocked = bool(blocked)
        return previous


class _SpinBox:
    def __init__(self):
        self.enabled = False
        self.current_value = 0
        self.signals_blocked = False

    def setEnabled(self, enabled):
        self.enabled = bool(enabled)

    def setValue(self, value):
        self.current_value = int(value)

    def value(self):
        return self.current_value

    def blockSignals(self, blocked):
        previous = self.signals_blocked
        self.signals_blocked = bool(blocked)
        return previous


class _Label:
    def __init__(self):
        self.text = ""

    def setText(self, text):
        self.text = text


class _ListWidget:
    def __init__(self, row=-1):
        self.row = row

    def currentRow(self):
        return self.row

    def setCurrentRow(self, row):
        self.row = int(row)


class _StatusBar:
    def __init__(self):
        self.message = ""

    def showMessage(self, message):
        self.message = message


class _VolumeManager:
    def __init__(self, loaded=True):
        self.image = object() if loaded else None

    def get_vtk_image(self):
        return self.image


class _ScrewTool:
    def __init__(self, screws):
        self.screws = screws

    def get_screws(self):
        return list(self.screws)


class _Window:
    def __init__(self, screws, row=0):
        self.axial_viewer = _Viewer()
        self.sagittal_viewer = _Viewer()
        self.coronal_viewer = _Viewer()
        self.screw_list_widget = _ListWidget(row)
        self.screw_axis_mpr_btn = _Button()
        self.standard_mpr_btn = _Button()
        self.screw_axis_position_slider = _Slider()
        self.screw_axis_position_label = _Label()
        self.screw_axis_rotation_spin = _SpinBox()
        self.screw_mpr_reset_btn = _Button()
        self.statusbar = _StatusBar()
        self._tool_ctrl = SimpleNamespace(screw_tool=_ScrewTool(screws))
        self.layout_mode = None
        self.inspector_updates = []

    def set_view_layout(self, mode):
        self.layout_mode = mode

    def _get_mpr_viewers(self):
        return [self.axial_viewer, self.sagittal_viewer, self.coronal_viewer]

    def update_selected_screw_inspector(self, index, screw, review_active):
        self.inspector_updates.append((index, screw, review_active))


def _screw(entry=(0.0, 0.0, 0.0), target=(0.0, 0.0, 40.0)):
    return Screw(entry_point=entry, target_point=target, diameter=6.0)


def _make_controller(screws, row=0, loaded=True):
    window = _Window(screws, row=row)
    controller = ScrewMPRController(_VolumeManager(loaded=loaded), window)
    return controller, window


def test_no_selection_does_not_enter_screw_axis_mode():
    controller, window = _make_controller([_screw()], row=-1)

    assert controller.enter() is False
    assert controller.is_active is False
    assert "Select a screw" in window.statusbar.message


def test_valid_screw_applies_two_long_axes_and_midpoint_cross_section():
    controller, window = _make_controller([_screw()])

    assert controller.enter() is True

    assert controller.is_active is True
    assert window.layout_mode == "planning"
    assert window.axial_viewer.title == "Oblique Axial · Screw #1"
    assert window.sagittal_viewer.title == "Oblique Sagittal · Screw #1"
    assert window.coronal_viewer.title == "Cross-section · 50%"
    assert window.coronal_viewer.axes.GetElement(2, 3) == pytest.approx(20.0)
    assert window.standard_mpr_btn.enabled is True
    assert window.screw_axis_position_slider.enabled is True
    assert all(
        viewer.review_screw_id == 0
        for viewer in window._get_mpr_viewers()
    )


def test_position_slider_moves_only_cross_section_along_screw():
    controller, window = _make_controller([_screw()])
    controller.enter()
    original_long_axis = window.axial_viewer.axes

    controller.set_position(75)

    assert window.axial_viewer.axes is not original_long_axis
    assert window.coronal_viewer.axes.GetElement(2, 3) == pytest.approx(30.0)
    assert "75%" in window.screw_axis_position_label.text
    assert "30.0 mm" in window.screw_axis_position_label.text


def test_active_review_realigns_after_screw_update_at_same_percent():
    controller, window = _make_controller([_screw()])
    controller.enter()
    controller.set_position(75)
    window._tool_ctrl.screw_tool.screws[0] = _screw(
        target=(20.0, 0.0, 40.0)
    )

    controller.on_screw_updated(0)

    assert controller.position_percent == 75
    assert window.coronal_viewer.title == "Cross-section · 75%"
    assert window.coronal_viewer.axes.GetElement(0, 3) == pytest.approx(15.0)
    assert window.coronal_viewer.axes.GetElement(2, 3) == pytest.approx(30.0)


def test_active_screw_edit_keeps_ct_reslice_axes_fixed_while_geometry_moves():
    controller, window = _make_controller([_screw()])
    controller.enter()
    original_axes = tuple(
        viewer.axes
        for viewer in window._get_mpr_viewers()
    )
    window._screw_edit_ctrl = SimpleNamespace(is_active=True)
    window._tool_ctrl.screw_tool.screws[0] = _screw(
        entry=(10.0, 0.0, 0.0),
        target=(20.0, 0.0, 40.0),
    )

    controller.on_screw_updated(0)

    assert tuple(
        viewer.axes
        for viewer in window._get_mpr_viewers()
    ) == original_axes
    assert window.inspector_updates[-1][1].entry_point == (10.0, 0.0, 0.0)


def test_active_3d_edit_realigns_review_mpr_to_changed_screw():
    controller, window = _make_controller([_screw()])
    controller.enter()
    original_axes = window.axial_viewer.axes
    window._screw_edit_ctrl = SimpleNamespace(
        is_active=True,
        active_source="3D",
    )
    window._tool_ctrl.screw_tool.screws[0] = _screw(
        entry=(10.0, 0.0, 0.0),
        target=(20.0, 0.0, 40.0),
    )

    controller.on_screw_updated(0)

    assert window.axial_viewer.axes is not original_axes
    assert window.coronal_viewer.axes.GetElement(0, 3) == pytest.approx(15.0)


def test_deleting_active_screw_restores_standard_mpr():
    controller, window = _make_controller([_screw()])
    controller.enter()

    controller.on_screw_removed(0)

    assert controller.is_active is False
    assert all(
        viewer.clear_count == 1
        for viewer in (
            window.axial_viewer,
            window.sagittal_viewer,
            window.coronal_viewer,
        )
    )
    assert window.standard_mpr_btn.enabled is False
    assert window.screw_axis_position_slider.enabled is False
    assert all(
        viewer.review_screw_id is None
        for viewer in window._get_mpr_viewers()
    )


def test_zero_length_screw_is_rejected_without_changing_views():
    controller, window = _make_controller([_screw(target=(0.0, 0.0, 0.0))])

    assert controller.enter() is False

    assert controller.is_active is False
    assert window.axial_viewer.axes is None
    assert "zero-length" in window.statusbar.message


def test_active_review_realigns_when_another_screw_is_selected():
    controller, window = _make_controller(
        [
            _screw(target=(0.0, 0.0, 40.0)),
            _screw(target=(30.0, 0.0, 0.0)),
        ]
    )
    controller.enter()
    first = tuple(
        window.axial_viewer.axes.GetElement(row, column)
        for row in range(4)
        for column in range(4)
    )

    window.screw_list_widget.setCurrentRow(1)
    controller.on_screw_selection_changed(1)
    second = tuple(
        window.axial_viewer.axes.GetElement(row, column)
        for row in range(4)
        for column in range(4)
    )

    assert second != first
    assert window.axial_viewer.title == "Oblique Axial · Screw #2"
    assert window.inspector_updates[-1][0] == 1
    assert window.inspector_updates[-1][2] is True
    assert all(
        viewer.review_screw_id == 1
        for viewer in window._get_mpr_viewers()
    )


def test_deleting_screw_before_active_selection_reindexes_review_filter():
    controller, window = _make_controller(
        [_screw(), _screw(target=(20.0, 0.0, 30.0))],
        row=1,
    )
    controller.enter()

    controller.on_screw_removed(0)

    assert all(
        viewer.review_screw_id == 0
        for viewer in window._get_mpr_viewers()
    )


def test_previous_and_next_change_list_selection():
    controller, window = _make_controller(
        [_screw(), _screw(), _screw()],
        row=1,
    )

    controller.select_previous()
    assert window.screw_list_widget.currentRow() == 0

    controller.select_next()
    assert window.screw_list_widget.currentRow() == 1


def test_selection_updates_inspector_in_standard_mode():
    controller, window = _make_controller([_screw()], row=0)

    controller.on_screw_selection_changed(0)

    assert window.inspector_updates[-1] == (
        0,
        window._tool_ctrl.screw_tool.screws[0],
        False,
    )


def test_plain_wheel_on_cross_section_moves_one_mm_per_notch():
    # 40 mm screw: two notches = 2 mm = 5 % of the trajectory.
    controller, window = _make_controller([_screw()])
    controller.enter()

    controller.handle_scroll("coronal", 2, frozenset())

    assert controller.position_percent == 55
    assert window.screw_axis_position_slider.current_value == 55
    assert window.coronal_viewer.axes.GetElement(2, 3) == pytest.approx(22.0)
    assert window.coronal_viewer.title == "Cross-section · 55%"


def test_shift_wheel_rotates_five_degrees_per_notch_on_any_plane():
    controller, window = _make_controller([_screw()])
    controller.enter()

    controller.handle_scroll("axial", 1, frozenset({"shift"}))
    controller.handle_scroll("coronal", 2, frozenset({"shift"}))

    assert controller.rotation_deg == pytest.approx(15.0)
    assert window.screw_axis_rotation_spin.current_value == 15
    assert controller.position_percent == 50


def test_wheel_on_a_long_axis_view_offsets_only_that_plane():
    # Screw along +Z: superior fallback = +X, transverse = -Y.
    controller, window = _make_controller([_screw()])
    controller.enter()

    controller.handle_scroll("axial", 3, frozenset())

    assert controller.long_axis_offsets_mm == pytest.approx((3.0, 0.0))
    assert window.axial_viewer.axes.GetElement(0, 3) == pytest.approx(3.0)
    assert window.sagittal_viewer.axes.GetElement(0, 3) == pytest.approx(0.0)

    controller.handle_scroll("sagittal", -2, frozenset())

    assert controller.long_axis_offsets_mm == pytest.approx((3.0, -2.0))
    assert window.sagittal_viewer.axes.GetElement(1, 3) == pytest.approx(2.0)
    assert controller.position_percent == 50


def test_wheel_is_ignored_when_screw_mpr_is_not_active():
    controller, window = _make_controller([_screw()])

    controller.handle_scroll("coronal", 5, frozenset())
    controller.handle_scroll("axial", 5, frozenset({"shift"}))

    assert controller.position_percent == 50
    assert controller.rotation_deg == 0.0
    assert window.coronal_viewer.axes is None


def test_ctrl_drag_rotation_only_applies_to_the_cross_section():
    controller, _window = _make_controller([_screw()])
    controller.enter()

    controller.handle_rotate_drag("coronal", 12.0)
    assert controller.rotation_deg == pytest.approx(12.0)

    controller.handle_rotate_drag("axial", 30.0)
    assert controller.rotation_deg == pytest.approx(12.0)


def test_rotation_is_clamped_to_plus_minus_one_hundred_eighty_degrees():
    controller, _window = _make_controller([_screw()])
    controller.enter()

    controller.set_rotation(400.0)
    assert controller.rotation_deg == pytest.approx(180.0)

    controller.set_rotation(-400.0)
    assert controller.rotation_deg == pytest.approx(-180.0)


def test_reset_view_restores_position_rotation_and_offsets():
    controller, window = _make_controller([_screw()])
    controller.enter()
    controller.set_position(80)
    controller.set_rotation(30.0)
    controller.nudge_plane("oblique_axial", 4)
    controller.nudge_plane("oblique_sagittal", -3)

    controller.reset_view()

    assert controller.position_percent == 50
    assert controller.rotation_deg == 0.0
    assert controller.long_axis_offsets_mm == (0.0, 0.0)
    assert controller.cross_section_offset_mm == (0.0, 0.0)
    assert window.screw_axis_rotation_spin.current_value == 0
    assert window.screw_axis_position_slider.current_value == 50
    assert window.coronal_viewer.axes.GetElement(2, 3) == pytest.approx(20.0)
    assert window.axial_viewer.axes.GetElement(0, 3) == pytest.approx(0.0)


def test_exit_clears_rotation_and_plane_offsets():
    controller, window = _make_controller([_screw()])
    controller.enter()
    controller.set_rotation(45.0)
    controller.nudge_plane("oblique_axial", 2)

    controller.exit()

    assert controller.rotation_deg == 0.0
    assert controller.long_axis_offsets_mm == (0.0, 0.0)
    assert controller.position_percent == 50
    assert window.screw_axis_rotation_spin.current_value == 0
    assert all(
        viewer.readout is None for viewer in window._get_mpr_viewers()
    )


def test_readout_reports_rotation_and_the_plane_specific_offset():
    controller, window = _make_controller([_screw()])
    controller.enter()
    controller.set_rotation(15.0)
    controller.nudge_plane("oblique_axial", 2)

    assert window.axial_viewer.readout == "Screw-aligned · rot 15° · +2.0 mm"
    assert window.sagittal_viewer.readout == (
        "Screw-aligned · rot 15° · +0.0 mm"
    )
    assert window.coronal_viewer.readout == (
        "Screw-aligned · rot 15° · 20.0 mm from entry"
    )


def test_rotation_spin_and_reset_button_follow_the_active_state():
    controller, window = _make_controller([_screw()])

    controller.refresh_controls()
    assert window.screw_axis_rotation_spin.enabled is False
    assert window.screw_mpr_reset_btn.enabled is False

    controller.enter()
    assert window.screw_axis_rotation_spin.enabled is True
    assert window.screw_mpr_reset_btn.enabled is True

    controller.exit()
    assert window.screw_axis_rotation_spin.enabled is False
    assert window.screw_mpr_reset_btn.enabled is False


def test_help_text_documents_every_screw_mpr_gesture():
    for fragment in (
        "Wheel",
        "Shift",
        "Ctrl",
        "Middle",
        "right-hand rule",
    ):
        assert fragment in SCREW_MPR_CONTROLS_HELP
