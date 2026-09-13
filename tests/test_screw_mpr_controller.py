"""Tests for explicit screw-aligned MPR mode."""

from types import SimpleNamespace

import pytest

from src.controllers.screw_mpr_controller import (
    SCREW_MPR_CONTROLS_HELP,
    ScrewMPRController,
)
from src.core.mpr_geometry import build_screw_mpr_axes
from src.models.screw import Screw


def _elements(matrix):
    """Flatten a vtkMatrix4x4 into a 16-tuple for element-wise comparison."""
    return tuple(
        matrix.GetElement(row, column)
        for row in range(4)
        for column in range(4)
    )


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
    def __init__(self, screws, grader=None):
        self.screws = screws
        self.grader = grader

    def get_screws(self):
        return list(self.screws)


class _Grader:
    """Stub grader: returns a fixed label, or raises when told to."""

    def __init__(self, label=None, raises=False):
        self.label = label
        self.raises = raises
        self.calls = []

    def detect_label(self, entry, target):
        self.calls.append((entry, target))
        if self.raises:
            raise RuntimeError("grader boom")
        return self.label


class _WLSlider:
    """Stand-in for the Study tab's Window/Level QSlider."""

    def __init__(self, value):
        self._value = value

    def value(self):
        return self._value

    def setValue(self, value):
        self._value = value


class _Viewer3DRecorder:
    """Records every show_screw_mpr / clear_screw_mpr call for assertions."""

    def __init__(self):
        self.calls = []
        self.clear_count = 0

    def show_screw_mpr(self, oblique_axial, oblique_sagittal, cross_section, **kwargs):
        self.calls.append(
            {
                "oblique_axial": oblique_axial,
                "oblique_sagittal": oblique_sagittal,
                "cross_section": cross_section,
                "vertebra_label": kwargs.get("vertebra_label"),
                "window_level": kwargs.get("window_level"),
            }
        )

    def clear_screw_mpr(self):
        self.clear_count += 1


class _Window:
    def __init__(
        self,
        screws,
        row=0,
        viewer_3d=None,
        grader=None,
        window_level=None,
    ):
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
        self._tool_ctrl = SimpleNamespace(screw_tool=_ScrewTool(screws, grader=grader))
        self.layout_mode = None
        self.inspector_updates = []
        self.viewer_3d = viewer_3d
        if window_level is not None:
            self.window_slider = _WLSlider(window_level[0])
            self.level_slider = _WLSlider(window_level[1])

    def set_view_layout(self, mode):
        self.layout_mode = mode

    def _get_mpr_viewers(self):
        return [self.axial_viewer, self.sagittal_viewer, self.coronal_viewer]

    def update_selected_screw_inspector(self, index, screw, review_active):
        self.inspector_updates.append((index, screw, review_active))


def _screw(entry=(0.0, 0.0, 0.0), target=(0.0, 0.0, 40.0), vertebra_level=""):
    return Screw(
        entry_point=entry,
        target_point=target,
        diameter=6.0,
        vertebra_level=vertebra_level,
    )


def _make_controller(
    screws,
    row=0,
    loaded=True,
    viewer_3d=None,
    grader=None,
    window_level=None,
):
    window = _Window(
        screws,
        row=row,
        viewer_3d=viewer_3d,
        grader=grader,
        window_level=window_level,
    )
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


def test_rotation_wraps_instead_of_clamping_at_the_boundary():
    # Convention: -180 <= rotation_deg < 180, so the boundary itself
    # normalises to -180 rather than +180.
    controller, _window = _make_controller([_screw()])
    controller.enter()

    controller.set_rotation(175.0)
    controller.rotate(10.0)
    assert controller.rotation_deg == pytest.approx(-175.0)

    controller.set_rotation(540.0)
    assert controller.rotation_deg == pytest.approx(-180.0)

    controller.set_rotation(-400.0)
    assert controller.rotation_deg == pytest.approx(-40.0)


def test_a_full_turn_of_drag_returns_to_the_starting_rotation():
    controller, _window = _make_controller([_screw()])
    controller.enter()

    controller.set_rotation(20.0)
    for _ in range(36):
        controller.rotate(10.0)

    assert controller.rotation_deg == pytest.approx(20.0)


def test_rotation_spin_box_shows_the_normalised_value():
    controller, window = _make_controller([_screw()])
    controller.enter()

    controller.set_rotation(185.0)

    assert controller.rotation_deg == pytest.approx(-175.0)
    assert window.screw_axis_rotation_spin.current_value == -175


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


def test_3d_view_receives_the_graders_label():
    viewer_3d = _Viewer3DRecorder()
    grader = _Grader(label=28)
    controller, _window = _make_controller(
        [_screw(vertebra_level="L3")], viewer_3d=viewer_3d, grader=grader,
    )

    controller.enter()

    assert len(viewer_3d.calls) == 1
    assert viewer_3d.calls[0]["vertebra_label"] == 28
    # The grader is given the raw entry/target points, not the level name.
    assert grader.calls == [((0.0, 0.0, 0.0), (0.0, 0.0, 40.0))]


def test_label_falls_back_to_the_screws_level_name():
    viewer_3d = _Viewer3DRecorder()
    controller, _window = _make_controller(
        [_screw(vertebra_level="L4")], viewer_3d=viewer_3d, grader=None,
    )

    controller.enter()

    assert viewer_3d.calls[0]["vertebra_label"] == 28


def test_label_is_none_without_a_grader_or_a_recognised_level():
    viewer_3d = _Viewer3DRecorder()
    controller, _window = _make_controller(
        [_screw(vertebra_level="")], viewer_3d=viewer_3d, grader=None,
    )

    controller.enter()

    assert viewer_3d.calls[0]["vertebra_label"] is None


def test_a_failing_grader_falls_back_to_the_level_name():
    viewer_3d = _Viewer3DRecorder()
    grader = _Grader(raises=True)
    controller, _window = _make_controller(
        [_screw(vertebra_level="L4")], viewer_3d=viewer_3d, grader=grader,
    )

    controller.enter()

    assert viewer_3d.calls[0]["vertebra_label"] == 28


def test_label_is_cached_per_screw_geometry():
    viewer_3d = _Viewer3DRecorder()
    grader = _Grader(label=28)
    controller, _window = _make_controller(
        [_screw(vertebra_level="L3")], viewer_3d=viewer_3d, grader=grader,
    )
    controller.enter()
    assert len(grader.calls) == 1

    # Position-only changes keep the same entry/target -- must not re-sample.
    controller.set_position(75)

    assert len(grader.calls) == 1
    assert viewer_3d.calls[-1]["vertebra_label"] == 28


def test_exit_clears_the_label_cache_so_a_new_grader_is_resampled():
    """A re-run segmentation between sessions must not keep a stale label.

    Same screw geometry (same row, entry, target) across two sessions, but
    the grader (segmentation) changes in between -- exit() must drop the
    cache so the new session re-samples instead of replaying the old label.
    """
    viewer_3d = _Viewer3DRecorder()
    old_grader = _Grader(label=28)
    screw = _screw(vertebra_level="L3")
    controller, window = _make_controller(
        [screw], viewer_3d=viewer_3d, grader=old_grader,
    )
    controller.enter()
    assert viewer_3d.calls[-1]["vertebra_label"] == 28
    controller.exit()

    new_grader = _Grader(label=29)
    window._tool_ctrl.screw_tool.grader = new_grader

    controller.enter()

    assert len(new_grader.calls) == 1
    assert viewer_3d.calls[-1]["vertebra_label"] == 29


@pytest.mark.parametrize(
    "level, old_grader",
    [
        ("", None),
        ("L5", _Grader(label=27)),
    ],
)
def test_a_replaced_grader_is_resampled_without_leaving_screw_mpr(
    level, old_grader,
):
    """A re-run segmentation replaces the grader without exiting Screw MPR.

    A segmentation re-run swaps the grader via ``screw_tool.set_grader(...)``,
    then ``ToolController.regrade_all -> refresh_screw -> on_screw_updated``
    reapplies the active screw without calling ``exit()`` -- the cached label
    from the old grader (or the old level-name fallback) must not survive
    that swap.
    """
    viewer_3d = _Viewer3DRecorder()
    controller, window = _make_controller(
        [_screw(vertebra_level=level)], viewer_3d=viewer_3d, grader=old_grader,
    )
    controller.enter()
    expected_old_label = None if old_grader is None else 27
    assert viewer_3d.calls[-1]["vertebra_label"] == expected_old_label

    new_grader = _Grader(label=28)
    window._tool_ctrl.screw_tool.grader = new_grader
    window._tool_ctrl.screw_tool.screws[0] = _screw(vertebra_level="L4")

    controller.on_screw_updated(0)

    assert controller.is_active is True
    assert new_grader.calls == [((0.0, 0.0, 0.0), (0.0, 0.0, 40.0))]
    assert viewer_3d.calls[-1]["vertebra_label"] == 28


def test_a_detached_grader_drops_the_cached_label():
    """SegmentationController.reset_state calls screw_tool.set_grader(None)
    while Screw MPR stays active."""
    viewer_3d = _Viewer3DRecorder()
    grader = _Grader(label=28)
    controller, window = _make_controller(
        [_screw(vertebra_level="")], viewer_3d=viewer_3d, grader=grader,
    )
    controller.enter()
    assert viewer_3d.calls[-1]["vertebra_label"] == 28

    window._tool_ctrl.screw_tool.grader = None
    controller.on_screw_updated(0)

    assert viewer_3d.calls[-1]["vertebra_label"] is None


def test_window_level_comes_from_the_panel_sliders():
    viewer_3d = _Viewer3DRecorder()
    controller, _window = _make_controller(
        [_screw()], viewer_3d=viewer_3d, window_level=(1200.0, 300.0),
    )

    controller.enter()

    assert viewer_3d.calls[0]["window_level"] == pytest.approx((1200.0, 300.0))


def test_window_level_is_none_without_panel_sliders():
    viewer_3d = _Viewer3DRecorder()
    controller, _window = _make_controller(
        [_screw()], viewer_3d=viewer_3d, window_level=None,
    )

    controller.enter()

    assert viewer_3d.calls[0]["window_level"] is None


def test_on_window_level_changed_repushes_only_while_active():
    viewer_3d = _Viewer3DRecorder()
    controller, _window = _make_controller(
        [_screw()], viewer_3d=viewer_3d, window_level=(1500.0, 400.0),
    )

    # Inactive: must not push anything (and must not raise for lack of state).
    controller.on_window_level_changed()
    assert len(viewer_3d.calls) == 0

    controller.enter()
    assert len(viewer_3d.calls) == 1

    controller.on_window_level_changed()

    assert len(viewer_3d.calls) == 2
    assert viewer_3d.calls[-1]["window_level"] == pytest.approx((1500.0, 400.0))


def test_exit_clears_the_3d_screw_mpr_view():
    viewer_3d = _Viewer3DRecorder()
    controller, _window = _make_controller([_screw()], viewer_3d=viewer_3d)
    controller.enter()

    controller.exit()

    assert viewer_3d.clear_count == 1


def test_3d_view_receives_the_same_three_planes_as_the_2d_viewers():
    """show_screw_mpr's three matrices must match the 2D viewers exactly.

    Guards against swapping the oblique_axial/oblique_sagittal arguments (or
    passing a different matrix than the 2D viewers got) in _apply_screw's 3D
    call.
    """
    viewer_3d = _Viewer3DRecorder()
    controller, window = _make_controller([_screw()], viewer_3d=viewer_3d)
    controller.enter()
    controller.set_rotation(15.0)
    controller.nudge_plane("oblique_axial", 2)
    controller.set_position(75)

    expected = build_screw_mpr_axes(
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 40.0),
        0.75,
        rotation_deg=15.0,
        long_axis_offset_mm=(2.0, 0.0),
        cross_section_offset_mm=(0.0, 0.0),
    )
    call = viewer_3d.calls[-1]

    for name in ("oblique_axial", "oblique_sagittal", "cross_section"):
        assert _elements(call[name]) == pytest.approx(
            _elements(getattr(expected, name))
        )
    assert call["oblique_axial"] is window.axial_viewer.axes
    assert call["oblique_sagittal"] is window.sagittal_viewer.axes
    assert call["cross_section"] is window.coronal_viewer.axes
    # Not vacuous: the two long-axis planes must actually differ.
    assert _elements(call["oblique_axial"]) != _elements(call["oblique_sagittal"])
