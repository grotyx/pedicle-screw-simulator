"""Tests for one-click MPR screw trajectory editing."""

from types import SimpleNamespace

import pytest

from src.controllers.screw_edit_controller import ScrewEditController
from src.models.screw import Screw
from src.tools.screw_tool import ScrewTool


class _VolumeManager:
    bounds = (-100.0, 100.0, -100.0, 100.0, -100.0, 100.0)

    def __init__(self):
        self.crosshair_updates = []

    def get_vtk_image(self):
        return object()

    def get_voxel_value(self, _x, _y, _z):
        return 1000.0

    def set_crosshair_position(self, x, y, z, source_plane=None):
        self.crosshair_updates.append(((x, y, z), source_plane))


class _ListWidget:
    def __init__(self, row=0):
        self.row = row

    def currentRow(self):
        return self.row

    def setCurrentRow(self, row):
        self.row = int(row)


class _Button:
    def __init__(self):
        self.enabled = False
        self.checked = False

    def setEnabled(self, enabled):
        self.enabled = bool(enabled)

    def setChecked(self, checked):
        self.checked = bool(checked)


class _StatusBar:
    def __init__(self):
        self.message = ""

    def showMessage(self, message):
        self.message = str(message)


class _ToolController:
    def __init__(self, screw):
        self.screw_tool = ScrewTool(_VolumeManager())
        self.screw_tool.add_screw(screw)
        self.refreshed = []
        self.active_tool = None

    def set_tool(self, tool):
        self.active_tool = tool

    def refresh_screw(self, index, screw):
        self.refreshed.append((index, screw))


class _ScrewMPRController:
    def __init__(self):
        self.updated = []

    def on_screw_updated(self, index):
        self.updated.append(int(index))


class _CheckableButton(_Button):
    pass


class _Action:
    def __init__(self):
        self.enabled = True
        self.checked = False

    def setEnabled(self, enabled):
        self.enabled = bool(enabled)

    def setChecked(self, checked):
        self.checked = bool(checked)


class _Window:
    def __init__(self, screw):
        self.screw_list_widget = _ListWidget()
        self.screw_edit_btn = _CheckableButton()
        self._screw_edit_move_entry_action = _Action()
        self._screw_edit_move_tip_action = _Action()
        self._screw_edit_move_whole_action = _Action()
        self._screw_edit_cancel_action = _Action()
        # Thin shims for the removed hidden legacy buttons.
        self.screw_edit_entry_btn = self._screw_edit_move_entry_action
        self.screw_edit_tip_btn = self._screw_edit_move_tip_action
        self.screw_edit_move_btn = self._screw_edit_move_whole_action
        self.screw_edit_cancel_btn = self._screw_edit_cancel_action
        self.statusbar = _StatusBar()
        self._tool_ctrl = _ToolController(screw)
        self._screw_mpr_ctrl = _ScrewMPRController()


def _make_controller(screw=None):
    screw = screw or Screw(
        entry_point=(0.0, 0.0, 0.0),
        target_point=(0.0, 0.0, 40.0),
        diameter=6.0,
    )
    window = _Window(screw)
    controller = ScrewEditController(_VolumeManager(), window)
    return controller, window


def test_entry_mode_changes_only_entry():
    controller, window = _make_controller()

    assert controller.start("entry") is True
    assert controller.handle_click("axial", 10.0, 5.0, 2.0) is True

    screw = window._tool_ctrl.screw_tool.get_screws()[0]
    assert screw.entry_point == (10.0, 5.0, 2.0)
    assert screw.target_point == (0.0, 0.0, 40.0)
    assert controller.mode == "idle"
    assert window._tool_ctrl.refreshed == [(0, screw)]


def test_tip_mode_changes_only_tip():
    controller, window = _make_controller()

    controller.start("tip")
    controller.handle_click("sagittal", 5.0, 4.0, 30.0)

    screw = window._tool_ctrl.screw_tool.get_screws()[0]
    assert screw.entry_point == (0.0, 0.0, 0.0)
    assert screw.target_point == (5.0, 4.0, 30.0)
    assert controller.mode == "idle"


def test_move_places_midpoint_at_click_and_preserves_vector():
    controller, window = _make_controller()

    controller.start("move")
    controller.handle_click("coronal", 10.0, 20.0, 30.0)

    screw = window._tool_ctrl.screw_tool.get_screws()[0]
    assert screw.entry_point == pytest.approx((10.0, 20.0, 10.0))
    assert screw.target_point == pytest.approx((10.0, 20.0, 50.0))
    assert controller.mode == "idle"


def test_short_result_is_rejected_and_mode_stays_active():
    controller, window = _make_controller()

    controller.start("tip")
    controller.handle_click("axial", 0.0, 0.0, 0.5)

    screw = window._tool_ctrl.screw_tool.get_screws()[0]
    assert screw.target_point == (0.0, 0.0, 40.0)
    assert controller.mode == "tip"
    assert window._tool_ctrl.refreshed == []
    assert "at least 1.0 mm" in window.statusbar.message


def test_out_of_bounds_result_is_rejected():
    controller, window = _make_controller()

    controller.start("entry")
    controller.handle_click("axial", 150.0, 0.0, 0.0)

    screw = window._tool_ctrl.screw_tool.get_screws()[0]
    assert screw.entry_point == (0.0, 0.0, 0.0)
    assert controller.mode == "entry"
    assert "inside the CT volume" in window.statusbar.message


def test_cancel_returns_to_idle_without_model_change():
    controller, window = _make_controller()
    original = window._tool_ctrl.screw_tool.get_screws()[0]

    controller.start("move")
    controller.cancel()

    assert controller.mode == "idle"
    assert window._tool_ctrl.screw_tool.get_screws()[0] is original


def test_selection_change_cancels_active_edit():
    controller, window = _make_controller()
    controller.start("entry")

    window.screw_list_widget.row = -1
    controller.on_screw_selection_changed(-1)

    assert controller.mode == "idle"
    assert window.screw_edit_btn.enabled is False


def test_selection_change_clears_pointer_lock_from_every_viewer():
    controller, window = _make_controller()
    mpr_viewers = [
        SimpleNamespace(cancelled=0, is_screw_interaction_active=True)
        for _ in range(3)
    ]
    viewer_3d = SimpleNamespace(
        cancelled=0,
        is_screw_interaction_active=True,
    )
    for viewer in [*mpr_viewers, viewer_3d]:
        viewer.cancel_screw_interaction = (
            lambda target=viewer: setattr(
                target,
                "cancelled",
                target.cancelled + 1,
            )
        )
    window._get_mpr_viewers = lambda: mpr_viewers
    window.viewer_3d = viewer_3d
    assert controller.begin_drag(0, "shaft", (0.0, 0.0, 20.0)) is True

    window.screw_list_widget.row = -1
    controller.on_screw_selection_changed(-1)

    assert controller.mode == "idle"
    assert [viewer.cancelled for viewer in mpr_viewers] == [1, 1, 1]
    assert viewer_3d.cancelled == 1


def test_entry_handle_drag_updates_only_entry_continuously():
    controller, window = _make_controller()

    assert controller.begin_drag(0, "entry", (0.0, 0.0, 0.0)) is True
    assert controller.update_drag((4.0, 5.0, 6.0), source="Axial MPR") is True

    screw = window._tool_ctrl.screw_tool.get_screws()[0]
    assert screw.entry_point == (4.0, 5.0, 6.0)
    assert screw.target_point == (0.0, 0.0, 40.0)
    assert controller.mode == "entry"
    assert window._tool_ctrl.refreshed[-1] == (0, screw)
    assert "double-click again" in window.statusbar.message
    assert controller._vm.crosshair_updates == []


def test_tip_handle_drag_updates_only_tip():
    controller, window = _make_controller()

    controller.begin_drag(0, "tip", (0.0, 0.0, 40.0))
    controller.update_drag((3.0, 2.0, 35.0), source="3D")

    screw = window._tool_ctrl.screw_tool.get_screws()[0]
    assert screw.entry_point == (0.0, 0.0, 0.0)
    assert screw.target_point == (3.0, 2.0, 35.0)
    assert controller.active_source == "3D"
    assert controller._vm.crosshair_updates == [
        ((3.0, 2.0, 35.0), "3d_screw_edit")
    ]


def test_shaft_drag_translates_both_points_by_same_delta():
    controller, window = _make_controller()

    controller.begin_drag(0, "shaft", (0.0, 0.0, 20.0))
    controller.update_drag((10.0, 5.0, 22.0), source="3D")

    screw = window._tool_ctrl.screw_tool.get_screws()[0]
    assert screw.entry_point == pytest.approx((10.0, 5.0, 2.0))
    assert screw.target_point == pytest.approx((10.0, 5.0, 42.0))
    assert tuple(
        screw.target_point[index] - screw.entry_point[index]
        for index in range(3)
    ) == pytest.approx((0.0, 0.0, 40.0))
    assert controller._vm.crosshair_updates == [
        ((10.0, 5.0, 22.0), "3d_screw_edit")
    ]


def test_invalid_drag_keeps_last_valid_geometry_and_drag_active():
    controller, window = _make_controller()

    controller.begin_drag(0, "tip", (0.0, 0.0, 40.0))
    assert controller.update_drag((0.0, 0.0, 0.5), source="Axial MPR") is False

    screw = window._tool_ctrl.screw_tool.get_screws()[0]
    assert screw.target_point == (0.0, 0.0, 40.0)
    assert controller.mode == "tip"
    assert window._tool_ctrl.refreshed == []


def test_begin_drag_selects_screw_and_end_drag_returns_idle():
    controller, window = _make_controller()
    window.screw_list_widget.row = -1

    assert controller.begin_drag(0, "shaft", (0.0, 0.0, 20.0)) is True
    assert window.screw_list_widget.currentRow() == 0
    assert controller.mode == "move"

    controller.end_drag()

    assert controller.mode == "idle"


def test_ending_mpr_drag_realigns_review_to_final_screw_geometry():
    controller, window = _make_controller()
    controller.begin_drag(0, "tip", (0.0, 0.0, 40.0))
    controller.update_drag((8.0, 4.0, 35.0), source="Axial MPR")

    controller.end_drag()

    assert window._screw_mpr_ctrl.updated == [0]


def test_rapid_drag_samples_coalesce_to_one_rebuild():
    """Pointer bursts must not rebuild the 3D actor per event."""
    controller, window = _make_controller()

    assert controller.begin_drag(0, "entry", (0.0, 0.0, 0.0)) is True
    assert controller.update_drag((1.0, 0.0, 0.0), source="Axial MPR") is True
    assert controller.update_drag((2.0, 0.0, 0.0), source="Axial MPR") is True
    assert controller.update_drag((3.0, 0.0, 0.0), source="Axial MPR") is True

    # Only the first sample rebuilds; the rest stage the pending geometry.
    assert len(window._tool_ctrl.refreshed) == 1

    controller.end_drag()

    screw = window._tool_ctrl.screw_tool.get_screws()[0]
    assert screw.entry_point == (3.0, 0.0, 0.0)
    assert screw.target_point == (0.0, 0.0, 40.0)
    assert len(window._tool_ctrl.refreshed) == 2
    assert window._screw_mpr_ctrl.updated == [0]


def test_begin_drag_resets_throttle_state_for_a_rapid_re_drag():
    """A drag that starts inside the previous drag's throttle window applies."""
    import time

    controller, window = _make_controller()

    assert controller.begin_drag(0, "entry", (0.0, 0.0, 0.0)) is True
    assert controller.update_drag((1.0, 0.0, 0.0), source="Axial MPR") is True
    assert len(window._tool_ctrl.refreshed) == 1

    # Still inside the throttle window: a fresh drag must not inherit the
    # old drag's clock or its staged sample.
    controller._last_drag_apply_s = time.monotonic()
    controller._pending_drag = ((9.0, 9.0, 9.0), (0.0, 0.0, 40.0), "Axial MPR")
    assert controller.begin_drag(0, "entry", (0.0, 0.0, 0.0)) is True
    assert controller._pending_drag is None
    assert controller.update_drag((2.0, 0.0, 0.0), source="Axial MPR") is True

    assert len(window._tool_ctrl.refreshed) == 2
    screw = window._tool_ctrl.screw_tool.get_screws()[0]
    assert screw.entry_point == (2.0, 0.0, 0.0)


def test_visible_edit_controls_mirror_mode_and_selection():
    controller, window = _make_controller()
    controller.refresh_controls()

    assert window.screw_edit_btn.enabled is True
    assert window._screw_edit_move_entry_action.enabled is True
    assert window._screw_edit_move_tip_action.enabled is True
    assert window._screw_edit_move_whole_action.enabled is True
    # Menu mode actions stay enabled so start() itself can report why an
    # edit cannot begin; the button carries the can_edit state instead.

    assert controller.start("entry") is True

    assert window.screw_edit_btn.enabled is True
    assert window._screw_edit_move_entry_action.checked is True
    assert window._screw_edit_move_tip_action.checked is False

    controller.cancel(show_message=False)

    assert window._screw_edit_move_entry_action.checked is False


def test_release_replays_a_rejected_sample_as_a_boundary_note():
    """Staging never validates: an out-of-volume burst still warns on release."""
    controller, window = _make_controller()

    assert controller.begin_drag(0, "entry", (0.0, 0.0, 0.0)) is True
    assert controller.update_drag((1.0, 0.0, 0.0), source="Axial MPR") is True
    assert controller.update_drag((150.0, 0.0, 0.0), source="Axial MPR") is True

    controller.end_drag()

    screw = window._tool_ctrl.screw_tool.get_screws()[0]
    assert screw.entry_point == (1.0, 0.0, 0.0)
    assert "inside the CT volume" in window.statusbar.message
    assert window._screw_mpr_ctrl.updated == [0]
