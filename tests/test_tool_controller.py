"""Direct tests for ToolController.set_tool dispatch.

Regression scope: src/controllers/tool_controller.py — tool switching,
the "measure" alias resolving through the mode combo, path-tool handling,
unknown-tool rejection, and edit-mode cancellation on switch. Uses a
lightweight fake window (no Qt, no MainWindow) so these stay independent
of UI construction.
"""

from types import SimpleNamespace

import pytest

from src.controllers.tool_controller import ToolController


class _Action:
    def __init__(self):
        self.checked = False

    def isChecked(self):
        return self.checked

    def setChecked(self, value):
        self.checked = bool(value)


class _Combo:
    def __init__(self, modes=("distance", "path", "angle")):
        self._modes = list(modes)
        self._index = 0
        self.signals_blocked = False

    def currentData(self):
        return self._modes[self._index]

    def findData(self, mode):
        try:
            return self._modes.index(mode)
        except ValueError:
            return -1

    def blockSignals(self, block):
        previous = self.signals_blocked
        self.signals_blocked = bool(block)
        return previous

    def setCurrentIndex(self, index):
        self._index = index


class _Button:
    def __init__(self):
        self.enabled = False

    def setEnabled(self, value):
        self.enabled = bool(value)


class _ListWidget:
    def __init__(self):
        self.rows = []
        self.current = -1

    def currentRow(self):
        return self.current

    def setCurrentRow(self, row):
        self.current = row

    def clear(self):
        self.rows = []

    def addItem(self, text):
        self.rows.append(text)


class _StatusBar:
    def __init__(self):
        self.messages = []

    def showMessage(self, message):
        self.messages.append(message)


class _Viewer:
    def __init__(self, plane):
        self.plane = plane

    def add_measurement(self, measurement_id, points, label):
        return None

    def remove_measurement(self, measurement_id):
        return None

    def clear_measurements(self):
        return None

    def set_selected_measurement(self, measurement_id):
        return None


def _window():
    window = SimpleNamespace(
        measure_mode_combo=_Combo(),
        measure_finish_btn=_Button(),
        statusbar=_StatusBar(),
        _select_tool_action=_Action(),
        _add_screw_tool_action=_Action(),
        _distance_tool_action=_Action(),
        _angle_tool_action=_Action(),
        _coord_label=SimpleNamespace(setText=lambda text: None),
        _hu_label=SimpleNamespace(setText=lambda text: None),
        _screw_edit_ctrl=None,
        axial_viewer=_Viewer("axial"),
        sagittal_viewer=_Viewer("sagittal"),
        coronal_viewer=_Viewer("coronal"),
        measurement_list_widget=_ListWidget(),
        viewer_3d=_Viewer("3d"),
    )
    window._get_mpr_viewers = lambda: [
        window.axial_viewer,
        window.sagittal_viewer,
        window.coronal_viewer,
    ]
    return window


class _VolumeManager:
    def get_vtk_image(self):
        return object()

    def get_voxel_value(self, x, y, z):
        return None


def _controller(window=None):
    window = window or _window()
    return ToolController(volume_manager=_VolumeManager(), main_window=window)


def test_initial_tool_is_navigate():
    assert _controller().active_tool == "navigate"


def test_set_tool_screw_updates_action_and_status():
    controller = _controller()
    window = controller._window

    controller.set_tool("screw")

    assert controller.active_tool == "screw"
    assert window._add_screw_tool_action.isChecked()
    assert window.statusbar.messages[-1].startswith("Add Screw")


def test_set_tool_rejects_unknown_tool():
    with pytest.raises(ValueError, match="Unknown tool"):
        _controller().set_tool("laser")


def test_set_tool_measure_resolves_combo_mode():
    controller = _controller()
    window = controller._window
    window.measure_mode_combo.setCurrentIndex(
        window.measure_mode_combo.findData("angle")
    )

    controller.set_tool("measure")

    assert controller.active_tool == "angle"
    assert controller.measurement_tool.get_mode() == "angle"


def test_set_tool_path_enables_finish_button():
    controller = _controller()
    window = controller._window

    controller.set_tool("path")

    assert controller.active_tool == "path"
    assert controller.measurement_tool.get_mode() == "path"
    assert window.measure_finish_btn.enabled


def test_set_tool_distance_disables_finish_button():
    controller = _controller()
    window = controller._window
    controller.set_tool("path")
    assert window.measure_finish_btn.enabled

    controller.set_tool("distance")

    assert controller.active_tool == "distance"
    assert not window.measure_finish_btn.enabled


def test_switching_tool_cancels_pending_screw_state():
    controller = _controller()
    controller.set_tool("screw")
    controller.screw_tool.on_click(0.0, 0.0, 0.0, plane="axial")
    assert controller.screw_tool.get_state() == "waiting_target"

    controller.set_tool("navigate")

    assert controller.screw_tool.get_state() == "waiting_entry"


def test_switching_tool_cancels_pending_measurement():
    controller = _controller()
    controller.set_tool("distance")
    controller.measurement_tool.on_click(0.0, 0.0, 0.0)
    assert len(controller.measurement_tool.get_pending_points()) == 1

    controller.set_tool("navigate")

    assert controller.measurement_tool.get_pending_points() == []


def test_switching_tool_cancels_measurement_edit():
    from src.models.measurement import Measurement

    controller = _controller()
    window = controller._window
    controller.set_tool("distance")
    controller.on_viewer_click("axial", 0.0, 0.0, 0.0)
    controller.on_viewer_click("axial", 3.0, 4.0, 0.0)
    window.measurement_list_widget.setCurrentRow(0)
    controller.begin_edit_selected_measurement()
    assert controller._editing_measurement_row == 0

    controller.set_tool("navigate")

    assert controller._editing_measurement_row is None
    assert isinstance(
        controller.measurement_tool.get_measurements()[0], Measurement
    )


class _ScrewListStub:
    """Minimal ScrewPlanTable surface for removal/undo tests."""

    def __init__(self):
        self.rows = []
        self.current = -1

    def count(self):
        return len(self.rows)

    def currentRow(self):
        return self.current

    def setCurrentRow(self, row):
        self.current = int(row)

    def selectedIndexes(self):
        return []

    def blockSignals(self, block):
        return False

    def visible_rows(self):
        return list(range(len(self.rows)))

    def clear(self):
        self.rows = []
        self.current = -1

    def addScrewRow(self, screw, index=None):
        self.rows.append(screw)

    def updateScrewRow(self, _index, _screw):
        return None


class _Viewer3DStub:
    def __init__(self):
        self.added = 0
        self.cleared = 0

    def add_screw(self, *_args, **_kwargs):
        self.added += 1
        return object()

    def remove_screw(self, _actor):
        return None

    def clear_screws(self):
        self.cleared += 1


class _MPRStub:
    def add_screw_overlay(self, *_args, **_kwargs):
        return None

    def clear_screw_overlays(self):
        return None


class _ScrewMPRStub:
    def __init__(self):
        self.updated = []
        self.removed = []
        self.exited = 0
        self.selections = []

    def on_screw_updated(self, index):
        self.updated.append(int(index))

    def on_screw_removed(self, row):
        self.removed.append(int(row))

    def on_screws_removed(self, rows):
        self.removed.extend(int(row) for row in rows)

    def on_screw_selection_changed(self, row):
        self.selections.append(int(row))

    def exit(self):
        self.exited += 1


class _EditStub:
    def __init__(self):
        self.removed = []
        self.refreshed = 0
        self.reset_calls = 0
        self.selections = []

    def on_screw_removed(self, row):
        self.removed.append(int(row))

    def on_screw_selection_changed(self, row):
        self.selections.append(int(row))

    def refresh_controls(self):
        self.refreshed += 1

    def reset(self):
        self.reset_calls += 1


def _removal_window(screws):
    window = _window()
    window.screw_list_widget = _ScrewListStub()
    window.viewer_3d = _Viewer3DStub()
    window._screw_mpr_ctrl = _ScrewMPRStub()
    window._screw_edit_ctrl = _EditStub()
    window._get_mpr_viewers = lambda: [_MPRStub()]
    controller = ToolController(
        volume_manager=_VolumeManager(), main_window=window
    )
    for screw in screws:
        controller.add_existing_screw(screw)
    return controller, window


def _screw(level="L4", side="left"):
    from src.models.screw import Screw

    return Screw(
        entry_point=(0.0, 0.0, 0.0),
        target_point=(0.0, 0.0, 40.0),
        diameter=6.0,
        vertebra_level=level,
        side=side,
    )


def test_single_remove_keeps_old_status_text():
    controller, window = _removal_window([_screw(), _screw(side="right")])
    window.screw_list_widget.setCurrentRow(0)

    controller.remove_selected_screw()

    assert len(controller.screw_tool.get_screws()) == 1
    assert window.statusbar.messages[-1] == "Selected screw removed"


def test_multi_remove_deletes_every_selected_row_at_once():
    controller, window = _removal_window(
        [_screw(), _screw(side="right"), _screw(level="L5")]
    )
    table = window.screw_list_widget

    class _Index:
        def __init__(self, row):
            self._row = row

        def row(self):
            return self._row

    table.selectedIndexes = lambda: [_Index(0), _Index(2)]
    table.setCurrentRow(0)

    assert controller.remove_screws_at_rows([0, 2]) == 2
    remaining = controller.screw_tool.get_screws()
    assert len(remaining) == 1
    assert remaining[0].side == "right"


def test_remove_selected_reports_the_batch_size():
    controller, window = _removal_window(
        [_screw(), _screw(side="right"), _screw(level="L5")]
    )
    table = window.screw_list_widget

    class _Index:
        def __init__(self, row):
            self._row = row

        def row(self):
            return self._row

    table.selectedIndexes = lambda: [_Index(0), _Index(2)]
    table.setCurrentRow(0)

    controller.remove_selected_screw()

    assert len(controller.screw_tool.get_screws()) == 1
    assert window.statusbar.messages[-1] == "2 screws removed"


def test_undo_restores_the_removed_batch_in_order():
    controller, window = _removal_window(
        [_screw(), _screw(side="right"), _screw(level="L5")]
    )

    assert controller.remove_screws_at_rows([0, 2]) == 2
    assert controller.undo_remove_screws() is True

    screws = controller.screw_tool.get_screws()
    assert [s.vertebra_level for s in screws] == ["L4", "L4", "L5"]
    assert [s.side for s in screws] == ["left", "right", "left"]
    assert "Restored 2 screws" in window.statusbar.messages[-1]


def test_undo_with_empty_stack_reports_nothing_to_undo():
    controller, window = _removal_window([_screw()])

    assert controller.undo_remove_screws() is False
    assert window.statusbar.messages[-1] == "Nothing to undo"


def test_clear_screws_drops_the_undo_stack():
    controller, window = _removal_window([_screw(), _screw(side="right")])
    window.screw_list_widget.setCurrentRow(0)
    controller.remove_selected_screw()
    assert len(controller.screw_tool.get_screws()) == 1

    controller.clear_screws()

    assert controller._screw_undo_stack == []
    assert controller.undo_remove_screws() is False
    assert window.statusbar.messages[-1] == "Nothing to undo"


def test_clear_screw_undo_forgets_a_pending_removal():
    controller, window = _removal_window([_screw(), _screw(side="right")])
    assert controller.remove_screws_at_rows([0]) == 1

    controller.clear_screw_undo()

    assert controller.undo_remove_screws() is False


def test_hidden_rows_are_excluded_from_delete_selection():
    controller, window = _removal_window(
        [_screw(), _screw(side="right"), _screw(level="L5")]
    )
    table = window.screw_list_widget

    class _Index:
        def __init__(self, row):
            self._row = row

        def row(self):
            return self._row

    table.selectedIndexes = lambda: [_Index(0), _Index(2)]
    table.setCurrentRow(2)
    table.visible_rows = lambda: [0, 1]

    assert controller._selected_screw_rows() == [0]
    # The rows passed explicitly still delete; only the *selection*
    # derivation filters hidden rows.
    controller2, window2 = _removal_window(
        [_screw(), _screw(side="right"), _screw(level="L5")]
    )
    window2.screw_list_widget.selectedIndexes = lambda: [_Index(0), _Index(2)]
    window2.screw_list_widget.setCurrentRow(2)
    window2.screw_list_widget.visible_rows = lambda: [0, 1]
    window2.screw_list_widget.setCurrentRow(0)
    controller2.remove_selected_screw()
    assert [s.side for s in controller2.screw_tool.get_screws()] == [
        "right",
        "left",
    ]


def test_rebuild_keeps_listeners_on_the_reselected_row_without_review_switch():
    controller, window = _removal_window(
        [_screw(), _screw(side="right"), _screw(level="L5")]
    )
    window._on_screw_visual_selection_changed = lambda row: window.statusbar.showMessage(
        f"visual {row}"
    )

    assert controller.remove_screws_at_rows([0]) == 1

    assert window._screw_mpr_ctrl.selections[-1] == 0
    assert window._screw_edit_ctrl.selections[-1] == 0
    assert window.statusbar.messages[-1] == "visual 0"
