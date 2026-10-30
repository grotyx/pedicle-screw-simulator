"""Undo/redo of screw plan edits: the history object and the window wiring."""

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from src.models.screw import Screw
from src.models.screw_history import ScrewHistory


def _screw(x, **kwargs):
    return Screw((float(x), 2.0, 3.0), (float(x), 2.0, 33.0), **kwargs)


# ── ScrewHistory ──────────────────────────────────────────────────────────


def test_history_starts_empty():
    history = ScrewHistory()
    assert not history.can_undo and not history.can_redo
    assert history.undo([]) is None
    assert history.redo([]) is None


def test_undo_returns_the_snapshot_and_redo_the_state_it_replaced():
    history = ScrewHistory()
    before = [_screw(1)]
    after = [_screw(1), _screw(2)]
    history.push("add screw", before)
    assert history.undo_label == "add screw"

    label, restored = history.undo(after)
    assert label == "add screw" and restored == before
    assert history.redo_label == "add screw" and not history.can_undo

    label, redone = history.redo(restored)
    assert label == "add screw" and redone == after
    assert history.can_undo and not history.can_redo


def test_snapshots_are_deep_copies():
    history = ScrewHistory()
    screws = [_screw(1, warnings=["w"])]
    history.push("edit", screws)
    screws[0].warnings.append("later")
    screws[0].diameter = 9.0

    _, restored = history.undo([])
    assert restored[0].warnings == ["w"]
    assert restored[0].diameter != 9.0


def test_new_edit_clears_redo():
    history = ScrewHistory()
    history.push("a", [])
    history.undo([_screw(1)])
    assert history.can_redo
    history.push("b", [])
    assert not history.can_redo


def test_history_is_bounded():
    history = ScrewHistory(limit=3)
    for index in range(5):
        history.push(f"step {index}", [_screw(index)])
    labels = []
    while history.can_undo:
        labels.append(history.undo([])[0])
    assert labels == ["step 4", "step 3", "step 2"]


def test_same_coalesce_key_merges_a_burst_into_one_step():
    history = ScrewHistory()
    first = [_screw(1, diameter=6.0)]
    history.push("change diameter", first, coalesce_key=("diameter", 0))
    history.push("change diameter", [_screw(1, diameter=6.5)],
                 coalesce_key=("diameter", 0))
    history.push("change diameter", [_screw(1, diameter=7.0)],
                 coalesce_key=("diameter", 0))

    _, restored = history.undo([_screw(1, diameter=7.5)])
    assert restored == first
    assert not history.can_undo


def test_coalescing_stops_after_another_step_or_an_undo():
    history = ScrewHistory()
    history.push("change diameter", [], coalesce_key=("diameter", 0))
    history.push("move screw", [_screw(1)])
    history.push("change diameter", [_screw(2)], coalesce_key=("diameter", 0))
    assert len(history._undo) == 3

    history.undo([_screw(3)])
    history.push("change diameter", [_screw(2)], coalesce_key=("diameter", 0))
    assert len(history._undo) == 3


def test_clear_drops_both_stacks():
    history = ScrewHistory()
    history.push("a", [])
    history.push("b", [])
    history.undo([])
    history.clear()
    assert not history.can_undo and not history.can_redo


# ── MainWindow integration ────────────────────────────────────────────────

pytest.importorskip("pytestqt")
pytest.importorskip("SimpleITK")

import tests.test_ui_integration as _ui  # noqa: E402

_create_test_image = _ui._create_test_image
_planned_l4 = _ui._planned_l4
_ProgressStub = _ui._ProgressStub
# Shared fixtures, bound here so pytest finds them in this module.
isolated_qsettings = _ui.isolated_qsettings
ui_main_window = _ui.ui_main_window


def _loaded(window, series="SERIES-UNDO"):
    window._on_dicom_loaded(
        image=_create_test_image(),
        metadata={"series_id": series, "num_slices": 12},
        progress=_ProgressStub(),
    )
    return window


def _geometry(screws):
    # Undo regrades what it restores, so compare where the screws are.
    return [(s.entry_point, s.target_point, s.diameter) for s in screws]


def _overlay_entries(window):
    return {
        viewer.plane: {k: v["entry"] for k, v in viewer.screw_overlays.items()}
        for viewer in window._get_mpr_viewers()
    }


def test_undo_redo_actions_live_in_an_edit_menu(ui_main_window):
    window = ui_main_window
    menus = [a.text() for a in window.menuBar().actions()]
    assert "Edit" in menus
    assert window._undo_action.shortcut().toString() == "Ctrl+Z"
    redo_keys = {k.toString() for k in window._redo_action.shortcuts()}
    assert {"Ctrl+Shift+Z", "Ctrl+Y"} <= redo_keys
    assert not window._undo_action.isEnabled()
    assert not window._redo_action.isEnabled()


def test_manual_insert_undo_and_redo(ui_main_window):
    window = _loaded(ui_main_window)
    window._add_screw_tool_action.trigger()
    window.axial_viewer.crosshair_moved.emit("axial", 2.0, 3.0, 4.0)
    assert not window._undo_action.isEnabled()  # half-placed: nothing yet
    window.axial_viewer.crosshair_moved.emit("axial", 2.0, 3.0, 34.0)
    placed = window.screw_tool.get_screws()[0]
    assert window._undo_action.isEnabled()

    window._undo_action.trigger()
    assert window.screw_tool.get_screws() == []
    assert window.screw_list_widget.count() == 0
    assert window.viewer_3d.screws == []
    assert all(not v.screw_overlays for v in window._get_mpr_viewers())
    assert window.statusbar.currentMessage() == "Undo: add screw"
    assert window._redo_action.isEnabled()
    assert not window._undo_action.isEnabled()

    window._redo_action.trigger()
    assert _geometry(window.screw_tool.get_screws()) == _geometry([placed])
    assert window.screw_list_widget.count() == 1
    assert window.screw_list_widget.currentRow() == 0
    assert window.viewer_3d.screws[0].screw_id == 0
    assert window.axial_viewer.screw_overlays[0]["entry"] == (2.0, 3.0, 4.0)
    assert window.statusbar.currentMessage() == "Redo: add screw"


def test_delete_undo_restores_the_same_screw_at_the_same_index(ui_main_window):
    window = _loaded(ui_main_window)
    screws = [
        _screw(i + 1, vertebra_level=f"L{i + 1}", side="left") for i in range(3)
    ]
    screws[1].grade = "C"
    screws[1].breach_distance = 2.5
    for screw in screws:
        window._tool_ctrl.add_existing_screw(screw)
    window.screw_list_widget.setCurrentRow(1)
    window._tool_ctrl.remove_selected_screw()
    assert len(window.screw_tool.get_screws()) == 2

    window._undo_action.trigger()

    restored = window.screw_tool.get_screws()
    assert [(r.entry_point, r.target_point) for r in restored] == [
        (s.entry_point, s.target_point) for s in screws
    ]
    # Regraded on restore: no segmentation here, so no grade to bring back.
    assert restored[1].grade == "N/A"
    assert window.screw_list_widget.currentRow() == 1
    assert [a.screw_id for a in window.viewer_3d.screws] == [0, 1, 2]
    assert window.axial_viewer.screw_overlays[1]["entry"] == screws[1].entry_point
    assert "L2" in window.screw_list_widget.rowText(1)
    assert window.statusbar.currentMessage() == "Undo: delete screw"


def test_drag_is_one_step_and_undo_puts_the_screw_back(ui_main_window):
    window = _loaded(ui_main_window)
    screw = Screw((2.0, 3.0, 4.0), (2.0, 3.0, 10.0))
    window._tool_ctrl.add_existing_screw(screw, select=True)
    callbacks = window.viewer_3d.screw_interaction_callbacks

    assert callbacks["on_begin"](0, "entry", screw.entry_point)
    callbacks["on_drag"]((3.0, 4.0, 5.0), "3D")
    window._undo_action.trigger()  # ignored mid-drag
    callbacks["on_drag"]((4.0, 5.0, 6.0), "3D")
    callbacks["on_end"]()
    assert window.screw_tool.get_screws()[0].entry_point == (4.0, 5.0, 6.0)

    window._undo_action.trigger()
    assert _geometry(window.screw_tool.get_screws()) == _geometry([screw])
    assert window.viewer_3d.screws[0].entry_point == screw.entry_point
    assert not window._undo_action.isEnabled()  # the whole drag was one step


def test_insert_at_crosshair_is_undoable(ui_main_window):
    window = ui_main_window
    _ui._load_large_study(window, "SERIES-UNDO-INSERT")
    window.length_spin.setValue(20)
    window.volume_manager.set_crosshair_position(10.0, 50.0, 5.0)
    window.insert_left_btn.click()
    assert len(window.screw_tool.get_screws()) == 1

    window._undo_action.trigger()

    assert window.screw_tool.get_screws() == []
    assert window.statusbar.currentMessage() == "Undo: insert screw"


def test_guide_rotation_is_one_undoable_step(ui_main_window):
    from src.core.screw_geometry import GUIDE_REACH_MM, guide_handle_point

    window = _loaded(ui_main_window)
    screw = Screw((2.0, 3.0, 4.0), (2.0, 3.0, 10.0))
    window._tool_ctrl.add_existing_screw(screw, select=True)
    callbacks = window.viewer_3d.screw_interaction_callbacks
    handle = guide_handle_point(screw.entry_point, screw.target_point, GUIDE_REACH_MM)

    assert callbacks["on_begin"](0, "guide", handle)
    callbacks["on_drag"]((handle[0] + 5.0, handle[1], handle[2]), "3D")
    callbacks["on_drag"]((handle[0] + 9.0, handle[1] + 2.0, handle[2]), "3D")
    callbacks["on_end"]()
    assert window.screw_tool.get_screws()[0].target_point != screw.target_point

    window._undo_action.trigger()

    assert _geometry(window.screw_tool.get_screws()) == _geometry([screw])
    assert not window._undo_action.isEnabled()


def test_drag_that_changes_nothing_records_nothing(ui_main_window):
    window = _loaded(ui_main_window)
    screw = Screw((2.0, 3.0, 4.0), (2.0, 3.0, 10.0))
    window._tool_ctrl.add_existing_screw(screw, select=True)
    callbacks = window.viewer_3d.screw_interaction_callbacks

    callbacks["on_begin"](0, "tip", screw.target_point)
    callbacks["on_end"]()

    assert not window._undo_action.isEnabled()


def test_click_edit_is_undoable(ui_main_window):
    window = _loaded(ui_main_window)
    screw = Screw((2.0, 3.0, 4.0), (2.0, 3.0, 10.0))
    window._tool_ctrl.add_existing_screw(screw, select=True)
    window._screw_edit_ctrl.start("tip")
    window.axial_viewer.crosshair_moved.emit("axial", 5.0, 6.0, 11.0)
    assert window.screw_tool.get_screws()[0].target_point == (5.0, 6.0, 11.0)

    window._undo_action.trigger()
    assert _geometry(window.screw_tool.get_screws()) == _geometry([screw])


def test_diameter_burst_is_one_step(ui_main_window):
    window = _loaded(ui_main_window)
    window._tool_ctrl.add_existing_screw(_screw(2, diameter=5.5), select=True)

    window.selected_screw_diameter.setValue(6.0)
    window.selected_screw_diameter.setValue(6.5)
    window.selected_screw_diameter.setValue(7.0)

    window._undo_action.trigger()
    assert window.screw_tool.get_screws()[0].diameter == pytest.approx(5.5)
    assert window.selected_screw_diameter.value() == pytest.approx(5.5)
    assert window.viewer_3d.screws[0].radius == pytest.approx(2.75)
    assert not window._undo_action.isEnabled()


def test_auto_plan_batch_is_one_step(ui_main_window):
    window = _loaded(ui_main_window)
    ctrl = window._auto_placement_ctrl
    old_auto = _screw(1, vertebra_level="L4", side="left", source="auto")
    manual = _screw(4, vertebra_level="L4", side="right")
    window._tool_ctrl.add_existing_screw(old_auto, select=True)
    window._tool_ctrl.add_existing_screw(manual)

    class _Thread:
        generation = ctrl._run_generation
        cancelled = False
        replace_levels = ("L4",)

    ctrl._thread = _Thread
    ctrl._on_finished(_Thread, [_planned_l4()])
    assert [s.source for s in window.screw_tool.get_screws()] == [
        "manual", "auto",
    ]

    window._undo_action.trigger()
    assert _geometry(window.screw_tool.get_screws()) == _geometry([old_auto, manual])
    assert window.statusbar.currentMessage() == "Undo: auto plan"
    assert not window._undo_action.isEnabled()


def test_history_is_cleared_on_workspace_reset_and_plan_load(ui_main_window):
    window = _loaded(ui_main_window)
    window._tool_ctrl.add_existing_screw(_screw(2), select=True)
    window._tool_ctrl.remove_selected_screw()
    assert window._undo_action.isEnabled()

    window.reset_workspace()
    assert not window._undo_action.isEnabled()
    assert not window._tool_ctrl.history.can_undo

    window._tool_ctrl.add_existing_screw(_screw(2), select=True)
    window._tool_ctrl.remove_selected_screw()
    window._undo_action.trigger()
    assert window._redo_action.isEnabled()
    window._plan_ctrl._apply_loaded_plan([_screw(3)], [], [])
    assert not window._undo_action.isEnabled()
    assert not window._redo_action.isEnabled()


def test_new_edit_after_undo_clears_redo(ui_main_window):
    window = _loaded(ui_main_window)
    window._tool_ctrl.add_existing_screw(_screw(2), select=True)
    window._tool_ctrl.add_existing_screw(_screw(5))
    window.screw_list_widget.setCurrentRow(0)
    window._tool_ctrl.remove_selected_screw()
    window._undo_action.trigger()
    assert window._redo_action.isEnabled()

    window.screw_list_widget.setCurrentRow(1)
    window._tool_ctrl.remove_selected_screw()
    assert not window._redo_action.isEnabled()
    assert window._undo_action.text() == "Undo Delete Screw"


def test_a_drag_left_with_esc_is_still_one_undo_step(ui_main_window):
    window = _loaded(ui_main_window)
    screw = Screw((2.0, 3.0, 4.0), (2.0, 3.0, 10.0))
    window._tool_ctrl.add_existing_screw(screw, select=True)
    callbacks = window.viewer_3d.screw_interaction_callbacks

    callbacks["on_begin"](0, "entry", screw.entry_point)
    callbacks["on_drag"]((4.0, 5.0, 6.0), "3D")
    window._screw_edit_ctrl.cancel()            # Esc keeps the geometry

    assert window._undo_action.text() == "Undo Drag Screw"
    window._undo_action.trigger()
    assert _geometry(window.screw_tool.get_screws()) == _geometry([screw])


def test_undo_regrades_against_the_current_segmentation(ui_main_window, monkeypatch):
    # Without a grader (segmentation cleared) a restored grade must not come
    # back as if it were current; each screw is graded once, before display.
    window = _loaded(ui_main_window)
    graded = Screw((2.0, 3.0, 4.0), (2.0, 3.0, 10.0))
    graded.grade = "A"
    window._tool_ctrl.add_existing_screw(graded)
    window.screw_list_widget.setCurrentRow(0)
    window._tool_ctrl.remove_selected_screw()
    evaluated = []
    real = window.screw_tool.evaluate_screw
    monkeypatch.setattr(
        window.screw_tool, "evaluate_screw", lambda screw: (evaluated.append(1), real(screw))
    )
    monkeypatch.setattr(
        window._tool_ctrl, "regrade_all", lambda: pytest.fail("rebuilds every view twice")
    )

    window._undo_action.trigger()

    assert evaluated == [1]
    assert window.screw_tool.get_screws()[0].grade == "N/A"


def test_undo_keeps_review_marks_on_screws_it_does_not_move(ui_main_window):
    window = _loaded(ui_main_window)
    for x in (1.0, 2.0, 3.0):
        window._tool_ctrl.add_existing_screw(Screw((x, 3.0, 4.0), (x, 3.0, 30.0)))
    window.screw_list_widget.setCurrentRow(1)
    window._tool_ctrl.remove_selected_screw()
    for screw in window.screw_tool.get_screws():
        screw.reviewed = True

    window._undo_action.trigger()

    assert [s.reviewed for s in window.screw_tool.get_screws()] == [True, False, True]


def test_undo_keeps_screw_mpr_open(ui_main_window):
    window = _loaded(ui_main_window)
    window._tool_ctrl.add_existing_screw(Screw((8.0, 9.0, 4.0), (8.0, 9.0, 24.0), diameter=6.0))
    window.screw_list_widget.setCurrentRow(0)
    window.screw_axis_mpr_btn.click()
    assert window._screw_mpr_ctrl.is_active
    window._tool_ctrl.set_selected_screw_diameter(7.0)

    window._undo_action.trigger()

    assert window.screw_tool.get_screws()[0].diameter == 6.0
    assert window._screw_mpr_ctrl.is_active


def test_diameter_edits_separated_by_a_selection_change_are_two_steps(ui_main_window):
    window = _loaded(ui_main_window)
    for x in (1.0, 2.0):
        window._tool_ctrl.add_existing_screw(Screw((x, 3.0, 4.0), (x, 3.0, 30.0), diameter=6.5))
    window.screw_list_widget.setCurrentRow(0)
    window._tool_ctrl.set_selected_screw_diameter(7.0)
    window.screw_list_widget.setCurrentRow(1)
    window.screw_list_widget.setCurrentRow(0)
    window._tool_ctrl.set_selected_screw_diameter(7.5)

    window._undo_action.trigger()

    assert window.screw_tool.get_screws()[0].diameter == 7.0
