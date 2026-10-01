"""Per-screw "Reviewed" state: model, persistence, edits, and the Review page."""

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("pytestqt")

from PyQt6.QtWidgets import QLabel  # noqa: E402

from src.models.screw import Screw  # noqa: E402
from src.ui.workflow_bar import workflow_states  # noqa: E402
from src.utils.planning_io import (  # noqa: E402
    screw_from_dict,
    screw_to_dict,
)
from tests.test_main_window_redesign import (  # noqa: E402, F401
    _add_levelled_screw,
    _flush,
    isolated_qsettings,
    ui_main_window,
)


@pytest.fixture
def window(ui_main_window):  # noqa: F811
    return ui_main_window


def _screw():
    return Screw(entry_point=(0.0, 0.0, 0.0), target_point=(0.0, 0.0, 30.0))


def test_screw_defaults_to_not_reviewed():
    assert _screw().reviewed is False


def test_plan_round_trip_keeps_reviewed():
    screw = _screw()
    screw.reviewed = True
    payload = screw_to_dict(screw)
    assert payload["reviewed"] is True
    assert screw_from_dict(payload).reviewed is True


def test_old_plan_without_the_key_loads_not_reviewed():
    payload = screw_to_dict(_screw())
    del payload["reviewed"]
    assert screw_from_dict(payload).reviewed is False


def _states(**overrides):
    args = dict(
        has_volume=True,
        segment_available=True,
        has_mask=True,
        plan_available=True,
        has_plan=True,
        has_screws=True,
    )
    args.update(overrides)
    return workflow_states(**args)


def test_review_step_is_done_only_when_every_screw_is_reviewed():
    assert _states()[3].done is False
    assert _states(all_reviewed=True)[3].done is True
    # No screws: nothing to have reviewed.
    assert _states(has_screws=False, all_reviewed=True)[3].done is False


def _window_with_screw(window):
    _add_levelled_screw(window, "L4", "left")
    window.screw_list_widget.setCurrentRow(0)
    _flush()
    return window._tool_ctrl.screw_tool.get_screws()[0]


def _review_done(window):
    return "✓" in window.workflow_bar.buttons[3].text()


def _chip(window):
    return window.construct_map.findChildren(QLabel, "constructChip")[0]


def test_button_marks_the_selected_screw_reviewed_everywhere(window):
    screw = _window_with_screw(window)
    button = window.screw_reviewed_btn
    assert button.isEnabled() and not button.isChecked()
    assert not _review_done(window)

    button.click()
    _flush()

    assert screw.reviewed is True
    assert button.isChecked()
    assert window.screw_list_widget.item(0, 0).text() == "1 ✓"
    assert _chip(window).property("reviewed") == "true"
    assert _review_done(window)
    assert "1 reviewed" in window.selected_screw_counter.text()

    button.click()
    _flush()
    assert screw.reviewed is False
    assert window.screw_list_widget.item(0, 0).text() == "1"
    assert _chip(window).property("reviewed") == "false"
    assert not _review_done(window)


def test_button_is_disabled_without_a_selected_screw(window):
    assert not window.screw_reviewed_btn.isEnabled()


def test_geometry_edit_clears_reviewed(window):
    screw = _window_with_screw(window)
    window.screw_reviewed_btn.click()
    tool = window._tool_ctrl.screw_tool

    updated = tool.replace_screw(0, (2.0, 3.0, 4.0), (8.0, 9.0, 30.0))
    window._tool_ctrl.refresh_screw(0, updated)
    _flush()

    assert screw.reviewed is True  # the old object is untouched
    assert updated.reviewed is False
    assert not window.screw_reviewed_btn.isChecked()
    assert window.screw_list_widget.item(0, 0).text() == "1"
    assert not _review_done(window)


def test_diameter_edit_clears_reviewed(window):
    _window_with_screw(window)
    window.screw_reviewed_btn.click()

    window._tool_ctrl.set_selected_screw_diameter(7.5)
    _flush()

    assert window._tool_ctrl.screw_tool.get_screws()[0].reviewed is False
    assert not window.screw_reviewed_btn.isChecked()


def test_regrade_that_changes_the_grade_clears_reviewed(window):
    screw = _window_with_screw(window)
    screw.grade = "B"  # as if graded earlier; the grader (none) now says N/A
    screw.reviewed = True

    window._tool_ctrl.regrade_all()
    _flush()

    assert screw.grade == "N/A"
    assert screw.reviewed is False


def test_regrade_that_keeps_the_grade_keeps_reviewed(window):
    screw = _window_with_screw(window)
    screw.grade = "N/A"
    screw.reviewed = True

    window._tool_ctrl.regrade_all()

    assert screw.reviewed is True


def test_selection_change_does_not_clear_reviewed(window):
    first = _window_with_screw(window)
    _add_levelled_screw(window, "L5", "right")
    window.screw_reviewed_btn.click()

    window.screw_list_widget.setCurrentRow(1)
    _flush()
    assert not window.screw_reviewed_btn.isChecked()
    window.screw_list_widget.setCurrentRow(0)
    _flush()

    assert first.reviewed is True
    assert window.screw_reviewed_btn.isChecked()
