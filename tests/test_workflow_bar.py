"""Unit and widget tests for the workflow bar: the state rule in isolation
(``workflow_states``) and the QWidget that renders it (``WorkflowBar``).
"""

import itertools
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("pytestqt")

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QLabel

from src.ui.styles import THEMES, load_stylesheet
from src.ui.workflow_bar import (
    DONE_MARK,
    StepState,
    WorkflowBar,
    WorkflowStep,
    workflow_states,
)


def _states(**overrides):
    defaults = dict(
        has_volume=False,
        segment_available=True,
        has_mask=False,
        plan_available=False,
        has_plan=False,
    )
    defaults.update(overrides)
    return workflow_states(**defaults)


# ---------------------------------------------------------------------------
# workflow_states: pure, Qt-free.
# ---------------------------------------------------------------------------


def test_fresh_session_points_at_open_dicom():
    states = _states()

    assert [state.done for state in states] == [False, False, False]
    assert [state.enabled for state in states] == [True, False, False]
    assert [state.current for state in states] == [True, False, False]
    assert [state.hint for state in states] == [
        "Open a DICOM series folder",
        "Open a DICOM series first",
        "Run segmentation first",
    ]


def test_loaded_volume_finishes_step_one_and_points_at_segment():
    states = _states(has_volume=True)

    assert [state.done for state in states] == [True, False, False]
    assert [state.enabled for state in states] == [True, True, False]
    assert [state.current for state in states] == [False, True, False]
    assert states[1].hint == "Run TotalSegmentator on the loaded study"
    assert states[2].hint == "Run segmentation first"


def test_segment_stays_current_while_its_button_is_disabled():
    states = _states(has_volume=True, segment_available=False)

    assert states[1].enabled is False
    assert states[1].current is True


def test_mask_without_levels_points_at_a_disabled_plan_step():
    states = _states(has_volume=True, has_mask=True, plan_available=False)

    assert [state.done for state in states] == [True, True, False]
    assert states[2].current is True
    assert states[2].enabled is False
    assert states[2].hint == "Select vertebral levels in the Planning tab"


def test_mask_with_levels_enables_the_plan_step():
    states = _states(has_volume=True, has_mask=True, plan_available=True)

    assert states[2].enabled is True
    assert states[2].hint == "Plan screws for the selected vertebral levels"


def test_finished_plan_leaves_no_current_step():
    states = _states(
        has_volume=True, has_mask=True, plan_available=True, has_plan=True
    )

    assert all(state.done for state in states)
    assert not any(state.current for state in states)


def test_plan_step_needs_a_mask_even_when_its_button_is_enabled():
    states = _states(has_volume=True, has_mask=False, plan_available=True)

    assert states[2].enabled is False


def test_a_previous_studys_mask_or_plan_finishes_nothing_without_a_volume():
    states = _states(has_volume=False, has_mask=True, has_plan=True)
    assert [state.done for state in states] == [False, False, False]
    assert [state.current for state in states] == [True, False, False]

    states = _states(has_volume=True, has_mask=False, has_plan=True)
    assert [state.done for state in states] == [True, False, False]


@pytest.mark.parametrize(
    "combo", list(itertools.product([False, True], repeat=5))
)
def test_states_are_consistent_for_every_input(combo):
    has_volume, segment_available, has_mask, plan_available, has_plan = combo
    states = workflow_states(
        has_volume=has_volume,
        segment_available=segment_available,
        has_mask=has_mask,
        plan_available=plan_available,
        has_plan=has_plan,
    )

    assert len(states) == 3
    assert states[0].enabled is True
    for i in (1, 2):
        assert not states[i].done or states[i - 1].done

    current_indices = [i for i, state in enumerate(states) if state.current]
    first_not_done = next(
        (i for i, state in enumerate(states) if not state.done), None
    )
    expected = [] if first_not_done is None else [first_not_done]
    assert current_indices == expected

    if states[1].enabled:
        assert has_volume
    if states[2].enabled:
        assert has_mask and plan_available
    if states[2].done:
        assert has_plan


# ---------------------------------------------------------------------------
# WorkflowBar: the widget that renders those states.
# ---------------------------------------------------------------------------


def _build_bar(qtbot):
    calls = []

    def _recorder(label):
        def _run():
            calls.append(label)

        return _run

    steps = [
        WorkflowStep("Open DICOM", _recorder("Open DICOM")),
        WorkflowStep("Segment", _recorder("Segment")),
        WorkflowStep("Plan Screws", _recorder("Plan Screws")),
    ]
    bar = WorkflowBar(steps)
    qtbot.addWidget(bar)
    return bar, calls


def test_bar_numbers_each_step_and_joins_them_with_arrows(qtbot):
    bar, _calls = _build_bar(qtbot)

    assert [button.text() for button in bar.buttons] == [
        "①  Open DICOM",
        "②  Segment",
        "③  Plan Screws",
    ]
    assert bar.objectName() == "workflowBar"
    assert all(button.objectName() == "workflowStep" for button in bar.buttons)

    chevrons = [
        child
        for child in bar.findChildren(QLabel)
        if child.objectName() == "workflowChevron"
    ]
    assert len(chevrons) == 2
    assert all(chevron.text() == "→" for chevron in chevrons)


def test_done_steps_show_a_check_mark_instead_of_their_number(qtbot):
    bar, _calls = _build_bar(qtbot)

    bar.set_states(
        [
            StepState(True, done=True),
            StepState(True, current=True),
            StepState(False),
        ]
    )

    texts = [button.text() for button in bar.buttons]
    assert texts == ["✓ Open DICOM", "②  Segment", "③  Plan Screws"]
    assert texts[0] == f"{DONE_MARK} Open DICOM"

    bar.set_states(
        [StepState(True), StepState(True), StepState(False)]
    )
    assert [button.text() for button in bar.buttons] == [
        "①  Open DICOM",
        "②  Segment",
        "③  Plan Screws",
    ]


def test_roles_mark_the_current_step_primary_and_finished_ones_secondary(qtbot):
    bar, _calls = _build_bar(qtbot)

    bar.set_states(
        [
            StepState(True, done=True),
            StepState(True, current=True),
            StepState(False),
        ]
    )
    assert [button.property("role") for button in bar.buttons] == [
        "secondary",
        "primary",
        "",
    ]

    bar.set_states(
        [
            StepState(True, done=True),
            StepState(True, done=True),
            StepState(True, current=True),
        ]
    )
    assert [button.property("role") for button in bar.buttons] == [
        "secondary",
        "secondary",
        "primary",
    ]


def test_set_states_applies_enabled_state_and_hint(qtbot):
    bar, _calls = _build_bar(qtbot)

    states = [
        StepState(True, hint="first"),
        StepState(False, hint="second"),
        StepState(True, hint="third"),
    ]
    bar.set_states(states)

    for button, state in zip(bar.buttons, states, strict=True):
        assert button.isEnabled() == state.enabled
        assert button.toolTip() == state.hint


def test_set_states_ignores_extra_and_missing_states(qtbot):
    bar, _calls = _build_bar(qtbot)

    before_text = bar.buttons[2].text()
    before_enabled = bar.buttons[2].isEnabled()
    before_tooltip = bar.buttons[2].toolTip()

    bar.set_states([StepState(True), StepState(False)])

    assert bar.buttons[2].text() == before_text
    assert bar.buttons[2].isEnabled() == before_enabled
    assert bar.buttons[2].toolTip() == before_tooltip

    # Four states for a three-button bar must not raise.
    bar.set_states([StepState(True)] * 4)


def test_clicking_a_step_runs_its_action_only_when_enabled(qtbot):
    bar, calls = _build_bar(qtbot)
    bar.set_states(
        [StepState(True), StepState(False), StepState(True)]
    )

    qtbot.mouseClick(bar.buttons[0], Qt.MouseButton.LeftButton)
    assert calls == ["Open DICOM"]

    qtbot.mouseClick(bar.buttons[1], Qt.MouseButton.LeftButton)
    assert calls == ["Open DICOM"]


def test_steps_past_the_circled_digits_are_numbered_plainly(qtbot):
    steps = [WorkflowStep(f"S{i}", lambda: None) for i in range(1, 7)]
    bar = WorkflowBar(steps)
    qtbot.addWidget(bar)

    assert bar.buttons[5].text() == "6.  S6"


def test_every_theme_styles_the_workflow_bar():
    for name in THEMES:
        stylesheet = load_stylesheet(name)
        assert "QWidget#workflowBar" in stylesheet
        assert "QPushButton#workflowStep" in stylesheet
        assert "QLabel#workflowChevron" in stylesheet
