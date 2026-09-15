"""Unit and widget tests for the workflow bar: the state rule in isolation
(``workflow_states``) and the QWidget that renders it (``WorkflowBar``).

The bar now navigates a four-page step panel (Study/Segment/Plan/Review)
rather than running the three primary actions directly, so every step stays
clickable and ``workflow_states`` returns four states.
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
    STEP_NAMES,
    StepState,
    WorkflowBar,
    WorkflowStep,
    workflow_states,
)


def _states(**overrides):
    defaults = dict(
        has_volume=False,
        has_mask=False,
        has_plan=False,
        selected_levels=False,
    )
    defaults.update(overrides)
    return workflow_states(**defaults)


# ---------------------------------------------------------------------------
# workflow_states: pure, Qt-free.
# ---------------------------------------------------------------------------


def test_workflow_states_has_study_segment_plan_review():
    assert STEP_NAMES == ("Study", "Segment", "Plan", "Review")
    assert len(_states()) == 4


def test_fresh_session_points_at_study():
    states = _states()

    assert [state.done for state in states] == [False, False, False, False]
    assert [state.enabled for state in states] == [True, True, True, True]
    assert [state.current for state in states] == [True, False, False, False]
    assert states[0].hint == "Open a DICOM series folder"
    assert states[1].hint == "Open a DICOM series first"
    assert states[2].hint == "Run segmentation first"
    assert states[3].hint == "Plan or place screws first"


def test_every_step_stays_clickable():
    for combo in itertools.product([False, True], repeat=4):
        (
            has_volume,
            has_mask,
            has_plan,
            has_screws,
        ) = combo
        states = workflow_states(
            has_volume=has_volume,
            has_mask=has_mask,
            has_plan=has_plan,
            has_screws=has_screws,
        )
        assert [state.enabled for state in states] == [True, True, True, True]


def test_loaded_volume_finishes_step_one_and_points_at_segment():
    states = _states(has_volume=True)

    assert [state.done for state in states] == [True, False, False, False]
    assert [state.current for state in states] == [False, True, False, False]
    assert states[0].hint == "Study loaded — open another one here"
    assert states[1].hint == "Run TotalSegmentator on the loaded study"
    assert states[2].hint == "Run segmentation first"


def test_mask_without_levels_points_at_plan():
    states = _states(has_volume=True, has_mask=True, selected_levels=False)

    assert [state.done for state in states] == [True, True, False, False]
    assert states[2].current is True
    assert states[2].hint == "Select vertebral levels in the Plan step"


def test_mask_with_levels_names_the_plan_hint():
    states = _states(has_volume=True, has_mask=True, selected_levels=True)

    assert states[2].hint == "Plan screws for the selected vertebral levels"


def test_legacy_button_kwargs_are_accepted_but_ignored():
    """Old callers pass inferred button state; session state still rules."""
    planned = workflow_states(
        has_volume=True,
        has_mask=True,
        selected_levels=True,
        has_plan=False,
        plan_available=True,
        segment_available=False,
    )
    assert planned[2].hint == "Plan screws for the selected vertebral levels"

    unplanned = workflow_states(
        has_volume=True,
        has_mask=True,
        selected_levels=False,
        has_plan=False,
        plan_available=True,
        segment_available=True,
    )
    assert unplanned[2].hint == "Select vertebral levels in the Plan step"


def test_review_becomes_current_once_a_plan_exists():
    states = _states(
        has_volume=True, has_mask=True, selected_levels=True, has_plan=True
    )

    assert [state.done for state in states] == [True, True, True, False]
    assert [state.current for state in states] == [False, False, False, True]


def test_running_job_marks_its_step_with_a_spinner(qtbot):
    from src.ui.workflow_bar import RUNNING_MARK

    bar, _calls = _build_bar(qtbot)
    bar.set_states(
        workflow_states(
            has_volume=True,
            has_mask=False,
            has_plan=False,
            is_running=True,
            running_step=0,
        )
    )

    assert bar.buttons[0].text() == f"{RUNNING_MARK} Study"
    assert bar.buttons[0].toolTip() == "Loading DICOM…"

    bar.set_states(
        workflow_states(
            has_volume=True,
            has_mask=False,
            has_plan=False,
            is_running=True,
            running_step=1,
        )
    )

    assert bar.buttons[1].text() == f"{RUNNING_MARK} Segment"
    assert bar.buttons[1].toolTip() == "Segmentation is running…"
    assert bar.buttons[1].isEnabled() is True


def test_review_hint_reflects_whether_screws_exist():
    states = _states(has_screws=False)
    assert states[3].hint == "Plan or place screws first"

    states = _states(has_screws=True)
    assert states[3].hint == "Review the screws level by level"


def test_a_previous_studys_mask_or_plan_finishes_nothing_without_a_volume():
    states = _states(has_volume=False, has_mask=True, has_plan=True)
    assert [state.done for state in states] == [False, False, False, False]
    assert [state.current for state in states] == [True, False, False, False]

    states = _states(has_volume=True, has_mask=False, has_plan=True)
    assert [state.done for state in states] == [True, False, False, False]


@pytest.mark.parametrize(
    "combo", list(itertools.product([False, True], repeat=4))
)
def test_states_are_consistent_for_every_input(combo):
    (
        has_volume,
        has_mask,
        has_plan,
        has_screws,
    ) = combo
    states = workflow_states(
        has_volume=has_volume,
        has_mask=has_mask,
        has_plan=has_plan,
        has_screws=has_screws,
    )

    assert len(states) == 4
    assert all(state.enabled for state in states)
    for i in (1, 2):
        assert not states[i].done or states[i - 1].done
    assert states[3].done is False

    current_indices = [i for i, state in enumerate(states) if state.current]
    first_not_done = next(
        (i for i, state in enumerate(states) if not state.done), None
    )
    expected = [] if first_not_done is None else [first_not_done]
    assert current_indices == expected

    # The Plan step is never marked done unless a plan actually exists.
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
        WorkflowStep("Study", _recorder("Study")),
        WorkflowStep("Segment", _recorder("Segment")),
        WorkflowStep("Plan", _recorder("Plan")),
        WorkflowStep("Review", _recorder("Review")),
    ]
    bar = WorkflowBar(steps)
    qtbot.addWidget(bar)
    return bar, calls


def test_bar_numbers_each_step_and_joins_them_with_arrows(qtbot):
    bar, _calls = _build_bar(qtbot)

    assert [button.text() for button in bar.buttons] == [
        "①  Study",
        "②  Segment",
        "③  Plan",
        "④  Review",
    ]
    assert bar.objectName() == "workflowBar"
    assert all(button.objectName() == "workflowStep" for button in bar.buttons)

    chevrons = [
        child
        for child in bar.findChildren(QLabel)
        if child.objectName() == "workflowChevron"
    ]
    assert len(chevrons) == 3
    assert all(chevron.text() == "→" for chevron in chevrons)


def test_done_steps_show_a_check_mark_instead_of_their_number(qtbot):
    bar, _calls = _build_bar(qtbot)

    bar.set_states(
        [
            StepState(True, done=True),
            StepState(True, current=True),
            StepState(True),
            StepState(True),
        ]
    )

    texts = [button.text() for button in bar.buttons]
    assert texts[0] == f"{DONE_MARK} Study"
    assert texts[1] == "②  Segment"

    # A re-render back to not-done (e.g. a new study loaded) clears the
    # check mark rather than leaving the previous render's text behind.
    bar.set_states(
        [
            StepState(True),
            StepState(True, current=True),
            StepState(True),
            StepState(True),
        ]
    )
    assert bar.buttons[0].text() == "①  Study"


def test_roles_mark_the_current_step_primary_and_finished_ones_secondary(qtbot):
    bar, _calls = _build_bar(qtbot)

    bar.set_states(
        [
            StepState(True, done=True),
            StepState(True, current=True),
            StepState(True),
            StepState(True),
        ]
    )
    assert [button.property("role") for button in bar.buttons] == [
        "secondary",
        "primary",
        "",
        "",
    ]

    # Three steps done plus the current Review step: secondary, secondary,
    # secondary, primary -- the last step is never itself marked "done".
    bar.set_states(
        [
            StepState(True, done=True),
            StepState(True, done=True),
            StepState(True, done=True),
            StepState(True, current=True),
        ]
    )
    assert [button.property("role") for button in bar.buttons] == [
        "secondary",
        "secondary",
        "secondary",
        "primary",
    ]


def test_set_active_marks_only_the_shown_step(qtbot):
    bar, _calls = _build_bar(qtbot)

    bar.set_active(2)
    assert [button.property("active") for button in bar.buttons] == [
        "false",
        "false",
        "true",
        "false",
    ]

    bar.set_active(None)
    assert [button.property("active") for button in bar.buttons] == ["false"] * 4


def test_set_states_applies_enabled_state_and_hint(qtbot):
    bar, _calls = _build_bar(qtbot)

    states = [
        StepState(True, hint="first"),
        StepState(True, hint="second"),
        StepState(True, hint="third"),
        StepState(True, hint="fourth"),
    ]
    bar.set_states(states)

    for button, state in zip(bar.buttons, states, strict=True):
        assert button.isEnabled() == state.enabled
        assert button.toolTip() == state.hint


def test_set_states_ignores_extra_and_missing_states(qtbot):
    bar, _calls = _build_bar(qtbot)

    # Fewer states than buttons: the untouched buttons are left alone rather
    # than raising or being blanked out.
    bar.set_states([StepState(True, hint="only one")])
    assert bar.buttons[0].toolTip() == "only one"
    assert bar.buttons[1].text() == "②  Segment"

    # More states than buttons: the extra state is silently ignored.
    bar.set_states(
        [
            StepState(True, hint="a"),
            StepState(True, hint="b"),
            StepState(True, hint="c"),
            StepState(True, hint="d"),
            StepState(True, hint="e"),
        ]
    )
    assert [button.toolTip() for button in bar.buttons] == ["a", "b", "c", "d"]


def test_clicking_a_step_always_runs_its_action(qtbot):
    bar, calls = _build_bar(qtbot)
    bar.set_states(
        [StepState(True), StepState(True), StepState(True), StepState(True)]
    )

    qtbot.mouseClick(bar.buttons[0], Qt.MouseButton.LeftButton)
    qtbot.mouseClick(bar.buttons[3], Qt.MouseButton.LeftButton)
    assert calls == ["Study", "Review"]


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
        assert 'QPushButton#workflowStep[active="true"]' in stylesheet
