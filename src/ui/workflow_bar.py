"""The three primary actions of a planning session, in order, above the views.

Opening a study, segmenting it and planning screws are the whole workflow,
but the last two buttons lived inside the Study and Planning tabs, below the
fold of a scrolled panel -- the one thing a first-time user needs to find was
the hardest thing on screen to find.  This bar puts them in a row, numbered,
where the eye starts: ``① Open DICOM → ② Segment → ③ Plan Screws``.

It is a view, not a second copy of the workflow.  Each step runs the existing
action it names, and :class:`MainWindow` tells it what state each step is in;
the tab buttons stay exactly as they were, so nothing that relies on them
changes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QSizePolicy, QWidget

#: Circled digits for the step numbers; a plain "1." read as a list, not a path.
_STEP_NUMBERS = ("①", "②", "③", "④", "⑤")

#: Shown before a finished step's label.
DONE_MARK = "✓"


@dataclass(frozen=True)
class StepState:
    """What one step looks like right now.

    ``done`` marks a step already completed for this study; ``current`` is the
    next thing to do, drawn as the primary action.  ``hint`` is the tooltip,
    which for a disabled step should say what unlocks it.
    """

    enabled: bool
    done: bool = False
    current: bool = False
    hint: str = ""


@dataclass(frozen=True)
class WorkflowStep:
    """One numbered step: its label and the action it runs."""

    label: str
    action: Callable[[], None]


class WorkflowBar(QWidget):
    """A row of numbered, state-aware step buttons joined by arrows."""

    def __init__(self, steps: Sequence[WorkflowStep], parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setObjectName("workflowBar")
        # QWidget subclasses ignore stylesheet backgrounds without this.
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 6, 10, 6)
        layout.setSpacing(8)

        self._steps: List[WorkflowStep] = list(steps)
        self._buttons: List[QPushButton] = []
        for index, step in enumerate(self._steps):
            if index:
                chevron = QLabel("→")
                chevron.setObjectName("workflowChevron")
                chevron.setAlignment(Qt.AlignmentFlag.AlignCenter)
                layout.addWidget(chevron)
            button = QPushButton(self._text(index, step.label, done=False))
            button.setObjectName("workflowStep")
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            button.clicked.connect(step.action)
            layout.addWidget(button)
            self._buttons.append(button)
        layout.addStretch(1)

    @property
    def buttons(self) -> List[QPushButton]:
        """The step buttons, in order."""
        return list(self._buttons)

    def set_states(self, states: Sequence[StepState]) -> None:
        """Redraw every step from its state.  Extra or missing states are ignored."""
        for index, (button, state) in enumerate(zip(self._buttons, states, strict=False)):
            button.setEnabled(bool(state.enabled))
            button.setText(self._text(index, self._steps[index].label, done=state.done))
            button.setToolTip(state.hint)
            # The existing role styles carry the theme: the next step is the
            # primary action, a finished one is quiet, the rest are plain.
            role = "primary" if state.current else "secondary" if state.done else ""
            if button.property("role") != role:
                button.setProperty("role", role)
                # A dynamic property only restyles once the style is re-applied.
                button.style().unpolish(button)
                button.style().polish(button)

    @staticmethod
    def _text(index: int, label: str, *, done: bool) -> str:
        number = _STEP_NUMBERS[index] if index < len(_STEP_NUMBERS) else f"{index + 1}."
        return f"{DONE_MARK} {label}" if done else f"{number}  {label}"


def workflow_states(
    *,
    has_volume: bool,
    segment_available: bool,
    has_mask: bool,
    plan_available: bool,
    has_plan: bool,
) -> List[StepState]:
    """The three steps' states from what the session has done so far.

    Kept free of Qt so the rule can be tested directly.  The *current* step is
    the first one not yet done -- highlighted even while it is disabled, so the
    bar always points at what comes next, and its tooltip says what it waits
    for.  A later step never looks done while an earlier one is not: a plan
    from a previous study must not show ③ as finished on a fresh one.
    """
    done = [has_volume, has_volume and has_mask, has_volume and has_mask and has_plan]
    enabled = [True, has_volume and segment_available, has_mask and plan_available]
    hints = [
        "Open a DICOM series folder",
        (
            "Run TotalSegmentator on the loaded study"
            if has_volume
            else "Open a DICOM series first"
        ),
        (
            "Plan screws for the selected vertebral levels"
            if has_mask and plan_available
            else "Select vertebral levels in the Planning tab"
            if has_mask
            else "Run segmentation first"
        ),
    ]
    first_open = next((i for i, finished in enumerate(done) if not finished), None)
    return [
        StepState(
            enabled=enabled[i],
            done=done[i],
            current=(i == first_open),
            hint=hints[i],
        )
        for i in range(3)
    ]
