"""The four workflow steps of a planning session, above the views.

The bar used to run the three primary actions of a session directly (Open
DICOM, Segment, Plan Screws); this now **navigates**, one step per page of
the right-hand :class:`~src.ui.step_panel.StepPanel`. Every step is always
clickable -- unlike the old three-action bar, a step is never disabled, since
every page (Study/Segment/Plan/Review) is reachable at any time and the
buttons that actually run an action live on the pages themselves.
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

#: The four workflow steps, in panel order.
STEP_NAMES = ("Study", "Segment", "Plan", "Review")


@dataclass(frozen=True)
class StepState:
    """What one step looks like right now.

    ``done`` marks a step already completed for this study; ``current`` is the
    next thing to do, drawn as the primary action.  ``hint`` is the tooltip.
    Every step is ``enabled`` now that the bar only navigates: every page is
    always reachable.
    """

    enabled: bool
    done: bool = False
    current: bool = False
    hint: str = ""


@dataclass(frozen=True)
class WorkflowStep:
    """One numbered step: its label and the action it runs (show its page)."""

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

    def set_active(self, index: Optional[int]) -> None:
        """Mark the button at *index* as the currently shown step (or none)."""
        for i, button in enumerate(self._buttons):
            active = "true" if i == index else "false"
            if button.property("active") != active:
                button.setProperty("active", active)
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
    has_screws: bool = False,
) -> List[StepState]:
    """The four steps' states from what the session has done so far.

    Kept free of Qt so the rule can be tested directly. Every step is always
    ``enabled``: the bar only navigates, and every page is reachable whatever
    the study's state. ``current`` is the first step not yet done, so once a
    plan exists the Review step is the one highlighted.
    """
    done = [
        has_volume,
        has_volume and has_mask,
        has_volume and has_mask and has_plan,
        False,
    ]
    hints = [
        (
            "Study loaded — open another one here"
            if has_volume
            else "Open a DICOM series folder"
        ),
        (
            "Run TotalSegmentator on the loaded study"
            if has_volume
            else "Open a DICOM series first"
        ),
        (
            "Plan screws for the selected vertebral levels"
            if has_mask and plan_available
            else "Select vertebral levels in the Plan step"
            if has_mask
            else "Run segmentation first"
        ),
        (
            "Review the screws level by level"
            if has_screws
            else "Plan or place screws first"
        ),
    ]
    first_open = next((i for i, finished in enumerate(done) if not finished), None)
    return [
        StepState(
            enabled=True,
            done=done[i],
            current=(i == first_open),
            hint=hints[i],
        )
        for i in range(4)
    ]
