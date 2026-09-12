"""A four-page step stack for the right control panel.

The control panel used to be a pinned "Planning Cockpit" summary sitting
above a three-tab widget (Study / Planning / Tools); the Screw Review table
ended up squeezed to a couple of visible rows at the bottom of the Planning
tab. This module gives :class:`~src.ui.main_window.MainWindow` a step-based
replacement that mirrors the workflow bar (Study / Segment / Plan / Review):
each step is its own scrollable page, and the Review page is built around the
screw list rather than around a fixed-height inspector card.
"""

from __future__ import annotations

from typing import Sequence

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QLabel,
    QScrollArea,
    QSizePolicy,
    QStackedWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

#: The four workflow steps, in panel order.
STEP_NAMES = ("Study", "Segment", "Plan", "Review")

#: Minimum height the Review screw table must keep at a 1600x900 window --
#: used by the layout test, not enforced at runtime.
REVIEW_TABLE_MIN_VISIBLE_PX = 300


class StepPanel(QWidget):
    """A QStackedWidget of scrollable pages, one per workflow step."""

    step_changed = pyqtSignal(str)

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self.stack = QStackedWidget(self)
        layout.addWidget(self.stack)
        self._pages: dict[str, QWidget] = {}
        self._scroll_areas: dict[str, QScrollArea] = {}
        self._current: str | None = None

    def add_page(self, name: str, page: QWidget, *, scrollable: bool = True) -> None:
        """Add one named page, wrapped in a scroll area unless told otherwise."""
        if scrollable:
            scroll = QScrollArea(self.stack)
            scroll.setObjectName(f"stepPage{name}")
            scroll.setWidgetResizable(True)
            scroll.setHorizontalScrollBarPolicy(
                Qt.ScrollBarPolicy.ScrollBarAlwaysOff
            )
            scroll.setWidget(page)
            self._scroll_areas[name] = scroll
            self.stack.addWidget(scroll)
        else:
            page.setObjectName(f"stepPage{name}")
            self.stack.addWidget(page)
        self._pages[name] = page
        if self._current is None:
            self._current = name

    def show_step(self, name: str) -> None:
        """Show the named page; raises ValueError for an unknown name."""
        if name not in self._pages:
            raise ValueError(f"Unknown step: {name}")
        widget = self._scroll_areas.get(name, self._pages[name])
        if self._current != name:
            self._current = name
            self.stack.setCurrentWidget(widget)
            self.step_changed.emit(name)
        else:
            self.stack.setCurrentWidget(widget)

    @property
    def current_step(self) -> str | None:
        """Name of the currently shown page."""
        return self._current

    def page(self, name: str) -> QWidget:
        """Return the raw page widget for *name* (not its scroll wrapper)."""
        return self._pages[name]

    def scroll_area(self, name: str) -> QScrollArea | None:
        """Return the QScrollArea wrapping *name*'s page, or None if unscrollable."""
        return self._scroll_areas.get(name)


def warning_summary_text(count: int) -> str:
    """One-line summary text for a warning count."""
    count = int(count)
    if count <= 0:
        return "✓ No warnings"
    if count == 1:
        return "⚠ 1 warning"
    return f"⚠ {count} warnings"


class WarningSummary(QWidget):
    """A collapsed one-line warning count that expands to the full text."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        self.toggle = QToolButton(self)
        self.toggle.setObjectName("screwWarningsToggle")
        self.toggle.setCheckable(True)
        self.toggle.setChecked(False)
        self.toggle.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.toggle.setArrowType(Qt.ArrowType.RightArrow)
        self.toggle.setCursor(Qt.CursorShape.PointingHandCursor)
        self.toggle.toggled.connect(self._on_toggled)
        layout.addWidget(self.toggle)

        self.label = QLabel(self)
        self.label.setObjectName("selectedScrewWarning")
        self.label.setWordWrap(True)
        self.label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred
        )
        self.label.hide()
        layout.addWidget(self.label)

        self._count = 0
        self.set_lines(["Select a screw to inspect its trajectory."], 0)

    def _on_toggled(self, checked: bool) -> None:
        self.toggle.setArrowType(
            Qt.ArrowType.DownArrow if checked else Qt.ArrowType.RightArrow
        )
        self.label.setVisible(checked)

    def set_lines(self, lines: Sequence[str], count: int) -> None:
        """Set the toggle's one-line summary and the full expandable text."""
        self._count = int(count)
        self.toggle.setEnabled(True)
        self.toggle.setText(warning_summary_text(self._count))
        # Clear the placeholder's tooltip: without this, a screw selected
        # after the "no screw selected" placeholder was shown would keep
        # showing that stale placeholder text on hover.
        self.toggle.setToolTip("")
        self.label.setText("\n".join(lines))

    def set_placeholder(self, text: str) -> None:
        """Show a fixed disabled caption (used when nothing is selected).

        The toggle is a QToolButton, which never wraps its text -- a long
        sentence there would force the whole key-numbers card wide enough to
        fit it on one line. The full text still reaches the user, as the
        toggle's tooltip.
        """
        short = text if len(text) <= 24 else "No screw selected"
        self.toggle.setText(short)
        self.toggle.setToolTip(text)
        self.toggle.setEnabled(False)
        self.toggle.setChecked(False)
        self.label.setText(text)
