"""Floating tool palette shown over the views."""

from __future__ import annotations

from typing import List, Optional, Sequence

from PyQt6.QtCore import QPoint, QSize, Qt, pyqtSignal
from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import QFrame, QLabel, QSizePolicy, QToolButton, QVBoxLayout


class ToolDock(QFrame):
    """A column of icon buttons driven by existing QActions; ``None`` is a separator.

    It floats over the views: drag the grip at the top to put it anywhere,
    and it snaps to an edge it is dropped near. A plain QFrame, so it can sit
    over the VTK panes without them ever changing parent.
    """

    #: Distance in pixels at which a dropped palette snaps to an edge.
    SNAP = 24
    #: Gap kept between the palette and the edge it snaps to.
    MARGIN = 8

    moved = pyqtSignal(QPoint)

    def __init__(self, actions: Sequence[Optional[QAction]], parent=None):
        super().__init__(parent)
        self.setObjectName("toolDock")
        self.setProperty("viewerOverlay", "true")
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 2, 4, 4)
        layout.setSpacing(2)
        self._grip = QLabel("⠿", self)
        self._grip.setObjectName("toolDockGrip")
        self._grip.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._grip.setCursor(Qt.CursorShape.SizeAllCursor)
        self._grip.setToolTip("Drag to move the tools")
        layout.addWidget(self._grip)
        self._buttons: List[QToolButton] = []
        for action in actions:
            if action is None:
                separator = QFrame(self)
                separator.setObjectName("toolDockSeparator")
                # A plain 1 px fill from the stylesheet; an HLine frame
                # draws its own shaded bar on top of that.
                separator.setFixedHeight(1)
                layout.addWidget(separator)
                continue
            button = QToolButton(self)
            button.setDefaultAction(action)
            button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonIconOnly)
            button.setIconSize(QSize(18, 18))
            button.setAutoRaise(True)
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            layout.addWidget(button)
            self._buttons.append(button)
        self._drag_start: Optional[tuple] = None
        self._grip.installEventFilter(self)

    def buttons(self) -> List[QToolButton]:
        return list(self._buttons)

    def place(self, top_left: QPoint) -> QPoint:
        """Move inside the parent, snapping to an edge it lands close to."""
        parent = self.parentWidget()
        if parent is None:
            self.move(top_left)
            return self.pos()
        self.adjustSize()
        max_x = max(self.MARGIN, parent.width() - self.width() - self.MARGIN)
        max_y = max(self.MARGIN, parent.height() - self.height() - self.MARGIN)
        x = min(max(top_left.x(), self.MARGIN), max_x)
        y = min(max(top_left.y(), self.MARGIN), max_y)
        if x - self.MARGIN < self.SNAP:
            x = self.MARGIN
        elif max_x - x < self.SNAP:
            x = max_x
        if y - self.MARGIN < self.SNAP:
            y = self.MARGIN
        elif max_y - y < self.SNAP:
            y = max_y
        self.move(x, y)
        self.raise_()
        return self.pos()

    def eventFilter(self, obj, event):  # noqa: N802 - Qt override
        if obj is self._grip:
            kind = event.type()
            if kind == event.Type.MouseButtonPress and event.button() == Qt.MouseButton.LeftButton:
                self._drag_start = (event.globalPosition().toPoint(), self.pos())
                return True
            if kind == event.Type.MouseMove and self._drag_start is not None:
                pressed_at, start_pos = self._drag_start
                self.place(start_pos + event.globalPosition().toPoint() - pressed_at)
                return True
            if kind == event.Type.MouseButtonRelease and self._drag_start is not None:
                self._drag_start = None
                self.moved.emit(self.pos())
                return True
        return super().eventFilter(obj, event)
