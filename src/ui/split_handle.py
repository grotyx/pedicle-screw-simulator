"""A thin bar that resizes the grid rows above and below it."""

from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import QFrame


class RowSplitHandle(QFrame):
    """Emits the pointer's y (in the parent's coordinates) while dragged.

    The views share one grid and must never change parent, so a QSplitter is
    out; the window turns this y into row stretches instead.
    """

    dragged = pyqtSignal(int)
    released = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("rowSplitHandle")
        self.setFixedHeight(6)
        self.setCursor(Qt.CursorShape.SplitVCursor)
        self.setToolTip("Drag to resize the main view and the thumbnails")

    def mouseMoveEvent(self, event):  # noqa: N802 - Qt override
        if event.buttons() & Qt.MouseButton.LeftButton:
            self.dragged.emit(self.mapToParent(event.position().toPoint()).y())

    def mouseReleaseEvent(self, event):  # noqa: N802 - Qt override
        if event.button() == Qt.MouseButton.LeftButton:
            self.released.emit()
