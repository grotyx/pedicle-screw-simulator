"""Clickable viewer title bar shared by the MPR and 3D viewports."""

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QLabel


class ViewerHeaderLabel(QLabel):
    """A viewer title that reports double-clicks for single-view maximise."""

    doubleClicked = pyqtSignal(str)

    def __init__(self, text: str, view_name: str, parent=None):
        super().__init__(text, parent)
        self.setObjectName("viewerHeader")
        self._view_name = str(view_name)
        self.setToolTip("Double-click to maximize or restore this view")

    @property
    def view_name(self) -> str:
        """The name this header reports when it is double-clicked."""
        return self._view_name

    def mouseDoubleClickEvent(self, event) -> None:  # noqa: N802 (Qt override)
        self.doubleClicked.emit(self._view_name)
        event.accept()
