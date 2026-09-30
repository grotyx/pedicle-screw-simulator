"""Floating tool palette shown over the main view."""

from __future__ import annotations

from typing import List, Optional, Sequence

from PyQt6.QtCore import QSize, Qt
from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import QFrame, QHBoxLayout, QSizePolicy, QToolButton


class ToolDock(QFrame):
    """A row of icon buttons driven by existing QActions; ``None`` is a separator.

    A plain QFrame, so it can move between viewers' overlay cells freely --
    unlike the VTK panes, which must never change parent.
    """

    def __init__(self, actions: Sequence[Optional[QAction]], parent=None):
        super().__init__(parent)
        self.setObjectName("toolDock")
        self.setProperty("viewerOverlay", "true")
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(4)
        self._buttons: List[QToolButton] = []
        for action in actions:
            if action is None:
                separator = QFrame(self)
                separator.setObjectName("toolDockSeparator")
                # A plain 1 px fill from the stylesheet; a VLine frame
                # draws its own shaded bar on top of that.
                separator.setFixedWidth(1)
                layout.addWidget(separator)
                continue
            button = QToolButton(self)
            button.setDefaultAction(action)
            button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonIconOnly)
            button.setIconSize(QSize(20, 20))
            button.setAutoRaise(True)
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            layout.addWidget(button)
            self._buttons.append(button)

    def buttons(self) -> List[QToolButton]:
        return list(self._buttons)
