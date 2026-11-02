"""Floating tool palette shown over the views."""

from __future__ import annotations

from typing import List, Optional, Sequence

from PyQt6.QtCore import QPoint, QSize, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import QApplication, QFrame, QLabel, QSizePolicy, QToolButton, QVBoxLayout


class ToolDock(QFrame):
    """A column of icon buttons driven by existing QActions; ``None`` is a separator.

    The default strip sits outside the views. Drag the grip away to float
    over the views, then drop on the strip or double-click the grip to attach.
    Only this frame changes parent; the native VTK panes never do.
    """

    #: Distance in pixels at which a dropped palette snaps to an edge.
    SNAP = 24
    #: Gap kept between the palette and the edge it snaps to.
    MARGIN = 8

    moved = pyqtSignal(QPoint)
    mode_changed = pyqtSignal(str)

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
        self._grip.setToolTip("Drag away to detach; drop on the left strip or double-click to attach")
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
        self.is_attached = False
        self._strip = None
        self._floating_host = parent
        self._floating_position = QPoint(self.MARGIN, 40)
        self._pending_position = None
        self._move_timer = QTimer(self)
        self._move_timer.setInterval(34)
        self._move_timer.setSingleShot(True)
        self._move_timer.timeout.connect(self._flush_move)
        self._sized = False

    def configure_attachment(self, strip, floating_host) -> None:
        """Give the palette an independent strip and its floating overlay host."""
        self._strip = strip
        self._floating_host = floating_host
        layout = QVBoxLayout(strip)
        layout.setContentsMargins(2, 6, 2, 6)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.set_attached(True)

    def set_attached(self, attached: bool) -> None:
        if self._strip is None or attached == self.is_attached:
            return
        self._move_timer.stop()
        self._pending_position = None
        if attached:
            self._floating_position = self.pos()
            self.moved.emit(self._floating_position)
            self.setParent(self._strip)
            self._strip.setFixedWidth(self.sizeHint().width() + 4)
            self._strip.layout().addWidget(self)
        else:
            self._strip.layout().removeWidget(self)
            self.setParent(self._floating_host)
            self._strip.setFixedWidth(16)
        self.is_attached = attached
        self.show()
        if not attached:
            self.place(self._floating_position)
            self.raise_()
        self.mode_changed.emit("attached" if attached else "floating")

    def _over_strip(self, global_pos) -> bool:
        return (self._strip is not None
                and self._strip.rect().contains(self._strip.mapFromGlobal(global_pos)))

    def _flush_move(self) -> None:
        if self._pending_position is not None:
            position = self._pending_position
            self._pending_position = None
            self.place(position)

    def buttons(self) -> List[QToolButton]:
        return list(self._buttons)

    def place(self, top_left: QPoint) -> QPoint:
        """Move inside the parent, snapping to an edge it lands close to."""
        if self.is_attached:
            self._floating_position = QPoint(top_left)
            return self._floating_position
        if not self._sized:
            self.adjustSize()
            self._sized = True
        parent = self.parentWidget()
        if parent is None:
            self.move(top_left)
            return self.pos()
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
        self._floating_position = self.pos()
        return self.pos()

    def eventFilter(self, obj, event):  # noqa: N802 - Qt override
        if obj is self._grip:
            kind = event.type()
            if kind == event.Type.MouseButtonDblClick and event.button() == Qt.MouseButton.LeftButton:
                self._drag_start = None
                self.set_attached(True)
                return True
            if kind == event.Type.MouseButtonPress and event.button() == Qt.MouseButton.LeftButton:
                pressed = event.globalPosition().toPoint()
                self._drag_start = (pressed, self.mapToGlobal(QPoint()))
                return True
            if kind == event.Type.MouseMove and self._drag_start is not None:
                pressed_at, start_global = self._drag_start
                current = event.globalPosition().toPoint()
                if self.is_attached:
                    if ((current - pressed_at).manhattanLength() < QApplication.startDragDistance()
                            or self._over_strip(current)):
                        return True
                    self.set_attached(False)
                    self._grip.grabMouse()
                self._pending_position = self._floating_host.mapFromGlobal(
                    start_global + current - pressed_at
                )
                if not self._move_timer.isActive():
                    self._move_timer.start()
                return True
            if kind == event.Type.MouseButtonRelease and self._drag_start is not None:
                self._move_timer.stop()
                if not self.is_attached:
                    pressed_at, start_global = self._drag_start
                    current = event.globalPosition().toPoint()
                    self._pending_position = self._floating_host.mapFromGlobal(
                        start_global + current - pressed_at
                    )
                    self._flush_move()
                    if self._over_strip(current):
                        self.set_attached(True)
                    else:
                        self.moved.emit(self.pos())
                    self._grip.releaseMouse()
                self._drag_start = None
                return True
        return super().eventFilter(obj, event)
