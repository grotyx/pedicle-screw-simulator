"""
Collapsible Group Box widget for the control panel.

Click the group header to expand/collapse content with smooth animation.
Uses QPropertyAnimation for height transitions.
"""

from PyQt6.QtCore import (
    QAbstractAnimation,
    QParallelAnimationGroup,
    QPropertyAnimation,
    Qt,
)
from PyQt6.QtWidgets import QLayout, QSizePolicy, QToolButton, QVBoxLayout, QWidget


class CollapsibleGroupBox(QWidget):
    """
    A group box that collapses/expands its content area on header click.

    Usage:
        group = CollapsibleGroupBox("Window/Level")
        layout = QVBoxLayout()
        layout.addWidget(some_widget)
        group.set_content_layout(layout)
    """

    ANIMATION_DURATION_MS = 200

    def __init__(self, title: str = "", parent: QWidget = None):
        super().__init__(parent)
        self.setObjectName("collapsibleSection")

        self._is_collapsed = False
        self._animation_group = QParallelAnimationGroup(self)

        # Header toggle button
        self._toggle_button = QToolButton(self)
        self._toggle_button.setObjectName("sectionHeader")
        self._toggle_button.setToolButtonStyle(
            Qt.ToolButtonStyle.ToolButtonTextBesideIcon
        )
        self._toggle_button.setArrowType(Qt.ArrowType.DownArrow)
        self._toggle_button.setText(title)
        self._toggle_button.setCheckable(True)
        self._toggle_button.setChecked(False)
        self._toggle_button.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        self._toggle_button.clicked.connect(self._on_toggle)

        # Content area
        self._content_area = QWidget(self)
        self._content_area.setObjectName("sectionContent")
        self._content_area.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        self._content_area.setMaximumHeight(0)
        self._content_area.setMinimumHeight(0)

        # Animation for content height
        self._content_animation = QPropertyAnimation(
            self._content_area, b"maximumHeight"
        )
        self._content_animation.setDuration(self.ANIMATION_DURATION_MS)
        self._animation_group.addAnimation(self._content_animation)

        # Main layout
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(0)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(self._toggle_button)
        main_layout.addWidget(self._content_area)

        # Start expanded
        self._is_collapsed = True
        self._toggle_button.setChecked(False)
        # Will expand on first set_content_layout call

    def set_content_layout(self, layout: QLayout) -> None:
        """
        Set the layout for the collapsible content area.

        Must be called exactly once after construction.
        """
        # Replace any existing layout
        old_layout = self._content_area.layout()
        if old_layout is not None:
            # QWidget doesn't allow replacing layouts directly; delete children
            while old_layout.count():
                item = old_layout.takeAt(0)
                widget = item.widget()
                if widget:
                    widget.deleteLater()

        self._content_area.setLayout(layout)
        content_height = layout.sizeHint().height()

        # Start expanded
        self._content_area.setMaximumHeight(content_height)
        self._is_collapsed = False
        self._toggle_button.setChecked(False)
        self._toggle_button.setArrowType(Qt.ArrowType.DownArrow)

    def _on_toggle(self, checked: bool) -> None:
        """Handle header click: animate expand/collapse."""
        if self._animation_group.state() == QAbstractAnimation.State.Running:
            self._animation_group.stop()

        content_height = self._content_area.layout().sizeHint().height()

        if checked:
            # Collapse
            self._toggle_button.setArrowType(Qt.ArrowType.RightArrow)
            self._content_animation.setStartValue(content_height)
            self._content_animation.setEndValue(0)
            self._is_collapsed = True
        else:
            # Expand
            self._toggle_button.setArrowType(Qt.ArrowType.DownArrow)
            self._content_animation.setStartValue(0)
            self._content_animation.setEndValue(content_height)
            self._is_collapsed = False

        self._animation_group.start()

    @property
    def is_collapsed(self) -> bool:
        return self._is_collapsed

    def set_title(self, title: str) -> None:
        self._toggle_button.setText(title)

    def collapse(self) -> None:
        """Programmatically collapse the group."""
        if not self._is_collapsed:
            self._toggle_button.setChecked(True)
            self._on_toggle(True)

    def expand(self) -> None:
        """Programmatically expand the group."""
        if self._is_collapsed:
            self._toggle_button.setChecked(False)
            self._on_toggle(False)
