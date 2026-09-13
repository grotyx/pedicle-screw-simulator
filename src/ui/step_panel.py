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
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMenu,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QStackedWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ..core.planner_config import PlannerConfig
from ..utils.constants import (
    DEFAULT_SCREW_DIAMETER,
    DEFAULT_SCREW_LENGTH,
    MAX_SCREW_DIAMETER,
    MIN_SCREW_DIAMETER,
    TRANSFER_FUNCTION_PRESETS,
)
from .collapsible_group import CollapsibleGroupBox
from .screw_plan_table import ScrewPlanTable
from .spin_boxes import DiameterSpinBox

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


def build_study_page() -> tuple[QWidget, dict]:
    """Build the Study step page; returns (page, refs keyed by attribute name)."""
    study_page = QWidget()
    study_layout = QVBoxLayout(study_page)
    study_layout.setContentsMargins(6, 6, 6, 6)
    study_layout.setSpacing(4)

    open_dicom_btn = QPushButton("Open DICOM…")
    open_dicom_btn.setProperty("role", "primary")
    study_layout.addWidget(open_dicom_btn)

    info_group = CollapsibleGroupBox("Study")
    info_layout = QVBoxLayout()
    info_label = QLabel("No volume loaded")
    info_label.setWordWrap(True)
    info_layout.addWidget(info_label)
    info_group.set_content_layout(info_layout)
    study_layout.addWidget(info_group)

    wl_group = CollapsibleGroupBox("Window/Level")
    wl_layout = QGridLayout()
    wl_layout.addWidget(QLabel("Window:"), 0, 0)
    window_slider = QSlider(Qt.Orientation.Horizontal)
    window_slider.setRange(1, 4000)
    window_slider.setValue(1500)
    wl_layout.addWidget(window_slider, 0, 1)
    wl_layout.addWidget(QLabel("Level:"), 1, 0)
    level_slider = QSlider(Qt.Orientation.Horizontal)
    level_slider.setRange(-1000, 3000)
    level_slider.setValue(400)
    wl_layout.addWidget(level_slider, 1, 1)
    preset_layout = QHBoxLayout()
    _btn_bone = QPushButton("Bone")
    _btn_soft = QPushButton("Soft Tissue")
    preset_layout.addWidget(_btn_bone)
    preset_layout.addWidget(_btn_soft)
    wl_layout.addLayout(preset_layout, 2, 0, 1, 2)
    wl_group.set_content_layout(wl_layout)
    study_layout.addWidget(wl_group)

    view_group = CollapsibleGroupBox("3D Rendering")
    vr_layout = QVBoxLayout()
    tf_preset_layout = QHBoxLayout()
    tf_preset_layout.addWidget(QLabel("Preset:"))
    tf_preset_combo = QComboBox()
    for name in TRANSFER_FUNCTION_PRESETS:
        tf_preset_combo.addItem(name)
    tf_preset_layout.addWidget(tf_preset_combo)
    vr_layout.addLayout(tf_preset_layout)
    opacity_layout = QHBoxLayout()
    opacity_layout.addWidget(QLabel("Opacity:"))
    opacity_slider = QSlider(Qt.Orientation.Horizontal)
    opacity_slider.setRange(0, 100)
    opacity_slider.setValue(100)
    opacity_layout.addWidget(opacity_slider)
    vr_layout.addLayout(opacity_layout)
    view_group.set_content_layout(vr_layout)
    study_layout.addWidget(view_group)
    study_layout.addStretch()

    refs = {
        "open_dicom_btn": open_dicom_btn,
        "info_label": info_label,
        "window_slider": window_slider,
        "level_slider": level_slider,
        "_btn_bone": _btn_bone,
        "_btn_soft": _btn_soft,
        "tf_preset_combo": tf_preset_combo,
        "opacity_slider": opacity_slider,
        "info_group": info_group,
        "wl_group": wl_group,
        "view_group": view_group,
    }
    return study_page, refs


def build_segment_page() -> tuple[QWidget, dict]:
    """Build the Segment step page; returns (page, refs keyed by attribute name).

    The vertebra-level selector container is created here (detected levels
    first arrive via Segmentation) but added to the Plan page by the caller,
    mirroring the original layout.
    """
    segment_page = QWidget()
    segment_layout = QVBoxLayout(segment_page)
    segment_layout.setContentsMargins(6, 6, 6, 6)
    segment_layout.setSpacing(4)

    seg_group = CollapsibleGroupBox("Segmentation")
    seg_layout = QVBoxLayout()
    seg_layout.setContentsMargins(6, 2, 6, 4)
    seg_layout.setSpacing(3)

    seg_run_btn = QPushButton("Run Auto Segmentation")
    seg_run_btn.setProperty("role", "primary")
    seg_layout.addWidget(seg_run_btn)

    seg_refine_check = QCheckBox("Refine boundaries against CT")
    seg_refine_check.setChecked(True)
    seg_refine_check.setToolTip(
        "Smooth the segmentation to the CT grid and snap its boundaries to "
        "the bone cortex. Turn off to keep TotalSegmentator's raw 1.5 mm "
        "label map."
    )
    seg_layout.addWidget(seg_refine_check)

    seg_status_label = QLabel("Ready for automatic segmentation")
    seg_status_label.setWordWrap(True)
    seg_status_label.setObjectName("segmentationStatus")
    seg_layout.addWidget(seg_status_label)

    vertebra_isolate_btn = QPushButton("Isolate Vertebrae")
    vertebra_isolate_btn.setEnabled(False)
    seg_layout.addWidget(vertebra_isolate_btn)

    seg_group.set_content_layout(seg_layout)
    segment_layout.addWidget(seg_group)

    isolation_hint_label = QLabel(
        "Vertebrae are isolated automatically after TotalSegmentator. "
        "Switch with Vertebrae / Full CT on the 3D view."
    )
    isolation_hint_label.setWordWrap(True)
    isolation_hint_label.setObjectName("mutedHint")
    segment_layout.addWidget(isolation_hint_label)

    # Levels-to-plan selection now lives on the Plan page; the grid and
    # its bookkeeping are still built here because Segmentation is where
    # detected levels first arrive (update_vertebra_level_checks below).
    vertebra_level_container = QWidget()
    _vertebra_level_grid = QGridLayout(vertebra_level_container)
    _vertebra_level_grid.setContentsMargins(0, 0, 0, 0)
    _vertebra_level_grid.setHorizontalSpacing(8)
    _vertebra_level_grid.setVerticalSpacing(4)
    _vertebra_level_checks: dict = {}
    _updating_vertebra_level_checks = False
    _vertebra_level_placeholder = QLabel("Run segmentation first")
    _vertebra_level_placeholder.setObjectName("segmentationStatus")
    _vertebra_level_grid.addWidget(
        _vertebra_level_placeholder,
        0,
        0,
        1,
        4,
    )

    seg_advanced_panel = QWidget()
    advanced_layout = QGridLayout(seg_advanced_panel)
    advanced_layout.setContentsMargins(0, 0, 0, 0)

    advanced_layout.addWidget(QLabel("Task:"), 0, 0)
    seg_task_combo = QComboBox()
    seg_task_combo.addItem("Spine Only", "spine_only")
    seg_task_combo.addItem("Total (117 structures)", "total")
    advanced_layout.addWidget(seg_task_combo, 0, 1)

    advanced_layout.addWidget(QLabel("Device:"), 1, 0)
    seg_device_combo = QComboBox()
    seg_device_combo.addItem("GPU (preferred)", "gpu")
    seg_device_combo.addItem("CPU fallback", "cpu")
    advanced_layout.addWidget(seg_device_combo, 1, 1)

    advanced_layout.addWidget(QLabel("Quality:"), 2, 0)
    seg_quality_combo = QComboBox()
    seg_quality_combo.addItem("Accurate (1.5 mm)", "accurate")
    seg_quality_combo.addItem("Memory Saver (3 mm)", "fast")
    seg_quality_combo.setToolTip(
        "Accurate mode uses more memory. Memory Saver is faster but may "
        "reduce boundary precision."
    )
    advanced_layout.addWidget(seg_quality_combo, 2, 1)

    advanced_layout.addWidget(QLabel("Label ID:"), 3, 0)
    seg_label_spin = QSpinBox()
    seg_label_spin.setRange(0, 1000)
    seg_label_spin.setValue(0)
    seg_label_spin.setToolTip(
        "0 shows all labels, other values show only that label."
    )
    advanced_layout.addWidget(seg_label_spin, 3, 1)

    advanced_layout.addWidget(QLabel("Label Name:"), 4, 0)
    seg_label_combo = QComboBox()
    advanced_layout.addWidget(seg_label_combo, 4, 1)

    seg_label_info = QLabel("Selected: All labels")
    seg_label_info.setWordWrap(True)
    advanced_layout.addWidget(seg_label_info, 5, 0, 1, 2)

    seg_show_2d_check = QCheckBox("Show 2D Overlay")
    seg_show_2d_check.setChecked(True)
    advanced_layout.addWidget(seg_show_2d_check, 6, 0, 1, 2)

    seg_show_3d_check = QCheckBox("Show 3D Overlay")
    seg_show_3d_check.setChecked(False)
    advanced_layout.addWidget(seg_show_3d_check, 7, 0, 1, 2)

    seg_clear_btn = QPushButton("Clear Segmentation Overlay")
    advanced_layout.addWidget(seg_clear_btn, 8, 0, 1, 2)

    vertebra_restore_btn = QPushButton("Show Full Volume")
    vertebra_restore_btn.setEnabled(False)
    vertebra_restore_btn.hide()
    advanced_layout.addWidget(vertebra_restore_btn, 9, 0, 1, 2)

    advanced_layout.addWidget(QLabel("3D Vertebrae:"), 10, 0, 1, 2)
    vertebra_display_list = QListWidget()
    vertebra_display_list.setSelectionMode(
        QAbstractItemView.SelectionMode.ExtendedSelection
    )
    vertebra_display_list.setMaximumHeight(110)
    vertebra_display_list.setToolTip(
        "Select one or more segmented vertebrae to show in 3D"
    )
    advanced_layout.addWidget(vertebra_display_list, 11, 0, 1, 2)

    vertebra_display_buttons = QHBoxLayout()
    vertebra_show_selected_btn = QPushButton("Show Selected 3D")
    vertebra_show_all_btn = QPushButton("Show All Segmented")
    vertebra_show_selected_btn.setEnabled(False)
    vertebra_show_all_btn.setEnabled(False)
    vertebra_display_buttons.addWidget(vertebra_show_selected_btn)
    vertebra_display_buttons.addWidget(vertebra_show_all_btn)
    advanced_layout.addLayout(vertebra_display_buttons, 12, 0, 1, 2)

    seg_use_subregion_check = QCheckBox("Use pedicle subregion model")
    seg_use_subregion_check.setChecked(False)
    seg_use_subregion_check.setToolTip(
        "Refine pedicle detection with a locally installed nnU-Net "
        "subregion model. Leave off to use TotalSegmentator alone."
    )
    advanced_layout.addWidget(seg_use_subregion_check, 13, 0, 1, 2)

    subregion_dir_row = QHBoxLayout()
    seg_subregion_dir_edit = QLineEdit()
    seg_subregion_dir_edit.setPlaceholderText(
        "Model directory (dataset.json)"
    )
    subregion_dir_row.addWidget(seg_subregion_dir_edit, 1)
    seg_subregion_browse_btn = QPushButton("Browse...")
    subregion_dir_row.addWidget(seg_subregion_browse_btn)
    advanced_layout.addLayout(subregion_dir_row, 14, 0, 1, 2)

    seg_advanced_panel.hide()

    seg_advanced_toggle = QToolButton()
    seg_advanced_toggle.setObjectName("segAdvancedToggle")
    seg_advanced_toggle.setToolButtonStyle(
        Qt.ToolButtonStyle.ToolButtonTextBesideIcon
    )
    seg_advanced_toggle.setArrowType(Qt.ArrowType.RightArrow)
    seg_advanced_toggle.setText("Advanced options")
    seg_advanced_toggle.setCheckable(True)
    seg_advanced_toggle.setChecked(False)
    segment_layout.addWidget(seg_advanced_toggle)
    segment_layout.addWidget(seg_advanced_panel)
    segment_layout.addStretch()

    refs = {
        "segmentation_group": seg_group,
        "seg_group": seg_group,
        "seg_run_btn": seg_run_btn,
        "seg_refine_check": seg_refine_check,
        "seg_status_label": seg_status_label,
        "vertebra_isolate_btn": vertebra_isolate_btn,
        "isolation_hint_label": isolation_hint_label,
        "vertebra_level_container": vertebra_level_container,
        "_vertebra_level_grid": _vertebra_level_grid,
        "_vertebra_level_checks": _vertebra_level_checks,
        "_updating_vertebra_level_checks": _updating_vertebra_level_checks,
        "_vertebra_level_placeholder": _vertebra_level_placeholder,
        "seg_advanced_panel": seg_advanced_panel,
        "seg_task_combo": seg_task_combo,
        "seg_device_combo": seg_device_combo,
        "seg_quality_combo": seg_quality_combo,
        "seg_label_spin": seg_label_spin,
        "seg_label_combo": seg_label_combo,
        "seg_label_info": seg_label_info,
        "seg_show_2d_check": seg_show_2d_check,
        "seg_show_3d_check": seg_show_3d_check,
        "seg_clear_btn": seg_clear_btn,
        "vertebra_restore_btn": vertebra_restore_btn,
        "vertebra_display_list": vertebra_display_list,
        "vertebra_show_selected_btn": vertebra_show_selected_btn,
        "vertebra_show_all_btn": vertebra_show_all_btn,
        "seg_use_subregion_check": seg_use_subregion_check,
        "seg_subregion_dir_edit": seg_subregion_dir_edit,
        "seg_subregion_browse_btn": seg_subregion_browse_btn,
        "seg_advanced_toggle": seg_advanced_toggle,
    }
    return segment_page, refs


def build_plan_page(vertebra_level_container: QWidget) -> tuple[QWidget, dict]:
    """Build the Plan step page; returns (page, refs keyed by attribute name)."""
    plan_page = QWidget()
    plan_layout = QVBoxLayout(plan_page)
    plan_layout.setContentsMargins(6, 6, 6, 6)
    plan_layout.setSpacing(4)

    auto_screw_group = CollapsibleGroupBox("Planning")
    auto_layout = QVBoxLayout()
    auto_layout.setContentsMargins(6, 2, 6, 4)
    auto_layout.setSpacing(3)

    vertebra_header = QHBoxLayout()
    vertebra_header.addWidget(QLabel("Levels to plan (also shown in 3D)"), 1)
    vertebra_select_all_btn = QPushButton("All")
    vertebra_clear_all_btn = QPushButton("Clear")
    vertebra_select_all_btn.setEnabled(False)
    vertebra_clear_all_btn.setEnabled(False)
    vertebra_header.addWidget(vertebra_select_all_btn)
    vertebra_header.addWidget(vertebra_clear_all_btn)
    auto_layout.addLayout(vertebra_header)
    auto_layout.addWidget(vertebra_level_container)

    workspace_mode_row = QHBoxLayout()
    workspace_mode_row.addWidget(QLabel("Mode:"))
    workspace_mode_combo = QComboBox()
    workspace_mode_combo.setObjectName("workspaceMode")
    workspace_mode_combo.addItem("Planning", "planning")
    workspace_mode_combo.addItem("Guided (Coming Soon)", "guided")
    guided_index = workspace_mode_combo.findData("guided")
    guided_item = workspace_mode_combo.model().item(guided_index)
    if guided_item is not None:
        guided_item.setEnabled(False)
    workspace_mode_combo.setToolTip(
        "Guided Workflow is reserved for a later release"
    )
    workspace_mode_combo.setMaximumWidth(100)
    workspace_mode_combo.setMaximumHeight(26)
    workspace_mode_row.addWidget(workspace_mode_combo, 1)
    auto_layout.addLayout(workspace_mode_row)

    auto_screw_review_notice = QLabel(
        "Generated screws are added immediately. Select and adjust them below."
    )
    auto_screw_review_notice.setWordWrap(True)
    auto_layout.addWidget(auto_screw_review_notice)

    # Plan button
    auto_screw_plan_btn = QPushButton("Plan Screws")
    auto_screw_plan_btn.setEnabled(False)
    auto_screw_plan_btn.setProperty("role", "primary")
    auto_layout.addWidget(auto_screw_plan_btn)

    # Status
    auto_screw_status = QLabel("No auto plan")
    auto_screw_status.setWordWrap(True)
    auto_layout.addWidget(auto_screw_status)

    _btn_clear_screws = QPushButton("Clear All Screws")
    _btn_clear_screws.setProperty("role", "danger")
    auto_layout.addWidget(_btn_clear_screws)

    auto_screw_group.set_content_layout(auto_layout)
    plan_layout.addWidget(auto_screw_group)

    # ── Planning parameters (persisted between sessions) ──
    plan_params_group = CollapsibleGroupBox("Planning parameters")
    params_layout = QGridLayout()
    params_layout.setContentsMargins(6, 2, 6, 4)
    params_layout.setHorizontalSpacing(8)
    params_layout.setVerticalSpacing(4)

    planner_defaults = PlannerConfig()

    plan_mode_combo = QComboBox()
    plan_mode_combo.addItem("Optimizer", "optimizer")
    plan_mode_combo.addItem("Legacy", "legacy")
    plan_mode_combo.setToolTip(
        "Optimizer ranks a dense candidate grid; Legacy uses the original "
        "greedy entry/target search"
    )
    params_layout.addWidget(QLabel("Planner"), 0, 0)
    params_layout.addWidget(plan_mode_combo, 0, 1, 1, 2)

    plan_trajectory_combo = QComboBox()
    plan_trajectory_combo.addItem("Traditional", "traditional")
    plan_trajectory_combo.addItem("Cortical bone trajectory", "cbt")
    plan_trajectory_combo.setToolTip(
        "Trajectory family the planner aims for"
    )
    params_layout.addWidget(QLabel("Trajectory"), 1, 0)
    params_layout.addWidget(plan_trajectory_combo, 1, 1, 1, 2)

    plan_fill_ratio_spin = QDoubleSpinBox()
    plan_fill_ratio_spin.setRange(0.50, 1.00)
    plan_fill_ratio_spin.setSingleStep(0.05)
    plan_fill_ratio_spin.setDecimals(2)
    plan_fill_ratio_spin.setValue(planner_defaults.pedicle_fill_ratio)
    plan_fill_ratio_spin.setToolTip(
        "Screw diameter as a fraction of the narrowest pedicle width"
    )
    params_layout.addWidget(QLabel("Pedicle fill"), 2, 0)
    params_layout.addWidget(plan_fill_ratio_spin, 2, 1, 1, 2)

    plan_wall_clearance_spin = QDoubleSpinBox()
    plan_wall_clearance_spin.setRange(0.0, 3.0)
    plan_wall_clearance_spin.setSingleStep(0.1)
    plan_wall_clearance_spin.setDecimals(1)
    plan_wall_clearance_spin.setSuffix(" mm")
    plan_wall_clearance_spin.setValue(planner_defaults.wall_clearance_mm)
    plan_wall_clearance_spin.setToolTip(
        "Minimum distance kept between the screw and the cortical wall"
    )
    params_layout.addWidget(QLabel("Wall clearance"), 3, 0)
    params_layout.addWidget(plan_wall_clearance_spin, 3, 1, 1, 2)

    plan_anterior_margin_spin = QDoubleSpinBox()
    plan_anterior_margin_spin.setRange(0.0, 15.0)
    plan_anterior_margin_spin.setSingleStep(0.5)
    plan_anterior_margin_spin.setDecimals(1)
    plan_anterior_margin_spin.setSuffix(" mm")
    plan_anterior_margin_spin.setValue(
        planner_defaults.anterior_margin_mm
    )
    plan_anterior_margin_spin.setToolTip(
        "Safety margin kept behind the anterior vertebral body cortex"
    )
    params_layout.addWidget(QLabel("Anterior margin"), 4, 0)
    params_layout.addWidget(plan_anterior_margin_spin, 4, 1, 1, 2)

    plan_max_convergence_spin = QDoubleSpinBox()
    plan_max_convergence_spin.setRange(5.0, 60.0)
    plan_max_convergence_spin.setSingleStep(1.0)
    plan_max_convergence_spin.setDecimals(1)
    plan_max_convergence_spin.setSuffix(" °")
    plan_max_convergence_spin.setValue(
        planner_defaults.max_convergence_deg
    )
    plan_max_convergence_spin.setToolTip(
        "Largest medial convergence angle the planner may use"
    )
    params_layout.addWidget(QLabel("Max convergence"), 5, 0)
    params_layout.addWidget(plan_max_convergence_spin, 5, 1, 1, 2)

    plan_hu_threshold_spin = QDoubleSpinBox()
    plan_hu_threshold_spin.setRange(50.0, 300.0)
    plan_hu_threshold_spin.setSingleStep(5.0)
    plan_hu_threshold_spin.setDecimals(0)
    plan_hu_threshold_spin.setSuffix(" HU")
    plan_hu_threshold_spin.setValue(
        planner_defaults.trajectory_hu_threshold
    )
    plan_hu_threshold_spin.setToolTip(
        "Trajectory HU below which loosening risk is flagged"
    )
    params_layout.addWidget(QLabel("HU threshold"), 6, 0)
    params_layout.addWidget(plan_hu_threshold_spin, 6, 1, 1, 2)

    plan_narrow_pedicle_spin = QDoubleSpinBox()
    plan_narrow_pedicle_spin.setRange(3.0, 8.0)
    plan_narrow_pedicle_spin.setSingleStep(0.5)
    plan_narrow_pedicle_spin.setDecimals(1)
    plan_narrow_pedicle_spin.setSuffix(" mm")
    plan_narrow_pedicle_spin.setValue(planner_defaults.narrow_pedicle_mm)
    plan_narrow_pedicle_spin.setToolTip(
        "Pedicle width below which the smallest screw is planned and the "
        "level is marked narrow"
    )
    params_layout.addWidget(QLabel("Narrow pedicle"), 7, 0)
    params_layout.addWidget(plan_narrow_pedicle_spin, 7, 1, 1, 2)

    plan_narrow_lateral_spin = QDoubleSpinBox()
    plan_narrow_lateral_spin.setRange(0.0, 6.0)
    plan_narrow_lateral_spin.setSingleStep(0.5)
    plan_narrow_lateral_spin.setDecimals(1)
    plan_narrow_lateral_spin.setSuffix(" mm")
    plan_narrow_lateral_spin.setValue(
        planner_defaults.narrow_lateral_breach_mm
    )
    plan_narrow_lateral_spin.setToolTip(
        "Lateral (in-out-in) breach a narrow pedicle may accept; the medial "
        "wall is never breached"
    )
    params_layout.addWidget(QLabel("Lateral breach cap"), 8, 0)
    params_layout.addWidget(plan_narrow_lateral_spin, 8, 1, 1, 2)

    weight_rows = (
        ("Safety weight", "safety", "Importance of cortical wall clearance"),
        ("Density weight", "density",
         "Importance of dense bone along the trajectory"),
        ("Construct alignment", "rod",
         "Importance of lining the screw heads up for the rod and of "
         "agreeing on convergence across levels"),
    )
    _planner_weight_value_labels = {}
    _planner_weight_captions = {}
    weight_sliders = {}
    for row, (caption, name, tip) in enumerate(weight_rows, start=9):
        attribute = f"plan_weight_{name}"
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(0, 300)
        slider.setSingleStep(5)
        slider.setPageStep(25)
        # Percent of the nominal weight, so the catalogue default is 100 %.
        slider.setValue(
            int(round(getattr(planner_defaults.weights, name) * 100.0))
        )
        slider.setToolTip(f"{tip} (percent of the nominal weight)")
        weight_sliders[attribute] = slider
        value_label = QLabel()
        value_label.setMinimumWidth(34)
        value_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        _planner_weight_value_labels[attribute] = value_label
        caption_label = QLabel(caption)
        _planner_weight_captions[attribute] = caption_label
        params_layout.addWidget(caption_label, row, 0)
        params_layout.addWidget(slider, row, 1)
        params_layout.addWidget(value_label, row, 2)

    # Rows are taken from the grid rather than hard-coded, so this block
    # keeps working whatever rows the parameter list above grows.
    endplate_row = params_layout.rowCount()
    plan_endplate_parallel_check = QCheckBox("Parallel to upper endplate")
    plan_endplate_parallel_check.setChecked(planner_defaults.endplate_parallel)
    plan_endplate_parallel_check.setToolTip(
        "Aim the trajectory along the upper endplate instead of horizontally"
    )
    params_layout.addWidget(plan_endplate_parallel_check, endplate_row, 0, 1, 3)

    plan_endplate_tolerance_spin = QDoubleSpinBox()
    plan_endplate_tolerance_spin.setRange(0.0, 30.0)
    plan_endplate_tolerance_spin.setSingleStep(1.0)
    plan_endplate_tolerance_spin.setDecimals(1)
    plan_endplate_tolerance_spin.setSuffix(" °")
    plan_endplate_tolerance_spin.setValue(planner_defaults.endplate_tolerance_deg)
    plan_endplate_tolerance_spin.setToolTip(
        "How far from the endplate direction the optimizer may angle the screw"
    )
    params_layout.addWidget(QLabel("Endplate band"), endplate_row + 1, 0)
    params_layout.addWidget(plan_endplate_tolerance_spin, endplate_row + 1, 1, 1, 2)

    plan_reset_defaults_btn = QPushButton("Reset Defaults")
    params_layout.addWidget(plan_reset_defaults_btn, params_layout.rowCount(), 0, 1, 3)

    plan_params_group.set_content_layout(params_layout)
    plan_layout.addWidget(plan_params_group)

    # ── Manual Screw Defaults (collapsed) ──
    screw_defaults_group = CollapsibleGroupBox("Manual Screw Defaults")
    screw_layout = QGridLayout()

    screw_layout.addWidget(QLabel("Length (mm):"), 0, 0)
    length_spin = QSpinBox()
    length_spin.setRange(20, 70)
    length_spin.setValue(int(DEFAULT_SCREW_LENGTH))
    screw_layout.addWidget(length_spin, 0, 1)

    screw_layout.addWidget(QLabel("Diameter (mm):"), 1, 0)
    diameter_spin = DiameterSpinBox()
    diameter_spin.setRange(MIN_SCREW_DIAMETER, MAX_SCREW_DIAMETER)
    diameter_spin.setDecimals(1)
    diameter_spin.setSingleStep(0.5)
    diameter_spin.setValue(DEFAULT_SCREW_DIAMETER)
    diameter_spin.setSuffix(" mm")
    screw_layout.addWidget(diameter_spin, 1, 1)

    screw_defaults_group.set_content_layout(screw_layout)
    plan_layout.addWidget(screw_defaults_group)
    plan_layout.addStretch()

    refs = {
        "planning_group": auto_screw_group,
        "auto_screw_group": auto_screw_group,
        "vertebra_select_all_btn": vertebra_select_all_btn,
        "vertebra_clear_all_btn": vertebra_clear_all_btn,
        "workspace_mode_combo": workspace_mode_combo,
        "auto_screw_review_notice": auto_screw_review_notice,
        "auto_screw_plan_btn": auto_screw_plan_btn,
        "auto_screw_status": auto_screw_status,
        "_btn_clear_screws": _btn_clear_screws,
        "planning_params_group": plan_params_group,
        "plan_params_group": plan_params_group,
        "plan_mode_combo": plan_mode_combo,
        "plan_trajectory_combo": plan_trajectory_combo,
        "plan_fill_ratio_spin": plan_fill_ratio_spin,
        "plan_wall_clearance_spin": plan_wall_clearance_spin,
        "plan_anterior_margin_spin": plan_anterior_margin_spin,
        "plan_max_convergence_spin": plan_max_convergence_spin,
        "plan_hu_threshold_spin": plan_hu_threshold_spin,
        "plan_narrow_pedicle_spin": plan_narrow_pedicle_spin,
        "plan_narrow_lateral_spin": plan_narrow_lateral_spin,
        "_planner_weight_value_labels": _planner_weight_value_labels,
        "_planner_weight_captions": _planner_weight_captions,
        "plan_endplate_parallel_check": plan_endplate_parallel_check,
        "plan_endplate_tolerance_spin": plan_endplate_tolerance_spin,
        "plan_reset_defaults_btn": plan_reset_defaults_btn,
        "screw_parameters_group": screw_defaults_group,
        "screw_defaults_group": screw_defaults_group,
        "length_spin": length_spin,
        "diameter_spin": diameter_spin,
    }
    refs.update(weight_sliders)
    # Refresh the weight value labels without a window instance.
    for attribute, slider in weight_sliders.items():
        _planner_weight_value_labels[attribute].setText(
            f"{slider.value() / 100.0:.2f}"
        )
    return plan_page, refs


def build_review_page(theme_name: str) -> tuple[QWidget, dict]:
    """Build the Review step page; returns (page, refs keyed by attribute name)."""
    # A plain widget, not a collapsible group: this step is centred on
    # the screw list, which needs all the vertical room it can get.
    review_page = QWidget()
    review_page.setProperty("role", "review")
    review_layout = QVBoxLayout(review_page)
    review_layout.setContentsMargins(4, 4, 4, 4)
    review_layout.setSpacing(5)

    header_row = QHBoxLayout()
    review_title_label = QLabel("Review")
    review_title_label.setObjectName("stepTitle")
    header_row.addWidget(review_title_label)
    selected_screw_counter = QLabel("No screws")
    selected_screw_counter.setObjectName("screwReviewCounter")
    selected_screw_counter.setAlignment(
        Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
    )
    header_row.addWidget(selected_screw_counter, 1)
    review_layout.addLayout(header_row)

    # Key numbers: only what a surgeon glances at while flipping between
    # screws -- level/side, diameter, length, grade, and a one-line
    # warning count that expands into the full text on demand.
    review_key_numbers = QWidget()
    review_key_numbers.setObjectName("reviewKeyNumbers")
    key_layout = QVBoxLayout(review_key_numbers)
    key_layout.setContentsMargins(8, 6, 8, 6)
    key_layout.setSpacing(4)

    title_row = QHBoxLayout()
    selected_screw_title = QLabel("No screw selected")
    selected_screw_title.setObjectName("selectedScrewTitle")
    # Word-wrap rather than growing the label's minimum width to fit the
    # full placeholder text on one line: an unwrapped QLabel's minimum
    # size hint equals its full text width, and "No screw selected" (the
    # widest text it ever holds) alone pushed the whole Review page's
    # minimum width well past the panel's real launch width.
    selected_screw_title.setWordWrap(True)
    title_row.addWidget(selected_screw_title, 1)
    selected_screw_grade = QLabel("Grade --")
    selected_screw_grade.setObjectName("screwGradeChip")
    selected_screw_grade.setProperty("grade", "NA")
    selected_screw_grade.setAlignment(Qt.AlignmentFlag.AlignCenter)
    title_row.addWidget(selected_screw_grade)
    key_layout.addLayout(title_row)

    values_row = QHBoxLayout()
    values_row.addWidget(QLabel("Ø"))
    selected_screw_diameter = DiameterSpinBox()
    selected_screw_diameter.setRange(
        MIN_SCREW_DIAMETER,
        MAX_SCREW_DIAMETER,
    )
    selected_screw_diameter.setDecimals(1)
    selected_screw_diameter.setSingleStep(0.5)
    selected_screw_diameter.setSuffix(" mm")
    selected_screw_diameter.setValue(DEFAULT_SCREW_DIAMETER)
    selected_screw_diameter.setEnabled(False)
    selected_screw_diameter.setObjectName("reviewKeyValue")
    selected_screw_diameter.setToolTip(
        "Adjust the selected screw diameter (4.0–7.5 mm)"
    )
    values_row.addWidget(selected_screw_diameter)
    values_row.addWidget(QLabel("Length"))
    selected_screw_length = QLabel("--")
    selected_screw_length.setObjectName("reviewKeyValue")
    values_row.addWidget(selected_screw_length)
    values_row.addStretch(1)
    key_layout.addLayout(values_row)

    screw_warnings_summary = WarningSummary()
    screw_warnings_toggle = screw_warnings_summary.toggle
    selected_screw_warning = screw_warnings_summary.label
    key_layout.addWidget(screw_warnings_summary)

    review_layout.addWidget(review_key_numbers)

    screw_list_widget = ScrewPlanTable()
    screw_list_widget.apply_theme(theme_name)
    review_layout.addWidget(screw_list_widget, 1)

    screw_filter_row = QHBoxLayout()
    screw_filter_row.addWidget(QLabel("Filter:"))
    screw_filter_edit = QLineEdit()
    screw_filter_edit.setPlaceholderText("level, side, or grade — e.g. L4 left B")
    screw_filter_edit.setClearButtonEnabled(True)
    screw_filter_edit.setToolTip(
        "Show only rows matching every word (level, side, grade, warnings)"
    )
    screw_filter_row.addWidget(screw_filter_edit, 1)
    review_layout.addLayout(screw_filter_row)

    # Actions sit right under the list -- exactly what a reviewer does
    # next with the selected row.
    # A two-row grid rather than one wide QHBoxLayout: "Screw MPR" and
    # "Delete Screw" together are wider than the panel's floor width
    # (390 px), and a QHBoxLayout has no way to wrap -- it would just
    # force the whole Review page wider than the space the splitter
    # actually gives it.
    action_row = QGridLayout()
    action_row.setSpacing(3)
    action_row.setContentsMargins(0, 0, 0, 0)
    screw_previous_btn = QPushButton("‹")
    screw_previous_btn.setObjectName("screwReviewNav")
    screw_previous_btn.setToolTip("Previous screw")
    screw_previous_btn.setEnabled(False)
    # Fixed rather than the Qt default min-width (~80 px): four columns
    # at the default would push the Review page's minimum width past the
    # 400 px (390 px minimum) the splitter actually gives the panel,
    # clipping the '›' button and the table's right edge with no
    # horizontal scrollbar to reach them.
    screw_previous_btn.setFixedWidth(32)
    action_row.addWidget(screw_previous_btn, 0, 0)

    screw_axis_mpr_btn = QPushButton("Screw MPR")
    screw_axis_mpr_btn.setEnabled(False)
    screw_axis_mpr_btn.setProperty("role", "primary")
    standard_mpr_btn = QPushButton("Std MPR")
    standard_mpr_btn.setEnabled(False)
    standard_mpr_btn.setProperty("role", "secondary")
    standard_mpr_btn.hide()
    action_row.addWidget(screw_axis_mpr_btn, 0, 1)
    action_row.addWidget(standard_mpr_btn, 0, 1)

    screw_edit_btn = QToolButton()
    screw_edit_btn.setText("Edit")
    screw_edit_btn.setPopupMode(
        QToolButton.ToolButtonPopupMode.InstantPopup
    )
    screw_edit_btn.setEnabled(False)
    screw_edit_btn.setToolTip(
        "Move entry, tip, or the whole screw. "
        "Double-click again to finish."
    )
    screw_edit_menu = QMenu(screw_edit_btn)
    _screw_edit_move_entry_action = screw_edit_menu.addAction(
        "Move entry point"
    )
    _screw_edit_move_tip_action = screw_edit_menu.addAction(
        "Move tip point"
    )
    _screw_edit_move_whole_action = screw_edit_menu.addAction(
        "Move whole screw"
    )
    screw_edit_menu.addSeparator()
    _screw_edit_cancel_action = screw_edit_menu.addAction("Cancel edit")
    screw_edit_btn.setMenu(screw_edit_menu)
    action_row.addWidget(screw_edit_btn, 0, 2)

    screw_next_btn = QPushButton("›")
    screw_next_btn.setObjectName("screwReviewNav")
    screw_next_btn.setToolTip("Next screw")
    screw_next_btn.setEnabled(False)
    screw_next_btn.setFixedWidth(32)
    action_row.addWidget(screw_next_btn, 0, 3)

    remove_screw_btn = QPushButton("Delete Screw")
    remove_screw_btn.setProperty("role", "danger")
    action_row.addWidget(remove_screw_btn, 1, 0, 1, 4)
    review_layout.addLayout(action_row)

    # Hidden legacy edit buttons: ScrewEditController.refresh_controls
    # still reads and writes their enabled/checked state directly, so
    # they stay alive (never shown) rather than being renamed away.
    screw_edit_entry_btn = QPushButton("Entry")
    screw_edit_tip_btn = QPushButton("Tip")
    screw_edit_move_btn = QPushButton("Move")
    screw_edit_cancel_btn = QPushButton("Cancel")
    for button in (
        screw_edit_entry_btn,
        screw_edit_tip_btn,
        screw_edit_move_btn,
    ):
        button.setCheckable(True)
        button.setEnabled(False)
        review_layout.addWidget(button)
    screw_edit_cancel_btn.setEnabled(False)
    review_layout.addWidget(screw_edit_cancel_btn)
    for button in (
        screw_edit_entry_btn,
        screw_edit_tip_btn,
        screw_edit_move_btn,
        screw_edit_cancel_btn,
    ):
        button.hide()

    # Screw MPR's own position/rotation controls: only meaningful while
    # it is active, so refresh_mode_indicators toggles this whole row.
    screw_mpr_controls = QWidget()
    mpr_controls_layout = QVBoxLayout(screw_mpr_controls)
    mpr_controls_layout.setContentsMargins(0, 0, 0, 0)
    mpr_controls_layout.setSpacing(3)

    screw_position_layout = QHBoxLayout()
    screw_axis_position_label = QLabel("Position: 50%")
    screw_position_layout.addWidget(screw_axis_position_label)
    screw_axis_position_slider = QSlider(Qt.Orientation.Horizontal)
    screw_axis_position_slider.setRange(0, 100)
    screw_axis_position_slider.setValue(50)
    screw_axis_position_slider.setEnabled(False)
    screw_axis_position_slider.setToolTip(
        "Move the perpendicular cut from screw entry to target"
    )
    screw_position_layout.addWidget(screw_axis_position_slider, 1)
    mpr_controls_layout.addLayout(screw_position_layout)

    screw_rotation_layout = QHBoxLayout()
    screw_rotation_layout.addWidget(QLabel("Rotation:"))
    screw_axis_rotation_spin = QSpinBox()
    screw_axis_rotation_spin.setRange(-180, 180)
    screw_axis_rotation_spin.setSingleStep(5)
    screw_axis_rotation_spin.setValue(0)
    screw_axis_rotation_spin.setSuffix(" °")
    screw_axis_rotation_spin.setEnabled(False)
    screw_axis_rotation_spin.setToolTip(
        "Spin both long-axis cuts about the screw. Positive follows the "
        "right-hand rule about the entry-to-target axis."
    )
    screw_rotation_layout.addWidget(screw_axis_rotation_spin, 1)
    screw_mpr_reset_btn = QPushButton("Reset view")
    screw_mpr_reset_btn.setEnabled(False)
    screw_mpr_reset_btn.setProperty("role", "secondary")
    screw_mpr_reset_btn.setToolTip(
        "Reset Screw MPR position, rotation, and plane offsets"
    )
    screw_rotation_layout.addWidget(screw_mpr_reset_btn)
    mpr_controls_layout.addLayout(screw_rotation_layout)
    screw_mpr_controls.setVisible(False)
    review_layout.addWidget(screw_mpr_controls)

    details_group = CollapsibleGroupBox("Details")
    details_layout = QVBoxLayout()
    details_layout.setContentsMargins(6, 2, 6, 4)
    details_layout.setSpacing(4)

    selected_screw_metrics = QWidget()
    selected_screw_metrics.setObjectName("screwMetrics")
    selected_screw_details = QGridLayout()
    selected_screw_details.setContentsMargins(4, 2, 4, 2)
    selected_screw_details.setHorizontalSpacing(8)
    selected_screw_details.setVerticalSpacing(5)
    # One label/value pair per row, both spanning the same two columns:
    # an earlier layout put Craniocaudal in its own third column, which
    # forced that column (and so the whole grid) to fit "Craniocaudal"
    # on top of every other row's width -- wide enough to push the
    # Review page past the panel's available width at a typical 1600 px
    # window. A uniform two-column grid costs a little height (free,
    # since Details starts collapsed) for a much narrower minimum width.
    selected_screw_details.addWidget(QLabel("Convergence"), 0, 0)
    selected_screw_convergence = QLabel("--")
    selected_screw_details.addWidget(selected_screw_convergence, 0, 1)
    selected_screw_details.addWidget(QLabel("Craniocaudal"), 1, 0)
    selected_screw_craniocaudal = QLabel("--")
    selected_screw_details.addWidget(selected_screw_craniocaudal, 1, 1)
    selected_screw_details.addWidget(QLabel("Endplate"), 2, 0)
    selected_screw_endplate = QLabel("--")
    selected_screw_endplate.setToolTip(
        "Screw angle relative to the upper endplate; + is tip-cranial, 0 is parallel"
    )
    selected_screw_details.addWidget(selected_screw_endplate, 2, 1)
    selected_screw_details.addWidget(QLabel("Alignment"), 3, 0)
    selected_screw_alignment = QLabel("--")
    selected_screw_alignment.setToolTip(
        "How this screw sits in the construct: rod-line offset and how far "
        "its convergence differs from its neighbours"
    )
    selected_screw_details.addWidget(selected_screw_alignment, 3, 1)
    selected_screw_details.addWidget(QLabel("Trajectory HU"), 4, 0)
    selected_screw_hu = QLabel("--")
    selected_screw_details.addWidget(selected_screw_hu, 4, 1)
    selected_screw_details.addWidget(QLabel("Source"), 5, 0)
    selected_screw_source = QLabel("--")
    selected_screw_details.addWidget(selected_screw_source, 5, 1)
    selected_screw_details.addWidget(QLabel("Body HU"), 6, 0)
    selected_screw_body_hu = QLabel("--")
    selected_screw_details.addWidget(selected_screw_body_hu, 6, 1)
    selected_screw_details.addWidget(QLabel("Wall margin"), 7, 0)
    selected_screw_wall = QLabel("--")
    selected_screw_details.addWidget(selected_screw_wall, 7, 1)
    selected_screw_details.addWidget(QLabel("Facet"), 8, 0)
    selected_screw_facet = QLabel("--")
    selected_screw_facet.setWordWrap(True)
    selected_screw_details.addWidget(selected_screw_facet, 8, 1)
    selected_screw_details.addWidget(QLabel("Heary"), 9, 0)
    selected_screw_heary = QLabel("--")
    selected_screw_details.addWidget(selected_screw_heary, 9, 1)
    selected_screw_details.addWidget(QLabel("Trajectory"), 10, 0)
    selected_screw_trajectory = QLabel("—")
    selected_screw_details.addWidget(selected_screw_trajectory, 10, 1)
    selected_screw_details.addWidget(QLabel("Pedicle"), 11, 0)
    selected_screw_pedicle = QLabel("--")
    selected_screw_details.addWidget(selected_screw_pedicle, 11, 1)
    selected_screw_metrics.setLayout(selected_screw_details)
    details_layout.addWidget(selected_screw_metrics)

    screw_narrow_legend = QLabel(
        "Red screw = narrow pedicle: smallest implant, medial wall protected."
    )
    screw_narrow_legend.setWordWrap(True)
    screw_narrow_legend.setObjectName("screwNarrowLegend")
    details_layout.addWidget(screw_narrow_legend)

    screw_drag_hint = QLabel(
        "Double-click head, tip, or shaft; move the pointer; double-click again to finish."
    )
    screw_drag_hint.setWordWrap(True)
    screw_drag_hint.setObjectName("screwDragHint")
    details_layout.addWidget(screw_drag_hint)

    details_group.set_content_layout(details_layout)
    review_layout.addWidget(details_group)

    measurements_group = CollapsibleGroupBox("Measurements")
    measurements_layout = QVBoxLayout()
    measurements_layout.setContentsMargins(6, 2, 6, 4)
    measurements_layout.setSpacing(4)

    measure_mode_row = QHBoxLayout()
    measure_mode_row.addWidget(QLabel("Mode:"))
    measure_mode_combo = QComboBox()
    measure_mode_combo.addItem("Distance", "distance")
    measure_mode_combo.addItem("Path", "path")
    measure_mode_combo.addItem("Angle", "angle")
    measure_mode_row.addWidget(measure_mode_combo, 1)
    measurements_layout.addLayout(measure_mode_row)

    measure_finish_btn = QPushButton("Finish Path")
    measure_finish_btn.setEnabled(False)
    measurements_layout.addWidget(measure_finish_btn)

    measure_clear_btn = QPushButton("Clear Measurements")
    measurements_layout.addWidget(measure_clear_btn)

    measurement_list_widget = QListWidget()
    measurements_layout.addWidget(measurement_list_widget)

    measurement_actions = QHBoxLayout()
    show_measurement_btn = QPushButton("Show Cut")
    edit_measurement_btn = QPushButton("Edit")
    remove_measurement_btn = QPushButton("Delete")
    measurement_actions.addWidget(show_measurement_btn)
    measurement_actions.addWidget(edit_measurement_btn)
    measurement_actions.addWidget(remove_measurement_btn)
    measurements_layout.addLayout(measurement_actions)

    measurements_group.set_content_layout(measurements_layout)
    review_layout.addWidget(measurements_group)

    refs = {
        "selected_screw_group": review_page,
        "review_title_label": review_title_label,
        "selected_screw_counter": selected_screw_counter,
        "review_key_numbers": review_key_numbers,
        "selected_screw_title": selected_screw_title,
        "selected_screw_grade": selected_screw_grade,
        "selected_screw_diameter": selected_screw_diameter,
        "selected_screw_length": selected_screw_length,
        "screw_warnings_summary": screw_warnings_summary,
        "screw_warnings_toggle": screw_warnings_toggle,
        "selected_screw_warning": selected_screw_warning,
        "screw_list_widget": screw_list_widget,
        "screw_filter_edit": screw_filter_edit,
        "screw_previous_btn": screw_previous_btn,
        "screw_axis_mpr_btn": screw_axis_mpr_btn,
        "standard_mpr_btn": standard_mpr_btn,
        "screw_edit_btn": screw_edit_btn,
        "_screw_edit_move_entry_action": _screw_edit_move_entry_action,
        "_screw_edit_move_tip_action": _screw_edit_move_tip_action,
        "_screw_edit_move_whole_action": _screw_edit_move_whole_action,
        "_screw_edit_cancel_action": _screw_edit_cancel_action,
        "screw_next_btn": screw_next_btn,
        "remove_screw_btn": remove_screw_btn,
        "screw_edit_entry_btn": screw_edit_entry_btn,
        "screw_edit_tip_btn": screw_edit_tip_btn,
        "screw_edit_move_btn": screw_edit_move_btn,
        "screw_edit_cancel_btn": screw_edit_cancel_btn,
        "screw_mpr_controls": screw_mpr_controls,
        "screw_axis_position_label": screw_axis_position_label,
        "screw_axis_position_slider": screw_axis_position_slider,
        "screw_axis_rotation_spin": screw_axis_rotation_spin,
        "screw_mpr_reset_btn": screw_mpr_reset_btn,
        "details_group": details_group,
        "selected_screw_metrics": selected_screw_metrics,
        "selected_screw_convergence": selected_screw_convergence,
        "selected_screw_craniocaudal": selected_screw_craniocaudal,
        "selected_screw_endplate": selected_screw_endplate,
        "selected_screw_alignment": selected_screw_alignment,
        "selected_screw_hu": selected_screw_hu,
        "selected_screw_source": selected_screw_source,
        "selected_screw_body_hu": selected_screw_body_hu,
        "selected_screw_wall": selected_screw_wall,
        "selected_screw_facet": selected_screw_facet,
        "selected_screw_heary": selected_screw_heary,
        "selected_screw_trajectory": selected_screw_trajectory,
        "selected_screw_pedicle": selected_screw_pedicle,
        "screw_narrow_legend": screw_narrow_legend,
        "screw_drag_hint": screw_drag_hint,
        "measurements_group": measurements_group,
        "measure_mode_combo": measure_mode_combo,
        "measure_finish_btn": measure_finish_btn,
        "measure_clear_btn": measure_clear_btn,
        "measurement_list_widget": measurement_list_widget,
        "show_measurement_btn": show_measurement_btn,
        "edit_measurement_btn": edit_measurement_btn,
        "remove_measurement_btn": remove_measurement_btn,
    }
    return review_page, refs
