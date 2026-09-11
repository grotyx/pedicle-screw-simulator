"""Main application window with VWORKS-style and MPR-focus layouts."""

import logging
import sys
from typing import List, Optional

from PyQt6 import sip
from PyQt6.QtCore import QEvent, QObject, QSettings, Qt, QTimer
from PyQt6.QtGui import QAction, QActionGroup
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QSplitter,
    QStatusBar,
    QTabWidget,
    QToolBar,
    QVBoxLayout,
    QWidget,
    QWidgetAction,
)

from src.controllers.auto_placement_controller import AutoPlacementController
from src.controllers.dicom_controller import DicomController
from src.controllers.plan_controller import PlanController
from src.controllers.screw_edit_controller import ScrewEditController
from src.controllers.screw_mpr_controller import (
    SCREW_MPR_CONTROLS_HELP,
    ScrewMPRController,
)
from src.controllers.segmentation_controller import SegmentationController
from src.controllers.tool_controller import ToolController
from src.controllers.view_controller import ViewController
from src.utils.screw_metrics import is_narrow_pedicle, pedicle_row_text

from .. import (
    __academic_affiliation__,
    __author__,
    __department__,
    __email__,
    __license__,
    __organization__,
    __repository__,
    __title__,
    __version__,
    __website__,
)
from ..core.planner_config import PlannerConfig
from ..core.volume_manager import VolumeManager
from ..utils.constants import (
    DEFAULT_SCREW_DIAMETER,
    DEFAULT_SCREW_LENGTH,
    MAX_SCREW_DIAMETER,
    MIN_SCREW_DIAMETER,
    TRANSFER_FUNCTION_PRESETS,
)
from .collapsible_group import CollapsibleGroupBox
from .mpr_viewer import MPRViewer
from .screw_plan_table import ScrewPlanTable
from .spin_boxes import DiameterSpinBox
from .styles import (
    DEFAULT_THEME,
    THEME_LABELS,
    THEMES,
    load_stylesheet,
)
from .tool_icons import create_tool_icon
from .viewer_3d import Viewer3D
from .workflow_bar import WorkflowBar, WorkflowStep, workflow_states

logger = logging.getLogger(__name__)

#: The unified QSettings scope for the whole application. Older builds split
#: settings across two organization/application pairs; new code must use
#: this scope exclusively (see :func:`app_settings`).
_LEGACY_SETTINGS_SCOPE = ("ScrewFixation", "PedicleScrewPlanner")
_APP_SETTINGS_SCOPE = ("SNUBH", "PedicleScrewSimulator")

#: Keys copied out of the legacy scope on first use of the unified scope.
_MIGRATED_SETTINGS_KEYS = ("appearance/theme", "geometry", "windowState")

#: Shown once when a planner mode persisted as Legacy before construct
#: harmonisation existed is migrated to Optimizer.
PLANNER_MODE_MIGRATION_MESSAGE = (
    "Planner mode set to Optimizer (construct alignment needs it); "
    "change it in Planning parameters."
)

#: Settings flag recording that the one-time Legacy -> Optimizer migration ran.
_PLANNER_MODE_MIGRATION_KEY = "mode_migrated_v2"

#: The wall clearance older builds shipped as the planner default.  W7 moved
#: it to 0 mm because the optimiser could not satisfy ``min_wall >= 1.0`` on
#: most sides of a real study and fell back to the legacy planner; a store
#: written by one of those builds still carries it, so it has to be retired
#: rather than simply out-defaulted.
_LEGACY_WALL_CLEARANCE_MM = 1.0

#: Shown once when that persisted 1.0 mm clearance is reset to the new default.
PLANNER_CLEARANCE_MIGRATION_MESSAGE = (
    "Wall clearance reset to 0.0 mm (the new default); raise it in "
    "Planning parameters if you want a buffer."
)

#: Settings flag recording that the one-time clearance reset ran.
_PLANNER_CLEARANCE_MIGRATION_KEY = "clearance_migrated_v2"

#: Set once the legacy-scope migration check has run for this process, so
#: app_settings() only ever opens the legacy scope once, not on every call.
_migrated = False


def app_settings() -> QSettings:
    """Return the single QSettings scope every part of the app should use.

    Earlier code split settings between ``ScrewFixation/PedicleScrewPlanner``
    (theme, window geometry) and ``SNUBH/PedicleScrewSimulator`` (planner,
    segmentation), so "reset the app's settings" meant clearing two
    registry/plist locations. This unifies on the latter scope and, the
    first time it is called in this process, migrates ``appearance/theme``
    and the window-geometry keys out of the legacy scope (if present) so
    existing users keep their preferences. That check runs at most once per
    process, guarded by a module-level flag, rather than re-opening the
    legacy scope on every call until the migration happens to succeed.
    """
    global _migrated
    settings = QSettings(*_APP_SETTINGS_SCOPE)
    if not _migrated:
        _migrated = True
        if settings.value("appearance/theme") is None:
            legacy = QSettings(*_LEGACY_SETTINGS_SCOPE)
            if legacy.value("appearance/theme") is not None:
                for key in _MIGRATED_SETTINGS_KEYS:
                    value = legacy.value(key)
                    if value is not None:
                        settings.setValue(key, value)
                settings.sync()
    return settings


def build_about_html() -> str:
    """Return stable creator, version, license, and safety information."""
    return f"""
    <h2>{__title__}</h2>
    <p><b>Version:</b> {__version__}</p>
    <p>
      <b>Created by</b><br>
      {__author__}<br>
      Professor<br>
      {__department__}<br>
      {__organization__}<br>
      {__academic_affiliation__}
    </p>
    <p>
      <b>Email:</b> <a href="mailto:{__email__}">{__email__}</a><br>
      <b>Website:</b> <a href="{__website__}">{__website__}</a><br>
      <b>Source:</b> <a href="{__repository__}">{__repository__}</a>
    </p>
    <p><b>License:</b> {__license__} License</p>
    <hr>
    <p><b>Research and education use only.</b><br>
    This software is not a certified medical device and must not be used as
    the sole basis for diagnosis, surgery, navigation, or patient care.</p>
    """.strip()


class MainWindow(QMainWindow):
    """
    Main application window with quad-view layout.

    Features:
    - DICOM folder loading
    - Synchronized MPR views (axial, sagittal, coronal)
    - 3D bone surface view
    - Screw placement tools
    - Window/Level controls
    """

    def __init__(self):
        super().__init__()

        self.setWindowTitle(f"{__title__} {__version__}")
        self.setMinimumSize(1280, 760)
        self._apply_initial_window_size()
        self._settings = app_settings()
        app = QApplication.instance()
        app_theme = app.property("themeName") if app is not None else None
        stored_theme = self._settings.value(
            "appearance/theme",
            DEFAULT_THEME,
            type=str,
        )
        self._theme_name = (
            str(app_theme)
            if app_theme in THEMES
            else stored_theme if stored_theme in THEMES else DEFAULT_THEME
        )
        self._active_selection_kind: Optional[str] = None
        #: (QAction | QAbstractButton, icon kind) pairs repainted by apply_theme.
        self._themed_icon_targets: list[tuple[object, str]] = []
        #: Which collapsible sections live in which control-panel tab, in order.
        #: Filled in by _create_control_panel from control_tab_sections.
        self.control_section_order: dict[str, list[str]] = {}
        self.control_tab_sections: dict[str, tuple] = {}

        # Core components
        self.volume_manager = VolumeManager()
        self._current_series_id: Optional[str] = None

        # Viewers (initialized in _setup_ui)
        self.axial_viewer: Optional[MPRViewer] = None
        self.sagittal_viewer: Optional[MPRViewer] = None
        self.coronal_viewer: Optional[MPRViewer] = None
        self.viewer_3d: Optional[Viewer3D] = None

        # Create controllers (before UI -- they only need vm + self reference)
        self._dicom_ctrl = DicomController(self.volume_manager, self)
        self._seg_ctrl = SegmentationController(self.volume_manager, self)
        self._plan_ctrl = PlanController(self.volume_manager, self)
        self._tool_ctrl = ToolController(self.volume_manager, self)
        self._view_ctrl = ViewController(self.volume_manager, self)
        self._auto_placement_ctrl = AutoPlacementController(self.volume_manager, self)
        self._screw_mpr_ctrl = ScrewMPRController(self.volume_manager, self)
        self._screw_edit_ctrl = ScrewEditController(self.volume_manager, self)

        # load_planner_settings() runs before the status bar exists, so a
        # migration message waits here until _setup_statusbar can show it.
        self._planner_mode_migration_message: Optional[str] = None
        self._planner_clearance_migration_message: Optional[str] = None

        # Build UI (widgets only, no signal connections to controllers)
        self._setup_ui()
        self.load_planner_settings()
        self.load_segmentation_settings()
        self._setup_menubar()
        self._setup_toolbar()
        self._setup_statusbar()

        # Wire all signals to controllers
        self._connect_signals()

    @staticmethod
    def _calculate_initial_window_size(
        available_width: int,
        available_height: int,
    ) -> tuple[int, int]:
        """Return a screen-aware initial size for the planning workstation."""
        width = min(1680, int(available_width * 0.94))
        height = min(1050, int(available_height))
        return max(1280, width), max(760, height)

    def _apply_initial_window_size(self) -> None:
        """Fit the first window inside the current screen's available area."""
        screen = QApplication.primaryScreen()
        if screen is None:
            self.resize(1440, 900)
            return
        available = screen.availableGeometry()
        self.resize(
            *self._calculate_initial_window_size(
                available.width(),
                available.height(),
            )
        )

    def _setup_ui(self):
        """Setup the main UI layout."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)

        # Splitter: quad-view (left) | controls (right), user-resizable
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left panel: the workflow bar above the reconfigurable MPR/3D view.
        # Each step runs the existing action it names, looked up at click time
        # so the buttons it presses need not exist yet.
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(0)
        self.workflow_bar = WorkflowBar(
            [
                WorkflowStep("Open DICOM", lambda: self._dicom_ctrl.open_folder()),
                WorkflowStep("Segment", lambda: self.seg_run_btn.click()),
                WorkflowStep("Plan Screws", lambda: self.auto_screw_plan_btn.click()),
            ]
        )
        left_layout.addWidget(self.workflow_bar)

        view_container = QWidget()
        left_layout.addWidget(view_container, 1)
        self._view_layout = QGridLayout(view_container)
        self._view_layout.setSpacing(6)
        self._view_layout.setContentsMargins(6, 6, 6, 6)
        self._view_layout_mode = "planning"
        self._maximized_view: Optional[str] = None
        self._restore_layout_mode = "planning"
        self._focused_view_name = "axial"

        # Create viewers
        self.axial_viewer = MPRViewer("axial", self.volume_manager)
        self.sagittal_viewer = MPRViewer("sagittal", self.volume_manager)
        self.coronal_viewer = MPRViewer("coronal", self.volume_manager)
        self.viewer_3d = Viewer3D(self.volume_manager)

        self.set_view_layout("planning")

        splitter.addWidget(left_panel)

        # Right panel: Controls
        control_panel = self._create_control_panel()
        splitter.addWidget(control_panel)

        # Explicit initial sizes: ~75% quad view, ~25% control panel
        control_width = 400
        splitter.setSizes([max(self.width() - control_width, 880), control_width])
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)
        splitter.setCollapsible(0, False)
        splitter.setCollapsible(1, False)

        self.main_splitter = splitter
        main_layout.addWidget(splitter)

    def set_view_layout(self, mode: str) -> None:
        """Switch layouts, or maximise one pane with ``maximize:<name>``.

        A maximise leaves ``_view_layout_mode`` on the base layout, so
        restoring returns to exactly the layout the user was in.
        """
        mode = str(mode)
        maximized: Optional[str] = None
        if mode.startswith("maximize:"):
            maximized = mode.split(":", 1)[1]
            if maximized not in self.MAXIMIZABLE_VIEWS:
                raise ValueError(f"Unknown view layout: {mode}")
        elif mode not in {"planning", "mpr_focus"}:
            raise ValueError(f"Unknown view layout: {mode}")

        while self._view_layout.count():
            self._view_layout.takeAt(0)

        for row in range(3):
            self._view_layout.setRowStretch(row, 0)
        for column in range(2):
            self._view_layout.setColumnStretch(column, 0)

        panes = {
            "axial": self.axial_viewer,
            "sagittal": self.sagittal_viewer,
            "coronal": self.coronal_viewer,
            "3d": self.viewer_3d,
        }

        if maximized is not None:
            if self._maximized_view is None:
                self._restore_layout_mode = self._view_layout_mode
            for name, pane in panes.items():
                pane.setVisible(name == maximized)
            self._view_layout.addWidget(panes[maximized], 0, 0)
            self._view_layout.setRowStretch(0, 1)
            self._view_layout.setColumnStretch(0, 1)
            self._maximized_view = maximized
        else:
            for pane in panes.values():
                pane.setVisible(True)
            self._maximized_view = None
            if mode == "planning":
                self._view_layout.addWidget(self.axial_viewer, 0, 0)
                self._view_layout.addWidget(self.sagittal_viewer, 1, 0)
                self._view_layout.addWidget(self.coronal_viewer, 2, 0)
                self._view_layout.addWidget(self.viewer_3d, 0, 1, 3, 1)
                for row in range(3):
                    self._view_layout.setRowStretch(row, 1)
                self._view_layout.setColumnStretch(0, 3)
                self._view_layout.setColumnStretch(1, 5)
            else:
                self._view_layout.addWidget(self.axial_viewer, 0, 0)
                self._view_layout.addWidget(self.sagittal_viewer, 0, 1)
                self._view_layout.addWidget(self.coronal_viewer, 1, 0)
                self._view_layout.addWidget(self.viewer_3d, 1, 1)
                self._view_layout.setRowStretch(0, 1)
                self._view_layout.setRowStretch(1, 1)
                self._view_layout.setColumnStretch(0, 1)
                self._view_layout.setColumnStretch(1, 1)
            self._view_layout_mode = mode
            self._restore_layout_mode = mode

        self._render_shown_panes(panes, maximized)

        if hasattr(self, "layout_combo"):
            combo_index = self.layout_combo.findData(self._view_layout_mode)
            if combo_index >= 0 and combo_index != self.layout_combo.currentIndex():
                previous = self.layout_combo.blockSignals(True)
                self.layout_combo.setCurrentIndex(combo_index)
                self.layout_combo.blockSignals(previous)
        if hasattr(self, "_planning_layout_action"):
            self._planning_layout_action.setChecked(
                self._view_layout_mode == "planning"
            )
            self._mpr_focus_layout_action.setChecked(
                self._view_layout_mode == "mpr_focus"
            )
        if hasattr(self, "_maximize_view_action"):
            previous = self._maximize_view_action.blockSignals(True)
            self._maximize_view_action.setChecked(self._maximized_view is not None)
            self._maximize_view_action.blockSignals(previous)
        QTimer.singleShot(0, self.fit_mpr_views)

    def _render_shown_panes(self, panes: dict, maximized: Optional[str]) -> None:
        """Repaint every pane the new layout leaves on screen.

        Re-showing a hidden VTK pane does not repaint its surface: the widget
        only renders when its dirty flag is set, and a plain ``setVisible``
        does not set it. Restoring from a maximised view therefore left the
        other panes -- most visibly the 3D view -- showing a stale, wrongly
        scaled copy of the maximised pane until the next interaction.
        """
        for name, pane in panes.items():
            if maximized is not None and name != maximized:
                continue
            for hook_name in ("_request_render", "safe_render"):
                hook = getattr(pane, hook_name, None)
                if callable(hook):
                    hook()
                    break

    def toggle_maximized_view(self, view_name: str) -> None:
        """Maximise one pane, or restore the previous layout if it already is."""
        name = str(view_name)
        if self._maximized_view == name:
            self.set_view_layout(self._restore_layout_mode)
        else:
            self.set_view_layout(f"maximize:{name}")

    def _toggle_maximize_current_view(self) -> None:
        """View-menu / Ctrl+M entry point for single-view maximise."""
        if self._maximized_view is not None:
            self.set_view_layout(self._restore_layout_mode)
        else:
            self.toggle_maximized_view(self._focused_view_name)

    def _remember_focused_view(self, plane: str) -> None:
        """Track the last pane the user interacted with."""
        if plane in self.MAXIMIZABLE_VIEWS:
            self._focused_view_name = str(plane)

    def _install_focus_tracking(self) -> None:
        """Follow the pointer into any pane, whatever the active tool.

        ``crosshair_moved`` is emitted only from ``MPRViewer._on_left_click``,
        past the pan, measure and screw-pick branches, and the 3D pane emits
        nothing comparable at all.  So panning the sagittal view, dragging a
        screw or orbiting in 3D left ``_focused_view_name`` wherever it last
        happened to be -- and since maximising used to pin it to the maximised
        pane, Ctrl+M after a maximise/restore re-maximised that same pane
        instead of the one being worked in.

        The filter goes on the application rather than on each pane: the panes
        own VTK widgets that are created and replaced outside this class, and a
        filter installed per child would miss every one of them made later.
        """
        app = QApplication.instance()
        if app is not None:
            app.installEventFilter(self)

    #: Events that count as "the user is working in this pane".
    _FOCUS_EVENT_TYPES = frozenset(
        {QEvent.Type.MouseButtonPress, QEvent.Type.Wheel}
    )

    def eventFilter(self, obj: QObject, event: QEvent) -> bool:
        """Track the pane in use and the workflow buttons; never consume anything.

        A press or wheel records which pane it landed in (see
        :meth:`_install_focus_tracking`).  An enabled-state change on one of the
        tab buttons the workflow bar presses redraws the bar, which is the one
        signal that covers every place those buttons are switched: a run
        starting or ending, levels being selected, a threshold mask.
        """
        event_type = event.type()
        if event_type in self._FOCUS_EVENT_TYPES:
            name = self._pane_name_for(obj)
            if name is not None:
                self._focused_view_name = name
        elif event_type == QEvent.Type.EnabledChange and obj in (
            self.__dict__.get("seg_run_btn"),
            self.__dict__.get("auto_screw_plan_btn"),
        ):
            self._refresh_workflow_bar()
        return super().eventFilter(obj, event)

    def _refresh_workflow_bar(self) -> None:
        """Redraw the workflow bar from what this session has done so far.

        Reads the state the rest of the window already keeps rather than
        tracking its own: a loaded volume, a TotalSegmentator mask (a
        threshold fallback cannot be planned on, so it does not finish step
        2), auto-planned screws, and whether the two tab buttons the steps
        press are enabled right now.

        This has several entry points -- the app-wide focus-tracking event
        filter, controller callbacks, and table-model signals -- so it must
        be a safe no-op both on a half-built window (called before
        ``workflow_bar`` or the tab buttons exist) and on a half-destroyed
        one (called after they have been ``deleteLater``'d but before Qt has
        finished tearing down this window's C++ side).  ``sip.isdeleted``
        catches the latter case, which a plain ``hasattr``/``is None`` check
        cannot: the Python attribute is still there, but touching the
        underlying C++ object raises.
        """
        bar = self.__dict__.get("workflow_bar")
        seg_run_btn = self.__dict__.get("seg_run_btn")
        auto_screw_plan_btn = self.__dict__.get("auto_screw_plan_btn")
        if (
            bar is None
            or seg_run_btn is None
            or auto_screw_plan_btn is None
            or sip.isdeleted(bar)
            or sip.isdeleted(seg_run_btn)
            or sip.isdeleted(auto_screw_plan_btn)
        ):
            return
        seg = self._seg_ctrl
        has_mask = (
            getattr(seg, "_last_segmentation_mask_path", None) is not None
            and getattr(seg, "_last_segmentation_method", None) == "totalsegmentator"
        )
        has_plan = any(
            getattr(screw, "source", "") == "auto"
            for screw in self._tool_ctrl.screw_tool.get_screws()
        )
        bar.set_states(
            workflow_states(
                has_volume=self.volume_manager.get_sitk_image() is not None,
                segment_available=seg_run_btn.isEnabled(),
                has_mask=has_mask,
                plan_available=auto_screw_plan_btn.isEnabled(),
                has_plan=has_plan,
            )
        )

    def _on_screw_rows_changed(self, *_args) -> None:
        """A screw row appeared or went, so step 3 may have finished or reopened.

        Connected to the screw table's rowsInserted/rowsRemoved/modelReset
        rather than kept as a lambda: a bound method uses this window as
        Qt's receiver, so the connection is torn down together with it.  A
        lambda has no receiver, so it can outlive this window's buttons
        during teardown -- pytest-qt's deleteLater() on a previous test's
        window can run its DeferredDelete inside a *later* test's event
        loop, after seg_run_btn's C++ object is already gone, and a lambda
        calling _refresh_workflow_bar() then raises on the deleted button.
        """
        self._refresh_workflow_bar()

    def _pane_name_for(self, obj: QObject) -> Optional[str]:
        """Which maximisable pane contains *obj*, if any.

        Walks up from the widget the event reached, so a press on a VTK render
        window deep inside a pane still names the pane.  Guarded with
        ``getattr``: the filter is live from construction, and an event can
        arrive before every viewer attribute exists.
        """
        panes = [
            (name, getattr(self, attribute, None))
            for name, attribute in (
                ("axial", "axial_viewer"),
                ("sagittal", "sagittal_viewer"),
                ("coronal", "coronal_viewer"),
                ("3d", "viewer_3d"),
            )
        ]
        node = obj
        while node is not None:
            for name, pane in panes:
                if pane is not None and node is pane:
                    return name
            node = node.parent()
        return None

    def _create_planning_cockpit(self) -> QWidget:
        """Build the pinned screw inspector shown above the control tabs."""
        cockpit = QWidget()
        cockpit.setObjectName("planningCockpit")
        cockpit_layout = QVBoxLayout(cockpit)
        cockpit_layout.setContentsMargins(8, 6, 8, 6)
        cockpit_layout.setSpacing(6)

        screw_review_nav = QHBoxLayout()
        screw_review_nav.setSpacing(5)
        self.screw_previous_btn = QPushButton("‹ Previous")
        self.screw_previous_btn.setObjectName("screwReviewNav")
        self.selected_screw_counter = QLabel("No screws")
        self.selected_screw_counter.setObjectName("screwReviewCounter")
        self.selected_screw_counter.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.screw_next_btn = QPushButton("Next ›")
        self.screw_next_btn.setObjectName("screwReviewNav")
        self.screw_previous_btn.setEnabled(False)
        self.screw_next_btn.setEnabled(False)
        screw_review_nav.addWidget(self.screw_previous_btn)
        screw_review_nav.addWidget(self.selected_screw_counter, 1)
        screw_review_nav.addWidget(self.screw_next_btn)
        cockpit_layout.addLayout(screw_review_nav)

        self.selected_screw_title = QLabel("No screw selected")
        self.selected_screw_title.setObjectName("selectedScrewTitle")
        cockpit_layout.addWidget(self.selected_screw_title)

        self.selected_screw_metrics = QWidget()
        self.selected_screw_metrics.setObjectName("screwMetrics")
        selected_screw_details = QGridLayout()
        selected_screw_details.setContentsMargins(8, 6, 8, 6)
        selected_screw_details.setHorizontalSpacing(8)
        selected_screw_details.setVerticalSpacing(5)
        selected_screw_details.addWidget(QLabel("Diameter"), 0, 0)
        self.selected_screw_diameter = DiameterSpinBox()
        self.selected_screw_diameter.setRange(
            MIN_SCREW_DIAMETER,
            MAX_SCREW_DIAMETER,
        )
        self.selected_screw_diameter.setDecimals(1)
        self.selected_screw_diameter.setSingleStep(0.5)
        self.selected_screw_diameter.setSuffix(" mm")
        self.selected_screw_diameter.setValue(DEFAULT_SCREW_DIAMETER)
        self.selected_screw_diameter.setEnabled(False)
        self.selected_screw_diameter.setToolTip(
            "Adjust the selected screw diameter (4.0–7.5 mm)"
        )
        selected_screw_details.addWidget(self.selected_screw_diameter, 0, 1)
        selected_screw_details.addWidget(QLabel("Length"), 0, 2)
        self.selected_screw_length = QLabel("--")
        selected_screw_details.addWidget(self.selected_screw_length, 0, 3)
        selected_screw_details.addWidget(QLabel("Convergence"), 1, 0)
        self.selected_screw_convergence = QLabel("--")
        selected_screw_details.addWidget(self.selected_screw_convergence, 1, 1)
        selected_screw_details.addWidget(QLabel("Craniocaudal"), 1, 2)
        self.selected_screw_craniocaudal = QLabel("--")
        selected_screw_details.addWidget(self.selected_screw_craniocaudal, 1, 3)
        selected_screw_details.addWidget(QLabel("Endplate"), 2, 0)
        self.selected_screw_endplate = QLabel("--")
        self.selected_screw_endplate.setToolTip(
            "Screw angle relative to the upper endplate; + is tip-cranial, 0 is parallel"
        )
        selected_screw_details.addWidget(self.selected_screw_endplate, 2, 1)
        selected_screw_details.addWidget(QLabel("Alignment"), 3, 0)
        self.selected_screw_alignment = QLabel("--")
        self.selected_screw_alignment.setToolTip(
            "How this screw sits in the construct: rod-line offset and how far "
            "its convergence differs from its neighbours"
        )
        selected_screw_details.addWidget(self.selected_screw_alignment, 3, 1, 1, 3)
        selected_screw_details.addWidget(QLabel("Safety"), 4, 0)
        self.selected_screw_grade = QLabel("Grade --")
        selected_screw_details.addWidget(self.selected_screw_grade, 4, 1, 1, 3)
        selected_screw_details.addWidget(QLabel("Trajectory HU"), 5, 0)
        self.selected_screw_hu = QLabel("--")
        selected_screw_details.addWidget(self.selected_screw_hu, 5, 1, 1, 3)
        selected_screw_details.addWidget(QLabel("Source"), 6, 0)
        self.selected_screw_source = QLabel("--")
        selected_screw_details.addWidget(self.selected_screw_source, 6, 1, 1, 3)
        selected_screw_details.addWidget(QLabel("Body HU"), 7, 0)
        self.selected_screw_body_hu = QLabel("--")
        selected_screw_details.addWidget(self.selected_screw_body_hu, 7, 1, 1, 3)
        selected_screw_details.addWidget(QLabel("Wall margin"), 8, 0)
        self.selected_screw_wall = QLabel("--")
        selected_screw_details.addWidget(self.selected_screw_wall, 8, 1, 1, 3)
        selected_screw_details.addWidget(QLabel("Facet"), 9, 0)
        self.selected_screw_facet = QLabel("--")
        self.selected_screw_facet.setWordWrap(True)
        selected_screw_details.addWidget(self.selected_screw_facet, 9, 1, 1, 3)
        selected_screw_details.addWidget(QLabel("Heary"), 10, 0)
        self.selected_screw_heary = QLabel("--")
        selected_screw_details.addWidget(self.selected_screw_heary, 10, 1, 1, 3)
        selected_screw_details.addWidget(QLabel("Trajectory"), 11, 0)
        self.selected_screw_trajectory = QLabel("—")
        selected_screw_details.addWidget(self.selected_screw_trajectory, 11, 1, 1, 3)
        selected_screw_details.addWidget(QLabel("Pedicle"), 12, 0)
        self.selected_screw_pedicle = QLabel("--")
        selected_screw_details.addWidget(self.selected_screw_pedicle, 12, 1, 1, 3)
        self.selected_screw_metrics.setLayout(selected_screw_details)
        cockpit_layout.addWidget(self.selected_screw_metrics)

        self.selected_screw_warning = QLabel(
            "Select a screw to inspect its trajectory."
        )
        self.selected_screw_warning.setWordWrap(True)
        self.selected_screw_warning.setMinimumHeight(42)
        self.selected_screw_warning.setObjectName("selectedScrewWarning")
        cockpit_layout.addWidget(self.selected_screw_warning)

        self.screw_narrow_legend = QLabel(
            "Red screw = narrow pedicle: smallest implant, medial wall protected."
        )
        self.screw_narrow_legend.setWordWrap(True)
        self.screw_narrow_legend.setObjectName("screwNarrowLegend")
        cockpit_layout.addWidget(self.screw_narrow_legend)

        self.screw_drag_hint = QLabel(
            "Double-click head, tip, or shaft; move the pointer; double-click again to finish."
        )
        self.screw_drag_hint.setWordWrap(True)
        self.screw_drag_hint.setObjectName("screwDragHint")
        cockpit_layout.addWidget(self.screw_drag_hint)

        return cockpit

    def _create_control_panel(self) -> QWidget:
        """Create the right-side control panel.

        The panel is a pinned Planning Cockpit above a three-tab widget;
        each tab holds its sections in its own scroll area, so the cockpit
        stays visible while any tab is scrolled.

        Creates all widgets and stores references. Signal connections
        are handled in _connect_signals().
        """
        # Sections are built into one throwaway column layout first and then
        # distributed into the tabs at the end of this method, so the group
        # construction below stays exactly as it was.
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(2)

        # ── Auto Segmentation (top priority) ──
        seg_group = CollapsibleGroupBox("Segmentation")
        self.segmentation_group = seg_group
        seg_layout = QVBoxLayout()
        seg_layout.setContentsMargins(6, 2, 6, 4)
        seg_layout.setSpacing(3)

        self.seg_run_btn = QPushButton("Run Auto Segmentation")
        seg_layout.addWidget(self.seg_run_btn)

        self.seg_refine_check = QCheckBox("Refine boundaries against CT")
        self.seg_refine_check.setChecked(True)
        self.seg_refine_check.setToolTip(
            "Smooth the segmentation to the CT grid and snap its boundaries to "
            "the bone cortex. Turn off to keep TotalSegmentator's raw 1.5 mm "
            "label map."
        )
        seg_layout.addWidget(self.seg_refine_check)

        self.vertebra_isolate_btn = QPushButton("Isolate Vertebrae")
        self.vertebra_isolate_btn.setEnabled(False)
        seg_layout.addWidget(self.vertebra_isolate_btn)

        vertebra_header = QHBoxLayout()
        vertebra_header.addWidget(QLabel("Visible / Plan Levels"), 1)
        self.vertebra_select_all_btn = QPushButton("All")
        self.vertebra_clear_all_btn = QPushButton("Clear")
        self.vertebra_select_all_btn.setEnabled(False)
        self.vertebra_clear_all_btn.setEnabled(False)
        vertebra_header.addWidget(self.vertebra_select_all_btn)
        vertebra_header.addWidget(self.vertebra_clear_all_btn)
        seg_layout.addLayout(vertebra_header)

        self.vertebra_level_container = QWidget()
        self._vertebra_level_grid = QGridLayout(
            self.vertebra_level_container
        )
        self._vertebra_level_grid.setContentsMargins(0, 0, 0, 0)
        self._vertebra_level_grid.setHorizontalSpacing(8)
        self._vertebra_level_grid.setVerticalSpacing(4)
        self._vertebra_level_checks: dict[int, QCheckBox] = {}
        self._updating_vertebra_level_checks = False
        self._vertebra_level_placeholder = QLabel("Run segmentation first")
        self._vertebra_level_placeholder.setObjectName("segmentationStatus")
        self._vertebra_level_grid.addWidget(
            self._vertebra_level_placeholder,
            0,
            0,
            1,
            4,
        )
        seg_layout.addWidget(self.vertebra_level_container)

        self.seg_status_label = QLabel("Ready for automatic segmentation")
        self.seg_status_label.setWordWrap(True)
        self.seg_status_label.setObjectName("segmentationStatus")
        seg_layout.addWidget(self.seg_status_label)

        self.seg_advanced_panel = QWidget()
        advanced_layout = QGridLayout(self.seg_advanced_panel)
        advanced_layout.setContentsMargins(0, 0, 0, 0)

        advanced_layout.addWidget(QLabel("Task:"), 0, 0)
        self.seg_task_combo = QComboBox()
        self.seg_task_combo.addItem("Spine Only", "spine_only")
        self.seg_task_combo.addItem("Total (117 structures)", "total")
        advanced_layout.addWidget(self.seg_task_combo, 0, 1)

        advanced_layout.addWidget(QLabel("Device:"), 1, 0)
        self.seg_device_combo = QComboBox()
        self.seg_device_combo.addItem("GPU (preferred)", "gpu")
        self.seg_device_combo.addItem("CPU fallback", "cpu")
        advanced_layout.addWidget(self.seg_device_combo, 1, 1)

        advanced_layout.addWidget(QLabel("Quality:"), 2, 0)
        self.seg_quality_combo = QComboBox()
        self.seg_quality_combo.addItem("Accurate (1.5 mm)", "accurate")
        self.seg_quality_combo.addItem("Memory Saver (3 mm)", "fast")
        self.seg_quality_combo.setToolTip(
            "Accurate mode uses more memory. Memory Saver is faster but may "
            "reduce boundary precision."
        )
        advanced_layout.addWidget(self.seg_quality_combo, 2, 1)

        advanced_layout.addWidget(QLabel("Label ID:"), 3, 0)
        self.seg_label_spin = QSpinBox()
        self.seg_label_spin.setRange(0, 1000)
        self.seg_label_spin.setValue(0)
        self.seg_label_spin.setToolTip(
            "0 shows all labels, other values show only that label."
        )
        advanced_layout.addWidget(self.seg_label_spin, 3, 1)

        advanced_layout.addWidget(QLabel("Label Name:"), 4, 0)
        self.seg_label_combo = QComboBox()
        advanced_layout.addWidget(self.seg_label_combo, 4, 1)

        self.seg_label_info = QLabel("Selected: All labels")
        self.seg_label_info.setWordWrap(True)
        advanced_layout.addWidget(self.seg_label_info, 5, 0, 1, 2)

        self.seg_show_2d_check = QCheckBox("Show 2D Overlay")
        self.seg_show_2d_check.setChecked(True)
        advanced_layout.addWidget(self.seg_show_2d_check, 6, 0, 1, 2)

        self.seg_show_3d_check = QCheckBox("Show 3D Overlay")
        self.seg_show_3d_check.setChecked(False)
        advanced_layout.addWidget(self.seg_show_3d_check, 7, 0, 1, 2)

        self.seg_clear_btn = QPushButton("Clear Segmentation Overlay")
        advanced_layout.addWidget(self.seg_clear_btn, 8, 0, 1, 2)

        self.vertebra_restore_btn = QPushButton("Show Full Volume")
        self.vertebra_restore_btn.setEnabled(False)
        self.vertebra_restore_btn.hide()
        advanced_layout.addWidget(self.vertebra_restore_btn, 9, 0, 1, 2)

        advanced_layout.addWidget(QLabel("3D Vertebrae:"), 10, 0, 1, 2)
        self.vertebra_display_list = QListWidget()
        self.vertebra_display_list.setSelectionMode(
            QAbstractItemView.SelectionMode.ExtendedSelection
        )
        self.vertebra_display_list.setMaximumHeight(110)
        self.vertebra_display_list.setToolTip(
            "Select one or more segmented vertebrae to show in 3D"
        )
        advanced_layout.addWidget(self.vertebra_display_list, 11, 0, 1, 2)

        vertebra_display_buttons = QHBoxLayout()
        self.vertebra_show_selected_btn = QPushButton("Show Selected 3D")
        self.vertebra_show_all_btn = QPushButton("Show All Segmented")
        self.vertebra_show_selected_btn.setEnabled(False)
        self.vertebra_show_all_btn.setEnabled(False)
        vertebra_display_buttons.addWidget(self.vertebra_show_selected_btn)
        vertebra_display_buttons.addWidget(self.vertebra_show_all_btn)
        advanced_layout.addLayout(vertebra_display_buttons, 12, 0, 1, 2)

        self.seg_use_subregion_check = QCheckBox("Use pedicle subregion model")
        self.seg_use_subregion_check.setChecked(False)
        self.seg_use_subregion_check.setToolTip(
            "Refine pedicle detection with a locally installed nnU-Net "
            "subregion model. Leave off to use TotalSegmentator alone."
        )
        advanced_layout.addWidget(self.seg_use_subregion_check, 13, 0, 1, 2)

        subregion_dir_row = QHBoxLayout()
        self.seg_subregion_dir_edit = QLineEdit()
        self.seg_subregion_dir_edit.setPlaceholderText(
            "Model directory (dataset.json)"
        )
        subregion_dir_row.addWidget(self.seg_subregion_dir_edit, 1)
        self.seg_subregion_browse_btn = QPushButton("Browse...")
        subregion_dir_row.addWidget(self.seg_subregion_browse_btn)
        advanced_layout.addLayout(subregion_dir_row, 14, 0, 1, 2)

        self.seg_advanced_panel.hide()
        seg_layout.addWidget(self.seg_advanced_panel)

        seg_group.set_content_layout(seg_layout)
        layout.addWidget(seg_group)

        # ── Auto Pedicle Screw (second priority) ──
        auto_screw_group = CollapsibleGroupBox("Planning")
        self.planning_group = auto_screw_group
        auto_layout = QVBoxLayout()
        auto_layout.setContentsMargins(6, 2, 6, 4)
        auto_layout.setSpacing(3)

        workspace_mode_row = QHBoxLayout()
        workspace_mode_row.addWidget(QLabel("Mode:"))
        self.workspace_mode_combo = QComboBox()
        self.workspace_mode_combo.setObjectName("workspaceMode")
        self.workspace_mode_combo.addItem("Planning", "planning")
        self.workspace_mode_combo.addItem("Guided (Coming Soon)", "guided")
        guided_index = self.workspace_mode_combo.findData("guided")
        guided_item = self.workspace_mode_combo.model().item(guided_index)
        if guided_item is not None:
            guided_item.setEnabled(False)
        self.workspace_mode_combo.setToolTip(
            "Guided Workflow is reserved for a later release"
        )
        self.workspace_mode_combo.setMaximumWidth(100)
        self.workspace_mode_combo.setMaximumHeight(26)
        workspace_mode_row.addWidget(self.workspace_mode_combo, 1)
        auto_layout.addLayout(workspace_mode_row)

        self.auto_screw_review_notice = QLabel(
            "Generated screws are added immediately. Select and adjust them below."
        )
        self.auto_screw_review_notice.setWordWrap(True)
        auto_layout.addWidget(self.auto_screw_review_notice)

        # Plan button
        self.auto_screw_plan_btn = QPushButton("Plan Screws")
        self.auto_screw_plan_btn.setEnabled(False)
        auto_layout.addWidget(self.auto_screw_plan_btn)

        # Status
        self.auto_screw_status = QLabel("No auto plan")
        self.auto_screw_status.setWordWrap(True)
        auto_layout.addWidget(self.auto_screw_status)

        auto_screw_group.set_content_layout(auto_layout)
        layout.addWidget(auto_screw_group)

        # ── Planning parameters (persisted between sessions) ──
        plan_params_group = CollapsibleGroupBox("Planning parameters")
        self.planning_params_group = plan_params_group
        params_layout = QGridLayout()
        params_layout.setContentsMargins(6, 2, 6, 4)
        params_layout.setHorizontalSpacing(8)
        params_layout.setVerticalSpacing(4)

        planner_defaults = PlannerConfig()

        self.plan_mode_combo = QComboBox()
        self.plan_mode_combo.addItem("Optimizer", "optimizer")
        self.plan_mode_combo.addItem("Legacy", "legacy")
        self.plan_mode_combo.setToolTip(
            "Optimizer ranks a dense candidate grid; Legacy uses the original "
            "greedy entry/target search"
        )
        params_layout.addWidget(QLabel("Planner"), 0, 0)
        params_layout.addWidget(self.plan_mode_combo, 0, 1, 1, 2)

        self.plan_trajectory_combo = QComboBox()
        self.plan_trajectory_combo.addItem("Traditional", "traditional")
        self.plan_trajectory_combo.addItem("Cortical bone trajectory", "cbt")
        self.plan_trajectory_combo.setToolTip(
            "Trajectory family the planner aims for"
        )
        params_layout.addWidget(QLabel("Trajectory"), 1, 0)
        params_layout.addWidget(self.plan_trajectory_combo, 1, 1, 1, 2)

        self.plan_fill_ratio_spin = QDoubleSpinBox()
        self.plan_fill_ratio_spin.setRange(0.50, 1.00)
        self.plan_fill_ratio_spin.setSingleStep(0.05)
        self.plan_fill_ratio_spin.setDecimals(2)
        self.plan_fill_ratio_spin.setValue(planner_defaults.pedicle_fill_ratio)
        self.plan_fill_ratio_spin.setToolTip(
            "Screw diameter as a fraction of the narrowest pedicle width"
        )
        params_layout.addWidget(QLabel("Pedicle fill"), 2, 0)
        params_layout.addWidget(self.plan_fill_ratio_spin, 2, 1, 1, 2)

        self.plan_wall_clearance_spin = QDoubleSpinBox()
        self.plan_wall_clearance_spin.setRange(0.0, 3.0)
        self.plan_wall_clearance_spin.setSingleStep(0.1)
        self.plan_wall_clearance_spin.setDecimals(1)
        self.plan_wall_clearance_spin.setSuffix(" mm")
        self.plan_wall_clearance_spin.setValue(planner_defaults.wall_clearance_mm)
        self.plan_wall_clearance_spin.setToolTip(
            "Minimum distance kept between the screw and the cortical wall"
        )
        params_layout.addWidget(QLabel("Wall clearance"), 3, 0)
        params_layout.addWidget(self.plan_wall_clearance_spin, 3, 1, 1, 2)

        self.plan_anterior_margin_spin = QDoubleSpinBox()
        self.plan_anterior_margin_spin.setRange(0.0, 15.0)
        self.plan_anterior_margin_spin.setSingleStep(0.5)
        self.plan_anterior_margin_spin.setDecimals(1)
        self.plan_anterior_margin_spin.setSuffix(" mm")
        self.plan_anterior_margin_spin.setValue(
            planner_defaults.anterior_margin_mm
        )
        self.plan_anterior_margin_spin.setToolTip(
            "Safety margin kept behind the anterior vertebral body cortex"
        )
        params_layout.addWidget(QLabel("Anterior margin"), 4, 0)
        params_layout.addWidget(self.plan_anterior_margin_spin, 4, 1, 1, 2)

        self.plan_max_convergence_spin = QDoubleSpinBox()
        self.plan_max_convergence_spin.setRange(5.0, 60.0)
        self.plan_max_convergence_spin.setSingleStep(1.0)
        self.plan_max_convergence_spin.setDecimals(1)
        self.plan_max_convergence_spin.setSuffix(" °")
        self.plan_max_convergence_spin.setValue(
            planner_defaults.max_convergence_deg
        )
        self.plan_max_convergence_spin.setToolTip(
            "Largest medial convergence angle the planner may use"
        )
        params_layout.addWidget(QLabel("Max convergence"), 5, 0)
        params_layout.addWidget(self.plan_max_convergence_spin, 5, 1, 1, 2)

        self.plan_hu_threshold_spin = QDoubleSpinBox()
        self.plan_hu_threshold_spin.setRange(50.0, 300.0)
        self.plan_hu_threshold_spin.setSingleStep(5.0)
        self.plan_hu_threshold_spin.setDecimals(0)
        self.plan_hu_threshold_spin.setSuffix(" HU")
        self.plan_hu_threshold_spin.setValue(
            planner_defaults.trajectory_hu_threshold
        )
        self.plan_hu_threshold_spin.setToolTip(
            "Trajectory HU below which loosening risk is flagged"
        )
        params_layout.addWidget(QLabel("HU threshold"), 6, 0)
        params_layout.addWidget(self.plan_hu_threshold_spin, 6, 1, 1, 2)

        self.plan_narrow_pedicle_spin = QDoubleSpinBox()
        self.plan_narrow_pedicle_spin.setRange(3.0, 8.0)
        self.plan_narrow_pedicle_spin.setSingleStep(0.5)
        self.plan_narrow_pedicle_spin.setDecimals(1)
        self.plan_narrow_pedicle_spin.setSuffix(" mm")
        self.plan_narrow_pedicle_spin.setValue(planner_defaults.narrow_pedicle_mm)
        self.plan_narrow_pedicle_spin.setToolTip(
            "Pedicle width below which the smallest screw is planned and the "
            "level is marked narrow"
        )
        params_layout.addWidget(QLabel("Narrow pedicle"), 7, 0)
        params_layout.addWidget(self.plan_narrow_pedicle_spin, 7, 1, 1, 2)

        self.plan_narrow_lateral_spin = QDoubleSpinBox()
        self.plan_narrow_lateral_spin.setRange(0.0, 6.0)
        self.plan_narrow_lateral_spin.setSingleStep(0.5)
        self.plan_narrow_lateral_spin.setDecimals(1)
        self.plan_narrow_lateral_spin.setSuffix(" mm")
        self.plan_narrow_lateral_spin.setValue(
            planner_defaults.narrow_lateral_breach_mm
        )
        self.plan_narrow_lateral_spin.setToolTip(
            "Lateral (in-out-in) breach a narrow pedicle may accept; the medial "
            "wall is never breached"
        )
        params_layout.addWidget(QLabel("Lateral breach cap"), 8, 0)
        params_layout.addWidget(self.plan_narrow_lateral_spin, 8, 1, 1, 2)

        weight_rows = (
            ("Safety weight", "safety", "Importance of cortical wall clearance"),
            ("Density weight", "density",
             "Importance of dense bone along the trajectory"),
            ("Construct alignment", "rod",
             "Importance of lining the screw heads up for the rod and of "
             "agreeing on convergence across levels"),
        )
        self._planner_weight_value_labels = {}
        self._planner_weight_captions = {}
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
            setattr(self, attribute, slider)
            value_label = QLabel()
            value_label.setMinimumWidth(34)
            value_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            self._planner_weight_value_labels[attribute] = value_label
            caption_label = QLabel(caption)
            self._planner_weight_captions[attribute] = caption_label
            params_layout.addWidget(caption_label, row, 0)
            params_layout.addWidget(slider, row, 1)
            params_layout.addWidget(value_label, row, 2)
        self._refresh_planner_weight_labels()

        # Rows are taken from the grid rather than hard-coded, so this block
        # keeps working whatever rows the parameter list above grows.
        endplate_row = params_layout.rowCount()
        self.plan_endplate_parallel_check = QCheckBox("Parallel to upper endplate")
        self.plan_endplate_parallel_check.setChecked(planner_defaults.endplate_parallel)
        self.plan_endplate_parallel_check.setToolTip(
            "Aim the trajectory along the upper endplate instead of horizontally"
        )
        params_layout.addWidget(self.plan_endplate_parallel_check, endplate_row, 0, 1, 3)

        self.plan_endplate_tolerance_spin = QDoubleSpinBox()
        self.plan_endplate_tolerance_spin.setRange(0.0, 30.0)
        self.plan_endplate_tolerance_spin.setSingleStep(1.0)
        self.plan_endplate_tolerance_spin.setDecimals(1)
        self.plan_endplate_tolerance_spin.setSuffix(" °")
        self.plan_endplate_tolerance_spin.setValue(planner_defaults.endplate_tolerance_deg)
        self.plan_endplate_tolerance_spin.setToolTip(
            "How far from the endplate direction the optimizer may angle the screw"
        )
        params_layout.addWidget(QLabel("Endplate band"), endplate_row + 1, 0)
        params_layout.addWidget(self.plan_endplate_tolerance_spin, endplate_row + 1, 1, 1, 2)

        self.plan_reset_defaults_btn = QPushButton("Reset Defaults")
        params_layout.addWidget(self.plan_reset_defaults_btn, params_layout.rowCount(), 0, 1, 3)

        plan_params_group.set_content_layout(params_layout)
        layout.addWidget(plan_params_group)

        # ── Remaining groups (collapsed by default) ──

        # DICOM info group
        info_group = CollapsibleGroupBox("Study")
        info_layout = QVBoxLayout()
        self.info_label = QLabel("No volume loaded")
        self.info_label.setWordWrap(True)
        info_layout.addWidget(self.info_label)
        info_group.set_content_layout(info_layout)
        layout.addWidget(info_group)

        # Window/Level group
        wl_group = CollapsibleGroupBox("Window/Level")
        wl_layout = QGridLayout()

        wl_layout.addWidget(QLabel("Window:"), 0, 0)
        self.window_slider = QSlider(Qt.Orientation.Horizontal)
        self.window_slider.setRange(1, 4000)
        self.window_slider.setValue(1500)
        wl_layout.addWidget(self.window_slider, 0, 1)

        wl_layout.addWidget(QLabel("Level:"), 1, 0)
        self.level_slider = QSlider(Qt.Orientation.Horizontal)
        self.level_slider.setRange(-1000, 3000)
        self.level_slider.setValue(400)
        wl_layout.addWidget(self.level_slider, 1, 1)

        # Presets
        preset_layout = QHBoxLayout()
        self._btn_bone = QPushButton("Bone")
        self._btn_soft = QPushButton("Soft Tissue")
        preset_layout.addWidget(self._btn_bone)
        preset_layout.addWidget(self._btn_soft)
        wl_layout.addLayout(preset_layout, 2, 0, 1, 2)

        wl_group.set_content_layout(wl_layout)
        layout.addWidget(wl_group)

        # Screw Parameters group
        screw_group = CollapsibleGroupBox("Screw Parameters")
        screw_layout = QGridLayout()

        screw_layout.addWidget(QLabel("Length (mm):"), 0, 0)
        self.length_spin = QSpinBox()
        self.length_spin.setRange(20, 70)
        self.length_spin.setValue(int(DEFAULT_SCREW_LENGTH))
        screw_layout.addWidget(self.length_spin, 0, 1)

        screw_layout.addWidget(QLabel("Diameter (mm):"), 1, 0)
        self.diameter_spin = DiameterSpinBox()
        self.diameter_spin.setRange(MIN_SCREW_DIAMETER, MAX_SCREW_DIAMETER)
        self.diameter_spin.setDecimals(1)
        self.diameter_spin.setSingleStep(0.5)
        self.diameter_spin.setValue(DEFAULT_SCREW_DIAMETER)
        self.diameter_spin.setSuffix(" mm")
        screw_layout.addWidget(self.diameter_spin, 1, 1)

        screw_group.set_content_layout(screw_layout)
        self.screw_parameters_group = screw_group
        layout.addWidget(screw_group)

        # Measurement group
        measure_group = CollapsibleGroupBox("Measurement")
        measure_layout = QGridLayout()
        measure_layout.addWidget(QLabel("Mode:"), 0, 0)
        self.measure_mode_combo = QComboBox()
        self.measure_mode_combo.addItem("Distance", "distance")
        self.measure_mode_combo.addItem("Path", "path")
        self.measure_mode_combo.addItem("Angle", "angle")
        measure_layout.addWidget(self.measure_mode_combo, 0, 1)

        self.measure_finish_btn = QPushButton("Finish Path")
        self.measure_finish_btn.setEnabled(False)
        measure_layout.addWidget(self.measure_finish_btn, 1, 0, 1, 2)

        self.measure_clear_btn = QPushButton("Clear Measurements")
        measure_layout.addWidget(self.measure_clear_btn, 2, 0, 1, 2)

        measure_group.set_content_layout(measure_layout)
        layout.addWidget(measure_group)

        # Measurement list group
        measure_list_group = CollapsibleGroupBox("Measurement List")
        measure_list_layout = QVBoxLayout()
        self.measurement_list_widget = QListWidget()
        measure_list_layout.addWidget(self.measurement_list_widget)

        measurement_actions = QHBoxLayout()
        self.show_measurement_btn = QPushButton("Show Cut")
        self.edit_measurement_btn = QPushButton("Edit")
        self.remove_measurement_btn = QPushButton("Delete")
        measurement_actions.addWidget(self.show_measurement_btn)
        measurement_actions.addWidget(self.edit_measurement_btn)
        measurement_actions.addWidget(self.remove_measurement_btn)
        measure_list_layout.addLayout(measurement_actions)
        measure_list_group.set_content_layout(measure_list_layout)
        layout.addWidget(measure_list_group)

        # Screw review group
        screw_list_group = CollapsibleGroupBox("Screw Review")
        self.selected_screw_group = screw_list_group
        self.selected_screw_group.setProperty("role", "review")
        screw_list_layout = QVBoxLayout()
        screw_list_layout.setContentsMargins(8, 4, 8, 8)
        screw_list_layout.setSpacing(6)

        self.screw_list_widget = ScrewPlanTable()
        self.screw_list_widget.apply_theme(self._theme_name)
        screw_list_layout.addWidget(self.screw_list_widget)

        screw_edit_buttons = QHBoxLayout()
        self.screw_edit_entry_btn = QPushButton("Entry")
        self.screw_edit_tip_btn = QPushButton("Tip")
        self.screw_edit_move_btn = QPushButton("Move")
        self.screw_edit_cancel_btn = QPushButton("Cancel")
        for button in (
            self.screw_edit_entry_btn,
            self.screw_edit_tip_btn,
            self.screw_edit_move_btn,
        ):
            button.setCheckable(True)
            button.setEnabled(False)
            screw_edit_buttons.addWidget(button)
        self.screw_edit_cancel_btn.setEnabled(False)
        screw_edit_buttons.addWidget(self.screw_edit_cancel_btn)
        for button in (
            self.screw_edit_entry_btn,
            self.screw_edit_tip_btn,
            self.screw_edit_move_btn,
            self.screw_edit_cancel_btn,
        ):
            button.hide()

        screw_mpr_buttons = QHBoxLayout()
        self.screw_axis_mpr_btn = QPushButton("Screw MPR")
        self.screw_axis_mpr_btn.setEnabled(False)
        self.standard_mpr_btn = QPushButton("Std MPR")
        self.standard_mpr_btn.setEnabled(False)
        screw_mpr_buttons.addWidget(self.screw_axis_mpr_btn)
        screw_mpr_buttons.addWidget(self.standard_mpr_btn)
        screw_list_layout.addLayout(screw_mpr_buttons)

        screw_position_layout = QHBoxLayout()
        self.screw_axis_position_label = QLabel("Position: 50%")
        screw_position_layout.addWidget(self.screw_axis_position_label)
        self.screw_axis_position_slider = QSlider(Qt.Orientation.Horizontal)
        self.screw_axis_position_slider.setRange(0, 100)
        self.screw_axis_position_slider.setValue(50)
        self.screw_axis_position_slider.setEnabled(False)
        self.screw_axis_position_slider.setToolTip(
            "Move the perpendicular cut from screw entry to target"
        )
        screw_position_layout.addWidget(self.screw_axis_position_slider, 1)
        screw_list_layout.addLayout(screw_position_layout)

        screw_rotation_layout = QHBoxLayout()
        screw_rotation_layout.addWidget(QLabel("Rotation:"))
        self.screw_axis_rotation_spin = QSpinBox()
        self.screw_axis_rotation_spin.setRange(-180, 180)
        self.screw_axis_rotation_spin.setSingleStep(5)
        self.screw_axis_rotation_spin.setValue(0)
        self.screw_axis_rotation_spin.setSuffix(" °")
        self.screw_axis_rotation_spin.setEnabled(False)
        self.screw_axis_rotation_spin.setToolTip(
            "Spin both long-axis cuts about the screw. Positive follows the "
            "right-hand rule about the entry-to-target axis."
        )
        screw_rotation_layout.addWidget(self.screw_axis_rotation_spin, 1)
        self.screw_mpr_reset_btn = QPushButton("Reset view")
        self.screw_mpr_reset_btn.setEnabled(False)
        self.screw_mpr_reset_btn.setToolTip(
            "Reset Screw MPR position, rotation, and plane offsets"
        )
        screw_rotation_layout.addWidget(self.screw_mpr_reset_btn)
        screw_list_layout.addLayout(screw_rotation_layout)

        self.remove_screw_btn = QPushButton("Delete Screw")
        screw_list_layout.addWidget(self.remove_screw_btn)
        screw_list_group.set_content_layout(screw_list_layout)
        layout.addWidget(screw_list_group)

        # 3D View controls
        view_group = CollapsibleGroupBox("Validation")
        vr_layout = QVBoxLayout()

        # Transfer function preset selector
        tf_preset_layout = QHBoxLayout()
        tf_preset_layout.addWidget(QLabel("Preset:"))
        self.tf_preset_combo = QComboBox()
        for name in TRANSFER_FUNCTION_PRESETS:
            self.tf_preset_combo.addItem(name)
        tf_preset_layout.addWidget(self.tf_preset_combo)
        vr_layout.addLayout(tf_preset_layout)

        # Volume opacity slider
        opacity_layout = QHBoxLayout()
        opacity_layout.addWidget(QLabel("Opacity:"))
        self.opacity_slider = QSlider(Qt.Orientation.Horizontal)
        self.opacity_slider.setRange(0, 100)
        self.opacity_slider.setValue(100)
        opacity_layout.addWidget(self.opacity_slider)
        vr_layout.addLayout(opacity_layout)

        view_group.set_content_layout(vr_layout)
        layout.addWidget(view_group)

        self.secondary_control_groups = (
            info_group,
            plan_params_group,
            wl_group,
            screw_group,
            measure_group,
            measure_list_group,
            view_group,
        )

        # Clear screws button
        self._btn_clear_screws = QPushButton("Clear All Screws")
        layout.addWidget(self._btn_clear_screws)

        for button in (
            self.seg_run_btn,
            self.auto_screw_plan_btn,
            self.screw_axis_mpr_btn,
        ):
            button.setProperty("role", "primary")
        self._register_themed_icon(self.seg_run_btn, "run")
        self._register_themed_icon(self.auto_screw_plan_btn, "run")
        for button in (
            self.standard_mpr_btn,
            self.screw_mpr_reset_btn,
        ):
            button.setProperty("role", "secondary")
        for button in (
            self.remove_screw_btn,
            self._btn_clear_screws,
            self.seg_clear_btn,
        ):
            button.setProperty("role", "danger")

        # ── Assemble: pinned cockpit + three tabs of collapsible sections ──
        self.planning_cockpit = self._create_planning_cockpit()

        self.control_tab_sections = {
            "Study": (info_group, seg_group, wl_group),
            "Planning": (
                auto_screw_group,
                plan_params_group,
                screw_group,
                screw_list_group,
            ),
            "Tools": (measure_group, measure_list_group, view_group),
        }
        self.control_section_order = {
            tab: [group.title for group in groups]
            for tab, groups in self.control_tab_sections.items()
        }

        # Detach every section from the temporary column layout.
        for groups in self.control_tab_sections.values():
            for group in groups:
                layout.removeWidget(group)
                group.setParent(None)
        layout.removeWidget(self._btn_clear_screws)
        self._btn_clear_screws.setParent(None)

        self.control_tabs = QTabWidget()
        self.control_tabs.setObjectName("controlTabs")
        self.control_tab_scrolls: dict[str, QScrollArea] = {}
        for tab, groups in self.control_tab_sections.items():
            page = QWidget()
            page_layout = QVBoxLayout(page)
            page_layout.setContentsMargins(6, 6, 6, 6)
            page_layout.setSpacing(2)
            for group in groups:
                page_layout.addWidget(group)
            if tab == "Planning":
                page_layout.addWidget(self._btn_clear_screws)
            page_layout.addStretch()

            page_scroll = QScrollArea()
            page_scroll.setWidgetResizable(True)
            page_scroll.setHorizontalScrollBarPolicy(
                Qt.ScrollBarPolicy.ScrollBarAsNeeded
            )
            page_scroll.setWidget(page)
            self.control_tab_scrolls[tab] = page_scroll
            self.control_tabs.addTab(page_scroll, tab)

        self.control_tabs.setCurrentIndex(1)
        #: Kept for callers that used to reach for the one panel scroll area.
        self.control_scroll = self.control_tab_scrolls["Planning"]

        container = QWidget()
        container.setMinimumWidth(390)
        container_layout = QVBoxLayout(container)
        container_layout.setContentsMargins(6, 6, 6, 6)
        container_layout.setSpacing(6)
        container_layout.addWidget(self.planning_cockpit)
        container_layout.addWidget(self.control_tabs, 1)

        # Keep the main planning stages visible and secondary tools collapsed.
        info_group.collapse()
        plan_params_group.collapse()
        wl_group.collapse()
        screw_group.collapse()
        measure_group.collapse()
        measure_list_group.collapse()
        view_group.collapse()

        panel.deleteLater()
        return container

    def _connect_signals(self):
        """Connect all widget signals to controller methods.

        Called once after both UI widgets and controllers are created.
        """
        # Window/Level
        self.window_slider.valueChanged.connect(
            self._view_ctrl.on_window_level_changed
        )
        self.level_slider.valueChanged.connect(
            self._view_ctrl.on_window_level_changed
        )
        self._btn_bone.clicked.connect(
            lambda: self._view_ctrl.set_preset(400, 1500)
        )
        self._btn_soft.clicked.connect(
            lambda: self._view_ctrl.set_preset(40, 400)
        )

        # Measurement
        self.measure_mode_combo.currentIndexChanged.connect(
            self._tool_ctrl.on_measure_mode_changed
        )
        self.measure_finish_btn.clicked.connect(
            self._tool_ctrl.finish_pending_measurement
        )
        self.measure_clear_btn.clicked.connect(
            self._tool_ctrl.clear_measurements
        )
        self.remove_measurement_btn.clicked.connect(
            self._tool_ctrl.remove_selected_measurement
        )
        self.show_measurement_btn.clicked.connect(
            self._tool_ctrl.jump_to_selected_measurement
        )
        self.edit_measurement_btn.clicked.connect(
            self._tool_ctrl.begin_edit_selected_measurement
        )
        self.measurement_list_widget.itemDoubleClicked.connect(
            lambda _item: self._tool_ctrl.jump_to_selected_measurement()
        )
        self.remove_screw_btn.clicked.connect(
            self._tool_ctrl.remove_selected_screw
        )
        self.screw_list_widget.currentRowChanged.connect(
            self._screw_mpr_ctrl.on_screw_selection_changed
        )
        self.screw_list_widget.currentRowChanged.connect(
            self._screw_edit_ctrl.on_screw_selection_changed
        )
        self.screw_list_widget.currentRowChanged.connect(
            self._on_screw_visual_selection_changed
        )
        self.screw_previous_btn.clicked.connect(
            self._screw_mpr_ctrl.select_previous
        )
        self.screw_next_btn.clicked.connect(
            self._screw_mpr_ctrl.select_next
        )
        self.screw_axis_mpr_btn.clicked.connect(self._screw_mpr_ctrl.enter)
        self.standard_mpr_btn.clicked.connect(self._screw_mpr_ctrl.exit)
        self.screw_axis_position_slider.valueChanged.connect(
            self._screw_mpr_ctrl.set_position
        )
        self.screw_axis_rotation_spin.valueChanged.connect(
            self._screw_mpr_ctrl.set_rotation
        )
        self.screw_mpr_reset_btn.clicked.connect(
            self._screw_mpr_ctrl.reset_view
        )
        self.selected_screw_diameter.valueChanged.connect(
            self._tool_ctrl.set_selected_screw_diameter
        )
        self.screw_edit_entry_btn.clicked.connect(
            lambda: self._screw_edit_ctrl.start("entry")
        )
        self.screw_edit_tip_btn.clicked.connect(
            lambda: self._screw_edit_ctrl.start("tip")
        )
        self.screw_edit_move_btn.clicked.connect(
            lambda: self._screw_edit_ctrl.start("move")
        )
        self.screw_edit_cancel_btn.clicked.connect(
            self._screw_edit_ctrl.cancel
        )

        # Segmentation
        self.seg_task_combo.currentIndexChanged.connect(
            self._seg_ctrl.on_task_changed
        )
        self.seg_label_spin.valueChanged.connect(
            self._seg_ctrl.apply_label_filter
        )
        self.seg_label_combo.currentIndexChanged.connect(
            self._seg_ctrl.on_label_combo_changed
        )
        self.seg_run_btn.clicked.connect(self._seg_ctrl.run)
        self.seg_refine_check.stateChanged.connect(
            self._on_segmentation_setting_changed
        )
        self.seg_show_2d_check.stateChanged.connect(
            self._seg_ctrl.update_visibility
        )
        self.seg_show_3d_check.stateChanged.connect(
            self._seg_ctrl.update_visibility
        )
        self.seg_clear_btn.clicked.connect(self._seg_ctrl.clear_overlay)
        self.seg_use_subregion_check.stateChanged.connect(
            self._on_segmentation_setting_changed
        )
        self.seg_subregion_dir_edit.textChanged.connect(
            self._on_segmentation_setting_changed
        )
        self.seg_subregion_browse_btn.clicked.connect(
            self._browse_subregion_model_dir
        )
        self.vertebra_isolate_btn.clicked.connect(
            self._seg_ctrl.toggle_vertebrae_isolation
        )
        self.vertebra_restore_btn.clicked.connect(
            self._seg_ctrl.restore_full_volume
        )
        self.vertebra_show_selected_btn.clicked.connect(
            self._seg_ctrl.show_selected_vertebrae
        )
        self.vertebra_show_all_btn.clicked.connect(
            self._seg_ctrl.show_all_vertebrae
        )
        self.vertebra_select_all_btn.clicked.connect(
            lambda: self._toggle_vertebra_level_checks(True)
        )
        self.vertebra_clear_all_btn.clicked.connect(
            lambda: self._toggle_vertebra_level_checks(False)
        )

        # Auto Pedicle Screw
        self.auto_screw_plan_btn.clicked.connect(
            self._auto_placement_ctrl.run_planning
        )

        # Planning parameters
        for spin_box in self._planner_spin_boxes().values():
            spin_box.valueChanged.connect(self._on_planner_parameter_changed)
        for combo in self._planner_choice_combos().values():
            combo.currentIndexChanged.connect(self._on_planner_parameter_changed)
        for check_box in self._planner_check_boxes().values():
            check_box.toggled.connect(self._on_planner_parameter_changed)
        for slider in self._planner_weight_sliders().values():
            slider.valueChanged.connect(self._on_planner_weight_changed)
        self.plan_reset_defaults_btn.clicked.connect(
            self.reset_planner_settings
        )
        # Transfer function preset
        self.tf_preset_combo.currentTextChanged.connect(
            self._view_ctrl.on_preset_changed
        )

        # Opacity
        self.opacity_slider.valueChanged.connect(
            self._view_ctrl.on_opacity_changed
        )

        # Clear screws button
        self._btn_clear_screws.clicked.connect(self._tool_ctrl.clear_screws)

        # Viewer signals
        for viewer in self._get_mpr_viewers():
            viewer.slice_changed.connect(self._on_slice_changed)
            viewer.crosshair_moved.connect(self._tool_ctrl.on_viewer_click)
            viewer.set_screw_interaction_callbacks(
                on_begin=self._screw_edit_ctrl.begin_drag,
                on_drag=self._screw_edit_ctrl.update_drag,
                on_end=self._screw_edit_ctrl.end_drag,
                on_select=self._select_screw_from_view,
            )
            viewer.set_measurement_interaction_callbacks(
                on_select=self._tool_ctrl.select_measurement_from_view,
                on_drag=self._tool_ctrl.update_measurement_point,
            )
            viewer.set_custom_scroll_handler(self._screw_mpr_ctrl.handle_scroll)

        # Only the cross-section view rotates under Ctrl+left-drag.
        self.coronal_viewer.set_custom_rotate_handler(
            self._screw_mpr_ctrl.handle_rotate_drag
        )
        self.measurement_list_widget.currentRowChanged.connect(
            self._tool_ctrl.on_measurement_selection_changed
        )
        self.viewer_3d.set_screw_interaction_callbacks(
            on_begin=self._screw_edit_ctrl.begin_drag,
            on_drag=self._screw_edit_ctrl.update_drag,
            on_end=self._screw_edit_ctrl.end_drag,
            on_select=self._select_screw_from_view,
        )

        # Initialize segmentation label options after signals are connected
        self._seg_ctrl.refresh_label_options()
        self._screw_mpr_ctrl.refresh_controls()
        self._screw_mpr_ctrl.refresh_selected_screw()
        self._screw_edit_ctrl.refresh_controls()

        # Single-view maximise: header double-click on every pane, and a
        # record of the last MPR pane the user touched for Ctrl+M.
        for name, pane in (
            ("axial", self.axial_viewer),
            ("sagittal", self.sagittal_viewer),
            ("coronal", self.coronal_viewer),
            ("3d", self.viewer_3d),
        ):
            signal = getattr(pane, "header_double_clicked", None)
            if signal is not None:
                signal.connect(self.toggle_maximized_view)
            del name
        for viewer in self._get_mpr_viewers():
            viewer.crosshair_moved.connect(
                lambda plane, *_unused: self._remember_focused_view(plane)
            )
        self._install_focus_tracking()
        # The workflow bar follows the study: a finished plan or a removed
        # screw changes which step comes next. (A new volume is handled by
        # reset_workspace() itself, once the previous study's mask and
        # screws are actually cleared -- see the end of reset_workspace().)
        table_model = self.screw_list_widget.model()
        # A bound method, not a lambda: PyQt uses ``self`` as the receiver
        # for a bound method, so the connection is torn down together with
        # this window.  A lambda has no such receiver, so during teardown it
        # can still fire -- from a table row change queued while this
        # window's children (including seg_run_btn) are already destroyed --
        # and raise ``wrapped C/C++ object ... has been deleted``.
        for signal in (table_model.rowsInserted, table_model.rowsRemoved, table_model.modelReset):
            signal.connect(self._on_screw_rows_changed)
        self._refresh_workflow_bar()
        self.screw_list_widget.currentRowChanged.connect(
            lambda *_unused: self.refresh_mode_indicators()
        )

    def _select_screw_from_view(self, screw_id: int) -> None:
        """Synchronize a screw picked in MPR/3D with the selection list."""
        if 0 <= int(screw_id) < self.screw_list_widget.count():
            self._active_selection_kind = "screw"
            self.screw_list_widget.setCurrentRow(int(screw_id))

    def _on_screw_visual_selection_changed(self, row: int) -> None:
        """Highlight the list-selected screw in the 3D scene."""
        if row >= 0:
            self._active_selection_kind = "screw"
        self.viewer_3d.set_selected_screw(row if row >= 0 else None)

    def _delete_active_selection(self) -> None:
        """Delete the most recently selected screw or measurement."""
        if (
            self._active_selection_kind == "measurement"
            and self.measurement_list_widget.currentRow() >= 0
        ):
            self._tool_ctrl.remove_selected_measurement()
            return
        self._tool_ctrl.remove_selected_screw()

    def _setup_menubar(self):
        """Setup the application menu bar."""
        menubar = self.menuBar()

        self._cancel_screw_edit_action = QAction(
            "Cancel Screw Edit",
            self,
        )
        self._cancel_screw_edit_action.setShortcut("Esc")
        self._cancel_screw_edit_action.setShortcutContext(
            Qt.ShortcutContext.WindowShortcut
        )
        self._cancel_screw_edit_action.triggered.connect(
            self._cancel_screw_interaction
        )
        self.addAction(self._cancel_screw_edit_action)

        self._delete_screw_action = QAction("Delete Selected Item", self)
        self._delete_screw_action.setShortcut("Delete")
        self._delete_screw_action.setShortcutContext(
            Qt.ShortcutContext.WindowShortcut
        )
        self._delete_screw_action.triggered.connect(
            self._delete_active_selection
        )
        self.addAction(self._delete_screw_action)

        # File menu
        file_menu = menubar.addMenu("File")

        open_action = QAction("Open DICOM Folder...", self)
        self._register_themed_icon(open_action, "open")
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self._dicom_ctrl.open_folder)
        file_menu.addAction(open_action)

        save_plan_action = QAction("Save Plan...", self)
        self._register_themed_icon(save_plan_action, "save")
        save_plan_action.setShortcut("Ctrl+S")
        save_plan_action.triggered.connect(self._plan_ctrl.save_dialog)
        file_menu.addAction(save_plan_action)

        load_plan_action = QAction("Load Plan...", self)
        load_plan_action.setShortcut("Ctrl+L")
        load_plan_action.triggered.connect(self._plan_ctrl.load_dialog)
        file_menu.addAction(load_plan_action)

        export_csv_action = QAction("Export Screws CSV...", self)
        export_csv_action.triggered.connect(self._plan_ctrl.export_csv_dialog)
        file_menu.addAction(export_csv_action)

        export_stl_action = QAction("Export Bone STL...", self)
        export_stl_action.triggered.connect(self._plan_ctrl.export_stl_dialog)
        file_menu.addAction(export_stl_action)

        file_menu.addSeparator()

        exit_action = QAction("Exit", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # Tools menu
        tools_menu = menubar.addMenu("Tools")

        select_action = QAction("Select", self)
        self._register_themed_icon(select_action, "select")
        select_action.triggered.connect(
            lambda: self._tool_ctrl.set_tool("navigate")
        )
        tools_menu.addAction(select_action)

        screw_action = QAction("Add Screw", self)
        self._register_themed_icon(screw_action, "screw")
        screw_action.triggered.connect(
            lambda: self._tool_ctrl.set_tool("screw")
        )
        tools_menu.addAction(screw_action)

        distance_action = QAction("Measure Distance", self)
        self._register_themed_icon(distance_action, "distance")
        distance_action.triggered.connect(
            lambda: self._tool_ctrl.set_tool("distance")
        )
        tools_menu.addAction(distance_action)

        angle_action = QAction("Measure Angle", self)
        self._register_themed_icon(angle_action, "angle")
        angle_action.triggered.connect(
            lambda: self._tool_ctrl.set_tool("angle")
        )
        tools_menu.addAction(angle_action)

        # View menu
        view_menu = menubar.addMenu("View")

        reset_view = QAction("Reset Camera", self)
        reset_view.setShortcut("R")
        reset_view.triggered.connect(self._view_ctrl.reset_camera)
        view_menu.addAction(reset_view)

        view_menu.addSeparator()
        self._layout_action_group = QActionGroup(self)
        self._layout_action_group.setExclusive(True)
        self._planning_layout_action = QAction("Planning Layout", self)
        self._planning_layout_action.setCheckable(True)
        self._planning_layout_action.setChecked(True)
        self._planning_layout_action.triggered.connect(
            lambda: self.set_view_layout("planning")
        )
        self._layout_action_group.addAction(self._planning_layout_action)
        view_menu.addAction(self._planning_layout_action)

        self._mpr_focus_layout_action = QAction("MPR Focus Layout", self)
        self._mpr_focus_layout_action.setCheckable(True)
        self._mpr_focus_layout_action.triggered.connect(
            lambda: self.set_view_layout("mpr_focus")
        )
        self._layout_action_group.addAction(self._mpr_focus_layout_action)
        view_menu.addAction(self._mpr_focus_layout_action)

        self._maximize_view_action = QAction("Maximize Current View", self)
        self._maximize_view_action.setShortcut("Ctrl+M")
        self._maximize_view_action.setCheckable(True)
        self._maximize_view_action.setToolTip(
            "Maximize the last-used view, or restore the previous layout"
        )
        self._maximize_view_action.triggered.connect(
            self._toggle_maximize_current_view
        )
        self._register_themed_icon(self._maximize_view_action, "layout")
        view_menu.addAction(self._maximize_view_action)

        view_menu.addSeparator()

        self.theme_combo = QComboBox()
        self.theme_combo.setObjectName("themeSelector")
        for theme_name, label in THEME_LABELS.items():
            self.theme_combo.addItem(label, theme_name)
        theme_index = self.theme_combo.findData(self._theme_name)
        self.theme_combo.setCurrentIndex(max(theme_index, 0))
        self.theme_combo.setMaximumWidth(110)
        self.theme_combo.setMaximumHeight(26)
        self.theme_combo.currentIndexChanged.connect(
            lambda: self.apply_theme(self.theme_combo.currentData())
        )
        view_menu.addAction(
            self._menu_combo_action("Theme:", self.theme_combo)
        )

        self.layout_combo = QComboBox()
        self.layout_combo.addItem("Planning (3D Large)", "planning")
        self.layout_combo.addItem("MPR Focus (2 x 2)", "mpr_focus")
        self.layout_combo.setCurrentIndex(
            self.layout_combo.findData(self._view_layout_mode)
        )
        self.layout_combo.currentIndexChanged.connect(
            lambda: self.set_view_layout(self.layout_combo.currentData())
        )
        view_menu.addAction(
            self._menu_combo_action("Layout:", self.layout_combo)
        )

        help_menu = menubar.addMenu("Help")
        self._screw_mpr_help_action = QAction("Screw MPR controls", self)
        self._screw_mpr_help_action.triggered.connect(
            self.show_screw_mpr_help
        )
        help_menu.addAction(self._screw_mpr_help_action)
        help_menu.addSeparator()
        self._about_action = QAction(f"About {__title__}", self)
        self._about_action.triggered.connect(self.show_about_dialog)
        help_menu.addAction(self._about_action)

    def _menu_combo_action(self, caption: str, combo: QComboBox) -> QWidgetAction:
        """Wrap a labelled combo box so it can live inside a QMenu."""
        row = QWidget(self)
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(10, 3, 10, 3)
        row_layout.setSpacing(6)
        row_layout.addWidget(QLabel(caption))
        row_layout.addWidget(combo, 1)
        action = QWidgetAction(self)
        action.setDefaultWidget(row)
        return action

    def show_screw_mpr_help(self) -> None:
        """Show the Screw MPR mouse and keyboard gesture map."""
        QMessageBox.information(
            self, "Screw MPR controls", SCREW_MPR_CONTROLS_HELP
        )

    def show_about_dialog(self) -> None:
        """Display version, creator credit, contact, license, and safety scope."""
        QMessageBox.about(self, f"About {__title__}", build_about_html())

    def _cancel_screw_interaction(self) -> None:
        """Exit any locked MPR/3D screw movement and preserve its last state."""
        for viewer in self._get_mpr_viewers():
            viewer.cancel_screw_interaction()
        self.viewer_3d.cancel_screw_interaction()
        self._screw_edit_ctrl.cancel()
        self._tool_ctrl.cancel_measurement_edit(show_message=False)

    def _setup_toolbar(self):
        """Setup the application toolbar and unified icon tool palette."""
        toolbar = QToolBar("Main Toolbar")
        toolbar.setObjectName("mainToolbar")
        toolbar.setMovable(False)
        toolbar.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.addToolBar(toolbar)
        self.main_toolbar = toolbar

        open_action = QAction("Open DICOM", self)
        open_action.setToolTip("Open a DICOM series folder")
        open_action.triggered.connect(self._dicom_ctrl.open_folder)
        self._register_themed_icon(open_action, "open")
        toolbar.addAction(open_action)
        self._open_toolbar_action = open_action

        toolbar.addSeparator()

        # Unified tool palette (mutually exclusive)
        self._tool_group = QActionGroup(self)
        self._tool_group.setExclusive(True)

        self._select_tool_action = QAction("Select", self)
        self._select_tool_action.setToolTip("Select, navigate, and inspect")
        self._select_tool_action.setCheckable(True)
        self._select_tool_action.setChecked(True)
        self._select_tool_action.triggered.connect(
            lambda: self._tool_ctrl.set_tool("navigate")
        )
        self._register_themed_icon(self._select_tool_action, "select")
        self._tool_group.addAction(self._select_tool_action)
        toolbar.addAction(self._select_tool_action)

        self._add_screw_tool_action = QAction("Add Screw", self)
        self._add_screw_tool_action.setToolTip(
            "Add a screw: click entry, then tip, in an MPR view"
        )
        self._add_screw_tool_action.setCheckable(True)
        self._add_screw_tool_action.triggered.connect(
            lambda: self._tool_ctrl.set_tool("screw")
        )
        self._register_themed_icon(self._add_screw_tool_action, "screw")
        self._tool_group.addAction(self._add_screw_tool_action)
        toolbar.addAction(self._add_screw_tool_action)

        self._distance_tool_action = QAction("Distance", self)
        self._distance_tool_action.setToolTip(
            "Measure distance: click two points in one MPR view"
        )
        self._distance_tool_action.setCheckable(True)
        self._distance_tool_action.triggered.connect(
            lambda: self._tool_ctrl.set_tool("distance")
        )
        self._register_themed_icon(self._distance_tool_action, "distance")
        self._tool_group.addAction(self._distance_tool_action)
        toolbar.addAction(self._distance_tool_action)

        self._angle_tool_action = QAction("Angle", self)
        self._angle_tool_action.setToolTip(
            "Measure angle: click three points in one MPR view"
        )
        self._angle_tool_action.setCheckable(True)
        self._angle_tool_action.triggered.connect(
            lambda: self._tool_ctrl.set_tool("angle")
        )
        self._register_themed_icon(self._angle_tool_action, "angle")
        self._tool_group.addAction(self._angle_tool_action)
        toolbar.addAction(self._angle_tool_action)

        self._nav_action = self._select_tool_action
        self._screw_action = self._add_screw_tool_action
        self._measure_action = self._distance_tool_action

        toolbar.addSeparator()

        self._fit_mpr_action = QAction("Fit MPR", self)
        self._fit_mpr_action.setToolTip("Fit CT images tightly in all MPR views")
        self._fit_mpr_action.triggered.connect(self.fit_mpr_views)
        self._register_themed_icon(self._fit_mpr_action, "fit")
        toolbar.addAction(self._fit_mpr_action)

        self._zoom_out_3d_action = QAction("3D -", self)
        self._zoom_out_3d_action.setToolTip("Zoom out the 3D view")
        self._zoom_out_3d_action.triggered.connect(
            lambda: self.viewer_3d.zoom_camera(1.0 / 1.2)
        )
        self._register_themed_icon(self._zoom_out_3d_action, "zoom_out")
        toolbar.addAction(self._zoom_out_3d_action)

        self._zoom_in_3d_action = QAction("3D +", self)
        self._zoom_in_3d_action.setToolTip("Zoom in the 3D view")
        self._zoom_in_3d_action.triggered.connect(
            lambda: self.viewer_3d.zoom_camera(1.2)
        )
        self._register_themed_icon(self._zoom_in_3d_action, "zoom_in")
        toolbar.addAction(self._zoom_in_3d_action)

        self._fit_3d_action = QAction("Fit 3D", self)
        self._fit_3d_action.setToolTip("Fit all visible objects in the 3D view")
        self._fit_3d_action.triggered.connect(self.viewer_3d.fit_to_view)
        self._register_themed_icon(self._fit_3d_action, "reset")
        toolbar.addAction(self._fit_3d_action)

        toolbar.addSeparator()

        self._screw_mpr_action = QAction("Screw MPR", self)
        self._screw_mpr_action.setToolTip(
            "Toggle screw-aligned MPR for the selected screw"
        )
        self._screw_mpr_action.setCheckable(True)
        self._screw_mpr_action.setEnabled(False)
        self._screw_mpr_action.triggered.connect(self._toggle_screw_mpr)
        self._register_themed_icon(self._screw_mpr_action, "screw_mpr")
        toolbar.addAction(self._screw_mpr_action)

        self._refresh_themed_icons()
        self.refresh_mode_indicators()

    def _toggle_screw_mpr(self, checked: bool) -> None:
        """Enter or leave Screw MPR from the toolbar toggle.

        Both controller entry points end in ``refresh_controls()``, which
        calls back into ``refresh_mode_indicators``; refreshing again here
        would rewrite the status-bar mode text twice per toggle.
        """
        if checked:
            self._screw_mpr_ctrl.enter()
        else:
            self._screw_mpr_ctrl.exit()

    def _setup_statusbar(self):
        """Setup the status bar with mode, coordinate, and HU display."""
        self.statusbar = QStatusBar()
        self.setStatusBar(self.statusbar)

        self._mode_label = QLabel("Select")
        self._mode_label.setObjectName("statusModeLabel")
        self._coord_label = QLabel("X: --  Y: --  Z: --")
        self._hu_label = QLabel("HU: --")
        self.statusbar.addPermanentWidget(self._mode_label)
        self.statusbar.addPermanentWidget(self._coord_label)
        self.statusbar.addPermanentWidget(self._hu_label)

        self.statusbar.showMessage("Ready")
        self.refresh_mode_indicators()

        # Both one-time migrations can fire on the same first launch, and the
        # user has to learn about both -- the second must not silently replace
        # the first.
        migrations = [
            message
            for message in (
                self._planner_mode_migration_message,
                self._planner_clearance_migration_message,
            )
            if message
        ]
        if migrations:
            self.statusbar.showMessage(" · ".join(migrations))

    def _get_mpr_viewers(self) -> List[MPRViewer]:
        """Return list of active MPR viewers."""
        return [
            viewer
            for viewer in [self.axial_viewer, self.sagittal_viewer, self.coronal_viewer]
            if viewer is not None
        ]

    def fit_mpr_views(self) -> None:
        """Fit every CT slice to its current MPR viewport."""
        for viewer in self._get_mpr_viewers():
            viewer.fit_to_view()

    def apply_theme(self, theme_name: str, persist: bool = True) -> None:
        """Apply one named palette immediately and optionally persist it."""
        if theme_name not in THEMES:
            theme_name = DEFAULT_THEME
        app = QApplication.instance()
        if app is not None:
            app.setProperty("themeName", theme_name)
            app.setStyleSheet(load_stylesheet(theme_name))
        self._theme_name = theme_name
        self._refresh_themed_icons()
        if hasattr(self, "screw_list_widget"):
            self.screw_list_widget.apply_theme(theme_name)
        # An inline colour the application stylesheet cannot reach, so the
        # previous palette's red would otherwise stay on the narrow-pedicle row
        # until the next selection.
        self._repaint_pedicle_row()
        pan_icon_color = THEMES[theme_name]["viewer_foreground"]
        for viewer in self._get_mpr_viewers():
            viewer.refresh_orientation_markers(render=True)
            set_pan_icon_color = getattr(viewer, "set_pan_icon_color", None)
            if callable(set_pan_icon_color):
                set_pan_icon_color(pan_icon_color)

        if hasattr(self, "theme_combo"):
            index = self.theme_combo.findData(theme_name)
            if index >= 0 and index != self.theme_combo.currentIndex():
                previous = self.theme_combo.blockSignals(True)
                self.theme_combo.setCurrentIndex(index)
                self.theme_combo.blockSignals(previous)

        if persist:
            self._settings.setValue("appearance/theme", theme_name)
            self._settings.sync()
        if hasattr(self, "statusbar"):
            self.statusbar.showMessage(
                f"Theme changed to {THEME_LABELS[theme_name]}"
            )

    def _register_themed_icon(self, target, kind: str) -> None:
        """Track a widget/action whose icon must follow the active palette."""
        self._themed_icon_targets.append((target, kind))
        target.setIcon(create_tool_icon(kind, THEMES[self._theme_name]["accent"]))

    def _refresh_themed_icons(self) -> None:
        """Repaint every registered icon with the active theme's accent."""
        accent = THEMES[self._theme_name]["accent"]
        for target, kind in self._themed_icon_targets:
            target.setIcon(create_tool_icon(kind, accent))

    def refresh_mode_indicators(self) -> None:
        """Sync the Screw MPR toggle and the status-bar mode label."""
        action = getattr(self, "_screw_mpr_action", None)
        if action is not None:
            active = bool(self._screw_mpr_ctrl.is_active)
            if action.isChecked() != active:
                previous = action.blockSignals(True)
                action.setChecked(active)
                action.blockSignals(previous)
            action.setEnabled(active or self.screw_axis_mpr_btn.isEnabled())
        label = getattr(self, "_mode_label", None)
        if label is not None:
            label.setText(self.current_mode_text())

    def current_mode_text(self) -> str:
        """Return the status-bar wording for the active tool or review mode."""
        if self._screw_mpr_ctrl.is_active:
            row = self.screw_list_widget.currentRow()
            screws = self._tool_ctrl.screw_tool.get_screws()
            if 0 <= row < len(screws):
                screw = screws[row]
                level = str(getattr(screw, "vertebra_level", "") or "").strip()
                side = str(getattr(screw, "side", "") or "").strip()
                identity = " ".join(
                    part for part in (f"#{row + 1}", level, side) if part
                )
                return f"Screw MPR · {identity}"
            return "Screw MPR"
        tool = self._tool_ctrl.active_tool
        return self.TOOL_MODE_LABELS.get(tool, str(tool).capitalize())

    # ------------------------------------------------------------------
    # Planning parameters
    # ------------------------------------------------------------------

    # Only settings with an editable widget belong here: `min_convergence_deg`
    # has no spin box, so persisting it implied an editability that never
    # existed and pinned the value at the dataclass default regardless of
    # what was stored.
    PLANNER_SETTINGS_KEYS = (
        "pedicle_fill_ratio",
        "wall_clearance_mm",
        "anterior_margin_mm",
        "max_convergence_deg",
        "trajectory_hu_threshold",
        "narrow_pedicle_mm",
        "narrow_lateral_breach_mm",
        "endplate_tolerance_deg",
    )

    #: Planner settings whose widget is a check box.  Kept apart from
    #: PLANNER_SETTINGS_KEYS because that tuple's loader coerces with float(),
    #: and because QSettings gives a bool back as the string "true"/"false" --
    #: PlannerConfig.from_mapping does that coercion.
    PLANNER_BOOL_SETTINGS_KEYS = ("endplate_parallel",)

    #: Views that "maximize:<name>" accepts.
    MAXIMIZABLE_VIEWS = ("axial", "sagittal", "coronal", "3d")

    #: Status-bar wording for each active tool id.
    TOOL_MODE_LABELS = {
        "navigate": "Select",
        "screw": "Add Screw",
        "distance": "Distance",
        "angle": "Angle",
        "path": "Path",
    }

    @staticmethod
    def _segmentation_settings() -> QSettings:
        """Return QSettings positioned inside the persisted segmentation group."""
        settings = app_settings()
        settings.beginGroup("segmentation")
        return settings

    def save_segmentation_settings(self) -> None:
        """Persist the optional subregion-model choices for the next session."""
        settings = self._segmentation_settings()
        settings.setValue(
            "use_subregion_model", self.seg_use_subregion_check.isChecked()
        )
        settings.setValue(
            "subregion_model_dir", self.seg_subregion_dir_edit.text().strip()
        )
        settings.setValue("refine_mask", self.seg_refine_check.isChecked())
        settings.endGroup()
        settings.sync()

    def load_segmentation_settings(self) -> None:
        """Restore the persisted segmentation choices."""
        settings = self._segmentation_settings()
        raw_enabled = settings.value("use_subregion_model", False)
        model_dir = settings.value("subregion_model_dir", "")
        raw_refine = settings.value("refine_mask", True)
        settings.endGroup()

        enabled = str(raw_enabled).strip().lower() in ("true", "1", "yes")
        refine = str(raw_refine).strip().lower() in ("true", "1", "yes")
        for widget in (
            self.seg_use_subregion_check,
            self.seg_subregion_dir_edit,
            self.seg_refine_check,
        ):
            previous = widget.blockSignals(True)
            try:
                if widget is self.seg_use_subregion_check:
                    widget.setChecked(enabled)
                elif widget is self.seg_refine_check:
                    widget.setChecked(refine)
                else:
                    widget.setText(str(model_dir or ""))
            finally:
                widget.blockSignals(previous)

    def _on_segmentation_setting_changed(self, _value=None) -> None:
        """Persist the subregion-model choices whenever the user edits one."""
        self.save_segmentation_settings()

    def _browse_subregion_model_dir(self) -> None:
        """Let the user pick the nnU-Net subregion model directory."""
        directory = QFileDialog.getExistingDirectory(
            self,
            "Select Pedicle Subregion Model Directory",
            self.seg_subregion_dir_edit.text().strip(),
        )
        if directory:
            self.seg_subregion_dir_edit.setText(directory)

    @staticmethod
    def _planner_settings() -> QSettings:
        """Return QSettings positioned inside the persisted planner group."""
        settings = app_settings()
        settings.beginGroup("planner")
        return settings

    def _planner_spin_boxes(self) -> dict:
        """Map PlannerConfig field names to their editing spin boxes."""
        return {
            "pedicle_fill_ratio": self.plan_fill_ratio_spin,
            "wall_clearance_mm": self.plan_wall_clearance_spin,
            "anterior_margin_mm": self.plan_anterior_margin_spin,
            "max_convergence_deg": self.plan_max_convergence_spin,
            "trajectory_hu_threshold": self.plan_hu_threshold_spin,
            "narrow_pedicle_mm": self.plan_narrow_pedicle_spin,
            "narrow_lateral_breach_mm": self.plan_narrow_lateral_spin,
            "endplate_tolerance_deg": self.plan_endplate_tolerance_spin,
        }

    def _planner_choice_combos(self) -> dict:
        """Map PlannerConfig enum field names to their combo boxes."""
        return {
            "mode": self.plan_mode_combo,
            "trajectory": self.plan_trajectory_combo,
        }

    def _planner_check_boxes(self) -> dict:
        """Map PlannerConfig boolean field names to their check boxes."""
        return {"endplate_parallel": self.plan_endplate_parallel_check}

    def _planner_weight_sliders(self) -> dict:
        """Map OptimizerWeights field names to their percent sliders."""
        return {
            "safety": self.plan_weight_safety,
            "density": self.plan_weight_density,
            "rod": self.plan_weight_rod,
        }

    def _refresh_planner_weight_labels(self) -> None:
        """Show each weight slider's value as the weight the planner receives."""
        for name, slider in self._planner_weight_sliders().items():
            label = self._planner_weight_value_labels[f"plan_weight_{name}"]
            label.setText(f"{slider.value() / 100.0:.2f}")

    def planner_config(self) -> PlannerConfig:
        """Return the planner configuration currently shown in the panel."""
        data = {
            key: spin_box.value()
            for key, spin_box in self._planner_spin_boxes().items()
        }
        for key, combo in self._planner_choice_combos().items():
            data[key] = combo.currentData()
        for key, check_box in self._planner_check_boxes().items():
            data[key] = check_box.isChecked()
        # Sliders hold percents; the optimiser wants plain multipliers.
        data["weights"] = {
            name: slider.value() / 100.0
            for name, slider in self._planner_weight_sliders().items()
        }
        return PlannerConfig.from_mapping(data)

    def save_planner_settings(self) -> None:
        """Persist the panel's planner parameters for the next session.

        Only values with an editable widget are written: the scalar spin
        boxes, the mode/trajectory combos, and the three weight sliders.
        Writing the full `PlannerConfig` mapping would also persist fields
        (like the two non-slider weights) that no widget ever re-applies on
        load, implying an editability that does not exist.
        """
        settings = self._planner_settings()
        values = self.planner_config().to_mapping()
        for key in self.PLANNER_SETTINGS_KEYS:
            settings.setValue(key, values[key])
        for key in self._planner_choice_combos():
            settings.setValue(key, values[key])
        for key in self.PLANNER_BOOL_SETTINGS_KEYS:
            settings.setValue(key, values[key])
        for name in self._planner_weight_sliders():
            settings.setValue(f"weights/{name}", values["weights"][name])
        settings.endGroup()
        settings.sync()

    def load_planner_settings(self) -> None:
        """Restore persisted planner parameters, falling back to defaults."""
        settings = self._planner_settings()
        defaults = PlannerConfig().to_mapping()
        stored = {}
        for key in self.PLANNER_SETTINGS_KEYS:
            raw = settings.value(key, defaults[key])
            try:
                stored[key] = float(raw)
            except (TypeError, ValueError):
                logger.warning(
                    "Ignoring unreadable planner setting %s=%r", key, raw
                )
                stored = {}
                break
        for key in self._planner_choice_combos():
            stored[key] = str(settings.value(key, defaults[key]) or defaults[key])
        for key in self.PLANNER_BOOL_SETTINGS_KEYS:
            # Handed to PlannerConfig.from_mapping as-is: QSettings returns
            # "false" as a string, and bool("false") is True.
            stored[key] = settings.value(key, defaults[key])
        weights = {}
        for name in self._planner_weight_sliders():
            default_weight = defaults["weights"][name]
            raw = settings.value(f"weights/{name}", default_weight)
            try:
                weights[name] = float(raw)
            except (TypeError, ValueError):
                logger.warning(
                    "Ignoring unreadable planner weight %s=%r", name, raw
                )
                weights[name] = float(default_weight)
        stored["weights"] = weights
        raw_migrated = settings.value(_PLANNER_MODE_MIGRATION_KEY, False)
        if str(raw_migrated).strip().lower() not in ("true", "1", "yes"):
            # Runs once per settings store, whatever the stored mode is, so a
            # user who picks Legacy *after* this keeps it.
            settings.setValue(_PLANNER_MODE_MIGRATION_KEY, True)
            if stored.get("mode") == "legacy":
                stored["mode"] = "optimizer"
                settings.setValue("mode", "optimizer")
                self._planner_mode_migration_message = PLANNER_MODE_MIGRATION_MESSAGE
        self._migrate_wall_clearance(settings, stored)
        settings.endGroup()
        settings.sync()

        try:
            config = PlannerConfig.from_mapping(stored)
        except ValueError as exc:
            logger.warning(
                "Stored planner settings are invalid (%s); using defaults", exc
            )
            config = PlannerConfig()
        self._apply_planner_config(config)

    def _migrate_wall_clearance(self, settings: QSettings, stored: dict) -> None:
        """Retire the 1.0 mm wall clearance older builds persisted as the default.

        W7 moved the planner default to 0 mm: at 1 mm the optimiser could not
        satisfy ``min_wall >= 1.0`` on most sides of a real study (8 of 12
        measured) and silently fell back to the legacy planner, so a user
        upgrading from an older build would keep getting legacy trajectories
        from a number they never chose.  Out-defaulting it is not enough -- the
        old value is *persisted*, so it wins over the new default forever.

        Mirrors the Legacy -> Optimizer migration next to it: it runs once per
        settings store, whatever is stored, so a clearance the user picks
        *afterwards* (1.0 mm included) is respected.  A persisted value that is
        not the old default -- 0.5 or 1.5 mm -- was a deliberate choice and is
        left alone; only the flag is written.

        The one case that does *not* count as a run is an unreadable store.
        ``load_planner_settings`` empties ``stored`` on the first numeric key it
        cannot parse, so a single malformed value hides the persisted clearance
        from this check entirely.  Marking the migration done there retired it
        permanently while the old 1.0 mm survived -- exactly the state it
        exists to end -- so the flag is written only once the value has
        actually been read.
        """
        raw_migrated = settings.value(_PLANNER_CLEARANCE_MIGRATION_KEY, False)
        if str(raw_migrated).strip().lower() in ("true", "1", "yes"):
            return
        clearance = stored.get("wall_clearance_mm")
        persisted = settings.contains("wall_clearance_mm")
        if persisted and not isinstance(clearance, float):
            return                       # unreadable; retry on a later launch
        settings.setValue(_PLANNER_CLEARANCE_MIGRATION_KEY, True)
        if not persisted or not isinstance(clearance, float):
            return                       # nothing persisted; the new default applies
        if abs(clearance - _LEGACY_WALL_CLEARANCE_MM) > 1e-6:
            return
        stored["wall_clearance_mm"] = 0.0
        settings.setValue("wall_clearance_mm", 0.0)
        self._planner_clearance_migration_message = PLANNER_CLEARANCE_MIGRATION_MESSAGE

    def reset_planner_settings(self) -> None:
        """Restore the built-in planner defaults and persist them."""
        self._apply_planner_config(PlannerConfig())
        self.save_planner_settings()
        if hasattr(self, "statusbar"):
            self.statusbar.showMessage("Planning parameters reset to defaults")

    def _apply_planner_config(self, config: PlannerConfig) -> None:
        """Write config values into the panel widgets without re-saving them."""
        values = config.to_mapping()
        for key, spin_box in self._planner_spin_boxes().items():
            previous = spin_box.blockSignals(True)
            spin_box.setValue(float(values[key]))
            spin_box.blockSignals(previous)
        for key, combo in self._planner_choice_combos().items():
            index = combo.findData(values[key])
            if index < 0:
                logger.warning("Unknown planner %s %r; keeping the current choice",
                               key, values[key])
                continue
            previous = combo.blockSignals(True)
            combo.setCurrentIndex(index)
            combo.blockSignals(previous)
        for key, check_box in self._planner_check_boxes().items():
            previous = check_box.blockSignals(True)
            check_box.setChecked(bool(values[key]))
            check_box.blockSignals(previous)
        for name, slider in self._planner_weight_sliders().items():
            previous = slider.blockSignals(True)
            slider.setValue(int(round(float(values["weights"][name]) * 100.0)))
            slider.blockSignals(previous)
        self._refresh_planner_weight_labels()
        self._push_screw_tool_thresholds()

    def _push_screw_tool_thresholds(self) -> None:
        """Judge an edited screw by the parameters the panel currently shows.

        The screw tool re-grades on every drag and every diameter change, and
        two of its verdicts are thresholds rather than geometry: the cortical
        clearance worth a note, and the width below which a pedicle is narrow.
        Left on its own defaults the tool would disagree with the planner that
        produced the screw, and the *edit* would appear to have changed it.

        Called from both directions a parameter can move -- a config written
        into the panel, and a user turning a spin box -- so the tool is never
        a step behind.  Guarded because ``load_planner_settings`` runs during
        construction, when the tool controller may not exist yet.
        """
        tool_ctrl = getattr(self, "_tool_ctrl", None)
        screw_tool = getattr(tool_ctrl, "screw_tool", None)
        if screw_tool is None:
            return
        screw_tool.set_wall_clearance_mm(self.plan_wall_clearance_spin.value())
        screw_tool.set_narrow_pedicle_mm(self.plan_narrow_pedicle_spin.value())

    def _on_planner_parameter_changed(self, _value=None) -> None:
        """Persist planner parameters whenever the user edits one."""
        self.save_planner_settings()
        self._push_screw_tool_thresholds()

    def _on_planner_weight_changed(self, _value=None) -> None:
        """Refresh the weight readouts and persist the new objective weights."""
        self._refresh_planner_weight_labels()
        self.save_planner_settings()

    # ------------------------------------------------------------------
    # Screw inspector
    # ------------------------------------------------------------------

    def _clear_screw_metric_rows(self) -> None:
        """Reset the clinical metric rows of the screw inspector."""
        for label in (
            self.selected_screw_endplate,
            self.selected_screw_body_hu,
            self.selected_screw_wall,
            self.selected_screw_facet,
            self.selected_screw_heary,
            self.selected_screw_pedicle,
            self.selected_screw_alignment,
        ):
            label.setText("--")
        self.selected_screw_trajectory.setText("—")
        self._pedicle_row_narrow = False
        self._repaint_pedicle_row()

    #: Whether the cockpit's Pedicle row is currently showing a narrow pedicle.
    #: Its colour is an inline style, so the application stylesheet cannot
    #: repaint it on a theme change; remembering the verdict is what lets
    #: :meth:`apply_theme` redraw it without a fresh selection.
    _pedicle_row_narrow = False

    def _repaint_pedicle_row(self) -> None:
        """Paint the cockpit's Pedicle row in the active palette's danger colour."""
        if not hasattr(self, "selected_screw_pedicle"):
            return
        self.selected_screw_pedicle.setStyleSheet(
            f"color: {THEMES[self._theme_name]['danger']};"
            if self._pedicle_row_narrow
            else ""
        )

    def _update_screw_metric_rows(self, metrics: dict) -> None:
        """Fill the clinical metric rows from a screw's metric bundle."""
        self._clear_screw_metric_rows()

        trajectory_type = str(metrics.get("trajectory_type") or "").strip().lower()
        if trajectory_type == "cbt":
            trajectory_text = "CBT"
            cranial = metrics.get("cbt_cranial_angle_deg")
            if cranial is not None:
                trajectory_text += f" (cranial {float(cranial):.1f}°)"
            self.selected_screw_trajectory.setText(trajectory_text)
        elif trajectory_type == "traditional":
            self.selected_screw_trajectory.setText("Traditional")

        body_hu = metrics.get("body_mean_hu")
        if body_hu is not None:
            self.selected_screw_body_hu.setText(f"{float(body_hu):.0f} HU")

        min_wall = metrics.get("min_wall_mm")
        if min_wall is not None:
            self.selected_screw_wall.setText(f"{float(min_wall):.1f} mm")

        endplate_angle = metrics.get("endplate_angle_deg")
        if isinstance(endplate_angle, (int, float)) and not isinstance(endplate_angle, bool):
            self.selected_screw_endplate.setText(f"{float(endplate_angle):+.1f}°")

        facet_grade = metrics.get("facet_grade")
        facet_text = str(metrics.get("facet_text") or "").strip()
        if facet_grade is not None:
            facet = f"Grade {int(facet_grade)}"
            if facet_text:
                facet = f"{facet} — {facet_text}"
            self.selected_screw_facet.setText(facet)
        elif facet_text:
            self.selected_screw_facet.setText(facet_text)

        heary = str(metrics.get("heary_direction") or "").strip()
        if heary:
            self.selected_screw_heary.setText(heary)

        self.selected_screw_pedicle.setText(pedicle_row_text(metrics))
        self._pedicle_row_narrow = is_narrow_pedicle(metrics)
        self._repaint_pedicle_row()

        # Type-guarded like the endplate angle above: these arrive from a plan
        # file, which the app does not own -- a hand-edited or third-party JSON
        # with a string here used to raise straight out of screw selection.
        rod = metrics.get("rod_misalignment_mm")
        deviation = metrics.get("convergence_deviation_deg")
        alignment: List[str] = []
        if isinstance(rod, (int, float)) and not isinstance(rod, bool):
            alignment.append(f"rod {float(rod):.1f} mm")
        if isinstance(deviation, (int, float)) and not isinstance(deviation, bool):
            # Signed: which way this level differs from its neighbours is the
            # part a surgeon acts on.
            alignment.append(f"conv {float(deviation):+.1f}°")
        if alignment:
            self.selected_screw_alignment.setText(" · ".join(alignment))

    def update_selected_screw_inspector(
        self,
        index: int,
        screw,
        review_active: bool,
    ) -> None:
        """Update the Planning Cockpit summary for the selected screw."""
        self.selected_screw_title.setProperty(
            "reviewActive",
            bool(review_active),
        )
        title_style = self.selected_screw_title.style()
        title_style.unpolish(self.selected_screw_title)
        title_style.polish(self.selected_screw_title)
        if screw is None:
            self.selected_screw_counter.setText("No screws")
            self.selected_screw_title.setText("No screw selected")
            previous = self.selected_screw_diameter.blockSignals(True)
            self.selected_screw_diameter.setValue(DEFAULT_SCREW_DIAMETER)
            self.selected_screw_diameter.blockSignals(previous)
            self.selected_screw_diameter.setEnabled(False)
            self.selected_screw_length.setText("--")
            self.selected_screw_convergence.setText("--")
            self.selected_screw_craniocaudal.setText("--")
            self.selected_screw_grade.setText("Grade --")
            self.selected_screw_hu.setText("--")
            self.selected_screw_source.setText("--")
            self._clear_screw_metric_rows()
            self.selected_screw_warning.setText(
                "Select a screw to inspect its trajectory."
            )
            return

        screw_count = self.screw_list_widget.count()
        self.selected_screw_counter.setText(
            f"Screw {index + 1} of {screw_count}"
        )
        self.screw_list_widget.scrollToRow(index)

        level = str(getattr(screw, "vertebra_level", "") or "").strip()
        side = str(getattr(screw, "side", "") or "").strip().capitalize()
        identity = " ".join(part for part in (level, side) if part)
        if not identity:
            identity = f"Screw #{index + 1}"

        from src.core.auto_screw_planner import NARROW_PEDICLE_WARNING_PREFIX
        from src.core.screw_geometry import convergence_angle_deg, craniocaudal_angle_deg
        convergence = convergence_angle_deg(screw.entry_point, screw.target_point, screw.side or None)
        craniocaudal = craniocaudal_angle_deg(screw.entry_point, screw.target_point)

        self.selected_screw_title.setText(identity)
        previous = self.selected_screw_diameter.blockSignals(True)
        self.selected_screw_diameter.setValue(float(screw.diameter))
        self.selected_screw_diameter.blockSignals(previous)
        self.selected_screw_diameter.setEnabled(True)
        self.selected_screw_length.setText(f"{screw.length:.1f} mm")
        self.selected_screw_convergence.setText(f"{convergence:+.1f}°")
        self.selected_screw_craniocaudal.setText(f"{craniocaudal:+.1f}°")
        self.selected_screw_grade.setText(f"Grade {screw.grade}")

        mean_hu = getattr(screw, "mean_hu", None)
        min_hu = getattr(screw, "min_hu", None)
        if mean_hu is None or min_hu is None:
            self.selected_screw_hu.setText("-- (no CT sample)")
        else:
            self.selected_screw_hu.setText(
                f"mean {mean_hu:.0f} / min {min_hu:.0f} HU"
            )
        self.selected_screw_source.setText(
            "Auto plan"
            if getattr(screw, "source", "manual") == "auto"
            else "Manual"
        )
        self._update_screw_metric_rows(getattr(screw, "metrics", None) or {})

        breach_distance = float(getattr(screw, "breach_distance", 0.0))
        lines = list(getattr(screw, "warnings", []) or [])
        if screw.grade == "N/A":
            lines.insert(
                0, "Grade N/A — run segmentation to grade this screw."
            )
        elif breach_distance > 0.0:
            lines.insert(
                0,
                f"Estimated breach {breach_distance:.1f} mm — verify on CT.",
            )
        else:
            lines.insert(
                0, "No estimated breach — CT review is still required."
            )
        # The narrow note explains why the screw is the size it is, so it reads
        # before the breach line rather than after it.
        lines.sort(key=lambda line: not line.startswith(NARROW_PEDICLE_WARNING_PREFIX))
        self.selected_screw_warning.setText("\n".join(lines))

    def update_vertebra_level_checks(self, detected_labels: list) -> None:
        """Build the shared 3D visibility and planning checkbox set."""
        from src.core.pedicle_analyzer import VERTEBRA_LABELS

        self._clear_vertebra_level_grid()
        self.vertebra_display_list.clear()
        valid_labels = [
            int(label_id)
            for label_id in sorted(set(detected_labels), reverse=True)
            if int(label_id) in VERTEBRA_LABELS
            and VERTEBRA_LABELS[int(label_id)] != "sacrum"
        ]

        self._updating_vertebra_level_checks = True
        for index, label_id in enumerate(valid_labels):
            name = VERTEBRA_LABELS[label_id]
            checkbox = QCheckBox(name)
            checkbox.setObjectName("vertebraLevelCheck")
            checkbox.setChecked(True)
            checkbox.toggled.connect(self._on_vertebra_level_selection_changed)
            self._vertebra_level_checks[label_id] = checkbox
            self._vertebra_level_grid.addWidget(
                checkbox,
                index // 4,
                index % 4,
            )

            item = QListWidgetItem(name)
            item.setData(Qt.ItemDataRole.UserRole, label_id)
            item.setSelected(True)
            self.vertebra_display_list.addItem(item)
        self._updating_vertebra_level_checks = False

        if not valid_labels:
            self._add_vertebra_level_placeholder()
        has_levels = bool(valid_labels)
        self.vertebra_select_all_btn.setEnabled(has_levels)
        self.vertebra_clear_all_btn.setEnabled(has_levels)
        self.vertebra_show_selected_btn.setEnabled(has_levels)
        self.vertebra_show_all_btn.setEnabled(has_levels)
        self._on_vertebra_level_selection_changed(apply_to_view=False)
        # Called as segmentation finishes, after its mask is recorded: step 2
        # is done even when the plan button's enabled state did not change.
        self._refresh_workflow_bar()

    def update_vertebra_checkboxes(self, detected_labels: list) -> None:
        """Compatibility alias for the shared vertebra checkbox set."""
        self.update_vertebra_level_checks(detected_labels)

    def update_vertebra_display_options(self, detected_labels: list) -> None:
        """Compatibility alias for the shared vertebra checkbox set."""
        self.update_vertebra_level_checks(detected_labels)

    def _toggle_vertebra_level_checks(self, checked: bool) -> None:
        """Set every detected vertebra checkbox and apply once."""
        self._updating_vertebra_level_checks = True
        for checkbox in self._vertebra_level_checks.values():
            checkbox.setChecked(checked)
        self._updating_vertebra_level_checks = False
        self._on_vertebra_level_selection_changed()

    def _on_vertebra_level_selection_changed(
        self,
        _checked=None,
        *,
        apply_to_view: bool = True,
    ) -> None:
        """Apply one shared level selection to 3D visibility and planning."""
        if self._updating_vertebra_level_checks:
            return
        selected_labels = sorted(
            label_id
            for label_id, checkbox in self._vertebra_level_checks.items()
            if checkbox.isChecked()
        )
        selected_set = set(selected_labels)
        for index in range(self.vertebra_display_list.count()):
            item = self.vertebra_display_list.item(index)
            item.setSelected(
                int(item.data(Qt.ItemDataRole.UserRole)) in selected_set
            )

        has_mask = self._seg_ctrl._last_segmentation_mask_path is not None
        self.auto_screw_plan_btn.setEnabled(has_mask and bool(selected_labels))
        if apply_to_view and self._seg_ctrl._last_vtk_mask is not None:
            self._seg_ctrl.show_vertebrae_by_labels(selected_labels)

    def clear_vertebra_display_options(self) -> None:
        """Clear shared vertebra selection and disable planning."""
        self._clear_vertebra_level_grid()
        self._add_vertebra_level_placeholder()
        self.vertebra_display_list.clear()
        self.vertebra_select_all_btn.setEnabled(False)
        self.vertebra_clear_all_btn.setEnabled(False)
        self.vertebra_show_selected_btn.setEnabled(False)
        self.vertebra_show_all_btn.setEnabled(False)
        self.auto_screw_plan_btn.setEnabled(False)
        # Mirrors the refresh at the end of update_vertebra_level_checks:
        # this is where a cleared segmentation lands, and step 2 must reopen
        # even when the plan button was already disabled (so no
        # EnabledChange event fires to trigger the refresh on its own).
        self._refresh_workflow_bar()

    def _reset_all_vertebra_checkboxes(self) -> None:
        """Compatibility reset for segmentation clear paths."""
        self.clear_vertebra_display_options()

    def _clear_vertebra_level_grid(self) -> None:
        self._vertebra_level_checks.clear()
        while self._vertebra_level_grid.count():
            item = self._vertebra_level_grid.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

    def _add_vertebra_level_placeholder(self) -> None:
        self._vertebra_level_placeholder = QLabel("Run segmentation first")
        self._vertebra_level_placeholder.setObjectName("segmentationStatus")
        self._vertebra_level_grid.addWidget(
            self._vertebra_level_placeholder,
            0,
            0,
            1,
            4,
        )

    def _on_slice_changed(self, plane: str, position: float):
        """Handle slice position change from a viewer."""
        self.viewer_3d._update_plane_positions()

    def reset_workspace(self):
        """Clear all UI state for a fresh DICOM load.

        Called by DicomController after setting a new volume. Delegates
        clearing to each controller so tool/segmentation state is
        consistent.
        """
        self._screw_edit_ctrl.reset()
        self._screw_mpr_ctrl.exit()

        # Clear tools
        self._tool_ctrl.screw_tool.clear_screws()
        self._tool_ctrl.screw_tool.cancel()
        self._tool_ctrl.measurement_tool.clear_measurements()
        self._tool_ctrl.measurement_tool.cancel()
        self._tool_ctrl._screw_actors.clear()
        self._tool_ctrl._measurement_entries = []
        self._tool_ctrl._next_measurement_id = 1
        self._tool_ctrl._active_measure_plane = None
        self._tool_ctrl._editing_measurement_row = None
        self._tool_ctrl._clear_measurement_overlays()

        # Clear viewer overlays
        for viewer in self._get_mpr_viewers():
            viewer.clear_segmentation_mask()
        self.viewer_3d.clear_screws()
        self.viewer_3d.clear_segmentation_mask()

        # Clear list widgets
        self.screw_list_widget.clear()
        self.measurement_list_widget.clear()

        # Reset segmentation state
        self._seg_ctrl.reset_state()

        # Reset auto placement state
        self._auto_placement_ctrl.reset_state()

        # Reset vertebra checkboxes to show all
        self._reset_all_vertebra_checkboxes()

        # DicomController sets the new volume before it calls this method, so
        # until the clearing above has run this window still holds the
        # previous study's mask and screws, and a redraw then would open the
        # new study with step 2 already ticked. This is where the bar is
        # redrawn, now that the study is actually a fresh one.
        self._refresh_workflow_bar()

    def closeEvent(self, event):
        """Handle window close."""
        if self._dicom_ctrl.is_running:
            QMessageBox.warning(
                self,
                "DICOM Loading",
                "DICOM data is still loading. Please wait for completion.",
            )
            event.ignore()
            return

        if self._seg_ctrl.is_running:
            QMessageBox.warning(
                self,
                "Segmentation Running",
                "Auto segmentation is still running. "
                "Please wait for completion.",
            )
            event.ignore()
            return

        if self._auto_placement_ctrl.is_running:
            self._auto_placement_ctrl.request_cancel()
            self.statusbar.showMessage(
                "Cancelling planning — close again once it stops"
            )
            event.ignore()
            return

        # Cleanup viewers
        for viewer in [self.axial_viewer, self.sagittal_viewer,
                       self.coronal_viewer, self.viewer_3d]:
            if viewer:
                viewer.cleanup()

        # The pane-focus filter is installed on the application, which outlives
        # this window; taking it off here keeps a closing window from being
        # asked about events on the way out.
        app = QApplication.instance()
        if app is not None:
            app.removeEventFilter(self)

        self._seg_ctrl.reset_state()

        event.accept()

    # ------------------------------------------------------------------
    # Backward-compatibility shims for tests that call private methods
    # directly on the window object. These thin delegations preserve the
    # existing test API without duplicating logic.
    # ------------------------------------------------------------------

    def _on_dicom_loaded(self, image, metadata, progress):
        """Shim: delegate to DicomController._on_loaded."""
        self._dicom_ctrl._on_loaded(image, metadata, progress)

    def _on_segmentation_finished(self, result):
        """Shim: delegate to SegmentationController._on_finished."""
        self._seg_ctrl._on_finished(result)

    def _update_segmentation_visibility(self, _state=None):
        """Shim: delegate to SegmentationController.update_visibility."""
        self._seg_ctrl.update_visibility(_state)

    def _save_plan_dialog(self):
        """Shim: delegate to PlanController.save_dialog."""
        self._plan_ctrl.save_dialog()

    @property
    def screw_tool(self):
        """Shim: expose tool_ctrl.screw_tool for backward compatibility."""
        return self._tool_ctrl.screw_tool

    @property
    def measurement_tool(self):
        """Shim: expose tool_ctrl.measurement_tool for backward compatibility."""
        return self._tool_ctrl.measurement_tool

    @property
    def _last_segmentation_mask_path(self):
        """Shim: expose seg_ctrl mask path for backward compatibility."""
        return self._seg_ctrl._last_segmentation_mask_path

    @_last_segmentation_mask_path.setter
    def _last_segmentation_mask_path(self, value):
        self._seg_ctrl._last_segmentation_mask_path = value


def main():
    """Application entry point."""
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    settings = app_settings()
    saved_theme = settings.value(
        "appearance/theme",
        DEFAULT_THEME,
        type=str,
    )
    theme_name = saved_theme if saved_theme in THEMES else DEFAULT_THEME
    app.setProperty("themeName", theme_name)
    app.setStyleSheet(load_stylesheet(theme_name))

    window = MainWindow()
    window.show()
    window.raise_()
    window.activateWindow()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
