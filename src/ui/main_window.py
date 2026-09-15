"""Main application window with VWORKS-style and MPR-focus layouts."""

import functools
import logging
import sys
from typing import List, Optional

from PyQt6 import sip
from PyQt6.QtCore import QEvent, QObject, QSettings, Qt, QTimer
from PyQt6.QtGui import QAction, QActionGroup
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QSplitter,
    QStatusBar,
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
from ..utils.constants import DEFAULT_SCREW_DIAMETER
from .mpr_viewer import MPRViewer
from .step_panel import (
    StepPanel,
    build_plan_page,
    build_review_page,
    build_segment_page,
    build_study_page,
)
from .styles import (
    DEFAULT_THEME,
    THEME_LABELS,
    THEMES,
    load_stylesheet,
)
from .tool_icons import create_tool_icon
from .viewer_3d import Viewer3D
from .workflow_bar import STEP_NAMES, WorkflowBar, WorkflowStep, workflow_states

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
        #: Which collapsible sections live on which step page, in order.
        #: Filled in by _create_control_panel from step_sections.
        self.step_section_order: dict[str, list[str]] = {}
        self.step_sections: dict[str, tuple] = {}
        #: Tracks the Screw MPR active/inactive edge so refresh_mode_indicators
        #: can switch to the Review step only on the inactive -> active flip.
        self._screw_mpr_was_active = False

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
                # functools.partial binds ``name`` immediately, at each
                # iteration -- a plain ``lambda: self.show_step(name)`` here
                # would instead close over the comprehension's shared ``name``
                # cell, so every button's lambda would call show_step with
                # whatever value ``name`` held last ("Review") once the loop
                # finished, not the value at the iteration that built it.
                WorkflowStep(name, functools.partial(self.show_step, name))
                for name in STEP_NAMES
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
        """Track the pane in use; never consume anything.

        A press or wheel records which pane it landed in (see
        :meth:`_install_focus_tracking`). Workflow-bar refreshes come from
        explicit calls (controller callbacks and table-model signals), not
        from watching other widgets' enabled state.
        """
        event_type = event.type()
        if event_type in self._FOCUS_EVENT_TYPES:
            name = self._pane_name_for(obj)
            if name is not None:
                self._focused_view_name = name
        return super().eventFilter(obj, event)

    def _workflow_running_step(self) -> tuple[bool, Optional[int]]:
        """Which workflow step a background job is running on, if any."""
        dicom = self.__dict__.get("_dicom_ctrl")
        if dicom is not None and bool(
            getattr(dicom, "is_running", False)
        ):
            return True, 0
        seg = self.__dict__.get("_seg_ctrl")
        auto = self.__dict__.get("_auto_placement_ctrl")
        if seg is not None and bool(
            getattr(seg, "is_running", False)
        ):
            return True, 1
        if auto is not None and bool(
            getattr(auto, "is_running", False)
        ):
            return True, 2
        return False, None

    def _refresh_workflow_bar(self) -> None:
        """Redraw the workflow bar from what this session has done so far.

        Reads the state the rest of the window already keeps rather than
        tracking its own: a loaded volume, a TotalSegmentator mask (a
        threshold fallback cannot be planned on, so it does not finish step
        2), selected vertebra levels, auto-planned screws, any screws at
        all, and whether segmentation or planning is running right now.

        This has several entry points -- controller callbacks and
        table-model signals -- so it must be a safe no-op both on a
        half-built window (called before ``workflow_bar`` exists) and on a
        half-destroyed one (called after it has been ``deleteLater``'d but
        before Qt has finished tearing down this window's C++ side).
        ``sip.isdeleted`` catches the latter case, which a plain
        ``hasattr``/``is None`` check cannot: the Python attribute is still
        there, but touching the underlying C++ object raises.
        """
        bar = self.__dict__.get("workflow_bar")
        if bar is None or sip.isdeleted(bar):
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
        checks = self.__dict__.get("_vertebra_level_checks") or {}
        has_screws = False
        widget = self.__dict__.get("screw_list_widget")
        if widget is not None and not sip.isdeleted(widget):
            try:
                has_screws = widget.count() > 0
            except RuntimeError:
                return
        is_running, running_step = self._workflow_running_step()
        bar.set_states(
            workflow_states(
                has_volume=self.volume_manager.get_sitk_image() is not None,
                has_mask=has_mask,
                selected_levels=bool(
                    any(box.isChecked() for box in checks.values())
                ),
                has_plan=has_plan,
                has_screws=has_screws,
                is_running=is_running,
                running_step=running_step,
            )
        )

    def show_step(self, name: str) -> None:
        """Show one page of the right-hand step panel and mark it active."""
        self.step_panel.show_step(name)
        index = STEP_NAMES.index(name)
        self.workflow_bar.set_active(index)

    def _on_seg_advanced_toggled(self, checked: bool) -> None:
        """Show or hide the Segment page's advanced options panel.

        The panel (task/device/quality/label controls and the pedicle
        subregion model settings) starts hidden so the common path -- Run
        Auto Segmentation with defaults -- stays uncluttered; this toggle is
        its only way back into view.
        """
        self.seg_advanced_toggle.setArrowType(
            Qt.ArrowType.DownArrow if checked else Qt.ArrowType.RightArrow
        )
        self.seg_advanced_panel.setVisible(checked)

    def _on_screw_rows_changed(self, *_args) -> None:
        """A screw row appeared or went, so Plan/Review may have finished or reopened.

        Connected to the screw table's rowsInserted/rowsRemoved/modelReset
        rather than kept as a lambda: a bound method uses this window as
        Qt's receiver, so the connection is torn down together with it.  A
        lambda has no receiver, so it can outlive this window during
        teardown -- pytest-qt's deleteLater() on a previous test's window
        can run its DeferredDelete inside a *later* test's event loop,
        after this window's C++ side is already gone.

        Rows also arrive outside of a selection change -- an auto-plan run
        selects the first screw while the table still holds one row, then
        appends the rest -- so the "Screw N of M" counter must be
        re-derived here too, or it is left stuck at "Screw 1 of 1".
        """
        self._refresh_workflow_bar()
        self._refresh_screw_counter()

    @staticmethod
    def _format_screw_counter(current_row: int, count: int) -> str:
        """Format the Review header's screw counter for one shared spot.

        Used both when a screw is selected and when the table's row count
        changes out from under the current selection, so the two paths
        cannot disagree on the wording. Rows can exist with no selection --
        a loaded plan before it selects one, a cleared selection -- and that
        case shows the count rather than claiming there are no screws while
        the table and workflow bar (see :meth:`_refresh_workflow_bar`) say
        otherwise.

        ``count`` is the table's total row count, hidden rows included: the
        counter answers "how many screws exist", while the text filter only
        answers "how many are shown". A filtered-out screw still exists and
        is still deleted/undone with the rest, so the header must not drop
        it from the count.
        """
        if count <= 0:
            return "No screws"
        if current_row < 0:
            return f"{count} screw" if count == 1 else f"{count} screws"
        return f"Screw {current_row + 1} of {count}"

    def _refresh_screw_counter(self) -> None:
        """Re-derive the Review header's screw counter from the table.

        Guarded like :meth:`_refresh_workflow_bar`: this fires from a
        table-model signal, which can arrive on a half-built window (before
        ``screw_list_widget`` exists) or a half-destroyed one (after Qt has
        started tearing down this window's C++ side).
        """
        widget = self.__dict__.get("screw_list_widget")
        counter = self.__dict__.get("selected_screw_counter")
        if (
            widget is None
            or counter is None
            or sip.isdeleted(widget)
            or sip.isdeleted(counter)
        ):
            return
        counter.setText(
            self._format_screw_counter(widget.currentRow(), widget.count())
        )

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

    def _create_control_panel(self) -> QWidget:
        """Create the right-side control panel: a step-based page stack.

        The panel follows the workflow bar (Study/Segment/Plan/Review): each
        step gets its own scrollable page, so the Review page's screw list
        can take most of the panel instead of being squeezed under a pinned
        summary card. Page content is built by step_panel builders; this
        method only installs the refs, registers pages, and handles layout.
        Creates all widgets and stores references; signal connections are
        handled in _connect_signals().
        """
        self.step_panel = StepPanel()

        study_page, study_refs = build_study_page()
        for name, widget in study_refs.items():
            setattr(self, name, widget)
        info_group = study_refs["info_group"]
        wl_group = study_refs["wl_group"]
        view_group = study_refs["view_group"]
        self.step_panel.add_page("Study", study_page)

        segment_page, segment_refs = build_segment_page()
        for name, widget in segment_refs.items():
            setattr(self, name, widget)
        seg_group = segment_refs["seg_group"]
        segment_refs["seg_advanced_toggle"].toggled.connect(
            self._on_seg_advanced_toggled
        )
        self.step_panel.add_page("Segment", segment_page)

        plan_page, plan_refs = build_plan_page(
            self.vertebra_level_container
        )
        for name, widget in plan_refs.items():
            setattr(self, name, widget)
        auto_screw_group = plan_refs["auto_screw_group"]
        plan_params_group = plan_refs["plan_params_group"]
        screw_defaults_group = plan_refs["screw_defaults_group"]
        self._refresh_planner_weight_labels()
        self.step_panel.add_page("Plan", plan_page)

        review_page, review_refs = build_review_page(self._theme_name)
        for name, widget in review_refs.items():
            setattr(self, name, widget)
        details_group = review_refs["details_group"]
        measurements_group = review_refs["measurements_group"]
        self.step_panel.add_page("Review", review_page, scrollable=True)

        self.secondary_control_groups = (
            wl_group,
            view_group,
            plan_params_group,
            screw_defaults_group,
            details_group,
            measurements_group,
        )
        self.step_sections = {
            "Study": (info_group, wl_group, view_group),
            "Segment": (seg_group,),
            "Plan": (auto_screw_group, plan_params_group, screw_defaults_group),
            "Review": (details_group, measurements_group),
        }
        self.step_section_order = {
            step: [group.title for group in groups]
            for step, groups in self.step_sections.items()
        }

        self._register_themed_icon(self.seg_run_btn, "run")
        self._register_themed_icon(self.auto_screw_plan_btn, "run")
        self._register_themed_icon(self.screw_axis_mpr_btn, "run")
        self.seg_clear_btn.setProperty("role", "danger")

        for group in self.secondary_control_groups:
            group.collapse()

        container = QWidget()
        container.setMinimumWidth(390)
        container_layout = QVBoxLayout(container)
        container_layout.setContentsMargins(6, 6, 6, 6)
        container_layout.setSpacing(6)
        container_layout.addWidget(self.step_panel, 1)

        self.show_step("Study")
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
        # Screw MPR's 3D cross-section slice samples the CT with its own
        # window/level, so it needs to hear about a change too -- not just
        # the standard MPR views ViewController already refreshes.
        self.window_slider.valueChanged.connect(
            self._screw_mpr_ctrl.on_window_level_changed
        )
        self.level_slider.valueChanged.connect(
            self._screw_mpr_ctrl.on_window_level_changed
        )
        self.open_dicom_btn.clicked.connect(self._dicom_ctrl.open_folder)
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
        self.screw_filter_edit.textChanged.connect(
            self._on_screw_filter_changed
        )
        self.screw_previous_btn.clicked.connect(
            self._select_previous_screw_guarded
        )
        self.screw_next_btn.clicked.connect(
            self._select_next_screw_guarded
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
        self._screw_edit_move_entry_action.triggered.connect(
            lambda: self._start_screw_edit("entry")
        )
        self._screw_edit_move_tip_action.triggered.connect(
            lambda: self._start_screw_edit("tip")
        )
        self._screw_edit_move_whole_action.triggered.connect(
            lambda: self._start_screw_edit("move")
        )
        self._screw_edit_cancel_action.triggered.connect(
            self._cancel_screw_edit
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
        self.viewer_3d.isolation_requested.connect(
            self._seg_ctrl.set_vertebrae_isolated
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
        # window's children are already destroyed -- and raise
        # ``wrapped C/C++ object ... has been deleted``.
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
            # A viewport pick is a user request to look at that screw.
            self.show_step("Review")

    def _on_screw_filter_changed(self, text: str) -> None:
        """Show only screw rows matching every whitespace-separated word."""
        words = str(text or "").strip().lower().split()
        table = self.screw_list_widget
        for row in range(table.rowCount()):
            haystack = table.rowText(row).lower()
            table.set_row_visible(row, all(word in haystack for word in words))
        current = table.currentRow()
        if current >= 0 and table.isRowHidden(current):
            # Move the selection onto a visible row and drop the hidden
            # rows from the extended selection: they keep no business
            # being deleted, and prev/next never land on them anyway.
            table.clearSelection()
            visible = table.visible_rows()
            if visible:
                table.selectRow(visible[0])
            table.setCurrentRow(visible[0] if visible else -1)

    def _on_screw_visual_selection_changed(self, row: int) -> None:
        """Highlight the list-selected screw in the 3D scene.

        Never switches the step panel: the selection moves on rebuilds,
        filter fixups, and loads too, and only an explicit user pick (a
        viewport pick, prev/next, the Screw MPR entry) means the surgeon
        is now looking at one screw. Those paths call show_step("Review")
        themselves.
        """
        if row >= 0:
            self._active_selection_kind = "screw"
        self.viewer_3d.set_selected_screw(row if row >= 0 else None)
        self._refresh_mode_chip()

    #: Text edits whose keystrokes the Review-list shortcuts must not steal.
    #: QSpinBox/QDoubleSpinBox/QComboBox all host an editable line internally
    #: (their up/down buttons do not take focus), and the custom spin boxes
    #: consume plain digits themselves, so Delete/Ctrl+Z/[/] must reach them.
    _SHORTCUT_TEXT_TYPES = None

    def _review_shortcut_blocked(self) -> bool:
        """Whether a Review-list shortcut must yield to a text edit.

        Delete/Ctrl+Z/[/] are WindowShortcut, so they fire while the focus
        sits in the Review filter or any spin box/combo. Typing there must
        never delete a screw, undo a removal, or move the selection.
        """
        widget = QApplication.focusWidget()
        if widget is None:
            return False
        text_types = self._SHORTCUT_TEXT_TYPES
        if text_types is None:
            from PyQt6.QtWidgets import (
                QComboBox,
                QDoubleSpinBox,
                QLineEdit,
                QSpinBox,
            )

            text_types = self._SHORTCUT_TEXT_TYPES = (
                QLineEdit,
                QSpinBox,
                QDoubleSpinBox,
                QComboBox,
            )
        return isinstance(widget, text_types)

    def _delete_active_selection(self) -> None:
        """Delete the most recently selected screw or measurement."""
        if self._review_shortcut_blocked():
            return
        if (
            self._active_selection_kind == "measurement"
            and self.measurement_list_widget.currentRow() >= 0
        ):
            self._tool_ctrl.remove_selected_measurement()
            return
        self._tool_ctrl.remove_selected_screw()

    def _undo_remove_screws_guarded(self) -> None:
        """Undo a screw removal unless a text edit owns the keystroke."""
        if self._review_shortcut_blocked():
            return
        self._tool_ctrl.undo_remove_screws()

    def _select_previous_screw_guarded(self) -> None:
        """Step to the previous visible screw unless a text edit owns it."""
        if self._review_shortcut_blocked():
            return
        self._pick_screw_step(self._screw_mpr_ctrl.select_previous)

    def _select_next_screw_guarded(self) -> None:
        """Step to the next visible screw unless a text edit owns it."""
        if self._review_shortcut_blocked():
            return
        self._pick_screw_step(self._screw_mpr_ctrl.select_next)

    def _pick_screw_step(self, step) -> None:
        """Run one user-initiated screw step and open its Review page."""
        before = self.screw_list_widget.currentRow()
        step()
        if self.screw_list_widget.currentRow() != before:
            self.show_step("Review")

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

        self._undo_remove_screw_action = QAction("Undo Screw Delete", self)
        self._undo_remove_screw_action.setShortcut("Ctrl+Z")
        self._undo_remove_screw_action.setShortcutContext(
            Qt.ShortcutContext.WindowShortcut
        )
        self._undo_remove_screw_action.triggered.connect(
            self._undo_remove_screws_guarded
        )
        self.addAction(self._undo_remove_screw_action)

        self._screw_prev_action = QAction("Previous Screw", self)
        self._screw_prev_action.setShortcut("[")
        self._screw_prev_action.setShortcutContext(
            Qt.ShortcutContext.WindowShortcut
        )
        self._screw_prev_action.triggered.connect(
            self._select_previous_screw_guarded
        )
        self.addAction(self._screw_prev_action)

        self._screw_next_action = QAction("Next Screw", self)
        self._screw_next_action.setShortcut("]")
        self._screw_next_action.setShortcutContext(
            Qt.ShortcutContext.WindowShortcut
        )
        self._screw_next_action.triggered.connect(
            self._select_next_screw_guarded
        )
        self.addAction(self._screw_next_action)

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
        select_action.setData("navigate")
        select_action.triggered.connect(
            lambda: self._tool_ctrl.set_tool("navigate")
        )
        tools_menu.addAction(select_action)

        screw_action = QAction("Add Screw", self)
        self._register_themed_icon(screw_action, "screw")
        screw_action.setData("screw")
        screw_action.triggered.connect(
            lambda: self._tool_ctrl.set_tool("screw")
        )
        tools_menu.addAction(screw_action)

        distance_action = QAction("Measure Distance", self)
        self._register_themed_icon(distance_action, "distance")
        distance_action.setData("distance")
        distance_action.triggered.connect(
            lambda: self._tool_ctrl.set_tool("distance")
        )
        tools_menu.addAction(distance_action)

        path_action = QAction("Measure Path", self)
        self._register_themed_icon(path_action, "distance")
        path_action.setData("path")
        path_action.triggered.connect(
            lambda: self._tool_ctrl.set_tool("path")
        )
        tools_menu.addAction(path_action)

        angle_action = QAction("Measure Angle", self)
        self._register_themed_icon(angle_action, "angle")
        angle_action.setData("angle")
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
        self._select_tool_action.setData("navigate")
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
        self._add_screw_tool_action.setData("screw")
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
        self._distance_tool_action.setData("distance")
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
        self._angle_tool_action.setData("angle")
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
        """Sync the Screw MPR toggle, the Review action row, and the mode chip."""
        active = bool(self._screw_mpr_ctrl.is_active)
        action = getattr(self, "_screw_mpr_action", None)
        if action is not None:
            if action.isChecked() != active:
                previous = action.blockSignals(True)
                action.setChecked(active)
                action.blockSignals(previous)
            action.setEnabled(active or self.screw_axis_mpr_btn.isEnabled())
        self._sync_tool_actions()
        if hasattr(self, "standard_mpr_btn"):
            self.screw_axis_mpr_btn.setVisible(not active)
            self.standard_mpr_btn.setVisible(active)
        if hasattr(self, "screw_mpr_controls"):
            self.screw_mpr_controls.setVisible(active)
        # Turning Screw MPR on is a request to look at that screw's review
        # data; only the inactive -> active edge should switch pages, so
        # leaving MPR active does not keep yanking the panel back.
        if active and not self._screw_mpr_was_active and hasattr(self, "step_panel"):
            self.show_step("Review")
        self._screw_mpr_was_active = active
        self._refresh_mode_chip()

    def _sync_tool_actions(self) -> None:
        """Check the toolbar/menu action matching the controller's active tool.

        ToolController.set_tool already checks its own action; this covers
        the reverse path (combo-driven "path" mode) so the chip and the
        checked action never disagree about which tool is live. Distance
        stands in for Path in the toolbar, matching its action_map the way
        the Tools menu does with its own Path entry below.
        """
        tool_ctrl = self.__dict__.get("_tool_ctrl")
        if tool_ctrl is None:
            return
        tool = getattr(tool_ctrl, "active_tool", "navigate")
        action_map = {
            "navigate": self.__dict__.get("_select_tool_action"),
            "screw": self.__dict__.get("_add_screw_tool_action"),
            "distance": self.__dict__.get("_distance_tool_action"),
            "path": self.__dict__.get("_distance_tool_action"),
            "angle": self.__dict__.get("_angle_tool_action"),
        }
        action = action_map.get(tool)
        if action is not None and not action.isChecked():
            previous = action.blockSignals(True)
            action.setChecked(True)
            action.blockSignals(previous)

    def _start_screw_edit(self, mode: str) -> None:
        """Start one screw-edit mode, then refresh the mode chip."""
        self._screw_edit_ctrl.start(mode)
        self._refresh_mode_chip()

    def _cancel_screw_edit(self) -> None:
        """Cancel a screw edit, then refresh the mode chip."""
        self._screw_edit_ctrl.cancel()
        self._refresh_mode_chip()

    def _on_screw_visual_selection_changed(self, row: int) -> None:
        """Highlight the list-selected screw in the 3D scene and the mode chip.

        Never switches the step panel: see the list-side handler above. Only
        explicit user picks (viewport pick, prev/next, Screw MPR entry) open
        the Review page, and those call show_step("Review") themselves.
        """
        if row >= 0:
            self._active_selection_kind = "screw"
        self.viewer_3d.set_selected_screw(row if row >= 0 else None)
        self._refresh_mode_chip()

    def current_mode_text(self) -> str:
        """Return the status-bar wording for the active tool or review mode."""
        return self.mode_chip_text()

    def mode_chip_text(self) -> str:
        """One status-bar chip: tool + armed edit + Screw-MPR identity + isolation."""
        tool_ctrl = self.__dict__.get("_tool_ctrl")
        screw_mpr_ctrl = self.__dict__.get("_screw_mpr_ctrl")
        mpr_active = bool(
            getattr(screw_mpr_ctrl, "is_active", False)
        )
        if mpr_active:
            widget = self.__dict__.get("screw_list_widget")
            row = widget.currentRow() if widget is not None else -1
            screws = tool_ctrl.screw_tool.get_screws() if tool_ctrl else []
            if 0 <= row < len(screws):
                screw = screws[row]
                level = str(getattr(screw, "vertebra_level", "") or "").strip()
                side = str(getattr(screw, "side", "") or "").strip()
                identity = " ".join(
                    part for part in (f"#{row + 1}", level, side) if part
                )
                parts = [f"Screw MPR · {identity}"]
            else:
                parts = ["Screw MPR"]
        else:
            tool = getattr(tool_ctrl, "active_tool", "navigate") if tool_ctrl else "navigate"
            parts = [self.TOOL_MODE_LABELS.get(tool, str(tool).capitalize())]
            edit_ctrl = self.__dict__.get("_screw_edit_ctrl")
            edit_mode = getattr(edit_ctrl, "mode", "idle") if edit_ctrl else "idle"
            if edit_mode != "idle":
                parts.append(f"Edit {str(edit_mode).capitalize()}")
        seg_ctrl = self.__dict__.get("_seg_ctrl")
        if seg_ctrl is not None and bool(
            getattr(seg_ctrl, "_vertebrae_isolated", False)
        ):
            parts.append("Isolated")
        return " · ".join(parts)

    def _refresh_mode_chip(self) -> None:
        """Write the mode chip text; safe on half-built or tearing-down windows."""
        label = self.__dict__.get("_mode_label")
        if label is None or sip.isdeleted(label):
            return
        label.setText(self.mode_chip_text())

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
        three of its verdicts are thresholds rather than geometry: the cortical
        clearance worth a note, the width below which a pedicle is narrow, and
        the bound-vs-headline disagreement that sizes from the lower bound.
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
        screw_tool.set_width_bound_disagreement_mm(
            float(self.planner_config().width_bound_disagreement_mm)
        )

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
        reference = str(metrics.get("endplate_reference") or "none")
        reference_levels = str(metrics.get("endplate_reference_levels") or "")
        if isinstance(endplate_angle, (int, float)) and not isinstance(endplate_angle, bool):
            if reference == "neighbours" and reference_levels:
                self.selected_screw_endplate.setText(
                    f"{float(endplate_angle):+.1f}° vs {reference_levels}"
                )
            else:
                self.selected_screw_endplate.setText(f"{float(endplate_angle):+.1f}°")
            tooltip = {
                "own": "Screw angle relative to this level's own upper-endplate fit",
                "neighbours": (
                    "This level's endplate fit is too rough to trust; angle is "
                    f"shown against the nearest well-fitted level(s) ({reference_levels})"
                    if reference_levels
                    else "Angle is shown against neighbouring levels' endplate fit"
                ),
            }.get(
                reference,
                "Screw angle relative to the upper endplate; + is tip-cranial, 0 is parallel",
            )
            self.selected_screw_endplate.setToolTip(tooltip)
        else:
            self.selected_screw_endplate.setText("--")

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
        secondary = str(metrics.get("heary_secondary") or "").strip()
        if heary:
            self.selected_screw_heary.setText(
                f"{heary} + {secondary}" if secondary and secondary != heary else heary
            )

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

    def _set_grade_chip(self, grade: str) -> None:
        """Set the Review page's grade chip text and dynamic ``grade`` property."""
        grade = str(grade).strip().upper()
        token = grade if grade in ("A", "B", "C", "D", "E") else "NA"
        self.selected_screw_grade.setText(f"Grade {grade if grade != 'NA' else '--'}")
        if self.selected_screw_grade.property("grade") != token:
            self.selected_screw_grade.setProperty("grade", token)
            chip_style = self.selected_screw_grade.style()
            chip_style.unpolish(self.selected_screw_grade)
            chip_style.polish(self.selected_screw_grade)

    def update_selected_screw_inspector(
        self,
        index: int,
        screw,
        review_active: bool,
    ) -> None:
        """Update the Review step's key numbers and details for one screw."""
        self.selected_screw_title.setProperty(
            "reviewActive",
            bool(review_active),
        )
        title_style = self.selected_screw_title.style()
        title_style.unpolish(self.selected_screw_title)
        title_style.polish(self.selected_screw_title)
        if screw is None:
            self.selected_screw_counter.setText(
                self._format_screw_counter(-1, self.screw_list_widget.count())
            )
            self.selected_screw_title.setText("No screw selected")
            previous = self.selected_screw_diameter.blockSignals(True)
            self.selected_screw_diameter.setValue(DEFAULT_SCREW_DIAMETER)
            self.selected_screw_diameter.blockSignals(previous)
            self.selected_screw_diameter.setEnabled(False)
            self.selected_screw_length.setText("--")
            self.selected_screw_convergence.setText("--")
            self.selected_screw_craniocaudal.setText("--")
            self._set_grade_chip("NA")
            self.selected_screw_hu.setText("--")
            self.selected_screw_source.setText("--")
            self._clear_screw_metric_rows()
            self.screw_warnings_summary.set_placeholder(
                "Select a screw to inspect its trajectory."
            )
            if hasattr(self, "screw_edit_btn"):
                self.screw_edit_btn.setEnabled(False)
            return

        screw_count = self.screw_list_widget.count()
        self.selected_screw_counter.setText(
            self._format_screw_counter(index, screw_count)
        )
        self.screw_list_widget.scrollToRow(index)

        level = str(getattr(screw, "vertebra_level", "") or "").strip()
        side = str(getattr(screw, "side", "") or "").strip().capitalize()
        identity = " ".join(part for part in (level, side) if part)
        if not identity:
            identity = f"Screw #{index + 1}"

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
        self._set_grade_chip(str(screw.grade))
        if hasattr(self, "screw_edit_btn"):
            self.screw_edit_btn.setEnabled(True)

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

        from src.ui.screw_plan_table import screw_warning_count, screw_warning_lines

        self.screw_warnings_summary.set_lines(
            screw_warning_lines(screw), screw_warning_count(screw)
        )

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
        self._refresh_workflow_bar()

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
        # A cleared segmentation reopens step 2; refresh from session state.
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

        # A fresh study's screws are unrelated to the old one's: an undo
        # after the load must not resurrect the previous study's removals.
        # reset_workspace() itself clears the table above, so clear the
        # tool-controller stack here rather than hiding it inside clear_screws.
        self._tool_ctrl.clear_screw_undo()

        # Reset vertebra checkboxes to show all
        self._reset_all_vertebra_checkboxes()

        # DicomController sets the new volume before it calls this method, so
        # until the clearing above has run this window still holds the
        # previous study's mask and screws, and a redraw then would open the
        # new study with step 2 already ticked. This is where the bar is
        # redrawn, now that the study is actually a fresh one.
        self._refresh_workflow_bar()
        # A study just loaded: point the panel at what comes next.
        self.show_step("Segment")

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
