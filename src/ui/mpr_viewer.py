"""
MPR Viewer - Multiplanar Reconstruction viewer using VTK

Provides axial, sagittal, and coronal slice views with:
- Synchronized crosshairs
- Window/Level adjustment
- Slice navigation
- Coordinate display
"""

import logging
import math
import time
from typing import Callable, Dict, List, Optional, Tuple

import vtk
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QApplication,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ..core.mpr_geometry import (
    orientation_letters,
    project_screw_to_slice,
    slice_to_world,
    world_to_slice,
)
from ..core.volume_manager import VolumeManager
from ..utils.constants import (
    COLOR_AXIAL,
    COLOR_CORONAL,
    COLOR_SAGITTAL,
    DEFAULT_WINDOW_CENTER,
    DEFAULT_WINDOW_WIDTH,
)
from .click_detector import DoubleClickDetector
from .styles import DEFAULT_THEME, theme_rgb_float
from .vtk_widget import create_vtk_widget

logger = logging.getLogger(__name__)

# Orientation markers: 14 px letters kept 6 px inside the viewport edges.
ORIENTATION_MARKER_FONT_SIZE = 14
ORIENTATION_MARKER_MARGIN_PX = 6
ORIENTATION_MARKER_SIDES = ("top", "bottom", "left", "right")


class MPRViewer(QWidget):
    """
    Single MPR plane viewer widget.

    Features:
    - vtkImageReslice for arbitrary plane slicing
    - Window/Level adjustment
    - Crosshair overlay
    - Synchronized scroll with other viewers
    """

    # Signals for cross-viewer synchronization
    slice_changed = pyqtSignal(str, float)  # (plane, position)
    crosshair_moved = pyqtSignal(str, float, float, float)  # (plane, x, y, z)

    ENTRY_HIT_RADIUS_PX = 16.0
    TIP_HIT_RADIUS_PX = 20.0
    SHAFT_HIT_RADIUS_PX = 10.0
    TIP_SHAFT_START_FRACTION = 0.70

    def __init__(
        self,
        plane: str,
        volume_manager: VolumeManager,
        parent: Optional[QWidget] = None
    ):
        """
        Initialize MPR viewer.

        Args:
            plane: 'axial', 'sagittal', or 'coronal'
            volume_manager: Shared VolumeManager instance
            parent: Parent widget
        """
        super().__init__(parent)
        self.setProperty("reviewActive", False)
        self.setSizePolicy(
            QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored
        )

        self.plane = plane
        self.volume_manager = volume_manager

        # Window/Level settings
        self._window_center = DEFAULT_WINDOW_CENTER
        self._window_width = DEFAULT_WINDOW_WIDTH

        # VTK pipeline components
        self._reslice: Optional[vtk.vtkImageReslice] = None
        self._color_map: Optional[vtk.vtkImageMapToWindowLevelColors] = None
        self._actor: Optional[vtk.vtkImageActor] = None
        self._renderer: Optional[vtk.vtkRenderer] = None
        self._crosshair_actors: list = []
        self._measurement_props: Dict[int, List[vtk.vtkProp]] = {}
        self._measurement_data: Dict[int, dict] = {}
        self._measurement_prop_parts: Dict[str, Tuple[int, object]] = {}
        self._selected_measurement_id: Optional[int] = None
        self._measurement_drag_active = False
        self._active_measurement_point: Optional[Tuple[int, int]] = None
        self._measurement_select_callback: Optional[Callable] = None
        self._measurement_drag_callback: Optional[Callable] = None
        self._mask_reslice: Optional[vtk.vtkImageReslice] = None
        # Either vtkImageBinaryThreshold or vtkImageThreshold, whichever the
        # running VTK provides (see _build_segmentation_overlay).
        self._mask_threshold: Optional[vtk.vtkImageAlgorithm] = None
        self._mask_color_map: Optional[vtk.vtkImageMapToColors] = None
        self._mask_actor: Optional[vtk.vtkImageActor] = None
        self._mask_actor_added = False
        self._segmentation_mask_image: Optional[vtk.vtkImageData] = None
        self._segmentation_label_value: int = 0
        self._segmentation_color: Tuple[float, float, float] = (1.0, 0.2, 0.8)
        self._segmentation_opacity: float = 0.35
        self._actor_added = False
        self._crosshairs_added = False
        self._last_hu_value: Optional[float] = None
        self._render_guard_active: bool = False
        self._custom_reslice_axes: Optional[vtk.vtkMatrix4x4] = None

        # Screw projection overlays
        self._screw_overlays: Dict[int, List[vtk.vtkProp]] = {}
        self._screw_data: Dict[int, dict] = {}  # id -> {entry, target, color, diameter}
        self._review_screw_id: Optional[int] = None
        self._screw_prop_parts: Dict[str, Tuple[int, str]] = {}
        self._screw_drag_active = False
        self._screw_drag_begin_callback: Optional[Callable] = None
        self._screw_drag_callback: Optional[Callable] = None
        self._screw_drag_end_callback: Optional[Callable] = None
        self._screw_select_callback: Optional[Callable] = None
        self._double_click_detector = DoubleClickDetector()
        self._active_screw_part: Optional[Tuple[int, str]] = None
        self._pending_screw_press: Optional[
            Tuple[int, str, Tuple[float, float, float]]
        ] = None
        self._mpr_pan_mode_active = False
        self._mpr_pan_drag_active = False
        self._mpr_pan_last_display: Optional[Tuple[int, int]] = None
        self._orientation_actors: Dict[str, vtk.vtkTextActor] = {}

        self._setup_ui()
        self._setup_vtk_pipeline()
        self._setup_interactor()

        # Register with volume manager
        self.volume_manager.add_observer(
            f"mpr_{plane}",
            self._on_volume_event
        )

    def _setup_ui(self):
        """Setup the Qt UI layout."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Header label
        colors = {
            "axial": COLOR_AXIAL,
            "sagittal": COLOR_SAGITTAL,
            "coronal": COLOR_CORONAL,
        }
        color = colors.get(self.plane, (255, 255, 255))
        self.label = QLabel(self.plane.capitalize())
        self.label.setObjectName("viewerHeader")
        self.label.setStyleSheet(
            f"color: rgb({color[0]}, {color[1]}, {color[2]}); "
            f"font-weight: bold; padding: 2px;"
        )

        self.viewport_container = QWidget(self)
        viewport_layout = QGridLayout(self.viewport_container)
        viewport_layout.setContentsMargins(0, 0, 0, 0)
        viewport_layout.setSpacing(0)

        self.vtk_widget = create_vtk_widget(self.viewport_container)
        viewport_layout.addWidget(self.vtk_widget, 0, 0)

        self.mpr_zoom_controls = QWidget(self.viewport_container)
        self.mpr_zoom_controls.setObjectName("mprZoomControls")
        zoom_layout = QHBoxLayout(self.mpr_zoom_controls)
        zoom_layout.setContentsMargins(3, 3, 3, 3)
        zoom_layout.setSpacing(2)
        self.pan_button = QToolButton(self.mpr_zoom_controls)
        self.pan_button.setText("Pan")
        self.pan_button.setToolTip(
            "Move this MPR image: enable Pan, then left-drag"
        )
        self.pan_button.setCheckable(True)
        self.pan_button.setChecked(False)
        self.zoom_out_button = QToolButton(self.mpr_zoom_controls)
        self.zoom_out_button.setText("−")
        self.zoom_out_button.setToolTip("Zoom out this MPR")
        self.zoom_in_button = QToolButton(self.mpr_zoom_controls)
        self.zoom_in_button.setText("+")
        self.zoom_in_button.setToolTip("Zoom in this MPR")
        self.zoom_fit_button = QToolButton(self.mpr_zoom_controls)
        self.zoom_fit_button.setText("Fit")
        self.zoom_fit_button.setToolTip("Fit the CT image to this MPR")
        for button in (
            self.pan_button,
            self.zoom_out_button,
            self.zoom_in_button,
            self.zoom_fit_button,
        ):
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            zoom_layout.addWidget(button)
        self.mpr_zoom_controls.setStyleSheet(
            "QWidget#mprZoomControls {"
            "background: rgba(24, 29, 35, 210);"
            "border: 1px solid rgba(205, 218, 228, 90);"
            "border-radius: 6px; }"
            "QToolButton {"
            "background: transparent; color: #e8eef4; border: none;"
            "min-width: 22px; min-height: 20px; padding: 1px 3px;"
            "font-size: 12px; font-weight: 700; }"
            "QToolButton:hover { background: rgba(70, 82, 94, 220); }"
            "QToolButton:checked {"
            "background: rgba(42, 112, 124, 225); color: white; }"
        )
        self.pan_button.toggled.connect(self.set_mpr_pan_mode)
        self.zoom_out_button.clicked.connect(self.zoom_out)
        self.zoom_in_button.clicked.connect(self.zoom_in)
        self.zoom_fit_button.clicked.connect(lambda: self.fit_to_view())
        viewport_layout.addWidget(
            self.mpr_zoom_controls,
            0,
            0,
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignTop,
        )

        # Info label
        self.info_label = QLabel("Slice: 0.0 mm")
        self.info_label.setObjectName("viewerReadout")
        self.info_label.setStyleSheet("color: white; padding: 2px;")

        layout.addWidget(self.label)
        layout.addWidget(self.viewport_container, stretch=1)
        layout.addWidget(self.info_label)

    def resizeEvent(self, event):
        """Keep the orientation letters pinned to the viewport edges."""
        super().resizeEvent(event)
        self.refresh_orientation_markers()

    def _setup_vtk_pipeline(self):
        """Setup the VTK rendering pipeline."""
        # Create renderer
        self._renderer = vtk.vtkRenderer()
        self._renderer.SetBackground(0.1, 0.1, 0.1)

        # Add renderer to render window
        render_window = self.vtk_widget.GetRenderWindow()
        render_window.AddRenderer(self._renderer)

        # VTK-level render guard: abort any render when guard is active.
        render_window.AddObserver("StartEvent", self._render_guard_callback)

        # Create image actor (will be populated when volume is loaded)
        self._actor = vtk.vtkImageActor()
        self._mask_actor = vtk.vtkImageActor()

        # Create crosshair actors
        self._create_crosshairs()
        self._create_orientation_markers()

    def _setup_interactor(self):
        """Setup interactor for mouse events."""
        interactor = self.vtk_widget.GetRenderWindow().GetInteractor()

        style = vtk.vtkInteractorStyleImage()
        interactor.SetInteractorStyle(style)

        # Scroll events — observer fires when style invokes the event.
        # Zoom vs. slice is decided inside the callback using Qt modifiers
        # (more reliable on macOS than VTK GetControlKey).
        style.AddObserver("MouseWheelForwardEvent", self._on_scroll_forward)
        style.AddObserver("MouseWheelBackwardEvent", self._on_scroll_backward)

        style.AddObserver("LeftButtonPressEvent", self._on_left_click)
        style.AddObserver("LeftButtonReleaseEvent", self._on_left_release)
        style.AddObserver("MouseMoveEvent", self._on_mouse_move)
        style.AddObserver("RightButtonPressEvent", self._on_right_click)
        style.AddObserver("RightButtonReleaseEvent", self._on_right_release)

        self._right_button_down = False
        self._last_mouse_y = 0

    def _render_guard_callback(self, obj, event):
        """VTK render guard — abort render attempts during pipeline setup."""
        if self._render_guard_active:
            obj.SetAbortRender(1)

    def _request_render(self):
        """Request a VTK render."""
        if hasattr(self.vtk_widget, 'safe_render'):
            self.vtk_widget.safe_render()
        else:
            self.vtk_widget.GetRenderWindow().Render()

    def fit_to_view(self, render: bool = True) -> None:
        """Fit the active CT slice tightly inside the current MPR viewport."""
        if self._reslice is None or self._renderer is None:
            return
        self._reslice.Update()
        output = self._reslice.GetOutput()
        if output is None:
            return
        bounds = output.GetBounds()
        image_width = max(float(bounds[1] - bounds[0]), 1e-6)
        image_height = max(float(bounds[3] - bounds[2]), 1e-6)
        viewport_width = max(int(self.vtk_widget.width()), 1)
        viewport_height = max(int(self.vtk_widget.height()), 1)
        viewport_aspect = viewport_width / viewport_height

        center_x = (bounds[0] + bounds[1]) / 2.0
        center_y = (bounds[2] + bounds[3]) / 2.0
        camera = self._renderer.GetActiveCamera()
        camera.ParallelProjectionOn()
        camera.SetFocalPoint(center_x, center_y, 0.0)
        camera.SetPosition(
            center_x,
            center_y,
            max(image_width, image_height) * 2.0,
        )
        camera.SetViewUp(0.0, 1.0, 0.0)
        camera.SetParallelScale(
            0.505 * max(image_height, image_width / viewport_aspect)
        )
        self._renderer.ResetCameraClippingRange(bounds)
        if render:
            self._request_render()

    def _create_crosshairs(self):
        """Create crosshair overlay actors."""
        colors = {
            "axial": [(COLOR_SAGITTAL, "h"), (COLOR_CORONAL, "v")],
            "sagittal": [(COLOR_CORONAL, "h"), (COLOR_AXIAL, "v")],
            "coronal": [(COLOR_SAGITTAL, "h"), (COLOR_AXIAL, "v")],
        }

        for color, orientation in colors.get(self.plane, []):
            line_source = vtk.vtkLineSource()
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(line_source.GetOutputPort())

            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            actor.GetProperty().SetColor(
                color[0] / 255.0, color[1] / 255.0, color[2] / 255.0
            )
            actor.GetProperty().SetLineWidth(1)

            self._crosshair_actors.append({
                "actor": actor,
                "source": line_source,
                "orientation": orientation,
            })

    def _create_orientation_markers(self) -> None:
        """Create the four corner letters that name the patient axes."""
        for side in ORIENTATION_MARKER_SIDES:
            actor = vtk.vtkTextActor()
            actor.SetObjectName(f"orientation-{side}")
            actor.SetInput("")
            actor.SetVisibility(False)
            text_property = actor.GetTextProperty()
            text_property.SetFontFamilyToArial()
            text_property.SetFontSize(ORIENTATION_MARKER_FONT_SIZE)
            text_property.SetBold(True)
            if side == "top":
                text_property.SetJustificationToCentered()
                text_property.SetVerticalJustificationToTop()
            elif side == "bottom":
                text_property.SetJustificationToCentered()
                text_property.SetVerticalJustificationToBottom()
            elif side == "left":
                text_property.SetJustificationToLeft()
                text_property.SetVerticalJustificationToCentered()
            else:
                text_property.SetJustificationToRight()
                text_property.SetVerticalJustificationToCentered()
            text_property.SetColor(*self._orientation_marker_color())
            self._orientation_actors[side] = actor
            if self._renderer is not None:
                self._renderer.AddViewProp(actor)

    def _orientation_marker_color(self) -> Tuple[float, float, float]:
        """Marker colour taken from the running application's theme."""
        app = QApplication.instance()
        theme_name = None if app is None else app.property("themeName")
        return theme_rgb_float(theme_name or DEFAULT_THEME, "viewer_foreground")

    def _orientation_marker_positions(
        self,
        width: int,
        height: int,
    ) -> Dict[str, Tuple[int, int]]:
        """Display positions (VTK pixels, y increasing upwards) per side."""
        margin = ORIENTATION_MARKER_MARGIN_PX
        return {
            "top": (width // 2, max(height - margin, 0)),
            "bottom": (width // 2, margin),
            "left": (margin, height // 2),
            "right": (max(width - margin, 0), height // 2),
        }

    def refresh_orientation_markers(self, render: bool = False) -> None:
        """Re-derive, re-colour and re-place the four orientation letters.

        Markers are hidden whenever there is no slice to label -- before a
        volume is loaded ``_active_reslice_axes`` returns ``None``.
        """
        # Use the instance dict directly: a bare, not-yet-``__init__``-ed
        # QWidget raises RuntimeError (not AttributeError) on ordinary
        # attribute access, which ``getattr(..., default)`` cannot catch.
        actors = self.__dict__.get("_orientation_actors")
        if not actors:
            return
        axes = self._active_reslice_axes()
        letters = {} if axes is None else orientation_letters(axes)
        color = self._orientation_marker_color()
        widget = getattr(self, "vtk_widget", None)
        width = max(int(widget.width()), 1) if widget is not None else 1
        height = max(int(widget.height()), 1) if widget is not None else 1
        positions = self._orientation_marker_positions(width, height)
        for side, actor in actors.items():
            text = letters.get(side, "")
            actor.SetInput(text)
            actor.SetVisibility(bool(text))
            actor.GetTextProperty().SetColor(*color)
            actor.SetDisplayPosition(*positions[side])
        if render:
            self._request_render()

    def update_volume(self):
        """Update the viewer when volume data changes."""
        vtk_image = self.volume_manager.get_vtk_image()
        if vtk_image is None:
            return
        self._last_hu_value = None

        # Block renders during pipeline setup
        self._render_guard_active = True
        self.vtk_widget.setUpdatesEnabled(False)
        logger.info("MPR[%s]: update_volume — render guard ON", self.plane)

        t0 = time.perf_counter()

        # Create reslice filter
        self._reslice = self.volume_manager.create_mpr_reslice(self.plane)

        # Apply window/level
        self._color_map = vtk.vtkImageMapToWindowLevelColors()
        self._color_map.SetInputConnection(self._reslice.GetOutputPort())
        self._color_map.SetWindow(self._window_width)
        self._color_map.SetLevel(self._window_center)
        self._color_map.Update()

        # Update actor - vtkImageActor uses SetInputData, not mapper
        self._actor.SetInputData(self._color_map.GetOutput())
        if not self._actor_added:
            self._renderer.AddActor(self._actor)
            self._actor_added = True

        # Add crosshairs
        if not self._crosshairs_added:
            for ch in self._crosshair_actors:
                self._renderer.AddActor(ch["actor"])
            self._crosshairs_added = True

        self.fit_to_view(render=False)
        self.refresh_orientation_markers()

        # Update display
        self._update_slice_info()

        elapsed = time.perf_counter() - t0
        logger.info("MPR[%s]: update_volume done (%.3fs, render deferred)", self.plane, elapsed)

    def _deferred_initial_render(self):
        """Deferred initial render after volume load."""
        logger.info("MPR[%s]: _deferred_initial_render firing...", self.plane)
        # Lift guard and enable updates before rendering
        self._render_guard_active = False
        self.vtk_widget.setUpdatesEnabled(True)
        t0 = time.perf_counter()
        if hasattr(self.vtk_widget, 'safe_render'):
            self.vtk_widget.safe_render()
        else:
            self.vtk_widget.GetRenderWindow().Render()
        elapsed = time.perf_counter() - t0
        logger.info("MPR[%s]: Render done (%.3fs)", self.plane, elapsed)

    def set_slice_position(self, position: float):
        """
        Set the slice position.

        Args:
            position: World coordinate position for this plane
        """
        self.volume_manager.set_slice_position(self.plane, position)
        self._update_reslice_position()
        self._update_slice_info()
        self._request_render()

    def _update_reslice_position(self):
        """Update reslice axes to current position."""
        if self._reslice is None and self._mask_reslice is None:
            return

        from ..utils.vtk_helpers import create_reslice_axes

        position = self.volume_manager.get_slice_position(self.plane)
        center = self.volume_manager.center

        if self.plane == "axial":
            slice_center = (center[0], center[1], position)
        elif self.plane == "sagittal":
            slice_center = (position, center[1], center[2])
        else:  # coronal
            slice_center = (center[0], position, center[2])

        axes = self._custom_reslice_axes
        if axes is None:
            axes = create_reslice_axes(self.plane, slice_center)
        if self._reslice is not None:
            self._reslice.SetResliceAxes(axes)
            self._reslice.Update()
        if self._mask_reslice is not None:
            self._mask_reslice.SetResliceAxes(axes)
            self._mask_reslice.Update()

        # Propagate reslice changes through downstream filters.
        # Actors use SetInputData (not pipeline connection), so they
        # don't auto-update during render — we must Update() explicitly.
        if self._color_map is not None:
            self._color_map.Update()
        if self._mask_color_map is not None:
            self._mask_color_map.Update()

        # Update screw projections for new slice position
        if self._screw_data:
            self._update_screw_projections()
        self._update_measurement_visibility()

    def _active_reslice_axes(self) -> Optional[vtk.vtkMatrix4x4]:
        """Return the matrix mapping slice-local points into DICOM world space."""
        if self._custom_reslice_axes is not None:
            return self._custom_reslice_axes
        if self._reslice is not None:
            return self._reslice.GetResliceAxes()
        return None

    def _slice_display_to_world(
        self,
        point: Tuple[float, float, float],
    ) -> Tuple[float, float, float]:
        """Convert a picked reslice-display point to DICOM world coordinates."""
        axes = self._active_reslice_axes()
        if axes is None:
            return tuple(float(value) for value in point)
        return slice_to_world(point, axes)

    def _world_to_slice_display(
        self,
        point: Tuple[float, float, float],
    ) -> Tuple[float, float, float]:
        """Convert a DICOM world point to the active reslice display space."""
        axes = self._active_reslice_axes()
        if axes is None:
            return tuple(float(value) for value in point)
        return world_to_slice(point, axes)

    def set_custom_reslice_axes(
        self,
        axes: vtk.vtkMatrix4x4,
        title: str,
    ) -> None:
        """Show an arbitrary MPR plane while preserving world-coordinate data."""
        was_custom = self._custom_reslice_axes is not None
        copied_axes = vtk.vtkMatrix4x4()
        copied_axes.DeepCopy(axes)
        self._custom_reslice_axes = copied_axes
        self._set_review_active(True)
        self.label.setText(title)
        self._update_reslice_position()
        self._update_slice_info()
        self.refresh_orientation_markers()
        if not was_custom:
            self.fit_to_view(render=False)
        self._request_render()

    def clear_custom_reslice_axes(self) -> None:
        """Restore the viewer's standard axial, sagittal, or coronal plane."""
        self._custom_reslice_axes = None
        self._set_review_active(False)
        self.label.setText(self.plane.capitalize())
        self._update_reslice_position()
        self._update_slice_info()
        self.refresh_orientation_markers()
        self.fit_to_view(render=False)
        self._request_render()

    def set_review_screw(self, screw_id: Optional[int]) -> None:
        """Show only one selected screw during Screw MPR."""
        self._review_screw_id = (
            None if screw_id is None else int(screw_id)
        )
        if self._screw_data:
            self._update_screw_projections()

    def set_screw_interaction_callbacks(
        self,
        on_begin: Optional[Callable] = None,
        on_drag: Optional[Callable] = None,
        on_end: Optional[Callable] = None,
        on_select: Optional[Callable] = None,
    ) -> None:
        """Connect direct screw selection and drag callbacks."""
        self._screw_drag_begin_callback = on_begin
        self._screw_drag_callback = on_drag
        self._screw_drag_end_callback = on_end
        self._screw_select_callback = on_select

    def _begin_screw_drag(
        self,
        screw_id: int,
        part: str,
        world_point: Tuple[float, float, float],
    ) -> bool:
        if self._screw_select_callback is not None:
            self._screw_select_callback(int(screw_id))
        accepted = True
        if self._screw_drag_begin_callback is not None:
            accepted = bool(
                self._screw_drag_begin_callback(
                    int(screw_id), part, tuple(world_point)
                )
            )
        self._screw_drag_active = accepted
        self._active_screw_part = (
            (int(screw_id), str(part)) if accepted else None
        )
        self._refresh_active_screw_style()
        return accepted

    def _drag_screw_to(self, world_point: Tuple[float, float, float]) -> None:
        if self._screw_drag_active and self._screw_drag_callback is not None:
            self._screw_drag_callback(
                tuple(world_point), f"{self.plane.capitalize()} MPR"
            )

    def _end_screw_drag(self) -> None:
        if not self._screw_drag_active:
            return
        self._screw_drag_active = False
        self._active_screw_part = None
        self._pending_screw_press = None
        self._refresh_active_screw_style(rebuild=True)
        if self._screw_drag_end_callback is not None:
            self._screw_drag_end_callback()

    def cancel_screw_interaction(self) -> None:
        """Clear a pointer-move lock without changing the last valid geometry."""
        self._screw_drag_active = False
        self._active_screw_part = None
        self._pending_screw_press = None
        self._double_click_detector.reset()
        self._refresh_active_screw_style(rebuild=True)

    @property
    def is_screw_interaction_active(self) -> bool:
        """Whether this viewport currently owns a pointer-move lock."""
        return bool(self._screw_drag_active)

    def _process_screw_press(
        self,
        x: int,
        y: int,
        screw_part: Optional[Tuple[int, str]],
        world_point: Optional[Tuple[float, float, float]],
        timestamp: Optional[float] = None,
    ) -> str:
        """Apply single-select and double-click toggle semantics."""
        is_double = self._double_click_detector.register(
            x, y, timestamp=timestamp
        )
        pending = getattr(self, "_pending_screw_press", None)
        if self._screw_drag_active:
            if is_double:
                self._end_screw_drag()
                return "ended"
            return "locked"
        if is_double and pending is not None:
            self._pending_screw_press = None
            screw_id, part, first_world_point = pending
            if self._begin_screw_drag(
                screw_id, part, first_world_point
            ):
                return "started"
        if screw_part is None or world_point is None:
            self._pending_screw_press = None
            return "navigate"

        screw_id, part = screw_part
        self._pending_screw_press = (
            int(screw_id),
            str(part),
            tuple(float(value) for value in world_point),
        )
        if self._screw_select_callback is not None:
            self._screw_select_callback(int(screw_id))
        return "selected"

    def _set_review_active(self, active: bool) -> None:
        """Refresh dynamic Qt styling for Std/Screw MPR state."""
        self.setProperty("reviewActive", bool(active))
        style = self.style()
        style.unpolish(self)
        style.polish(self)

    def _update_slice_info(self):
        """Update the slice position info label."""
        if self._last_hu_value is None:
            hu_text = "HU: --"
        else:
            hu_text = f"HU: {self._last_hu_value:.0f}"
        if self._custom_reslice_axes is not None:
            self.info_label.setText(f"Screw-aligned | {hu_text}")
        else:
            position = self.volume_manager.get_slice_position(self.plane)
            self.info_label.setText(f"Slice: {position:.1f} mm | {hu_text}")

    def _update_hover_hu(self, x: int, y: int):
        """Update HU display for current cursor location."""
        if self._reslice is None or self._renderer is None:
            return

        picker = vtk.vtkCellPicker()
        picker.Pick(x, y, 0, self._renderer)
        if picker.GetCellId() < 0:
            self._last_hu_value = None
            self._update_slice_info()
            return

        world_pos = self._slice_display_to_world(picker.GetPickPosition())
        self._last_hu_value = self.volume_manager.get_voxel_value(
            world_pos[0], world_pos[1], world_pos[2]
        )
        self._update_slice_info()

    def update_crosshairs(self, x: float, y: float, z: float):
        """
        Update crosshair positions.

        Args:
            x, y, z: World coordinates of crosshair intersection
        """
        if self._reslice is None:
            return
        self._reslice.Update()
        bounds = self._reslice.GetOutput().GetBounds()
        local = self._world_to_slice_display((x, y, z))
        overlay_z = 0.2

        for ch in self._crosshair_actors:
            source = ch["source"]
            if ch["orientation"] == "h":
                source.SetPoint1(bounds[0], local[1], overlay_z)
                source.SetPoint2(bounds[1], local[1], overlay_z)
            else:
                source.SetPoint1(local[0], bounds[2], overlay_z)
                source.SetPoint2(local[0], bounds[3], overlay_z)

        self._request_render()

    def set_window_level(self, window: float, level: float):
        """Set window/level values."""
        self._window_width = window
        self._window_center = level

        # Update existing color map if available
        if hasattr(self, '_color_map') and self._color_map is not None:
            self._color_map.SetWindow(window)
            self._color_map.SetLevel(level)
            self._color_map.Update()
            self._actor.SetInputData(self._color_map.GetOutput())
            self._request_render()
        else:
            self.update_volume()

    def set_segmentation_mask(
        self,
        mask_image: vtk.vtkImageData,
        label_value: int = 0,
        color: Tuple[float, float, float] = (1.0, 0.2, 0.8),
        opacity: float = 0.35,
    ) -> None:
        """Attach a segmentation mask overlay to this MPR view."""
        if mask_image is None:
            self.clear_segmentation_mask()
            return

        self._segmentation_mask_image = mask_image
        self._segmentation_label_value = max(0, int(label_value))
        self._segmentation_color = color
        self._segmentation_opacity = max(0.0, min(1.0, float(opacity)))
        self._build_segmentation_overlay()

    def set_segmentation_label(self, label_value: int) -> None:
        """Update active segmentation label filter (0 means all labels)."""
        self._segmentation_label_value = max(0, int(label_value))
        if self._segmentation_mask_image is None:
            return
        self._build_segmentation_overlay()

    def set_segmentation_visible(self, visible: bool) -> None:
        """Toggle visibility of segmentation overlay."""
        if self._mask_actor is None:
            return
        self._mask_actor.SetVisibility(visible)
        self._request_render()

    def _build_segmentation_overlay(self) -> None:
        """Build or rebuild segmentation overlay pipeline for current label."""
        if self._segmentation_mask_image is None:
            return
        if self._renderer is None or self._mask_actor is None:
            return

        # Match mask reslice output to main CT reslice so overlays align.
        # TotalSegmentator outputs at 1.5mm isotropic — different from CT spacing.
        # We force the mask reslice to use the same output spacing/extent as the
        # main reslice so both actors cover the same pixel grid.
        self._mask_reslice = vtk.vtkImageReslice()
        self._mask_reslice.SetInputData(self._segmentation_mask_image)
        self._mask_reslice.SetOutputDimensionality(2)
        self._mask_reslice.SetInterpolationModeToNearestNeighbor()
        self._mask_reslice.SetBackgroundLevel(0)

        # Copy output spacing/extent from main reslice so overlays align pixel-for-pixel
        if self._reslice is not None:
            self._reslice.Update()
            main_output = self._reslice.GetOutput()
            if main_output is not None:
                sp = main_output.GetSpacing()
                ext = main_output.GetExtent()
                self._mask_reslice.SetOutputSpacing(sp[0], sp[1], sp[2])
                self._mask_reslice.SetOutputExtent(
                    ext[0], ext[1], ext[2], ext[3], ext[4], ext[5]
                )
                origin = main_output.GetOrigin()
                self._mask_reslice.SetOutputOrigin(origin[0], origin[1], origin[2])

        self._update_reslice_position()

        # vtkImageBinaryThreshold (VTK >= 9.7) replaces the deprecated
        # vtkImageThreshold.ThresholdBetween(); see the same fallback in
        # extract_vertebral_mesh (src/core/vertebral_mesh.py).
        if self._segmentation_label_value <= 0:
            lower, upper = 1, 1000000
        else:
            lower = upper = self._segmentation_label_value
        if hasattr(vtk, "vtkImageBinaryThreshold"):
            threshold = vtk.vtkImageBinaryThreshold()
            threshold.SetInputConnection(self._mask_reslice.GetOutputPort())
            threshold.SetLowerThreshold(lower)
            threshold.SetUpperThreshold(upper)
            threshold.SetInValue(1)
            threshold.SetOutValue(0)
            threshold.SetReplaceIn(True)
            threshold.SetReplaceOut(True)
            threshold.SetOutputScalarTypeToUnsignedChar()
        else:
            threshold = vtk.vtkImageThreshold()
            threshold.SetInputConnection(self._mask_reslice.GetOutputPort())
            threshold.ThresholdBetween(lower, upper)
            threshold.SetInValue(1)
            threshold.SetOutValue(0)
            threshold.SetOutputScalarTypeToUnsignedChar()
        threshold.Update()
        self._mask_threshold = threshold

        lut = vtk.vtkLookupTable()
        lut.SetNumberOfTableValues(2)
        lut.SetRange(0, 1)
        lut.SetTableValue(0, 0.0, 0.0, 0.0, 0.0)
        lut.SetTableValue(
            1,
            self._segmentation_color[0],
            self._segmentation_color[1],
            self._segmentation_color[2],
            self._segmentation_opacity,
        )
        lut.Build()

        color_map = vtk.vtkImageMapToColors()
        color_map.SetLookupTable(lut)
        color_map.SetOutputFormatToRGBA()
        color_map.SetInputConnection(threshold.GetOutputPort())
        color_map.Update()
        self._mask_color_map = color_map

        self._mask_actor.SetInputData(color_map.GetOutput())
        # Force mask actor slightly in front of CT to avoid z-fighting
        self._mask_actor.SetPosition(0, 0, 0.01)
        self._mask_actor.ForceTranslucentOn()
        if not self._mask_actor_added:
            self._renderer.AddActor(self._mask_actor)
            self._mask_actor_added = True

        logger.info(
            "MPR[%s]: segmentation overlay built — label=%d, "
            "mask dims=%s, threshold nonzero=%s",
            self.plane,
            self._segmentation_label_value,
            self._segmentation_mask_image.GetDimensions(),
            threshold.GetOutput().GetScalarRange(),
        )
        self._request_render()

    def clear_segmentation_mask(self) -> None:
        """Remove segmentation overlay from this MPR view."""
        had_mask = self._mask_actor_added
        self._segmentation_mask_image = None
        self._mask_reslice = None
        self._mask_threshold = None
        self._mask_color_map = None
        if self._renderer is not None and self._mask_actor is not None and had_mask:
            self._renderer.RemoveActor(self._mask_actor)
            self._mask_actor_added = False
            if not self._render_guard_active:
                self._request_render()

    def add_measurement(
        self,
        measurement_id: int,
        points: list[Tuple[float, float, float]],
        label: str,
        color: Tuple[float, float, float] = (1.0, 0.8, 0.2),
    ) -> None:
        """Add measurement line segments and text label in this MPR view."""
        if not points or len(points) < 2:
            return
        if measurement_id in self._measurement_props:
            self.remove_measurement(measurement_id)

        props: List[vtk.vtkProp] = []
        measurement_prop_parts = self.__dict__.setdefault(
            "_measurement_prop_parts", {}
        )
        display_points = [self._world_to_slice_display(point) for point in points]
        display_points = [(point[0], point[1], 0.2) for point in display_points]

        for index in range(1, len(display_points)):
            line = vtk.vtkLineSource()
            line.SetPoint1(*display_points[index - 1])
            line.SetPoint2(*display_points[index])

            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(line.GetOutputPort())

            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            actor.GetProperty().SetColor(*color)
            actor.GetProperty().SetLineWidth(2)
            actor.GetProperty().SetLighting(False)

            self._renderer.AddViewProp(actor)
            props.append(actor)
            measurement_prop_parts[self._prop_key(actor)] = (
                int(measurement_id),
                "line",
            )

        text_actor = vtk.vtkBillboardTextActor3D()
        text_actor.SetInput(label)
        text_actor.SetPosition(*display_points[-1])
        text_actor.GetTextProperty().SetColor(*color)
        text_actor.GetTextProperty().SetBold(True)
        text_actor.GetTextProperty().SetFontSize(16)
        self._renderer.AddViewProp(text_actor)
        props.append(text_actor)
        measurement_prop_parts[self._prop_key(text_actor)] = (
            int(measurement_id),
            "label",
        )

        for point_index, display_point in enumerate(display_points):
            handle_source = vtk.vtkRegularPolygonSource()
            handle_source.SetNumberOfSides(24)
            handle_source.SetRadius(2.2)
            handle_source.SetCenter(*display_point)
            handle_source.GeneratePolygonOn()
            handle_mapper = vtk.vtkPolyDataMapper()
            handle_mapper.SetInputConnection(handle_source.GetOutputPort())
            handle_actor = vtk.vtkActor()
            handle_actor.SetMapper(handle_mapper)
            handle_actor.SetObjectName(f"measurement-handle-{point_index}")
            handle_actor.GetProperty().SetColor(1.0, 0.92, 0.25)
            handle_actor.GetProperty().SetLighting(False)
            handle_actor.GetProperty().EdgeVisibilityOn()
            handle_actor.GetProperty().SetEdgeColor(0.12, 0.12, 0.12)
            handle_actor.GetProperty().SetLineWidth(1.5)
            handle_actor.SetVisibility(False)
            self._renderer.AddViewProp(handle_actor)
            props.append(handle_actor)
            measurement_prop_parts[self._prop_key(handle_actor)] = (
                int(measurement_id),
                int(point_index),
            )
        self._measurement_props[measurement_id] = props
        axes = self._active_reslice_axes()
        reference_normal = None
        if axes is not None:
            normal = tuple(
                float(axes.GetElement(row, 2)) for row in range(3)
            )
            length = math.sqrt(sum(value * value for value in normal))
            if length > 1e-9:
                reference_normal = tuple(value / length for value in normal)
        self.__dict__.setdefault("_measurement_data", {})[measurement_id] = {
            "points": [tuple(float(value) for value in point) for point in points],
            "normal": reference_normal,
        }
        self._update_measurement_visibility(measurement_id)
        self._request_render()

    def _update_measurement_visibility(
        self,
        measurement_id: Optional[int] = None,
    ) -> None:
        """Show measurements only when their original MPR cut is active."""
        measurement_data = self.__dict__.get("_measurement_data", {})
        measurement_props = self.__dict__.get("_measurement_props", {})
        axes = self._active_reslice_axes()
        if axes is None:
            return

        current_normal = tuple(
            float(axes.GetElement(row, 2)) for row in range(3)
        )
        normal_length = math.sqrt(
            sum(value * value for value in current_normal)
        )
        if normal_length > 1e-9:
            current_normal = tuple(
                value / normal_length for value in current_normal
            )

        spacing = getattr(self.volume_manager, "spacing", (1.0, 1.0, 1.0))
        tolerance = max(min(float(value) for value in spacing) * 0.45, 0.1)
        measurement_ids = (
            [int(measurement_id)]
            if measurement_id is not None
            else list(measurement_data)
        )
        for current_id in measurement_ids:
            data = measurement_data.get(current_id)
            props = measurement_props.get(current_id, [])
            if data is None:
                continue
            reference_normal = data.get("normal")
            same_orientation = True
            if reference_normal is not None and normal_length > 1e-9:
                alignment = abs(sum(
                    reference_normal[index] * current_normal[index]
                    for index in range(3)
                ))
                same_orientation = alignment >= 0.999
            distances = [
                abs(self._world_to_slice_display(point)[2])
                for point in data.get("points", [])
            ]
            visible = bool(
                same_orientation
                and distances
                and max(distances) <= tolerance
            )
            for prop in props:
                part = self.__dict__.get(
                    "_measurement_prop_parts", {}
                ).get(self._prop_key(prop), (current_id, "line"))[1]
                is_handle = isinstance(part, int)
                prop.SetVisibility(
                    visible
                    and (
                        not is_handle
                        or current_id
                        == self.__dict__.get("_selected_measurement_id")
                    )
                )

    def set_measurement_interaction_callbacks(
        self,
        on_select: Optional[Callable] = None,
        on_drag: Optional[Callable] = None,
    ) -> None:
        """Connect PACS-style measurement selection and point dragging."""
        self._measurement_select_callback = on_select
        self._measurement_drag_callback = on_drag

    def set_selected_measurement(
        self,
        measurement_id: Optional[int],
    ) -> None:
        """Highlight one measurement and expose its draggable point handles."""
        self._selected_measurement_id = (
            None if measurement_id is None else int(measurement_id)
        )
        self._update_measurement_visibility()
        for current_id, props in self._measurement_props.items():
            selected = current_id == self._selected_measurement_id
            for prop in props:
                part = self._measurement_prop_parts.get(
                    self._prop_key(prop),
                    (current_id, "line"),
                )[1]
                if part == "line":
                    prop.GetProperty().SetLineWidth(3.5 if selected else 2.0)
        self._request_render()

    def _pick_measurement_part_at_display(
        self,
        x: int,
        y: int,
    ) -> Optional[Tuple[int, object]]:
        """Pick measurement points or lines in display pixels."""
        click = (float(x), float(y))
        endpoint_candidates = []
        line_candidates = []
        for measurement_id, data in self._measurement_data.items():
            props = self._measurement_props.get(measurement_id, [])
            if not any(prop.GetVisibility() for prop in props):
                continue
            points = [
                self._slice_point_to_display(
                    self._world_to_slice_display(point)
                )
                for point in data.get("points", [])
            ]
            for point_index, display_point in enumerate(points):
                distance = math.hypot(
                    click[0] - display_point[0],
                    click[1] - display_point[1],
                )
                if distance <= 12.0:
                    endpoint_candidates.append(
                        (distance, int(measurement_id), int(point_index))
                    )
            for point_index in range(1, len(points)):
                distance = self._distance_to_display_segment(
                    click,
                    points[point_index - 1],
                    points[point_index],
                )
                if distance <= 8.0:
                    line_candidates.append(
                        (distance, int(measurement_id), "line")
                    )
        if endpoint_candidates:
            _, measurement_id, part = min(endpoint_candidates)
            return measurement_id, part
        if line_candidates:
            _, measurement_id, part = min(line_candidates)
            return measurement_id, part
        return None

    def remove_measurement(self, measurement_id: int) -> None:
        """Remove one measurement overlay from this MPR view."""
        props = self._measurement_props.pop(measurement_id, None)
        self.__dict__.get("_measurement_data", {}).pop(measurement_id, None)
        if not props:
            return
        for prop in props:
            self.__dict__.get("_measurement_prop_parts", {}).pop(
                self._prop_key(prop),
                None,
            )
            self._renderer.RemoveViewProp(prop)
        if self.__dict__.get("_selected_measurement_id") == int(measurement_id):
            self._selected_measurement_id = None
        if not self._render_guard_active:
            self._request_render()

    def clear_measurements(self) -> None:
        """Clear all measurement overlays from this MPR view."""
        if not self._measurement_props:
            self.__dict__.get("_measurement_data", {}).clear()
            return
        for props in self._measurement_props.values():
            for prop in props:
                self._renderer.RemoveViewProp(prop)
        self._measurement_props = {}
        self._measurement_data = {}
        self._measurement_prop_parts = {}
        self._selected_measurement_id = None
        if not self._render_guard_active:
            self._request_render()

    # --- Reslice input swap for vertebral-only mode ---

    def set_reslice_input(self, vtk_image: vtk.vtkImageData) -> None:
        """Swap the reslice input to a different volume (e.g. vertebral-only).

        Args:
            vtk_image: Replacement vtkImageData to reslice.
        """
        if self._reslice is None:
            return
        self._reslice.SetInputData(vtk_image)
        self._update_reslice_position()
        if self._color_map is not None:
            self._color_map.Update()
            self._actor.SetInputData(self._color_map.GetOutput())
        self._request_render()

    def restore_original_input(self) -> None:
        """Restore the reslice input to the original CT volume."""
        original = self.volume_manager.get_vtk_image()
        if original is None or self._reslice is None:
            return
        self.set_reslice_input(original)

    # --- Screw projection overlays ---

    def add_screw_overlay(
        self,
        screw_id: int,
        entry: Tuple[float, float, float],
        target: Tuple[float, float, float],
        color: Tuple[float, float, float] = (0.2, 0.8, 0.2),
        diameter: float = 6.0,
    ) -> None:
        """Add a screw projection overlay on this MPR plane.

        The screw trajectory is projected onto the current slice plane
        and redrawn each time the slice position changes.

        Args:
            screw_id: Unique identifier for this screw.
            entry: Entry point in world coordinates (LPS).
            target: Target point in world coordinates (LPS).
            color: RGB color tuple (0-1).
            diameter: Screw diameter in mm (affects line width).
        """
        if screw_id in self._screw_overlays:
            self.remove_screw_overlay(screw_id)
        self._screw_data[screw_id] = {
            "entry": entry,
            "target": target,
            "color": color,
            "diameter": diameter,
        }
        self._draw_screw_projection(screw_id)
        if not self._render_guard_active:
            self._request_render()

    def remove_screw_overlay(self, screw_id: int) -> None:
        """Remove one screw projection overlay."""
        self._screw_data.pop(screw_id, None)
        props = self._screw_overlays.pop(screw_id, None)
        if not props:
            return
        for prop in props:
            self._screw_prop_parts.pop(self._prop_key(prop), None)
            self._renderer.RemoveViewProp(prop)
        if not self._render_guard_active:
            self._request_render()

    def clear_screw_overlays(self) -> None:
        """Remove all screw projection overlays."""
        for props in self._screw_overlays.values():
            for prop in props:
                self._screw_prop_parts.pop(self._prop_key(prop), None)
                self._renderer.RemoveViewProp(prop)
        self._screw_overlays.clear()
        self._screw_data.clear()
        if not self._render_guard_active:
            self._request_render()

    def _update_screw_projections(self) -> None:
        """Recalculate all screw projections for the current slice position."""
        # Remove existing actors
        for props in self._screw_overlays.values():
            for prop in props:
                self._screw_prop_parts.pop(self._prop_key(prop), None)
                self._renderer.RemoveViewProp(prop)
        self._screw_overlays.clear()

        for screw_id in list(self._screw_data.keys()):
            self._draw_screw_projection(screw_id)
        if not self._render_guard_active:
            self._request_render()

    def _draw_screw_projection(self, screw_id: int) -> None:
        """Draw a faint trajectory and the moving screw/slice intersection."""
        data = self._screw_data.get(screw_id)
        if data is None or self._renderer is None:
            return

        entry = data["entry"]
        target = data["target"]
        color = data["color"]
        diameter = data["diameter"]
        props: List[vtk.vtkProp] = []
        axes = self._active_reslice_axes()
        if axes is None:
            self._screw_overlays[screw_id] = props
            return

        projection = project_screw_to_slice(entry, target, axes, diameter)
        overlay_z = 0.2
        p1 = (*projection.entry_slice[:2], overlay_z)
        p2 = (*projection.target_slice[:2], overlay_z)
        local_delta = tuple(
            projection.target_slice[index] - projection.entry_slice[index]
            for index in range(3)
        )
        local_length = math.sqrt(sum(value * value for value in local_delta))
        normal_fraction = (
            abs(local_delta[2]) / local_length if local_length > 1e-9 else 1.0
        )
        custom_long_axis = (
            self._custom_reslice_axes is not None and normal_fraction < 0.15
        )
        if self._review_screw_id is not None:
            if screw_id != self._review_screw_id:
                self._screw_overlays[screw_id] = props
                return
        elif not projection.intersects:
            self._screw_overlays[screw_id] = props
            return

        if custom_long_axis:
            bar_actor = self._create_screw_bar(
                p1,
                p2,
                diameter,
                color,
            )
            self._renderer.AddViewProp(bar_actor)
            props.append(bar_actor)
            self._register_screw_prop(bar_actor, screw_id, "shaft")
        else:
            line = vtk.vtkLineSource()
            line.SetPoint1(*p1)
            line.SetPoint2(*p2)

            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(line.GetOutputPort())

            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            actor.SetObjectName("screw-guide")
            actor.GetProperty().SetColor(*color)
            actor.GetProperty().SetLineWidth(1.5)
            actor.GetProperty().SetOpacity(0.16)
            actor.GetProperty().SetLighting(False)

            self._renderer.AddViewProp(actor)
            props.append(actor)
            self._register_screw_prop(actor, screw_id, "shaft")

        radius = max(float(diameter) / 2.0, 1.0)
        marker_radius = max(1.8, diameter * 0.65)
        endpoint_data = (
            (
                "screw-entry",
                "entry",
                32,
                p1,
                color,
                abs(projection.entry_slice[2]) <= radius,
            ),
            (
                "screw-target",
                "tip",
                3,
                p2,
                (0.65, 0.95, 1.0),
                abs(projection.target_slice[2]) <= radius,
            ),
        )
        for name, part, sides, point, marker_color, near_slice in endpoint_data:
            if not custom_long_axis and not near_slice:
                continue
            marker = vtk.vtkRegularPolygonSource()
            marker.SetNumberOfSides(sides)
            marker.SetRadius(
                marker_radius if part == "entry" else max(1.5, diameter * 0.45)
            )
            marker.SetCenter(*point)
            marker.SetNormal(0, 0, 1)
            marker.GeneratePolygonOn()

            marker_mapper = vtk.vtkPolyDataMapper()
            marker_mapper.SetInputConnection(marker.GetOutputPort())
            marker_actor = vtk.vtkActor()
            marker_actor.SetMapper(marker_mapper)
            marker_actor.SetObjectName(name)
            marker_actor.GetProperty().SetColor(*marker_color)
            marker_actor.GetProperty().SetLineWidth(2.0)
            marker_actor.GetProperty().SetOpacity(0.95)
            marker_actor.GetProperty().SetLighting(False)
            self._renderer.AddViewProp(marker_actor)
            props.append(marker_actor)
            self._register_screw_prop(marker_actor, screw_id, part)

        if projection.intersection_slice is not None and not custom_long_axis:
            projected_length = math.hypot(local_delta[0], local_delta[1])
            if projected_length > 1e-9:
                major_direction = (
                    local_delta[0] / projected_length,
                    local_delta[1] / projected_length,
                )
            else:
                major_direction = (1.0, 0.0)
            minor_direction = (-major_direction[1], major_direction[0])
            semi_major = min(radius / max(normal_fraction, 0.2), radius * 4.0)

            ellipse_points = vtk.vtkPoints()
            ellipse = vtk.vtkPolygon()
            sides = 48
            ellipse.GetPointIds().SetNumberOfIds(sides)
            for index in range(sides):
                angle = 2.0 * math.pi * index / sides
                major_offset = semi_major * math.cos(angle)
                minor_offset = radius * math.sin(angle)
                point_x = (
                    projection.intersection_slice[0]
                    + major_offset * major_direction[0]
                    + minor_offset * minor_direction[0]
                )
                point_y = (
                    projection.intersection_slice[1]
                    + major_offset * major_direction[1]
                    + minor_offset * minor_direction[1]
                )
                ellipse_points.InsertNextPoint(point_x, point_y, overlay_z)
                ellipse.GetPointIds().SetId(index, index)

            ellipse_cells = vtk.vtkCellArray()
            ellipse_cells.InsertNextCell(ellipse)
            ellipse_data = vtk.vtkPolyData()
            ellipse_data.SetPoints(ellipse_points)
            ellipse_data.SetPolys(ellipse_cells)

            intersection_mapper = vtk.vtkPolyDataMapper()
            intersection_mapper.SetInputData(ellipse_data)
            intersection_actor = vtk.vtkActor()
            intersection_actor.SetMapper(intersection_mapper)
            intersection_actor.SetObjectName("screw-section")
            intersection_actor.GetProperty().SetColor(*color)
            intersection_actor.GetProperty().SetOpacity(0.82)
            intersection_actor.GetProperty().SetLighting(False)
            intersection_actor.GetProperty().EdgeVisibilityOn()
            intersection_actor.GetProperty().SetEdgeColor(1.0, 1.0, 1.0)
            intersection_actor.GetProperty().SetLineWidth(2.5)
            self._renderer.AddViewProp(intersection_actor)
            props.append(intersection_actor)
            self._register_screw_prop(intersection_actor, screw_id, "shaft")

        self._screw_overlays[screw_id] = props

    def _register_screw_prop(self, prop, screw_id: int, part: str) -> None:
        """Associate one pickable MPR prop with a model screw and part."""
        self._screw_prop_parts[self._prop_key(prop)] = (
            int(screw_id), str(part)
        )
        prop.PickableOn()
        if self._active_screw_part == (int(screw_id), str(part)):
            prop.GetProperty().SetOpacity(1.0)
            prop.GetProperty().SetLineWidth(4.0)
            prop.GetProperty().EdgeVisibilityOn()
            prop.GetProperty().SetEdgeColor(0.85, 1.0, 1.0)

    def _resolve_screw_pick_part(
        self,
        screw_id: int,
        fallback_part: str,
        pick_position: Tuple[float, float, float],
    ) -> str:
        """Prefer visible endpoint hit zones over an overlapping shaft prop."""
        if fallback_part in {"entry", "tip"}:
            return fallback_part
        data = self._screw_data.get(int(screw_id))
        axes = self._active_reslice_axes()
        if data is None or axes is None:
            return fallback_part

        visible_parts = {
            part
            for prop in self._screw_overlays.get(int(screw_id), [])
            for prop_screw_id, part in [
                self._screw_prop_parts.get(
                    self._prop_key(prop), (-1, "")
                )
            ]
            if prop_screw_id == int(screw_id)
        }
        projection = project_screw_to_slice(
            data["entry"], data["target"], axes, data["diameter"]
        )
        if fallback_part == "shaft":
            fraction = self._display_segment_fraction(
                (float(pick_position[0]), float(pick_position[1])),
                projection.entry_slice[:2],
                projection.target_slice[:2],
            )
            if fraction >= self.TIP_SHAFT_START_FRACTION:
                return "tip"
        threshold = max(2.0, float(data["diameter"]) * 0.75)
        candidates = []
        for part, endpoint in (
            ("entry", projection.entry_slice),
            ("tip", projection.target_slice),
        ):
            if part not in visible_parts:
                continue
            distance = math.hypot(
                float(pick_position[0]) - endpoint[0],
                float(pick_position[1]) - endpoint[1],
            )
            if distance <= threshold:
                candidates.append((distance, part))
        return min(candidates)[1] if candidates else fallback_part

    def _slice_point_to_display(
        self,
        point: Tuple[float, float, float],
    ) -> Tuple[float, float]:
        """Convert a slice-local point to viewport pixel coordinates."""
        self._renderer.SetWorldPoint(
            float(point[0]), float(point[1]), 0.2, 1.0
        )
        self._renderer.WorldToDisplay()
        display = self._renderer.GetDisplayPoint()
        return float(display[0]), float(display[1])

    def _pick_screw_part_at_display(
        self,
        x: int,
        y: int,
    ) -> Optional[Tuple[int, str]]:
        """Resolve screw parts in screen space before CT image picking."""
        axes = self._active_reslice_axes()
        if axes is None:
            return None

        endpoint_candidates = []
        shaft_candidates = []
        click = (float(x), float(y))
        for screw_id, data in self._screw_data.items():
            props = self._screw_overlays.get(int(screw_id), [])
            if not props:
                continue
            visible_parts = {
                part
                for prop in props
                for prop_screw_id, part in [
                    self._screw_prop_parts.get(
                        self._prop_key(prop), (-1, "")
                    )
                ]
                if prop_screw_id == int(screw_id)
            }
            projection = project_screw_to_slice(
                data["entry"], data["target"], axes, data["diameter"]
            )
            entry_display = self._slice_point_to_display(
                projection.entry_slice
            )
            tip_display = self._slice_point_to_display(
                projection.target_slice
            )
            for part, display_point, radius in (
                ("entry", entry_display, self.ENTRY_HIT_RADIUS_PX),
                ("tip", tip_display, self.TIP_HIT_RADIUS_PX),
            ):
                if part not in visible_parts:
                    continue
                distance = math.hypot(
                    click[0] - display_point[0],
                    click[1] - display_point[1],
                )
                if distance <= radius:
                    endpoint_candidates.append(
                        (distance / radius, int(screw_id), part)
                    )
            if "shaft" in visible_parts:
                distance = self._distance_to_display_segment(
                    click, entry_display, tip_display
                )
                if distance <= self.SHAFT_HIT_RADIUS_PX:
                    fraction = self._display_segment_fraction(
                        click, entry_display, tip_display
                    )
                    part = (
                        "tip"
                        if fraction >= self.TIP_SHAFT_START_FRACTION
                        else "shaft"
                    )
                    shaft_candidates.append(
                        (distance, int(screw_id), part)
                    )

        if endpoint_candidates:
            _, screw_id, part = min(endpoint_candidates)
            return screw_id, part
        if shaft_candidates:
            _, screw_id, part = min(shaft_candidates)
            return screw_id, part
        return None

    @staticmethod
    def _display_segment_fraction(
        point: Tuple[float, float],
        start: Tuple[float, float],
        end: Tuple[float, float],
    ) -> float:
        delta_x = end[0] - start[0]
        delta_y = end[1] - start[1]
        length_squared = delta_x * delta_x + delta_y * delta_y
        if length_squared <= 1e-9:
            return 1.0
        fraction = (
            (point[0] - start[0]) * delta_x
            + (point[1] - start[1]) * delta_y
        ) / length_squared
        return min(max(fraction, 0.0), 1.0)

    @staticmethod
    def _distance_to_display_segment(
        point: Tuple[float, float],
        start: Tuple[float, float],
        end: Tuple[float, float],
    ) -> float:
        delta_x = end[0] - start[0]
        delta_y = end[1] - start[1]
        length_squared = delta_x * delta_x + delta_y * delta_y
        if length_squared <= 1e-9:
            return math.hypot(point[0] - start[0], point[1] - start[1])
        fraction = (
            (point[0] - start[0]) * delta_x
            + (point[1] - start[1]) * delta_y
        ) / length_squared
        fraction = min(max(fraction, 0.0), 1.0)
        nearest = (
            start[0] + fraction * delta_x,
            start[1] + fraction * delta_y,
        )
        return math.hypot(point[0] - nearest[0], point[1] - nearest[1])

    def _refresh_active_screw_style(self, rebuild: bool = False) -> None:
        if rebuild and self._screw_data and self._renderer is not None:
            self._update_screw_projections()
            return
        if self._active_screw_part is None:
            return
        for props in self._screw_overlays.values():
            for prop in props:
                if self._screw_prop_parts.get(
                    self._prop_key(prop)
                ) == self._active_screw_part:
                    prop.GetProperty().SetOpacity(1.0)
                    prop.GetProperty().SetLineWidth(4.0)
                    prop.GetProperty().EdgeVisibilityOn()
                    prop.GetProperty().SetEdgeColor(0.85, 1.0, 1.0)

    @staticmethod
    def _prop_key(prop) -> str:
        return prop.GetAddressAsString("")

    def _create_screw_bar(
        self,
        entry_point: Tuple[float, float, float],
        target_point: Tuple[float, float, float],
        diameter: float,
        color: Tuple[float, float, float],
    ) -> vtk.vtkActor:
        """Create a filled diameter-aware bar in slice-local coordinates."""
        delta_x = target_point[0] - entry_point[0]
        delta_y = target_point[1] - entry_point[1]
        projected_length = math.hypot(delta_x, delta_y)
        radius = max(float(diameter) / 2.0, 0.5)

        if projected_length <= 1e-9:
            direction_x, direction_y = 1.0, 0.0
        else:
            direction_x = delta_x / projected_length
            direction_y = delta_y / projected_length
        normal_x = -direction_y * radius
        normal_y = direction_x * radius
        overlay_z = max(entry_point[2], target_point[2])

        coordinates = (
            (entry_point[0] + normal_x, entry_point[1] + normal_y, overlay_z),
            (target_point[0] + normal_x, target_point[1] + normal_y, overlay_z),
            (target_point[0] - normal_x, target_point[1] - normal_y, overlay_z),
            (entry_point[0] - normal_x, entry_point[1] - normal_y, overlay_z),
        )
        points = vtk.vtkPoints()
        polygon = vtk.vtkPolygon()
        polygon.GetPointIds().SetNumberOfIds(4)
        for index, coordinate in enumerate(coordinates):
            points.InsertNextPoint(*coordinate)
            polygon.GetPointIds().SetId(index, index)

        cells = vtk.vtkCellArray()
        cells.InsertNextCell(polygon)
        data = vtk.vtkPolyData()
        data.SetPoints(points)
        data.SetPolys(cells)

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(data)
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.SetObjectName("screw-bar")
        actor.GetProperty().SetColor(*color)
        actor.GetProperty().SetOpacity(0.88)
        actor.GetProperty().SetLighting(False)
        actor.GetProperty().EdgeVisibilityOn()
        actor.GetProperty().SetEdgeColor(1.0, 0.96, 0.65)
        actor.GetProperty().SetLineWidth(1.5)
        return actor

    def _on_volume_event(self, event: str, **kwargs):
        """Handle volume manager events."""
        if event == "volume_loaded":
            self.update_volume()
        elif event == "slice_changed":
            if kwargs.get("plane") == self.plane:
                self._update_reslice_position()
                self._update_slice_info()
                self._request_render()
        elif event == "crosshair_changed":
            pos = kwargs.get("position")
            if pos and kwargs.get("source") != self.plane:
                self.update_crosshairs(*pos)

    def _on_left_click(self, obj, event):
        """Handle left click for crosshair positioning."""
        if self._reslice is None:
            return

        interactor = self.vtk_widget.GetRenderWindow().GetInteractor()
        click_pos = interactor.GetEventPosition()
        self._update_hover_hu(click_pos[0], click_pos[1])

        if self.__dict__.get("_mpr_pan_mode_active", False):
            self._mpr_pan_drag_active = True
            self._mpr_pan_last_display = (int(click_pos[0]), int(click_pos[1]))
            self.vtk_widget.setCursor(Qt.CursorShape.ClosedHandCursor)
            return

        measurement_part = self._pick_measurement_part_at_display(
            click_pos[0], click_pos[1]
        )
        if measurement_part is not None:
            measurement_id, part = measurement_part
            self.set_selected_measurement(measurement_id)
            if self._measurement_select_callback is not None:
                self._measurement_select_callback(int(measurement_id))
            if isinstance(part, int):
                self._measurement_drag_active = True
                self._active_measurement_point = (
                    int(measurement_id), int(part)
                )
                self.vtk_widget.setCursor(Qt.CursorShape.ClosedHandCursor)
            return

        if self._screw_drag_active:
            self._process_screw_press(
                click_pos[0], click_pos[1], None, None
            )
            return

        screen_screw_part = self._pick_screw_part_at_display(
            click_pos[0], click_pos[1]
        )
        if screen_screw_part is not None:
            world_pos = self._pick_world_on_slice(
                click_pos[0], click_pos[1]
            )
            if world_pos is None:
                data = self._screw_data.get(screen_screw_part[0], {})
                if screen_screw_part[1] == "shaft":
                    entry = data.get("entry")
                    target = data.get("target")
                    if entry is not None and target is not None:
                        world_pos = tuple(
                            (entry[index] + target[index]) / 2.0
                            for index in range(3)
                        )
                else:
                    endpoint_key = (
                        "entry"
                        if screen_screw_part[1] == "entry"
                        else "target"
                    )
                    world_pos = data.get(endpoint_key)
            if world_pos is not None:
                self._process_screw_press(
                    click_pos[0],
                    click_pos[1],
                    screen_screw_part,
                    world_pos,
                )
                return

        # Convert display coordinates to world coordinates
        picker = vtk.vtkCellPicker()
        picker.SetTolerance(0.01)
        picker.Pick(click_pos[0], click_pos[1], 0, self._renderer)
        if picker.GetCellId() < 0:
            if getattr(self, "_pending_screw_press", None) is not None:
                self._process_screw_press(
                    click_pos[0], click_pos[1], None, None
                )
            return

        picked_prop = picker.GetViewProp()
        world_pos = self._slice_display_to_world(picker.GetPickPosition())
        screw_part = self._screw_prop_parts.get(self._prop_key(picked_prop))
        if screw_part is not None:
            screw_part = (
                screw_part[0],
                self._resolve_screw_pick_part(
                    screw_part[0],
                    screw_part[1],
                    picker.GetPickPosition(),
                ),
            )
            self._process_screw_press(
                click_pos[0], click_pos[1], screw_part, world_pos
            )
            return
        if getattr(self, "_pending_screw_press", None) is not None:
            result = self._process_screw_press(
                click_pos[0], click_pos[1], None, None
            )
            if result == "started":
                return
        self.volume_manager.set_crosshair_position(
            world_pos[0], world_pos[1], world_pos[2],
            source_plane=self.plane
        )
        self.crosshair_moved.emit(
            self.plane, world_pos[0], world_pos[1], world_pos[2]
        )

    def _on_mouse_move(self, obj, event):
        """Handle mouse move for window/level adjustment."""
        interactor = self.vtk_widget.GetRenderWindow().GetInteractor()
        x, y = interactor.GetEventPosition()
        self._update_hover_hu(x, y)

        if self.__dict__.get("_mpr_pan_drag_active", False):
            previous = self._mpr_pan_last_display
            if previous is not None:
                self._pan_camera_by_pixels(x - previous[0], y - previous[1])
            self._mpr_pan_last_display = (int(x), int(y))
            return

        if self.__dict__.get("_measurement_drag_active", False):
            active_point = self._active_measurement_point
            world_pos = self._pick_world_on_slice(x, y)
            if (
                active_point is not None
                and world_pos is not None
                and self._measurement_drag_callback is not None
            ):
                self._measurement_drag_callback(
                    active_point[0],
                    active_point[1],
                    tuple(world_pos),
                )
            return

        if self._screw_drag_active:
            world_pos = self._pick_world_on_slice(x, y)
            if world_pos is not None:
                self._drag_screw_to(world_pos)
            return

        if self._right_button_down:
            delta = y - self._last_mouse_y
            self._last_mouse_y = y

            self._window_center += delta * 5
            self.set_window_level(self._window_width, self._window_center)

    def _on_left_release(self, obj, event):
        """Release does not end double-click-locked screw movement."""
        if self.__dict__.get("_mpr_pan_drag_active", False):
            self._mpr_pan_drag_active = False
            self._mpr_pan_last_display = None
            self.vtk_widget.setCursor(Qt.CursorShape.OpenHandCursor)
        if self.__dict__.get("_measurement_drag_active", False):
            self._measurement_drag_active = False
            self._active_measurement_point = None
            self.vtk_widget.setCursor(Qt.CursorShape.ArrowCursor)

    def _pick_world_on_slice(
        self, x: int, y: int
    ) -> Optional[Tuple[float, float, float]]:
        """Map a display position onto the active image reslice plane."""
        if self._actor is None:
            return None
        picker = vtk.vtkCellPicker()
        picker.PickFromListOn()
        picker.AddPickList(self._actor)
        if not picker.Pick(x, y, 0, self._renderer):
            return None
        return self._slice_display_to_world(picker.GetPickPosition())

    def _get_scroll_step(self) -> float:
        """Get slice step size in mm for current plane."""
        spacing = self.volume_manager.spacing
        if self.plane == "axial":
            return spacing[2]
        if self.plane == "sagittal":
            return spacing[0]
        return spacing[1]

    def _on_scroll_forward(self, obj, event):
        """Scroll forward: Ctrl/Cmd+Scroll = zoom in, plain = next slice."""
        modifiers = QApplication.keyboardModifiers()
        if modifiers & (Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.MetaModifier):
            self._zoom(1.15)
        elif self._custom_reslice_axes is not None:
            return
        else:
            step = self._get_scroll_step()
            current = self.volume_manager.get_slice_position(self.plane)
            self.set_slice_position(current + step)
            self.slice_changed.emit(self.plane, current + step)

    def _on_scroll_backward(self, obj, event):
        """Scroll backward: Ctrl/Cmd+Scroll = zoom out, plain = prev slice."""
        modifiers = QApplication.keyboardModifiers()
        if modifiers & (Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.MetaModifier):
            self._zoom(0.85)
        elif self._custom_reslice_axes is not None:
            return
        else:
            step = self._get_scroll_step()
            current = self.volume_manager.get_slice_position(self.plane)
            self.set_slice_position(current - step)
            self.slice_changed.emit(self.plane, current - step)

    def _zoom(self, factor: float) -> None:
        """Zoom in/out by adjusting camera zoom factor."""
        if self._renderer is None:
            return
        camera = self._renderer.GetActiveCamera()
        camera.Zoom(factor)
        self._request_render()

    def zoom_in(self) -> None:
        """Zoom into this MPR viewport."""
        self._zoom(1.2)

    def zoom_out(self) -> None:
        """Zoom out of this MPR viewport."""
        self._zoom(1.0 / 1.2)

    def set_mpr_pan_mode(self, active: bool) -> None:
        """Enable or disable grab-and-drag camera movement for this MPR."""
        self._mpr_pan_mode_active = bool(active)
        self._mpr_pan_drag_active = False
        self._mpr_pan_last_display = None
        self.vtk_widget.setCursor(
            Qt.CursorShape.OpenHandCursor
            if self._mpr_pan_mode_active
            else Qt.CursorShape.ArrowCursor
        )

    def _pan_camera_by_pixels(
        self,
        delta_x: float,
        delta_y: float,
        viewport_height: Optional[int] = None,
    ) -> None:
        """Translate the parallel camera so the image follows pointer drag."""
        if self._renderer is None:
            return
        height = max(
            int(viewport_height)
            if viewport_height is not None
            else int(self.vtk_widget.height()),
            1,
        )
        camera = self._renderer.GetActiveCamera()
        world_per_pixel = 2.0 * float(camera.GetParallelScale()) / height
        offset_x = -float(delta_x) * world_per_pixel
        offset_y = -float(delta_y) * world_per_pixel
        focal = camera.GetFocalPoint()
        position = camera.GetPosition()
        camera.SetFocalPoint(
            focal[0] + offset_x,
            focal[1] + offset_y,
            focal[2],
        )
        camera.SetPosition(
            position[0] + offset_x,
            position[1] + offset_y,
            position[2],
        )
        self._request_render()

    def _on_right_click(self, obj, event):
        """Handle right click start for window/level."""
        self._right_button_down = True
        interactor = self.vtk_widget.GetRenderWindow().GetInteractor()
        self._last_mouse_y = interactor.GetEventPosition()[1]

    def _on_right_release(self, obj, event):
        """Handle right click release."""
        self._right_button_down = False

    def start(self):
        """Start the interactor."""
        self.vtk_widget.GetRenderWindow().GetInteractor().Initialize()
        self.vtk_widget.GetRenderWindow().GetInteractor().Start()

    def cleanup(self):
        """Cleanup resources."""
        self.clear_segmentation_mask()
        self.clear_measurements()
        self.volume_manager.remove_observer(f"mpr_{self.plane}")
        self.vtk_widget.GetRenderWindow().Finalize()
