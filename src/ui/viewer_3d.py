"""
3D Viewer - Volume Rendering

Provides:
- Volume rendering (vtkSmartVolumeMapper, or a CPU fallback on macOS)
- Transfer function presets (Bone, Soft Tissue, CT Angiography, MIP)
- MPR plane indicators
- Screw visualization
- Measurement visualization
- Segmentation overlay
- Interactive rotation/zoom

Why the mapper depends on the platform:
  macOS OpenGL is emulated on Metal. The hardware volume mappers trigger
  glFinish() during 3D texture upload, blocking the main thread for 60-312s.
  No parameter tuning or downsampling avoids this - the bottleneck is the
  OpenGL->Metal translation layer itself (AppleMetalOpenGLRenderer). On macOS
  the fixed-point CPU mapper therefore does all ray casting on CPU and only
  blits a 2D result image to screen (~1 MB), completely bypassing the hang.
  Everywhere else vtkSmartVolumeMapper picks hardware ray casting when a
  usable context exists and falls back to its own CPU caster otherwise.
"""

import logging
import math
import sys
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import vtk
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtWidgets import (
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QSlider,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ..core.vertebral_mesh import MESH_SMOOTHING_ITERATIONS, MESH_SMOOTHING_PASSBAND
from ..core.volume_manager import VolumeManager
from ..core.volume_scale import assess_volume_scale
from ..utils.constants import (
    COLOR_SCREW,
    TRANSFER_FUNCTION_PRESETS,
)
from ..utils.vtk_helpers import (
    MAPPER_KIND_CPU,
    MAPPER_KIND_SMART,
    downsample_vtk_image,
    shrink_factors,
)
from .click_detector import DoubleClickDetector
from .viewer_header import ViewerHeaderLabel
from .vtk_widget import create_vtk_widget

logger = logging.getLogger(__name__)

VIEWPORT_BACKGROUND = (0.035, 0.035, 0.035)
PLANE_INDICATOR_COLORS = {
    "axial": (255, 242, 178),
    "sagittal": (168, 225, 235),
    "coronal": (226, 183, 218),
}
PLANE_INDICATOR_OPACITY = 0.14
PLANE_OUTLINE_OPACITY = 0.48

# vtkSmartVolumeMapper render-mode enum -> log-friendly name.
RENDER_MODE_NAMES = {
    0: "default",
    1: "cpu-raycast",
    2: "gpu",
    3: "ospray",
    4: "anari",
    5: "undefined",
    6: "invalid",
}


def select_volume_mapper_kind(platform: str) -> str:
    """Pick the volume mapper kind for a ``sys.platform`` string.

    macOS keeps the fixed-point CPU ray caster (see the module docstring for
    the Metal glFinish() hang); every other platform gets the smart mapper.
    """
    return MAPPER_KIND_CPU if str(platform) == "darwin" else MAPPER_KIND_SMART


def create_volume_mapper(kind: str) -> vtk.vtkVolumeMapper:
    """Build and configure the volume mapper for ``kind``.

    Raises:
        ValueError: if ``kind`` is not a known mapper kind.
    """
    if kind == MAPPER_KIND_CPU:
        mapper = vtk.vtkFixedPointVolumeRayCastMapper()
        mapper.SetBlendModeToComposite()
        mapper.SetAutoAdjustSampleDistances(False)
        mapper.LockSampleDistanceToInputSpacingOff()
        mapper.SetInteractiveSampleDistance(3.0)
        return mapper
    if kind == MAPPER_KIND_SMART:
        # VTK 9.7's vtkSmartVolumeMapper exposes neither
        # SetInteractiveSampleDistance nor LockSampleDistanceToInputSpacing;
        # the two-phase LOD in _on_interaction_start / _execute_phase2 drives
        # SetSampleDistance directly instead, which both mappers support.
        mapper = vtk.vtkSmartVolumeMapper()
        mapper.SetRequestedRenderModeToDefault()
        mapper.SetBlendModeToComposite()
        mapper.SetAutoAdjustSampleDistances(False)
        mapper.SetInteractiveUpdateRate(1.0 / 15.0)
        return mapper
    raise ValueError(f"unknown mapper kind: {kind!r}")


def describe_render_mode(mapper) -> str:
    """Name the render mode a mapper used on its last render.

    vtkFixedPointVolumeRayCastMapper has no GetLastUsedRenderMode(); it is
    always a CPU fixed-point cast, so say so instead of failing.
    """
    getter = getattr(mapper, "GetLastUsedRenderMode", None)
    if getter is None:
        return "cpu-fixed-point"
    return RENDER_MODE_NAMES.get(int(getter()), "unknown")


# Per-tier target sample distance: (spacing multiplier, absolute floor in mm).
# These reproduce the pre-GPU formulas exactly and do not depend on the mapper.
_SAMPLE_DISTANCE_RULES = {
    "xl": (4.0, 2.0),
    "large": (3.0, 1.5),
    "medium": (2.5, 1.2),
    "small": (2.0, 1.0),
}


def volume_render_settings(
    tier: str,
    spacing: Sequence[float],
    mapper_kind: str,
) -> Tuple[float, Tuple[int, int, int]]:
    """Return the fine sample distance (mm) and shrink factors for a volume.

    Args:
        tier: Tier string from ``assess_volume_scale``.
        spacing: Volume spacing (x, y, z) in millimetres.
        mapper_kind: ``MAPPER_KIND_CPU`` or ``MAPPER_KIND_SMART``.

    Returns:
        ``(sample_distance_mm, (sx, sy, sz))``.

    Raises:
        ValueError: if the tier or the mapper kind is unknown.
    """
    rule = _SAMPLE_DISTANCE_RULES.get(tier)
    if rule is None:
        raise ValueError(f"unknown volume tier: {tier!r}")
    multiplier, floor_mm = rule
    min_spacing = min(float(value) for value in spacing)
    return max(min_spacing * multiplier, floor_mm), shrink_factors(tier, mapper_kind)


@dataclass
class ScrewVisual:
    """Three independently pickable props representing one screw."""

    screw_id: int
    entry_point: Tuple[float, float, float]
    target_point: Tuple[float, float, float]
    entry_actor: vtk.vtkActor
    shaft_actor: vtk.vtkActor
    tip_actor: vtk.vtkActor

    @property
    def props(self) -> Tuple[vtk.vtkActor, vtk.vtkActor, vtk.vtkActor]:
        return (self.entry_actor, self.shaft_actor, self.tip_actor)

    def GetProperty(self):
        """Keep actor-like opacity compatibility for legacy callers."""
        return self.shaft_actor.GetProperty()


def _cylinder_actor(
    start: Tuple[float, float, float],
    end: Tuple[float, float, float],
    radius: float,
) -> vtk.vtkActor:
    delta = tuple(end[index] - start[index] for index in range(3))
    length = math.sqrt(sum(value * value for value in delta))
    direction = tuple(value / length for value in delta)
    center = tuple((start[index] + end[index]) / 2.0 for index in range(3))

    cylinder = vtk.vtkCylinderSource()
    cylinder.SetRadius(radius)
    cylinder.SetHeight(length)
    cylinder.SetResolution(32)
    cylinder.CappingOn()

    transform = vtk.vtkTransform()
    transform.Translate(center)
    y_axis = (0.0, 1.0, 0.0)
    cross = (
        y_axis[1] * direction[2] - y_axis[2] * direction[1],
        y_axis[2] * direction[0] - y_axis[0] * direction[2],
        y_axis[0] * direction[1] - y_axis[1] * direction[0],
    )
    dot = sum(y_axis[index] * direction[index] for index in range(3))
    angle = math.degrees(math.acos(max(-1.0, min(1.0, dot))))
    cross_length = math.sqrt(sum(value * value for value in cross))
    if cross_length > 1e-6:
        transform.RotateWXYZ(
            angle, *(value / cross_length for value in cross)
        )
    elif dot < 0.0:
        transform.RotateX(180.0)

    # vtkTransformFilter (VTK >= 9.7) replaces the deprecated
    # vtkTransformPolyDataFilter. Its declared output type is vtkPointSet,
    # but it mirrors the input type at runtime, so a vtkPolyData input
    # (from vtkCylinderSource here) still yields a vtkPolyData output that
    # vtkPolyDataMapper can consume directly via GetOutputPort().
    transform_filter = (
        vtk.vtkTransformFilter()
        if hasattr(vtk, "vtkTransformFilter")
        else vtk.vtkTransformPolyDataFilter()
    )
    transform_filter.SetInputConnection(cylinder.GetOutputPort())
    transform_filter.SetTransform(transform)
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(transform_filter.GetOutputPort())
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    return actor


def _style_screw_actor(
    actor: vtk.vtkActor,
    color: Tuple[float, float, float],
) -> None:
    actor.GetProperty().SetColor(*color)
    actor.GetProperty().SetOpacity(1.0)
    actor.GetProperty().SetAmbient(0.3)
    actor.GetProperty().SetDiffuse(0.7)
    actor.GetProperty().SetSpecular(0.8)
    actor.GetProperty().SetSpecularPower(40.0)
    actor.GetProperty().EdgeVisibilityOff()
    actor.PickableOn()


def create_screw_visual(
    screw_id: int,
    entry_point: Tuple[float, float, float],
    target_point: Tuple[float, float, float],
    radius: float = 3.0,
    color: Optional[Tuple[float, float, float]] = None,
) -> ScrewVisual:
    """Create a spherical head, cylindrical shaft, and pointed conical tip."""
    entry = tuple(float(value) for value in entry_point)
    target = tuple(float(value) for value in target_point)
    delta = tuple(target[index] - entry[index] for index in range(3))
    length = math.sqrt(sum(value * value for value in delta))
    if length <= 1e-6:
        raise ValueError("Screw length must be greater than zero")
    radius = max(float(radius), 0.25)
    direction = tuple(value / length for value in delta)
    base_color = color or (
        COLOR_SCREW[0] / 255.0,
        COLOR_SCREW[1] / 255.0,
        COLOR_SCREW[2] / 255.0,
    )
    tip_length = min(max(radius * 2.2, 3.0), length * 0.35)
    shaft_end = tuple(
        target[index] - direction[index] * tip_length for index in range(3)
    )

    head = vtk.vtkSphereSource()
    head.SetCenter(*entry)
    head.SetRadius(radius * 1.5)
    head.SetThetaResolution(32)
    head.SetPhiResolution(24)
    head_mapper = vtk.vtkPolyDataMapper()
    head_mapper.SetInputConnection(head.GetOutputPort())
    entry_actor = vtk.vtkActor()
    entry_actor.SetMapper(head_mapper)
    entry_actor.SetObjectName("screw-entry-head")

    shaft_actor = _cylinder_actor(entry, shaft_end, radius)
    shaft_actor.SetObjectName("screw-shaft")

    cone = vtk.vtkConeSource()
    cone.SetCenter(*tuple(
        target[index] - direction[index] * tip_length / 2.0
        for index in range(3)
    ))
    cone.SetDirection(*direction)
    cone.SetHeight(tip_length)
    cone.SetRadius(radius)
    cone.SetResolution(32)
    cone.CappingOn()
    cone_mapper = vtk.vtkPolyDataMapper()
    cone_mapper.SetInputConnection(cone.GetOutputPort())
    tip_actor = vtk.vtkActor()
    tip_actor.SetMapper(cone_mapper)
    tip_actor.SetObjectName("screw-pointed-tip")

    entry_color = tuple(min(1.0, value * 1.12 + 0.05) for value in base_color)
    tip_color = tuple(max(0.0, value * 0.82) for value in base_color)
    _style_screw_actor(entry_actor, entry_color)
    _style_screw_actor(shaft_actor, base_color)
    _style_screw_actor(tip_actor, tip_color)
    return ScrewVisual(
        int(screw_id),
        entry,
        target,
        entry_actor,
        shaft_actor,
        tip_actor,
    )


def intersect_ray_plane(
    near_point: Tuple[float, float, float],
    far_point: Tuple[float, float, float],
    plane_origin: Tuple[float, float, float],
    plane_normal: Tuple[float, float, float],
) -> Optional[Tuple[float, float, float]]:
    """Intersect a display ray with a world-space drag plane."""
    direction = tuple(far_point[index] - near_point[index] for index in range(3))
    denominator = sum(direction[index] * plane_normal[index] for index in range(3))
    if abs(denominator) <= 1e-9:
        return None
    numerator = sum(
        (plane_origin[index] - near_point[index]) * plane_normal[index]
        for index in range(3)
    )
    factor = numerator / denominator
    return tuple(
        near_point[index] + factor * direction[index] for index in range(3)
    )


def _flush_logs():
    """Force flush all log handlers so we see output before a hang."""
    for handler in logging.getLogger().handlers:
        handler.flush()


class Viewer3D(QWidget):
    """
    3D viewer widget for volume rendering.

    The volume mapper is chosen per platform (see ``select_volume_mapper_kind``):
    vtkFixedPointVolumeRayCastMapper (CPU) on macOS, to avoid the OpenGL→Metal
    glFinish() hang, and vtkSmartVolumeMapper (GPU-capable) everywhere else.
    No mesh generation, no background threads — volume is rendered directly
    from vtkImageData with transfer functions.
    """

    header_double_clicked = pyqtSignal(str)  # ("3d") — maximise request

    # Render state: GUARD blocks renders during pipeline setup, NORMAL allows them.
    _RS_GUARD = 0
    _RS_NORMAL = 1
    TIP_SHAFT_START_FRACTION = 0.70
    MIN_VERTEBRAL_SURFACE_OPACITY = 0.20

    @property
    def _render_guard_active(self) -> bool:
        """Backward-compatible guard check (used by screw/measurement methods)."""
        return self._render_state == self._RS_GUARD

    @_render_guard_active.setter
    def _render_guard_active(self, value: bool):
        if value:
            self._render_state = self._RS_GUARD
        elif self._render_state == self._RS_GUARD:
            self._render_state = self._RS_NORMAL

    def __init__(
        self,
        volume_manager: VolumeManager,
        parent: Optional[QWidget] = None
    ):
        super().__init__(parent)
        self.setSizePolicy(
            QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored
        )

        self.volume_manager = volume_manager

        # Volume rendering pipeline
        self._volume: Optional[vtk.vtkVolume] = None
        self._volume_mapper: Optional[vtk.vtkVolumeMapper] = None
        self._mapper_kind: str = select_volume_mapper_kind(sys.platform)
        self._volume_property: Optional[vtk.vtkVolumeProperty] = None
        self._color_tf: Optional[vtk.vtkColorTransferFunction] = None
        self._opacity_tf: Optional[vtk.vtkPiecewiseFunction] = None
        self._gradient_opacity_tf: Optional[vtk.vtkPiecewiseFunction] = None
        self._volume_added: bool = False
        self._render_state: int = self._RS_NORMAL
        self._render_generation: int = 0

        # VTK scene objects
        self._renderer: Optional[vtk.vtkRenderer] = None
        self._plane_actors: dict = {}
        self._plane_actors_added: bool = False
        self._planes_visible: bool = True

        # Screw visualization
        self._screw_actors: List[ScrewVisual] = []
        self._screw_prop_parts: Dict[str, Tuple[int, str]] = {}
        self._selected_screw_id: Optional[int] = None
        self._screw_drag_active = False
        self._screw_drag_plane_origin: Optional[Tuple[float, float, float]] = None
        self._screw_drag_plane_normal: Optional[Tuple[float, float, float]] = None
        self._screw_drag_begin_callback: Optional[Callable] = None
        self._screw_drag_callback: Optional[Callable] = None
        self._screw_drag_end_callback: Optional[Callable] = None
        self._screw_select_callback: Optional[Callable] = None
        self._double_click_detector = DoubleClickDetector()
        self._focus_double_click_detector = DoubleClickDetector()
        self._active_screw_part: Optional[Tuple[int, str]] = None
        self._pan_mode_active = False
        self._pan_forced_shift = False
        self._model_transparency = 0.50
        self._model_opacity = 1.0
        self._vertebral_transparency = 0.50
        self._vertebral_surface_opacity = 0.50

        # Measurement visualization
        self._measurement_props: Dict[int, List[vtk.vtkProp]] = {}

        # Segmentation overlay
        self._segmentation_actor: Optional[vtk.vtkActor] = None
        self._segmentation_mask_image: Optional[vtk.vtkImageData] = None
        self._segmentation_label_value: int = 0
        self._segmentation_color: Tuple[float, float, float] = (0.92, 0.82, 0.70)
        self._segmentation_opacity: float = 0.35
        self._segmentation_default_opacity: float = 0.35

        # Vertebral body mesh overlay
        self._vertebral_mesh_actor: Optional[vtk.vtkActor] = None
        self._vertebral_mesh_default_opacity: float = 0.86

        self._setup_ui()
        self._setup_vtk_pipeline()

        # Register with volume manager
        self.volume_manager.add_observer("viewer_3d", self._on_volume_event)

    def _setup_ui(self):
        """Setup the Qt UI layout."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.label = ViewerHeaderLabel("3D View", "3d", self)
        self.label.setStyleSheet(
            "color: white; font-weight: bold; padding: 2px;"
        )
        self.label.doubleClicked.connect(self.header_double_clicked)

        self.viewport_container = QWidget(self)
        viewport_layout = QGridLayout(self.viewport_container)
        viewport_layout.setContentsMargins(0, 0, 0, 0)
        viewport_layout.setSpacing(0)

        self.vtk_widget = create_vtk_widget(self.viewport_container)
        viewport_layout.addWidget(self.vtk_widget, 0, 0)

        self.plane_visibility_toggle = QToolButton(self.viewport_container)
        self.plane_visibility_toggle.setObjectName("planeVisibilityToggle")
        self.plane_visibility_toggle.setText("Planes On")
        self.plane_visibility_toggle.setToolTip(
            "Show or hide the axial, sagittal, and coronal planes"
        )
        self.plane_visibility_toggle.setCheckable(True)
        self.plane_visibility_toggle.setChecked(True)
        self.plane_visibility_toggle.setCursor(Qt.CursorShape.PointingHandCursor)
        self.plane_visibility_toggle.setStyleSheet(
            "QToolButton#planeVisibilityToggle {"
            "background: rgba(24, 29, 35, 210); color: #e8eef4;"
            "border: 1px solid rgba(205, 218, 228, 90);"
            "border-radius: 7px; padding: 5px 9px;"
            "font-size: 11px; font-weight: 600; }"
            "QToolButton#planeVisibilityToggle:hover {"
            "background: rgba(45, 54, 64, 225); }"
            "QToolButton#planeVisibilityToggle:checked {"
            "border-color: rgba(168, 225, 235, 165); }"
        )
        self.plane_visibility_toggle.toggled.connect(
            self.set_plane_indicators_visible
        )
        viewport_layout.addWidget(
            self.plane_visibility_toggle,
            0,
            0,
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignBottom,
        )

        self.reset_view_button = QToolButton(self.viewport_container)
        self.reset_view_button.setObjectName("resetViewButton")
        self.reset_view_button.setText("Reset View")
        self.reset_view_button.setToolTip(
            "Restore the initial 3D orientation and fit all visible objects"
        )
        self.reset_view_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.reset_view_button.setStyleSheet(
            "QToolButton#resetViewButton {"
            "background: rgba(24, 29, 35, 210); color: #e8eef4;"
            "border: 1px solid rgba(205, 218, 228, 90);"
            "border-radius: 7px; padding: 5px 9px;"
            "font-size: 11px; font-weight: 600; }"
            "QToolButton#resetViewButton:hover {"
            "background: rgba(45, 54, 64, 225); }"
        )
        self.reset_view_button.clicked.connect(self.reset_to_initial_view)
        viewport_layout.addWidget(
            self.reset_view_button,
            0,
            0,
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop,
        )

        self.pan_mode_toggle = QToolButton(self.viewport_container)
        self.pan_mode_toggle.setObjectName("panModeToggle")
        self.pan_mode_toggle.setText("Pan")
        self.pan_mode_toggle.setToolTip(
            "Move the whole 3D model: enable Pan and drag, or use Shift+drag. "
            "Double-click anatomy to center and zoom."
        )
        self.pan_mode_toggle.setCheckable(True)
        self.pan_mode_toggle.setChecked(False)
        self.pan_mode_toggle.setCursor(Qt.CursorShape.PointingHandCursor)
        self.pan_mode_toggle.setStyleSheet(
            "QToolButton#panModeToggle {"
            "background: rgba(24, 29, 35, 210); color: #e8eef4;"
            "border: 1px solid rgba(205, 218, 228, 90);"
            "border-radius: 7px; padding: 5px 9px;"
            "font-size: 11px; font-weight: 600; }"
            "QToolButton#panModeToggle:hover {"
            "background: rgba(45, 54, 64, 225); }"
            "QToolButton#panModeToggle:checked {"
            "background: rgba(42, 112, 124, 225);"
            "border-color: rgba(113, 210, 222, 200); }"
        )
        self.pan_mode_toggle.toggled.connect(self.set_pan_mode)
        viewport_layout.addWidget(
            self.pan_mode_toggle,
            0,
            0,
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignTop,
        )

        self.model_opacity_controls = QWidget(self.viewport_container)
        self.model_opacity_controls.setObjectName("modelOpacityControls")
        opacity_layout = QHBoxLayout(self.model_opacity_controls)
        opacity_layout.setContentsMargins(7, 3, 7, 3)
        opacity_layout.setSpacing(5)
        self.model_opacity_label = QLabel("Vertebra Transparency 50%")
        self.model_opacity_slider = QSlider(
            Qt.Orientation.Horizontal,
            self.viewport_container,
        )
        self.model_opacity_slider.setRange(0, 100)
        self.model_opacity_slider.setValue(50)
        self.model_opacity_slider.setFixedWidth(105)
        self.model_opacity_slider.setToolTip(
            "Adjust vertebral body transparency: 0% hides internal screws, "
            "100% reveals them while keeping the body visible"
        )
        opacity_layout.addWidget(self.model_opacity_label)
        opacity_layout.addWidget(self.model_opacity_slider)
        self.model_opacity_controls.setStyleSheet(
            "QWidget#modelOpacityControls {"
            "background: rgba(24, 29, 35, 210);"
            "border: 1px solid rgba(205, 218, 228, 90);"
            "border-radius: 7px; }"
            "QLabel { color: #e8eef4; font-size: 11px; font-weight: 600; }"
        )
        self.model_opacity_slider.valueChanged.connect(
            lambda value: self.set_vertebral_transparency(value / 100.0)
        )
        viewport_layout.addWidget(
            self.model_opacity_controls,
            0,
            0,
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignBottom,
        )

        layout.addWidget(self.label)
        layout.addWidget(self.viewport_container, stretch=1)

    def _build_volume_mapper(self) -> vtk.vtkVolumeMapper:
        """Select and configure the volume mapper for the current platform."""
        self._mapper_kind = select_volume_mapper_kind(sys.platform)
        mapper = create_volume_mapper(self._mapper_kind)
        logger.info(
            "_setup_vtk_pipeline: volume mapper=%s (kind=%s, platform=%s)",
            type(mapper).__name__,
            self._mapper_kind,
            sys.platform,
        )
        return mapper

    def _setup_vtk_pipeline(self):
        """Setup the VTK volume rendering pipeline (mapper chosen per platform)."""
        logger.info("_setup_vtk_pipeline: initializing volume renderer")
        # Flat neutral background keeps anatomy and implant colors dominant.
        self._renderer = vtk.vtkRenderer()
        self._renderer.SetBackground(*VIEWPORT_BACKGROUND)
        self._renderer.GradientBackgroundOff()

        render_window = self.vtk_widget.GetRenderWindow()
        render_window.AddRenderer(self._renderer)
        render_window.SetMultiSamples(0)  # Disable MSAA

        # VTK-level render guard: abort any render when guard is active.
        render_window.AddObserver("StartEvent", self._render_guard_callback)

        # Trackball interactor with LOD update rates
        interactor = render_window.GetInteractor()
        style = vtk.vtkInteractorStyleTrackballCamera()
        interactor.SetInteractorStyle(style)
        interactor.SetDesiredUpdateRate(15.0)
        interactor.SetStillUpdateRate(2.0)

        # Manual LOD: coarsen during interaction, refine after
        interactor.AddObserver("StartInteractionEvent", self._on_interaction_start)
        interactor.AddObserver("EndInteractionEvent", self._on_interaction_end)
        interactor.AddObserver("LeftButtonPressEvent", self._on_screw_left_press, 1.0)
        interactor.AddObserver("MouseMoveEvent", self._on_screw_mouse_move, 1.0)
        interactor.AddObserver("LeftButtonReleaseEvent", self._on_screw_left_release, 1.0)

        self._volume_mapper = self._build_volume_mapper()

        self._color_tf = vtk.vtkColorTransferFunction()
        self._opacity_tf = vtk.vtkPiecewiseFunction()
        self._gradient_opacity_tf = vtk.vtkPiecewiseFunction()

        self._volume_property = vtk.vtkVolumeProperty()
        self._volume_property.SetColor(self._color_tf)
        self._volume_property.SetScalarOpacity(self._opacity_tf)
        self._volume_property.SetGradientOpacity(self._gradient_opacity_tf)
        self._volume_property.SetInterpolationTypeToLinear()

        self._volume = vtk.vtkVolume()
        self._volume.SetMapper(self._volume_mapper)
        self._volume.SetProperty(self._volume_property)

        # Slice plane indicators
        self._create_plane_indicators()

        # Axes widget
        self._add_axes_widget()

    def _render_guard_callback(self, obj, event):
        """VTK StartEvent callback — abort renders during pipeline setup."""
        if self._render_state == self._RS_GUARD:
            obj.SetAbortRender(1)

    def _on_interaction_start(self, obj, event):
        """Use coarse sampling during interaction for responsive rotation/zoom."""
        if self._volume_mapper and hasattr(self, '_target_sample_dist'):
            coarse = max(self._target_sample_dist * 3.0, 3.0)
            self._volume_mapper.SetSampleDistance(coarse)

    def _on_interaction_end(self, obj, event):
        """Restore fine sampling after interaction ends."""
        if self._volume_mapper and hasattr(self, '_target_sample_dist'):
            self._volume_mapper.SetSampleDistance(self._target_sample_dist)
            self._request_render()

    def zoom_camera(self, factor: float) -> None:
        """Zoom the 3D camera; factors above one zoom in."""
        if factor <= 0:
            raise ValueError("zoom factor must be positive")
        if self._renderer is None:
            return
        self._renderer.GetActiveCamera().Dolly(float(factor))
        self._renderer.ResetCameraClippingRange()
        self._request_render()

    def fit_to_view(self) -> None:
        """Fit all visible 3D props inside the viewport."""
        if self._renderer is None:
            return
        self._renderer.ResetCamera()
        self._renderer.ResetCameraClippingRange()
        self._request_render()

    def _orient_camera_to_initial(self) -> None:
        """Set the sagittal startup direction before fitting scene bounds."""
        if self._renderer is None:
            return
        camera = self._renderer.GetActiveCamera()
        camera.SetFocalPoint(0.0, 0.0, 0.0)
        camera.SetPosition(1.0, 0.0, 0.0)
        camera.SetViewUp(0.0, 0.0, 1.0)
        camera.OrthogonalizeViewUp()

    def reset_to_initial_view(self) -> None:
        """Restore the startup orientation and fit all visible 3D props."""
        if self._renderer is None:
            return
        self._orient_camera_to_initial()
        self._renderer.ResetCamera()
        self._renderer.ResetCameraClippingRange()
        pan_toggle = self.__dict__.get("pan_mode_toggle")
        if pan_toggle is not None:
            pan_toggle.setChecked(False)
        self._request_render()

    def set_pan_mode(self, active: bool) -> None:
        """Use left drag as camera pan for trackpad-friendly navigation."""
        self._pan_mode_active = bool(active)
        cursor = (
            Qt.CursorShape.OpenHandCursor
            if self._pan_mode_active
            else Qt.CursorShape.ArrowCursor
        )
        if hasattr(self, "vtk_widget"):
            self.vtk_widget.setCursor(cursor)

    def focus_on_world_point(
        self,
        point: Tuple[float, float, float],
        zoom_factor: float = 1.6,
    ) -> None:
        """Center a picked anatomy point and zoom without changing view angle."""
        if self._renderer is None:
            return
        zoom_factor = max(float(zoom_factor), 1.0)
        camera = self._renderer.GetActiveCamera()
        old_focal = camera.GetFocalPoint()
        old_position = camera.GetPosition()
        offset = tuple(
            old_position[index] - old_focal[index]
            for index in range(3)
        )
        target = tuple(float(value) for value in point)
        camera.SetFocalPoint(*target)
        camera.SetPosition(*tuple(
            target[index] + offset[index] / zoom_factor
            for index in range(3)
        ))
        self._renderer.ResetCameraClippingRange()
        self._request_render()

    def focus_at_display(self, x: int, y: int) -> bool:
        """Focus and zoom on visible anatomy under a display coordinate."""
        picker = vtk.vtkCellPicker()
        picker.SetTolerance(0.005)
        picker.PickFromListOn()
        for prop in (
            self._vertebral_mesh_actor,
            self._segmentation_actor,
        ):
            if prop is not None and prop.GetVisibility():
                picker.AddPickList(prop)
        if picker.GetPickList().GetNumberOfItems() == 0:
            return False
        if not picker.Pick(int(x), int(y), 0, self._renderer):
            return False
        self.focus_on_world_point(picker.GetPickPosition())
        return True

    # --- Volume Rendering ---

    def update_volume(self):
        """
        Update the viewer when volume data changes.

        Uses downsampled volume + CPU renderer for fast, hang-free rendering.

        Strategy:
        - Phase 1: Downsampled volume + coarse sampling → fast first frame
        - Phase 2: Same downsampled volume + fine sampling → improved quality
          (scheduled via QTimer, 500ms after Phase 1)
        - Full-resolution stays in VolumeManager for MPR views only.
        """
        t_total = time.perf_counter()
        vtk_image = self.volume_manager.get_vtk_image()
        if vtk_image is None:
            return

        # Block ALL renders during pipeline setup
        self._render_guard_active = True
        self._render_generation += 1
        self.vtk_widget.setUpdatesEnabled(False)
        logger.info("update_volume: render guard ON, gen=%d, widget updates DISABLED",
                     self._render_generation)

        # Quality control per volume tier
        dims = self.volume_manager.dimensions
        spacing = self.volume_manager.spacing
        assessment = assess_volume_scale(dims)
        sample_dist, shrink = volume_render_settings(
            assessment.tier, spacing, self._mapper_kind
        )
        logger.info(
            "  dims=%s spacing=%s tier=%s mapper=%s shrink=%s sample_dist=%.2f",
            dims, spacing, assessment.tier, self._mapper_kind, shrink, sample_dist,
        )

        self._target_sample_dist = sample_dist

        # Downsampled volume per the mapper's tier table
        t0 = time.perf_counter()
        self._downsampled_image = downsample_vtk_image(vtk_image, shrink)
        ds_dims = self._downsampled_image.GetDimensions()
        logger.info(
            "  downsample: shrink=%s → dims=%s  (%.3fs)",
            shrink, ds_dims, time.perf_counter() - t0,
        )

        t0 = time.perf_counter()
        self._volume_mapper.SetInputData(self._downsampled_image)
        logger.info("  SetInputData: %.3fs", time.perf_counter() - t0)

        # Phase 1: Coarse sampling for fast first frame
        self._volume_mapper.SetSampleDistance(max(sample_dist * 2.0, 2.0))
        logger.info(
            "  sample_dist=%.2f (phase1=%.2f)",
            sample_dist, max(sample_dist * 2.0, 2.0),
        )

        # Apply current transfer function preset
        preset_name = self.volume_manager.transfer_function_preset
        logger.info("  applying preset '%s'...", preset_name)
        self.apply_transfer_function_preset(preset_name)
        logger.info("  preset applied")

        t0 = time.perf_counter()
        if not self._volume_added:
            logger.info("  AddVolume()...")
            self._renderer.AddVolume(self._volume)
            self._volume_added = True
            logger.info("  AddVolume() done: %.3fs", time.perf_counter() - t0)

        # Update plane indicators
        t0 = time.perf_counter()
        self._update_plane_positions()
        if not self._plane_actors_added:
            for plane_data in self._plane_actors.values():
                self._renderer.AddActor(plane_data["actor"])
                self._renderer.AddActor(plane_data["outline_actor"])
            self._plane_actors_added = True
        logger.info("  plane indicators: %.3fs", time.perf_counter() - t0)

        t0 = time.perf_counter()
        logger.info("  ResetCamera()...")
        self._orient_camera_to_initial()
        self._renderer.ResetCamera()
        logger.info("  ResetCamera() done: %.3fs", time.perf_counter() - t0)

        total_sec = time.perf_counter() - t_total
        logger.info("update_volume total: %.3fs (render deferred)", total_sec)
        _flush_logs()

    def _deferred_render_phase1(self):
        """Phase 1: Deferred coarse render on a clean event loop.

        Called by _coordinated_initial_render after render guard is lifted.
        Renders with coarse sampling for a fast first frame, then schedules
        Phase 2 (fine sampling) via QTimer.
        """
        if not self._volume_added or not hasattr(self, '_target_sample_dist'):
            self._render_state = self._RS_NORMAL
            self.vtk_widget.setUpdatesEnabled(True)
            return

        # Coarse sampling for fast first frame
        coarse_dist = max(self._target_sample_dist * 2.0, 2.0)
        self._volume_mapper.SetSampleDistance(coarse_dist)

        logger.info("  Phase 1 render (sample_dist=%.2f)...", coarse_dist)
        _flush_logs()

        # Lift guard and enable widget updates before rendering
        self._render_state = self._RS_NORMAL
        self.vtk_widget.setUpdatesEnabled(True)

        t0 = time.perf_counter()
        if hasattr(self.vtk_widget, 'safe_render'):
            self.vtk_widget.safe_render()
        else:
            self.vtk_widget.GetRenderWindow().Render()
        render_sec = time.perf_counter() - t0
        logger.info(
            "  Phase 1 done: %.3fs (mapper=%s, render mode=%s)",
            render_sec,
            self._mapper_kind,
            describe_render_mode(self._volume_mapper),
        )

        # Schedule Phase 2 via QTimer (works now — no Cocoa event loop starvation)
        gen = self._render_generation
        QTimer.singleShot(500, lambda: self._execute_phase2(gen))
        logger.info("  Phase 2 timer started (gen=%d, 0.5s)", gen)

    def _execute_phase2(self, generation: int):
        """Phase 2: Fine quality render after delay.

        Called by QTimer 500ms after Phase 1 completes.
        Sets fine sampling distance and re-renders.
        """
        if generation != self._render_generation:
            logger.info("  Phase 2 STALE (gen=%d, current=%d)",
                        generation, self._render_generation)
            return
        if hasattr(self, '_target_sample_dist'):
            self._volume_mapper.SetSampleDistance(self._target_sample_dist)
            logger.info("  Phase 2 render (sample_dist=%.2f)",
                        self._target_sample_dist)
        self._request_render()
        logger.info("  Phase 2 done")

    def _request_render(self):
        """Request a VTK render."""
        if hasattr(self.vtk_widget, 'safe_render'):
            self.vtk_widget.safe_render()
        else:
            self.vtk_widget.GetRenderWindow().Render()

    def apply_transfer_function_preset(self, preset_name: str) -> None:
        """
        Apply a transfer function preset to the volume rendering.

        This is instant — no recomputation needed.
        """
        if preset_name not in TRANSFER_FUNCTION_PRESETS:
            return

        config = TRANSFER_FUNCTION_PRESETS[preset_name]

        # Color transfer function
        self._color_tf.RemoveAllPoints()
        for hu, r, g, b in config["color_points"]:
            self._color_tf.AddRGBPoint(hu, r, g, b)

        # Scalar opacity
        model_opacity = self._model_opacity
        self._opacity_tf.RemoveAllPoints()
        for hu, opacity in config["opacity_points"]:
            self._opacity_tf.AddPoint(hu, opacity * model_opacity)

        # Gradient opacity
        self._gradient_opacity_tf.RemoveAllPoints()
        for grad_mag, opacity in config["gradient_opacity_points"]:
            self._gradient_opacity_tf.AddPoint(grad_mag, opacity)

        # Shading parameters
        if config["shade"]:
            self._volume_property.ShadeOn()
        else:
            self._volume_property.ShadeOff()

        self._volume_property.SetAmbient(config["ambient"])
        self._volume_property.SetDiffuse(config["diffuse"])
        self._volume_property.SetSpecular(config["specular"])
        self._volume_property.SetSpecularPower(config["specular_power"])

        # Blend mode
        if config["blend_mode"] == "maximum_intensity":
            self._volume_mapper.SetBlendModeToMaximumIntensity()
        else:
            self._volume_mapper.SetBlendModeToComposite()

        if self._volume_added:
            self._request_render()

    def set_volume_visible(self, visible: bool) -> None:
        """Toggle visibility of the volume rendering actor.

        Vertebral mesh and screw actors remain visible regardless.

        Args:
            visible: True to show volume rendering, False to hide.
        """
        if self._volume is not None:
            self._volume.SetVisibility(int(visible))
        # Also toggle segmentation overlay if present
        if self._segmentation_actor is not None:
            self._segmentation_actor.SetVisibility(int(visible))
        if self._volume_added:
            self._request_render()

    def set_volume_opacity(self, opacity: float) -> None:
        """Scale the overall volume opacity (0.0–1.0)."""
        self._model_opacity = max(0.0, min(1.0, float(opacity)))
        self._update_volume_opacity_transfer_function(self._model_opacity)

        if self._volume_added:
            self._request_render()

    def _update_volume_opacity_transfer_function(self, opacity: float) -> None:
        """Update volume opacity points without forcing a render."""
        if self._opacity_tf is None:
            return
        opacity = max(0.0, min(1.0, float(opacity)))
        config = self.volume_manager.get_transfer_function_config()
        self._opacity_tf.RemoveAllPoints()
        for hu, base_opacity in config["opacity_points"]:
            self._opacity_tf.AddPoint(hu, base_opacity * opacity)

    def set_model_opacity(self, opacity: float) -> None:
        """Apply a direct anatomy opacity scale for backward compatibility."""
        self._model_opacity = max(0.0, min(1.0, float(opacity)))
        self._update_volume_opacity_transfer_function(self._model_opacity)
        if self._vertebral_mesh_actor is not None:
            self._vertebral_mesh_actor.GetProperty().SetOpacity(
                self._vertebral_mesh_default_opacity * self._model_opacity
            )
        if self._segmentation_actor is not None:
            self._segmentation_actor.GetProperty().SetOpacity(
                self._segmentation_default_opacity * self._model_opacity
            )
        if self._renderer is not None:
            self._request_render()

    def set_model_transparency(self, transparency: float) -> None:
        """Backward-compatible alias for vertebral body transparency."""
        self.set_vertebral_transparency(transparency)

    def set_vertebral_transparency(self, transparency: float) -> None:
        """Adjust vertebral surfaces without changing CT volume or screws."""
        self._vertebral_transparency = max(
            0.0,
            min(1.0, float(transparency)),
        )
        self._model_transparency = self._vertebral_transparency
        self._vertebral_surface_opacity = max(
            self.MIN_VERTEBRAL_SURFACE_OPACITY,
            1.0 - self._vertebral_transparency,
        )
        self._apply_screw_focus()
        opacity_label = self.__dict__.get("model_opacity_label")
        if opacity_label is not None:
            opacity_label.setText(
                "Vertebra Transparency "
                f"{round(self._vertebral_transparency * 100):d}%"
            )
        if self._renderer is not None:
            self._request_render()

    # --- Plane Indicators ---

    def _create_plane_indicators(self):
        """Create colored plane indicators for MPR positions."""
        for plane_name, color in PLANE_INDICATOR_COLORS.items():
            plane = vtk.vtkPlaneSource()
            plane.SetResolution(1, 1)

            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(plane.GetOutputPort())

            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            actor.GetProperty().SetColor(
                color[0] / 255.0, color[1] / 255.0, color[2] / 255.0
            )
            actor.GetProperty().SetOpacity(PLANE_INDICATOR_OPACITY)
            actor.GetProperty().SetLighting(False)
            actor.SetVisibility(self._planes_visible)

            outline = vtk.vtkOutlineFilter()
            outline.SetInputConnection(plane.GetOutputPort())

            outline_mapper = vtk.vtkPolyDataMapper()
            outline_mapper.SetInputConnection(outline.GetOutputPort())

            outline_actor = vtk.vtkActor()
            outline_actor.SetMapper(outline_mapper)
            outline_actor.GetProperty().SetColor(
                color[0] / 255.0, color[1] / 255.0, color[2] / 255.0
            )
            outline_actor.GetProperty().SetOpacity(PLANE_OUTLINE_OPACITY)
            outline_actor.GetProperty().SetLineWidth(1.5)
            outline_actor.SetVisibility(self._planes_visible)

            self._plane_actors[plane_name] = {
                "source": plane,
                "actor": actor,
                "outline_actor": outline_actor,
            }

    def set_plane_indicators_visible(self, visible: bool) -> None:
        """Show or hide all three MPR planes in the 3D viewport."""
        self._planes_visible = bool(visible)
        for plane_data in self._plane_actors.values():
            plane_data["actor"].SetVisibility(self._planes_visible)
            plane_data["outline_actor"].SetVisibility(self._planes_visible)

        toggle = getattr(self, "plane_visibility_toggle", None)
        if toggle is not None:
            if toggle.isChecked() != self._planes_visible:
                toggle.setChecked(self._planes_visible)
            toggle.setText("Planes On" if self._planes_visible else "Planes Off")
        if self._renderer is not None:
            self._request_render()

    def _update_plane_positions(self):
        """Update positions of slice plane indicators."""
        bounds = self.volume_manager.bounds
        if bounds == (0, 0, 0, 0, 0, 0):
            return

        # Axial plane (XY at Z)
        axial_z = self.volume_manager.get_slice_position("axial")
        axial = self._plane_actors["axial"]["source"]
        axial.SetOrigin(bounds[0], bounds[2], axial_z)
        axial.SetPoint1(bounds[1], bounds[2], axial_z)
        axial.SetPoint2(bounds[0], bounds[3], axial_z)

        # Sagittal plane (YZ at X)
        sagittal_x = self.volume_manager.get_slice_position("sagittal")
        sagittal = self._plane_actors["sagittal"]["source"]
        sagittal.SetOrigin(sagittal_x, bounds[2], bounds[4])
        sagittal.SetPoint1(sagittal_x, bounds[3], bounds[4])
        sagittal.SetPoint2(sagittal_x, bounds[2], bounds[5])

        # Coronal plane (XZ at Y)
        coronal_y = self.volume_manager.get_slice_position("coronal")
        coronal = self._plane_actors["coronal"]["source"]
        coronal.SetOrigin(bounds[0], coronal_y, bounds[4])
        coronal.SetPoint1(bounds[1], coronal_y, bounds[4])
        coronal.SetPoint2(bounds[0], coronal_y, bounds[5])

    def _add_axes_widget(self):
        """Add orientation axes widget to corner."""
        axes = vtk.vtkAxesActor()

        widget = vtk.vtkOrientationMarkerWidget()
        widget.SetOrientationMarker(axes)
        widget.SetInteractor(
            self.vtk_widget.GetRenderWindow().GetInteractor()
        )
        widget.SetViewport(0.0, 0.0, 0.2, 0.2)
        widget.EnabledOn()
        widget.InteractiveOff()

        self._axes_widget = widget

    # --- Screw Visualization ---

    def add_screw(
        self,
        entry_point: Tuple[float, float, float],
        target_point: Tuple[float, float, float],
        radius: float = 3.0,
        color: Tuple[float, float, float] = None,
        screw_id: Optional[int] = None,
    ) -> ScrewVisual:
        """Add a pickable head/shaft/tip screw visual."""
        resolved_id = len(self._screw_actors) if screw_id is None else int(screw_id)
        visual = create_screw_visual(
            resolved_id, entry_point, target_point, radius, color
        )
        for part, actor in (
            ("entry", visual.entry_actor),
            ("shaft", visual.shaft_actor),
            ("tip", visual.tip_actor),
        ):
            self._renderer.AddActor(actor)
            self._screw_prop_parts[self._prop_key(actor)] = (resolved_id, part)
        self._screw_actors.append(visual)
        self._apply_screw_selection_style(visual)
        self._request_render()

        return visual

    def remove_screw(self, actor):
        """Remove a complete screw visual or a legacy single actor."""
        if isinstance(actor, ScrewVisual) and actor in self._screw_actors:
            for prop in actor.props:
                self._screw_prop_parts.pop(self._prop_key(prop), None)
                self._renderer.RemoveActor(prop)
            self._screw_actors.remove(actor)
            if not self._render_guard_active:
                self._request_render()
        elif isinstance(actor, vtk.vtkActor):
            self._renderer.RemoveActor(actor)

    def clear_screws(self):
        """Remove all screw visuals."""
        if not self._screw_actors:
            self._selected_screw_id = None
            self._apply_screw_focus()
            return
        for visual in self._screw_actors:
            for actor in visual.props:
                self._renderer.RemoveActor(actor)
        self._screw_actors.clear()
        self._screw_prop_parts.clear()
        self._selected_screw_id = None
        self._apply_screw_focus()
        if not self._render_guard_active:
            self._request_render()

    def set_screw_interaction_callbacks(
        self,
        on_begin: Optional[Callable] = None,
        on_drag: Optional[Callable] = None,
        on_end: Optional[Callable] = None,
        on_select: Optional[Callable] = None,
    ) -> None:
        """Connect 3D direct screw selection and manipulation callbacks."""
        self._screw_drag_begin_callback = on_begin
        self._screw_drag_callback = on_drag
        self._screw_drag_end_callback = on_end
        self._screw_select_callback = on_select

    def set_selected_screw(self, screw_id: Optional[int]) -> None:
        """Highlight one screw without hiding the remaining plan."""
        self._selected_screw_id = None if screw_id is None else int(screw_id)
        for visual in self._screw_actors:
            self._apply_screw_selection_style(visual)
        self._apply_screw_focus()
        if self._screw_actors:
            self._request_render()

    def _apply_screw_focus(self) -> None:
        """Keep vertebral transparency under explicit user control."""
        surface_opacity = self.__dict__.get(
            "_vertebral_surface_opacity",
            0.50,
        )
        if self._vertebral_mesh_actor is not None:
            self._vertebral_mesh_actor.GetProperty().SetOpacity(
                surface_opacity
            )
        if self._segmentation_actor is not None:
            self._segmentation_actor.GetProperty().SetOpacity(
                surface_opacity
            )

    def _build_screw_only_picker(self) -> vtk.vtkCellPicker:
        """Build a picker that ignores anatomy occluding screw geometry."""
        picker = vtk.vtkCellPicker()
        picker.SetTolerance(0.01)
        picker.PickFromListOn()
        for visual in self._screw_actors:
            for prop in visual.props:
                picker.AddPickList(prop)
        return picker

    def _screw_visual_by_id(self, screw_id: int) -> Optional[ScrewVisual]:
        return next(
            (
                visual
                for visual in self._screw_actors
                if visual.screw_id == int(screw_id)
            ),
            None,
        )

    def _resolve_screw_pick_part(
        self,
        visual: ScrewVisual,
        fallback_part: str,
        picked_world: Tuple[float, float, float],
    ) -> str:
        """Treat the distal shaft as part of the tip interaction handle."""
        if fallback_part != "shaft":
            return fallback_part
        trajectory = tuple(
            visual.target_point[index] - visual.entry_point[index]
            for index in range(3)
        )
        length_squared = sum(value * value for value in trajectory)
        if length_squared <= 1e-9:
            return fallback_part
        fraction = sum(
            (picked_world[index] - visual.entry_point[index])
            * trajectory[index]
            for index in range(3)
        ) / length_squared
        if fraction >= self.TIP_SHAFT_START_FRACTION:
            return "tip"
        return fallback_part

    def _apply_screw_selection_style(self, visual: ScrewVisual) -> None:
        selected = visual.screw_id == self._selected_screw_id
        for part, actor in (
            ("entry", visual.entry_actor),
            ("shaft", visual.shaft_actor),
            ("tip", visual.tip_actor),
        ):
            active = self._active_screw_part == (visual.screw_id, part)
            actor.GetProperty().SetAmbient(0.8 if active else 0.55 if selected else 0.3)
            actor.GetProperty().SetSpecular(1.0 if active or selected else 0.8)
            actor.GetProperty().SetLineWidth(3.0 if active else 2.0 if selected else 1.0)
            actor.GetProperty().SetEdgeVisibility(active)
            if active:
                actor.GetProperty().SetEdgeColor(0.85, 1.0, 1.0)

    def _on_screw_left_press(self, obj, event) -> None:
        """Select on one click and toggle pointer movement on double-click."""
        interactor = self.vtk_widget.GetRenderWindow().GetInteractor()
        x, y = interactor.GetEventPosition()
        if self._screw_drag_active:
            result = self._process_screw_press(x, y, None, None, None)
            if result == "ended":
                self._restore_trackball_after_event()
            return

        if self._pan_mode_active:
            interactor.SetShiftKey(1)
            self._pan_forced_shift = True
            self.vtk_widget.setCursor(Qt.CursorShape.ClosedHandCursor)
            return

        picker = self._build_screw_only_picker()
        if not picker.Pick(x, y, 0, self._renderer):
            if self._focus_double_click_detector.register(x, y):
                self.focus_at_display(x, y)
            return
        part_data = self._screw_prop_parts.get(
            self._prop_key(picker.GetViewProp())
        )
        if part_data is None:
            return

        screw_id, part = part_data
        picked_world = tuple(float(value) for value in picker.GetPickPosition())
        visual = self._screw_visual_by_id(screw_id)
        if visual is not None:
            part = self._resolve_screw_pick_part(
                visual, part, picked_world
            )
        self._focus_double_click_detector.reset()
        self.set_selected_screw(screw_id)
        camera = self._renderer.GetActiveCamera()
        camera_normal = tuple(
            float(value) for value in camera.GetDirectionOfProjection()
        )
        result = self._process_screw_press(
            x, y, (screw_id, part), picked_world, camera_normal
        )
        if result == "started":
            self._set_trackball_enabled(False)

    def _on_screw_mouse_move(self, obj, event) -> None:
        """Move the active screw part on a camera-facing world plane."""
        if (
            not self._screw_drag_active
            or self._screw_drag_plane_origin is None
            or self._screw_drag_plane_normal is None
        ):
            return
        interactor = self.vtk_widget.GetRenderWindow().GetInteractor()
        x, y = interactor.GetEventPosition()
        world_point = self._display_to_drag_plane(x, y)
        if world_point is not None:
            self._emit_screw_pointer_move(world_point)

    def _on_screw_left_release(self, obj, event) -> None:
        """Release does not end double-click-locked screw movement."""
        if self.__dict__.get("_pan_forced_shift", False):
            self._pan_forced_shift = False
            interactor = self.vtk_widget.GetRenderWindow().GetInteractor()
            QTimer.singleShot(0, lambda: interactor.SetShiftKey(0))
            self.vtk_widget.setCursor(Qt.CursorShape.OpenHandCursor)

    def _process_screw_press(
        self,
        x: int,
        y: int,
        screw_part: Optional[Tuple[int, str]],
        world_point: Optional[Tuple[float, float, float]],
        camera_normal: Optional[Tuple[float, float, float]],
        timestamp: Optional[float] = None,
    ) -> str:
        """Apply single-select and double-click toggle semantics in 3D."""
        is_double = self._double_click_detector.register(
            x, y, timestamp=timestamp
        )
        if self._screw_drag_active:
            if is_double:
                self._finish_screw_pointer_move(notify=True)
                return "ended"
            return "locked"
        if screw_part is None or world_point is None or camera_normal is None:
            return "navigate"

        screw_id, part = screw_part
        if is_double:
            accepted = True
            if self._screw_select_callback is not None:
                self._screw_select_callback(int(screw_id))
            if self._screw_drag_begin_callback is not None:
                accepted = bool(
                    self._screw_drag_begin_callback(
                        int(screw_id), str(part), tuple(world_point)
                    )
                )
            if accepted:
                self._screw_drag_active = True
                self._active_screw_part = (int(screw_id), str(part))
                self._screw_drag_plane_origin = tuple(world_point)
                self._screw_drag_plane_normal = tuple(camera_normal)
                self._refresh_screw_visual_styles()
                return "started"
            return "selected"

        if self._screw_select_callback is not None:
            self._screw_select_callback(int(screw_id))
        return "selected"

    def _emit_screw_pointer_move(
        self, world_point: Tuple[float, float, float]
    ) -> None:
        if self._screw_drag_active and self._screw_drag_callback is not None:
            self._screw_drag_callback(tuple(world_point), "3D")

    def _finish_screw_pointer_move(self, notify: bool) -> None:
        self._screw_drag_active = False
        self._active_screw_part = None
        self._screw_drag_plane_origin = None
        self._screw_drag_plane_normal = None
        self._refresh_screw_visual_styles()
        if notify and self._screw_drag_end_callback is not None:
            self._screw_drag_end_callback()

    def cancel_screw_interaction(self) -> None:
        """Exit pointer movement at its last valid position."""
        self._finish_screw_pointer_move(notify=False)
        self._double_click_detector.reset()
        self._set_trackball_enabled(True)

    @property
    def is_screw_interaction_active(self) -> bool:
        """Whether the 3D viewport currently owns a pointer-move lock."""
        return bool(self._screw_drag_active)

    def _set_trackball_enabled(self, enabled: bool) -> None:
        if not hasattr(self, "vtk_widget"):
            return
        interactor = self.vtk_widget.GetRenderWindow().GetInteractor()
        style = interactor.GetInteractorStyle()
        if style is not None:
            style.SetEnabled(1 if enabled else 0)

    def _restore_trackball_after_event(self) -> None:
        QTimer.singleShot(0, lambda: self._set_trackball_enabled(True))

    def _refresh_screw_visual_styles(self) -> None:
        for visual in getattr(self, "_screw_actors", []):
            self._apply_screw_selection_style(visual)
        if getattr(self, "_screw_actors", None) and hasattr(self, "vtk_widget"):
            self._request_render()

    def _display_to_drag_plane(
        self, x: int, y: int
    ) -> Optional[Tuple[float, float, float]]:
        if (
            self._screw_drag_plane_origin is None
            or self._screw_drag_plane_normal is None
        ):
            return None
        near_point = self._display_to_world(x, y, 0.0)
        far_point = self._display_to_world(x, y, 1.0)
        return intersect_ray_plane(
            near_point,
            far_point,
            self._screw_drag_plane_origin,
            self._screw_drag_plane_normal,
        )

    def _display_to_world(
        self, x: int, y: int, depth: float
    ) -> Tuple[float, float, float]:
        self._renderer.SetDisplayPoint(float(x), float(y), float(depth))
        self._renderer.DisplayToWorld()
        homogeneous = self._renderer.GetWorldPoint()
        scale = homogeneous[3] if abs(homogeneous[3]) > 1e-12 else 1.0
        return tuple(float(homogeneous[index] / scale) for index in range(3))

    @staticmethod
    def _prop_key(prop) -> str:
        return prop.GetAddressAsString("")

    # --- Measurement Visualization ---

    def add_measurement(
        self,
        measurement_id: int,
        points: List[Tuple[float, float, float]],
        label: str,
        color: Tuple[float, float, float] = (1.0, 0.8, 0.2),
    ) -> None:
        """Add 3D measurement line segments and a label."""
        if not points or len(points) < 2:
            return
        if measurement_id in self._measurement_props:
            self.remove_measurement(measurement_id)

        props: List[vtk.vtkProp] = []

        for index in range(1, len(points)):
            p1 = points[index - 1]
            p2 = points[index]

            line = vtk.vtkLineSource()
            line.SetPoint1(*p1)
            line.SetPoint2(*p2)

            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(line.GetOutputPort())

            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            actor.GetProperty().SetColor(*color)
            actor.GetProperty().SetLineWidth(3)
            actor.GetProperty().SetLighting(False)

            self._renderer.AddViewProp(actor)
            props.append(actor)

        end_point = points[-1]
        text_actor = vtk.vtkBillboardTextActor3D()
        text_actor.SetInput(label)
        text_actor.SetPosition(*end_point)
        text_actor.GetTextProperty().SetColor(*color)
        text_actor.GetTextProperty().SetBold(True)
        text_actor.GetTextProperty().SetFontSize(18)

        self._renderer.AddViewProp(text_actor)
        props.append(text_actor)
        self._measurement_props[measurement_id] = props
        self._request_render()

    def remove_measurement(self, measurement_id: int) -> None:
        """Remove one measurement's actors by identifier."""
        props = self._measurement_props.pop(measurement_id, None)
        if not props:
            return
        for prop in props:
            self._renderer.RemoveViewProp(prop)
        if not self._render_guard_active:
            self._request_render()

    def clear_measurements(self) -> None:
        """Remove all measurement actors and labels."""
        if not self._measurement_props:
            return
        for props in self._measurement_props.values():
            for prop in props:
                self._renderer.RemoveViewProp(prop)
        self._measurement_props = {}
        if not self._render_guard_active:
            self._request_render()

    # --- Segmentation Overlay ---

    def set_segmentation_mask(
        self,
        mask_image: vtk.vtkImageData,
        label_value: int = 0,
        color: Tuple[float, float, float] = (0.92, 0.82, 0.70),
        opacity: float = 0.35,
    ) -> None:
        """Render segmentation mask as translucent 3D surface."""
        if mask_image is None:
            self.clear_segmentation_mask()
            return

        self._segmentation_mask_image = mask_image
        self._segmentation_label_value = max(0, int(label_value))
        self._segmentation_color = color
        self._segmentation_opacity = max(0.0, min(1.0, float(opacity)))
        self._segmentation_default_opacity = self._segmentation_opacity
        self._render_segmentation_actor()

    def set_segmentation_label(self, label_value: int) -> None:
        """Update active segmentation label filter (0 means all labels)."""
        self._segmentation_label_value = max(0, int(label_value))
        if self._segmentation_mask_image is None:
            return
        self._render_segmentation_actor()

    def set_segmentation_visible(self, visible: bool) -> None:
        """Toggle visibility of segmentation overlay actor."""
        if self._segmentation_actor is None:
            return
        self._segmentation_actor.SetVisibility(visible)
        self._request_render()

    def _render_segmentation_actor(self) -> None:
        """Build or rebuild segmentation actor with current label filter."""
        self.clear_segmentation_actor_only()
        if self._segmentation_mask_image is None:
            return

        # vtkImageBinaryThreshold (VTK >= 9.7) replaces the deprecated
        # vtkImageThreshold.ThresholdBetween(); see the same fallback in
        # extract_vertebral_mesh (src/core/vertebral_mesh.py).
        if self._segmentation_label_value <= 0:
            lower, upper = 1, 1000000
        else:
            lower = upper = self._segmentation_label_value
        if hasattr(vtk, "vtkImageBinaryThreshold"):
            threshold = vtk.vtkImageBinaryThreshold()
            threshold.SetInputData(self._segmentation_mask_image)
            threshold.SetLowerThreshold(lower)
            threshold.SetUpperThreshold(upper)
            threshold.SetInValue(1)
            threshold.SetOutValue(0)
            threshold.SetReplaceIn(True)
            threshold.SetReplaceOut(True)
            threshold.SetOutputScalarTypeToUnsignedChar()
        else:
            threshold = vtk.vtkImageThreshold()
            threshold.SetInputData(self._segmentation_mask_image)
            threshold.ThresholdBetween(lower, upper)
            threshold.SetInValue(1)
            threshold.SetOutValue(0)
            threshold.SetOutputScalarTypeToUnsignedChar()
        threshold.Update()

        scalar_range = threshold.GetOutput().GetScalarRange()
        logger.info(
            "3D seg: label=%d, mask dims=%s, threshold range=%s",
            self._segmentation_label_value,
            self._segmentation_mask_image.GetDimensions(),
            scalar_range,
        )

        surface_input = threshold.GetOutputPort()
        voxel_count = math.prod(self._segmentation_mask_image.GetDimensions())
        gaussian = None
        if voxel_count <= 32_000_000:
            gaussian = vtk.vtkImageGaussianSmooth()
            gaussian.SetInputConnection(threshold.GetOutputPort())
            gaussian.SetStandardDeviations(0.8, 0.8, 0.8)
            gaussian.SetRadiusFactors(1.5, 1.5, 1.5)
            gaussian.SetDimensionality(3)
            surface_input = gaussian.GetOutputPort()

        # vtkFlyingEdges3D is the same sub-voxel extractor used by
        # extract_vertebral_mesh (src/core/vertebral_mesh.py); it produces the
        # same isosurface as vtkMarchingCubes with far less staircase noise
        # on anisotropic grids.
        surface_extractor = vtk.vtkFlyingEdges3D()
        surface_extractor.SetInputConnection(surface_input)
        surface_extractor.SetValue(0, 0.5)
        surface_extractor.ComputeNormalsOff()
        surface_extractor.ComputeGradientsOff()
        surface_extractor.Update()

        n_points = surface_extractor.GetOutput().GetNumberOfPoints()
        logger.info("3D seg: flying edges produced %d points", n_points)
        if n_points == 0:
            return

        decimator = vtk.vtkDecimatePro()
        decimator.SetInputConnection(surface_extractor.GetOutputPort())
        decimator.SetTargetReduction(0.45)
        decimator.PreserveTopologyOn()

        smoother = vtk.vtkWindowedSincPolyDataFilter()
        smoother.SetInputConnection(decimator.GetOutputPort())
        smoother.SetNumberOfIterations(MESH_SMOOTHING_ITERATIONS)
        smoother.SetPassBand(MESH_SMOOTHING_PASSBAND)
        smoother.BoundarySmoothingOff()
        smoother.FeatureEdgeSmoothingOff()
        smoother.NonManifoldSmoothingOn()
        smoother.NormalizeCoordinatesOn()

        normals = vtk.vtkPolyDataNormals()
        normals.SetInputConnection(smoother.GetOutputPort())
        normals.ComputePointNormalsOn()
        normals.ComputeCellNormalsOff()
        normals.SplittingOff()

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(normals.GetOutputPort())
        mapper.ScalarVisibilityOff()

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(*self._segmentation_color)
        actor.GetProperty().SetOpacity(
            self.__dict__.get("_vertebral_surface_opacity", 0.50)
        )
        actor.GetProperty().SetLighting(True)
        actor.GetProperty().SetAmbient(0.24)
        actor.GetProperty().SetDiffuse(0.76)
        actor.GetProperty().SetSpecular(0.18)
        actor.GetProperty().SetSpecularPower(18.0)

        self._renderer.AddActor(actor)
        self._segmentation_actor = actor
        self._request_render()

    def clear_segmentation_actor_only(self) -> None:
        """Remove current segmentation actor while keeping mask state."""
        if self._segmentation_actor is None:
            return
        self._renderer.RemoveActor(self._segmentation_actor)
        self._segmentation_actor = None
        if not self._render_guard_active:
            self._request_render()

    def clear_segmentation_mask(self) -> None:
        """Remove segmentation overlay actor if present."""
        self.clear_segmentation_actor_only()
        self.clear_vertebral_mesh()
        self._segmentation_mask_image = None
        self._segmentation_label_value = 0

    # --- Vertebral Body Mesh ---

    def set_vertebral_mesh(
        self,
        mask_image: vtk.vtkImageData,
        labels: Optional[list[int]] = None,
    ) -> None:
        """Render vertebral body mesh from segmentation mask.

        Extracts vertebral labels (25-43: sacrum through T1) from the
        multilabel segmentation mask and renders them as a combined
        colored 3D mesh. Each vertebra gets a distinct warm-gradient color.

        Args:
            mask_image: Multilabel vtkImageData from TotalSegmentator.
            labels: Optional labels already detected by the segmentation step.
        """
        from ..core.vertebral_mesh import extract_vertebral_mesh

        self.clear_vertebral_mesh()

        if mask_image is None:
            return

        poly_data = extract_vertebral_mesh(mask_image, labels=labels)
        if poly_data is None or poly_data.GetNumberOfCells() == 0:
            logger.info("set_vertebral_mesh: no vertebral mesh produced")
            return

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(poly_data)
        mapper.ScalarVisibilityOn()
        mapper.SetScalarModeToUseCellData()
        mapper.SelectColorArray("VertebraColors")

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetOpacity(
            self.__dict__.get("_vertebral_surface_opacity", 0.50)
        )
        actor.GetProperty().SetAmbient(0.24)
        actor.GetProperty().SetDiffuse(0.76)
        actor.GetProperty().SetSpecular(0.22)
        actor.GetProperty().SetSpecularPower(24.0)
        actor.GetProperty().SetInterpolationToPhong()

        self._renderer.AddActor(actor)
        self._vertebral_mesh_actor = actor
        self._request_render()

        logger.info(
            "set_vertebral_mesh: added actor with %d cells",
            poly_data.GetNumberOfCells(),
        )

    def clear_vertebral_mesh(self) -> None:
        """Remove vertebral mesh actor from the 3D scene."""
        if self._vertebral_mesh_actor is None:
            return
        self._renderer.RemoveActor(self._vertebral_mesh_actor)
        self._vertebral_mesh_actor = None
        if not self._render_guard_active:
            self._request_render()

    # --- Backward Compatibility ---

    def set_bone_opacity(self, opacity: float):
        """Backward-compatible: now controls volume opacity."""
        self.set_volume_opacity(opacity)

    def set_plane_visibility(self, plane: str, visible: bool):
        """Set visibility of a plane indicator."""
        if plane in self._plane_actors:
            self._plane_actors[plane]["actor"].SetVisibility(visible)
            self._plane_actors[plane]["outline_actor"].SetVisibility(visible)
            self._request_render()

    # --- Events ---

    def _on_volume_event(self, event: str, **kwargs):
        """Handle volume manager events."""
        logger.debug("_on_volume_event: %s %s", event, kwargs)
        if event == "volume_loaded":
            self.update_volume()
        elif event == "transfer_function_changed":
            preset = kwargs.get("preset", "Bone")
            self.apply_transfer_function_preset(preset)
        elif event in ("slice_changed", "crosshair_changed"):
            self._update_plane_positions()
            self._request_render()

    def start(self):
        """Start the interactor."""
        self.vtk_widget.GetRenderWindow().GetInteractor().Initialize()
        self.vtk_widget.GetRenderWindow().GetInteractor().Start()

    def cleanup(self):
        """Cleanup resources."""
        self.clear_segmentation_mask()
        self.volume_manager.remove_observer("viewer_3d")
        self.vtk_widget.GetRenderWindow().Finalize()
