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
    QButtonGroup,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QSlider,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ..core.vertebral_mesh import (
    MESH_SMOOTHING_ITERATIONS,
    MESH_SMOOTHING_PASSBAND,
    label_world_bounds,
    split_mesh_by_label,
)
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
    label_threshold_filter,
    request_render,
    shared_cell_picker,
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

#: Screw MPR plane -> the standard plane whose pane shows it, so an oblique
#: indicator in 3D takes the colour of the MPR header the user is reading.
SCREW_MPR_PLANE_COLORS = {
    "oblique_axial": PLANE_INDICATOR_COLORS["axial"],
    "oblique_sagittal": PLANE_INDICATOR_COLORS["sagittal"],
    "cross_section": PLANE_INDICATOR_COLORS["coronal"],
}

#: Half the side of the square drawn for each screw-aligned plane (mm).  The
#: planes are unbounded; 40 mm either side of the screw covers a vertebra
#: without an oblique quad sprawling across the whole volume.
SCREW_MPR_PLANE_HALF_SIZE_MM = 40.0

#: The textured cross-section slice is cropped to the same footprint as the
#: plane indicator quad, so the slice never reads as bigger or smaller than
#: the frame drawn around it.
SCREW_MPR_SLICE_HALF_SIZE_MM = SCREW_MPR_PLANE_HALF_SIZE_MM

#: Alpha (0-1) applied to slice pixels outside the screw's own vertebra.
#: Kept translucent rather than hidden so the plane still reads as a slice
#: through real anatomy, while opacity still marks where the vertebra (and
#: therefore the cut) actually is.
SCREW_MPR_SLICE_CONTEXT_ALPHA = 0.35

#: Padding (mm) added around a vertebra's voxel bounding box before it is
#: used to crop the main volume / size the cut volume, so the cut face sits
#: just outside the bone rather than shaving its own surface.
SCREW_MPR_CUT_MARGIN_MM = 2.0

#: Window/level (HU) used to color the cross-section slice when the caller
#: has none of its own -- the same bone default the MPR panel starts on.
SCREW_MPR_DEFAULT_WINDOW_LEVEL = (1500.0, 400.0)

#: Cropping-region flag set for a vtkVolumeMapper's 3x3x3 region grid that
#: keeps every region EXCEPT the centre one (VTK_CROP_SUBVOLUME). Used to cut
#: the vertebra's own box out of the main volume, leaving that box for the
#: separate cut volume to render (clipped at the cross-section) so no level
#: other than the screw's own is ever touched.
VOLUME_CROP_OUTSIDE_BOX = 0x7FFFFFF & ~vtk.VTK_CROP_SUBVOLUME

#: Distance (mm) the "Cut View" camera sits back from the cross-section
#: centre, on the entry side, looking down the screw at the open cut face.
SCREW_MPR_CUT_VIEW_DISTANCE_MM = 250.0

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


#: Render-mode names that mean "the mapper has not rendered yet", so the log
#: should wait for a real frame rather than report them.
PENDING_RENDER_MODES = frozenset({"undefined", "unknown"})

#: The mode a ``vtkSmartVolumeMapper`` reports when it found no usable GPU
#: context and fell back to its own CPU ray caster.
CPU_RAYCAST_MODE = "cpu-raycast"


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
    #: Emitted only from a user click on the header Vertebrae/Full CT toggle
    #: (True = Vertebrae requested); never from set_isolation_state, which
    #: blocks signals while it reflects the controller's actual state.
    isolation_requested = pyqtSignal(bool)

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
        # Which shrink table the volume is downsampled by.  Normally the
        # mapper's own, but a smart mapper that turns out to be ray casting on
        # the CPU is demoted to the CPU table (see _apply_cpu_raycast_fallback).
        self._shrink_kind: str = self._mapper_kind
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
        # Screw MPR in 3D: the three screw-aligned plane indicators (created on
        # first use) and the clipping plane that opens the volume at the
        # cross-section.  While active the standard indicators are hidden.
        self._screw_mpr_actors: dict = {}
        self._screw_mpr_active: bool = False
        # The cross-section clipping plane -- now applied ONLY to the screw's
        # own vertebra's cut actor/volume, never to the main volume or mesh
        # mappers (see _apply_screw_mpr_cut / clear_screw_mpr).
        self._screw_mpr_clip: Optional[vtk.vtkPlane] = None
        # Textured cross-section slice pipeline (built lazily; the actor is
        # created once and reused so Position/rotation updates are cheap).
        self._screw_mpr_slice_actor: Optional[vtk.vtkImageActor] = None
        self._screw_mpr_slice_reslice: Optional[vtk.vtkImageReslice] = None
        self._screw_mpr_slice_mask_reslice: Optional[vtk.vtkImageReslice] = None
        self._screw_mpr_slice_color_map: Optional[vtk.vtkImageMapToWindowLevelColors] = None
        self._screw_mpr_slice_alpha_filter: Optional[vtk.vtkImageThreshold] = None
        # The local cut: the screw's own vertebra split out of the combined
        # mesh (cut_actor) and out of the volume (cut_volume), each carrying
        # _screw_mpr_clip as its only clipping plane. Every other level is
        # left whole.
        self._screw_mpr_cut_actor: Optional[vtk.vtkActor] = None
        self._screw_mpr_cut_volume: Optional[vtk.vtkVolume] = None
        self._screw_mpr_cut_volume_extract: Optional[vtk.vtkExtractVOI] = None
        self._screw_mpr_cut_label: Optional[int] = None
        self._screw_mpr_cut_volume_source: Optional[vtk.vtkImageData] = None
        self._screw_mpr_cross_section_axes: Optional[vtk.vtkMatrix4x4] = None
        self._screw_mpr_window_level: Optional[Tuple[float, float]] = None
        self._screw_mpr_view_button: Optional[QToolButton] = None
        # Unpadded world-box cache for the resolved cut label, keyed on
        # (id(_segmentation_mask_image), label) so a Position/rotation/offset
        # step (same label, same mask) never re-scans the mask with scipy's
        # find_objects -- only a new label or a swapped mask does. A freed
        # mask's id can be reused, so this cache is only safe because
        # set_segmentation_mask/clear_segmentation_mask explicitly clear it
        # whenever the mask identity changes.
        self._screw_mpr_label_box_cache: Dict[Tuple[int, int], Optional[Tuple[float, float, float, float, float, float]]] = {}
        # The full combined mesh, kept so the screw's vertebra can be split
        # back out again (rebuild, or a different screw's level).
        self._vertebral_mesh_poly: Optional[vtk.vtkPolyData] = None
        self._vertebral_mesh_split_cache: Dict[int, Tuple[vtk.vtkPolyData, vtk.vtkPolyData]] = {}

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

        # One focus picker per viewer, reused on every double-click: the old
        # path built a fresh vtkCellPicker per focus request.
        self._focus_picker = shared_cell_picker(0.005)
        self._focus_picker.PickFromListOn()
        # One screw picker per viewer, rebuilt only when the screw set
        # changes (see _build_screw_only_picker): per-click allocation on
        # the pointer path is what this replaces.
        self._screw_picker: Optional[vtk.vtkCellPicker] = None

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
        # No colour here on purpose: the app stylesheet's QLabel#viewerHeader
        # rule supplies the theme's viewer_foreground, and a hard-coded white
        # in this inline sheet used to override it on the light palettes.
        self.label.setStyleSheet("font-weight: bold; padding: 2px;")
        self.label.doubleClicked.connect(self.header_double_clicked)

        self.header_bar = QWidget(self)
        self.header_bar.setObjectName("viewerHeaderBar")
        header_bar_layout = QHBoxLayout(self.header_bar)
        header_bar_layout.setContentsMargins(0, 0, 0, 0)
        header_bar_layout.setSpacing(4)
        header_bar_layout.addWidget(self.label, 1)

        self.show_vertebrae_btn = QToolButton(self.header_bar)
        self.show_vertebrae_btn.setObjectName("isolationToggle")
        self.show_vertebrae_btn.setText("Vertebrae")
        self.show_vertebrae_btn.setCheckable(True)
        self.show_vertebrae_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.show_vertebrae_btn.setToolTip(
            "Show only the segmented vertebrae (MPR views are masked too)"
        )
        self.show_full_ct_btn = QToolButton(self.header_bar)
        self.show_full_ct_btn.setObjectName("isolationToggle")
        self.show_full_ct_btn.setText("Full CT")
        self.show_full_ct_btn.setCheckable(True)
        self.show_full_ct_btn.setChecked(True)
        self.show_full_ct_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.show_full_ct_btn.setToolTip("Show the whole CT volume")
        self._isolation_button_group = QButtonGroup(self.header_bar)
        self._isolation_button_group.setExclusive(True)
        self._isolation_button_group.addButton(self.show_vertebrae_btn)
        self._isolation_button_group.addButton(self.show_full_ct_btn)
        for button in (self.show_vertebrae_btn, self.show_full_ct_btn):
            button.setEnabled(False)
        self.show_vertebrae_btn.toggled.connect(self._on_isolation_toggle_clicked)
        header_bar_layout.addWidget(self.show_vertebrae_btn)
        header_bar_layout.addWidget(self.show_full_ct_btn)

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

        self.screw_mpr_view_button = QToolButton(self.viewport_container)
        self.screw_mpr_view_button.setObjectName("screwMprViewButton")
        self.screw_mpr_view_button.setText("Cut View")
        self.screw_mpr_view_button.setToolTip(
            "Look down the screw at the cross-section from the entry side"
        )
        self.screw_mpr_view_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.screw_mpr_view_button.setStyleSheet(
            "QToolButton#screwMprViewButton {"
            "background: rgba(24, 29, 35, 210); color: #e8eef4;"
            "border: 1px solid rgba(205, 218, 228, 90);"
            "border-radius: 7px; padding: 5px 9px;"
            "font-size: 11px; font-weight: 600; }"
            "QToolButton#screwMprViewButton:hover {"
            "background: rgba(45, 54, 64, 225); }"
        )
        self.screw_mpr_view_button.clicked.connect(self.focus_screw_mpr_cut)
        # Only meaningful while Screw MPR is active -- see show_screw_mpr /
        # clear_screw_mpr.
        self.screw_mpr_view_button.setVisible(False)

        # Reset View and Cut View share the top-left corner as one row so
        # neither button overlaps the plane toggle beneath it.
        self.top_left_controls = QWidget(self.viewport_container)
        top_left_layout = QHBoxLayout(self.top_left_controls)
        top_left_layout.setContentsMargins(0, 0, 0, 0)
        top_left_layout.setSpacing(5)
        top_left_layout.addWidget(self.reset_view_button)
        top_left_layout.addWidget(self.screw_mpr_view_button)
        viewport_layout.addWidget(
            self.top_left_controls,
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

        layout.addWidget(self.header_bar)
        layout.addWidget(self.viewport_container, stretch=1)

    def _on_isolation_toggle_clicked(self, checked: bool) -> None:
        """Emit a user request only -- never fired by set_isolation_state."""
        self.isolation_requested.emit(bool(checked))

    def set_isolation_state(self, isolated: bool, available: bool) -> None:
        """Reflect the controller's real isolation state without re-emitting.

        Called after every isolate/restore attempt (including a failed one,
        which snaps the toggle back to Full CT) so the header always shows
        what is actually on screen rather than what was last requested.
        """
        for button in (self.show_vertebrae_btn, self.show_full_ct_btn):
            button.setEnabled(bool(available))
        self.show_vertebrae_btn.setToolTip(
            "Show only the segmented vertebrae (MPR views are masked too)"
            if available
            else "Run TotalSegmentator segmentation first"
        )
        # Both buttons' signals must be blocked for the whole exchange: the
        # exclusive QButtonGroup can otherwise force the *other* button's
        # checked state (and fire its toggled signal) after this method has
        # already restored that button's own block, turning a state sync
        # into a spurious isolation_requested emission.
        blocked = [
            (button, button.blockSignals(True))
            for button in (self.show_vertebrae_btn, self.show_full_ct_btn)
        ]
        try:
            self.show_vertebrae_btn.setChecked(bool(isolated))
            self.show_full_ct_btn.setChecked(not isolated)
        finally:
            for button, previous in blocked:
                button.blockSignals(previous)

    def _build_volume_mapper(self) -> vtk.vtkVolumeMapper:
        """Select and configure the volume mapper for the current platform."""
        self._mapper_kind = select_volume_mapper_kind(sys.platform)
        self._shrink_kind = self._mapper_kind
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
            self._sync_screw_mpr_cut_volume_quality()

    def _on_interaction_end(self, obj, event):
        """Restore fine sampling after interaction ends."""
        if self._volume_mapper and hasattr(self, '_target_sample_dist'):
            self._volume_mapper.SetSampleDistance(self._target_sample_dist)
            self._sync_screw_mpr_cut_volume_quality()
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
        picker = self._focus_picker
        picker.ClearPickList()
        for prop in (
            self._vertebral_mesh_actor,
            self._segmentation_actor,
            self.__dict__.get("_screw_mpr_cut_actor"),
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
            assessment.tier, spacing, self._shrink_kind
        )
        logger.info(
            "  dims=%s spacing=%s tier=%s mapper=%s shrink=%s(%s) sample_dist=%.2f",
            dims, spacing, assessment.tier, self._mapper_kind, shrink,
            self._shrink_kind, sample_dist,
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
        self._sync_screw_mpr_cut_volume_quality()

        logger.info("  Phase 1 render (sample_dist=%.2f)...", coarse_dist)
        _flush_logs()

        # Lift guard and enable widget updates before rendering
        self._render_state = self._RS_NORMAL
        self.vtk_widget.setUpdatesEnabled(True)

        t0 = time.perf_counter()
        request_render(self.vtk_widget)
        render_sec = time.perf_counter() - t0
        logger.info(
            "  Phase 1 done: %.3fs (mapper=%s)", render_sec, self._mapper_kind
        )

        # safe_render() only marks the widget dirty; the VTK render itself
        # happens in the next paintEvent, so the mapper's last-used mode is
        # still undefined right here. Read it on the next event-loop turn.
        self._render_mode_logged = False
        QTimer.singleShot(0, lambda: self._log_render_mode("Phase 1"))

        # Schedule Phase 2 via QTimer (works now — no Cocoa event loop starvation)
        gen = self._render_generation
        QTimer.singleShot(500, lambda: self._execute_phase2(gen))
        logger.info("  Phase 2 timer started (gen=%d, 0.5s)", gen)

    def _log_render_mode(self, phase: str = "Phase 1", final: bool = False):
        """Log the render mode the mapper actually used, once it is known.

        Called on the event-loop turn after each phase's paint request. While
        the mode is still undefined the log is held back, so a mapper that
        only settles later is reported honestly; Phase 2 makes the final
        attempt and logs whatever it sees.
        """
        # Read through __dict__: a Viewer3D built with __new__ (as the tests
        # do) raises RuntimeError, not AttributeError, on attribute access.
        state = self.__dict__
        if state.get("_render_mode_logged", False):
            return
        mapper = state.get("_volume_mapper")
        if mapper is None:
            return
        mode = describe_render_mode(mapper)
        if mode in PENDING_RENDER_MODES and not final:
            return
        state["_render_mode_logged"] = True
        logger.info(
            "  %s render mode=%s (mapper=%s)",
            phase,
            mode,
            state.get("_mapper_kind", "?"),
        )
        _flush_logs()
        if mode == CPU_RAYCAST_MODE and state.get("_shrink_kind") == MAPPER_KIND_SMART:
            self._apply_cpu_raycast_fallback()

    def _apply_cpu_raycast_fallback(self) -> None:
        """Re-downsample after the smart mapper falls back to its CPU ray caster.

        ``vtkSmartVolumeMapper`` picks GPU ray casting only when it finds a
        usable context; over RDP, in a VM and on software GL it silently uses
        its own CPU caster instead.  The smart shrink table assumes the GPU is
        not the bottleneck and hands the "small" tier the volume at full
        resolution, so that fallback meant full-resolution CPU ray casting with
        ``AutoAdjustSampleDistances`` off -- several times the load the old
        fixed-point path ever carried, and a frozen GUI on every camera move.

        Detecting the mode and only logging it, which is what this used to do,
        told the log what the user was already suffering.  The volume is
        re-downsampled by the CPU table instead and the mapper is allowed to
        drop sample distances during interaction, which is exactly the deal the
        macOS fixed-point path takes.  Runs once: ``_shrink_kind`` is the flag.
        """
        self._shrink_kind = MAPPER_KIND_CPU
        mapper = self.__dict__.get("_volume_mapper")
        if mapper is None:
            return
        # The GPU path pins the sample distance for a stable frame time; a CPU
        # cast cannot afford that while the camera is moving.
        mapper.SetAutoAdjustSampleDistances(True)
        vtk_image = self.volume_manager.get_vtk_image()
        if vtk_image is None or not self.__dict__.get("_volume_added", False):
            return                       # nothing loaded yet; update_volume will use the new table
        tier = assess_volume_scale(self.volume_manager.dimensions).tier
        sample_dist, shrink = volume_render_settings(
            tier, self.volume_manager.spacing, MAPPER_KIND_CPU
        )
        logger.warning(
            "  smart mapper is ray casting on the CPU; re-downsampling "
            "tier=%s with shrink=%s sample_dist=%.2f",
            tier, shrink, sample_dist,
        )
        self._downsampled_image = downsample_vtk_image(vtk_image, shrink)
        mapper.SetInputData(self._downsampled_image)
        self._target_sample_dist = sample_dist
        mapper.SetSampleDistance(sample_dist)
        self._sync_screw_mpr_cut_volume_quality()
        # The cut volume's vtkExtractVOI reads from the old _downsampled_image
        # by identity; re-applying the cut (a no-op unless Screw MPR is
        # active) rebuilds it against the new one instead of leaving it to
        # render stale, mismatched data until the next Position step.
        if self.__dict__.get("_screw_mpr_active", False):
            self._apply_screw_mpr_cut(self.__dict__.get("_screw_mpr_cut_label"))
        self._request_render()
        _flush_logs()

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
            self._sync_screw_mpr_cut_volume_quality()
            logger.info("  Phase 2 render (sample_dist=%.2f)",
                        self._target_sample_dist)
        self._request_render()
        logger.info("  Phase 2 done")
        QTimer.singleShot(
            0, lambda: self._log_render_mode("Phase 2", final=True)
        )

    def _request_render(self):
        """Request a VTK render (shared dirty-flag path; see vtk_helpers)."""
        request_render(self.vtk_widget)

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
        self._sync_screw_mpr_cut_volume_quality()

        if self._volume_added:
            self._request_render()

    def set_volume_visible(self, visible: bool) -> None:
        """Toggle visibility of the volume rendering actor.

        Vertebral mesh and screw actors remain visible regardless, and so does
        the segmentation overlay: its visibility belongs to the "Show 3D"
        checkbox (``set_segmentation_visible``).  This used to switch the
        overlay too, so every caller that showed the volume -- restoring Full
        CT, or changing the checked levels in Full CT mode -- silently turned
        the overlay back on while the checkbox still read unchecked.  A caller
        that wants the overlay hidden with the volume, as isolation does, now
        says so itself.

        Args:
            visible: True to show volume rendering, False to hide.
        """
        if self._volume is not None:
            self._volume.SetVisibility(int(visible))
        # The Screw MPR cut volume always mirrors the main volume, so
        # isolated mode (volume hidden) never shows a stray cut box.
        cut_volume = self.__dict__.get("_screw_mpr_cut_volume")
        if cut_volume is not None:
            cut_volume.SetVisibility(int(visible))
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
        """Show or hide the MPR planes in the 3D viewport.

        While Screw MPR is active those are the three screw-aligned planes; the
        standard ones stay hidden, because they no longer describe any pane.
        """
        self._planes_visible = bool(visible)
        # Read through __dict__, as _log_render_mode does: a Viewer3D built with
        # __new__ (the tests do) has none of the Screw MPR state yet.
        state = self.__dict__
        screw_mpr_active = bool(state.get("_screw_mpr_active", False))
        standard_visible = self._planes_visible and not screw_mpr_active
        for plane_data in self._plane_actors.values():
            plane_data["actor"].SetVisibility(standard_visible)
            plane_data["outline_actor"].SetVisibility(standard_visible)
        oblique_visible = self._planes_visible and screw_mpr_active
        for name, plane_data in state.get("_screw_mpr_actors", {}).items():
            # The cross-section's FILLED quad stays hidden once the textured
            # slice actor covers the same plane, to avoid z-fighting between
            # the two; its outline still frames the slice.
            plane_data["actor"].SetVisibility(oblique_visible and name != "cross_section")
            plane_data["outline_actor"].SetVisibility(oblique_visible)

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
        self._invalidate_screw_picker()
        self._request_render()

        return visual

    def remove_screw(self, actor):
        """Remove a complete screw visual or a legacy single actor."""
        if isinstance(actor, ScrewVisual) and actor in self._screw_actors:
            for prop in actor.props:
                self._screw_prop_parts.pop(self._prop_key(prop), None)
                self._renderer.RemoveActor(prop)
            self._screw_actors.remove(actor)
            self._invalidate_screw_picker()
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
        self._invalidate_screw_picker()
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
        picker = shared_cell_picker(0.01)
        picker.PickFromListOn()
        for visual in self._screw_actors:
            for prop in visual.props:
                picker.AddPickList(prop)
        return picker

    def _screw_picker_cached(self) -> vtk.vtkCellPicker:
        """The screw picker, rebuilt only when the screw set changed."""
        if self._screw_picker is None:
            self._screw_picker = self._build_screw_only_picker()
        return self._screw_picker

    def _invalidate_screw_picker(self) -> None:
        """Drop the cached screw picker after screws are added or removed."""
        self._screw_picker = None

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

        picker = self._screw_picker_cached()
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
        # A new mask invalidates every cached label box, even one at the same
        # Python id as a previous (now-freed) mask.
        self._screw_mpr_label_box_cache = {}
        # A live cut was built against the old mask's label geometry: drop it
        # and let the slice fall back to full opacity (its alpha was resliced
        # from the old mask) until the next show_screw_mpr rebuilds both
        # against the new mask. Normal re-runs do get that show_screw_mpr;
        # an armed one-click edit, or a re-run that raises in between, do not.
        if self.__dict__.get("_screw_mpr_active", False):
            self._remove_screw_mpr_cut()
            axes = self.__dict__.get("_screw_mpr_cross_section_axes")
            if axes is not None:
                self._update_screw_mpr_slice(
                    axes, self.__dict__.get("_screw_mpr_window_level"), None
                )
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

        # Shared label filter (see vtk_helpers.label_threshold_filter).
        if self._segmentation_label_value <= 0:
            lower, upper = 1, 1000000
        else:
            lower = upper = self._segmentation_label_value
        threshold = label_threshold_filter(lower, upper)
        threshold.SetInputData(self._segmentation_mask_image)
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
        self._screw_mpr_label_box_cache = {}
        # No mask left to cut or shade the slice by: drop the local cut and
        # let the slice fall back to full opacity, even if Screw MPR is
        # still open.
        if self.__dict__.get("_screw_mpr_active", False):
            self._remove_screw_mpr_cut()
            axes = self.__dict__.get("_screw_mpr_cross_section_axes")
            if axes is not None:
                self._update_screw_mpr_slice(
                    axes, self.__dict__.get("_screw_mpr_window_level"), None
                )

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
        self._vertebral_mesh_poly = poly_data
        self._vertebral_mesh_split_cache = {}
        label = self.__dict__.get("_screw_mpr_cut_label")
        if label is not None:
            # A mesh rebuilt while Screw MPR is open must re-apply the same
            # cut: the main actor's mapper input just got replaced above, so
            # the previous split (main = 'rest') is gone until redone here.
            # Only the mesh split is redone -- the volume crop/extract did
            # not change and must not be rebuilt.
            self._apply_screw_mpr_mesh_cut(label)
        self._request_render()

        logger.info(
            "set_vertebral_mesh: added actor with %d cells",
            poly_data.GetNumberOfCells(),
        )

    def clear_vertebral_mesh(self) -> None:
        """Remove vertebral mesh actor from the 3D scene."""
        self._vertebral_mesh_poly = None
        self._vertebral_mesh_split_cache = {}
        if self.__dict__.get("_screw_mpr_cut_actor") is not None:
            # No mesh left to have split out a piece of.
            self._renderer.RemoveActor(self._screw_mpr_cut_actor)
            self._screw_mpr_cut_actor = None
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
            visible = bool(visible) and not self.__dict__.get("_screw_mpr_active", False)
            self._plane_actors[plane]["actor"].SetVisibility(visible)
            self._plane_actors[plane]["outline_actor"].SetVisibility(visible)
            self._request_render()

    # --- Screw MPR in 3D ---

    def show_screw_mpr(
        self,
        oblique_axial: vtk.vtkMatrix4x4,
        oblique_sagittal: vtk.vtkMatrix4x4,
        cross_section: vtk.vtkMatrix4x4,
        *,
        vertebra_label: Optional[int] = None,
        window_level: Optional[Tuple[float, float]] = None,
    ) -> None:
        """Show the screw-aligned planes and a textured cut through one vertebra.

        Screw MPR used to change only the three MPR panes; the 3D view went on
        drawing the standard axial, sagittal and coronal planes, which no pane
        showed any more. It used to also clip the WHOLE volume and the WHOLE
        vertebral mesh at one infinite plane, which for an oblique screw (e.g.
        S1, craniocaudal -25 deg) removed everything cranial to the cut and,
        at any level, stripped the posterior elements of every vertebra --
        not just the screw's own.

        Each matrix argument is a reslice-axes matrix exactly as the MPR panes
        receive it -- columns are screen-right, screen-up and the normal, and
        the last column is the plane centre -- so the 3D indicators and the
        panes are drawn from the same numbers. ``cross_section`` additionally
        drives a real textured CT slice (``_screw_mpr_slice_actor``) placed at
        that matrix via ``SetUserMatrix``, so it follows **Position**,
        rotation and offsets exactly like the cross-section pane.

        ``vertebra_label`` (a TotalSegmentator label, e.g. 28 for L4) is the
        ONLY thing that gets cut, and only when it resolves to real voxels in
        the current segmentation mask: the main volume mapper is cropped to
        exclude that vertebra's box (region outside the box stays), and a
        separate cut volume + cut mesh actor render that box/that vertebra's
        cells clipped at the cross-section plane. Every other level -- and
        the screw actors themselves -- are left untouched. With no resolvable
        label, the slice still shows but nothing is cut.

        The optional raw 3D segmentation overlay actor (``_segmentation_actor``,
        set_segmentation_mask) is never cut here -- only the vertebral mesh and
        the volume are.
        """
        if self._renderer is None:
            return
        self._screw_mpr_active = True
        for name, axes in (
            ("oblique_axial", oblique_axial),
            ("oblique_sagittal", oblique_sagittal),
            ("cross_section", cross_section),
        ):
            self._place_screw_mpr_plane(name, axes)
        for plane_data in self._plane_actors.values():
            plane_data["actor"].SetVisibility(False)
            plane_data["outline_actor"].SetVisibility(False)
        for name, plane_data in self._screw_mpr_actors.items():
            # The cross-section's filled quad stays hidden -- the textured
            # slice actor covers that plane instead, and drawing both would
            # z-fight. Its outline stays, framing the slice.
            plane_data["actor"].SetVisibility(
                self._planes_visible and name != "cross_section"
            )
            plane_data["outline_actor"].SetVisibility(self._planes_visible)

        centre = tuple(cross_section.GetElement(row, 3) for row in range(3))
        normal = tuple(cross_section.GetElement(row, 2) for row in range(3))
        if self._screw_mpr_clip is None:
            self._screw_mpr_clip = vtk.vtkPlane()
        self._screw_mpr_clip.SetOrigin(*centre)
        self._screw_mpr_clip.SetNormal(*normal)
        self._screw_mpr_clip.Modified()
        self._screw_mpr_cross_section_axes = cross_section
        self._screw_mpr_window_level = window_level

        resolved_label = self._resolve_screw_mpr_label(vertebra_label)
        self._update_screw_mpr_slice(cross_section, window_level, resolved_label)
        self._apply_screw_mpr_cut(resolved_label)
        button = self.__dict__.get("screw_mpr_view_button")
        if button is not None:
            button.setVisible(True)
        self._request_render()

    def clear_screw_mpr(self) -> None:
        """Put the standard planes back, drop the slice, and undo the cut."""
        if not self._screw_mpr_active and self._screw_mpr_clip is None:
            return
        self._screw_mpr_active = False
        self._screw_mpr_clip = None
        self._remove_screw_mpr_cut()
        if self._screw_mpr_slice_actor is not None:
            self._screw_mpr_slice_actor.SetVisibility(False)
        for plane_data in self._screw_mpr_actors.values():
            plane_data["actor"].SetVisibility(False)
            plane_data["outline_actor"].SetVisibility(False)
        for plane_data in self._plane_actors.values():
            plane_data["actor"].SetVisibility(self._planes_visible)
            plane_data["outline_actor"].SetVisibility(self._planes_visible)
        button = self.__dict__.get("screw_mpr_view_button")
        if button is not None:
            button.setVisible(False)
        if self._renderer is not None:
            self._request_render()

    @property
    def screw_mpr_active(self) -> bool:
        """Whether the 3D view is showing the screw-aligned planes."""
        return self._screw_mpr_active

    def _resolve_screw_mpr_label(self, vertebra_label: Optional[int]) -> Optional[int]:
        """Only cut with a label that actually has voxels in the current mask.

        A stale or mistaken label (no segmentation loaded, or a level not
        present in it) must fall back to "slice only, no cut" rather than
        cropping/clipping at a meaningless empty box.
        """
        mask = self.__dict__.get("_segmentation_mask_image")
        if vertebra_label is None or mask is None:
            return None
        if self._cached_label_bounds(mask, int(vertebra_label)) is None:
            return None
        return int(vertebra_label)

    def _cached_label_bounds(
        self, mask: vtk.vtkImageData, label: int
    ) -> Optional[Tuple[float, float, float, float, float, float]]:
        """Unpadded world bounds of ``label``'s voxels, scanned at most once.

        ``label_world_bounds`` runs scipy's ``find_objects`` over the whole
        mask; show_screw_mpr is called on every Position/rotation/offset
        step, so an uncached call would re-scan a large mask on every notch.
        Keyed on the mask's Python id together with the label. A freed
        mask's id can be reused by a later object, so the cache is only
        safe because set_segmentation_mask and clear_segmentation_mask both
        drop it outright whenever the mask changes.
        """
        # __dict__.setdefault, not a plain attribute read, because Viewer3D
        # is built with __new__ in tests and this attribute may not exist yet.
        cache = self.__dict__.setdefault("_screw_mpr_label_box_cache", {})
        key = (id(mask), int(label))
        if key not in cache:
            cache[key] = label_world_bounds(mask, int(label))
        return cache[key]

    def _cached_padded_label_bounds(
        self, mask: vtk.vtkImageData, label: int, margin_mm: float
    ) -> Optional[Tuple[float, float, float, float, float, float]]:
        """Padded box for the volume crop, derived from the cached unpadded box.

        Adding the margin here (instead of caching one entry per margin
        value) reuses the single scipy scan that ``_resolve_screw_mpr_label``
        already performed for this label.
        """
        bounds = self._cached_label_bounds(mask, label)
        if bounds is None:
            return None
        xmin, xmax, ymin, ymax, zmin, zmax = bounds
        mask_bounds = mask.GetBounds()
        return (
            max(xmin - margin_mm, mask_bounds[0]),
            min(xmax + margin_mm, mask_bounds[1]),
            max(ymin - margin_mm, mask_bounds[2]),
            min(ymax + margin_mm, mask_bounds[3]),
            max(zmin - margin_mm, mask_bounds[4]),
            min(zmax + margin_mm, mask_bounds[5]),
        )

    # --- Screw MPR: textured cross-section slice ---

    def _update_screw_mpr_slice(
        self,
        cross_section: vtk.vtkMatrix4x4,
        window_level: Optional[Tuple[float, float]],
        resolved_label: Optional[int],
    ) -> None:
        """(Re)build the textured CT slice at the cross-section plane.

        Always resliced from the volume manager's full-resolution CT -- even
        in isolated mode, where the volume itself is hidden -- so the tissue
        around the screw's own vertebra reads as real anatomy, not the
        downsampled render volume or a vertebra-only crop.
        """
        volume_manager = self.__dict__.get("volume_manager")
        ct_image = None
        if volume_manager is not None:
            try:
                ct_image = volume_manager.get_vtk_image()
            except Exception:
                ct_image = None
        if ct_image is None:
            return

        spacing = [abs(float(value)) for value in ct_image.GetSpacing()]
        in_plane_spacing = max(min(spacing), 1e-3)
        half = SCREW_MPR_SLICE_HALF_SIZE_MM
        n = max(1, int(math.ceil((2.0 * half) / in_plane_spacing)))
        extent = (0, n - 1, 0, n - 1, 0, 0)
        origin = (-half, -half, 0.0)

        reslice = self._screw_mpr_slice_reslice
        if reslice is None:
            reslice = vtk.vtkImageReslice()
            reslice.SetOutputDimensionality(2)
            reslice.SetInterpolationModeToLinear()
            reslice.SetBackgroundLevel(-1000.0)
            self._screw_mpr_slice_reslice = reslice
        reslice.SetInputData(ct_image)
        reslice.SetResliceAxes(cross_section)
        reslice.SetOutputSpacing(in_plane_spacing, in_plane_spacing, in_plane_spacing)
        reslice.SetOutputOrigin(*origin)
        reslice.SetOutputExtent(*extent)

        color_map = self._screw_mpr_slice_color_map
        if color_map is None:
            color_map = vtk.vtkImageMapToWindowLevelColors()
            color_map.SetOutputFormatToRGB()
            self._screw_mpr_slice_color_map = color_map
        color_map.SetInputConnection(reslice.GetOutputPort())
        window, level = (
            window_level if window_level is not None else SCREW_MPR_DEFAULT_WINDOW_LEVEL
        )
        color_map.SetWindow(float(window))
        color_map.SetLevel(float(level))
        color_map.Update()

        alpha_source = self._screw_mpr_slice_alpha_source(
            cross_section, resolved_label, in_plane_spacing, extent, origin
        )

        appender = vtk.vtkImageAppendComponents()
        appender.AddInputConnection(color_map.GetOutputPort())
        if isinstance(alpha_source, vtk.vtkImageData):
            appender.AddInputData(alpha_source)
        else:
            appender.AddInputConnection(alpha_source)
        appender.Update()

        slice_actor = self._screw_mpr_slice_actor
        if slice_actor is None:
            slice_actor = vtk.vtkImageActor()
            slice_actor.PickableOff()
            self._renderer.AddActor(slice_actor)
            self._screw_mpr_slice_actor = slice_actor
        slice_actor.SetInputData(appender.GetOutput())
        slice_actor.SetUserMatrix(cross_section)
        slice_actor.SetVisibility(True)

    def _screw_mpr_slice_alpha_source(
        self,
        cross_section: vtk.vtkMatrix4x4,
        resolved_label: Optional[int],
        in_plane_spacing: float,
        extent: Tuple[int, int, int, int, int, int],
        origin: Tuple[float, float, float],
    ):
        """Build the slice's alpha channel: opaque over the cut vertebra.

        Returns either a vtkImageData (constant full-opacity fallback) or an
        output port (the resliced-mask threshold), so the caller can feed
        either into vtkImageAppendComponents without branching twice.
        """
        mask = self.__dict__.get("_segmentation_mask_image")
        if mask is None or resolved_label is None:
            alpha_image = vtk.vtkImageData()
            alpha_image.SetExtent(*extent)
            alpha_image.SetSpacing(in_plane_spacing, in_plane_spacing, in_plane_spacing)
            alpha_image.SetOrigin(*origin)
            alpha_image.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 1)
            alpha_image.GetPointData().GetScalars().Fill(255)
            return alpha_image

        mask_reslice = self._screw_mpr_slice_mask_reslice
        if mask_reslice is None:
            mask_reslice = vtk.vtkImageReslice()
            mask_reslice.SetOutputDimensionality(2)
            mask_reslice.SetInterpolationModeToNearestNeighbor()
            mask_reslice.SetBackgroundLevel(0.0)
            self._screw_mpr_slice_mask_reslice = mask_reslice
        mask_reslice.SetInputData(mask)
        mask_reslice.SetResliceAxes(cross_section)
        mask_reslice.SetOutputSpacing(in_plane_spacing, in_plane_spacing, in_plane_spacing)
        mask_reslice.SetOutputOrigin(*origin)
        mask_reslice.SetOutputExtent(*extent)

        context_alpha = round(SCREW_MPR_SLICE_CONTEXT_ALPHA * 255)
        alpha_filter = self._screw_mpr_slice_alpha_filter
        if alpha_filter is None:
            alpha_filter = vtk.vtkImageThreshold()
            self._screw_mpr_slice_alpha_filter = alpha_filter
        alpha_filter.SetInputConnection(mask_reslice.GetOutputPort())
        alpha_filter.ThresholdBetween(float(resolved_label), float(resolved_label))
        alpha_filter.SetInValue(255)
        alpha_filter.SetOutValue(context_alpha)
        alpha_filter.SetOutputScalarTypeToUnsignedChar()
        alpha_filter.Update()
        return alpha_filter.GetOutputPort()

    # --- Screw MPR: local vertebra cut (mesh + volume) ---

    def _apply_screw_mpr_cut(self, resolved_label: Optional[int]) -> None:
        """Cut only ``resolved_label``'s vertebra, restoring any earlier cut.

        Called on every show_screw_mpr -- including Position/rotation-only
        updates, so it must be cheap when the label has not changed: the mesh
        split and the volume crop/extract are cached and only rebuilt when
        the label (or the downsampled volume identity, for the cut volume)
        actually changes.
        """
        if resolved_label != self._screw_mpr_cut_label:
            self._remove_screw_mpr_cut()
            self._screw_mpr_cut_label = resolved_label

        if resolved_label is None:
            return

        self._apply_screw_mpr_mesh_cut(resolved_label)
        self._apply_screw_mpr_volume_cut(resolved_label)

    def _apply_screw_mpr_mesh_cut(self, label: int) -> None:
        """Split the combined vertebral mesh so only ``label`` is clippable."""
        full_poly = self._vertebral_mesh_poly
        mesh_actor = self._vertebral_mesh_actor
        if full_poly is None or mesh_actor is None or mesh_actor.GetMapper() is None:
            return

        split = self._vertebral_mesh_split_cache.get(label)
        if split is None:
            matching, rest = split_mesh_by_label(full_poly, label)
            split = (matching, rest)
            self._vertebral_mesh_split_cache[label] = split
        matching, rest = split

        mesh_actor.GetMapper().SetInputData(rest)

        cut_actor = self._screw_mpr_cut_actor
        if cut_actor is None:
            cut_mapper = vtk.vtkPolyDataMapper()
            cut_actor = vtk.vtkActor()
            cut_actor.SetMapper(cut_mapper)
            # Share the main actor's property so opacity/focus changes on the
            # vertebral mesh apply to the cut piece too.
            cut_actor.SetProperty(mesh_actor.GetProperty())
            self._renderer.AddActor(cut_actor)
            self._screw_mpr_cut_actor = cut_actor
        cut_actor.GetMapper().SetInputData(matching)
        cut_actor.GetMapper().SetScalarModeToUseCellData()
        cut_actor.GetMapper().SelectColorArray("VertebraColors")
        cut_actor.GetMapper().ScalarVisibilityOn()
        # The cut actor's ONLY clipping plane -- the main mesh mapper never
        # carries one any more.
        clipping = cut_actor.GetMapper().GetClippingPlanes()
        if clipping is None or clipping.GetNumberOfItems() == 0:
            cut_actor.GetMapper().AddClippingPlane(self._screw_mpr_clip)

    def _apply_screw_mpr_volume_cut(self, label: int) -> None:
        """Crop the main volume to exclude ``label``'s box; render it separately."""
        mask = self.__dict__.get("_segmentation_mask_image")
        bounds = (
            self._cached_padded_label_bounds(mask, label, SCREW_MPR_CUT_MARGIN_MM)
            if mask is not None
            else None
        )
        if bounds is None or self._volume_mapper is None:
            return

        downsampled = self.__dict__.get("_downsampled_image")
        if downsampled is None:
            # No cut volume can be built to fill the hole a crop would leave
            # in the main volume, so leave the main volume uncropped rather
            # than cutting a chunk out of it with nothing standing in.
            return

        if (
            self._screw_mpr_cut_volume is None
            or self._screw_mpr_cut_volume_source is not downsampled
        ):
            self._build_screw_mpr_cut_volume(downsampled, bounds)
        else:
            self._update_screw_mpr_cut_volume_box(downsampled, bounds)
        self._sync_screw_mpr_cut_volume_quality()

        # Only now that a cut volume exists to fill the hole: crop the main
        # volume outside the box.
        self._volume_mapper.SetCropping(True)
        self._volume_mapper.SetCroppingRegionPlanes(*bounds)
        self._volume_mapper.SetCroppingRegionFlags(VOLUME_CROP_OUTSIDE_BOX)

        if self._screw_mpr_cut_volume is not None and self._volume is not None:
            self._screw_mpr_cut_volume.SetVisibility(self._volume.GetVisibility())

    def _voi_for_bounds(
        self,
        image: vtk.vtkImageData,
        bounds: Tuple[float, float, float, float, float, float],
    ) -> Tuple[int, int, int, int, int, int]:
        """Convert a world-space box into an image's clamped index extent."""
        origin = image.GetOrigin()
        spacing = image.GetSpacing()
        extent = image.GetExtent()
        xmin, xmax, ymin, ymax, zmin, zmax = bounds
        indices = []
        for lo, hi, o, s, e_lo, e_hi in (
            (xmin, xmax, origin[0], spacing[0], extent[0], extent[1]),
            (ymin, ymax, origin[1], spacing[1], extent[2], extent[3]),
            (zmin, zmax, origin[2], spacing[2], extent[4], extent[5]),
        ):
            if s == 0:
                indices.extend([e_lo, e_hi])
                continue
            i0 = (lo - o) / s
            i1 = (hi - o) / s
            if i0 > i1:
                i0, i1 = i1, i0
            i0 = max(e_lo, min(e_hi, int(math.floor(i0))))
            i1 = max(e_lo, min(e_hi, int(math.ceil(i1))))
            indices.extend([i0, i1])
        return tuple(indices)

    def _build_screw_mpr_cut_volume(
        self,
        downsampled: vtk.vtkImageData,
        bounds: Tuple[float, float, float, float, float, float],
    ) -> None:
        """Create the cut volume: the vertebra's box, clipped at the cross-section."""
        if self._screw_mpr_cut_volume is not None:
            self._renderer.RemoveVolume(self._screw_mpr_cut_volume)

        extract = vtk.vtkExtractVOI()
        extract.SetInputData(downsampled)
        extract.SetVOI(*self._voi_for_bounds(downsampled, bounds))

        mapper = create_volume_mapper(self._mapper_kind)
        mapper.SetInputConnection(extract.GetOutputPort())
        mapper.AddClippingPlane(self._screw_mpr_clip)

        cut_volume = vtk.vtkVolume()
        cut_volume.SetMapper(mapper)
        if self._volume_property is not None:
            cut_volume.SetProperty(self._volume_property)
        if self._volume is not None:
            cut_volume.SetVisibility(self._volume.GetVisibility())

        self._renderer.AddVolume(cut_volume)
        self._screw_mpr_cut_volume = cut_volume
        self._screw_mpr_cut_volume_source = downsampled
        self._screw_mpr_cut_volume_extract = extract
        # Sample distance and blend mode set once here would drift from the
        # main mapper's the moment interaction LOD, Phase 1/2, or a transfer
        # function preset changes it; mirror them explicitly on every build.
        self._sync_screw_mpr_cut_volume_quality()

    def _sync_screw_mpr_cut_volume_quality(self) -> None:
        """Mirror the main volume mapper's sample distance and blend mode.

        The cut volume must always render at the same quality and in the
        same mode as its surroundings, so it is re-synced everywhere the
        main mapper's sample distance or blend mode changes (interaction
        start/end, Phase 1/2 LOD, a transfer function preset) as well as
        right after the cut volume is (re)built.
        """
        main_mapper = self.__dict__.get("_volume_mapper")
        cut_volume = self.__dict__.get("_screw_mpr_cut_volume")
        if main_mapper is None or cut_volume is None:
            return
        cut_mapper = cut_volume.GetMapper()
        if cut_mapper is None:
            return
        if hasattr(main_mapper, "GetSampleDistance") and hasattr(cut_mapper, "SetSampleDistance"):
            try:
                cut_mapper.SetSampleDistance(main_mapper.GetSampleDistance())
            except AttributeError:
                pass
        if hasattr(main_mapper, "GetBlendMode") and hasattr(cut_mapper, "SetBlendMode"):
            try:
                cut_mapper.SetBlendMode(main_mapper.GetBlendMode())
            except AttributeError:
                pass

    def _update_screw_mpr_cut_volume_box(
        self,
        downsampled: vtk.vtkImageData,
        bounds: Tuple[float, float, float, float, float, float],
    ) -> None:
        """Move the existing cut volume's box (same label's crop, e.g. Position moved)."""
        extract = self.__dict__.get("_screw_mpr_cut_volume_extract")
        if extract is None:
            self._build_screw_mpr_cut_volume(downsampled, bounds)
            return
        extract.SetVOI(*self._voi_for_bounds(downsampled, bounds))

    def _remove_screw_mpr_cut(self) -> None:
        """Undo the local cut: restore the full mesh, drop the cut volume/actor."""
        mesh_actor = self._vertebral_mesh_actor
        if mesh_actor is not None and mesh_actor.GetMapper() is not None and self._vertebral_mesh_poly is not None:
            mesh_actor.GetMapper().SetInputData(self._vertebral_mesh_poly)
        if self._screw_mpr_cut_actor is not None:
            self._renderer.RemoveActor(self._screw_mpr_cut_actor)
            self._screw_mpr_cut_actor = None
        if self._volume_mapper is not None:
            self._volume_mapper.SetCropping(False)
        if self._screw_mpr_cut_volume is not None:
            self._renderer.RemoveVolume(self._screw_mpr_cut_volume)
            self._screw_mpr_cut_volume = None
        self._screw_mpr_cut_volume_source = None
        self.__dict__.pop("_screw_mpr_cut_volume_extract", None)
        self._screw_mpr_cut_label = None

    def focus_screw_mpr_cut(self) -> None:
        """Point the camera down the screw at the cross-section, from entry side.

        Never called automatically -- only from the "Cut View" button -- so a
        surgeon who has framed a shot manually is never yanked out of it.
        """
        clip = self._screw_mpr_clip
        if clip is None or self._renderer is None:
            return
        centre = clip.GetOrigin()
        normal = clip.GetNormal()
        up = None
        axes = self.__dict__.get("_screw_mpr_cross_section_axes")
        if axes is not None:
            up = tuple(axes.GetElement(row, 1) for row in range(3))
        camera = self._renderer.GetActiveCamera()
        position = tuple(
            centre[i] - normal[i] * SCREW_MPR_CUT_VIEW_DISTANCE_MM for i in range(3)
        )
        camera.SetFocalPoint(*centre)
        camera.SetPosition(*position)
        if up is not None and any(up):
            camera.SetViewUp(*up)
        self._renderer.ResetCameraClippingRange()
        self._request_render()

    def _place_screw_mpr_plane(self, name: str, axes: vtk.vtkMatrix4x4) -> None:
        """Create (once) and position one screw-aligned plane indicator."""
        data = self._screw_mpr_actors.get(name)
        if data is None:
            color = SCREW_MPR_PLANE_COLORS[name]
            source = vtk.vtkPlaneSource()
            source.SetResolution(1, 1)
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(source.GetOutputPort())
            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            actor.GetProperty().SetColor(*(c / 255.0 for c in color))
            actor.GetProperty().SetOpacity(PLANE_INDICATOR_OPACITY)
            actor.GetProperty().SetLighting(False)
            outline = vtk.vtkOutlineFilter()
            outline.SetInputConnection(source.GetOutputPort())
            outline_mapper = vtk.vtkPolyDataMapper()
            outline_mapper.SetInputConnection(outline.GetOutputPort())
            outline_actor = vtk.vtkActor()
            outline_actor.SetMapper(outline_mapper)
            outline_actor.GetProperty().SetColor(*(c / 255.0 for c in color))
            outline_actor.GetProperty().SetOpacity(PLANE_OUTLINE_OPACITY)
            outline_actor.GetProperty().SetLineWidth(1.5)
            self._renderer.AddActor(actor)
            self._renderer.AddActor(outline_actor)
            data = {"source": source, "actor": actor, "outline_actor": outline_actor}
            self._screw_mpr_actors[name] = data

        centre = [axes.GetElement(row, 3) for row in range(3)]
        right = [axes.GetElement(row, 0) for row in range(3)]
        up = [axes.GetElement(row, 1) for row in range(3)]
        half = SCREW_MPR_PLANE_HALF_SIZE_MM
        source = data["source"]
        source.SetOrigin(*(centre[i] - half * right[i] - half * up[i] for i in range(3)))
        source.SetPoint1(*(centre[i] + half * right[i] - half * up[i] for i in range(3)))
        source.SetPoint2(*(centre[i] - half * right[i] + half * up[i] for i in range(3)))
        source.Update()

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
