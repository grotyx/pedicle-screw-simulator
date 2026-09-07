"""
Volume Manager - Central volume data management and MPR/3D coordination

This is the SINGLE SOURCE OF TRUTH for volume data. All viewports reference
this manager to ensure coordinate consistency.
"""

import logging
import vtk
import SimpleITK as sitk
from typing import Optional, Tuple, Dict, Callable

logger = logging.getLogger(__name__)

from ..utils.vtk_helpers import sitk_to_vtk, create_reslice_axes
from ..utils.constants import TRANSFER_FUNCTION_PRESETS


class VolumeManager:
    """
    Central manager for volume data, ensuring MPR-3D coordinate consistency.

    CRITICAL: All viewports must reference this single VolumeManager instance
    to maintain synchronized coordinates.

    Architecture:
    - Single VTK vtkImageData instance shared by all viewers
    - Consistent origin, spacing, direction for all operations
    - Centralized slice position management
    - Transfer function preset management for CPU volume rendering
    - Event system for viewport synchronization
    """

    def __init__(self):
        self._vtk_image: Optional[vtk.vtkImageData] = None
        self._sitk_image: Optional[sitk.Image] = None

        # Volume geometry (CRITICAL - single source of truth)
        self._origin: Tuple[float, float, float] = (0, 0, 0)
        self._spacing: Tuple[float, float, float] = (1, 1, 1)
        self._dimensions: Tuple[int, int, int] = (0, 0, 0)
        self._bounds: Tuple[float, ...] = (0, 0, 0, 0, 0, 0)
        self._center: Tuple[float, float, float] = (0, 0, 0)

        # Current slice positions (world coordinates)
        self._slice_positions: Dict[str, float] = {
            "axial": 0.0,
            "sagittal": 0.0,
            "coronal": 0.0,
        }

        # Vertebral-only masked volume (set by segmentation controller)
        self._vertebral_only_image: Optional[vtk.vtkImageData] = None

        # Transfer function preset for CPU volume rendering
        self._transfer_function_preset: str = "Bone"

        # Observers for viewport synchronization
        self._observers: Dict[str, Callable] = {}

    def set_volume(self, sitk_image: sitk.Image) -> None:
        """
        Set the volume data from SimpleITK image.

        This converts to VTK format and caches all geometry information.

        Args:
            sitk_image: SimpleITK Image to use as volume data
        """
        self._sitk_image = sitk_image
        self._vtk_image = sitk_to_vtk(sitk_image)

        # Cache geometry from VTK image (SINGLE SOURCE OF TRUTH)
        self._dimensions = self._vtk_image.GetDimensions()
        self._spacing = self._vtk_image.GetSpacing()
        self._origin = self._vtk_image.GetOrigin()
        self._bounds = self._vtk_image.GetBounds()

        # Calculate center
        self._center = (
            (self._bounds[0] + self._bounds[1]) / 2,
            (self._bounds[2] + self._bounds[3]) / 2,
            (self._bounds[4] + self._bounds[5]) / 2,
        )

        # Initialize slice positions to center
        self._slice_positions = {
            "axial": self._center[2],
            "sagittal": self._center[0],
            "coronal": self._center[1],
        }

        # Notify observers
        self._notify_observers("volume_loaded")

    def get_vtk_image(self) -> Optional[vtk.vtkImageData]:
        """Get the VTK image data (shared reference)."""
        return self._vtk_image

    def get_sitk_image(self) -> Optional[sitk.Image]:
        """Get the SimpleITK image."""
        return self._sitk_image

    def set_vertebral_mask(self, mask_vtk: vtk.vtkImageData) -> None:
        """Create and store a vertebral-only masked volume.

        Args:
            mask_vtk: Multilabel segmentation mask from TotalSegmentator.
        """
        from ..core.vertebral_mesh import create_vertebral_only_volume
        self._vertebral_only_image = create_vertebral_only_volume(
            self._vtk_image, mask_vtk
        )

    def get_vertebral_only_image(self) -> Optional[vtk.vtkImageData]:
        """Return the vertebral-only masked volume, or None if not created."""
        return self._vertebral_only_image

    def clear_vertebral_mask(self) -> None:
        """Discard the vertebral-only masked volume."""
        self._vertebral_only_image = None

    @property
    def origin(self) -> Tuple[float, float, float]:
        """Volume origin in world coordinates."""
        return self._origin

    @property
    def spacing(self) -> Tuple[float, float, float]:
        """Voxel spacing (mm)."""
        return self._spacing

    @property
    def dimensions(self) -> Tuple[int, int, int]:
        """Volume dimensions (voxels)."""
        return self._dimensions

    @property
    def bounds(self) -> Tuple[float, ...]:
        """Volume bounds (x_min, x_max, y_min, y_max, z_min, z_max)."""
        return self._bounds

    @property
    def center(self) -> Tuple[float, float, float]:
        """Volume center in world coordinates."""
        return self._center

    def get_slice_position(self, plane: str) -> float:
        """Get current slice position for given plane."""
        return self._slice_positions.get(plane, 0.0)

    def set_slice_position(self, plane: str, position: float) -> None:
        """
        Set slice position and notify observers.

        Args:
            plane: 'axial', 'sagittal', or 'coronal'
            position: World coordinate position
        """
        if plane not in self._slice_positions:
            raise ValueError(f"Unknown plane: {plane}")

        # Clamp to bounds
        if plane == "axial":
            position = max(self._bounds[4], min(self._bounds[5], position))
        elif plane == "sagittal":
            position = max(self._bounds[0], min(self._bounds[1], position))
        elif plane == "coronal":
            position = max(self._bounds[2], min(self._bounds[3], position))

        self._slice_positions[plane] = position
        self._notify_observers("slice_changed", plane=plane, position=position)

    def get_crosshair_position(self) -> Tuple[float, float, float]:
        """
        Get current crosshair position (intersection of all slice planes).

        Returns:
            (x, y, z) world coordinates
        """
        return (
            self._slice_positions["sagittal"],
            self._slice_positions["coronal"],
            self._slice_positions["axial"],
        )

    def set_crosshair_position(
        self,
        x: float,
        y: float,
        z: float,
        source_plane: Optional[str] = None
    ) -> None:
        """
        Set crosshair position and update all slice positions.

        Args:
            x, y, z: World coordinates
            source_plane: Plane that initiated the change (won't be notified)
        """
        self._slice_positions["sagittal"] = max(
            self._bounds[0], min(self._bounds[1], x)
        )
        self._slice_positions["coronal"] = max(
            self._bounds[2], min(self._bounds[3], y)
        )
        self._slice_positions["axial"] = max(
            self._bounds[4], min(self._bounds[5], z)
        )

        for plane in ("sagittal", "coronal", "axial"):
            if plane != source_plane:
                self._notify_observers(
                    "slice_changed", plane=plane, position=self._slice_positions[plane]
                )

        self._notify_observers(
            "crosshair_changed",
            position=(x, y, z),
            source=source_plane
        )

    def create_mpr_reslice(self, plane: str) -> vtk.vtkImageReslice:
        """
        Create a vtkImageReslice configured for the specified plane.

        CRITICAL: This ensures consistent coordinate handling across all MPR views.

        Args:
            plane: 'axial', 'sagittal', or 'coronal'

        Returns:
            Configured vtkImageReslice ready for connection
        """
        if self._vtk_image is None:
            raise RuntimeError("No volume loaded")

        reslice = vtk.vtkImageReslice()
        reslice.SetInputData(self._vtk_image)
        reslice.SetOutputDimensionality(2)
        reslice.SetInterpolationModeToLinear()
        reslice.SetBackgroundLevel(-1000)  # Air HU

        # Create reslice axes for this plane
        position = self._slice_positions[plane]
        if plane == "axial":
            center = (self._center[0], self._center[1], position)
        elif plane == "sagittal":
            center = (position, self._center[1], self._center[2])
        else:  # coronal
            center = (self._center[0], position, self._center[2])

        axes = create_reslice_axes(plane, center)
        reslice.SetResliceAxes(axes)

        return reslice

    # --- Transfer Function Presets ---

    @property
    def transfer_function_preset(self) -> str:
        """Current transfer function preset name."""
        return self._transfer_function_preset

    def set_transfer_function_preset(self, name: str) -> None:
        """
        Set the active transfer function preset.

        Args:
            name: Preset name (must exist in TRANSFER_FUNCTION_PRESETS)

        Raises:
            ValueError: if name is not a valid preset
        """
        if name not in TRANSFER_FUNCTION_PRESETS:
            raise ValueError(
                f"Unknown transfer function preset: '{name}'. "
                f"Valid presets: {list(TRANSFER_FUNCTION_PRESETS.keys())}"
            )
        self._transfer_function_preset = name
        self._notify_observers("transfer_function_changed", preset=name)

    def get_transfer_function_config(self) -> dict:
        """Get the current transfer function configuration dict."""
        return TRANSFER_FUNCTION_PRESETS[self._transfer_function_preset]

    # --- Observer System ---

    def add_observer(self, name: str, callback: Callable) -> None:
        """Add an observer for volume events."""
        self._observers[name] = callback

    def remove_observer(self, name: str) -> None:
        """Remove an observer."""
        self._observers.pop(name, None)

    def _notify_observers(self, event: str, **kwargs) -> None:
        """Notify all observers of an event."""
        for callback in self._observers.values():
            try:
                callback(event, **kwargs)
            except Exception as e:
                logger.exception("Observer notification failed")

    # --- Coordinate Conversion ---

    def world_to_index(
        self,
        x: float,
        y: float,
        z: float
    ) -> Tuple[int, int, int]:
        """
        Convert world coordinates to voxel indices.

        Uses the volume's cached geometry for consistency.
        """
        i = int(round((x - self._origin[0]) / self._spacing[0]))
        j = int(round((y - self._origin[1]) / self._spacing[1]))
        k = int(round((z - self._origin[2]) / self._spacing[2]))

        # Clamp to valid range
        i = max(0, min(self._dimensions[0] - 1, i))
        j = max(0, min(self._dimensions[1] - 1, j))
        k = max(0, min(self._dimensions[2] - 1, k))

        return (i, j, k)

    def index_to_world(
        self,
        i: int,
        j: int,
        k: int
    ) -> Tuple[float, float, float]:
        """
        Convert voxel indices to world coordinates.

        Uses the volume's cached geometry for consistency.
        """
        x = self._origin[0] + i * self._spacing[0]
        y = self._origin[1] + j * self._spacing[1]
        z = self._origin[2] + k * self._spacing[2]

        return (x, y, z)

    def get_voxel_value(self, x: float, y: float, z: float) -> float:
        """
        Get voxel value at world coordinate.

        Returns HU value at the given position.
        """
        if self._vtk_image is None:
            return -1000  # Air

        i, j, k = self.world_to_index(x, y, z)
        return self._vtk_image.GetScalarComponentAsFloat(i, j, k, 0)
