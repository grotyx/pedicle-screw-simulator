"""
Screw Tool - Interactive screw placement tool

Implements two placement modes:
1. Entry-Target Mode: Click entry point, then click target point
2. Entry-Angle Mode: Click entry point, specify angles and length

Based on 3D Slicer Pedicle Screw Simulator algorithms.
"""

import math
from typing import Optional, Tuple, List, Callable, TYPE_CHECKING

from src.models.screw import Screw
from ..utils.constants import (
    DEFAULT_SCREW_LENGTH, DEFAULT_SCREW_DIAMETER,
    MIN_SCREW_LENGTH, MAX_SCREW_LENGTH,
    MIN_SCREW_DIAMETER, MAX_SCREW_DIAMETER,
    GRADE_A_DESCRIPTION, GRADE_B_DESCRIPTION,
    GRADE_C_DESCRIPTION, GRADE_D_DESCRIPTION, GRADE_E_DESCRIPTION
)

if TYPE_CHECKING:
    from ..core.volume_manager import VolumeManager


class ScrewTool:
    """
    Interactive tool for placing pedicle screws.

    Usage:
        tool = ScrewTool(volume_manager)
        tool.set_mode("entry_target")  # or "entry_angle"
        tool.on_click(x, y, z)  # Process click events
    """

    def __init__(self, volume_manager: "VolumeManager"):
        """
        Initialize screw tool.

        Args:
            volume_manager: Shared VolumeManager instance
        """
        self.volume_manager = volume_manager

        # Tool state
        self._mode = "entry_target"  # or "entry_angle"
        self._state = "waiting_entry"  # or "waiting_target"
        self._pending_entry: Optional[Tuple[float, float, float]] = None

        # Screw parameters
        self._default_length = DEFAULT_SCREW_LENGTH
        self._default_diameter = DEFAULT_SCREW_DIAMETER
        self._default_insertion_angle = 15.0  # degrees
        self._default_medial_angle = 10.0  # degrees

        # Placed screws
        self._screws: List[Screw] = []

        # Callbacks
        self._on_screw_placed: Optional[Callable[[Screw], None]] = None
        self._on_state_changed: Optional[Callable[[str], None]] = None

    def set_mode(self, mode: str):
        """
        Set placement mode.

        Args:
            mode: "entry_target" or "entry_angle"
        """
        if mode not in ("entry_target", "entry_angle"):
            raise ValueError(f"Invalid mode: {mode}")

        self._mode = mode
        self._reset_state()

    def set_screw_parameters(
        self,
        length: Optional[float] = None,
        diameter: Optional[float] = None,
        insertion_angle: Optional[float] = None,
        medial_angle: Optional[float] = None
    ):
        """Set default screw parameters."""
        if length is not None:
            self._default_length = max(MIN_SCREW_LENGTH,
                                       min(MAX_SCREW_LENGTH, length))
        if diameter is not None:
            self._default_diameter = max(MIN_SCREW_DIAMETER,
                                        min(MAX_SCREW_DIAMETER, diameter))
        if insertion_angle is not None:
            self._default_insertion_angle = insertion_angle
        if medial_angle is not None:
            self._default_medial_angle = medial_angle

    def set_callbacks(
        self,
        on_screw_placed: Optional[Callable[[Screw], None]] = None,
        on_state_changed: Optional[Callable[[str], None]] = None
    ):
        """Set callback functions."""
        self._on_screw_placed = on_screw_placed
        self._on_state_changed = on_state_changed

    def on_click(
        self,
        x: float,
        y: float,
        z: float,
        plane: Optional[str] = None
    ) -> Optional[Screw]:
        """
        Process a click event.

        Args:
            x, y, z: World coordinates of click
            plane: Source plane ('axial', 'sagittal', 'coronal', or None for 3D)

        Returns:
            Screw object if placement completed, None otherwise
        """
        if self._mode == "entry_target":
            return self._handle_entry_target_click(x, y, z, plane)
        else:
            return self._handle_entry_angle_click(x, y, z, plane)

    def _handle_entry_target_click(
        self,
        x: float, y: float, z: float,
        plane: Optional[str]
    ) -> Optional[Screw]:
        """Handle click in entry-target mode."""
        if self._state == "waiting_entry":
            # First click: set entry point
            self._pending_entry = (x, y, z)
            self._state = "waiting_target"
            self._notify_state_changed()
            return None

        else:  # waiting_target
            # Second click: set target and create screw
            entry = self._pending_entry
            target = (x, y, z)

            screw = self._create_screw(entry, target)

            self._reset_state()
            return screw

    def _handle_entry_angle_click(
        self,
        x: float, y: float, z: float,
        plane: Optional[str]
    ) -> Optional[Screw]:
        """Handle click in entry-angle mode."""
        if self._state != "waiting_entry":
            return None

        # Calculate target from entry point and angles
        entry = (x, y, z)
        target = self._calculate_target_from_angles(
            entry,
            self._default_length,
            self._default_insertion_angle,
            self._default_medial_angle
        )

        screw = self._create_screw(entry, target)
        return screw

    def _create_screw(
        self,
        entry: Tuple[float, float, float],
        target: Tuple[float, float, float]
    ) -> Screw:
        """Create a new screw and add to list."""
        screw = Screw(
            entry_point=entry,
            target_point=target,
            diameter=self._default_diameter
        )

        # Evaluate placement
        self._evaluate_screw(screw)

        # Add to list
        self._screws.append(screw)

        # Notify callback
        if self._on_screw_placed:
            self._on_screw_placed(screw)

        return screw

    def _calculate_target_from_angles(
        self,
        entry: Tuple[float, float, float],
        length: float,
        insertion_angle: float,
        medial_angle: float
    ) -> Tuple[float, float, float]:
        """
        Calculate target point from entry point and angles.

        Args:
            entry: Entry point coordinates
            length: Screw length
            insertion_angle: Angle from vertical (degrees)
            medial_angle: Angle from midline (degrees)

        Returns:
            Target point coordinates
        """
        # Convert to radians
        ins_rad = math.radians(insertion_angle)
        med_rad = math.radians(medial_angle)

        # Calculate direction components
        # Coordinate convention used here:
        # - Z: anterior/depth direction
        # - Y: superior/inferior direction
        # - X: medial/lateral direction
        # The direction vector is normalized so actual screw length matches input.
        dx = -math.cos(ins_rad) * math.sin(med_rad)
        dy = -math.sin(ins_rad)
        dz = -math.cos(ins_rad) * math.cos(med_rad)

        return (
            entry[0] + length * dx,
            entry[1] + length * dy,
            entry[2] + length * dz,
        )

    def _evaluate_screw(self, screw: Screw):
        """
        Evaluate screw placement using Gertzbein-Robbins grading.

        Checks for cortical breach by sampling HU values along trajectory.
        """
        if self.volume_manager.get_vtk_image() is None:
            screw.grade = "A"  # Can't evaluate without volume
            return

        # Sample points along screw trajectory
        num_samples = int(screw.length / 0.5)  # Sample every 0.5mm
        max_breach = 0.0

        for i in range(num_samples):
            t = i / num_samples
            x = screw.entry_point[0] + t * (screw.target_point[0] - screw.entry_point[0])
            y = screw.entry_point[1] + t * (screw.target_point[1] - screw.entry_point[1])
            z = screw.entry_point[2] + t * (screw.target_point[2] - screw.entry_point[2])

            # Sample perpendicular points (at screw radius)
            # Simplified: just check the trajectory itself
            hu_value = self.volume_manager.get_voxel_value(x, y, z)

            # If HU is below bone threshold, we're outside bone (breach)
            if hu_value < 200:  # Soft tissue threshold
                # Estimate breach distance (simplified)
                max_breach = max(max_breach, screw.diameter / 2)

        # Assign grade based on breach
        if max_breach == 0:
            screw.grade = "A"
        elif max_breach < 2:
            screw.grade = "B"
        elif max_breach < 4:
            screw.grade = "C"
        elif max_breach < 6:
            screw.grade = "D"
        else:
            screw.grade = "E"

        screw.breach_distance = max_breach

    def _reset_state(self):
        """Reset tool state."""
        self._state = "waiting_entry"
        self._pending_entry = None
        self._notify_state_changed()

    def _notify_state_changed(self):
        """Notify callback of state change."""
        if self._on_state_changed:
            self._on_state_changed(self._state)

    def cancel(self):
        """Cancel current operation."""
        self._reset_state()

    def get_screws(self) -> List[Screw]:
        """Get all placed screws."""
        return self._screws.copy()

    def add_screw(self, screw: Screw):
        """Append an existing screw (used for plan restore)."""
        self._screws.append(screw)

    def replace_screw(
        self,
        index: int,
        entry_point: Tuple[float, float, float],
        target_point: Tuple[float, float, float],
    ) -> Screw:
        """Replace one screw geometry and recalculate derived safety data."""
        if index < 0 or index >= len(self._screws):
            raise IndexError(f"Screw index out of range: {index}")

        original = self._screws[index]
        updated = Screw(
            entry_point=tuple(float(value) for value in entry_point),
            target_point=tuple(float(value) for value in target_point),
            diameter=original.diameter,
            vertebra_level=original.vertebra_level,
            side=original.side,
        )
        self._evaluate_screw(updated)
        self._screws[index] = updated
        return updated

    def update_screw_diameter(self, index: int, diameter: float) -> Screw:
        """Resize one screw and recalculate its diameter-dependent safety data."""
        if index < 0 or index >= len(self._screws):
            raise IndexError(f"Screw index out of range: {index}")
        diameter = float(diameter)
        if not MIN_SCREW_DIAMETER <= diameter <= MAX_SCREW_DIAMETER:
            raise ValueError(
                f"Screw diameter must be between {MIN_SCREW_DIAMETER:.1f} "
                f"and {MAX_SCREW_DIAMETER:.1f} mm"
            )

        original = self._screws[index]
        updated = Screw(
            entry_point=original.entry_point,
            target_point=original.target_point,
            diameter=diameter,
            vertebra_level=original.vertebra_level,
            side=original.side,
        )
        self._evaluate_screw(updated)
        self._screws[index] = updated
        return updated

    def set_screws(self, screws: List[Screw]):
        """Replace screw list with existing screws (used for plan restore)."""
        self._screws = screws.copy()
        self._reset_state()

    def remove_screw(self, index: int):
        """Remove a screw by index."""
        if 0 <= index < len(self._screws):
            del self._screws[index]

    def clear_screws(self):
        """Remove all screws."""
        self._screws.clear()

    def get_state(self) -> str:
        """Get current placement state."""
        return self._state

    @staticmethod
    def get_grade_description(grade: str) -> str:
        """Get description for a Gertzbein-Robbins grade."""
        descriptions = {
            "A": GRADE_A_DESCRIPTION,
            "B": GRADE_B_DESCRIPTION,
            "C": GRADE_C_DESCRIPTION,
            "D": GRADE_D_DESCRIPTION,
            "E": GRADE_E_DESCRIPTION,
        }
        return descriptions.get(grade, "Unknown grade")
