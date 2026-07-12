"""
Screw data model for pedicle screw placement.
"""

import math
from typing import Tuple
from dataclasses import dataclass, field

from src.utils.constants import (
    DEFAULT_SCREW_LENGTH, DEFAULT_SCREW_DIAMETER,
)


@dataclass
class Screw:
    """
    Data class representing a pedicle screw.

    Attributes:
        entry_point: World coordinates of entry point
        target_point: World coordinates of target (tip)
        length: Screw length in mm
        diameter: Screw diameter in mm
        vertebra_level: Vertebral level (e.g., "L4", "T12")
        side: "left" or "right"
        trajectory: Unit vector from entry to target
        insertion_angle: Angle from vertical in sagittal plane (degrees)
        medial_angle: Angle from midline in axial plane (degrees)
        grade: Gertzbein-Robbins grade (A-E)
        breach_distance: Maximum breach distance in mm (0 for Grade A)
    """
    entry_point: Tuple[float, float, float]
    target_point: Tuple[float, float, float]
    length: float = DEFAULT_SCREW_LENGTH
    diameter: float = DEFAULT_SCREW_DIAMETER
    vertebra_level: str = ""
    side: str = ""
    trajectory: Tuple[float, float, float] = field(default=(0, 0, 1))
    insertion_angle: float = 0.0
    medial_angle: float = 0.0
    grade: str = "A"
    breach_distance: float = 0.0

    def __post_init__(self):
        """Calculate derived properties."""
        self._calculate_trajectory()
        self._calculate_angles()
        self._calculate_length()

    def _calculate_trajectory(self):
        """Calculate trajectory unit vector."""
        dx = self.target_point[0] - self.entry_point[0]
        dy = self.target_point[1] - self.entry_point[1]
        dz = self.target_point[2] - self.entry_point[2]

        length = math.sqrt(dx*dx + dy*dy + dz*dz)
        if length > 0:
            self.trajectory = (dx/length, dy/length, dz/length)

    def _calculate_angles(self):
        """Calculate insertion and medial angles."""
        # Insertion angle: angle from vertical (S direction) in sagittal plane
        # Project trajectory onto YZ plane
        yz_length = math.sqrt(self.trajectory[1]**2 + self.trajectory[2]**2)
        if yz_length > 0:
            self.insertion_angle = math.degrees(
                math.atan2(self.trajectory[1], self.trajectory[2])
            )

        # Medial angle: angle from midline in axial plane
        xy_length = math.sqrt(self.trajectory[0]**2 + self.trajectory[1]**2)
        if xy_length > 0:
            self.medial_angle = math.degrees(
                math.atan2(abs(self.trajectory[0]), self.trajectory[1])
            )

    def _calculate_length(self):
        """Calculate actual length from entry to target."""
        dx = self.target_point[0] - self.entry_point[0]
        dy = self.target_point[1] - self.entry_point[1]
        dz = self.target_point[2] - self.entry_point[2]
        self.length = math.sqrt(dx*dx + dy*dy + dz*dz)
