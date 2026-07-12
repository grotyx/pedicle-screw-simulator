"""
Measurement data model for surgical planning.
"""

from typing import Optional, Tuple, List
from dataclasses import dataclass


@dataclass
class Measurement:
    """Data class for a measurement."""
    points: List[Tuple[float, float, float]]
    distance: float  # Total distance in mm
    mode: str = "distance"
    angle: Optional[float] = None
    label: str = ""

    @property
    def num_points(self) -> int:
        return len(self.points)
