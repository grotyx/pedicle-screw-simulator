"""
Measurement data model for surgical planning.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple


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
