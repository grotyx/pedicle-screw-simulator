"""
Screw data model for pedicle screw placement.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from src.core.screw_geometry import (
    convergence_angle_deg,
    craniocaudal_angle_deg,
    unit_trajectory,
)
from src.utils.constants import DEFAULT_SCREW_DIAMETER, DEFAULT_SCREW_LENGTH


@dataclass
class Screw:
    """A pedicle screw defined by entry and tip points in LPS millimetres.

    Attributes:
        entry_point: Entry (head) point.
        target_point: Tip point.
        length: Entry-to-tip distance (derived).
        diameter: Screw diameter in mm.
        vertebra_level: "L4", "T12", ... or "".
        side: "left", "right", or "".
        trajectory: Unit vector entry -> tip (derived).
        insertion_angle: Craniocaudal angle, degrees, cranial positive (derived).
        medial_angle: Axial convergence angle, degrees, medial positive when
            ``side`` is known, otherwise magnitude (derived).
        grade: Gertzbein-Robbins grade "A".."E" or "N/A".
        breach_distance: Maximum cortical breach in mm (0 for A).
        mean_hu / min_hu: Trajectory HU statistics when a CT was available.
        warnings: Planner or grader notes shown in the inspector.
        source: "manual" or "auto".
    """
    entry_point: Tuple[float, float, float]
    target_point: Tuple[float, float, float]
    length: float = DEFAULT_SCREW_LENGTH
    diameter: float = DEFAULT_SCREW_DIAMETER
    vertebra_level: str = ""
    side: str = ""
    trajectory: Tuple[float, float, float] = field(default=(0.0, 0.0, 1.0))
    insertion_angle: float = 0.0
    medial_angle: float = 0.0
    grade: str = "A"
    breach_distance: float = 0.0
    mean_hu: Optional[float] = None
    min_hu: Optional[float] = None
    warnings: List[str] = field(default_factory=list)
    source: str = "manual"

    def __post_init__(self):
        self.entry_point = tuple(float(v) for v in self.entry_point)
        self.target_point = tuple(float(v) for v in self.target_point)
        self._recompute_geometry()

    def _recompute_geometry(self) -> None:
        direction = unit_trajectory(self.entry_point, self.target_point)
        if direction != (0.0, 0.0, 0.0):
            self.trajectory = direction
        dx = self.target_point[0] - self.entry_point[0]
        dy = self.target_point[1] - self.entry_point[1]
        dz = self.target_point[2] - self.entry_point[2]
        self.length = (dx * dx + dy * dy + dz * dz) ** 0.5
        self.insertion_angle = craniocaudal_angle_deg(self.entry_point, self.target_point)
        self.medial_angle = convergence_angle_deg(
            self.entry_point, self.target_point, self.side or None
        )
