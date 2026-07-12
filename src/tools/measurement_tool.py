"""
Measurement Tool - Interactive distance measurement

Provides:
- Point-to-point distance measurement
- Multi-point path measurement
- Angle measurement
"""

import math
from typing import Optional, Tuple, List, Callable

from src.models.measurement import Measurement


class MeasurementTool:
    """
    Interactive measurement tool for medical imaging.

    Modes:
    - "distance": Two-point distance measurement
    - "path": Multi-point path measurement (double-click to finish)
    - "angle": Three-point angle measurement
    """

    def __init__(self):
        self._mode = "distance"
        self._pending_points: List[Tuple[float, float, float]] = []
        self._measurements: List[Measurement] = []

        # Callbacks
        self._on_measurement_complete: Optional[Callable[[Measurement], None]] = None
        self._on_point_added: Optional[Callable[[Tuple[float, float, float]], None]] = None

    def set_mode(self, mode: str):
        """Set measurement mode."""
        if mode not in ("distance", "path", "angle"):
            raise ValueError(f"Invalid mode: {mode}")

        self._mode = mode
        self._pending_points.clear()

    def get_mode(self) -> str:
        """Get current measurement mode."""
        return self._mode

    def set_callbacks(
        self,
        on_measurement_complete: Optional[Callable[[Measurement], None]] = None,
        on_point_added: Optional[Callable[[Tuple[float, float, float]], None]] = None
    ):
        """Set callback functions."""
        self._on_measurement_complete = on_measurement_complete
        self._on_point_added = on_point_added

    def on_click(
        self,
        x: float,
        y: float,
        z: float,
        double_click: bool = False
    ) -> Optional[Measurement]:
        """
        Process a click event.

        Args:
            x, y, z: World coordinates
            double_click: True if this was a double-click

        Returns:
            Measurement if complete, None otherwise
        """
        point = (x, y, z)
        self._pending_points.append(point)

        if self._on_point_added:
            self._on_point_added(point)

        # Check if measurement is complete
        if self._mode == "distance" and len(self._pending_points) >= 2:
            return self._finish_measurement()

        elif self._mode == "angle" and len(self._pending_points) >= 3:
            return self._finish_measurement()

        elif self._mode == "path" and double_click:
            return self._finish_measurement()

        return None

    def _finish_measurement(self) -> Optional[Measurement]:
        """Finish current measurement."""
        if len(self._pending_points) < 2:
            return None

        total_distance = self._calculate_total_distance(self._pending_points)
        angle = None
        label = self.format_distance(total_distance)
        if self._mode == "angle":
            if len(self._pending_points) < 3:
                return None
            angle = self._calculate_angle_from_points(self._pending_points[:3])
            if angle is None:
                return None
            label = self.format_angle(angle)

        measurement = Measurement(
            points=self._pending_points.copy(),
            distance=total_distance,
            mode=self._mode,
            angle=angle,
            label=label,
        )

        self._measurements.append(measurement)
        self._pending_points.clear()

        if self._on_measurement_complete:
            self._on_measurement_complete(measurement)

        return measurement

    def finish_pending(self) -> Optional[Measurement]:
        """Force-finish the current pending measurement when valid."""
        if self._mode == "distance" and len(self._pending_points) >= 2:
            return self._finish_measurement()
        if self._mode == "path" and len(self._pending_points) >= 2:
            return self._finish_measurement()
        if self._mode == "angle" and len(self._pending_points) >= 3:
            return self._finish_measurement()
        return None

    def calculate_angle(self) -> Optional[float]:
        """
        Calculate angle for three-point measurement.

        Returns angle at the middle point in degrees.
        """
        if len(self._pending_points) != 3:
            return None
        return self._calculate_angle_from_points(self._pending_points)

    def cancel(self):
        """Cancel current measurement."""
        self._pending_points.clear()

    def get_measurements(self) -> List[Measurement]:
        """Get all measurements."""
        return self._measurements.copy()

    def add_measurement(self, measurement: Measurement):
        """Append an existing measurement (used for plan restore)."""
        self._measurements.append(measurement)

    def set_measurements(self, measurements: List[Measurement]):
        """Replace measurement list with existing entries (plan restore)."""
        self._measurements = measurements.copy()
        self._pending_points.clear()

    def remove_measurement(self, index: int):
        """Remove a measurement by index."""
        if 0 <= index < len(self._measurements):
            del self._measurements[index]

    def replace_measurement(self, index: int, measurement: Measurement):
        """Replace one completed measurement while preserving list order."""
        if not 0 <= index < len(self._measurements):
            raise IndexError("measurement index out of range")
        self._measurements[index] = measurement

    def clear_measurements(self):
        """Clear all measurements."""
        self._measurements.clear()
        self._pending_points.clear()

    def get_pending_points(self) -> List[Tuple[float, float, float]]:
        """Get points for current incomplete measurement."""
        return self._pending_points.copy()

    @staticmethod
    def _calculate_total_distance(points: List[Tuple[float, float, float]]) -> float:
        """Calculate polyline length for given points."""
        total_distance = 0.0
        for i in range(1, len(points)):
            p1 = points[i - 1]
            p2 = points[i]
            segment_dist = math.sqrt(
                (p2[0] - p1[0]) ** 2
                + (p2[1] - p1[1]) ** 2
                + (p2[2] - p1[2]) ** 2
            )
            total_distance += segment_dist
        return total_distance

    @staticmethod
    def _calculate_angle_from_points(
        points: List[Tuple[float, float, float]]
    ) -> Optional[float]:
        """Calculate angle at middle point from 3 points."""
        if len(points) < 3:
            return None

        p1, p2, p3 = points[0], points[1], points[2]
        v1 = (p1[0] - p2[0], p1[1] - p2[1], p1[2] - p2[2])
        v2 = (p3[0] - p2[0], p3[1] - p2[1], p3[2] - p2[2])

        dot = v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2]
        mag1 = math.sqrt(v1[0] ** 2 + v1[1] ** 2 + v1[2] ** 2)
        mag2 = math.sqrt(v2[0] ** 2 + v2[1] ** 2 + v2[2] ** 2)
        if mag1 == 0 or mag2 == 0:
            return None

        cos_angle = max(-1, min(1, dot / (mag1 * mag2)))
        return math.degrees(math.acos(cos_angle))

    @staticmethod
    def format_distance(distance: float) -> str:
        """Format distance for display."""
        if distance < 10:
            return f"{distance:.2f} mm"
        else:
            return f"{distance:.1f} mm"

    @staticmethod
    def format_angle(angle: float) -> str:
        """Format angle for display."""
        return f"{angle:.1f}°"
