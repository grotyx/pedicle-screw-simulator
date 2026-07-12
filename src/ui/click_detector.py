"""Small platform-independent mouse click gesture helpers."""

from __future__ import annotations

import math
import time
from typing import Optional, Tuple


class DoubleClickDetector:
    """Recognize two nearby presses without depending on a GUI backend event."""

    def __init__(self, max_interval: float = 0.45, max_distance: float = 8.0):
        self.max_interval = float(max_interval)
        self.max_distance = float(max_distance)
        self._last_click: Optional[Tuple[float, float, float]] = None

    def register(
        self,
        x: float,
        y: float,
        timestamp: Optional[float] = None,
    ) -> bool:
        now = time.monotonic() if timestamp is None else float(timestamp)
        x = float(x)
        y = float(y)
        previous = self._last_click
        if previous is not None:
            elapsed = now - previous[2]
            distance = math.hypot(x - previous[0], y - previous[1])
            if 0.0 <= elapsed <= self.max_interval and distance <= self.max_distance:
                self.reset()
                return True
        self._last_click = (x, y, now)
        return False

    def reset(self) -> None:
        self._last_click = None
