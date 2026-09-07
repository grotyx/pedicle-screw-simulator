"""Shared screw trajectory angle definitions (LPS millimetres).

Conventions
-----------
* Convergence (axial) angle: angle between the trajectory projected onto the
  axial plane and the anterior axis (-Y).  Positive = medial (tip toward the
  midline) for the stated side.  Without a side, the magnitude is returned.
* Craniocaudal (sagittal) angle: elevation of the trajectory above the axial
  plane.  Positive = tip superior (+Z) to entry.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple

Point3 = Sequence[float]


def unit_trajectory(entry: Point3, target: Point3) -> Tuple[float, float, float]:
    dx = float(target[0]) - float(entry[0])
    dy = float(target[1]) - float(entry[1])
    dz = float(target[2]) - float(entry[2])
    length = math.sqrt(dx * dx + dy * dy + dz * dz)
    if length <= 1e-12:
        return (0.0, 0.0, 0.0)
    return (dx / length, dy / length, dz / length)


def convergence_angle_deg(entry: Point3, target: Point3, side: Optional[str]) -> float:
    dx = float(target[0]) - float(entry[0])
    dy = float(target[1]) - float(entry[1])
    if math.hypot(dx, dy) <= 1e-9:
        return 0.0
    magnitude = math.degrees(math.atan2(abs(dx), abs(dy)))
    if side == "left":
        return magnitude if dx <= 0.0 else -magnitude
    if side == "right":
        return magnitude if dx >= 0.0 else -magnitude
    return magnitude


def craniocaudal_angle_deg(entry: Point3, target: Point3) -> float:
    dx = float(target[0]) - float(entry[0])
    dy = float(target[1]) - float(entry[1])
    dz = float(target[2]) - float(entry[2])
    horizontal = math.hypot(dx, dy)
    if horizontal <= 1e-9 and abs(dz) <= 1e-9:
        return 0.0
    return math.degrees(math.atan2(dz, horizontal))
