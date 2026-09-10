"""Shared screw trajectory angle definitions (LPS millimetres).

Conventions
-----------
* Convergence (axial) angle: angle between the trajectory projected onto the
  axial plane and the anterior axis (-Y).  Positive = medial (tip toward the
  midline) for the stated side.  Without a side, the magnitude is returned.
* Craniocaudal (sagittal) angle: elevation of the trajectory above the axial
  plane.  Positive = tip superior (+Z) to entry.
* Endplate angle: craniocaudal angle of the trajectory minus the craniocaudal
  angle of the upper-endplate plane.  Positive = tip cranial to the endplate.
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


def endplate_slope_deg(upper_endplate_normal: Optional[Point3]) -> Optional[float]:
    """Craniocaudal angle of the upper-endplate plane, travelling anteriorly.

    ``upper_endplate_normal`` is the ``+Z``-oriented LPS plane normal produced by
    :meth:`src.core.pedicle_analyzer.PedicleAnalyzer._estimate_upper_endplate_normal`.
    For a plane ``z = a*x + b*y + c`` that normal is ``(-a, -b, 1)`` normalised, so
    ``n_y = -b`` and ``atan2(n_y, n_z) = atan(-b)`` — which is exactly the
    craniocaudal angle of the endplate-parallel direction when the trajectory
    travels anteriorly (``dy < 0``, ``dz = b*dy``).  Positive therefore means the
    endplate rises toward the front of the patient, matching the sign
    :meth:`~src.core.auto_screw_planner.AutoScrewPlanner._endplate_aligned_target_z`
    already builds its targets with: a screw aimed along that helper's target
    reads an :func:`endplate_angle_deg` of zero.

    ``None`` means "not measurable" rather than "flat": no normal, a normal that
    is not a 3-vector, or one lying in the axial plane (``n_z ~ 0``), which is not
    an endplate at all.
    """
    if upper_endplate_normal is None:
        return None
    values = tuple(float(v) for v in upper_endplate_normal)
    if len(values) != 3:
        return None
    _, n_y, n_z = values
    if abs(n_z) <= 1e-9:
        return None
    return math.degrees(math.atan2(n_y, n_z))


def endplate_angle_deg(
    entry: Point3,
    target: Point3,
    upper_endplate_normal: Optional[Point3],
) -> Optional[float]:
    """Signed angle between the screw axis and the upper endplate (degrees).

    Positive = the tip is cranial relative to the endplate; zero = parallel to it.
    ``None`` when the endplate plane is not measurable (see
    :func:`endplate_slope_deg`), which is "we do not know", not "zero".

    Measured with :func:`craniocaudal_angle_deg`, so it divides the rise by the
    *full* horizontal travel rather than by the AP travel alone.  A strongly
    convergent screw therefore reads a little closer to zero than a pure sagittal
    projection would give.  That is deliberate: the inspector shows this number
    next to the craniocaudal angle and the two have to be the same measurement.
    """
    slope = endplate_slope_deg(upper_endplate_normal)
    if slope is None:
        return None
    return craniocaudal_angle_deg(entry, target) - slope
