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


#: Largest |angle| accepted by :func:`target_from_angles`; at 90 degrees the
#: trajectory is vertical or lateral and the angle pair no longer defines it.
MAX_TRAJECTORY_ANGLE_DEG = 89.0

#: How far behind the entry the guide handle sits, so it stays outside the bone.
GUIDE_REACH_MM = 60.0


def _finite_point(point: Point3, name: str) -> Tuple[float, float, float]:
    values = tuple(float(v) for v in point)
    if len(values) != 3 or not all(math.isfinite(v) for v in values):
        raise ValueError(f"{name} must be three finite numbers")
    return values


def target_from_angles(
    entry: Point3,
    length: float,
    convergence_deg: float,
    craniocaudal_deg: float,
    side: str,
) -> Tuple[float, float, float]:
    """Tip of a screw of ``length`` mm aimed by the conventions above.

    The inverse of :func:`convergence_angle_deg` and
    :func:`craniocaudal_angle_deg` for an anteriorly directed screw.  ``side``
    is required ("left" or "right"): medial is -X for a left screw and +X for
    a right one, whatever the entry's X coordinate.
    """
    entry = _finite_point(entry, "entry")
    if side not in ("left", "right"):
        raise ValueError("side must be 'left' or 'right'")
    length = float(length)
    if not math.isfinite(length) or length <= 0.0:
        raise ValueError("length must be a positive finite number")
    for angle in (convergence_deg, craniocaudal_deg):
        if not math.isfinite(float(angle)) or abs(float(angle)) > MAX_TRAJECTORY_ANGLE_DEG:
            raise ValueError(f"angles must lie within ±{MAX_TRAJECTORY_ANGLE_DEG:g} degrees")
    con = math.radians(float(convergence_deg))
    cra = math.radians(float(craniocaudal_deg))
    lateral_sign = 1.0 if side == "left" else -1.0
    direction = (
        -lateral_sign * math.sin(con) * math.cos(cra),
        -math.cos(con) * math.cos(cra),
        math.sin(cra),
    )
    return tuple(entry[i] + length * direction[i] for i in range(3))


def guide_handle_point(
    entry: Point3, target: Point3, reach_mm: float
) -> Tuple[float, float, float]:
    """Centre of the guide handle ``reach_mm`` behind the entry, on the screw axis."""
    entry = _finite_point(entry, "entry")
    u = unit_trajectory(entry, _finite_point(target, "target"))
    return tuple(entry[i] - float(reach_mm) * u[i] for i in range(3))


def target_from_guide_drag(
    entry: Point3,
    target: Point3,
    handle: Point3,
    picked: Point3,
    pointer: Point3,
) -> Tuple[float, float, float]:
    """Tip after dragging the guide handle, pivoting about a fixed entry.

    The handle moves by ``pointer - picked`` (so picking the sphere's surface
    instead of its centre does not jump the screw); the screw keeps its entry
    and length and points away from the moved handle.
    """
    entry = _finite_point(entry, "entry")
    target = _finite_point(target, "target")
    handle = _finite_point(handle, "handle")
    picked = _finite_point(picked, "picked point")
    pointer = _finite_point(pointer, "pointer")
    moved = tuple(handle[i] + pointer[i] - picked[i] for i in range(3))
    lever = tuple(entry[i] - moved[i] for i in range(3))
    lever_length = math.sqrt(sum(v * v for v in lever))
    if lever_length <= 1e-6:
        raise ValueError("guide handle is on the entry point")
    length = math.dist(entry, target)
    return tuple(entry[i] + length * lever[i] / lever_length for i in range(3))


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
