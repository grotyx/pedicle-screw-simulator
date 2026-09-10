"""Classification of where a screw leaves the pedicle, and of facet violation.

Two complementary descriptions of a malpositioned screw:

* **Heary direction** — the anatomical direction of the worst cortical breach
  (Heary 2004), derived from the offset between the breaching sample and the
  centreline point it belongs to. A medial breach threatens the canal, a
  lateral one the segmental vessels, an inferior one the exiting root.
* **Facet violation grade** — a Babu (2012) 0-3 approximation measured from the
  proximal (head) end of the screw to the cephalad vertebra's segmentation.
  Violating the facet above the instrumented level is a known driver of
  adjacent-segment degeneration.

Both work in LPS millimetres (+X left, +Y posterior, +Z superior), the
convention every geometry module in this app uses.
"""

from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np

from ..utils.constants import FACET_CONTACT_DISTANCE_MM
from .screw_grading import ScrewGrader

Point3 = Sequence[float]

#: Fraction of the screw, measured from the entry point, that can reach the
#: facet joint above the instrumented level.
PROXIMAL_FRACTION = 0.3

#: Penetration at or beyond this depth counts as a frank facet violation.
FACET_PENETRATION_MM = 1.0

#: Prefix of the note a medial breach raises.  It is regenerated on every
#: re-grade (see :data:`src.tools.screw_tool._DERIVED_WARNING_PREFIXES`), so the
#: planner and the manual tool must build it from this one function or an edited
#: screw will lose -- or keep -- a canal warning its trajectory disproves.
MEDIAL_BREACH_WARNING_PREFIX = "Medial breach"


def medial_breach_warning(medial_breach_mm: float) -> str:
    """The note a screw that left the pedicle toward the canal carries."""
    return f"{MEDIAL_BREACH_WARNING_PREFIX} {float(medial_breach_mm):.1f} mm — canal side"


_NO_CEPHALAD = (0, "no cephalad vertebra segmented")


def heary_direction(breach_point_lps: Point3, centreline_point_lps: Point3, side: str) -> str:
    """Anatomical direction of a breach, as ``breach_point`` seen from the centreline.

    Returns ``"medial"``, ``"lateral"``, ``"anterior"``, ``"posterior"``,
    ``"superior"``, ``"inferior"`` or ``"none"`` (coincident points). The
    dominant axis of the offset decides; on the x axis the sign is read with
    ``side`` (``"left"`` or ``"right"``), since LPS +X points to the patient's
    left and medial is therefore -X on the left and +X on the right.
    """
    d = np.asarray(breach_point_lps, dtype=np.float64) - np.asarray(centreline_point_lps, dtype=np.float64)
    if not np.any(np.abs(d) > 1e-9):
        return "none"
    axis = int(np.argmax(np.abs(d)))
    if axis == 0:
        medial = d[0] < 0 if str(side).lower() == "left" else d[0] > 0
        return "medial" if medial else "lateral"
    if axis == 1:
        return "anterior" if d[1] < 0 else "posterior"
    return "superior" if d[2] > 0 else "inferior"


def facet_violation_grade(
    grader: ScrewGrader,
    entry: Point3,
    target: Point3,
    diameter_mm: float,
    label: int,
) -> Tuple[int, str]:
    """Babu-style facet violation grade (0-3) with its description.

    The cephalad vertebra is ``label + 1`` in TotalSegmentator's ``total``
    numbering (L4 = 28 -> L3 = 29). Only the proximal 30 % of the screw is
    considered: the shaft and tip lie inside the vertebral body and cannot
    reach the joint. The grades are

    * 0 — the screw stays more than ``FACET_CONTACT_DISTANCE_MM`` away,
    * 1 — it comes within that distance without entering the cephalad label,
    * 2 — it enters it by less than ``FACET_PENETRATION_MM``,
    * 3 — it penetrates it by at least that much.

    Grade 0 is also reported when the cephalad vertebra is missing (the top of
    the labelled range, or simply not segmented), with a distinguishing text.
    """
    from .pedicle_analyzer import VERTEBRA_LABELS  # local import: avoids a cycle

    cephalad = int(label) + 1
    if cephalad not in VERTEBRA_LABELS or not grader.has_label(cephalad):
        return _NO_CEPHALAD

    entry_arr = np.asarray(entry, dtype=np.float64)
    length = float(np.linalg.norm(np.asarray(target, dtype=np.float64) - entry_arr))
    points = grader.cylinder_points(entry, target, diameter_mm)
    proximal = points[np.linalg.norm(points - entry_arr, axis=1) <= PROXIMAL_FRACTION * length]
    if proximal.size == 0:        # only reachable for a non-finite trajectory
        return 0, "no facet contact"

    d_out, d_in = grader.distances_at_points(proximal, cephalad)
    entered = d_out == 0.0
    if entered.any():
        penetration = float(d_in[entered].max())
        if penetration >= FACET_PENETRATION_MM:
            return 3, "screw penetrates the cephalad facet"
        return 2, "screw enters the cephalad facet (<1 mm)"
    if float(d_out.min()) <= FACET_CONTACT_DISTANCE_MM:
        return 1, "screw abuts the cephalad facet"
    return 0, "no facet contact"
