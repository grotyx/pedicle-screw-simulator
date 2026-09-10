"""Reading the planner's pedicle-width verdict off a screw.

Three views show the same two numbers -- the plan table's Pedicle column, the
cockpit's Pedicle row and the colour of the screw in 3D and on the MPR planes --
and a screw's metric bundle is free-form, so every one of them would otherwise
grow its own "is this dict entry a truthy float" logic.  This module is Qt-free
and numpy-free on purpose: the table imports it, so does the controller, and so
could a report.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

#: Planner-owned metric keys.  Both survive a re-grade (they are absent from
#: :data:`src.tools.screw_tool._GRADER_MEASURED_METRIC_KEYS`), and both are
#: simply missing on a manually placed screw, which has no pedicle analysis
#: behind it -- "unknown", not "normal".
NARROW_PEDICLE_KEY = "narrow_pedicle"
PEDICLE_WIDTH_KEY = "pedicle_width_mm"

#: Shown wherever a width was never measured.
UNKNOWN_TEXT = "--"


def screw_metrics(screw: Any) -> Dict[str, Any]:
    """One screw's metric bundle as a plain dict (``{}`` when it has none)."""
    metrics = getattr(screw, "metrics", None)
    return dict(metrics) if isinstance(metrics, Mapping) else {}


def is_narrow_pedicle(metrics: Mapping[str, Any]) -> bool:
    """Whether the planner marked this screw's pedicle narrow."""
    return bool(metrics.get(NARROW_PEDICLE_KEY))


def pedicle_width_mm(metrics: Mapping[str, Any]) -> Optional[float]:
    """The measured pedicle width, or ``None`` when it was never measured."""
    value = metrics.get(PEDICLE_WIDTH_KEY)
    if value is None or isinstance(value, (bool, str)):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def pedicle_cell_text(metrics: Mapping[str, Any]) -> str:
    """The plan table's Pedicle cell: just the width, since the chip says narrow."""
    width = pedicle_width_mm(metrics)
    return UNKNOWN_TEXT if width is None else f"{width:.1f} mm"


def pedicle_row_text(metrics: Mapping[str, Any]) -> str:
    """The cockpit's Pedicle row, which has room to spell the verdict out."""
    text = pedicle_cell_text(metrics)
    if text != UNKNOWN_TEXT and is_narrow_pedicle(metrics):
        return f"{text} · narrow"
    return text
