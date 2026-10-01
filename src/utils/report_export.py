"""One-click planning report: a single self-contained HTML page.

Pure string building (no Qt window needed), so it can be tested directly.
The report deliberately takes no DICOM metadata, no folder path and no series
UID: the only study facts it can show are voxel spacing and image size.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass, field
from datetime import datetime
from html import escape
from typing import Any, Dict, Optional, Sequence, Tuple

from src import __version__
from src.utils.planning_io import _metric_number

TITLE = "Pedicle Screw Simulator — Planning report"
BANNER = (
    "RESEARCH USE ONLY — not a certified medical device; every value "
    "requires independent clinical review."
)
IMAGE_UNAVAILABLE = "image unavailable"

_STYLE = """
body{font-family:Segoe UI,Arial,sans-serif;margin:24px;color:#1b1f24;max-width:1100px}
h1{font-size:22px;margin:0 0 4px}h2{font-size:16px;margin:24px 0 8px;border-bottom:1px solid #ccd}
.banner{background:#7a1f1f;color:#fff;padding:10px 14px;font-weight:700;margin:12px 0;border-radius:4px}
.meta{color:#555;font-size:13px}
table{border-collapse:collapse;font-size:13px;width:100%}
th,td{border:1px solid #ccd;padding:4px 8px;text-align:left;vertical-align:top}th{background:#eef}
.construct td{text-align:center}.construct .lvl{text-align:right;font-weight:600;width:80px}
.chip{display:inline-block;min-width:44px;margin:1px;padding:2px 6px;border-radius:10px;color:#fff;font-weight:700}
.gA{background:#2e7d32}.gB{background:#689f38}.gC{background:#f9a825;color:#222}
.gD,.gE{background:#c62828}.gN{background:#777}
.shots{display:grid;grid-template-columns:1fr 1fr;gap:10px}
.shots figure{margin:0}.shots img{width:100%;border:1px solid #ccd}
.shots figcaption{font-size:12px;color:#555}.missing{color:#888;font-style:italic}
"""

#: Pane title -> embedded PNG bytes, or None when capture failed.
Images = Dict[str, Optional[bytes]]


@dataclass
class ReportContext:
    """Everything the report may show; there is no free-form metadata field."""

    screws: Sequence[Any] = ()
    planner: Any = None  # PlannerConfig, or None when unknown
    spacing: Optional[Tuple[float, float, float]] = None
    size: Optional[Tuple[int, int, int]] = None
    images: Images = field(default_factory=dict)
    generated_at: Optional[str] = None
    version: str = __version__


def capture_png(render_window) -> Optional[bytes]:
    """Grab a VTK render window as PNG bytes; ``None`` on any failure."""
    try:
        import vtk
        from vtk.util.numpy_support import vtk_to_numpy

        grab = vtk.vtkWindowToImageFilter()
        grab.SetInput(render_window)
        grab.ReadFrontBufferOff()
        grab.Update()
        writer = vtk.vtkPNGWriter()
        writer.SetInputConnection(grab.GetOutputPort())
        writer.WriteToMemoryOn()
        writer.Write()
        data = bytes(vtk_to_numpy(writer.GetResult()))
        return data or None
    except Exception:
        return None


def _row(label: str, value: Any) -> str:
    return f"<tr><th>{escape(label)}</th><td>{escape(str(value))}</td></tr>"


def _study_html(ctx: ReportContext) -> str:
    rows = []
    if ctx.spacing:
        rows.append(_row("Voxel spacing (mm)", " × ".join(f"{v:.3f}" for v in ctx.spacing)))
    if ctx.size:
        rows.append(_row("Image size (voxels)", " × ".join(str(int(v)) for v in ctx.size)))
    return "<table>" + "".join(rows) + "</table>" if rows else "<p>Not available.</p>"


def _settings_html(planner: Any) -> str:
    if planner is None:
        return "<p>Not available.</p>"
    rows = [
        _row("Planner mode", planner.mode),
        _row("Trajectory", planner.trajectory),
        _row("Accepted breach grade", planner.accepted_breach_grade),
        _row("Narrow-pedicle threshold (mm)", f"{planner.narrow_pedicle_mm:g}"),
        _row(
            "Place narrow screws even if not contained",
            "on" if planner.place_uncontained_narrow else "off",
        ),
    ]
    return "<table>" + "".join(rows) + "</table>"


def _screws_html(screws: Sequence[Any]) -> str:
    # Imported here: src.ui imports the controllers, which import this module.
    from src.ui.screw_plan_table import screw_row_cells, screw_warning_lines

    if not screws:
        return "<p>No screws planned.</p>"
    head = (
        "#", "Level", "Side", "Source", "Diameter × length (mm)", "Grade",
        "Breach (mm)", "Medial breach (mm)", "Trajectory HU mean", "Warnings",
    )
    out = ["<table><tr>" + "".join(f"<th>{h}</th>" for h in head) + "</tr>"]
    for index, screw in enumerate(screws):
        cells = screw_row_cells(index, screw)
        metrics = screw.metrics if isinstance(screw.metrics, dict) else {}
        values = (
            cells[0], cells[1], cells[2], screw.source,
            f"{cells[4]} × {cells[5]}", cells[6],
            f"{screw.breach_distance:.1f}",
            _metric_number(metrics, "medial_breach_mm", ".1f") or "—",
            _metric_number(metrics, "trajectory_mean_hu", ".0f") or "—",
        )
        warnings = "<br>".join(escape(line) for line in screw_warning_lines(screw))
        out.append(
            "<tr>" + "".join(f"<td>{escape(str(v))}</td>" for v in values)
            + f"<td>{warnings}</td></tr>"
        )
    return "".join(out) + "</table>"


def _chip(index: int, screw: Any) -> str:
    grade = str(screw.grade)
    css = "g" + (grade if grade in "ABCDE" and len(grade) == 1 else "N")
    return f'<span class="chip {css}">#{index + 1} {escape(grade)}</span>'


def _construct_html(screws: Sequence[Any]) -> str:
    from src.ui.construct_map import MANUAL_LEVEL, level_order_key

    grouped: Dict[str, Dict[str, list]] = {}
    for index, screw in enumerate(screws):
        level = str(screw.vertebra_level or "") or MANUAL_LEVEL
        side = str(screw.side or "").lower()
        slot = side if side in ("right", "left") else "mid"
        grouped.setdefault(level, {"right": [], "mid": [], "left": []})[slot].append(
            _chip(index, screw)
        )
    if not grouped:
        return "<p>No screws planned.</p>"
    out = ['<table class="construct"><tr><th>Level</th><th>Right</th><th></th><th>Left</th></tr>']
    for level in sorted(grouped, key=level_order_key):
        cells = grouped[level]
        out.append(
            f'<tr><td class="lvl">{escape(level)}</td><td>{"".join(cells["right"])}</td>'
            f'<td>{"".join(cells["mid"])}</td><td>{"".join(cells["left"])}</td></tr>'
        )
    return "".join(out) + "</table>"


def _images_html(images: Images) -> str:
    if not images:
        return f'<p class="missing">{IMAGE_UNAVAILABLE}</p>'
    figs = []
    for title, png in images.items():
        if png:
            body = f'<img alt="{escape(title)}" src="data:image/png;base64,{base64.b64encode(png).decode("ascii")}">'
        else:
            body = f'<p class="missing">{IMAGE_UNAVAILABLE}</p>'
        figs.append(f"<figure>{body}<figcaption>{escape(title)}</figcaption></figure>")
    return '<div class="shots">' + "".join(figs) + "</div>"


def build_report_html(ctx: ReportContext) -> str:
    """Return the complete report as one HTML string."""
    stamp = ctx.generated_at or datetime.now().strftime("%Y-%m-%d %H:%M")
    return (
        '<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">'
        f"<title>{escape(TITLE)}</title><style>{_STYLE}</style></head><body>"
        f"<h1>{escape(TITLE)}</h1>"
        f'<div class="meta">Version {escape(ctx.version)} · generated {escape(stamp)}</div>'
        f'<div class="banner">{escape(BANNER)}</div>'
        f"<h2>Study</h2>{_study_html(ctx)}"
        f"<h2>Planner settings</h2>{_settings_html(ctx.planner)}"
        f"<h2>Screws</h2>{_screws_html(ctx.screws)}"
        f"<h2>Construct map</h2>{_construct_html(ctx.screws)}"
        f"<h2>Views</h2>{_images_html(ctx.images)}"
        "</body></html>"
    )
