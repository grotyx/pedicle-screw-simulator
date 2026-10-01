"""CLI that shows what the 0.2.4 planning defaults change on a real study.

Loads a CT (a DICOM directory or any ITK-readable volume) and a vertebra
segmentation, analyses the pedicles once, then runs the auto planner twice on
those same analyses -- with :data:`PREVIOUS_CONFIG` and with
:data:`CURRENT_CONFIG` -- and prints one row per ``(level, side)``::

    LEVEL SIDE   PREVIOUS                         CURRENT                          CHANGE
    L2    right  placed 6.0x45 A med 0.0 lat 0.0  placed 6.0x45 A med 0.0 lat 0.0  same

followed by a summary (screws placed before and after, newly skipped sides
grouped by reason).  CHANGE is one of ``same``, ``smaller screw``,
``now skipped``, ``now placed`` or ``different trajectory``.

The output names vertebra levels and numbers only: no input path, no DICOM
metadata.  This is a report, so the exit code is 0 whatever it finds; it is
non-zero only for a usage or read error.  It is research output, not a
clinical validation.  Run it from the repo root; it is Qt-free.

Usage::

    python scripts/compare_planning_defaults.py --ct <ct> --mask <vertebra mask>
    python scripts/compare_planning_defaults.py --ct <ct> --mask <vertebra mask> \\
        [--pedicle-mask <pedicle mask>] [--levels L3,L4] [--csv out.csv]
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# Allow running as `python scripts/compare_planning_defaults.py` from the repo
# root without installing the package; scripts/ itself is on the path so the
# loading helpers of check_narrow_policy can be reused.
_SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPTS.parent))
sys.path.insert(0, str(_SCRIPTS))

from check_narrow_policy import _load_pedicle_mask, _read_volume  # noqa: E402

from src.core.auto_screw_planner import AutoScrewPlanner, PlannedScrew  # noqa: E402
from src.core.pedicle_analyzer import PedicleAnalysisResult, PedicleAnalyzer  # noqa: E402
from src.core.planner_config import PlannerConfig  # noqa: E402

#: Closest available stand-in for the 0.2.3 behaviour: grade-B breaches were
#: accepted and a narrow side was placed whether or not its trajectory was
#: contained.  It differs from 0.2.3 in one way that cannot be re-enabled:
#: 0.2.3 also accepted a canal-side (medial) breach on legacy normal-width
#: screws, whereas the planner now never accepts a medial breach, so a side
#: 0.2.3 placed that way can show up here as skipped or moved.
PREVIOUS_CONFIG = PlannerConfig(accepted_breach_grade="B", place_uncontained_narrow=True)

#: The shipped defaults: grade A (no cortical breach), no medial breach ever,
#: narrow sides that fail containment are skipped.
CURRENT_CONFIG = PlannerConfig()

#: A screw whose entry or target moved less than this (mm) is the same trajectory.
_SAME_TRAJECTORY_MM = 0.25

_HEADER = f"{'LEVEL':<6}{'SIDE':<7}{'PREVIOUS':<40}{'CURRENT':<40}CHANGE"


@dataclass(frozen=True)
class Outcome:
    """One planner run's result for a side: a screw, or the reason it has none."""

    screw: Optional[PlannedScrew]
    reason: str = ""

    def text(self) -> str:
        if self.screw is None:
            return f"skipped: {self.reason}"
        s = self.screw
        return (
            f"placed {s.diameter_mm:.1f}x{s.length_mm:.0f} {s.gertzbein_grade} "
            f"med {float(s.metrics.get('medial_breach_mm') or 0.0):.1f} "
            f"lat {float(s.metrics.get('lateral_breach_mm') or 0.0):.1f}"
        )


@dataclass(frozen=True)
class Row:
    level: str
    side: str
    previous: Outcome
    current: Outcome
    change: str


def classify(previous: Outcome, current: Outcome) -> str:
    """Name what changed between two outcomes for the same side."""
    p, c = previous.screw, current.screw
    if p is None and c is None:
        return "same"
    if p is None:
        return "now placed"
    if c is None:
        return "now skipped"
    if (c.diameter_mm, c.length_mm) < (p.diameter_mm, p.length_mm):
        return "smaller screw"
    moved = max(
        float(np.linalg.norm(c.entry_lps - p.entry_lps)),
        float(np.linalg.norm(c.target_lps - p.target_lps)),
    )
    same_size = (c.diameter_mm, c.length_mm) == (p.diameter_mm, p.length_mm)
    return "same" if same_size and moved <= _SAME_TRAJECTORY_MM else "different trajectory"


def _plan(ct, mask, analyses, config: PlannerConfig) -> Dict[Tuple[str, str], Outcome]:
    planner = AutoScrewPlanner(ct, mask, config=config)
    outcomes = {(s.vertebra_name, s.side): Outcome(s) for s in planner.plan_all(analyses)}
    for name, side, reason in planner.skipped_sides:
        outcomes[(name, side)] = Outcome(None, reason)
    return outcomes


def compare_defaults(
    ct,
    mask,
    analyses: Sequence[PedicleAnalysisResult],
    levels: Optional[Sequence[str]] = None,
    previous_config: PlannerConfig = PREVIOUS_CONFIG,
    current_config: PlannerConfig = CURRENT_CONFIG,
) -> List[Row]:
    """Plan ``analyses`` under both configs and return one row per (level, side)."""
    wanted = {name.strip().upper() for name in levels} if levels else None
    chosen = [a for a in analyses if wanted is None or a.vertebra.name.upper() in wanted]
    previous = _plan(ct, mask, chosen, previous_config)
    current = _plan(ct, mask, chosen, current_config)
    missing = Outcome(None, "not planned")
    rows = []
    for level, side in sorted(set(previous) | set(current)):
        p = previous.get((level, side), missing)
        c = current.get((level, side), missing)
        rows.append(Row(level, side, p, c, classify(p, c)))
    return rows


def _reason_group(reason: str) -> str:
    """A skip reason without its side and numbers, so equal causes group."""
    head = reason.split(" (")[0].split(" — ")[0]
    return re.sub(r"\b(left|right) ", "", head)


def format_report(rows: Sequence[Row]) -> List[str]:
    lines = [_HEADER]
    for r in rows:
        lines.append(
            f"{r.level:<6}{r.side:<7}{r.previous.text():<40}{r.current.text():<40}{r.change}"
        )
    lines.append("")
    lines.append(
        "screws placed: previous "
        f"{sum(r.previous.screw is not None for r in rows)}, "
        f"current {sum(r.current.screw is not None for r in rows)}"
    )
    changes = Counter(r.change for r in rows)
    lines.append("changes: " + ", ".join(f"{n} {k}" for k, n in sorted(changes.items())))
    newly = Counter(_reason_group(r.current.reason) for r in rows if r.change == "now skipped")
    for group, n in newly.most_common():
        lines.append(f"newly skipped ({n}): {group}")
    return lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ct", required=True, type=Path)
    parser.add_argument("--mask", required=True, type=Path)
    parser.add_argument("--pedicle-mask", type=Path, default=None)
    parser.add_argument("--levels", default=None, help="comma-separated levels, e.g. L3,L4")
    parser.add_argument("--csv", type=Path, default=None, help="also write the rows here")
    args = parser.parse_args(argv)

    # Skip reasons contain a typographic dash; never die on a narrow console.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")

    try:
        ct = _read_volume(args.ct)
        mask = _read_volume(args.mask)
        pedicle_mask = (
            _load_pedicle_mask(args.pedicle_mask, mask) if args.pedicle_mask else None
        )
    except (ValueError, RuntimeError, OSError):
        # The underlying message names the path, which the report must not.
        parser.error("could not read the CT, mask or pedicle mask")

    analyses = PedicleAnalyzer(mask, ct, pedicle_mask=pedicle_mask).analyze_all()
    levels = args.levels.split(",") if args.levels else None
    rows = compare_defaults(ct, mask, analyses, levels)
    print("\n".join(format_report(rows)))

    if args.csv is not None:
        try:
            with open(args.csv, "w", newline="", encoding="utf-8") as fh:
                writer = csv.writer(fh)
                writer.writerow(["level", "side", "previous", "current", "change"])
                writer.writerows(
                    [r.level, r.side, r.previous.text(), r.current.text(), r.change]
                    for r in rows
                )
        except OSError:
            parser.error("could not write the CSV file")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
