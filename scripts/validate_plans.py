"""CLI to compare a predicted pedicle screw plan against a reference plan.

Loads two plan JSON files (as produced by the desktop app or the automatic
planner), computes per-screw and cohort-level deviation metrics via
:mod:`src.core.plan_metrics`, prints the cohort summary, and writes a
per-screw CSV and a summary JSON report.

Usage::

    python scripts/validate_plans.py --pred pred.json --ref ref.json --out report

writes ``report.csv`` (one row per matched screw) and ``report.json`` (the
cohort summary) next to whatever base path ``--out`` names. ``--out`` is
optional: without it, only the cohort summary is printed to stdout and no
files are written. Screw level and side strings are case/whitespace-
normalised before matching, so plans from other tools ("l4"/"L4",
"Left"/"left") still pair up correctly. Exits with status 2 (and writes no
report) when no screws matched between the two plans, when either plan
file cannot be read or parsed, or when a plan file is valid JSON but has
the wrong structure (missing/malformed fields).

This script is Qt-free and can be run standalone or from CI.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

# Allow running as `python scripts/validate_plans.py` from the repo root
# without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.plan_metrics import compare_plans  # noqa: E402
from src.utils.planning_io import load_plan_json  # noqa: E402

SCREW_FIELDS = [
    "level",
    "side",
    "head_mad_mm",
    "tip_mad_mm",
    "axis_angle_deg",
    "convergence_delta_deg",
    "craniocaudal_delta_deg",
    "diameter_delta_mm",
    "length_delta_mm",
    "pedicle_center_offset_mm",
    "dice",
]


def _normalize_plan_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of ``payload`` with screw level/side casing normalised.

    ``match_screws`` pairs screws by raw ``(vertebra_level, side)`` string
    equality, so a reference plan exported by another tool with different
    casing (``"l4"`` vs. ``"L4"``, ``"Left"`` vs. ``"left"``) would otherwise
    fail to match. Normalise both fields the same way this script's own
    output uses (level upper-case, side lower-case) before comparing.
    """
    if not isinstance(payload, dict):
        return payload
    screws = payload.get("screws")
    if not isinstance(screws, list):
        return payload

    normalized_screws = []
    for item in screws:
        if isinstance(item, dict):
            item = dict(item)
            level = item.get("vertebra_level")
            if isinstance(level, str):
                item["vertebra_level"] = level.strip().upper()
            side = item.get("side")
            if isinstance(side, str):
                item["side"] = side.strip().lower()
        normalized_screws.append(item)

    normalized = dict(payload)
    normalized["screws"] = normalized_screws
    return normalized


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare a predicted pedicle screw plan against a reference plan."
    )
    parser.add_argument("--pred", required=True, help="Path to predicted plan JSON.")
    parser.add_argument("--ref", required=True, help="Path to reference plan JSON.")
    parser.add_argument(
        "--out",
        default=None,
        help=(
            "Output base path; writes <out>.csv and <out>.json. "
            "When omitted, only the summary is printed to stdout."
        ),
    )
    parser.add_argument(
        "--voxel-mm",
        type=float,
        default=0.5,
        help="Voxel edge length (mm) used to rasterise screws for Dice overlap.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        pred_payload = _normalize_plan_payload(load_plan_json(args.pred))
        ref_payload = _normalize_plan_payload(load_plan_json(args.ref))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"error: could not load plan file: {exc}", file=sys.stderr)
        return 2

    try:
        comparisons, summary = compare_plans(pred_payload, ref_payload, voxel_mm=args.voxel_mm)
    except (KeyError, TypeError, ValueError) as exc:
        print(f"error: invalid plan structure: {exc}", file=sys.stderr)
        return 2

    if summary.n_matched == 0:
        print(
            "No screws matched between pred and ref plans "
            f"(pred={summary.n_unmatched_pred}, ref={summary.n_unmatched_ref}); "
            "nothing to compare."
        )
        return 2

    print(f"voxel_mm: {args.voxel_mm}")
    for key, value in asdict(summary).items():
        print(f"{key}: {value}")

    if args.out is None:
        return 0

    out_base = Path(args.out)
    out_base.parent.mkdir(parents=True, exist_ok=True)

    csv_path = out_base.with_suffix(".csv")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(SCREW_FIELDS)
        for comparison in comparisons:
            row = asdict(comparison)
            writer.writerow([row[field] for field in SCREW_FIELDS])

    json_path = out_base.with_suffix(".json")
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(asdict(summary), handle, indent=2)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
