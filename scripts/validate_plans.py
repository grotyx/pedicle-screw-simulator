"""CLI to compare a predicted pedicle screw plan against a reference plan.

Loads two plan JSON files (as produced by the desktop app or the automatic
planner), computes per-screw and cohort-level deviation metrics via
:mod:`src.core.plan_metrics`, prints the cohort summary, and writes a
per-screw CSV and a summary JSON report.

Usage::

    python scripts/validate_plans.py --pred pred.json --ref ref.json --out report

writes ``report.csv`` (one row per matched screw) and ``report.json`` (the
cohort summary) next to whatever base path ``--out`` names.

This script is Qt-free and can be run standalone or from CI.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict
from pathlib import Path

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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare a predicted pedicle screw plan against a reference plan."
    )
    parser.add_argument("--pred", required=True, help="Path to predicted plan JSON.")
    parser.add_argument("--ref", required=True, help="Path to reference plan JSON.")
    parser.add_argument(
        "--out",
        required=True,
        help="Output base path; writes <out>.csv and <out>.json.",
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

    pred_payload = load_plan_json(args.pred)
    ref_payload = load_plan_json(args.ref)

    comparisons, summary = compare_plans(pred_payload, ref_payload, voxel_mm=args.voxel_mm)

    for key, value in asdict(summary).items():
        print(f"{key}: {value}")

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
