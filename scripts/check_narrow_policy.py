"""CLI that checks the narrow-pedicle policy against a real study.

Loads a CT (a DICOM directory or any ITK-readable volume) and a vertebra
segmentation, runs the pedicle analysis and the auto planner with the shipped
:class:`~src.core.planner_config.PlannerConfig` defaults, and prints one row per
``(level, side)``::

    LEVEL SIDE   WIDTH  NARROW    DIA   MEDIAL  LATERAL  GRADE
    L2    right   4.6   yes       4.0     0.00     1.25  B

then exits non-zero when any of W7's acceptance items fails:

* a side was skipped for a reason mentioning the pedicle width;
* a screw the optimiser placed on a narrow side has a medial breach;
* a screw the optimiser placed on a narrow side is not at
  :attr:`~src.core.auto_screw_planner.AutoScrewPlanner.MIN_SCREW_DIAMETER`;
* a level that produced no screw at all did so without a reason (except the
  sacrum, which the planner excludes by policy rather than dropping).

Pedicle analysis runs through
:meth:`~src.core.pedicle_analyzer.PedicleAnalyzer.analyze_all`, the same
entry point :class:`~src.controllers.auto_placement_controller._PlanningThread`
uses, so one vertebra whose geometry cannot be analyzed is reported back as a
failed :class:`~src.core.pedicle_analyzer.PedicleAnalysisResult` instead of
aborting the whole run.

An optional ``--pedicle-mask`` supplies a stage-2 pedicle subregion mask, the
same input :class:`~src.controllers.segmentation_controller.SegmentationController`
feeds through ``_load_pedicle_mask``.  That controller method resolves a
named "pedicle" class out of a multi-label subregion result (metadata this
CLI does not have, since it never runs the subregion model) and only then
resamples it; this CLI instead treats whatever file is passed as an
already-binary pedicle mask and resamples it onto the vertebra mask's grid
with the same :func:`~src.core.subregion_segmentation.resample_to_reference`
nearest-neighbour call the controller uses, so a mask that is not already on
that grid still lines up correctly. Any nonzero voxel in the resampled mask
counts as pedicle.

The planner runs with the shipped :class:`~src.core.planner_config.PlannerConfig`
defaults, except for ``--accept-grade-b``: the legacy diameter step-down
requires grade A (zero breach) by default and skips a normal-width side that
only grades B, matching the optimiser and CBT. Passing the flag sets
``accept_grade_b=True``, so the legacy loop also accepts grade B (< 2 mm
breach) and the accepted screw carries a persistent "Grade B accepted"
warning. The narrow-pedicle path is untouched either way: it is already on
the smallest implant and slides laterally instead of stepping down.

This is an acceptance aid for a human running the sample study, not a test:
``tests/`` never reads ``data/sample``, and this needs a real volume.  Run it
from the repo root; it is Qt-free.

Usage::

    python scripts/check_narrow_policy.py --ct data/sample/ct --mask data/sample/mask.nii.gz
    python scripts/check_narrow_policy.py --ct data/sample/ct --mask data/sample/mask.nii.gz \\
        --pedicle-mask data/sample/pedicle_mask.nii.gz
    python scripts/check_narrow_policy.py --ct data/sample/ct --mask data/sample/mask.nii.gz \\
        --accept-grade-b
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import SimpleITK as sitk

# Allow running as `python scripts/check_narrow_policy.py` from the repo root
# without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.auto_screw_planner import (  # noqa: E402
    _SACRUM_LABEL,
    OPTIMIZER_FALLBACK_WARNING,
    AutoScrewPlanner,
)
from src.core.pedicle_analyzer import PedicleAnalysisResult, PedicleAnalyzer  # noqa: E402
from src.core.planner_config import PlannerConfig  # noqa: E402
from src.core.subregion_segmentation import resample_to_reference  # noqa: E402

_HEADER = f"{'LEVEL':<6}{'SIDE':<7}{'WIDTH':>6}{'NARROW':>8}{'DIA':>7}{'MEDIAL':>9}{'LATERAL':>9}{'GRADE':>7}"

_SIDES = ("left", "right")


def _read_volume(path: Path) -> sitk.Image:
    """Read a DICOM series directory or a single ITK-readable file."""
    if path.is_dir():
        reader = sitk.ImageSeriesReader()
        names = reader.GetGDCMSeriesFileNames(str(path))
        if not names:
            raise ValueError(f"no DICOM series under {path}")
        reader.SetFileNames(names)
        return reader.Execute()
    return sitk.ReadImage(str(path))


def _load_pedicle_mask(path: Path, mask_image: sitk.Image):
    """Read an external pedicle mask and resample it onto the vertebra mask's grid.

    Mirrors the resampling half of
    ``SegmentationController._load_pedicle_mask`` (nearest-neighbour onto the
    vertebra mask's grid via :func:`resample_to_reference`); it does not
    replicate the label-lookup half, since a subregion model's
    ``subregion_labels`` mapping has no equivalent for a plain file handed to
    this CLI. Any nonzero voxel in the resampled volume is treated as pedicle.
    """
    pedicle_image = resample_to_reference(_read_volume(path), mask_image)
    return sitk.GetArrayFromImage(pedicle_image) != 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ct", required=True, type=Path)
    parser.add_argument("--mask", required=True, type=Path)
    parser.add_argument(
        "--pedicle-mask",
        type=Path,
        default=None,
        help=(
            "Optional stage-2 pedicle subregion mask (any ITK-readable "
            "volume). Resampled onto the vertebra mask's grid; any nonzero "
            "voxel is treated as pedicle."
        ),
    )
    parser.add_argument(
        "--accept-grade-b",
        action="store_true",
        help=(
            "Set PlannerConfig.accept_grade_b: the legacy diameter step-down "
            "also accepts grade B (< 2 mm breach) instead of requiring "
            "grade A. Off by default; narrow pedicles are unaffected."
        ),
    )
    args = parser.parse_args(argv)

    ct = _read_volume(args.ct)
    mask = _read_volume(args.mask)
    config = PlannerConfig(accept_grade_b=args.accept_grade_b)

    pedicle_mask = (
        _load_pedicle_mask(args.pedicle_mask, mask)
        if args.pedicle_mask is not None
        else None
    )

    analyzer = PedicleAnalyzer(mask, ct, pedicle_mask=pedicle_mask)
    analyses: List[PedicleAnalysisResult] = analyzer.analyze_all()
    planner = AutoScrewPlanner(ct, mask, config=config)
    screws = planner.plan_all(analyses)

    print(_HEADER)
    failures: list[str] = []
    fallback_count = 0
    for screw in sorted(screws, key=lambda s: (s.vertebra_name, s.side)):
        metrics = screw.metrics
        narrow = bool(metrics.get("narrow_pedicle"))
        medial = float(metrics.get("medial_breach_mm") or 0.0)
        is_fallback = OPTIMIZER_FALLBACK_WARNING in screw.warnings
        if is_fallback:
            fallback_count += 1
        print(
            f"{screw.vertebra_name:<6}{screw.side:<7}"
            f"{float(metrics.get('pedicle_width_mm') or 0.0):>6.1f}"
            f"{'yes' if narrow else 'no':>8}"
            f"{screw.diameter_mm:>7.1f}"
            f"{medial:>9.2f}"
            f"{float(metrics.get('lateral_breach_mm') or 0.0):>9.2f}"
            f"{screw.gertzbein_grade:>7}"
        )
        if narrow and medial > 0.0:
            # The "narrow => no medial breach" contract binds the optimiser's
            # own trajectory search, not the legacy fallback it falls back to
            # when no feasible trajectory is found -- that path predates the
            # zero-breach guarantee and is reported, not failed.
            if is_fallback:
                print(
                    f"  narrow (legacy fallback) medial {medial:.2f} mm"
                )
            else:
                failures.append(
                    f"{screw.vertebra_name} {screw.side}: narrow screw breaches "
                    f"medially by {medial:.2f} mm"
                )
        if narrow and abs(screw.diameter_mm - AutoScrewPlanner.MIN_SCREW_DIAMETER) > 1e-6:
            failures.append(
                f"{screw.vertebra_name} {screw.side}: narrow screw has diameter "
                f"{screw.diameter_mm:.1f} mm, expected "
                f"{AutoScrewPlanner.MIN_SCREW_DIAMETER:.1f} mm"
            )

    for name, side, reason in planner.skipped_sides:
        print(f"SKIPPED {name} {side}: {reason}")
        if "width" in reason or "narrow" in reason:
            failures.append(f"{name} {side} was dropped for its width: {reason}")

    # Every (level, side) that produced no screw must be explained in
    # `skipped_sides`, per `AutoScrewPlanner.plan_all`'s own contract -- except
    # the sacrum, which `_record_skip` never adds there because it is excluded
    # by policy rather than dropped (reason=None, not a blank reason).
    produced = {(screw.vertebra_name, screw.side) for screw in screws}
    skip_reasons: Dict[Tuple[str, str], str] = {
        (name, side): reason for name, side, reason in planner.skipped_sides
    }
    for analysis in analyses:
        if analysis.vertebra.label == _SACRUM_LABEL:
            continue
        for side in _SIDES:
            pair = (analysis.vertebra.name, side)
            if pair in produced:
                continue
            reason: Optional[str] = skip_reasons.get(pair)
            if not reason:
                failures.append(
                    f"{pair[0]} {pair[1]}: produced no screw and no reason was "
                    "recorded in skipped_sides"
                )

    print()
    print(
        f"{len(screws)} screws ({fallback_count} via legacy fallback), "
        f"{len(planner.skipped_sides)} skipped sides"
    )
    for failure in failures:
        print(f"FAIL: {failure}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
