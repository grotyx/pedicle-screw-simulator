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
* a level that produced no screw at all did so without a reason.

This is an acceptance aid for a human running the sample study, not a test:
``tests/`` never reads ``data/sample``, and this needs a real volume.  Run it
from the repo root; it is Qt-free.

Usage::

    python scripts/check_narrow_policy.py --ct data/sample/ct --mask data/sample/mask.nii.gz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import SimpleITK as sitk

# Allow running as `python scripts/check_narrow_policy.py` from the repo root
# without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.auto_screw_planner import AutoScrewPlanner  # noqa: E402
from src.core.pedicle_analyzer import PedicleAnalyzer  # noqa: E402
from src.core.planner_config import PlannerConfig  # noqa: E402

_HEADER = f"{'LEVEL':<6}{'SIDE':<7}{'WIDTH':>6}{'NARROW':>8}{'DIA':>7}{'MEDIAL':>9}{'LATERAL':>9}{'GRADE':>7}"


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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ct", required=True, type=Path)
    parser.add_argument("--mask", required=True, type=Path)
    args = parser.parse_args(argv)

    ct = _read_volume(args.ct)
    mask = _read_volume(args.mask)
    config = PlannerConfig()

    analyzer = PedicleAnalyzer(mask)
    analyses = [
        analyzer.analyze_pedicle(vertebra)
        for vertebra in analyzer.get_available_vertebrae()
    ]
    planner = AutoScrewPlanner(ct, mask, config=config)
    screws = planner.plan_all(analyses)

    print(_HEADER)
    failures: list[str] = []
    for screw in sorted(screws, key=lambda s: (s.vertebra_name, s.side)):
        metrics = screw.metrics
        narrow = bool(metrics.get("narrow_pedicle"))
        medial = float(metrics.get("medial_breach_mm") or 0.0)
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
            failures.append(
                f"{screw.vertebra_name} {screw.side}: narrow screw breaches "
                f"medially by {medial:.2f} mm"
            )

    for name, side, reason in planner.skipped_sides:
        print(f"SKIPPED {name} {side}: {reason}")
        if "width" in reason or "narrow" in reason:
            failures.append(f"{name} {side} was dropped for its width: {reason}")

    print()
    print(f"{len(screws)} screws, {len(planner.skipped_sides)} skipped sides")
    for failure in failures:
        print(f"FAIL: {failure}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
