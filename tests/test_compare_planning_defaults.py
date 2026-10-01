"""scripts/compare_planning_defaults.py: previous-vs-current planning defaults."""

import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import compare_planning_defaults as cpd  # noqa: E402

from tests.test_auto_screw_planner import (  # noqa: E402
    _make_analysis,
    _make_bone_cylinder,
    _make_thin_medial_wall_phantom,
)


def _uncontained_narrow():
    """A narrow left side whose best trajectory still breaches medially."""
    ct, mask = _make_thin_medial_wall_phantom(lateral_edge=41)
    analysis = _make_analysis(
        left_center=np.array([40.0, 37.0, 20.0]),
        body_center=np.array([40.0, 18.0, 20.0]),
        left_width=4.5,
    )
    return ct, mask, analysis


def test_configs_are_the_documented_pair():
    assert cpd.PREVIOUS_CONFIG.accepted_breach_grade == "B"
    assert cpd.PREVIOUS_CONFIG.place_uncontained_narrow is True
    assert cpd.CURRENT_CONFIG == cpd.PlannerConfig()


def test_previous_uncontained_narrow_screw_is_now_skipped():
    ct, mask, analysis = _uncontained_narrow()

    rows = cpd.compare_defaults(ct, mask, [analysis])

    left = next(r for r in rows if r.side == "left")
    assert left.level == "L5"
    assert left.previous.screw is not None
    assert left.previous.screw.metrics["medial_breach_mm"] > 0.0
    assert left.current.screw is None
    assert "no contained trajectory" in left.current.reason
    assert left.change == "now skipped"


def test_a_contained_side_is_unchanged_and_row_structure():
    ct, mask = _make_bone_cylinder()

    rows = cpd.compare_defaults(ct, mask, [_make_analysis()])

    assert [(r.level, r.side) for r in rows] == [("L5", "left"), ("L5", "right")]
    assert {r.change for r in rows} == {"same"}
    assert all(r.previous.screw is not None and r.current.screw is not None for r in rows)


def test_levels_filter_and_report_leak_no_path(tmp_path):
    ct, mask, analysis = _uncontained_narrow()
    assert cpd.compare_defaults(ct, mask, [analysis], levels=["L1"]) == []

    lines = cpd.format_report(cpd.compare_defaults(ct, mask, [analysis], levels=["l5"]))

    text = "\n".join(lines)
    assert "now skipped" in text and "screws placed: previous" in text
    assert "newly skipped (1): narrow pedicle: no contained trajectory" in text
    assert str(tmp_path) not in text and str(ROOT) not in text


def test_cli_help():
    out = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "compare_planning_defaults.py"), "--help"],
        capture_output=True, text=True, cwd=ROOT,
    )
    assert out.returncode == 0
    assert "--levels" in out.stdout and "--csv" in out.stdout
