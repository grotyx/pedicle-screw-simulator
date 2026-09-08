"""Tests for scripts/validate_plans.py (M8: optional --out, M9: clean error exits)."""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from scripts.validate_plans import main  # noqa: E402
from src.models.screw import Screw  # noqa: E402
from src.utils.planning_io import serialize_plan  # noqa: E402


def _write_plan(path, screws):
    payload = serialize_plan(series_id="SERIES-1", screws=screws, measurements=[])
    path.write_text(json.dumps(payload), encoding="utf-8")


def _matched_pair(tmp_path):
    screw = Screw(
        entry_point=(20.0, 30.0, 0.0),
        target_point=(10.0, -8.0, 0.0),
        diameter=6.0,
        vertebra_level="L4",
        side="left",
    )
    pred = tmp_path / "pred.json"
    ref = tmp_path / "ref.json"
    _write_plan(pred, [screw])
    _write_plan(ref, [screw])
    return pred, ref


def test_out_is_optional_and_prints_summary_without_writing_files(tmp_path, capsys):
    pred, ref = _matched_pair(tmp_path)

    exit_code = main(["--pred", str(pred), "--ref", str(ref)])

    assert exit_code == 0
    out = capsys.readouterr().out
    assert "n_matched: 1" in out
    # No --out was given, so nothing should have been written next to the inputs.
    assert sorted(p.name for p in tmp_path.iterdir()) == ["pred.json", "ref.json"]


def test_out_still_writes_csv_and_json_reports_when_given(tmp_path):
    pred, ref = _matched_pair(tmp_path)
    out_base = tmp_path / "report"

    exit_code = main(
        ["--pred", str(pred), "--ref", str(ref), "--out", str(out_base)]
    )

    assert exit_code == 0
    assert out_base.with_suffix(".csv").exists()
    assert out_base.with_suffix(".json").exists()


def test_missing_pred_file_exits_cleanly_without_traceback(tmp_path, capsys):
    ref = tmp_path / "ref.json"
    _write_plan(ref, [])
    missing = tmp_path / "does_not_exist.json"

    exit_code = main(["--pred", str(missing), "--ref", str(ref)])

    assert exit_code == 2
    captured = capsys.readouterr()
    assert "Traceback" not in captured.out
    assert "Traceback" not in captured.err
    assert "does_not_exist.json" in captured.err or "does_not_exist.json" in captured.out


def test_malformed_json_exits_cleanly_without_traceback(tmp_path, capsys):
    pred = tmp_path / "pred.json"
    pred.write_text("{not valid json", encoding="utf-8")
    ref = tmp_path / "ref.json"
    _write_plan(ref, [])

    exit_code = main(["--pred", str(pred), "--ref", str(ref)])

    assert exit_code == 2
    captured = capsys.readouterr()
    assert "Traceback" not in captured.out
    assert "Traceback" not in captured.err


def test_structurally_invalid_json_exits_cleanly_with_status_2(tmp_path, capsys):
    """Valid JSON, wrong shape (screws is not a list) should not traceback."""
    pred = tmp_path / "pred.json"
    pred.write_text(json.dumps({"screws": "not-a-list"}), encoding="utf-8")
    ref = tmp_path / "ref.json"
    _write_plan(ref, [])

    exit_code = main(["--pred", str(pred), "--ref", str(ref)])

    assert exit_code == 2
    captured = capsys.readouterr()
    assert "Traceback" not in captured.out
    assert "Traceback" not in captured.err


def test_plan_with_malformed_screw_fields_exits_cleanly_with_status_2(tmp_path, capsys):
    """Valid JSON, a screw entry missing required fields (ValueError deep in
    deserialisation) should also exit 2, not traceback."""
    pred = tmp_path / "pred.json"
    pred.write_text(json.dumps({"screws": [{"vertebra_level": "L4"}]}), encoding="utf-8")
    ref = tmp_path / "ref.json"
    _write_plan(ref, [])

    exit_code = main(["--pred", str(pred), "--ref", str(ref)])

    assert exit_code == 2
    captured = capsys.readouterr()
    assert "Traceback" not in captured.out
    assert "Traceback" not in captured.err


def test_plan_with_wrong_field_types_exits_cleanly_with_status_2(tmp_path, capsys):
    """Valid JSON, a screw field of the wrong type (TypeError from float())
    should also exit 2, not traceback."""
    pred = tmp_path / "pred.json"
    pred.write_text(
        json.dumps(
            {
                "screws": [
                    {
                        "entry_point": [0, 0, 0],
                        "target_point": [1, 1, 1],
                        "diameter": {"bad": 1},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    ref = tmp_path / "ref.json"
    _write_plan(ref, [])

    exit_code = main(["--pred", str(pred), "--ref", str(ref)])

    assert exit_code == 2
    captured = capsys.readouterr()
    assert "Traceback" not in captured.out
    assert "Traceback" not in captured.err


def test_out_argument_is_not_required_by_the_parser():
    """Regression for M8: the roadmap's acceptance command omits --out."""
    from scripts.validate_plans import parse_args

    args = parse_args(["--pred", "a.json", "--ref", "b.json"])
    assert args.out is None
