"""Display rules of the reference-plan comparison window."""

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PyQt6")

from src.core.plan_metrics import compare_screw_lists  # noqa: E402
from src.models.screw import Screw  # noqa: E402
from src.ui.reference_comparison_dialog import (  # noqa: E402
    NO_PAIRS_TEXT,
    comparison_rows,
    summary_text,
)


def _s(x, level="L4", side="left"):
    return Screw(entry_point=(x, 0.0, 0.0), target_point=(x, -40.0, 0.0),
                 diameter=6.0, vertebra_level=level, side=side)


def test_no_pairs_never_reads_as_perfect_agreement():
    result = compare_screw_lists([_s(0.0)], [_s(0.0, level="L5")])
    assert summary_text(result) == NO_PAIRS_TEXT


def test_one_pair_shows_means_without_spread_or_limits():
    result = compare_screw_lists([_s(1.0)], [_s(0.0)])
    text = summary_text(result)
    assert text.startswith("1 matched pair;")
    assert "±" not in text and "LoA" not in text


def test_two_pairs_add_spread_and_agreement_limits():
    result = compare_screw_lists(
        [_s(1.0), _s(20.0, side="right")], [_s(0.0), _s(22.0, side="right")]
    )
    text = summary_text(result)
    assert "±" in text and "95% LoA" in text


def test_rows_list_pairs_then_every_leftover_with_its_reason():
    current = [_s(0.0), _s(5.0, level=""), _s(9.0, level="L2")]
    current.append(Screw(entry_point=(0, 0, 0), target_point=(0, -40, 0), diameter=0.0,
                         vertebra_level="L1", side="left"))
    reference = [_s(0.0), _s(30.0, level="L3")]

    rows = comparison_rows(compare_screw_lists(current, reference), current, reference)

    assert [row[0] for row in rows] == [
        "matched",
        "only in current plan",
        "only in reference",
        "no level/side (current)",
        "invalid (current): diameter is not a positive number",
    ]
    assert rows[0][1:3] == ("1", "1")
    assert rows[2][1:5] == ("", "2", "L3", "left")
