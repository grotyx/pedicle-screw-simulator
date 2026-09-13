"""Direct tests for the screw pedicle-width verdict helpers.

Regression scope: src/utils/screw_metrics.py bound logic — narrow verdict,
width-uncertain flag precedence, unmeasured widths, and the exact display
strings of both the plan-table cell and the inspector row.
"""

from types import SimpleNamespace

import pytest

from src.utils.screw_metrics import (
    UNKNOWN_TEXT,
    is_narrow_pedicle,
    is_width_uncertain,
    pedicle_cell_text,
    pedicle_row_text,
    pedicle_width_mm,
    screw_metrics,
)


def test_screw_metrics_copies_mapping_and_defaults_empty():
    screw = SimpleNamespace(metrics={"narrow_pedicle": True})
    copied = screw_metrics(screw)
    assert copied == {"narrow_pedicle": True}

    copied["narrow_pedicle"] = False
    assert screw.metrics == {"narrow_pedicle": True}

    assert screw_metrics(SimpleNamespace()) == {}
    assert screw_metrics(SimpleNamespace(metrics=["not", "a", "mapping"])) == {}


def test_is_narrow_pedicle_reads_flag_only():
    assert is_narrow_pedicle({"narrow_pedicle": True}) is True
    assert is_narrow_pedicle({"narrow_pedicle": False}) is False
    # Manually placed screws carry no analysis: unknown, not normal.
    assert is_narrow_pedicle({}) is False


def test_width_uncertain_flag_is_independent():
    assert is_width_uncertain({"width_uncertain": True}) is True
    assert is_width_uncertain({"width_uncertain": False}) is False
    assert is_width_uncertain({}) is False


def test_pedicle_width_mm_rejects_non_numbers():
    assert pedicle_width_mm({"pedicle_width_mm": 4.5}) == pytest.approx(4.5)
    assert pedicle_width_mm({"pedicle_width_mm": "4.5"}) is None
    assert pedicle_width_mm({"pedicle_width_mm": True}) is None
    assert pedicle_width_mm({"pedicle_width_mm": None}) is None
    assert pedicle_width_mm({}) is None


def test_pedicle_cell_text_formats_width():
    assert pedicle_cell_text({"pedicle_width_mm": 4.54}) == "4.5 mm"
    assert pedicle_cell_text({}) == UNKNOWN_TEXT


def test_pedicle_cell_text_flags_untrusted_width_with_question_mark():
    text = pedicle_cell_text(
        {"pedicle_width_mm": 4.5, "width_uncertain": True}
    )
    assert text == "4.5 mm?"


def test_pedicle_row_text_spells_out_narrow():
    assert (
        pedicle_row_text({"pedicle_width_mm": 4.5, "narrow_pedicle": True})
        == "4.5 mm · narrow"
    )
    assert pedicle_row_text({"pedicle_width_mm": 9.2}) == "9.2 mm"
    assert pedicle_row_text({}) == UNKNOWN_TEXT


def test_pedicle_row_text_uncertain_wins_over_narrow():
    # An out-of-band width is flagged in either direction, so "narrow"
    # would be wrong for the wide half of those.
    assert (
        pedicle_row_text(
            {
                "pedicle_width_mm": 9.2,
                "narrow_pedicle": True,
                "width_uncertain": True,
            }
        )
        == "9.2 mm? · width not trusted"
    )
