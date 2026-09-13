"""Direct tests for construct-level alignment stamping.

Regression scope: src/core/construct_alignment.py rod fit and convergence
spread/deviation math over synthetic screw stand-ins. No volumes, no Qt.
"""

from types import SimpleNamespace

import pytest

from src.core.construct_alignment import (
    ALIGNMENT_KEYS,
    DEVIATION_KEY,
    ROD_KEY,
    SPREAD_KEY,
    restamp_carriers,
    stamp_alignment,
)


def _screw(side, entry, convergence, level=None):
    return SimpleNamespace(
        side=side,
        entry=entry,
        convergence=convergence,
        level=level,
        metrics={},
    )


def _accessors():
    return {
        "side_of": lambda s: s.side,
        "entry_of": lambda s: s.entry,
        "convergence_of": lambda s: s.convergence,
        "level_of": lambda s: s.level,
        "metrics_of": lambda s: s.metrics,
    }


def test_stamp_alignment_collinear_heads_score_zero_rod():
    screws = [
        _screw("left", (0.0, 0.0, 0.0), 10.0),
        _screw("left", (0.0, 0.0, 10.0), 10.0),
        _screw("left", (0.0, 0.0, 20.0), 10.0),
    ]

    stamp_alignment(screws, **_accessors())

    for screw in screws:
        assert screw.metrics[ROD_KEY] == pytest.approx(0.0)
        assert screw.metrics[SPREAD_KEY] == pytest.approx(0.0)
        assert screw.metrics[DEVIATION_KEY] == pytest.approx(0.0)


def test_stamp_alignment_offline_head_scores_positive_rod():
    screws = [
        _screw("left", (0.0, 0.0, 0.0), 10.0),
        _screw("left", (0.0, 0.0, 10.0), 10.0),
        _screw("left", (5.0, 0.0, 20.0), 10.0),
    ]

    stamp_alignment(screws, **_accessors())

    assert screws[0].metrics[ROD_KEY] > 0.0
    # Same rod value stamped on every screw of that side.
    assert screws[1].metrics[ROD_KEY] == pytest.approx(
        screws[0].metrics[ROD_KEY]
    )


def test_stamp_alignment_measures_each_side_independently():
    screws = [
        _screw("left", (0.0, 0.0, 0.0), 10.0),
        _screw("left", (0.0, 0.0, 10.0), 10.0),
        _screw("left", (0.0, 0.0, 20.0), 20.0),
        _screw("right", (1.0, 0.0, 0.0), 12.0),
        _screw("right", (1.0, 0.0, 10.0), 12.0),
        _screw("right", (1.0, 0.0, 20.0), 12.0),
    ]

    stamp_alignment(screws, **_accessors())

    assert screws[0].metrics[SPREAD_KEY] > 0.0
    assert screws[3].metrics[SPREAD_KEY] == pytest.approx(0.0)
    assert screws[3].metrics[DEVIATION_KEY] == pytest.approx(0.0)


def test_stamp_alignment_sacral_screw_gets_spread_but_no_deviation():
    screws = [
        _screw("left", (0.0, 0.0, 0.0), 10.0, level=28),
        _screw("left", (0.0, 0.0, 10.0), 12.0, level=26),  # S1 excluded
    ]

    stamp_alignment(screws, **_accessors())

    assert SPREAD_KEY in screws[1].metrics
    assert DEVIATION_KEY not in screws[1].metrics
    assert DEVIATION_KEY in screws[0].metrics


def test_stamp_alignment_drops_screws_without_a_side():
    screws = [
        _screw("", (0.0, 0.0, 0.0), 10.0),
        _screw("left", (0.0, 0.0, 10.0), 10.0),
    ]

    stamp_alignment(screws, **_accessors())

    assert screws[0].metrics == {}
    assert set(screws[1].metrics) == set(ALIGNMENT_KEYS)


def test_restamp_carriers_leaves_non_carriers_alone():
    harmonised = _screw("left", (0.0, 0.0, 0.0), 10.0)
    harmonised.metrics[ROD_KEY] = 99.0
    plain = _screw("left", (100.0, 0.0, 50.0), 30.0)

    restamp_carriers([harmonised, plain], **_accessors())

    assert harmonised.metrics[ROD_KEY] == pytest.approx(0.0)
    assert plain.metrics == {}


def test_restamp_carriers_with_no_carriers_changes_nothing():
    screws = [_screw("left", (0.0, 0.0, 0.0), 10.0)]

    restamp_carriers(screws, **_accessors())

    assert screws[0].metrics == {}
