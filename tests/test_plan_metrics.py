"""Tests for plan comparison metrics (src/core/plan_metrics.py)."""

import numpy as np
import pytest

from src.core.plan_metrics import (
    compare_plans,
    compare_screw,
    cylinder_dice,
    match_screws,
    summarize,
)
from src.models.screw import Screw
from src.utils.planning_io import serialize_plan


def _s(entry, target, level="L4", side="left", d=6.0):
    return Screw(
        entry_point=entry,
        target_point=target,
        diameter=d,
        vertebra_level=level,
        side=side,
    )


# --- compare_screw -------------------------------------------------------


def test_identical_screws_have_zero_deviation_and_dice_one():
    a = _s((20, 30, 0), (10, -8, 0))
    c = compare_screw(a, a)
    assert c.head_mad_mm == 0 and c.tip_mad_mm == 0
    assert c.axis_angle_deg == pytest.approx(0)
    assert c.dice == pytest.approx(1.0, abs=0.02)


def test_translated_screw_metrics():
    a = _s((20, 30, 0), (10, -8, 0))
    b = _s((22, 30, 0), (12, -8, 0))
    c = compare_screw(a, b)
    assert c.head_mad_mm == pytest.approx(2.0) and c.tip_mad_mm == pytest.approx(2.0)
    assert c.axis_angle_deg == pytest.approx(0.0, abs=1e-6)
    assert 0.5 < c.dice < 0.9


def test_pedicle_center_offset_is_line_distance():
    a = _s((20, 30, 0), (20, -10, 0))
    c = compare_screw(a, a, ref_pedicle_center=(23.0, 10.0, 0.0))
    assert c.pedicle_center_offset_mm == pytest.approx(3.0)


def test_pedicle_center_offset_is_none_without_reference_point():
    a = _s((20, 30, 0), (20, -10, 0))
    assert compare_screw(a, a).pedicle_center_offset_mm is None


def test_axis_angle_and_scalar_deltas():
    pred = _s((0, 30, 0), (0, -10, 0), d=6.5)
    ref = _s((0, 30, 0), (0, 30, -40), d=5.5)
    c = compare_screw(pred, ref)
    assert c.axis_angle_deg == pytest.approx(90.0)
    assert c.diameter_delta_mm == pytest.approx(1.0)
    assert c.length_delta_mm == pytest.approx(0.0)
    assert c.craniocaudal_delta_deg == pytest.approx(
        pred.insertion_angle - ref.insertion_angle
    )
    assert c.convergence_delta_deg == pytest.approx(
        pred.medial_angle - ref.medial_angle
    )
    assert c.level == "L4" and c.side == "left"


def test_disjoint_screws_have_zero_dice():
    a = _s((20, 30, 0), (10, -8, 0))
    b = _s((-20, 30, 0), (-10, -8, 0), side="right")
    assert compare_screw(a, b).dice == pytest.approx(0.0)


# --- cylinder_dice -------------------------------------------------------


def test_cylinder_dice_identical_is_one():
    dice = cylinder_dice((0, 0, 0), (0, 0, 40), 6.0, (0, 0, 0), (0, 0, 40), 6.0)
    assert dice == pytest.approx(1.0)


def test_cylinder_dice_nested_matches_volume_ratio():
    # Same axis, half the radius -> intersection = the smaller volume, so
    # dice = 2 * V_small / (V_small + V_large) = 2 * 1/4 / (1/4 + 1) = 0.4
    dice = cylinder_dice((0, 0, 0), (0, 0, 40), 6.0, (0, 0, 0), (0, 0, 40), 12.0)
    assert dice == pytest.approx(0.4, abs=0.03)


def test_cylinder_dice_zero_length_is_zero():
    assert cylinder_dice((0, 0, 0), (0, 0, 0), 6.0, (0, 0, 0), (0, 0, 40), 6.0) == 0.0


# --- match_screws --------------------------------------------------------


def test_match_by_level_side_then_nearest():
    pred = [
        _s((20, 30, 0), (10, -8, 0), "L4", "left"),
        _s((20, 30, 40), (10, -8, 40), "L3", "left"),
    ]
    ref = [
        _s((21, 30, 40), (11, -8, 40), "L3", "left"),
        _s((20, 30, 0), (10, -8, 0), "L4", "left"),
        _s((-20, 30, 0), (-10, -8, 0), "L4", "right"),
    ]
    pairs, up, ur = match_screws(pred, ref)
    assert len(pairs) == 2 and up == [] and len(ur) == 1
    assert {p.vertebra_level for p, _ in pairs} == {"L3", "L4"}
    assert ur[0].side == "right"


def test_match_duplicates_pair_by_nearest_entry():
    p1 = _s((20, 30, 0), (10, -8, 0))
    p2 = _s((60, 30, 0), (50, -8, 0))
    r_far = _s((62, 30, 0), (52, -8, 0))
    r_near = _s((21, 30, 0), (11, -8, 0))
    pairs, up, ur = match_screws([p1, p2], [r_far, r_near])
    assert up == [] and ur == []
    mapping = {id(pred): ref for pred, ref in pairs}
    assert mapping[id(p1)] is r_near
    assert mapping[id(p2)] is r_far


def test_match_reports_unmatched_on_both_sides():
    pred = [_s((20, 30, 0), (10, -8, 0), "L4", "left")]
    ref = [_s((20, 30, 0), (10, -8, 0), "L5", "left")]
    pairs, up, ur = match_screws(pred, ref)
    assert pairs == [] and len(up) == 1 and len(ur) == 1


# --- summarize -----------------------------------------------------------


def test_summary_bland_altman():
    a = _s((20, 30, 0), (10, -8, 0), d=6.0)
    b = _s((20, 30, 0), (10, -8, 0), d=6.5)
    s = summarize([compare_screw(a, b), compare_screw(a, b)], 0, 0)
    assert s.diameter_bias == pytest.approx(-0.5) and s.n_matched == 2
    assert s.diameter_loa == pytest.approx((-0.5, -0.5))


def test_summary_statistics_and_loa_spread():
    a = _s((20, 30, 0), (10, -8, 0), d=6.0)
    refs = [
        _s((20, 30, 0), (10, -8, 0), d=6.0),
        _s((20, 30, 0), (10, -8, 0), d=7.0),
    ]
    comparisons = [compare_screw(a, r) for r in refs]
    s = summarize(comparisons, 1, 2)
    assert s.n_unmatched_pred == 1 and s.n_unmatched_ref == 2
    deltas = [c.diameter_delta_mm for c in comparisons]
    bias = float(np.mean(deltas))
    sd = float(np.std(deltas, ddof=1))
    assert s.diameter_bias == pytest.approx(bias)
    assert s.diameter_loa[0] == pytest.approx(bias - 1.96 * sd)
    assert s.diameter_loa[1] == pytest.approx(bias + 1.96 * sd)
    assert s.head_mad_mean == pytest.approx(0.0)
    assert s.dice_mean == pytest.approx(
        float(np.mean([c.dice for c in comparisons]))
    )
    assert 0.8 < s.dice_mean < 1.0


def test_summary_single_comparison_has_zero_sd():
    a = _s((20, 30, 0), (10, -8, 0), d=6.0)
    b = _s((22, 30, 0), (12, -8, 0), d=6.5)
    s = summarize([compare_screw(a, b)], 0, 0)
    assert s.head_mad_sd == 0.0 and s.tip_mad_sd == 0.0 and s.axis_angle_sd == 0.0
    assert s.length_loa == pytest.approx((s.length_bias, s.length_bias))


def test_summary_of_empty_comparisons():
    s = summarize([], 3, 4)
    assert s.n_matched == 0 and s.n_unmatched_pred == 3 and s.n_unmatched_ref == 4
    assert s.head_mad_mean == 0.0 and s.dice_mean == 0.0
    assert s.diameter_loa == (0.0, 0.0)


# --- compare_plans -------------------------------------------------------


def test_compare_plans_matches_and_reads_pedicle_center():
    pred = [_s((20, 30, 0), (20, -10, 0), "L4", "left", d=6.0)]
    ref = [_s((20, 30, 0), (20, -10, 0), "L4", "left", d=5.5)]
    pred_payload = serialize_plan("s1", pred, [])
    ref_payload = serialize_plan("s1", ref, [])
    ref_payload["screws"][0]["pedicle_center"] = [23.0, 10.0, 0.0]

    comparisons, summary = compare_plans(pred_payload, ref_payload)
    assert len(comparisons) == 1 and summary.n_matched == 1
    assert comparisons[0].pedicle_center_offset_mm == pytest.approx(3.0)
    assert comparisons[0].diameter_delta_mm == pytest.approx(0.5)


def test_compare_plans_without_pedicle_center_and_with_unmatched():
    pred = [
        _s((20, 30, 0), (10, -8, 0), "L4", "left"),
        _s((-20, 30, 0), (-10, -8, 0), "L4", "right"),
    ]
    ref = [_s((20, 30, 0), (10, -8, 0), "L4", "left")]
    comparisons, summary = compare_plans(serialize_plan(None, pred, []),
                                         serialize_plan(None, ref, []))
    assert len(comparisons) == 1
    assert comparisons[0].pedicle_center_offset_mm is None
    assert summary.n_unmatched_pred == 1 and summary.n_unmatched_ref == 0


def test_compare_plans_accepts_minimal_v1_payload():
    payload_pred = {
        "version": 1,
        "screws": [
            {
                "entry_point": [20, 30, 0],
                "target_point": [10, -8, 0],
                "diameter": 6.0,
                "vertebra_level": "L4",
                "side": "left",
            }
        ],
    }
    payload_ref = {
        "version": 1,
        "screws": [
            {
                "entry_point": [21, 30, 0],
                "target_point": [11, -8, 0],
                "diameter": 6.0,
                "vertebra_level": "L4",
                "side": "left",
                "pedicle_center": [20.0, 10.0, 0.0],
            }
        ],
    }
    comparisons, summary = compare_plans(payload_pred, payload_ref)
    assert summary.n_matched == 1
    assert comparisons[0].head_mad_mm == pytest.approx(1.0)
    assert comparisons[0].pedicle_center_offset_mm is not None
