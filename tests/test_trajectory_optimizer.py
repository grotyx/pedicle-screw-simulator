"""Tests for the candidate-based multi-objective trajectory optimiser."""

import logging
import math
import os
import sys
import time

import numpy as np
import pytest
import SimpleITK as sitk

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.core.pedicle_analyzer import PedicleAnalyzer
from src.core.planner_config import PlannerConfig
from src.core.screw_grading import BatchResult, ScrewGrader
from src.core.trajectory_optimizer import (
    MAX_DIAMETER_STEPS,
    MAX_ENTRY_SHORTFALL_MM,
    TIP_SEGMENT_MM,
    OptimizerWeights,
    _seat_entries,
    generate_candidates,
    make_planner,
    optimize_screw,
    score_candidates,
)
from tests.test_pedicle_analyzer import _make_anatomical_phantom

LABEL = 28


def _setup(hu_gradient=False, with_arch=False):
    """Phantom + matching CT + pedicle analysis.

    The optimiser tests default to ``with_arch=False``: the phantom's laminar
    arch only touches the pedicles at their x axis and is separated from them by
    air elsewhere, so the posterior ray-cast entry lands inside the lamina with
    no drillable bone behind it -- an artefact of the phantom, which the
    reachability check correctly rejects (see
    ``test_buried_entry_on_arch_phantom_is_rejected``).
    """
    mask = _make_anatomical_phantom(with_arch=with_arch)
    arr = sitk.GetArrayFromImage(mask)
    hu = np.where(arr > 0, 300, -50).astype(np.int16)
    if hu_gradient:   # denser bone on the cranial half of the body
        hu[32:, :, :] = np.where(arr[32:, :, :] > 0, 600, -50)
    ct = sitk.GetImageFromArray(hu)
    ct.CopyInformation(mask)
    analyzer = PedicleAnalyzer(mask)
    analysis = analyzer.analyze_pedicle(analyzer.get_available_vertebrae()[0])
    return ct, mask, analysis


def test_candidates_are_generated_for_both_sides():
    ct, mask, analysis = _setup()
    grader = ScrewGrader(mask, ct)
    entries, targets, lengths = generate_candidates(grader, analysis, "left", PlannerConfig())
    assert entries.shape[0] > 100 and entries.shape == targets.shape
    assert set(np.unique(lengths)) <= set(PlannerConfig().implant_lengths_mm)


def test_best_candidate_is_feasible_and_through_pedicle():
    ct, mask, analysis = _setup()
    best = optimize_screw(ScrewGrader(mask, ct), analysis, "left", PlannerConfig())[0]
    assert best.breach_mm == 0.0 and best.min_wall_mm >= 1.0
    d = best.target - best.entry
    hits = sum(1 for t in np.linspace(0, 1, 40) if 48 <= (best.entry + d * t)[1] < 62 and abs((best.entry + d * t)[0] - 60) <= 4)
    assert hits >= 8


def test_density_weight_prefers_denser_bone():
    ct, mask, analysis = _setup(hu_gradient=True)
    grader = ScrewGrader(mask, ct)
    low = optimize_screw(grader, analysis, "left", PlannerConfig(), OptimizerWeights(density=0.0))[0]
    high = optimize_screw(grader, analysis, "left", PlannerConfig(), OptimizerWeights(density=3.0))[0]
    assert high.mean_hu >= low.mean_hu


def test_scores_have_named_components():
    ct, mask, analysis = _setup()
    best = optimize_screw(ScrewGrader(mask, ct), analysis, "left", PlannerConfig())[0]
    assert {"safety", "density", "length", "endplate", "centering"} <= set(best.components)


def test_rod_misalignment_zero_for_collinear_heads():
    from src.core.trajectory_optimizer import rod_misalignment_mm
    heads = np.array([[0, 0, 0], [0, 0, 30], [0, 0, 60]], float)
    assert rod_misalignment_mm(heads) == pytest.approx(0.0)
    assert rod_misalignment_mm(np.array([[0, 0, 0], [0, 5, 30], [0, 0, 60]], float)) > 1.0


def test_construct_prefers_aligned_heads_within_score_tolerance():
    from src.core.trajectory_optimizer import Candidate, OptimizerWeights, optimize_construct
    def cand(entry, score):
        return Candidate(entry=np.array(entry, float), target=np.array(entry, float) + [0, -40, 0], length=40, diameter=6,
                         breach_mm=0, min_wall_mm=2, mean_hu=300, convergence_deg=10, craniocaudal_deg=0, score=score, components={})
    per = {("L3", "left"): [cand([20, 30, 60], 1.00), cand([20, 30, 60], 0.99)],
           ("L4", "left"): [cand([26, 30, 30], 1.00), cand([20, 30, 30], 0.95)],   # misaligned best, aligned runner-up
           ("L5", "left"): [cand([20, 30, 0], 1.00)]}
    chosen = optimize_construct(per, OptimizerWeights(rod=1.0))
    assert chosen[("L4", "left")].entry[0] == pytest.approx(20.0)


# ---------------------------------------------------------------- reachability
def test_seated_entry_lies_on_corridor_surface():
    """The seated entry is the *last* point on the ray whose cross-section fits."""
    ct, mask, analysis = _setup()
    grader = ScrewGrader(mask, ct)
    center = np.asarray(analysis.left_pedicle_center, dtype=np.float64)
    direction = np.array([[0.0, -1.0, 0.0]])            # straight anterior
    radius = 3.0
    entries, travel = _seat_entries(
        grader, center[None, :], direction, np.array([20.0]), LABEL, radius
    )
    entry = entries[0]
    assert travel[0] == pytest.approx(float(np.linalg.norm(entry - center)))
    probe = np.array([entry, entry - direction[0] * 0.5])   # here, and 0.5 mm further back
    d_out, d_in = grader.distances_at_points(probe, LABEL)
    assert d_out[0] == 0.0 and d_in[0] >= radius            # the screw fits at the entry
    assert d_out[1] > 0.0 or d_in[1] < radius               # ...and not one step behind it


def test_buried_entry_on_arch_phantom_is_rejected():
    """The arch phantom's entry is 12+ mm inside the lamina, so nothing is reachable."""
    ct, mask, analysis = _setup(with_arch=True)
    grader = ScrewGrader(mask, ct)
    diagnostics = {}
    entries, _targets, _lengths = generate_candidates(
        grader, analysis, "left", PlannerConfig(), diagnostics=diagnostics
    )
    assert entries.shape[0] > 100
    assert diagnostics["surface_shortfall_mm"].min() > MAX_ENTRY_SHORTFALL_MM
    assert optimize_screw(grader, analysis, "left", PlannerConfig()) == []


def test_best_candidate_reaches_the_posterior_cortex():
    ct, mask, analysis = _setup()
    best = optimize_screw(ScrewGrader(mask, ct), analysis, "left", PlannerConfig())[0]
    assert best.surface_shortfall_mm <= MAX_ENTRY_SHORTFALL_MM


# --------------------------------------------------------------- runtime bound
def test_optimizer_bounds_runtime_and_caps_diameter_step_down():
    """An impossible anterior margin must fail fast, not walk the whole catalogue."""
    ct, mask, analysis = _setup()
    grader = ScrewGrader(mask, ct)
    tried = []
    graded = grader.evaluate_batch

    def counting(entries, targets, diameter, label):
        tried.append(diameter)
        return graded(entries, targets, diameter, label)

    grader.evaluate_batch = counting
    started = time.perf_counter()
    result = optimize_screw(grader, analysis, "left", PlannerConfig(anterior_margin_mm=15.0))
    elapsed = time.perf_counter() - started

    assert result == []
    # An order-of-magnitude regression guard, not a benchmark: the measured
    # runtime is ~3.2 s locally and a shared CI runner is slower still.
    assert elapsed <= 10.0, f"optimize_screw took {elapsed:.2f} s"
    assert len(set(tried)) == MAX_DIAMETER_STEPS + 1
    assert sorted(set(tried), reverse=True) == [6.5, 6.0, 5.5]


def test_tip_margin_rejects_trajectories_without_anterior_clearance():
    ct, mask, analysis = _setup()
    grader = ScrewGrader(mask, ct)
    best = optimize_screw(grader, analysis, "left", PlannerConfig())[0]
    entries, targets = best.entry[None, :], best.target[None, :]
    lengths = np.array([best.length])
    direction = (best.target - best.entry) / best.length
    batch = grader.evaluate_batch(entries, targets, best.diameter, LABEL)
    tip = grader.evaluate_batch(
        targets - direction * TIP_SEGMENT_MM, targets, best.diameter, LABEL
    )
    args = (batch, entries, targets, lengths, best.diameter, analysis, "left")
    assert score_candidates(*args, PlannerConfig(), OptimizerWeights(), tip_batch=tip)
    assert score_candidates(
        *args, PlannerConfig(anterior_margin_mm=15.0), OptimizerWeights(), tip_batch=tip
    ) == []


# ------------------------------------------------------------------- behaviour
def test_diameter_steps_down_when_recommendation_does_not_fit():
    ct, mask, analysis = _setup()
    grader = ScrewGrader(mask, ct)
    planner = make_planner(grader, PlannerConfig())
    recommended = planner._compute_diameter(analysis.left_pedicle_width, analysis.vertebra.name)
    assert recommended == 6.5
    best = optimize_screw(grader, analysis, "left", PlannerConfig())[0]
    assert best.diameter == 6.0
    assert "Diameter reduced from 6.5 to 6.0 mm for cortical containment" in best.warnings


def test_right_side_convergence_is_mirrored():
    ct, mask, analysis = _setup()
    grader = ScrewGrader(mask, ct)
    left = optimize_screw(grader, analysis, "left", PlannerConfig())[0]
    right = optimize_screw(grader, analysis, "right", PlannerConfig())[0]
    assert left.convergence_deg > 0.0 and right.convergence_deg > 0.0
    assert right.convergence_deg == pytest.approx(left.convergence_deg)
    assert left.target[0] - left.entry[0] < 0.0     # left pedicle aims toward -X
    assert right.target[0] - right.entry[0] > 0.0   # right pedicle aims toward +X


def test_density_weight_reorders_scored_candidates():
    """Score ordering on a synthetic batch: denser but shorter can outrank longer."""
    ct, mask, analysis = _setup()
    entries = np.array([[60.0, 58.0, 32.0], [60.0, 58.0, 32.0]])
    targets = np.array([[60.0, 28.0, 32.0], [60.0, 33.0, 32.0]])
    lengths = np.array([30.0, 25.0])                    # long/sparse, then short/dense
    batch = BatchResult(
        breach_mm=np.zeros(2),
        min_wall_mm=np.array([2.0, 2.0]),
        mean_hu=np.array([200.0, 600.0]),
        min_hu=np.array([200.0, 600.0]),
    )
    args = (batch, entries, targets, lengths, 6.0, analysis, "left", PlannerConfig())
    ignored = score_candidates(*args, OptimizerWeights(density=0.0))
    weighted = score_candidates(*args, OptimizerWeights(density=3.0))
    assert [c.length for c in ignored] == [30.0, 25.0]   # length alone decides
    assert [c.length for c in weighted] == [25.0, 30.0]  # density overturns it
    assert weighted[0].mean_hu == 600.0


def test_grader_without_ct_still_yields_candidates():
    _ct, mask, analysis = _setup()
    ranked = optimize_screw(ScrewGrader(mask), analysis, "left", PlannerConfig())
    assert ranked
    assert math.isnan(ranked[0].mean_hu)
    assert ranked[0].components["density"] == 0.0


def test_missing_ct_is_warned_once_per_grader(caplog):
    """Each CT-less study warns once; a second study is not silently downgraded."""
    _ct, mask, analysis = _setup()
    config = PlannerConfig()

    with caplog.at_level(logging.WARNING, logger="src.core.trajectory_optimizer"):
        for _ in range(2):                       # two distinct short-lived graders
            optimize_screw(ScrewGrader(mask), analysis, "left", config)
        reused = ScrewGrader(mask)               # one grader, graded twice
        optimize_screw(reused, analysis, "left", config)
        optimize_screw(reused, analysis, "right", config)

    warned = [r for r in caplog.records if "has no CT" in r.getMessage()]
    assert len(warned) == 3


def test_a_prebuilt_planner_gives_the_same_ranking():
    """Reusing a planner must save the volume copy, not change the answer."""
    ct, mask, analysis = _setup()
    grader = ScrewGrader(mask, ct)
    config = PlannerConfig()

    fresh = optimize_screw(grader, analysis, "left", config)
    reused = optimize_screw(
        grader, analysis, "left", config, planner=make_planner(grader, config)
    )

    assert fresh and len(fresh) == len(reused)
    for a, b in zip(fresh, reused, strict=True):
        assert a.score == b.score
        assert a.diameter == b.diameter
        np.testing.assert_array_equal(a.entry, b.entry)
        np.testing.assert_array_equal(a.target, b.target)


def test_plan_all_optimized_does_not_build_a_planner_per_pedicle(monkeypatch):
    from src.core import trajectory_optimizer
    from src.core.auto_screw_planner import AutoScrewPlanner

    ct, mask, analysis = _setup()
    built = []
    real = trajectory_optimizer.make_planner
    monkeypatch.setattr(
        trajectory_optimizer,
        "make_planner",
        lambda *args, **kwargs: (built.append(1), real(*args, **kwargs))[1],
    )

    planner = AutoScrewPlanner(ct, mask, config=PlannerConfig(mode="optimizer"))
    screws = planner.plan_all([analysis])

    assert screws
    assert built == []      # the planner lent itself to every pedicle
