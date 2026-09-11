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
from src.core.screw_geometry import endplate_angle_deg, endplate_slope_deg
from src.core.screw_grading import BatchResult, ScrewGrader
from src.core.trajectory_optimizer import (
    DEFAULT_WEIGHTS,
    ENDPLATE_BAND_RELAXED_PREFIX,
    MAX_DIAMETER_STEPS,
    MAX_ENTRY_SHORTFALL_MM,
    SAFETY_CAP_MM,
    TIP_SEGMENT_MM,
    OptimizerWeights,
    _seat_entries,
    endplate_band_relaxed_warning,
    generate_candidates,
    make_planner,
    optimize_screw,
    score_candidates,
)
from tests.test_pedicle_analyzer import (
    _make_anatomical_phantom,
    _make_tilted_endplate_phantom,
)

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

    def counting(entries, targets, diameter, label, side=None):
        tried.append(diameter)
        return graded(entries, targets, diameter, label, side=side)

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
    assert best.diameter == 5.5
    assert "Diameter reduced from 6.5 to 5.5 mm for cortical containment" in best.warnings


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


# ------------------------------------------------------------- narrow pedicles
def _narrow_setup(**phantom_kwargs):
    """The narrow phantom plus a matching CT and its analysis."""
    from tests.test_pedicle_analyzer import _make_narrow_pedicle_phantom

    mask = _make_narrow_pedicle_phantom(**phantom_kwargs)
    arr = sitk.GetArrayFromImage(mask)
    ct = sitk.GetImageFromArray(np.where(arr > 0, 300, -50).astype(np.int16))
    ct.CopyInformation(mask)
    analyzer = PedicleAnalyzer(mask)
    analysis = analyzer.analyze_pedicle(analyzer.get_available_vertebrae()[0])
    return ct, mask, analysis


@pytest.mark.parametrize("side", ["left", "right"])
def test_a_narrow_pedicle_has_no_contained_trajectory(side):
    """Without the narrow mode the 3.5 mm corridor simply has no answer."""
    ct, mask, analysis = _narrow_setup()

    assert optimize_screw(ScrewGrader(mask, ct), analysis, side, PlannerConfig()) == []


@pytest.mark.parametrize("side", ["left", "right"])
def test_the_narrow_mode_protects_the_medial_wall(side):
    ct, mask, analysis = _narrow_setup()
    config = PlannerConfig()

    ranked = optimize_screw(
        ScrewGrader(mask, ct), analysis, side, config, narrow=True
    )

    assert ranked
    best = ranked[0]
    assert best.diameter == pytest.approx(4.0)          # no step-down for a narrow side
    assert best.medial_breach_mm == 0.0
    assert best.craniocaudal_breach_mm == 0.0
    assert 0.0 < best.lateral_breach_mm <= config.narrow_lateral_breach_mm
    # The corridor was pushed away from the canal, so the head sits lateral of
    # the isthmus centre.
    centre = (
        analysis.left_pedicle_center if side == "left" else analysis.right_pedicle_center
    )
    medial_sign = -1.0 if side == "left" else 1.0
    assert medial_sign * (best.entry[0] - centre[0]) < -0.5
    assert best.components.keys() == {
        "safety", "density", "length", "endplate", "lateral"
    }
    assert best.components["safety"] == pytest.approx(
        min(best.medial_wall_mm, SAFETY_CAP_MM) / SAFETY_CAP_MM
    )


def _stepped_narrow_setup():
    """A narrow phantom whose candidates breach laterally by *different* amounts.

    The default phantom admits a 4.0 mm screw at exactly one height and one
    lateral offset, so every feasible candidate breaches by the same 1.5 mm and
    a cap could only take all of them or none -- which would make a cap test
    vacuous.  Padding the lateral cortex of the caudal half by 1 mm, in a
    corridor tall enough for the sweep's craniocaudal offsets to matter, spreads
    the survivors over several lateral breaches at an unchanged medial wall.
    """
    return _narrow_setup(half_height_mm=6.0, lateral_relief_mm=1.0)


@pytest.mark.parametrize("cap, expected", [(1.0, {0.0, 0.5, 1.0}), (2.0, {1.5, 2.0})])
def test_every_narrow_candidate_respects_the_lateral_cap(cap, expected):
    """The cap is a hard bound, and it is load-bearing: it removes candidates."""
    ct, mask, analysis = _stepped_narrow_setup()
    grader = ScrewGrader(mask, ct)

    def ranked_at(limit):
        return optimize_screw(
            grader,
            analysis,
            "left",
            PlannerConfig(narrow_lateral_breach_mm=limit),
            narrow=True,
            top_k=100000,
        )

    ranked = ranked_at(cap)

    assert ranked, "the cap test must not pass over an empty candidate list"
    assert all(c.medial_breach_mm == 0.0 for c in ranked)
    assert all(c.craniocaudal_breach_mm == 0.0 for c in ranked)
    assert all(c.lateral_breach_mm <= cap + 1e-9 for c in ranked)
    breaches = {round(c.lateral_breach_mm, 2) for c in ranked}
    assert breaches >= expected, "the phantom must offer a spread of lateral breaches"
    # Tightening the cap really does cut the survivors, rather than the cap
    # being satisfied vacuously by geometry that could never exceed it.
    assert 0 < len(ranked_at(cap - 0.5)) < len(ranked)


def test_the_lateral_term_prefers_the_smaller_breach():
    """At an equal medial wall the cheaper lateral breach wins the tie."""
    ct, mask, analysis = _setup()
    entries = np.array([[60.0, 58.0, 32.0], [60.0, 58.0, 32.0]])
    targets = np.array([[60.0, 28.0, 32.0], [60.0, 28.0, 32.0]])
    lengths = np.array([30.0, 30.0])
    batch = BatchResult(                                # identical but for lateral
        breach_mm=np.array([1.5, 0.5]),
        min_wall_mm=np.zeros(2),
        mean_hu=np.array([300.0, 300.0]),
        min_hu=np.array([300.0, 300.0]),
        medial_breach_mm=np.zeros(2),
        lateral_breach_mm=np.array([1.5, 0.5]),
        craniocaudal_breach_mm=np.zeros(2),
        medial_wall_mm=np.array([1.0, 1.0]),
    )

    ranked = score_candidates(
        batch, entries, targets, lengths, 4.0, analysis, "left",
        PlannerConfig(narrow_lateral_breach_mm=2.0), DEFAULT_WEIGHTS, narrow=True,
    )

    assert [c.lateral_breach_mm for c in ranked] == [0.5, 1.5]
    assert ranked[0].components["safety"] == ranked[1].components["safety"]
    assert ranked[0].components["lateral"] == pytest.approx(0.75)
    assert ranked[1].components["lateral"] == pytest.approx(0.25)


def test_a_narrow_side_is_planned_at_the_minimum_diameter():
    """An implausible width flags a side narrow even when it reads far too wide.

    ``_is_narrow_side`` fires on ``width_flags`` alone, so a 25 mm "measurement"
    reaches the narrow path with a level recommendation of a full-size screw.
    The relaxed feasibility rule must never be handed that screw: the narrow
    policy is the smallest implant, exactly as the legacy planner does it.
    """
    ct, mask, analysis = _setup()
    grader = ScrewGrader(mask, ct)
    planner = make_planner(grader, PlannerConfig())
    analysis.left_pedicle_width = 25.0
    analysis.width_flags["left"] = "implausible"
    assert planner._compute_diameter(25.0, analysis.vertebra.name) > 4.0

    ranked = optimize_screw(grader, analysis, "left", PlannerConfig(), narrow=True)

    assert ranked
    assert all(c.diameter == pytest.approx(planner.MIN_SCREW_DIAMETER) for c in ranked)
    assert all(c.diameter == pytest.approx(4.0) for c in ranked)
    # The minimum is the policy, not a containment step-down, so nothing claims it was.
    assert not any("Diameter reduced" in w for c in ranked for w in c.warnings)


def test_a_normal_pedicle_is_unchanged_by_the_narrow_plumbing():
    """narrow=False must reproduce today's answer on the anatomical phantom."""
    ct, mask, analysis = _setup()
    grader = ScrewGrader(mask, ct)

    best = optimize_screw(grader, analysis, "left", PlannerConfig())[0]

    assert best.breach_mm == 0.0
    assert best.components.keys() == {
        "safety", "density", "length", "endplate", "centering"
    }
    assert best.medial_breach_mm == 0.0 and best.lateral_breach_mm == 0.0
    assert best.medial_wall_mm >= best.min_wall_mm - 1e-9


def _tilted_setup(tilt_deg=10.0):
    """Tilted-endplate phantom + flat CT + analysis, for the endplate tests."""
    mask = _make_tilted_endplate_phantom(tilt_deg=tilt_deg)
    arr = sitk.GetArrayFromImage(mask)
    ct = sitk.GetImageFromArray(np.where(arr > 0, 300, -50).astype(np.int16))
    ct.CopyInformation(mask)
    analyzer = PedicleAnalyzer(mask)
    analysis = analyzer.analyze_pedicle(analyzer.get_available_vertebrae()[0])
    return ct, mask, analysis


def _endplate_angles(candidates, analysis):
    return [
        endplate_angle_deg(c.entry, c.target, analysis.upper_endplate_normal)
        for c in candidates
    ]


def test_endplate_band_holds_every_candidate_inside_the_tolerance():
    """Without the band this phantom yields trajectories 14 degrees off."""
    ct, mask, analysis = _tilted_setup()
    assert endplate_slope_deg(analysis.upper_endplate_normal) == pytest.approx(10.0, abs=1.5)

    ranked = optimize_screw(
        ScrewGrader(mask, ct), analysis, "left", PlannerConfig(), top_k=500
    )

    assert ranked
    angles = _endplate_angles(ranked, analysis)
    assert max(abs(a) for a in angles) <= 10.0 + 1e-6
    assert not any(w.startswith(ENDPLATE_BAND_RELAXED_PREFIX) for w in ranked[0].warnings)


def test_recentred_sweep_covers_the_band_symmetrically():
    """The generated grid brackets the endplate direction instead of the axis.

    Measured on this phantom (independent of the corridor radius): centred on
    the pedicle axis the grid spans -19.37 to +0.63 degrees of endplate angle;
    centred on the endplate it spans -10.0 to +10.0.
    """
    ct, mask, analysis = _tilted_setup()
    grader = ScrewGrader(mask, ct)
    normal = analysis.upper_endplate_normal

    on_entries, on_targets, _ = generate_candidates(
        grader, analysis, "left", PlannerConfig(), corridor_radius_mm=2.5
    )
    off_entries, off_targets, _ = generate_candidates(
        grader, analysis, "left", PlannerConfig(endplate_parallel=False), corridor_radius_mm=2.5
    )
    on_angles = [endplate_angle_deg(e, t, normal) for e, t in zip(on_entries, on_targets, strict=True)]
    off_angles = [endplate_angle_deg(e, t, normal) for e, t in zip(off_entries, off_targets, strict=True)]

    assert max(on_angles) > max(off_angles) + 4.0
    assert min(on_angles) > min(off_angles) + 4.0
    assert max(on_angles) == pytest.approx(-min(on_angles), abs=1.0)


def test_recentred_sweep_reaches_trajectories_parallel_to_a_tilted_endplate():
    """Because the band is populated, the winner is very nearly parallel.

    On the axis-centred grid the best feasible 7.0 mm candidate sits 4.4 degrees
    caudal of the endplate; with the sweep recentred it sits 0.1 degrees off.
    """
    ct, mask, analysis = _tilted_setup()

    ranked = optimize_screw(
        ScrewGrader(mask, ct), analysis, "left", PlannerConfig(), top_k=500
    )

    best = endplate_angle_deg(ranked[0].entry, ranked[0].target, analysis.upper_endplate_normal)
    assert abs(best) <= 2.0


def test_endplate_option_off_leaves_the_component_neutral_and_the_band_open():
    ct, mask, analysis = _tilted_setup()
    grader = ScrewGrader(mask, ct)
    config = PlannerConfig(endplate_parallel=False)

    ranked = optimize_screw(grader, analysis, "left", config, top_k=500)

    assert ranked
    assert all(c.components["endplate"] == pytest.approx(1.0) for c in ranked)
    # Not banded: this phantom's un-recentred sweep reaches 14 degrees off.
    assert max(abs(a) for a in _endplate_angles(ranked, analysis)) > 10.0


def test_neutral_endplate_component_does_not_change_the_ranking():
    """Setting the component to 1.0 for everyone must be rank-neutral, which is
    what makes "off" equivalent to today's behaviour on a flat phantom."""
    ct, mask, analysis = _tilted_setup()
    grader = ScrewGrader(mask, ct)
    config = PlannerConfig(endplate_parallel=False)

    with_weight = optimize_screw(grader, analysis, "left", config, OptimizerWeights(), top_k=5)
    without_weight = optimize_screw(
        grader, analysis, "left", config, OptimizerWeights(endplate=0.0), top_k=5
    )

    assert np.allclose(with_weight[0].entry, without_weight[0].entry)
    assert np.allclose(with_weight[0].target, without_weight[0].target)


def test_impossible_band_is_relaxed_with_a_warning_instead_of_dropping_the_side():
    ct, mask, analysis = _tilted_setup()
    config = PlannerConfig(endplate_tolerance_deg=0.0)

    ranked = optimize_screw(ScrewGrader(mask, ct), analysis, "left", config, top_k=500)

    assert ranked, "a band nothing satisfies must not cost the side its screw"
    warning = endplate_band_relaxed_warning(0.0)
    assert warning == "Endplate band relaxed: no trajectory within ±0° of the upper endplate"
    assert all(warning in c.warnings for c in ranked)


def test_flat_phantom_without_an_endplate_normal_is_unaffected():
    """The band and the recentring are no-ops when there is no plane to use."""
    ct, mask, analysis = _setup()
    analysis.upper_endplate_normal = None

    ranked = optimize_screw(ScrewGrader(mask, ct), analysis, "left", PlannerConfig(), top_k=10)

    assert ranked
    assert all(c.components["endplate"] == pytest.approx(1.0) for c in ranked)
    assert not any(w.startswith(ENDPLATE_BAND_RELAXED_PREFIX) for w in ranked[0].warnings)

# ------------------------------------------------------- convergence spread
def test_convergence_spread_is_zero_below_two_screws():
    from src.core.trajectory_optimizer import convergence_spread_deg

    assert convergence_spread_deg([]) == pytest.approx(0.0)
    assert convergence_spread_deg([12.0]) == pytest.approx(0.0)
    assert convergence_spread_deg([12.0, 12.0, 12.0]) == pytest.approx(0.0)


def test_convergence_spread_without_levels_is_rms_about_the_median():
    from src.core.trajectory_optimizer import convergence_spread_deg

    # median 10 -> deviations -5, 0, +5
    assert convergence_spread_deg([5.0, 10.0, 15.0]) == pytest.approx(math.sqrt(50.0 / 3.0))


def test_convergence_deviations_report_each_screws_offset():
    from src.core.trajectory_optimizer import convergence_deviations_deg

    assert convergence_deviations_deg([5.0, 10.0, 15.0]) == pytest.approx([-5.0, 0.0, 5.0])


def test_s1_is_excluded_from_the_convergence_term():
    from src.core.trajectory_optimizer import (
        convergence_deviations_deg,
        convergence_spread_deg,
    )

    angles = [10.0, 10.0, 40.0]
    levels = [28, 27, 26]           # L4, L5, S1 (see pedicle_analyzer.VERTEBRA_LABELS)

    assert convergence_spread_deg(angles, levels) == pytest.approx(0.0)
    assert convergence_deviations_deg(angles, levels)[2] is None
    # Including it would let one sacral screw dominate the whole side.
    assert convergence_spread_deg(angles, levels, exclude_s1=False) == pytest.approx(
        math.sqrt(300.0)
    )


def test_neighbouring_levels_count_double_in_the_median():
    """A construct that steps from 20 deg to 5 deg at L3/L4 is judged locally."""
    from src.core.trajectory_optimizer import (
        convergence_deviations_deg,
        convergence_spread_deg,
    )

    angles = [20.0, 20.0, 20.0, 5.0, 5.0]
    levels = [31, 30, 29, 28, 27]           # L1, L2, L3, L4, L5

    weighted = convergence_spread_deg(angles, levels)
    unweighted = convergence_spread_deg(angles)     # plain median of the whole side

    assert weighted == pytest.approx(math.sqrt(11.25))
    assert unweighted == pytest.approx(math.sqrt(90.0))
    assert weighted < unweighted
    assert convergence_deviations_deg(angles, levels) == pytest.approx(
        [0.0, 0.0, 0.0, -7.5, 0.0]
    )
