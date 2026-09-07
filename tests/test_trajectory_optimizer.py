"""Tests for the candidate-based multi-objective trajectory optimiser."""

import os
import sys

import numpy as np
import pytest
import SimpleITK as sitk

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.core.pedicle_analyzer import PedicleAnalyzer
from src.core.planner_config import PlannerConfig
from src.core.screw_grading import ScrewGrader
from src.core.trajectory_optimizer import OptimizerWeights, generate_candidates, optimize_screw
from tests.test_pedicle_analyzer import _make_anatomical_phantom


def _setup(hu_gradient=False):
    mask = _make_anatomical_phantom()
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
