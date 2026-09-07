"""Tests for the candidate-based multi-objective trajectory optimiser."""

import os
import sys

import numpy as np
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
