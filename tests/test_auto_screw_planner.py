"""
Tests for AutoScrewPlanner — automatic pedicle screw trajectory planning.

Uses synthetic SimpleITK volumes with known geometry so that expected
entry/target points, HU profiles, and Gertzbein grades can be verified
analytically.
"""

import math
import os
import sys

import numpy as np
import pytest
import SimpleITK as sitk

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.core.auto_screw_planner import AutoScrewPlanner, PlannedScrew
from src.core.vertebra import PedicleAnalysisResult, Vertebra

# ---------------------------------------------------------------------------
# Helpers for building synthetic CT and mask volumes
# ---------------------------------------------------------------------------

def _make_image(
    array: np.ndarray,
    spacing: tuple = (1.0, 1.0, 1.0),
    origin: tuple = (0.0, 0.0, 0.0),
) -> sitk.Image:
    """Wrap a numpy (z, y, x) array into a SimpleITK image.

    ``spacing`` and ``origin`` are given in numpy (z, y, x) order and
    will be reversed internally to match SimpleITK's (x, y, z) convention.
    """
    img = sitk.GetImageFromArray(array)
    img.SetSpacing(tuple(reversed(spacing)))
    img.SetOrigin(tuple(reversed(origin)))
    return img


def _make_bone_cylinder(
    shape: tuple = (40, 60, 60),
    label: int = 27,
    hu_value: float = 500.0,
    spacing: tuple = (1.0, 1.0, 1.0),
) -> tuple:
    """Create a synthetic CT + mask with a simple vertebra shape.

    The vertebra consists of:
    - A vertebral body (anterior, lower y): z=[10..30], y=[10..30], x=[20..40]
    - Left pedicle:  z=[10..30], y=[30..45], x=[35..45]
    - Right pedicle: z=[10..30], y=[30..45], x=[15..25]

    Returns (ct_image, mask_image).
    """
    ct_arr = np.full(shape, -1000.0, dtype=np.float32)  # Air
    mask_arr = np.zeros(shape, dtype=np.uint8)

    # Vertebral body (anterior = lower y in LPS with identity direction).
    ct_arr[10:30, 5:32, 20:40] = hu_value
    mask_arr[10:30, 5:32, 20:40] = label

    # Left pedicle (higher x = patient left).
    ct_arr[10:30, 30:45, 35:45] = hu_value
    mask_arr[10:30, 30:45, 35:45] = label

    # Right pedicle (lower x = patient right).
    ct_arr[10:30, 30:45, 15:25] = hu_value
    mask_arr[10:30, 30:45, 15:25] = label

    ct_img = _make_image(ct_arr, spacing)
    mask_img = _make_image(mask_arr, spacing)
    return ct_img, mask_img


def _make_vertebra(
    label: int = 27,
    name: str = "L5",
    centroid_lps: np.ndarray = None,
) -> Vertebra:
    """Create a minimal Vertebra for testing."""
    if centroid_lps is None:
        centroid_lps = np.array([30.0, 25.0, 20.0])
    return Vertebra(
        label=label,
        name=name,
        centroid_lps=centroid_lps,
        bounding_box=(np.array([15.0, 10.0, 10.0]), np.array([45.0, 45.0, 30.0])),
        volume_mm3=5000.0,
        mask_indices=np.array([[20, 20, 20], [30, 30, 30]]),
    )


def _make_analysis(
    label: int = 27,
    name: str = "L5",
    left_center: np.ndarray = None,
    right_center: np.ndarray = None,
    left_axis: np.ndarray = None,
    right_axis: np.ndarray = None,
    left_width: float = 8.0,
    right_width: float = 8.0,
    body_center: np.ndarray = None,
    upper_endplate_normal: np.ndarray = None,
    success: bool = True,
) -> PedicleAnalysisResult:
    """Create a PedicleAnalysisResult with sensible defaults for the
    synthetic bone cylinder geometry."""
    vertebra = _make_vertebra(label, name)

    if left_center is None:
        left_center = np.array([40.0, 37.0, 20.0])
    if right_center is None:
        right_center = np.array([20.0, 37.0, 20.0])
    if left_axis is None:
        # Points toward the body (anteriorly / -Y) from the pedicle.
        axis = np.array([0.0, -1.0, 0.0])
        left_axis = axis / np.linalg.norm(axis)
    if right_axis is None:
        axis = np.array([0.0, -1.0, 0.0])
        right_axis = axis / np.linalg.norm(axis)
    if body_center is None:
        body_center = np.array([30.0, 18.0, 20.0])

    return PedicleAnalysisResult(
        vertebra=vertebra,
        left_pedicle_center=left_center,
        right_pedicle_center=right_center,
        left_pedicle_axis=left_axis,
        right_pedicle_axis=right_axis,
        left_pedicle_width=left_width,
        right_pedicle_width=right_width,
        vertebral_body_center=body_center,
        upper_endplate_normal=upper_endplate_normal,
        success=success,
    )


# ---------------------------------------------------------------------------
# PlannedScrew dataclass tests
# ---------------------------------------------------------------------------

class TestPlannedScrewDataclass:
    """Verify PlannedScrew construction and validation."""

    def test_basic_construction(self):
        ps = PlannedScrew(
            vertebra_name="L4",
            side="left",
            entry_lps=np.array([1.0, 2.0, 3.0]),
            target_lps=np.array([1.0, 2.0, 33.0]),
            length_mm=30.0,
            diameter_mm=6.0,
            convergence_angle=10.0,
            craniocaudal_angle=5.0,
            mean_bone_density=450.0,
            min_bone_density=250.0,
            gertzbein_grade="A",
            confidence=0.85,
        )
        assert ps.vertebra_name == "L4"
        assert ps.side == "left"
        assert ps.entry_lps.dtype == np.float64
        assert ps.target_lps.shape == (3,)
        assert ps.warnings == []

    def test_warnings_list(self):
        ps = PlannedScrew(
            vertebra_name="T12",
            side="right",
            entry_lps=[0, 0, 0],
            target_lps=[0, 0, 30],
            length_mm=30.0,
            diameter_mm=5.0,
            convergence_angle=0.0,
            craniocaudal_angle=0.0,
            mean_bone_density=300.0,
            min_bone_density=200.0,
            gertzbein_grade="B",
            confidence=0.5,
            warnings=["test warning"],
        )
        assert len(ps.warnings) == 1
        assert ps.warnings[0] == "test warning"


# ---------------------------------------------------------------------------
# AutoScrewPlanner: entry point finding
# ---------------------------------------------------------------------------

class TestEntryPointFinding:
    """Test _find_entry_point with synthetic bone geometry."""

    def test_entry_point_on_posterior_surface(self):
        """Entry should be on or near the posterior boundary of bone."""
        ct, mask = _make_bone_cylinder()
        planner = AutoScrewPlanner(ct, mask)

        # Left pedicle centre is at (40, 37, 20) in LPS.
        # Pedicle spans y=[30..45] in array coords.
        # Posterior axis = +Y in LPS.
        center = np.array([40.0, 37.0, 20.0])
        axis = np.array([0.0, 1.0, 0.0])  # posterior direction

        entry = planner._find_entry_point(center, axis, vertebra_label=27)

        assert entry is not None
        # Entry y should be near the posterior boundary (y~44).
        # The bone extends to y=44 in array-index space.
        # With identity direction, physical y == index y.
        assert entry[1] > 40.0  # well into the posterior region
        assert entry[1] < 46.0  # not outside the bone

    def test_entry_returns_none_outside_volume(self):
        """If pedicle centre is outside volume, return None."""
        ct, mask = _make_bone_cylinder(shape=(10, 10, 10))
        planner = AutoScrewPlanner(ct, mask)

        center = np.array([100.0, 100.0, 100.0])
        axis = np.array([0.0, 1.0, 0.0])

        entry = planner._find_entry_point(center, axis, vertebra_label=27)
        assert entry is None

    def test_entry_uses_vertebra_label_not_adjacent_high_hu(self):
        _, mask = _make_bone_cylinder()
        ct_arr = np.full((40, 60, 60), 1200.0, dtype=np.float32)
        ct = _make_image(ct_arr)
        planner = AutoScrewPlanner(ct, mask)

        entry = planner._find_entry_point(
            np.array([40.0, 37.0, 20.0]),
            np.array([0.0, 1.0, 0.0]),
            vertebra_label=27,
        )

        assert entry is not None
        assert 40.0 < entry[1] < 45.0


# ---------------------------------------------------------------------------
# AutoScrewPlanner: target point finding
# ---------------------------------------------------------------------------

class TestTargetPointFinding:
    """Test _find_target_point with synthetic geometry."""

    def test_endplate_aligned_target_z_uses_sagittal_plane_slope(self):
        normal = np.array([0.0, -0.2, 1.0])
        normal /= np.linalg.norm(normal)
        entry = np.array([40.0, 43.0, 20.0])

        target_z = AutoScrewPlanner._endplate_aligned_target_z(
            entry,
            target_y=18.0,
            upper_endplate_normal=normal,
        )

        assert target_z == pytest.approx(15.0)

    def test_target_in_anterior_body(self):
        """Target should be inside the vertebral body, anterior to entry."""
        ct, mask = _make_bone_cylinder()
        planner = AutoScrewPlanner(ct, mask)

        entry = np.array([40.0, 43.0, 20.0])
        axis = np.array([0.0, 1.0, 0.0])
        body_center = np.array([30.0, 20.0, 20.0])

        target = planner._find_target_point(entry, axis, body_center, vertebra_label=27)

        assert target is not None
        # Target should be anterior to entry (lower y in LPS).
        assert target[1] < entry[1]

    def test_target_respects_length_constraint(self):
        """Target distance from entry should not exceed MAX_SCREW_LENGTH."""
        ct, mask = _make_bone_cylinder()
        planner = AutoScrewPlanner(ct, mask)

        entry = np.array([40.0, 43.0, 20.0])
        axis = np.array([0.0, 1.0, 0.0])
        body_center = np.array([30.0, 18.0, 20.0])

        target = planner._find_target_point(entry, axis, body_center, vertebra_label=27)
        length = np.linalg.norm(target - entry)
        assert length <= planner.MAX_SCREW_LENGTH + 0.1

    def test_local_target_search_prefers_lower_breach_candidate(self, monkeypatch):
        ct, mask = _make_bone_cylinder()
        planner = AutoScrewPlanner(ct, mask)
        entry = np.array([40.0, 43.0, 20.0])
        body_center = np.array([30.0, 18.0, 20.0])

        monkeypatch.setattr(
            planner,
            "_find_target_point",
            lambda _entry, _axis, anchor, _label: np.array(
                [anchor[0], 10.0, anchor[2]]
            ),
        )
        monkeypatch.setattr(
            planner,
            "_evaluate_gertzbein_grade",
            lambda _entry, target, _diameter, _label: (
                "A",
                abs(target[0] - 33.0),
            ),
        )
        monkeypatch.setattr(
            planner,
            "_sample_hu_along_trajectory",
            lambda *_args, **_kwargs: (500.0, 300.0, []),
        )

        target = planner._find_best_target(
            entry,
            np.array([0.0, 1.0, 0.0]),
            body_center,
            vertebra_label=27,
            diameter=6.0,
        )

        assert target is not None
        assert target[0] == pytest.approx(33.0)

    def test_local_target_search_keeps_target_on_entry_side_of_midline(
        self,
        monkeypatch,
    ):
        ct, mask = _make_bone_cylinder()
        planner = AutoScrewPlanner(ct, mask)
        entry = np.array([40.0, 43.0, 20.0])
        body_center = np.array([30.0, 18.0, 20.0])

        monkeypatch.setattr(
            planner,
            "_find_target_point",
            lambda _entry, _axis, anchor, _label: np.array(
                [anchor[0], 10.0, anchor[2]]
            ),
        )
        monkeypatch.setattr(
            planner,
            "_evaluate_gertzbein_grade",
            lambda _entry, target, _diameter, _label: (
                "A",
                0.0 if target[0] < body_center[0] else 1.0,
            ),
        )
        monkeypatch.setattr(
            planner,
            "_sample_hu_along_trajectory",
            lambda *_args, **_kwargs: (500.0, 300.0, []),
        )

        target = planner._find_best_target(
            entry,
            np.array([0.0, 1.0, 0.0]),
            body_center,
            vertebra_label=27,
            diameter=6.0,
        )

        assert target is not None
        assert target[0] >= body_center[0]

    def test_local_target_search_rejects_excessive_convergence(self, monkeypatch):
        ct, mask = _make_bone_cylinder()
        planner = AutoScrewPlanner(ct, mask)
        entry = np.array([55.0, 43.0, 20.0])
        body_center = np.array([30.0, 18.0, 20.0])

        monkeypatch.setattr(
            planner,
            "_find_target_point",
            lambda _entry, _axis, anchor, _label: np.array(
                [anchor[0], 10.0, anchor[2]]
            ),
        )
        monkeypatch.setattr(
            planner,
            "_evaluate_gertzbein_grade",
            lambda *_args: ("A", 0.0),
        )
        monkeypatch.setattr(
            planner,
            "_sample_hu_along_trajectory",
            lambda *_args, **_kwargs: (500.0, 300.0, []),
        )

        target = planner._find_best_target(
            entry,
            np.array([0.0, 1.0, 0.0]),
            body_center,
            vertebra_label=27,
            diameter=6.0,
        )

        assert target is not None
        assert planner._compute_convergence_angle(entry, target, "left") <= (
            planner.MAX_CONVERGENCE_ANGLE
        )

    def test_extended_lateral_grid_recovers_more_lumbar_candidates(self, monkeypatch):
        ct, mask = _make_bone_cylinder()
        planner = AutoScrewPlanner(ct, mask)
        entry = np.array([50.0, 43.0, 20.0])
        body_center = np.array([30.0, 18.0, 20.0])

        def find_target(_entry, _axis, anchor, _label):
            if anchor[0] >= 45.0 and anchor[2] == pytest.approx(20.0):
                return np.array([45.0, 10.0, 20.0])
            return None

        monkeypatch.setattr(planner, "_find_target_point", find_target)
        monkeypatch.setattr(
            planner,
            "_evaluate_gertzbein_grade",
            lambda *_args: ("A", 0.0),
        )
        monkeypatch.setattr(
            planner,
            "_sample_hu_along_trajectory",
            lambda *_args, **_kwargs: (500.0, 300.0, []),
        )

        target = planner._find_best_target(
            entry,
            np.array([0.0, 1.0, 0.0]),
            body_center,
            vertebra_label=27,
            diameter=6.5,
        )

        assert target == pytest.approx((45.0, 10.0, 20.0))


# ---------------------------------------------------------------------------
# AutoScrewPlanner: HU sampling
# ---------------------------------------------------------------------------

class TestHUSampling:
    """Test HU sampling along trajectory."""

    def test_uniform_hu_sampling(self):
        """A trajectory inside uniform bone should report that HU value."""
        hu_val = 600.0
        ct, mask = _make_bone_cylinder(hu_value=hu_val)
        planner = AutoScrewPlanner(ct, mask)

        # Trajectory entirely within bone, radius small enough to stay inside.
        entry = np.array([30.0, 25.0, 20.0])
        target = np.array([30.0, 15.0, 20.0])

        mean_hu, min_hu, samples = planner._sample_hu_along_trajectory(
            entry, target, diameter=5.0,
        )

        assert mean_hu == pytest.approx(hu_val)
        assert min_hu == pytest.approx(hu_val)
        # The raw sample list is no longer returned by the shared grader.
        assert samples == []

    def test_empty_trajectory(self):
        """Zero-length trajectory should return zeros."""
        ct, mask = _make_bone_cylinder()
        planner = AutoScrewPlanner(ct, mask)

        point = np.array([30.0, 20.0, 20.0])
        mean_hu, min_hu, samples = planner._sample_hu_along_trajectory(
            point, point, diameter=5.0,
        )
        assert mean_hu == 0.0
        assert min_hu == 0.0
        assert samples == []


# ---------------------------------------------------------------------------
# AutoScrewPlanner: Gertzbein grade evaluation
# ---------------------------------------------------------------------------

class TestGertzbeinGradeEvaluation:
    """Test Gertzbein-Robbins grading logic."""

    def test_grade_a_fully_intrapedicular(self):
        """Trajectory entirely within bone should give grade A."""
        ct, mask = _make_bone_cylinder()
        planner = AutoScrewPlanner(ct, mask)

        # Small diameter screw well inside the body.
        entry = np.array([30.0, 25.0, 20.0])
        target = np.array([30.0, 15.0, 20.0])

        grade, breach = planner._evaluate_gertzbein_grade(
            entry, target, diameter=4.0, vertebra_label=27,
        )

        assert grade == "A"
        assert breach == 0.0

    def test_grade_with_partial_breach(self):
        """Trajectory near the bone edge should produce a non-A grade."""
        ct, mask = _make_bone_cylinder()
        planner = AutoScrewPlanner(ct, mask)

        # Place screw at the very edge of the mask so radial samples
        # extend outside.
        entry = np.array([39.0, 25.0, 20.0])
        target = np.array([39.0, 15.0, 20.0])

        grade, breach = planner._evaluate_gertzbein_grade(
            entry, target, diameter=6.0, vertebra_label=27,
        )

        # With centreline at x=39 and radius=3, some radial points
        # will be at x=42 which is outside the body (ends at x=40).
        # Expect breach > 0.
        assert grade in ("B", "C", "D", "E")
        assert breach > 0.0

    def test_grade_e_outside_bone(self):
        """Trajectory fully outside bone with large diameter should give grade E.

        When the centreline is outside the mask, breach distance equals
        the screw radius.  We use a large diameter (14mm, radius 7mm)
        so that the breach exceeds 6mm and triggers Grade E.
        """
        ct, mask = _make_bone_cylinder()
        planner = AutoScrewPlanner(ct, mask)

        entry = np.array([5.0, 5.0, 5.0])
        target = np.array([5.0, 5.0, 10.0])

        grade, breach = planner._evaluate_gertzbein_grade(
            entry, target, diameter=14.0, vertebra_label=27,
        )
        assert grade == "E"
        assert breach >= 6.0


# ---------------------------------------------------------------------------
# AutoScrewPlanner: confidence scoring
# ---------------------------------------------------------------------------

class TestConfidenceScoring:
    """Test _calculate_confidence with known inputs."""

    def test_grade_a_high_hu_high_margin(self):
        """Grade A, high HU, wide pedicle should yield high confidence."""
        ct, mask = _make_bone_cylinder()
        planner = AutoScrewPlanner(ct, mask)

        confidence = planner._calculate_confidence(
            grade="A", mean_hu=700.0, pedicle_width=10.0, diameter=5.0,
        )
        # Base 0.6 + HU ~0.17 + margin ~0.2 = ~0.97
        assert confidence > 0.8
        assert confidence <= 1.0

    def test_grade_e_zero_confidence(self):
        """Grade E should produce very low confidence."""
        ct, mask = _make_bone_cylinder()
        planner = AutoScrewPlanner(ct, mask)

        confidence = planner._calculate_confidence(
            grade="E", mean_hu=100.0, pedicle_width=5.0, diameter=5.0,
        )
        assert confidence < 0.1

    def test_confidence_range(self):
        """Confidence should always be in [0.0, 1.0]."""
        ct, mask = _make_bone_cylinder()
        planner = AutoScrewPlanner(ct, mask)

        for grade in ("A", "B", "C", "D", "E"):
            for hu in (0, 200, 500, 1000):
                for width, diam in ((4.0, 4.0), (8.0, 6.0), (12.0, 8.5)):
                    c = planner._calculate_confidence(grade, hu, width, diam)
                    assert 0.0 <= c <= 1.0


# ---------------------------------------------------------------------------
# AutoScrewPlanner: plan_screw
# ---------------------------------------------------------------------------

class TestPlanScrew:
    """Integration tests for single screw planning."""

    def test_plan_screw_left_succeeds(self):
        """Planning a left pedicle screw should produce valid results."""
        ct, mask = _make_bone_cylinder()
        planner = AutoScrewPlanner(ct, mask)
        analysis = _make_analysis()

        result = planner.plan_screw(analysis, "left")

        assert result is not None
        assert result.vertebra_name == "L5"
        assert result.side == "left"
        assert result.length_mm >= planner.MIN_SCREW_LENGTH
        assert result.length_mm <= planner.MAX_SCREW_LENGTH
        assert result.diameter_mm >= planner.MIN_SCREW_DIAMETER
        assert result.diameter_mm <= planner.MAX_SCREW_DIAMETER
        assert result.gertzbein_grade in ("A", "B", "C", "D", "E")
        assert 0.0 <= result.confidence <= 1.0

    def test_plan_screw_right_succeeds(self):
        """Planning a right pedicle screw should produce valid results."""
        ct, mask = _make_bone_cylinder()
        planner = AutoScrewPlanner(ct, mask)
        analysis = _make_analysis()

        result = planner.plan_screw(analysis, "right")

        assert result is not None
        assert result.side == "right"
        assert result.length_mm > 0

    def test_horizontal_upper_endplate_produces_level_sagittal_trajectory(self):
        ct, mask = _make_bone_cylinder()
        planner = AutoScrewPlanner(ct, mask)
        analysis = _make_analysis(
            upper_endplate_normal=np.array([0.0, 0.0, 1.0]),
        )

        result = planner.plan_screw(analysis, "left")

        assert result is not None
        assert result.target_lps[2] == pytest.approx(result.entry_lps[2])
        assert result.craniocaudal_angle == pytest.approx(0.0, abs=0.1)

    def test_missing_upper_endplate_uses_documented_horizontal_fallback(self):
        ct, mask = _make_bone_cylinder()
        planner = AutoScrewPlanner(ct, mask)

        result = planner.plan_screw(_make_analysis(), "left")

        assert result is not None
        assert result.craniocaudal_angle == pytest.approx(0.0, abs=0.1)
        assert any("Upper endplate unavailable" in item for item in result.warnings)

    def test_plan_screw_sacrum_returns_none(self):
        """Sacrum (label 25) should be skipped."""
        ct, mask = _make_bone_cylinder()
        planner = AutoScrewPlanner(ct, mask)
        analysis = _make_analysis(label=25, name="sacrum")

        result = planner.plan_screw(analysis, "left")
        assert result is None

    def test_plan_screw_no_success_returns_none(self):
        """Analysis with success=False should return None."""
        ct, mask = _make_bone_cylinder()
        planner = AutoScrewPlanner(ct, mask)
        analysis = _make_analysis(success=False)

        result = planner.plan_screw(analysis, "left")
        assert result is None

    def test_plan_screw_no_pedicle_data_returns_none(self):
        """Missing pedicle data for requested side should return None."""
        ct, mask = _make_bone_cylinder()
        planner = AutoScrewPlanner(ct, mask)

        vertebra = _make_vertebra()
        analysis = PedicleAnalysisResult(
            vertebra=vertebra,
            left_pedicle_center=None,
            right_pedicle_center=None,
            success=True,
            vertebral_body_center=np.array([30.0, 20.0, 20.0]),
        )

        result = planner.plan_screw(analysis, "left")
        assert result is None

    def test_failed_entry_search_does_not_fabricate_trajectory(self, monkeypatch):
        ct, mask = _make_bone_cylinder()
        planner = AutoScrewPlanner(ct, mask)
        monkeypatch.setattr(planner, "_find_entry_point", lambda *_args: None)

        result = planner.plan_screw(_make_analysis(), "left")

        assert result is None

    def test_short_bone_corridor_does_not_extend_screw_outside_mask(
        self,
        monkeypatch,
    ):
        ct, mask = _make_bone_cylinder()
        planner = AutoScrewPlanner(ct, mask)
        monkeypatch.setattr(
            planner,
            "_find_target_point",
            lambda entry, *_args: entry + np.array([0.0, -10.0, 0.0]),
        )

        result = planner.plan_screw(_make_analysis(), "left")

        assert result is None

    def test_plan_screw_narrow_pedicle_is_planned_at_the_minimum_diameter(self):
        """A narrow pedicle gets the smallest screw, not a dropped side."""
        ct, mask = _make_bone_cylinder()
        planner = AutoScrewPlanner(ct, mask)
        analysis = _make_analysis(left_width=3.0, right_width=3.0)

        result = planner.plan_screw(analysis, "left")

        assert result is not None
        assert result.diameter_mm == pytest.approx(AutoScrewPlanner.MIN_SCREW_DIAMETER)
        assert result.metrics["narrow_pedicle"] is True

    def test_plan_screw_invalid_side_raises(self):
        """Invalid side string should raise ValueError."""
        ct, mask = _make_bone_cylinder()
        planner = AutoScrewPlanner(ct, mask)
        analysis = _make_analysis()

        with pytest.raises(ValueError, match="side must be"):
            planner.plan_screw(analysis, "middle")

    def test_entry_and_target_are_distinct(self):
        """Entry and target points should not be the same."""
        ct, mask = _make_bone_cylinder()
        planner = AutoScrewPlanner(ct, mask)
        analysis = _make_analysis()

        result = planner.plan_screw(analysis, "left")
        assert result is not None
        assert not np.allclose(result.entry_lps, result.target_lps)

    def test_screw_diameter_respects_pedicle_width(self):
        """Diameter follows the fill ratio once the clearance default is 0 mm."""
        ct, mask = _make_bone_cylinder()
        planner = AutoScrewPlanner(ct, mask)
        analysis = _make_analysis(left_width=6.0)

        result = planner.plan_screw(analysis, "left")
        assert result is not None
        assert result.diameter_mm == pytest.approx(4.5)

    def test_diameter_is_reduced_until_trajectory_is_grade_a_or_b(
        self,
        monkeypatch,
    ):
        ct, mask = _make_bone_cylinder()
        planner = AutoScrewPlanner(ct, mask)
        monkeypatch.setattr(
            planner,
            "_evaluate_gertzbein_grade",
            lambda _entry, _target, diameter, _label: (
                ("C", 2.5) if diameter > 6.5 else ("A", 0.0)
            ),
        )

        result = planner.plan_screw(_make_analysis(left_width=9.5), "left")

        assert result is not None
        assert result.gertzbein_grade in {"A", "B"}
        assert result.diameter_mm == pytest.approx(6.5)
        assert any("Diameter reduced" in warning for warning in result.warnings)

    def test_no_body_center_returns_none(self):
        """Missing vertebral body centre should return None."""
        ct, mask = _make_bone_cylinder()
        planner = AutoScrewPlanner(ct, mask)

        vertebra = _make_vertebra()
        analysis = PedicleAnalysisResult(
            vertebra=vertebra,
            left_pedicle_center=np.array([40.0, 37.0, 20.0]),
            left_pedicle_axis=np.array([0.0, -1.0, 0.0]),
            left_pedicle_width=8.0,
            success=True,
            vertebral_body_center=None,
        )

        result = planner.plan_screw(analysis, "left")
        assert result is None


# ---------------------------------------------------------------------------
# AutoScrewPlanner: plan_all
# ---------------------------------------------------------------------------

class TestPlanAll:
    """Test batch planning across multiple vertebrae."""

    def test_plan_all_both_sides(self):
        """plan_all with sides='both' should plan left and right."""
        ct, mask = _make_bone_cylinder()
        planner = AutoScrewPlanner(ct, mask)
        analyses = [_make_analysis()]

        results = planner.plan_all(analyses, sides="both")

        assert len(results) == 2
        sides_returned = {r.side for r in results}
        assert sides_returned == {"left", "right"}

    def test_plan_all_left_only(self):
        """plan_all with sides='left' should only plan left."""
        ct, mask = _make_bone_cylinder()
        planner = AutoScrewPlanner(ct, mask)
        analyses = [_make_analysis()]

        results = planner.plan_all(analyses, sides="left")

        assert len(results) == 1
        assert results[0].side == "left"

    def test_plan_all_multiple_vertebrae(self):
        """Multiple vertebrae should each get planned screws."""
        ct_arr = np.full((80, 60, 60), -1000.0, dtype=np.float32)
        mask_arr = np.zeros((80, 60, 60), dtype=np.uint8)
        for label, z_start in ((27, 10), (28, 45)):
            z_slice = slice(z_start, z_start + 20)
            ct_arr[z_slice, 5:32, 20:40] = 500.0
            mask_arr[z_slice, 5:32, 20:40] = label
            ct_arr[z_slice, 30:45, 35:45] = 500.0
            mask_arr[z_slice, 30:45, 35:45] = label
            ct_arr[z_slice, 30:45, 15:25] = 500.0
            mask_arr[z_slice, 30:45, 15:25] = label
        ct, mask = _make_image(ct_arr), _make_image(mask_arr)
        planner = AutoScrewPlanner(ct, mask)

        analyses = [
            _make_analysis(label=27, name="L5"),
            _make_analysis(
                label=28,
                name="L4",
                left_center=np.array([40.0, 37.0, 55.0]),
                right_center=np.array([20.0, 37.0, 55.0]),
                body_center=np.array([30.0, 18.0, 55.0]),
            ),
        ]

        results = planner.plan_all(analyses, sides="both")

        # 2 vertebrae x 2 sides = 4 screws.
        assert len(results) == 4
        names = {r.vertebra_name for r in results}
        assert names == {"L5", "L4"}

    def test_plan_all_skips_failed_analyses(self):
        """Unsuccessful analyses should be skipped without error."""
        ct, mask = _make_bone_cylinder()
        planner = AutoScrewPlanner(ct, mask)

        analyses = [
            _make_analysis(label=27, name="L5", success=True),
            _make_analysis(label=28, name="L4", success=False),
        ]

        results = planner.plan_all(analyses, sides="left")

        assert len(results) == 1
        assert results[0].vertebra_name == "L5"

    def test_plan_all_invalid_sides_raises(self):
        """Invalid sides string should raise ValueError."""
        ct, mask = _make_bone_cylinder()
        planner = AutoScrewPlanner(ct, mask)

        with pytest.raises(ValueError, match="sides must be"):
            planner.plan_all([], sides="invalid")


class TestSkippedSides:
    """Every path reports the sides it could not place a screw on."""

    @pytest.mark.parametrize("mode", ["legacy", "optimizer"])
    def test_a_side_with_no_pedicle_data_is_reported(self, mode):
        from src.core.planner_config import PlannerConfig

        ct, mask = _make_bone_cylinder()
        planner = AutoScrewPlanner(ct, mask, config=PlannerConfig(mode=mode))
        analysis = _make_analysis()
        analysis.left_pedicle_center = None      # a one-sided coronal miss

        results = planner.plan_all([analysis])

        assert [r.side for r in results] == ["right"]
        assert [(v, s) for v, s, _r in planner.skipped_sides] == [("L5", "left")]
        assert "left" in planner.skipped_sides[0][2]

    @pytest.mark.parametrize("mode", ["legacy", "optimizer"])
    def test_a_failed_analysis_reports_both_sides(self, mode):
        from src.core.planner_config import PlannerConfig

        ct, mask = _make_bone_cylinder()
        planner = AutoScrewPlanner(ct, mask, config=PlannerConfig(mode=mode))

        planner.plan_all([_make_analysis(success=False)])

        assert [(v, s) for v, s, _r in planner.skipped_sides] == [
            ("L5", "left"), ("L5", "right"),
        ]
        assert all("analysis" in reason for _v, _s, reason in planner.skipped_sides)

    @pytest.mark.parametrize("mode", ["legacy", "optimizer"])
    def test_the_sacrum_is_excluded_not_skipped(self, mode):
        """Not planning the sacrum is the policy, not a failure to report."""
        from src.core.planner_config import PlannerConfig

        ct, mask = _make_bone_cylinder(label=25)
        planner = AutoScrewPlanner(ct, mask, config=PlannerConfig(mode=mode))

        results = planner.plan_all([_make_analysis(label=25, name="S1")])

        assert results == []
        assert planner.skipped_sides == []

    def test_a_successful_plan_reports_nothing(self):
        ct, mask = _make_bone_cylinder()
        planner = AutoScrewPlanner(ct, mask)

        results = planner.plan_all([_make_analysis()])

        assert len(results) == 2
        assert planner.skipped_sides == []


# ---------------------------------------------------------------------------
# Helper method tests
# ---------------------------------------------------------------------------

class TestHelperMethods:
    """Tests for internal utility methods."""

    @pytest.mark.parametrize(
        ("safe_length", "expected"),
        [
            (24.9, None),
            (25.0, 25.0),
            (34.9, 30.0),
            (50.0, 50.0),
            (54.9, 50.0),
            (55.0, 55.0),
            (72.0, 55.0),
        ],
    )
    def test_select_standard_length(self, safe_length, expected):
        """The longest catalogue implant fitting the safe corridor wins."""
        ct, mask = _make_bone_cylinder()
        planner = AutoScrewPlanner(ct, mask)
        result = planner._select_standard_length(safe_length)

        if expected is None:
            assert result is None
        else:
            assert result == pytest.approx(expected)

    def test_orient_axis_posterior_already_posterior(self):
        """Axis already pointing +Y should not be flipped."""
        axis = np.array([0.1, 0.9, 0.1])
        result = AutoScrewPlanner._orient_axis_posterior(axis)
        np.testing.assert_array_almost_equal(result, axis)

    def test_orient_axis_posterior_flips_anterior(self):
        """Axis pointing -Y (anterior) should be negated."""
        axis = np.array([0.1, -0.9, 0.1])
        result = AutoScrewPlanner._orient_axis_posterior(axis)
        expected = np.array([-0.1, 0.9, -0.1])
        np.testing.assert_array_almost_equal(result, expected)

    @pytest.mark.parametrize(
        ("level", "pedicle_width", "expected"),
        [
            ("S1", 9.0, 6.5),
            ("L5", 9.0, 6.5),
            ("L3", 9.0, 6.5),
            ("L2", 8.5, 6.0),
            ("L1", 8.5, 6.0),
            ("T12", 8.0, 5.5),
            ("T4", 8.0, 5.5),
        ],
    )
    def test_compute_diameter_uses_level_preference(
        self,
        level,
        pedicle_width,
        expected,
    ):
        ct, mask = _make_bone_cylinder()
        planner = AutoScrewPlanner(ct, mask)

        assert planner._compute_diameter(pedicle_width, level) == pytest.approx(
            expected
        )

    @pytest.mark.parametrize(
        ("level", "pedicle_width", "expected"),
        [
            ("L5", 9.5, 7.0),
            ("L1", 9.0, 6.5),
            ("T8", 8.5, 6.0),
        ],
    )
    def test_compute_diameter_upsizes_only_one_step_for_wide_pedicle(
        self,
        level,
        pedicle_width,
        expected,
    ):
        ct, mask = _make_bone_cylinder()
        planner = AutoScrewPlanner(ct, mask)

        assert planner._compute_diameter(pedicle_width, level) == pytest.approx(
            expected
        )

    def test_compute_diameter_normal(self):
        """Narrow pedicles should reduce below the level preference."""
        ct, mask = _make_bone_cylinder()
        planner = AutoScrewPlanner(ct, mask)

        # min(0.8 * 6.0, 6.0 - 0) = 4.8 -> largest catalogue size at or under it.
        assert planner._compute_diameter(6.0, "L4") == pytest.approx(4.5)

    def test_compute_diameter_falls_back_to_the_minimum(self):
        """A pedicle under the smallest implant still gets the smallest implant."""
        ct, mask = _make_bone_cylinder()
        planner = AutoScrewPlanner(ct, mask)

        assert planner._compute_diameter(5.0, "L4") == pytest.approx(4.0)
        assert planner._compute_diameter(3.5, "T4") == pytest.approx(4.0)
        assert planner._compute_diameter(0.0, "L4") == pytest.approx(4.0)

    def test_compute_diameter_maximum_clamp(self):
        """Very wide pedicle should clamp to MAX_SCREW_DIAMETER."""
        ct, mask = _make_bone_cylinder()
        planner = AutoScrewPlanner(ct, mask)

        assert planner._compute_diameter(15.0, "L5") == pytest.approx(7.0)

    def test_get_hu_at_lps_inside_volume(self):
        """Should return the correct HU value for a point inside."""
        hu_val = 450.0
        ct, mask = _make_bone_cylinder(hu_value=hu_val)
        planner = AutoScrewPlanner(ct, mask)

        # Centre of the body block: (30, 20, 20) in LPS.
        hu = planner._get_hu_at_lps(np.array([30.0, 20.0, 20.0]))
        assert hu is not None
        assert hu == pytest.approx(hu_val)

    def test_get_hu_at_lps_outside_volume(self):
        """Should return None for a point outside the volume."""
        ct, mask = _make_bone_cylinder()
        planner = AutoScrewPlanner(ct, mask)

        hu = planner._get_hu_at_lps(np.array([999.0, 999.0, 999.0]))
        assert hu is None

    def test_is_point_inside_mask_true(self):
        """Point inside the labelled region should return True."""
        ct, mask = _make_bone_cylinder(label=27)
        planner = AutoScrewPlanner(ct, mask)

        inside = planner._is_point_inside_mask(
            np.array([30.0, 20.0, 20.0]), vertebra_label=27,
        )
        assert inside is True

    def test_is_point_inside_mask_false_different_label(self):
        """Point inside volume but wrong label should return False."""
        ct, mask = _make_bone_cylinder(label=27)
        planner = AutoScrewPlanner(ct, mask)

        inside = planner._is_point_inside_mask(
            np.array([30.0, 20.0, 20.0]), vertebra_label=28,
        )
        assert inside is False

    def test_is_point_inside_mask_false_air(self):
        """Point in air (outside bone) should return False."""
        ct, mask = _make_bone_cylinder(label=27)
        planner = AutoScrewPlanner(ct, mask)

        # (5, 5, 5) is outside all painted regions.
        inside = planner._is_point_inside_mask(
            np.array([5.0, 5.0, 5.0]), vertebra_label=27,
        )
        assert inside is False


# ---------------------------------------------------------------------------
# Angle calculation tests
# ---------------------------------------------------------------------------

class TestAngleCalculations:
    """Test convergence and craniocaudal angle computation."""

    def test_pure_anterior_trajectory_zero_convergence(self):
        """Pure anterior trajectory (entry to target along -Y) should
        have near-zero convergence angle."""
        ct, mask = _make_bone_cylinder()
        planner = AutoScrewPlanner(ct, mask)

        entry = np.array([30.0, 40.0, 20.0])
        target = np.array([30.0, 10.0, 20.0])

        angle = planner._compute_convergence_angle(entry, target, "left")
        assert abs(angle) < 1.0  # nearly zero

    def test_convergence_with_medial_direction(self):
        """Trajectory with lateral deviation should produce non-zero angle."""
        ct, mask = _make_bone_cylinder()
        planner = AutoScrewPlanner(ct, mask)

        entry = np.array([40.0, 40.0, 20.0])
        target = np.array([30.0, 10.0, 20.0])

        angle = planner._compute_convergence_angle(entry, target, "left")
        assert angle > 5.0  # noticeable medial convergence

    def test_pure_anterior_trajectory_zero_craniocaudal(self):
        """Pure anterior trajectory should have zero craniocaudal angle."""
        ct, mask = _make_bone_cylinder()
        planner = AutoScrewPlanner(ct, mask)

        entry = np.array([30.0, 40.0, 20.0])
        target = np.array([30.0, 10.0, 20.0])

        angle = planner._compute_craniocaudal_angle(entry, target)
        assert abs(angle) < 1.0

    def test_cranially_directed_screw(self):
        """Screw directed cranially (toward +Z) should have positive angle."""
        ct, mask = _make_bone_cylinder()
        planner = AutoScrewPlanner(ct, mask)

        entry = np.array([30.0, 40.0, 15.0])
        target = np.array([30.0, 10.0, 25.0])

        angle = planner._compute_craniocaudal_angle(entry, target)
        assert angle > 0.0

    def test_craniocaudal_angle_matches_elevation_magnitude(self):
        """A purely sagittal 10 degree rise should report exactly 10 degrees."""
        ct, mask = _make_bone_cylinder()
        planner = AutoScrewPlanner(ct, mask)

        entry = np.array([0.0, 30.0, 0.0])
        target = np.array([
            0.0,
            30.0 - 40.0 * math.cos(math.radians(10.0)),
            40.0 * math.sin(math.radians(10.0)),
        ])

        angle = planner._compute_craniocaudal_angle(entry, target)
        assert angle == pytest.approx(10.0)

    def test_craniocaudal_angle_is_elevation_not_sagittal_projection(self):
        """Convergence must not inflate the reported craniocaudal angle.

        The trajectory below rises 10 degrees above the axial plane while
        converging 20 degrees medially.  Elevation stays 10 degrees; the
        sagittal projection atan2(dz, |dy|) would read ~10.6 degrees.
        """
        ct, mask = _make_bone_cylinder()
        planner = AutoScrewPlanner(ct, mask)

        horizontal = 40.0 * math.cos(math.radians(10.0))
        entry = np.array([20.0, 30.0, 0.0])
        target = np.array([
            20.0 - horizontal * math.sin(math.radians(20.0)),
            30.0 - horizontal * math.cos(math.radians(20.0)),
            40.0 * math.sin(math.radians(10.0)),
        ])

        assert planner._compute_convergence_angle(entry, target, "left") == (
            pytest.approx(20.0)
        )
        angle = planner._compute_craniocaudal_angle(entry, target)
        assert angle == pytest.approx(10.0)
        projected = math.degrees(
            math.atan2(target[2] - entry[2], abs(target[1] - entry[1]))
        )
        assert projected == pytest.approx(10.63, abs=0.02)
        assert angle < projected


# ---------------------------------------------------------------------------
# Delegation to ScrewGrader / shared geometry / sizing constants
# ---------------------------------------------------------------------------

class TestGrading:
    def _cube(self, label=28):
        arr = np.zeros((60, 60, 60), dtype=np.uint8)
        arr[20:40, 20:40, 20:40] = label
        mask = _make_image(arr)
        ct = _make_image(np.where(arr > 0, 350, -50).astype(np.int16))
        return ct, mask

    def test_screw_fully_outside_bone_is_grade_E(self):
        ct, mask = self._cube()
        planner = AutoScrewPlanner(ct, mask)
        grade, breach = planner._evaluate_gertzbein_grade(
            np.array([5.0, 5.0, 30.0]), np.array([5.0, 50.0, 30.0]), 6.5, 28)
        assert grade == "E"
        assert breach >= 6.0

    def test_contained_screw_is_grade_A(self):
        ct, mask = self._cube()
        planner = AutoScrewPlanner(ct, mask)
        grade, breach = planner._evaluate_gertzbein_grade(
            np.array([30.0, 38.0, 30.0]), np.array([30.0, 22.0, 30.0]), 6.0, 28)
        assert (grade, breach) == ("A", 0.0)


class TestSignedConvergence:
    def test_lateral_divergence_is_negative_for_left(self):
        ct, mask = TestGrading()._cube()
        planner = AutoScrewPlanner(ct, mask)
        entry = np.array([10.0, 30.0, 0.0])
        assert planner._compute_convergence_angle(entry, np.array([0.0, 0.0, 0.0]), "left") > 0
        assert planner._compute_convergence_angle(entry, np.array([20.0, 0.0, 0.0]), "left") < 0


class TestDiameterRule:
    def test_diameter_capped_at_80_percent_and_clearance(self):
        ct, mask = TestGrading()._cube()
        planner = AutoScrewPlanner(ct, mask)
        # width 7.0 -> min(0.8*7=5.6, 7-0=7.0) -> floor to the catalogue -> 5.5
        assert planner._compute_diameter(7.0, "L4") == pytest.approx(5.5)
        # width 10 -> min(8.0, 10.0) -> capped by the level preset/automatic max
        assert planner._compute_diameter(10.0, "L4") == pytest.approx(7.0)
        # too narrow for the fill ratio: the smallest implant, never nothing
        assert planner._compute_diameter(5.5, "L4") == pytest.approx(4.0)


class TestPlannerConfig:
    def test_config_changes_sizing(self):
        from src.core.planner_config import PlannerConfig

        ct, mask = TestGrading()._cube()
        strict = AutoScrewPlanner(
            ct, mask, config=PlannerConfig(pedicle_fill_ratio=0.6, wall_clearance_mm=1.5)
        )
        assert strict._compute_diameter(10.0, "L4") == pytest.approx(6.0)  # min(6.0, 7.0) -> 6.0
        short = AutoScrewPlanner(ct, mask, config=PlannerConfig(implant_lengths_mm=(30.0, 35.0)))
        assert short._select_standard_length(60.0) == 35.0
        assert short._select_standard_length(29.0) is None

    def test_diameter_step_down_walks_the_configured_catalogue(self):
        from src.core.planner_config import PlannerConfig

        ct, mask = TestGrading()._cube()
        planner = AutoScrewPlanner(
            ct, mask, config=PlannerConfig(implant_diameters_mm=(4.5, 5.5, 6.5))
        )
        assert planner._next_smaller_diameter(6.5) == pytest.approx(5.5)
        assert planner._next_smaller_diameter(5.5) == pytest.approx(4.5)
        assert planner._next_smaller_diameter(4.5) is None
        # A recommendation between catalogue entries steps to the next one below.
        assert planner._next_smaller_diameter(6.0) == pytest.approx(5.5)


class TestGridAlignment:
    """A mask that drifted off the CT grid is resampled, not rejected."""

    def _shifted(self, delta):
        ct, mask = TestGrading()._cube()
        origin = mask.GetOrigin()
        mask.SetOrigin((origin[0], origin[1], origin[2] + delta))
        return ct, mask

    def test_geometry_drift_beyond_tolerance_is_resampled(self):
        ct, mask = self._shifted(0.5)   # half a voxel: same size, different grid
        planner = AutoScrewPlanner(ct, mask)
        assert planner._mask.GetOrigin() == pytest.approx(ct.GetOrigin())
        # The label survived the nearest-neighbour resample, one voxel down.
        arr = sitk.GetArrayFromImage(planner._mask)
        assert (arr == 28).any()

    def test_float32_origin_drift_does_not_resample_or_raise(self):
        ct, mask = TestGrading()._cube()
        ct.SetOrigin((0.0, 0.0, -2300.00012))
        mask.SetOrigin((0.0, 0.0, float(np.float32(-2300.00012))))
        planner = AutoScrewPlanner(ct, mask)
        assert planner._mask is mask   # nothing to repair
        np.testing.assert_array_equal(
            sitk.GetArrayFromImage(planner._mask), sitk.GetArrayFromImage(mask)
        )


class TestScrewMetrics:
    """Every planned screw carries the clinical metric bundle."""

    def test_planned_screw_metrics_populated(self):
        from src.core.pedicle_analyzer import PedicleAnalyzer
        from tests.test_pedicle_analyzer import _make_anatomical_phantom

        mask = _make_anatomical_phantom()
        arr = sitk.GetArrayFromImage(mask)
        ct = sitk.GetImageFromArray(np.where(arr > 0, 300, -50).astype(np.int16))
        ct.CopyInformation(mask)
        analyzer = PedicleAnalyzer(mask)
        result = analyzer.analyze_pedicle(analyzer.get_available_vertebrae()[0])
        screws = AutoScrewPlanner(ct, mask).plan_all([result])
        assert screws
        keys = {
            "trajectory_mean_hu", "trajectory_min_hu", "pedicle_mean_hu", "body_mean_hu",
            "trajectory_body_ratio", "min_wall_mm", "heary_direction",
            "facet_grade", "facet_text",
        }
        assert keys <= set(screws[0].metrics)
        assert screws[0].metrics["facet_grade"] == 0
        assert screws[0].metrics["trajectory_mean_hu"] == pytest.approx(300.0, abs=50.0)
        assert screws[0].metrics["min_wall_mm"] == pytest.approx(screws[0].min_wall_mm)
        # A breach direction is reported exactly when there is a breach.
        heary = screws[0].metrics["heary_direction"]
        if screws[0].breach_mm > 0:
            assert heary in {"medial", "lateral", "anterior", "posterior",
                             "superior", "inferior"}
        else:
            assert heary == "none"


class TestOptimizerMode:
    """The optimiser-backed plan_all must never be worse than the legacy path."""

    def test_not_worse_than_legacy(self):
        from src.core.planner_config import PlannerConfig
        from tests.test_trajectory_optimizer import _setup

        ct, mask, analysis = _setup()
        legacy = AutoScrewPlanner(ct, mask, config=PlannerConfig(mode="legacy")).plan_all([analysis])
        optim = AutoScrewPlanner(ct, mask, config=PlannerConfig(mode="optimizer")).plan_all([analysis])
        order = "ABCDE"
        for side in ("left", "right"):
            before = next(s for s in legacy if s.side == side)
            after = next(s for s in optim if s.side == side)
            assert order.index(after.gertzbein_grade) <= order.index(before.gertzbein_grade)
            assert after.metrics["min_wall_mm"] >= before.metrics["min_wall_mm"] - 0.5
            assert "score" in after.metrics and "rod_misalignment_mm" in after.metrics

    def test_optimizer_screws_carry_score_components_and_clinical_metrics(self):
        from src.core.planner_config import PlannerConfig
        from tests.test_trajectory_optimizer import _setup

        ct, mask, analysis = _setup()
        screws = AutoScrewPlanner(ct, mask, config=PlannerConfig(mode="optimizer")).plan_all([analysis])
        assert screws
        for screw in screws:
            assert {"safety", "density", "length", "endplate", "centering"} <= set(
                screw.metrics["score_components"]
            )
            # The shared metadata path from plan_screw must still be applied.
            assert {"trajectory_mean_hu", "pedicle_mean_hu", "min_wall_mm",
                    "heary_direction", "facet_grade"} <= set(screw.metrics)
            assert screw.metrics["min_wall_mm"] == pytest.approx(screw.min_wall_mm)

    def test_legacy_mode_matches_plan_screw(self):
        from src.core.planner_config import PlannerConfig

        ct, mask = _make_bone_cylinder()
        planner = AutoScrewPlanner(ct, mask, config=PlannerConfig(mode="legacy"))
        analysis = _make_analysis()
        batch = planner.plan_all([analysis], sides="left")
        direct = planner.plan_screw(analysis, "left")
        assert len(batch) == 1
        assert np.allclose(batch[0].entry_lps, direct.entry_lps)
        assert np.allclose(batch[0].target_lps, direct.target_lps)
        assert batch[0].diameter_mm == direct.diameter_mm

    def test_infeasible_optimizer_falls_back_to_legacy(self, monkeypatch):
        from src.core import auto_screw_planner as module
        from src.core.planner_config import PlannerConfig

        ct, mask = _make_bone_cylinder()
        planner = AutoScrewPlanner(ct, mask, config=PlannerConfig(mode="optimizer"))
        monkeypatch.setattr(module, "optimize_screw", lambda *a, **k: [])

        results = planner.plan_all([_make_analysis()], sides="left")

        assert len(results) == 1
        assert "Optimizer found no feasible trajectory; legacy planner used" in results[0].warnings


class TestPlanAllProgressAndCancel:
    """``plan_all`` narrates each ``(level, side)`` and can stop between them."""

    def _planner(self, mode):
        from src.core.planner_config import PlannerConfig

        ct, mask = _make_bone_cylinder()
        return AutoScrewPlanner(ct, mask, config=PlannerConfig(mode=mode))

    @pytest.mark.parametrize("mode", ["legacy", "optimizer"])
    def test_progress_reports_one_message_per_level_and_side(self, mode):
        planner = self._planner(mode)
        messages = []

        planner.plan_all([_make_analysis()], progress=messages.append)

        assert messages == [
            "Planning L5 left (1/2)…",
            "Planning L5 right (2/2)…",
        ]
        assert planner.last_run_cancelled is False
        assert planner.skipped_sides == []

    @pytest.mark.parametrize("mode", ["legacy", "optimizer"])
    def test_cancel_after_the_first_side_returns_a_partial_construct(self, mode):
        planner = self._planner(mode)
        analysis = _make_analysis()
        full = planner.plan_all([analysis])
        assert len(full) == 2

        asked = []

        def cancel():
            asked.append(True)
            return len(asked) > 1

        partial = planner.plan_all([analysis], cancel=cancel)

        assert [s.side for s in partial] == ["left"]
        assert planner.last_run_cancelled is True

    @pytest.mark.parametrize("mode", ["legacy", "optimizer"])
    def test_last_run_cancelled_resets_on_the_next_run(self, mode):
        planner = self._planner(mode)
        analysis = _make_analysis()
        planner.plan_all([analysis], cancel=lambda: True)
        assert planner.last_run_cancelled is True

        planner.plan_all([analysis])

        assert planner.last_run_cancelled is False


class TestNarrowPedicle:
    """A narrow or untrustworthy width is a finding, never a reason to skip."""

    def test_a_flagged_width_no_longer_drops_the_side(self):
        from src.core.auto_screw_planner import AutoScrewPlanner
        from src.core.pedicle_analyzer import PedicleAnalyzer
        from src.core.planner_config import PlannerConfig
        from tests.test_pedicle_analyzer import _make_narrow_corridor_phantom

        mask = _make_narrow_corridor_phantom(label=29)  # L3
        arr = sitk.GetArrayFromImage(mask)
        ct = sitk.GetImageFromArray(np.where(arr > 0, 400, -50).astype(np.int16))
        ct.CopyInformation(mask)
        analyzer = PedicleAnalyzer(mask)
        analysis = analyzer.analyze_pedicle(analyzer.get_available_vertebrae()[0])
        assert analysis.width_flags["left"] == "implausible"

        planner = AutoScrewPlanner(ct, mask, config=PlannerConfig(mode="legacy"))
        planner.plan_all([analysis])

        reasons = [reason for _name, _side, reason in planner.skipped_sides]
        assert not any("too narrow" in reason for reason in reasons)
        assert not any("width uncertain" in reason for reason in reasons)

    def test_a_planned_screw_on_a_flagged_side_carries_both_warnings(self):
        from src.core.auto_screw_planner import (
            WIDTH_UNCERTAIN_SCREW_WARNING,
            AutoScrewPlanner,
        )

        ct, mask = _make_bone_cylinder()
        analysis = _make_analysis(left_width=8.0)
        analysis.width_flags["left"] = "implausible"
        planner = AutoScrewPlanner(ct, mask)

        screw = planner.plan_screw(analysis, "left")

        assert screw is not None
        assert WIDTH_UNCERTAIN_SCREW_WARNING in screw.warnings
        # An untrustworthy measurement is planned as if it were narrow.
        assert screw.diameter_mm == pytest.approx(4.0)
        assert screw.metrics["narrow_pedicle"] is True

    def test_a_narrow_side_is_planned_with_the_smallest_screw(self):
        from src.core.auto_screw_planner import AutoScrewPlanner
        from src.core.planner_config import PlannerConfig

        ct, mask = _make_bone_cylinder()
        analysis = _make_analysis(left_width=4.5, right_width=4.5)
        planner = AutoScrewPlanner(ct, mask, config=PlannerConfig(mode="legacy"))

        screws = planner.plan_all([analysis], sides="left")

        assert planner.skipped_sides == []
        assert len(screws) == 1
        screw = screws[0]
        assert screw.diameter_mm == pytest.approx(4.0)
        assert screw.metrics["narrow_pedicle"] is True
        assert screw.metrics["pedicle_width_mm"] == pytest.approx(4.5)
        assert screw.warnings[0] == (
            "Narrow pedicle (4.5 mm): 4.0 mm screw is 89 % of the width — "
            "verify the measurement or accept a lateral (in-out-in) breach"
        )

    def test_a_normal_side_is_not_marked_narrow(self):
        from src.core.auto_screw_planner import AutoScrewPlanner

        ct, mask = _make_bone_cylinder()
        planner = AutoScrewPlanner(ct, mask)

        screw = planner.plan_screw(_make_analysis(left_width=8.0), "left")

        assert screw is not None
        assert screw.metrics["narrow_pedicle"] is False
        assert screw.metrics["pedicle_width_mm"] == pytest.approx(8.0)
        assert not any(w.startswith("Narrow pedicle") for w in screw.warnings)

    def test_the_narrow_threshold_is_configurable(self):
        from src.core.auto_screw_planner import AutoScrewPlanner
        from src.core.planner_config import PlannerConfig

        ct, mask = _make_bone_cylinder()
        analysis = _make_analysis(left_width=6.0)
        relaxed = AutoScrewPlanner(ct, mask)
        strict = AutoScrewPlanner(
            ct, mask, config=PlannerConfig(narrow_pedicle_mm=7.0)
        )

        assert relaxed._is_narrow_side(analysis, "left") is False
        assert strict._is_narrow_side(analysis, "left") is True

    def test_a_graded_screw_records_the_directional_breach(self):
        from src.core.auto_screw_planner import AutoScrewPlanner

        ct, mask = _make_bone_cylinder()
        planner = AutoScrewPlanner(ct, mask)

        screw = planner.plan_screw(_make_analysis(), "left")

        assert screw is not None
        for key in (
            "medial_breach_mm", "lateral_breach_mm",
            "craniocaudal_breach_mm", "medial_wall_mm",
        ):
            assert isinstance(screw.metrics[key], float)
        assert screw.metrics["medial_breach_mm"] <= screw.breach_mm + 1e-9

    def test_no_planner_path_still_says_too_narrow(self):
        """The width-based skip is gone from the product; nothing may put it back."""
        import pathlib

        source = pathlib.Path(__file__).resolve().parent.parent / "src"
        offenders = sorted(
            str(path.relative_to(source))
            for path in source.rglob("*.py")
            if "too narrow" in path.read_text(encoding="utf-8")
        )

        assert offenders == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
