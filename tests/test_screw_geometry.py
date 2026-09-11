import math

import pytest

from src.core.screw_geometry import (
    convergence_angle_deg,
    craniocaudal_angle_deg,
    endplate_angle_deg,
    endplate_slope_deg,
    unit_trajectory,
)


def _tip(entry, medial_deg, cranial_deg, length=40.0, side="left"):
    """Build a target for a left/right screw with known angles (LPS)."""
    med = math.radians(medial_deg)
    cra = math.radians(cranial_deg)
    lateral_sign = 1.0 if side == "left" else -1.0
    dx = -lateral_sign * math.sin(med) * math.cos(cra)  # medial = toward midline
    dy = -math.cos(med) * math.cos(cra)                 # anterior = -Y
    dz = math.sin(cra)
    return (entry[0] + length * dx, entry[1] + length * dy, entry[2] + length * dz)


class TestUnitTrajectory:
    def test_unit_vector(self):
        assert unit_trajectory((0, 0, 0), (0, -3, 4)) == pytest.approx((0.0, -0.6, 0.8))

    def test_degenerate_returns_zero(self):
        assert unit_trajectory((1, 1, 1), (1, 1, 1)) == (0.0, 0.0, 0.0)


class TestConvergence:
    def test_left_medial_12_degrees(self):
        entry = (20.0, 30.0, 0.0)
        assert convergence_angle_deg(entry, _tip(entry, 12.0, 0.0, side="left"), "left") == pytest.approx(12.0, abs=1e-6)

    def test_right_medial_12_degrees(self):
        entry = (-20.0, 30.0, 0.0)
        assert convergence_angle_deg(entry, _tip(entry, 12.0, 0.0, side="right"), "right") == pytest.approx(12.0, abs=1e-6)

    def test_left_lateral_divergence_is_negative(self):
        entry = (20.0, 30.0, 0.0)
        assert convergence_angle_deg(entry, _tip(entry, -12.0, 0.0, side="left"), "left") == pytest.approx(-12.0, abs=1e-6)

    def test_unknown_side_returns_magnitude(self):
        entry = (20.0, 30.0, 0.0)
        assert convergence_angle_deg(entry, _tip(entry, -12.0, 0.0, side="left"), None) == pytest.approx(12.0, abs=1e-6)

    def test_pure_anterior_is_zero(self):
        assert convergence_angle_deg((0, 30, 0), (0, -10, 0), "left") == pytest.approx(0.0)

    def test_degenerate_is_zero(self):
        assert convergence_angle_deg((0, 0, 0), (0, 0, 5), "left") == 0.0


class TestCraniocaudal:
    def test_horizontal_is_zero(self):
        assert craniocaudal_angle_deg((0, 30, 0), (0, -10, 0)) == pytest.approx(0.0)

    def test_cranial_10_degrees(self):
        entry = (20.0, 30.0, 0.0)
        assert craniocaudal_angle_deg(entry, _tip(entry, 0.0, 10.0)) == pytest.approx(10.0, abs=1e-6)

    def test_caudal_is_negative(self):
        entry = (20.0, 30.0, 0.0)
        assert craniocaudal_angle_deg(entry, _tip(entry, 0.0, -7.0)) == pytest.approx(-7.0, abs=1e-6)


def _endplate_normal(tilt_deg):
    """LPS +Z-oriented normal of an endplate that rises `tilt_deg` anteriorly.

    ``endplate_slope_deg`` is the craniocaudal angle of the endplate-parallel
    direction travelled anteriorly, so a normal of
    ``(0, sin t, cos t)`` is exactly a ``+t`` degree endplate.
    """
    return (0.0, math.sin(math.radians(tilt_deg)), math.cos(math.radians(tilt_deg)))


class TestEndplateAngle:
    def test_slope_of_a_horizontal_endplate_is_zero(self):
        assert endplate_slope_deg((0.0, 0.0, 1.0)) == pytest.approx(0.0)

    def test_slope_is_positive_for_an_anteriorly_rising_endplate(self):
        assert endplate_slope_deg(_endplate_normal(10.0)) == pytest.approx(10.0)
        assert endplate_slope_deg(_endplate_normal(-7.5)) == pytest.approx(-7.5)

    def test_screw_along_a_ten_degree_endplate_reads_zero(self):
        entry = (30.0, 40.0, 20.0)
        target = _tip(entry, medial_deg=0.0, cranial_deg=10.0)
        assert endplate_angle_deg(entry, target, _endplate_normal(10.0)) == pytest.approx(0.0, abs=1e-6)

    def test_horizontal_screw_under_a_ten_degree_endplate_reads_minus_ten(self):
        entry = (30.0, 40.0, 20.0)
        target = _tip(entry, medial_deg=0.0, cranial_deg=0.0)
        assert endplate_angle_deg(entry, target, _endplate_normal(10.0)) == pytest.approx(-10.0, abs=1e-6)

    def test_convergence_does_not_change_the_reading(self):
        """_tip keeps dz/hypot(dx, dy) = tan(cranial), so a medialised screw
        parallel to the endplate still reads zero."""
        entry = (30.0, 40.0, 20.0)
        target = _tip(entry, medial_deg=25.0, cranial_deg=10.0)
        assert endplate_angle_deg(entry, target, _endplate_normal(10.0)) == pytest.approx(0.0, abs=1e-6)

    def test_missing_or_degenerate_normal_is_not_measurable(self):
        entry = (30.0, 40.0, 20.0)
        target = _tip(entry, medial_deg=0.0, cranial_deg=5.0)
        assert endplate_slope_deg(None) is None
        assert endplate_slope_deg((0.0, 1.0, 0.0)) is None       # n_z = 0: not an endplate
        assert endplate_slope_deg((0.0, 1.0)) is None            # wrong shape
        assert endplate_angle_deg(entry, target, None) is None
