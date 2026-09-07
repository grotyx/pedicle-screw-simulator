import math

import pytest

from src.core.screw_geometry import (
    convergence_angle_deg,
    craniocaudal_angle_deg,
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
