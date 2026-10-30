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


class TestTargetFromAngles:
    @pytest.mark.parametrize("side", ["left", "right"])
    @pytest.mark.parametrize("medial, cranial", [(15.0, 0.0), (-8.0, 12.0), (30.0, -20.0)])
    def test_round_trips_through_the_shared_angle_helpers(self, side, medial, cranial):
        from src.core.screw_geometry import target_from_angles

        entry = (5.0, -2.0, 30.0)
        target = target_from_angles(entry, 45.0, medial, cranial, side)

        assert math.dist(entry, target) == pytest.approx(45.0)
        assert convergence_angle_deg(entry, target, side) == pytest.approx(medial)
        assert craniocaudal_angle_deg(entry, target) == pytest.approx(cranial)

    def test_direction_depends_only_on_the_stated_side_not_on_x(self):
        from src.core.screw_geometry import target_from_angles

        # Both entries sit far from x = 0 on the same sign; only the side decides.
        for entry in ((180.0, 10.0, 0.0), (220.0, 10.0, 0.0)):
            left = target_from_angles(entry, 40.0, 20.0, 0.0, "left")
            right = target_from_angles(entry, 40.0, 20.0, 0.0, "right")
            assert left[0] < entry[0] < right[0]

    @pytest.mark.parametrize(
        "args",
        [
            ((0, 0, 0), 40.0, 10.0, 0.0, ""),
            ((0, 0, 0), 40.0, 10.0, 0.0, "up"),
            ((0, 0, 0), 0.0, 10.0, 0.0, "left"),
            ((0, 0, 0), 40.0, 90.0, 0.0, "left"),
            ((0, 0, 0), 40.0, 0.0, -90.0, "left"),
            ((0, 0, math.nan), 40.0, 0.0, 0.0, "left"),
            ((0, 0, 0), 40.0, math.inf, 0.0, "left"),
        ],
    )
    def test_rejects_invalid_input(self, args):
        from src.core.screw_geometry import target_from_angles

        with pytest.raises(ValueError):
            target_from_angles(*args)


class TestGuideHandle:
    def test_handle_sits_behind_the_entry_on_the_axis(self):
        from src.core.screw_geometry import guide_handle_point

        handle = guide_handle_point((0.0, 0.0, 0.0), (0.0, -40.0, 0.0), 60.0)

        assert handle == pytest.approx((0.0, 60.0, 0.0))

    def test_drag_rotates_about_the_entry_and_keeps_the_length(self):
        from src.core.screw_geometry import guide_handle_point, target_from_guide_drag

        entry, target = (10.0, 0.0, 5.0), (10.0, -45.0, 5.0)
        handle = guide_handle_point(entry, target, 60.0)
        picked = (handle[0] + 3.0, handle[1], handle[2])  # surface, not centre

        # Moving the handle toward +X (patient left) swings the tip toward -X.
        moved = (picked[0] + 20.0, picked[1], picked[2] + 10.0)
        new_target = target_from_guide_drag(entry, target, handle, picked, moved)

        assert math.dist(entry, new_target) == pytest.approx(45.0)
        assert new_target[0] < entry[0]
        assert new_target[2] < entry[2]

    def test_no_jump_when_the_pick_is_off_centre(self):
        from src.core.screw_geometry import guide_handle_point, target_from_guide_drag

        entry, target = (0.0, 0.0, 0.0), (0.0, -40.0, 0.0)
        handle = guide_handle_point(entry, target, 60.0)
        picked = (handle[0] + 4.0, handle[1] - 1.0, handle[2] + 2.0)

        assert target_from_guide_drag(entry, target, handle, picked, picked) == pytest.approx(target)

    def test_degenerate_lever_is_rejected(self):
        from src.core.screw_geometry import guide_handle_point, target_from_guide_drag

        entry, target = (0.0, 0.0, 0.0), (0.0, -40.0, 0.0)
        handle = guide_handle_point(entry, target, 60.0)
        # Dragging the handle onto the entry leaves no direction.
        with pytest.raises(ValueError):
            target_from_guide_drag(entry, target, handle, handle, entry)
