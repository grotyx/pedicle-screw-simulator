"""Tests for standard and screw-aligned MPR geometry."""

import numpy as np
import pytest

from src.core.mpr_geometry import (
    build_screw_mpr_axes,
    project_screw_to_slice,
    slice_to_world,
    world_to_slice,
)
from src.utils.vtk_helpers import create_reslice_axes


def _column(matrix, index):
    return np.array(
        [matrix.GetElement(row, index) for row in range(3)],
        dtype=float,
    )


def _normalized(vector):
    vector = np.asarray(vector, dtype=float)
    return vector / np.linalg.norm(vector)


@pytest.mark.parametrize("plane", ["axial", "sagittal", "coronal"])
def test_slice_world_round_trip_with_nonzero_center(plane):
    axes = create_reslice_axes(plane, (12.5, -30.0, 84.0))
    local_point = (14.0, -8.5, 0.0)

    world_point = slice_to_world(local_point, axes)
    round_trip = world_to_slice(world_point, axes)

    assert round_trip == pytest.approx(local_point)


def test_axial_slice_point_maps_to_dicom_world_center():
    axes = create_reslice_axes("axial", (-7.0, 223.0, 503.0))

    world_point = slice_to_world((10.0, -20.0, 0.0), axes)

    assert world_point == pytest.approx((3.0, 243.0, 503.0))


def test_axial_point_10mm_anterior_of_centre_maps_to_positive_local_v():
    axes = create_reslice_axes("axial", (-7.0, 223.0, 503.0))

    anterior = world_to_slice((-7.0, 213.0, 503.0), axes)
    posterior = world_to_slice((-7.0, 233.0, 503.0), axes)

    assert anterior[1] == pytest.approx(10.0)
    assert posterior[1] == pytest.approx(-10.0)


def test_screw_projection_reports_centerline_intersection():
    axes = create_reslice_axes("axial", (0.0, 0.0, 10.0))

    projection = project_screw_to_slice(
        entry=(0.0, 0.0, 0.0),
        target=(20.0, 0.0, 20.0),
        axes=axes,
        diameter=6.0,
    )

    assert projection.intersects is True
    assert projection.entry_slice == pytest.approx((0.0, 0.0, -10.0))
    assert projection.target_slice == pytest.approx((20.0, 0.0, 10.0))
    assert projection.intersection_slice == pytest.approx((10.0, 0.0, 0.0))
    assert projection.distance_mm == pytest.approx(0.0)


def test_off_slice_screw_keeps_projected_trajectory():
    axes = create_reslice_axes("axial", (0.0, 0.0, 100.0))

    projection = project_screw_to_slice(
        entry=(0.0, 0.0, 0.0),
        target=(20.0, 0.0, 20.0),
        axes=axes,
        diameter=6.0,
    )

    assert projection.intersects is False
    assert projection.intersection_slice is None
    assert projection.entry_slice[:2] == pytest.approx((0.0, 0.0))
    assert projection.target_slice[:2] == pytest.approx((20.0, 0.0))
    assert projection.distance_mm == pytest.approx(80.0)


def test_parallel_screw_within_radius_intersects_slice():
    axes = create_reslice_axes("axial", (0.0, 0.0, 10.0))

    projection = project_screw_to_slice(
        entry=(0.0, 0.0, 12.0),
        target=(20.0, 0.0, 12.0),
        axes=axes,
        diameter=6.0,
    )

    assert projection.intersects is True
    assert projection.intersection_slice == pytest.approx((10.0, 0.0, 0.0))
    assert projection.distance_mm == pytest.approx(2.0)


@pytest.mark.parametrize(
    "target",
    [
        (0.0, 0.0, 40.0),
        (40.0, 0.0, 0.0),
        (0.0, 40.0, 0.0),
        (15.0, 25.0, 35.0),
    ],
)
def test_screw_mpr_axes_are_orthonormal(target):
    result = build_screw_mpr_axes(
        entry=(0.0, 0.0, 0.0),
        target=target,
        position_fraction=0.25,
    )

    for axes in (result.long_axis_1, result.long_axis_2, result.cross_section):
        rotation = np.array(
            [[axes.GetElement(row, column) for column in range(3)] for row in range(3)]
        )
        assert rotation.T @ rotation == pytest.approx(np.eye(3), abs=1e-7)
        assert np.linalg.det(rotation) == pytest.approx(1.0, abs=1e-7)

    assert result.cross_section_center == pytest.approx(
        tuple(0.25 * np.asarray(target))
    )
    assert result.distance_from_entry_mm == pytest.approx(
        0.25 * np.linalg.norm(target)
    )


def test_screw_mpr_axes_reject_zero_length_screw():
    with pytest.raises(ValueError, match="zero-length"):
        build_screw_mpr_axes(
            entry=(1.0, 2.0, 3.0),
            target=(1.0, 2.0, 3.0),
            position_fraction=0.5,
        )


def test_screw_review_planes_have_expected_axis_relationships():
    result = build_screw_mpr_axes(
        entry=(5.0, 2.0, 3.0),
        target=(35.0, -28.0, 18.0),
        position_fraction=0.5,
    )
    screw_axis = _normalized((30.0, -30.0, 15.0))

    assert abs(np.dot(_column(result.oblique_axial, 2), screw_axis)) < 1e-7
    assert abs(np.dot(_column(result.oblique_sagittal, 2), screw_axis)) < 1e-7
    assert abs(np.dot(_column(result.cross_section, 2), screw_axis)) == (
        pytest.approx(1.0, abs=1e-7)
    )


def test_only_screw_mpr_sagittal_is_rotated_counterclockwise_90_degrees():
    result = build_screw_mpr_axes(
        entry=(0.0, 0.0, 0.0),
        target=(0.0, 40.0, 0.0),
        position_fraction=0.5,
    )

    assert _column(result.oblique_sagittal, 0) == pytest.approx(
        (0.0, -1.0, 0.0)
    )
    assert _column(result.oblique_sagittal, 1) == pytest.approx(
        (0.0, 0.0, 1.0)
    )
    assert _column(result.oblique_sagittal, 2) == pytest.approx(
        (-1.0, 0.0, 0.0)
    )


def test_semantic_planes_keep_long_axis_aliases():
    result = build_screw_mpr_axes(
        entry=(0.0, 0.0, 0.0),
        target=(30.0, 20.0, 10.0),
        position_fraction=0.5,
    )

    assert result.long_axis_1 is result.oblique_axial
    assert result.long_axis_2 is result.oblique_sagittal


# Golden matrices captured from the pre-W5 implementation for
# entry=(5.0, 2.0, 3.0), target=(35.0, -28.0, 18.0), position_fraction=0.25.
# Exact float literals: the defaults must stay bit-for-bit identical.
_GOLDEN_DEFAULT_AXES = {
    "oblique_axial": [
        [0.7071067811865476, 0.6666666666666666, 0.23570226039551584, 20.0],
        [0.7071067811865476, -0.6666666666666666, -0.23570226039551584, -13.0],
        [0.0, 0.3333333333333333, -0.9428090415820634, 10.5],
        [0.0, 0.0, 0.0, 1.0],
    ],
    "oblique_sagittal": [
        [-0.6666666666666666, -0.23570226039551584, 0.7071067811865476, 20.0],
        [0.6666666666666666, 0.23570226039551584, 0.7071067811865476, -13.0],
        [-0.3333333333333333, 0.9428090415820634, 0.0, 10.5],
        [0.0, 0.0, 0.0, 1.0],
    ],
    "cross_section": [
        [0.7071067811865476, -0.23570226039551584, 0.6666666666666666, 12.5],
        [0.7071067811865476, 0.23570226039551584, -0.6666666666666666, -5.5],
        [0.0, 0.9428090415820634, 0.3333333333333333, 6.75],
        [0.0, 0.0, 0.0, 1.0],
    ],
}

_GOLDEN_ENTRY = (5.0, 2.0, 3.0)
_GOLDEN_TARGET = (35.0, -28.0, 18.0)


def _matrix_rows(matrix):
    return [
        [matrix.GetElement(row, column) for column in range(4)]
        for row in range(4)
    ]


def test_default_arguments_reproduce_previous_matrices_bit_for_bit():
    result = build_screw_mpr_axes(
        entry=_GOLDEN_ENTRY,
        target=_GOLDEN_TARGET,
        position_fraction=0.25,
    )

    for name, expected in _GOLDEN_DEFAULT_AXES.items():
        assert _matrix_rows(getattr(result, name)) == expected, name
    assert result.cross_section_center == (12.5, -5.5, 6.75)
    assert result.distance_from_entry_mm == 11.25
    assert result.rotation_deg == 0.0
    assert result.long_axis_offset_mm == (0.0, 0.0)
    assert result.cross_section_offset_mm == (0.0, 0.0)


def test_rotation_and_offsets_are_keyword_only():
    with pytest.raises(TypeError):
        build_screw_mpr_axes((0.0, 0.0, 0.0), (0.0, 0.0, 40.0), 0.5, 15.0)


def test_ninety_degree_rotation_follows_the_right_hand_rule_about_the_screw():
    # Screw along +X: transverse = +Y, superior = +Z, normal (screw) = +X.
    # A +90 deg right-hand rotation about +X sends +Y -> +Z and +Z -> -Y.
    rotated = build_screw_mpr_axes(
        entry=(0.0, 0.0, 0.0),
        target=(40.0, 0.0, 0.0),
        rotation_deg=90.0,
    )

    assert _column(rotated.cross_section, 0) == pytest.approx(
        (0.0, 0.0, 1.0), abs=1e-9
    )
    assert _column(rotated.cross_section, 1) == pytest.approx(
        (0.0, -1.0, 0.0), abs=1e-9
    )
    assert _column(rotated.cross_section, 2) == pytest.approx(
        (1.0, 0.0, 0.0), abs=1e-9
    )
    assert rotated.rotation_deg == 90.0


def test_rotation_maps_oblique_axial_x_onto_previous_oblique_sagittal_y():
    base = build_screw_mpr_axes(entry=_GOLDEN_ENTRY, target=_GOLDEN_TARGET)
    rotated = build_screw_mpr_axes(
        entry=_GOLDEN_ENTRY,
        target=_GOLDEN_TARGET,
        rotation_deg=90.0,
    )

    assert _column(rotated.oblique_axial, 0) == pytest.approx(
        _column(base.oblique_sagittal, 1), abs=1e-9
    )
    # The screw axis itself is invariant under the rotation.
    assert _column(rotated.cross_section, 2) == pytest.approx(
        _column(base.cross_section, 2), abs=1e-12
    )


def test_long_axis_offsets_move_each_plane_origin_by_the_requested_mm():
    base = build_screw_mpr_axes(entry=_GOLDEN_ENTRY, target=_GOLDEN_TARGET)
    shifted = build_screw_mpr_axes(
        entry=_GOLDEN_ENTRY,
        target=_GOLDEN_TARGET,
        long_axis_offset_mm=(3.0, -2.0),
    )
    superior = _column(base.oblique_sagittal, 1)
    transverse = _column(base.oblique_axial, 0)

    assert _column(shifted.oblique_axial, 3) == pytest.approx(
        _column(base.oblique_axial, 3) + 3.0 * superior
    )
    assert _column(shifted.oblique_sagittal, 3) == pytest.approx(
        _column(base.oblique_sagittal, 3) - 2.0 * transverse
    )
    assert _column(shifted.cross_section, 3) == pytest.approx(
        _column(base.cross_section, 3)
    )
    assert np.linalg.norm(
        _column(shifted.oblique_axial, 3) - _column(base.oblique_axial, 3)
    ) == pytest.approx(3.0)
    assert np.linalg.norm(
        _column(shifted.oblique_sagittal, 3)
        - _column(base.oblique_sagittal, 3)
    ) == pytest.approx(2.0)
    assert shifted.long_axis_offset_mm == (3.0, -2.0)


def test_cross_section_offset_shifts_in_plane_only():
    base = build_screw_mpr_axes(
        entry=_GOLDEN_ENTRY, target=_GOLDEN_TARGET, position_fraction=0.25
    )
    shifted = build_screw_mpr_axes(
        entry=_GOLDEN_ENTRY,
        target=_GOLDEN_TARGET,
        position_fraction=0.25,
        cross_section_offset_mm=(4.0, 5.0),
    )
    transverse = _column(base.cross_section, 0)
    superior = _column(base.cross_section, 1)
    screw_axis = _normalized((30.0, -30.0, 15.0))

    expected = (
        np.asarray(base.cross_section_center)
        + 4.0 * transverse
        + 5.0 * superior
    )
    assert np.asarray(shifted.cross_section_center) == pytest.approx(expected)
    assert shifted.distance_from_entry_mm == pytest.approx(
        base.distance_from_entry_mm
    )
    delta = np.asarray(shifted.cross_section_center) - np.asarray(
        base.cross_section_center
    )
    assert float(np.dot(delta, screw_axis)) == pytest.approx(0.0, abs=1e-9)
    assert shifted.cross_section_offset_mm == (4.0, 5.0)


def test_rotated_and_offset_axes_stay_orthonormal_and_right_handed():
    result = build_screw_mpr_axes(
        entry=_GOLDEN_ENTRY,
        target=_GOLDEN_TARGET,
        position_fraction=0.4,
        rotation_deg=37.5,
        long_axis_offset_mm=(2.0, -3.0),
        cross_section_offset_mm=(1.0, -1.5),
    )

    for axes in (
        result.oblique_axial,
        result.oblique_sagittal,
        result.cross_section,
    ):
        rotation = np.array(
            [
                [axes.GetElement(row, column) for column in range(3)]
                for row in range(3)
            ]
        )
        assert rotation.T @ rotation == pytest.approx(np.eye(3), abs=1e-9)
        assert np.linalg.det(rotation) == pytest.approx(1.0, abs=1e-9)
