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
