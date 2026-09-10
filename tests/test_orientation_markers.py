"""Tests for patient-orientation letters derived from reslice matrices."""

import numpy as np
import vtk

from src.core.mpr_geometry import build_screw_mpr_axes, orientation_letters
from src.utils.vtk_helpers import create_reslice_axes


def _matrix_from_columns(x_axis, y_axis, normal):
    """Build a bare reslice matrix from three column vectors."""
    axes = vtk.vtkMatrix4x4()
    axes.Identity()
    for column, vector in enumerate((x_axis, y_axis, normal)):
        for row in range(3):
            axes.SetElement(row, column, float(vector[row]))
    return axes


def test_standard_axial_reads_anterior_top_and_patient_left_right():
    letters = orientation_letters(create_reslice_axes("axial", (0.0, 0.0, 0.0)))

    assert letters == {"top": "A", "bottom": "P", "left": "R", "right": "L"}


def test_standard_coronal_reads_superior_top_and_patient_left_right():
    letters = orientation_letters(create_reslice_axes("coronal", (1.0, 2.0, 3.0)))

    assert letters == {"top": "S", "bottom": "I", "left": "R", "right": "L"}


def test_standard_sagittal_reads_superior_top_and_anterior_left():
    letters = orientation_letters(create_reslice_axes("sagittal", (1.0, 2.0, 3.0)))

    assert letters == {"top": "S", "bottom": "I", "left": "A", "right": "P"}


def test_thirty_degree_in_plane_oblique_keeps_plain_letters():
    angle = np.radians(30.0)
    axes = _matrix_from_columns(
        (np.cos(angle), np.sin(angle), 0.0),
        (np.sin(angle), -np.cos(angle), 0.0),
        (0.0, 0.0, 1.0),
    )

    letters = orientation_letters(axes)

    assert letters == {"top": "A", "bottom": "P", "left": "R", "right": "L"}


def test_screw_aligned_cross_section_flags_the_ambiguous_axis():
    # Screw direction (3, 4, 7): the cross-section's screen-up column is
    # (-0.488, -0.651, 0.581) -- no component reaches 0.7 -- while its
    # screen-right column is (-0.8, 0.6, 0) and stays unambiguous.
    axes = build_screw_mpr_axes((0.0, 0.0, 0.0), (30.0, 40.0, 70.0)).cross_section

    letters = orientation_letters(axes)

    assert letters == {"top": "A'", "bottom": "P'", "left": "L", "right": "R"}


def test_degenerate_column_is_reported_as_unknown():
    axes = _matrix_from_columns((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), (1.0, 0.0, 0.0))

    letters = orientation_letters(axes)

    assert letters["right"] == "?"
    assert letters["left"] == "?"
    assert letters["top"] == "S"


def test_columns_are_normalised_before_comparison():
    axes = _matrix_from_columns((5.0, 0.0, 0.0), (0.0, -5.0, 0.0), (0.0, 0.0, 5.0))

    assert orientation_letters(axes) == {
        "top": "A",
        "bottom": "P",
        "left": "R",
        "right": "L",
    }


def test_helper_needs_no_renderer(monkeypatch):
    """The helper must not touch Qt or a render window."""
    monkeypatch.delattr(vtk, "vtkRenderWindow", raising=False)

    assert orientation_letters(create_reslice_axes("axial", (0.0, 0.0, 0.0)))
