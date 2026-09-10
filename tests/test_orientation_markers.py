"""Tests for patient-orientation letters derived from reslice matrices."""

import inspect
from types import SimpleNamespace

import numpy as np
import pytest
import vtk

from src.core.mpr_geometry import build_screw_mpr_axes, orientation_letters
from src.ui.mpr_viewer import (
    ORIENTATION_MARKER_FONT_SIZE,
    ORIENTATION_MARKER_MARGIN_PX,
    MPRViewer,
)
from src.ui.styles import DEFAULT_THEME, THEMES, theme_rgb_float
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


def _marker_viewer(axes, width=400, height=300):
    """A bare MPRViewer with only what the orientation markers touch.

    A real MPRViewer builds a VTK render window, which crashes the
    interpreter under offscreen Qt, so the tests use the same __new__
    pattern as tests/test_mpr_viewer_projection.py.
    """
    viewer = MPRViewer.__new__(MPRViewer)
    viewer.plane = "axial"
    viewer._renderer = vtk.vtkRenderer()
    viewer._reslice = (
        None if axes is None else SimpleNamespace(GetResliceAxes=lambda: axes)
    )
    viewer._custom_reslice_axes = None
    viewer._orientation_actors = {}
    viewer.vtk_widget = SimpleNamespace(width=lambda: width, height=lambda: height)
    viewer._request_render = lambda: None
    viewer._create_orientation_markers()
    return viewer


def _marker_texts(viewer):
    return {
        side: (actor.GetInput() or "")
        for side, actor in viewer._orientation_actors.items()
    }


def test_axial_viewer_labels_anterior_top_and_patient_left_right():
    viewer = _marker_viewer(create_reslice_axes("axial", (0.0, 0.0, 0.0)))

    viewer.refresh_orientation_markers()

    assert _marker_texts(viewer) == {
        "top": "A",
        "bottom": "P",
        "left": "R",
        "right": "L",
    }
    assert all(actor.GetVisibility() for actor in viewer._orientation_actors.values())


def test_markers_are_hidden_until_a_volume_is_loaded():
    viewer = _marker_viewer(None)

    viewer.refresh_orientation_markers()

    assert _marker_texts(viewer) == {
        "top": "",
        "bottom": "",
        "left": "",
        "right": "",
    }
    assert not any(
        actor.GetVisibility() for actor in viewer._orientation_actors.values()
    )


def test_custom_screw_axes_drive_the_letters():
    viewer = _marker_viewer(create_reslice_axes("axial", (0.0, 0.0, 0.0)))
    viewer._custom_reslice_axes = build_screw_mpr_axes(
        (0.0, 0.0, 0.0), (30.0, 40.0, 70.0)
    ).cross_section

    viewer.refresh_orientation_markers()

    assert _marker_texts(viewer) == {
        "top": "A'",
        "bottom": "P'",
        "left": "L",
        "right": "R",
    }


def test_markers_are_added_to_the_renderer_with_the_spec_font():
    viewer = _marker_viewer(create_reslice_axes("axial", (0.0, 0.0, 0.0)))

    props = viewer._renderer.GetViewProps()
    names = {
        props.GetItemAsObject(index).GetObjectName()
        for index in range(props.GetNumberOfItems())
    }
    assert names == {
        "orientation-top",
        "orientation-bottom",
        "orientation-left",
        "orientation-right",
    }
    for actor in viewer._orientation_actors.values():
        assert actor.GetTextProperty().GetFontSize() == ORIENTATION_MARKER_FONT_SIZE


def test_markers_sit_one_margin_inside_the_viewport_edges():
    viewer = _marker_viewer(create_reslice_axes("axial", (0.0, 0.0, 0.0)))

    positions = viewer._orientation_marker_positions(400, 300)

    margin = ORIENTATION_MARKER_MARGIN_PX
    assert positions == {
        "top": (200, 300 - margin),
        "bottom": (200, margin),
        "left": (margin, 150),
        "right": (400 - margin, 150),
    }


def test_marker_colour_follows_the_application_theme(qapp):
    # `qapp` is pytest-qt's application fixture, already used by seven other
    # test modules in this suite; the property is restored so the theme this
    # test picks cannot leak into tests that run after it.
    previous = qapp.property("themeName")
    try:
        qapp.setProperty("themeName", "graphite_mint")
        viewer = _marker_viewer(create_reslice_axes("axial", (0.0, 0.0, 0.0)))

        viewer.refresh_orientation_markers()

        expected = theme_rgb_float("graphite_mint", "viewer_foreground")
        color = viewer._orientation_actors["top"].GetTextProperty().GetColor()
        assert color == pytest.approx(expected)
    finally:
        qapp.setProperty("themeName", previous)


def test_theme_rgb_float_converts_hex_to_vtk_floats():
    assert theme_rgb_float("graphite_blue", "viewer_foreground") == pytest.approx(
        (232 / 255.0, 237 / 255.0, 242 / 255.0)
    )
    assert theme_rgb_float("no_such_theme", "viewer_foreground") == pytest.approx(
        theme_rgb_float(DEFAULT_THEME, "viewer_foreground")
    )


def test_every_palette_defines_a_viewer_foreground():
    assert all("viewer_foreground" in palette for palette in THEMES.values())


def test_refresh_follows_the_current_viewport_size():
    viewer = _marker_viewer(create_reslice_axes("axial", (0.0, 0.0, 0.0)))

    viewer.refresh_orientation_markers()
    assert viewer._orientation_actors["right"].GetPosition() == pytest.approx(
        (394.0, 150.0)
    )

    viewer.vtk_widget = SimpleNamespace(width=lambda: 800, height=lambda: 600)
    viewer.refresh_orientation_markers()

    assert viewer._orientation_actors["right"].GetPosition() == pytest.approx(
        (794.0, 300.0)
    )
    assert viewer._orientation_actors["top"].GetPosition() == pytest.approx(
        (400.0, 594.0)
    )


def test_viewer_resize_hook_is_wired_to_the_marker_refresh():
    # resizeEvent itself is a two-line Qt delegation; a real MPRViewer
    # cannot be built in-process (its VTK render window crashes offscreen
    # Qt), so pin the wiring rather than the Qt event delivery.
    assert "resizeEvent" in MPRViewer.__dict__
    source = inspect.getsource(MPRViewer.resizeEvent)
    assert "refresh_orientation_markers" in source


def test_refresh_is_a_noop_before_the_actors_exist():
    viewer = MPRViewer.__new__(MPRViewer)

    viewer.refresh_orientation_markers()   # must not raise
