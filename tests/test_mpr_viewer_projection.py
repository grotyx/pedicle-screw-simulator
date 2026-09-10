"""Regression tests for MPR world/display conversion and screw overlays."""

from types import SimpleNamespace

import pytest
import vtk

from src.core.mpr_geometry import build_screw_mpr_axes
from src.ui.click_detector import DoubleClickDetector
from src.ui.mpr_viewer import MPRViewer
from src.utils.vtk_helpers import create_reslice_axes


class _VolumeManager:
    def __init__(self, position):
        self.position = position
        self.spacing = (1.0, 1.0, 1.0)

    def get_slice_position(self, _plane):
        return self.position


def _make_viewer(center, position):
    axes = create_reslice_axes("axial", center)
    viewer = MPRViewer.__new__(MPRViewer)
    viewer.plane = "axial"
    viewer._renderer = vtk.vtkRenderer()
    viewer._reslice = SimpleNamespace(GetResliceAxes=lambda: axes)
    viewer._custom_reslice_axes = None
    viewer._screw_data = {}
    viewer._screw_overlays = {}
    viewer._review_screw_id = None
    viewer._screw_prop_parts = {}
    viewer._screw_drag_active = False
    viewer._screw_drag_callback = None
    viewer._screw_drag_begin_callback = None
    viewer._screw_drag_end_callback = None
    viewer._screw_select_callback = None
    viewer._double_click_detector = DoubleClickDetector()
    viewer._active_screw_part = None
    viewer._pending_screw_press = None
    viewer.volume_manager = _VolumeManager(position)
    return viewer


def _actor_by_name(props, name):
    return next(prop for prop in props if prop.GetObjectName() == name)


def test_display_pick_is_converted_back_to_dicom_world():
    viewer = _make_viewer(center=(-7.0, 223.0, 503.0), position=503.0)

    world = viewer._slice_display_to_world((10.0, -20.0, 0.0))

    assert world == pytest.approx((3.0, 243.0, 503.0))


def test_world_screw_is_drawn_inside_slice_local_coordinates():
    viewer = _make_viewer(center=(-7.0, 223.0, 503.0), position=503.0)
    viewer._screw_data[1] = {
        "entry": (-20.0, 200.0, 503.0),
        "target": (20.0, 240.0, 520.0),
        "color": (0.2, 0.8, 0.2),
        "diameter": 6.0,
    }

    viewer._draw_screw_projection(1)

    props = viewer._screw_overlays[1]
    assert len(props) >= 3
    guide = _actor_by_name(props, "screw-guide")
    line_bounds = guide.GetBounds()
    assert line_bounds[0:2] == pytest.approx((-13.0, 27.0))
    assert line_bounds[2:4] == pytest.approx((-17.0, 23.0))
    assert 0.0 < line_bounds[4] < 1.0
    assert 0.0 < line_bounds[5] < 1.0
    assert guide.GetProperty().GetOpacity() == pytest.approx(0.16)


def test_standard_mpr_hides_off_slice_screw():
    viewer = _make_viewer(center=(-7.0, 223.0, 600.0), position=600.0)
    viewer._screw_data[1] = {
        "entry": (-20.0, 200.0, 503.0),
        "target": (20.0, 240.0, 520.0),
        "color": (0.2, 0.8, 0.2),
        "diameter": 6.0,
    }

    viewer._draw_screw_projection(1)

    props = viewer._screw_overlays[1]
    assert props == []


def test_review_filter_draws_only_selected_screw():
    viewer = _make_viewer(center=(0.0, 0.0, 0.0), position=0.0)
    viewer._review_screw_id = 2
    viewer._screw_data = {
        1: {
            "entry": (-20.0, 0.0, 0.0),
            "target": (20.0, 0.0, 0.0),
            "color": (1.0, 0.9, 0.05),
            "diameter": 6.0,
        },
        2: {
            "entry": (0.0, -20.0, 0.0),
            "target": (0.0, 20.0, 0.0),
            "color": (1.0, 0.9, 0.05),
            "diameter": 6.0,
        },
    }

    viewer._draw_screw_projection(1)
    viewer._draw_screw_projection(2)

    assert viewer._screw_overlays[1] == []
    assert viewer._screw_overlays[2]


def test_review_long_axis_uses_filled_diameter_bar():
    entry = (0.0, 0.0, 0.0)
    target = (0.0, 0.0, 40.0)
    viewer = _make_viewer(center=(0.0, 0.0, 20.0), position=20.0)
    viewer._custom_reslice_axes = build_screw_mpr_axes(
        entry,
        target,
    ).oblique_axial
    viewer._review_screw_id = 1
    viewer._screw_data[1] = {
        "entry": entry,
        "target": target,
        "color": (1.0, 0.9, 0.05),
        "diameter": 6.0,
    }

    viewer._draw_screw_projection(1)

    props = viewer._screw_overlays[1]
    names = [prop.GetObjectName() for prop in props]
    assert "screw-bar" in names
    assert "screw-guide" not in names
    bar_bounds = _actor_by_name(props, "screw-bar").GetBounds()
    spans = sorted((bar_bounds[1] - bar_bounds[0], bar_bounds[3] - bar_bounds[2]))
    assert spans == pytest.approx((6.0, 40.0), abs=0.1)


def test_projection_props_expose_entry_tip_and_shaft_pick_metadata():
    viewer = _make_viewer(center=(0.0, 0.0, 0.0), position=0.0)
    viewer._screw_data[7] = {
        "entry": (-20.0, 0.0, 0.0),
        "target": (20.0, 0.0, 0.0),
        "color": (0.1, 0.8, 1.0),
        "diameter": 6.0,
    }

    viewer._draw_screw_projection(7)

    parts = {
        viewer._screw_prop_parts[viewer._prop_key(prop)]
        for prop in viewer._screw_overlays[7]
        if viewer._prop_key(prop) in viewer._screw_prop_parts
    }
    assert (7, "entry") in parts
    assert (7, "tip") in parts
    assert (7, "shaft") in parts


def test_endpoint_hit_zone_overrides_overlapping_shaft_actor():
    entry = (0.0, 0.0, 0.0)
    target = (0.0, 0.0, 40.0)
    viewer = _make_viewer(center=(0.0, 0.0, 20.0), position=20.0)
    viewer._custom_reslice_axes = build_screw_mpr_axes(
        entry, target
    ).oblique_axial
    viewer._review_screw_id = 1
    viewer._screw_data[1] = {
        "entry": entry,
        "target": target,
        "color": (0.1, 0.8, 1.0),
        "diameter": 6.0,
    }
    viewer._draw_screw_projection(1)
    entry_local = viewer._world_to_slice_display(entry)
    tip_local = viewer._world_to_slice_display(target)
    midpoint = tuple(
        (entry_local[index] + tip_local[index]) / 2.0
        for index in range(3)
    )

    assert viewer._resolve_screw_pick_part(
        1, "shaft", entry_local
    ) == "entry"
    assert viewer._resolve_screw_pick_part(
        1, "shaft", tip_local
    ) == "tip"
    assert viewer._resolve_screw_pick_part(
        1, "shaft", midpoint
    ) == "shaft"


def test_tip_has_large_screen_space_hit_zone_beyond_triangle():
    viewer = _make_viewer(center=(0.0, 0.0, 0.0), position=0.0)
    viewer._screw_data[1] = {
        "entry": (0.0, 0.0, 0.0),
        "target": (40.0, 0.0, 0.0),
        "color": (0.1, 0.8, 1.0),
        "diameter": 6.0,
    }
    viewer._draw_screw_projection(1)
    viewer._slice_point_to_display = lambda point: (point[0], point[1])

    pick_at_display = getattr(MPRViewer, "_pick_screw_part_at_display", None)
    result = (
        None
        if pick_at_display is None
        else pick_at_display(viewer, 58, 0)
    )

    assert result == (1, "tip")


def test_distal_quarter_of_mpr_shaft_is_selected_as_tip():
    viewer = _make_viewer(center=(0.0, 0.0, 0.0), position=0.0)
    viewer._screw_data[1] = {
        "entry": (0.0, 0.0, 0.0),
        "target": (100.0, 0.0, 0.0),
        "color": (0.1, 0.8, 1.0),
        "diameter": 6.0,
    }
    viewer._draw_screw_projection(1)
    viewer._slice_point_to_display = lambda point: (point[0], point[1])

    assert viewer._pick_screw_part_at_display(76, 0) == (1, "tip")


def test_standard_sagittal_axes_keep_original_patient_orientation():
    axes = create_reslice_axes("sagittal", (10.0, 20.0, 30.0))

    x_axis = tuple(axes.GetElement(row, 0) for row in range(3))
    y_axis = tuple(axes.GetElement(row, 1) for row in range(3))
    normal = tuple(axes.GetElement(row, 2) for row in range(3))

    assert x_axis == pytest.approx((0.0, 1.0, 0.0))
    assert y_axis == pytest.approx((0.0, 0.0, 1.0))
    assert normal == pytest.approx((1.0, 0.0, 0.0))


def test_standard_axial_axes_show_anterior_up_left_on_right():
    axes = create_reslice_axes("axial", (10.0, 20.0, 30.0))

    x_axis = tuple(axes.GetElement(row, 0) for row in range(3))
    y_axis = tuple(axes.GetElement(row, 1) for row in range(3))
    normal = tuple(axes.GetElement(row, 2) for row in range(3))

    assert x_axis == pytest.approx((1.0, 0.0, 0.0))
    assert y_axis == pytest.approx((0.0, -1.0, 0.0))
    assert normal == pytest.approx((0.0, 0.0, 1.0))


def test_mpr_drag_callbacks_receive_selected_part_and_world_updates():
    viewer = _make_viewer(center=(0.0, 0.0, 0.0), position=0.0)
    events = []
    viewer.set_screw_interaction_callbacks(
        on_begin=lambda screw_id, part, world: events.append(
            ("begin", screw_id, part, world)
        ) or True,
        on_drag=lambda world, source: events.append(("drag", world, source)),
        on_end=lambda: events.append(("end",)),
        on_select=lambda screw_id: events.append(("select", screw_id)),
    )

    assert viewer._begin_screw_drag(3, "entry", (1.0, 2.0, 3.0)) is True
    viewer._drag_screw_to((4.0, 5.0, 6.0))
    viewer._end_screw_drag()

    assert events == [
        ("select", 3),
        ("begin", 3, "entry", (1.0, 2.0, 3.0)),
        ("drag", (4.0, 5.0, 6.0), "Axial MPR"),
        ("end",),
    ]


def test_single_click_selects_and_double_click_toggles_pointer_move_lock():
    viewer = _make_viewer(center=(0.0, 0.0, 0.0), position=0.0)
    events = []
    viewer.set_screw_interaction_callbacks(
        on_begin=lambda screw_id, part, world: events.append(
            ("begin", screw_id, part, world)
        ) or True,
        on_drag=lambda world, source: events.append(("drag", world, source)),
        on_end=lambda: events.append(("end",)),
        on_select=lambda screw_id: events.append(("select", screw_id)),
    )

    first = viewer._process_screw_press(
        100, 100, (2, "entry"), (1.0, 2.0, 3.0), timestamp=1.0
    )
    second = viewer._process_screw_press(
        102, 101, (2, "entry"), (1.0, 2.0, 3.0), timestamp=1.2
    )
    viewer._drag_screw_to((4.0, 5.0, 6.0))
    viewer._on_left_release(None, None)

    assert first == "selected"
    assert second == "started"
    assert viewer._screw_drag_active is True
    assert viewer._active_screw_part == (2, "entry")
    assert events == [
        ("select", 2),
        ("select", 2),
        ("begin", 2, "entry", (1.0, 2.0, 3.0)),
        ("drag", (4.0, 5.0, 6.0), "Axial MPR"),
    ]

    assert viewer._process_screw_press(
        200, 200, None, None, timestamp=2.0
    ) == "locked"
    assert viewer._process_screw_press(
        201, 201, None, None, timestamp=2.2
    ) == "ended"
    assert viewer._screw_drag_active is False
    assert events[-1] == ("end",)


def test_double_click_survives_screw_mpr_realign_between_presses():
    """The first pick remains valid when selection redraws under the cursor."""
    viewer = _make_viewer(center=(0.0, 0.0, 0.0), position=0.0)
    events = []
    viewer.set_screw_interaction_callbacks(
        on_begin=lambda screw_id, part, world: events.append(
            ("begin", screw_id, part, world)
        ) or True,
        on_select=lambda screw_id: events.append(("select", screw_id)),
    )

    first = viewer._process_screw_press(
        100,
        100,
        (4, "shaft"),
        (1.0, 2.0, 3.0),
        timestamp=1.0,
    )
    # Selecting screw #5 may immediately rebuild Screw MPR, so the same
    # display position can hit only the CT image on the second press.
    second = viewer._process_screw_press(
        102,
        101,
        None,
        None,
        timestamp=1.2,
    )

    assert first == "selected"
    assert second == "started"
    assert viewer._active_screw_part == (4, "shaft")
    assert events[-1] == ("begin", 4, "shaft", (1.0, 2.0, 3.0))


def test_cancel_screw_interaction_clears_lock_without_committing_callback():
    viewer = _make_viewer(center=(0.0, 0.0, 0.0), position=0.0)
    events = []
    viewer.set_screw_interaction_callbacks(on_end=lambda: events.append("end"))
    viewer._screw_drag_active = True
    viewer._active_screw_part = (1, "shaft")

    viewer.cancel_screw_interaction()

    assert viewer._screw_drag_active is False
    assert viewer._active_screw_part is None
    assert events == []


def test_visible_screw_section_moves_when_slice_position_changes():
    viewer = _make_viewer(center=(-7.0, 223.0, 503.0), position=503.0)
    viewer._screw_data[1] = {
        "entry": (-20.0, 200.0, 503.0),
        "target": (20.0, 240.0, 520.0),
        "color": (0.2, 0.8, 0.2),
        "diameter": 6.0,
    }
    viewer._draw_screw_projection(1)
    first_section = _actor_by_name(
        viewer._screw_overlays[1],
        "screw-section",
    )
    first_bounds = first_section.GetBounds()
    first_center = (
        (first_bounds[0] + first_bounds[1]) / 2.0,
        (first_bounds[2] + first_bounds[3]) / 2.0,
    )

    next_axes = create_reslice_axes("axial", (-7.0, 223.0, 510.0))
    viewer._reslice = SimpleNamespace(GetResliceAxes=lambda: next_axes)
    for prop in viewer._screw_overlays[1]:
        viewer._renderer.RemoveViewProp(prop)
    viewer._screw_overlays.clear()
    viewer._draw_screw_projection(1)
    next_section = _actor_by_name(
        viewer._screw_overlays[1],
        "screw-section",
    )
    next_bounds = next_section.GetBounds()
    next_center = (
        (next_bounds[0] + next_bounds[1]) / 2.0,
        (next_bounds[2] + next_bounds[3]) / 2.0,
    )

    assert first_center == pytest.approx((-13.0, 23.0), abs=0.1)
    assert next_center == pytest.approx((3.47, 6.53), abs=0.1)
    assert next_center != pytest.approx(first_center)


def test_fit_to_view_uses_image_bounds_and_viewport_aspect():
    image = vtk.vtkImageData()
    image.SetDimensions(101, 51, 1)
    image.SetSpacing(1.0, 1.0, 1.0)
    image.SetOrigin(-50.0, -25.0, 0.0)
    viewer = MPRViewer.__new__(MPRViewer)
    viewer._renderer = vtk.vtkRenderer()
    viewer._reslice = SimpleNamespace(Update=lambda: None, GetOutput=lambda: image)
    viewer.vtk_widget = SimpleNamespace(width=lambda: 400, height=lambda: 200)

    viewer.fit_to_view(render=False)

    camera = viewer._renderer.GetActiveCamera()
    assert camera.GetParallelProjection() == 1
    assert camera.GetFocalPoint() == pytest.approx((0.0, 0.0, 0.0))
    assert camera.GetParallelScale() == pytest.approx(25.25)


def test_public_mpr_zoom_controls_change_camera_scale_and_render():
    assert hasattr(MPRViewer, "zoom_in")
    assert hasattr(MPRViewer, "zoom_out")
    viewer = MPRViewer.__new__(MPRViewer)
    viewer._renderer = vtk.vtkRenderer()
    camera = viewer._renderer.GetActiveCamera()
    camera.ParallelProjectionOn()
    camera.SetParallelScale(100.0)
    render_calls = []
    viewer._request_render = lambda: render_calls.append(True)

    viewer.zoom_in()

    assert camera.GetParallelScale() == pytest.approx(100.0 / 1.2)
    viewer.zoom_out()
    assert camera.GetParallelScale() == pytest.approx(100.0)
    assert len(render_calls) == 2


def test_mpr_pan_moves_camera_without_changing_zoom():
    assert hasattr(MPRViewer, "_pan_camera_by_pixels")
    viewer = MPRViewer.__new__(MPRViewer)
    viewer._renderer = vtk.vtkRenderer()
    camera = viewer._renderer.GetActiveCamera()
    camera.ParallelProjectionOn()
    camera.SetParallelScale(100.0)
    camera.SetFocalPoint(0.0, 0.0, 0.0)
    camera.SetPosition(0.0, 0.0, 100.0)
    render_calls = []
    viewer._request_render = lambda: render_calls.append(True)

    viewer._pan_camera_by_pixels(10, -20, viewport_height=200)

    assert camera.GetFocalPoint() == pytest.approx((-10.0, 20.0, 0.0))
    assert camera.GetPosition() == pytest.approx((-10.0, 20.0, 100.0))
    assert camera.GetParallelScale() == pytest.approx(100.0)
    assert render_calls == [True]


def test_measurement_is_visible_only_on_its_original_mpr_cut():
    viewer = _make_viewer(center=(0.0, 0.0, 10.0), position=10.0)
    viewer._measurement_props = {}
    viewer._measurement_data = {}
    viewer._request_render = lambda: None

    viewer.add_measurement(
        measurement_id=7,
        points=[(0.0, 0.0, 10.0), (3.0, 4.0, 10.0)],
        label="5.00 mm",
    )

    display_props = [
        prop
        for prop in viewer._measurement_props[7]
        if not prop.GetObjectName().startswith("measurement-handle-")
    ]
    assert all(prop.GetVisibility() for prop in display_props)

    moved_axes = create_reslice_axes("axial", (0.0, 0.0, 12.0))
    viewer._reslice = SimpleNamespace(GetResliceAxes=lambda: moved_axes)
    viewer._update_measurement_visibility()

    assert not any(prop.GetVisibility() for prop in viewer._measurement_props[7])

    original_axes = create_reslice_axes("axial", (0.0, 0.0, 10.0))
    viewer._reslice = SimpleNamespace(GetResliceAxes=lambda: original_axes)
    viewer._update_measurement_visibility()

    assert all(prop.GetVisibility() for prop in display_props)


def test_selected_measurement_shows_pacs_style_point_handles():
    viewer = _make_viewer(center=(0.0, 0.0, 10.0), position=10.0)
    viewer._measurement_props = {}
    viewer._measurement_data = {}
    viewer._measurement_prop_parts = {}
    viewer._selected_measurement_id = None
    viewer._request_render = lambda: None

    viewer.add_measurement(
        measurement_id=4,
        points=[(0.0, 0.0, 10.0), (3.0, 4.0, 10.0)],
        label="5.00 mm",
    )

    handles = [
        prop
        for prop in viewer._measurement_props[4]
        if prop.GetObjectName().startswith("measurement-handle-")
    ]
    assert len(handles) == 2
    assert not any(handle.GetVisibility() for handle in handles)

    viewer.set_selected_measurement(4)

    assert all(handle.GetVisibility() for handle in handles)
