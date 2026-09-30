"""MPR reference lines: colour per plane and following the current position."""

import numpy as np
import pytest
import SimpleITK as sitk

from src.core.volume_manager import VolumeManager
from src.ui.mpr_viewer import MPRViewer
from src.utils import constants
from src.utils.vtk_helpers import create_reslice_axes

PLANES = ("axial", "sagittal", "coronal")
PLANE_COLOUR = {
    "axial": constants.COLOR_AXIAL,
    "sagittal": constants.COLOR_SAGITTAL,
    "coronal": constants.COLOR_CORONAL,
}
# world axis a plane is perpendicular to: sagittal -> x, coronal -> y, axial -> z
WORLD_AXIS_PLANE = {0: "sagittal", 1: "coronal", 2: "axial"}


def _bare_viewer(plane, vm=None):
    viewer = MPRViewer.__new__(MPRViewer)
    viewer.plane = plane
    viewer._crosshair_actors = []
    viewer._create_crosshairs()
    if vm is not None:
        viewer.volume_manager = vm
        viewer._reslice = vm.create_mpr_reslice(plane)
        viewer._custom_reslice_axes = None
        viewer._mask_reslice = None
        viewer._color_map = None
        viewer._mask_color_map = None
        viewer._screw_data = {}
        viewer._update_measurement_visibility = lambda: None
        viewer._update_slice_info = lambda: None
        viewer._request_render = lambda: None
    return viewer


@pytest.mark.parametrize("plane", PLANES)
def test_each_reference_line_uses_the_colour_of_the_plane_it_represents(plane):
    viewer = _bare_viewer(plane)
    axes = create_reslice_axes(plane, (0.0, 0.0, 0.0))
    seen = set()
    for ch in viewer._crosshair_actors:
        # "h" is constant local y, "v" constant local x; the plane it marks is
        # perpendicular to the world direction of that in-slice axis.
        column = 1 if ch["orientation"] == "h" else 0
        world = [axes.GetElement(row, column) for row in range(3)]
        represented = WORLD_AXIS_PLANE[int(np.argmax(np.abs(world)))]
        rgb = tuple(round(c * 255) for c in ch["actor"].GetProperty().GetColor())
        assert rgb == PLANE_COLOUR[represented]
        seen.add(represented)
    assert seen == set(PLANES) - {plane}


def _three_viewers():
    vm = VolumeManager()
    vm.set_volume(sitk.GetImageFromArray(np.zeros((20, 20, 20), np.int16)))
    viewers = {p: _bare_viewer(p, vm) for p in PLANES}
    for plane, viewer in viewers.items():
        vm.add_observer(plane, viewer._on_volume_event)
    return vm, viewers


def _assert_lines_at_crosshair(vm, viewers):
    point = vm.get_crosshair_position()
    for viewer in viewers.values():
        local = viewer._world_to_slice_display(point)
        for ch in viewer._crosshair_actors:
            p1 = ch["source"].GetPoint1()
            if ch["orientation"] == "h":
                assert p1[1] == pytest.approx(local[1])
            else:
                assert p1[0] == pytest.approx(local[0])


def test_scrolling_one_view_moves_the_lines_in_the_other_views():
    vm, viewers = _three_viewers()
    viewers["axial"].set_slice_position(vm.get_slice_position("axial") + 5.0)
    _assert_lines_at_crosshair(vm, viewers)


def test_clicking_moves_the_lines_in_every_view_including_the_clicked_one():
    vm, viewers = _three_viewers()
    vm.set_crosshair_position(3.0, 4.0, 6.0, source_plane="coronal")
    _assert_lines_at_crosshair(vm, viewers)
