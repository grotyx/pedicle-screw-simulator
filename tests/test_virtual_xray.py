"""Virtual AP / lateral X-ray projections of a synthetic CT phantom."""

import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from src.core.projection import (
    VIEW_LETTERS,
    project_point,
    project_volume,
)

# Phantom: (z, y, x) array, anisotropic spacing, off-zero origin.
SHAPE_ZYX = (20, 30, 40)
SPACING = (0.5, 1.0, 2.0)  # x, y, z in mm
ORIGIN = (10.0, -20.0, 5.0)
IDENTITY = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
# Rod 1 runs anterior-posterior at voxel i=12, k=7; rod 2 runs left-right at
# voxel j=9, k=4.
ROD_AP_I, ROD_AP_K = 12, 7
ROD_LR_J, ROD_LR_K = 9, 4


def _phantom():
    array = np.full(SHAPE_ZYX, -1000, dtype=np.int16)
    array[ROD_AP_K, :, ROD_AP_I] = 1000
    array[ROD_LR_K, ROD_LR_J, :] = 1000
    return array


def _peak(image):
    return tuple(int(v) for v in np.unravel_index(np.argmax(image), image.shape))


def test_raysum_ap_rod_lands_at_its_voxel_with_superior_up():
    proj = project_volume(_phantom(), SPACING, ORIGIN, IDENTITY, "ap", "raysum")

    # Columns follow patient left (viewer right), rows run superior to inferior.
    assert proj.image.shape == (20, 40)
    assert _peak(proj.image) == (20 - 1 - ROD_AP_K, ROD_AP_I)
    assert proj.width_mm == pytest.approx(40 * 0.5)
    assert proj.height_mm == pytest.approx(20 * 2.0)
    assert not proj.oblique


def test_raysum_lateral_rod_lands_at_its_voxel_with_anterior_left():
    proj = project_volume(_phantom(), SPACING, ORIGIN, IDENTITY, "lateral", "raysum")

    assert proj.image.shape == (20, 30)
    assert _peak(proj.image) == (20 - 1 - ROD_LR_K, ROD_LR_J)
    assert proj.width_mm == pytest.approx(30 * 1.0)
    assert proj.height_mm == pytest.approx(20 * 2.0)


def test_mip_keeps_full_brightness_on_every_rod_pixel():
    proj = project_volume(_phantom(), SPACING, ORIGIN, IDENTITY, "ap", "mip")

    assert proj.image[20 - 1 - ROD_AP_K, ROD_AP_I] == pytest.approx(1.0)
    # Rod 2 projects to a full-brightness line in AP MIP, unlike ray-sum.
    assert np.all(proj.image[20 - 1 - ROD_LR_K, :] == pytest.approx(1.0))
    assert proj.image[0, 0] == pytest.approx(0.0)


def test_screw_points_project_onto_the_same_pixels_as_the_rods():
    ap = project_volume(_phantom(), SPACING, ORIGIN, IDENTITY, "ap", "raysum")
    lateral = project_volume(_phantom(), SPACING, ORIGIN, IDENTITY, "lateral", "raysum")
    rod_ap_world = (10.0 + ROD_AP_I * 0.5, 123.0, 5.0 + ROD_AP_K * 2.0)
    rod_lr_world = (-50.0, -20.0 + ROD_LR_J * 1.0, 5.0 + ROD_LR_K * 2.0)

    assert project_point(ap, rod_ap_world) == pytest.approx((ROD_AP_I, 12.0))
    assert project_point(lateral, rod_lr_world) == pytest.approx((ROD_LR_J, 15.0))


def test_flipped_axis_keeps_patient_right_on_viewer_left():
    # Image x axis runs towards patient right: the projection must flip it so
    # that the rod's world position still maps onto its bright pixel.
    flipped = (-1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    proj = project_volume(_phantom(), SPACING, ORIGIN, flipped, "ap", "raysum")
    rod_world = (10.0 - ROD_AP_I * 0.5, 0.0, 5.0 + ROD_AP_K * 2.0)

    col, row = project_point(proj, rod_world)
    assert _peak(proj.image) == (round(row), round(col))
    assert col == pytest.approx(40 - 1 - ROD_AP_I)


def test_oblique_direction_is_flagged_and_near_identity_is_not():
    def rotated_about_z(degrees):
        a = np.radians(degrees)
        c, s = np.cos(a), np.sin(a)
        return (c, -s, 0.0, s, c, 0.0, 0.0, 0.0, 1.0)

    tilted = project_volume(_phantom(), SPACING, ORIGIN, rotated_about_z(10.0), "ap", "mip")
    almost = project_volume(_phantom(), SPACING, ORIGIN, rotated_about_z(0.5), "ap", "mip")

    assert tilted.oblique
    assert tilted.oblique_deg == pytest.approx(10.0, abs=1e-6)
    assert not almost.oblique


def test_unknown_view_or_mode_is_rejected():
    with pytest.raises(ValueError):
        project_volume(_phantom(), SPACING, ORIGIN, IDENTITY, "axial", "mip")
    with pytest.raises(ValueError):
        project_volume(_phantom(), SPACING, ORIGIN, IDENTITY, "ap", "average")


def test_edge_letters_follow_the_radiological_mpr_convention():
    assert VIEW_LETTERS["ap"] == {"top": "S", "bottom": "I", "left": "R", "right": "L"}
    assert VIEW_LETTERS["lateral"] == {"top": "S", "bottom": "I", "left": "A", "right": "P"}


# --- Dialog ---------------------------------------------------------------


def _dialog(qtbot):
    pytest.importorskip("pytestqt")
    from types import SimpleNamespace

    import SimpleITK as sitk

    from src.models.screw import Screw
    from src.ui.screw_plan_table import ScrewPlanTable
    from src.ui.virtual_xray import VirtualXrayDialog

    image = sitk.GetImageFromArray(_phantom())
    image.SetSpacing(SPACING)
    image.SetOrigin(ORIGIN)
    screws = [
        Screw((14.0, -15.0, 20.0), (14.0, 5.0, 18.0), diameter=6.5),
        Screw((24.0, -15.0, 20.0), (24.0, 5.0, 18.0), diameter=5.5),
    ]
    volume_manager = SimpleNamespace(get_sitk_image=lambda: image)
    screw_tool = SimpleNamespace(get_screws=lambda: list(screws))
    table = ScrewPlanTable()
    qtbot.addWidget(table)
    for index, screw in enumerate(screws):
        table.addScrewRow(screw, index)
    dialog = VirtualXrayDialog(volume_manager, screw_tool, table)
    qtbot.addWidget(dialog)
    return dialog, table, screws


def test_dialog_renders_both_views_and_overlays_every_screw(qtbot):
    dialog, _table, _screws = _dialog(qtbot)
    dialog.show()

    assert not dialog.ap_label.pixmap().isNull()
    assert not dialog.lateral_label.pixmap().isNull()
    assert dialog.overlay_counts() == {"ap": 2, "lateral": 2}
    assert dialog.mode() == "raysum"


def test_dialog_toggles_mode_and_screw_overlay(qtbot):
    dialog, _table, _screws = _dialog(qtbot)
    dialog.show()

    dialog.mip_button.click()
    assert dialog.mode() == "mip"
    dialog.screws_checkbox.setChecked(False)
    assert dialog.overlay_counts() == {"ap": 0, "lateral": 0}


def test_dialog_follows_screw_table_changes_and_selection(qtbot):
    dialog, table, screws = _dialog(qtbot)
    dialog.show()

    table.setCurrentRow(1)
    assert dialog.highlighted_screw() == 1
    screws.pop()
    table.clear()
    table.addScrewRow(screws[0], 0)
    qtbot.waitUntil(lambda: dialog.overlay_counts() == {"ap": 1, "lateral": 1})


def test_dialog_without_a_volume_says_so(qtbot):
    pytest.importorskip("pytestqt")
    from types import SimpleNamespace

    from src.ui.screw_plan_table import ScrewPlanTable
    from src.ui.virtual_xray import VirtualXrayDialog

    table = ScrewPlanTable()
    qtbot.addWidget(table)
    dialog = VirtualXrayDialog(
        SimpleNamespace(get_sitk_image=lambda: None),
        SimpleNamespace(get_screws=lambda: []),
        table,
    )
    qtbot.addWidget(dialog)
    dialog.show()

    assert "Load a CT" in dialog.status_label.text()
    assert dialog.overlay_counts() == {"ap": 0, "lateral": 0}


def test_dialog_reprojects_when_a_new_volume_loads(qtbot):
    pytest.importorskip("pytestqt")
    from types import SimpleNamespace

    import SimpleITK as sitk

    from src.ui.screw_plan_table import ScrewPlanTable
    from src.ui.virtual_xray import VirtualXrayDialog

    observers = {}
    state = {"image": None}
    volume_manager = SimpleNamespace(
        get_sitk_image=lambda: state["image"],
        add_observer=lambda name, callback: observers.__setitem__(name, callback),
    )
    table = ScrewPlanTable()
    qtbot.addWidget(table)
    dialog = VirtualXrayDialog(
        volume_manager, SimpleNamespace(get_screws=lambda: []), table
    )
    qtbot.addWidget(dialog)
    dialog.show()
    assert dialog.ap_label.pixmap().isNull()

    observers["virtual_xray"]("slice_changed")
    state["image"] = sitk.GetImageFromArray(_phantom())
    observers["virtual_xray"]("volume_loaded")
    qtbot.waitUntil(lambda: not dialog.ap_label.pixmap().isNull())


def test_a_hidden_dialog_lets_go_of_the_previous_ct(qtbot):
    pytest.importorskip("pytestqt")
    from types import SimpleNamespace

    import SimpleITK as sitk

    from src.ui.screw_plan_table import ScrewPlanTable
    from src.ui.virtual_xray import VirtualXrayDialog

    observers = {}
    state = {"image": sitk.GetImageFromArray(_phantom())}
    volume_manager = SimpleNamespace(
        get_sitk_image=lambda: state["image"],
        add_observer=lambda name, callback: observers.__setitem__(name, callback),
    )
    table = ScrewPlanTable()
    qtbot.addWidget(table)
    dialog = VirtualXrayDialog(volume_manager, SimpleNamespace(get_screws=lambda: []), table)
    qtbot.addWidget(dialog)
    dialog.show()
    assert dialog._cache_image is state["image"]
    dialog.hide()

    state["image"] = None
    observers["virtual_xray"]("volume_loaded")

    assert dialog._cache_image is None and not dialog._cache


def test_a_screw_change_repaints_the_overlay_without_rescaling_the_ct(qtbot, monkeypatch):
    pytest.importorskip("pytestqt")
    from types import SimpleNamespace

    import SimpleITK as sitk

    import src.ui.virtual_xray as module
    from src.ui.screw_plan_table import ScrewPlanTable

    image = sitk.GetImageFromArray(_phantom())
    volume_manager = SimpleNamespace(get_sitk_image=lambda: image, add_observer=lambda *a: None)
    table = ScrewPlanTable()
    qtbot.addWidget(table)
    dialog = module.VirtualXrayDialog(volume_manager, SimpleNamespace(get_screws=lambda: []), table)
    qtbot.addWidget(dialog)
    dialog.show()
    calls = []
    real = module.QPixmap.fromImage
    monkeypatch.setattr(
        module.QPixmap, "fromImage", staticmethod(lambda *a: (calls.append(1), real(*a))[1])
    )

    dialog.refresh()

    assert calls == []
