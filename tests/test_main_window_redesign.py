"""MainWindow wiring for the step rail, main view, tool dock and construct map."""

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("pytestqt")

from PyQt6.QtCore import QSettings  # noqa: E402
from PyQt6.QtWidgets import QApplication, QLabel, QToolBar  # noqa: E402

import src.ui.main_window as main_window_module  # noqa: E402
from src.models.screw import Screw  # noqa: E402
from tests.test_ui_integration import (  # noqa: E402
    DummyMPRViewer,
    DummyViewer3D,
    _load_study,
)


@pytest.fixture
def isolated_qsettings(tmp_path, monkeypatch):
    """Redirect MainWindow's QSettings into temp INI files."""
    directory = tmp_path / "qsettings"
    directory.mkdir(parents=True, exist_ok=True)

    def factory(organization="Default", application="App", *_args, **_kwargs):
        path = directory / f"{organization}-{application}.ini"
        return QSettings(str(path), QSettings.Format.IniFormat)

    monkeypatch.setattr(main_window_module, "QSettings", factory)
    return factory


@pytest.fixture
def ui_main_window(monkeypatch, qtbot, isolated_qsettings):
    """Build MainWindow with lightweight viewer stubs."""
    monkeypatch.setattr(main_window_module, "MPRViewer", DummyMPRViewer)
    monkeypatch.setattr(main_window_module, "Viewer3D", DummyViewer3D)
    QApplication.instance().setProperty("themeName", "graphite_blue")
    window = main_window_module.MainWindow()
    qtbot.addWidget(window)
    return window


def _position(window, widget):
    return window._view_layout.getItemPosition(window._view_layout.indexOf(widget))


def _pane_map(window):
    return {
        "axial": window.axial_viewer,
        "sagittal": window.sagittal_viewer,
        "coronal": window.coronal_viewer,
        "3d": window.viewer_3d,
    }


def _add_levelled_screw(window, level, side, grade="A", diameter=6.5):
    screw = Screw(
        entry_point=(2.0, 3.0, 4.0),
        target_point=(8.0, 9.0, 24.0),
        diameter=diameter,
        length=45.0,
        vertebra_level=level,
        side=side,
        grade=grade,
        source="auto",
    )
    window._tool_ctrl.screw_tool.add_screw(screw)
    window._tool_ctrl._add_screw_to_list(screw)
    return screw


def _flush():
    QApplication.processEvents()


def test_rail_sits_left_of_the_views(ui_main_window):
    window = ui_main_window
    rail = window.workflow_bar

    assert rail.objectName() == "stepRail"
    central = window.centralWidget().layout()
    assert central.indexOf(rail) == 0
    assert central.indexOf(window.main_splitter) == 1


def test_default_layout_is_a_3d_main_view_with_three_thumbnails(ui_main_window):
    window = ui_main_window

    assert window.hero_view == "3d"
    assert _position(window, window.viewer_3d) == (0, 0, 1, 3)
    thumbnails = [window.axial_viewer, window.sagittal_viewer, window.coronal_viewer]
    assert sorted(_position(window, pane)[1] for pane in thumbnails) == [0, 1, 2]
    assert all(_position(window, pane)[0] == 2 for pane in thumbnails)
    assert all(pane.is_thumbnail() for pane in thumbnails)
    assert not window.viewer_3d.is_thumbnail()
    assert window._view_layout.rowMinimumHeight(0) >= 320
    assert window._view_layout.rowMinimumHeight(2) >= 150


def test_promote_swaps_the_main_view_without_reparenting_any_pane(ui_main_window):
    window = ui_main_window
    parents = {name: pane.parentWidget() for name, pane in _pane_map(window).items()}

    window.axial_viewer.promote_button.click()

    assert window.hero_view == "axial"
    assert _position(window, window.axial_viewer) == (0, 0, 1, 3)
    assert window.viewer_3d.is_thumbnail()
    assert not window.axial_viewer.is_thumbnail()
    assert {name: pane.parentWidget() for name, pane in _pane_map(window).items()} == parents


def test_promoting_the_current_main_view_is_a_no_op(ui_main_window):
    window = ui_main_window
    window.set_hero_view("3d")
    assert window.hero_view == "3d"
    with pytest.raises(ValueError):
        window.set_hero_view("oblique")


def test_showing_planning_with_a_new_hero_lays_the_grid_out_once(ui_main_window):
    window = ui_main_window
    window.set_hero_view("sagittal")
    calls = []
    original = window.set_view_layout
    window.set_view_layout = lambda mode: (calls.append(mode), original(mode))

    window.set_hero_view("axial", show_planning=True)

    assert calls == ["planning"]
    assert window.hero_view == "axial"
    assert _position(window, window.axial_viewer) == (0, 0, 1, 3)

    window.set_view_layout("maximize:coronal")
    window.set_hero_view("axial", show_planning=True)
    assert window._maximized_view is None


def test_dock_sits_on_its_own_row_under_the_big_view(ui_main_window):
    window = ui_main_window
    dock = window.tool_dock

    view_container = window.axial_viewer.parentWidget()
    assert dock.parentWidget() is view_container
    assert _position(window, dock) == (1, 0, 1, 3)
    window.set_hero_view("sagittal")
    assert _position(window, dock) == (1, 0, 1, 3)
    window.set_view_layout("maximize:coronal")
    assert _position(window, dock) == (1, 0, 1, 1)
    window.set_view_layout(window._restore_layout_mode)
    assert _position(window, dock) == (1, 0, 1, 3)
    window.set_view_layout("mpr_focus")
    assert _position(window, dock) == (2, 0, 1, 2)
    # The dock never sits inside a viewer, so it can never cover the image.
    for pane in _pane_map(window).values():
        assert not pane.isAncestorOf(dock)
    assert not any(pane.is_thumbnail() for pane in _pane_map(window).values())


def test_swapping_while_maximised_keeps_the_maximise_then_restores_to_it(ui_main_window):
    window = ui_main_window

    window.set_view_layout("maximize:axial")
    window.set_hero_view("coronal")
    assert window._maximized_view == "axial"
    assert _position(window, window.tool_dock) == (1, 0, 1, 1)

    window.toggle_maximized_view("axial")

    assert window._maximized_view is None
    assert window.hero_view == "coronal"
    assert _position(window, window.coronal_viewer) == (0, 0, 1, 3)


def test_dock_hosts_the_existing_tool_actions(ui_main_window):
    window = ui_main_window
    hosted = [button.defaultAction() for button in window.tool_dock.buttons()]

    assert hosted == [
        window._select_tool_action,
        window._add_screw_tool_action,
        window._distance_tool_action,
        window._angle_tool_action,
        window._screw_mpr_action,
        window._fit_mpr_action,
        window._fit_3d_action,
    ]
    toolbar_actions = window.main_toolbar.actions()
    assert window._open_toolbar_action in toolbar_actions
    for action in hosted:
        assert action not in toolbar_actions


def test_view_menu_holds_live_zoom_and_fit_actions(ui_main_window):
    window = ui_main_window
    view_menu = next(
        action.menu() for action in window.menuBar().actions() if action.text() == "View"
    )
    actions = view_menu.actions()
    for action in (
        window._zoom_in_3d_action,
        window._zoom_out_3d_action,
        window._fit_mpr_action,
        window._fit_3d_action,
    ):
        assert action in actions

    window._zoom_in_3d_action.trigger()
    assert window.viewer_3d.zoom_factors == [1.2]


def test_toolbar_shows_study_chip_and_research_badge(ui_main_window):
    window = ui_main_window

    assert window.ruo_badge.text() == "RESEARCH USE ONLY"
    assert window.ruo_badge.objectName() == "ruoBadge"
    assert window.study_chip.objectName() == "studyChip"
    assert window.main_toolbar.isAncestorOf(window.study_chip)
    assert window.main_toolbar.isAncestorOf(window.ruo_badge)
    assert isinstance(window.main_toolbar, QToolBar)


def test_study_chip_without_a_volume(ui_main_window):
    window = ui_main_window
    window.update_study_chip()
    assert window.study_chip.text() == "No study loaded"


def test_study_chip_shows_voxel_spacing_only(ui_main_window):
    window = ui_main_window

    _load_study(window, "CHIP-SERIES-7")

    text = window.study_chip.text()
    assert text == "CT · 1.00 × 1.00 × 1.20 mm"
    assert "CHIP-SERIES-7" not in text


def test_construct_map_tracks_rows_and_selects_on_click(ui_main_window):
    window = ui_main_window
    _add_levelled_screw(window, "L4", "left")
    _add_levelled_screw(window, "L4", "right", grade="B")
    _flush()

    assert window.construct_map.rows() == [("L4", [1], [0], [])]

    window.construct_map.screw_clicked.emit(1)
    assert window.screw_list_widget.currentRow() == 1
    _flush()
    selected = [
        chip.property("screwIndex")
        for chip in window.construct_map.findChildren(QLabel, "constructChip")
        if chip.property("selected") == "true"
    ]
    assert selected == [1]


def test_construct_map_is_never_height_capped_by_its_container(ui_main_window):
    window = ui_main_window
    window.resize(1600, 900)
    window.show()
    window.show_step("Review")
    for level in ("L1", "L2", "L3", "L4", "L5"):
        _add_levelled_screw(window, level, "left")
    _flush()
    _flush()

    node = window.construct_map
    review_page = window.step_panel.page("Review")
    while node is not review_page:
        assert node.maximumHeight() > 10_000, node.objectName()
        node = node.parentWidget()
    assert window.construct_map.height() > 5 * 14


def test_construct_map_updates_when_a_row_is_edited_in_place(ui_main_window):
    window = ui_main_window
    _add_levelled_screw(window, "L5", "left", diameter=6.5)
    window.screw_list_widget.setCurrentRow(0)
    _flush()
    rows_before = window.screw_list_widget.count()

    window._tool_ctrl.set_selected_screw_diameter(7.5)
    _flush()

    assert window.screw_list_widget.count() == rows_before
    chips = window.construct_map.findChildren(QLabel, "constructChip")
    assert len(chips) == 1
    assert "7.5×" in chips[0].text()


def test_new_study_resets_the_main_view_and_the_map(ui_main_window):
    window = ui_main_window
    _add_levelled_screw(window, "L4", "left")
    window.set_hero_view("axial")
    _flush()

    window.reset_workspace()
    _flush()

    assert window.hero_view == "3d"
    assert window.construct_map.rows() == []
    assert _position(window, window.tool_dock) == (1, 0, 1, 3)


def test_new_study_leaves_no_pane_maximised(ui_main_window):
    window = ui_main_window
    window.set_hero_view("sagittal")
    window.set_view_layout("maximize:axial")

    window.reset_workspace()

    assert window._maximized_view is None
    assert window.hero_view == "3d"
    assert _position(window, window.viewer_3d) == (0, 0, 1, 3)
    assert all(pane.isVisible() or not window.isVisible() for pane in _pane_map(window).values())
    assert _position(window, window.tool_dock) == (1, 0, 1, 3)


def test_new_study_keeps_the_chosen_mpr_focus_layout(ui_main_window):
    window = ui_main_window
    window.set_view_layout("mpr_focus")
    window.set_view_layout("maximize:coronal")

    window.reset_workspace()

    assert window._maximized_view is None
    assert window._view_layout_mode == "mpr_focus"
    assert _position(window, window.viewer_3d) == (1, 1, 1, 1)
    assert _position(window, window.tool_dock) == (2, 0, 1, 2)


def test_dock_keeps_working_after_moves_and_a_theme_change(ui_main_window):
    window = ui_main_window
    window.show()
    window.set_hero_view("axial")
    window.set_view_layout("maximize:sagittal")
    window.set_view_layout(window._restore_layout_mode)
    window.apply_theme("graphite_mint", persist=False)
    _flush()

    dock = window.tool_dock
    assert _position(window, dock) == (1, 0, 1, 3)
    assert dock.isVisible()
    assert dock.property("viewerOverlay") == "true"
    assert 'QWidget[viewerOverlay="true"]' in QApplication.instance().styleSheet()

    add_screw = next(
        button
        for button in dock.buttons()
        if button.defaultAction() is window._add_screw_tool_action
    )
    add_screw.click()
    assert window._add_screw_tool_action.isChecked()


def test_step_pages_have_headers(ui_main_window):
    window = ui_main_window
    for name, title in (
        ("Study", "Load a study"),
        ("Segment", "Segment vertebrae"),
        ("Plan", "Plan screws"),
        ("Review", "Review screws"),
    ):
        header = window.step_panel.header(name)
        assert header is not None, name
        assert header.findChild(QLabel, "stepHeaderTitle").text() == title


def test_entering_screw_mpr_makes_axial_the_main_view(ui_main_window):
    window = ui_main_window
    _load_study(window, "MPR-SERIES-1")
    _add_levelled_screw(window, "L4", "left")
    window.screw_list_widget.setCurrentRow(0)

    assert window._screw_mpr_ctrl.enter() is True

    assert window.hero_view == "axial"
    assert _position(window, window.axial_viewer) == (0, 0, 1, 3)

    window._screw_mpr_ctrl.exit()
    assert window.hero_view == "axial"
