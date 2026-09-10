"""Offscreen UI-polish tests for the W6 workstream."""

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("pytestqt")

from PyQt6.QtCore import QSettings
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import QApplication, QWidgetAction

import src.ui.main_window as main_window_module
from src.ui.styles import THEMES
from src.ui.tool_icons import TOOL_ICON_KINDS, create_tool_icon
from tests.test_ui_integration import DummyMPRViewer, DummyViewer3D


@pytest.fixture
def isolated_qsettings(tmp_path, monkeypatch):
    """Redirect MainWindow's QSettings into temp INI files."""
    directory = tmp_path / "qsettings"
    directory.mkdir(parents=True, exist_ok=True)

    def factory(organization="Default", application="App", *_args, **_kwargs):
        path = directory / f"{organization}-{application}.ini"
        return QSettings(str(path), QSettings.Format.IniFormat)

    monkeypatch.setattr(main_window_module, "QSettings", factory)
    monkeypatch.setattr(main_window_module, "_migrated", False)
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


def _toolbar_entries(window):
    return [
        "|" if action.isSeparator() else action.text()
        for action in window.main_toolbar.actions()
    ]


def test_icon_factory_draws_every_kind_in_every_theme(qtbot):
    assert len(TOOL_ICON_KINDS) == 14
    for palette in THEMES.values():
        for kind in TOOL_ICON_KINDS:
            icon = create_tool_icon(kind, palette["accent"])
            assert not icon.isNull()
            assert not icon.pixmap(22, 22).isNull()


def test_icon_factory_rejects_an_unknown_kind():
    with pytest.raises(ValueError, match="Unknown tool icon"):
        create_tool_icon("not-a-glyph")


def test_toolbar_order_matches_the_clinical_grouping(ui_main_window):
    assert _toolbar_entries(ui_main_window) == [
        "Open DICOM",
        "|",
        "Select",
        "Add Screw",
        "Distance",
        "Angle",
        "|",
        "Fit MPR",
        "3D -",
        "3D +",
        "Fit 3D",
        "|",
        "Screw MPR",
    ]


def test_every_toolbar_action_carries_an_icon(ui_main_window):
    for action in ui_main_window.main_toolbar.actions():
        if action.isSeparator():
            continue
        assert not action.icon().isNull()


def test_theme_and_layout_combos_moved_into_the_view_menu(ui_main_window):
    window = ui_main_window
    view_menu = next(
        action.menu()
        for action in window.menuBar().actions()
        if action.text() == "View"
    )
    hosted = []
    for action in view_menu.actions():
        widget = getattr(action, "defaultWidget", lambda: None)()
        if widget is not None:
            hosted.extend(widget.findChildren(type(window.theme_combo)))

    assert window.theme_combo in hosted
    assert window.layout_combo in hosted
    # Qt6 auto-creates a QToolButton for every plain QAction, so the toolbar is
    # checked for hosted widgets instead: no QWidgetAction, and neither combo.
    assert not any(
        isinstance(action, QWidgetAction)
        for action in window.main_toolbar.actions()
    )
    assert not window.main_toolbar.isAncestorOf(window.theme_combo)
    assert not window.main_toolbar.isAncestorOf(window.layout_combo)


def test_workspace_mode_combo_moved_into_the_planning_group(ui_main_window):
    window = ui_main_window
    assert window.planning_group.isAncestorOf(window.workspace_mode_combo)
    assert window.workspace_mode_combo.currentData() == "planning"
    assert window.plan_mode_combo.currentData() == "optimizer"


def test_theme_change_repaints_the_toolbar_icons(ui_main_window):
    window = ui_main_window

    window.apply_theme("graphite_mint", persist=False)

    accent = QColor(THEMES["graphite_mint"]["accent"])
    image = window._select_tool_action.icon().pixmap(22, 22).toImage()
    hits = 0
    for y in range(image.height()):
        for x in range(image.width()):
            pixel = image.pixelColor(x, y)
            if pixel.alpha() < 200:
                continue
            if (
                abs(pixel.red() - accent.red()) <= 12
                and abs(pixel.green() - accent.green()) <= 12
                and abs(pixel.blue() - accent.blue()) <= 12
            ):
                hits += 1
    assert hits > 0


def test_screw_mpr_toolbar_action_tracks_the_controller(ui_main_window):
    window = ui_main_window

    assert window._screw_mpr_action.isCheckable()
    assert window._screw_mpr_action.isChecked() is False
    assert window._screw_mpr_action.isEnabled() is False

    window._screw_mpr_action.trigger()

    assert window._screw_mpr_ctrl.is_active is False
    assert window._screw_mpr_action.isChecked() is False
