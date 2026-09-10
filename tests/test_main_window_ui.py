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


def test_control_panel_has_three_tabs_with_the_expected_sections(ui_main_window):
    window = ui_main_window

    assert window.control_tabs.count() == 3
    assert [
        window.control_tabs.tabText(index)
        for index in range(window.control_tabs.count())
    ] == ["Study", "Planning", "Tools"]
    assert window.control_section_order == {
        "Study": ["Study", "Segmentation", "Window/Level"],
        "Planning": [
            "Planning",
            "Planning parameters",
            "Screw Parameters",
            "Screw Review",
        ],
        "Tools": ["Measurement", "Measurement List", "Validation"],
    }
    for tab, titles in window.control_section_order.items():
        assert [
            group.title for group in window.control_tab_sections[tab]
        ] == titles


def test_cockpit_is_pinned_outside_the_tab_widget(ui_main_window):
    window = ui_main_window

    assert window.planning_cockpit.objectName() == "planningCockpit"
    assert window.control_tabs.isAncestorOf(window.planning_cockpit) is False
    for widget in (
        window.selected_screw_counter,
        window.selected_screw_title,
        window.selected_screw_metrics,
        window.selected_screw_warning,
        window.screw_previous_btn,
        window.screw_next_btn,
    ):
        assert window.planning_cockpit.isAncestorOf(widget)
    assert window.control_tabs.isAncestorOf(window.screw_list_widget)


def test_sections_keep_their_default_collapsed_state(ui_main_window):
    window = ui_main_window

    assert window.segmentation_group.is_collapsed is False
    assert window.planning_group.is_collapsed is False
    assert window.selected_screw_group.is_collapsed is False
    assert window.selected_screw_group.property("role") == "review"
    assert all(
        group.is_collapsed for group in window.secondary_control_groups
    )


def test_screw_plan_table_round_trips_rows_and_selection(ui_main_window):
    from src.models.screw import Screw

    window = ui_main_window
    table = window.screw_list_widget

    assert table.count() == 0
    assert table.currentRow() == -1

    rows = []
    table.currentRowChanged.connect(rows.append)

    for offset in range(2):
        table.addScrewRow(
            Screw(
                entry_point=(0.0, 0.0, float(offset)),
                target_point=(0.0, -40.0, float(offset)),
                diameter=6.5,
                vertebra_level=f"L{offset + 3}",
                side="left" if offset == 0 else "right",
                grade="A" if offset == 0 else "D",
            ),
            offset,
        )

    assert table.count() == 2
    assert table.rowText(0) == "1 · L3 · Left · 6.5 · 40.0 · Grade A · Manual"
    assert "Grade D" in table.rowText(1)

    table.setCurrentRow(1)
    assert table.currentRow() == 1
    assert rows[-1] == 1

    table.clear()
    assert table.count() == 0
    assert rows[-1] == -1


def test_screw_plan_table_paints_grade_chips_from_the_theme(ui_main_window):
    from src.models.screw import Screw
    from src.ui.screw_plan_table import GRADE_COLUMN

    window = ui_main_window
    table = window.screw_list_widget

    for grade in ("A", "B", "C", "E", "N/A"):
        table.addScrewRow(
            Screw(
                entry_point=(0.0, 0.0, 0.0),
                target_point=(0.0, -40.0, 0.0),
                diameter=6.0,
                grade=grade,
            )
        )

    palette = THEMES["graphite_blue"]
    expected = [
        palette["grade_a"],
        palette["grade_b"],
        palette["grade_c"],
        palette["grade_d"],
        palette["grade_na"],
    ]
    for row, colour in enumerate(expected):
        chip = table.item(row, GRADE_COLUMN)
        assert chip.background().color() == QColor(colour)
        assert chip.foreground().color() == QColor(palette["grade_text"])

    window.apply_theme("graphite_mint", persist=False)

    mint = THEMES["graphite_mint"]
    assert table.item(0, GRADE_COLUMN).background().color() == QColor(
        mint["grade_a"]
    )
    assert table.item(4, GRADE_COLUMN).background().color() == QColor(
        mint["grade_na"]
    )


def test_screw_plan_table_updates_a_row_in_place(ui_main_window):
    from src.models.screw import Screw

    window = ui_main_window
    table = window.screw_list_widget
    table.addScrewRow(
        Screw(
            entry_point=(0.0, 0.0, 0.0),
            target_point=(0.0, -30.0, 0.0),
            diameter=6.0,
            vertebra_level="L4",
            side="right",
            grade="A",
        ),
        0,
    )

    table.updateScrewRow(
        0,
        Screw(
            entry_point=(0.0, 0.0, 0.0),
            target_point=(0.0, -50.0, 0.0),
            diameter=7.5,
            vertebra_level="L5",
            side="left",
            grade="C",
        ),
    )

    assert table.count() == 1
    assert table.rowText(0) == "1 · L5 · Left · 7.5 · 50.0 · Grade C · Manual"


def _grid_position(window, widget):
    index = window._view_layout.indexOf(widget)
    if index < 0:
        return None
    return window._view_layout.getItemPosition(index)


def test_maximize_hides_the_other_panes_and_restores_the_base_layout(
    ui_main_window,
):
    window = ui_main_window
    window.set_view_layout("mpr_focus")

    window.set_view_layout("maximize:sagittal")

    assert window._maximized_view == "sagittal"
    assert window._view_layout_mode == "mpr_focus"
    assert _grid_position(window, window.sagittal_viewer) == (0, 0, 1, 1)
    assert window.sagittal_viewer.isHidden() is False
    for viewer in (
        window.axial_viewer,
        window.coronal_viewer,
        window.viewer_3d,
    ):
        assert viewer.isHidden() is True

    window.toggle_maximized_view("sagittal")

    assert window._maximized_view is None
    assert window._view_layout_mode == "mpr_focus"
    assert _grid_position(window, window.axial_viewer) == (0, 0, 1, 1)
    assert _grid_position(window, window.viewer_3d) == (1, 1, 1, 1)
    for viewer in window._get_mpr_viewers():
        assert viewer.isHidden() is False


def test_header_double_click_toggles_maximize(ui_main_window):
    window = ui_main_window

    window.axial_viewer.header_double_clicked.emit("axial")
    assert window._maximized_view == "axial"

    window.axial_viewer.header_double_clicked.emit("axial")
    assert window._maximized_view is None
    assert window._view_layout_mode == "planning"


def test_maximizing_another_view_keeps_the_original_restore_target(
    ui_main_window,
):
    window = ui_main_window
    window.set_view_layout("mpr_focus")

    window.toggle_maximized_view("axial")
    window.toggle_maximized_view("coronal")

    assert window._maximized_view == "coronal"
    assert window._restore_layout_mode == "mpr_focus"

    window.toggle_maximized_view("coronal")

    assert window._view_layout_mode == "mpr_focus"


def test_maximize_menu_action_uses_ctrl_m_and_leaves_f11_alone(ui_main_window):
    window = ui_main_window

    assert window._maximize_view_action.shortcut().toString() == "Ctrl+M"
    shortcuts = {
        action.shortcut().toString()
        for action in window.findChildren(type(window._maximize_view_action))
    }
    assert "F11" not in shortcuts

    window._maximize_view_action.trigger()
    assert window._maximized_view == "axial"

    window._maximize_view_action.trigger()
    assert window._maximized_view is None


def test_set_view_layout_rejects_an_unknown_maximize_target(ui_main_window):
    with pytest.raises(ValueError, match="Unknown view layout"):
        ui_main_window.set_view_layout("maximize:nope")


def test_status_bar_mode_label_tracks_the_tool_and_screw_mpr(ui_main_window):
    from src.models.screw import Screw

    window = ui_main_window

    assert window._mode_label.text() == "Select"

    window._tool_ctrl.set_tool("screw")
    assert window._mode_label.text() == "Add Screw"

    window._tool_ctrl.set_tool("distance")
    assert window._mode_label.text() == "Distance"

    window._tool_ctrl.set_tool("navigate")
    window._tool_ctrl.screw_tool.add_screw(
        Screw(
            entry_point=(0.0, 0.0, 0.0),
            target_point=(0.0, -40.0, 0.0),
            diameter=6.0,
            vertebra_level="L4",
            side="left",
        )
    )
    window._tool_ctrl._add_screw_to_list(window._tool_ctrl.screw_tool.get_screws()[0])
    window.screw_list_widget.setCurrentRow(0)

    monkey = window._screw_mpr_ctrl
    monkey._active = True
    window.refresh_mode_indicators()

    assert window._mode_label.text() == "Screw MPR · #1 L4 left"
    assert window._screw_mpr_action.isChecked() is True


def test_viewer_header_label_reports_double_clicks(qtbot):
    from PyQt6.QtCore import QPoint, Qt
    from PyQt6.QtGui import QMouseEvent

    from src.ui.viewer_header import ViewerHeaderLabel

    label = ViewerHeaderLabel("Axial", "axial")
    qtbot.addWidget(label)
    seen = []
    label.doubleClicked.connect(seen.append)

    event = QMouseEvent(
        QMouseEvent.Type.MouseButtonDblClick,
        QPoint(4, 4).toPointF(),
        Qt.MouseButton.LeftButton,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )
    label.mouseDoubleClickEvent(event)

    assert seen == ["axial"]
    assert label.view_name == "axial"
    assert label.objectName() == "viewerHeader"
