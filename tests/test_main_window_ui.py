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


class RenderRecordingMPRViewer(DummyMPRViewer):
    """DummyMPRViewer that counts render requests and pan-icon repaints."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.render_requests = 0
        self.pan_icon_colors = []

    def _request_render(self):
        self.render_requests += 1

    def set_pan_icon_color(self, color):
        self.pan_icon_colors.append(str(color))


class RenderRecordingViewer3D(DummyViewer3D):
    """DummyViewer3D that counts render requests."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.render_requests = 0

    def _request_render(self):
        self.render_requests += 1


@pytest.fixture
def render_recording_main_window(monkeypatch, qtbot, isolated_qsettings):
    """MainWindow whose viewer stubs record every render request."""
    monkeypatch.setattr(main_window_module, "MPRViewer", RenderRecordingMPRViewer)
    monkeypatch.setattr(main_window_module, "Viewer3D", RenderRecordingViewer3D)
    QApplication.instance().setProperty("themeName", "graphite_blue")

    window = main_window_module.MainWindow()
    qtbot.addWidget(window)
    return window


def _pane_map(window):
    return {
        "axial": window.axial_viewer,
        "sagittal": window.sagittal_viewer,
        "coronal": window.coronal_viewer,
        "3d": window.viewer_3d,
    }


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


def test_panel_is_a_four_page_step_stack(ui_main_window):
    window = ui_main_window

    from src.ui.workflow_bar import STEP_NAMES

    assert STEP_NAMES == ("Study", "Segment", "Plan", "Review")
    for name in STEP_NAMES:
        assert window.step_panel.page(name) is not None
    assert window.step_section_order == {
        "Study": ["Study", "Window/Level", "3D Rendering"],
        "Segment": ["Segmentation"],
        "Plan": ["Planning", "Planning parameters", "Manual Screw Defaults"],
        "Review": ["Details", "Measurements"],
    }
    for step, titles in window.step_section_order.items():
        assert [group.title for group in window.step_sections[step]] == titles


def test_clicking_a_workflow_step_shows_its_page(ui_main_window):
    window = ui_main_window

    for index, name in enumerate(("Study", "Segment", "Plan", "Review")):
        window.workflow_bar.buttons[index].click()
        assert window.step_panel.current_step == name


def test_selected_screw_group_is_the_review_page(ui_main_window):
    window = ui_main_window

    assert window.selected_screw_group.property("role") == "review"
    for widget in (
        window.selected_screw_counter,
        window.selected_screw_title,
        window.selected_screw_metrics,
        window.selected_screw_warning,
        window.screw_previous_btn,
        window.screw_next_btn,
    ):
        assert window.selected_screw_group.isAncestorOf(widget)
    assert window.selected_screw_group.isAncestorOf(window.screw_list_widget)


def test_sections_keep_their_default_collapsed_state(ui_main_window):
    window = ui_main_window

    assert window.segmentation_group.is_collapsed is False
    assert window.planning_group.is_collapsed is False
    assert all(
        group.is_collapsed for group in window.secondary_control_groups
    )


def test_every_existing_control_lives_on_its_step_page(ui_main_window):
    window = ui_main_window
    study = window.step_panel.page("Study")
    segment = window.step_panel.page("Segment")
    plan = window.step_panel.page("Plan")
    review = window.step_panel.page("Review")

    attribute_to_page = {
        "open_dicom_btn": study,
        "info_label": study,
        "window_slider": study,
        "level_slider": study,
        "_btn_bone": study,
        "_btn_soft": study,
        "tf_preset_combo": study,
        "opacity_slider": study,
        "seg_run_btn": segment,
        "seg_refine_check": segment,
        "seg_status_label": segment,
        "vertebra_isolate_btn": segment,
        "isolation_hint_label": segment,
        "seg_advanced_panel": segment,
        "vertebra_restore_btn": segment,
        "vertebra_display_list": segment,
        "vertebra_select_all_btn": plan,
        "vertebra_clear_all_btn": plan,
        "vertebra_level_container": plan,
        "workspace_mode_combo": plan,
        "auto_screw_review_notice": plan,
        "auto_screw_plan_btn": plan,
        "auto_screw_status": plan,
        "_btn_clear_screws": plan,
        "plan_mode_combo": plan,
        "plan_trajectory_combo": plan,
        "plan_fill_ratio_spin": plan,
        "plan_wall_clearance_spin": plan,
        "plan_anterior_margin_spin": plan,
        "plan_max_convergence_spin": plan,
        "plan_hu_threshold_spin": plan,
        "plan_narrow_pedicle_spin": plan,
        "plan_narrow_lateral_spin": plan,
        "plan_endplate_parallel_check": plan,
        "plan_endplate_tolerance_spin": plan,
        "plan_reset_defaults_btn": plan,
        "length_spin": plan,
        "diameter_spin": plan,
        "review_title_label": review,
        "selected_screw_counter": review,
        "selected_screw_title": review,
        "selected_screw_grade": review,
        "selected_screw_diameter": review,
        "selected_screw_length": review,
        "screw_warnings_toggle": review,
        "selected_screw_warning": review,
        "screw_list_widget": review,
        "screw_previous_btn": review,
        "screw_next_btn": review,
        "screw_axis_mpr_btn": review,
        "standard_mpr_btn": review,
        "screw_edit_btn": review,
        "remove_screw_btn": review,
        "screw_edit_entry_btn": review,
        "screw_edit_tip_btn": review,
        "screw_edit_move_btn": review,
        "screw_edit_cancel_btn": review,
        "screw_axis_position_label": review,
        "screw_axis_position_slider": review,
        "screw_axis_rotation_spin": review,
        "screw_mpr_reset_btn": review,
        "screw_mpr_controls": review,
        "selected_screw_metrics": review,
        "screw_narrow_legend": review,
        "screw_drag_hint": review,
        "measure_mode_combo": review,
        "measure_finish_btn": review,
        "measure_clear_btn": review,
        "measurement_list_widget": review,
        "show_measurement_btn": review,
        "edit_measurement_btn": review,
        "remove_measurement_btn": review,
    }
    for attribute, page in attribute_to_page.items():
        widget = getattr(window, attribute)
        assert page.isAncestorOf(widget), attribute


def test_review_list_fills_the_panel_at_1600x900(ui_main_window, qtbot):
    from src.ui.step_panel import REVIEW_TABLE_MIN_VISIBLE_PX

    window = ui_main_window
    window.resize(1600, 900)
    window.show()
    qtbot.waitExposed(window)
    # Pin the panel to the width a fresh 1600x900 launch gives it (a little
    # over the 400 px the app starts every session at): splitter
    # auto-redistribution can drift depending on what ran earlier in the same
    # QApplication, and this test is about the Review page's own layout, not
    # incidental splitter arithmetic.
    window.main_splitter.setSizes([1100, 500])
    # The collapsed sections animate their height shut over 200ms; without
    # waiting for that the panel briefly reports its pre-collapse size.
    qtbot.wait(350)
    window.show_step("Review")
    qtbot.wait(350)
    window.main_splitter.setSizes([1100, 500])
    qtbot.wait(50)

    scroll = window.step_panel.scroll_area("Review")
    assert scroll.verticalScrollBar().maximum() == 0
    assert window.screw_list_widget.height() >= REVIEW_TABLE_MIN_VISIBLE_PX

    # The scrolled widget must not need more room than the viewport gives it
    # in either direction -- a wider content widget would silently clip
    # (there is deliberately no horizontal scrollbar to reveal it).
    viewport_size = scroll.viewport().size()
    content_size = scroll.widget().size()
    assert content_size.width() <= viewport_size.width()
    assert content_size.height() <= viewport_size.height()

    table_top_left = window.screw_list_widget.mapTo(scroll.widget(), window.screw_list_widget.rect().topLeft())
    assert table_top_left.x() >= 0
    assert table_top_left.y() >= 0
    table_right = table_top_left.x() + window.screw_list_widget.width()
    table_bottom = table_top_left.y() + window.screw_list_widget.height()
    assert table_right <= content_size.width()
    assert table_bottom <= content_size.height()

    # Also check the panel at its real launch widths (400 px splitter width,
    # 390 px container minimum): the action row's nav buttons and Screw
    # MPR/Edit buttons must fit without a minimum width wider than the
    # viewport, or the right edge of the page silently clips with no
    # horizontal scrollbar to reach it.
    for panel_width in (400, 390):
        window.main_splitter.setSizes([1600 - panel_width, panel_width])
        qtbot.wait(50)
        viewport_size = scroll.viewport().size()
        content_size = scroll.widget().size()
        assert content_size.width() <= viewport_size.width(), (
            f"Review page content ({content_size.width()}px) overflows the "
            f"viewport ({viewport_size.width()}px) at panel width {panel_width}"
        )


def test_key_numbers_sit_above_the_list_and_actions_below(ui_main_window, qtbot):
    window = ui_main_window
    window.resize(1600, 900)
    window.show()
    qtbot.waitExposed(window)
    window.show_step("Review")
    QApplication.processEvents()

    def top(widget):
        return widget.mapTo(window.selected_screw_group, widget.rect().topLeft()).y()

    assert top(window.selected_screw_title) < top(window.selected_screw_diameter)
    assert top(window.selected_screw_diameter) < top(window.screw_list_widget)
    assert top(window.screw_list_widget) < top(window.remove_screw_btn)
    assert top(window.remove_screw_btn) < top(window.details_group)


def test_warnings_collapse_into_one_line(ui_main_window, qtbot):
    from src.models.screw import Screw

    window = ui_main_window
    window.show()
    qtbot.waitExposed(window)
    window.show_step("Review")
    screw = Screw(
        entry_point=(0.0, 0.0, 0.0),
        target_point=(0.0, -40.0, 0.0),
        diameter=6.0,
        vertebra_level="L4",
        side="right",
        grade="B",
        breach_distance=1.2,
        warnings=["warning one", "warning two"],
    )
    window._tool_ctrl.screw_tool.add_screw(screw)
    window._tool_ctrl._add_screw_to_list(screw)
    window.screw_list_widget.setCurrentRow(0)

    assert window.screw_warnings_toggle.text() == "⚠ 3 warnings"
    assert window.selected_screw_warning.isVisible() is False
    full_text = window.selected_screw_warning.text()
    assert "warning one" in full_text
    assert "warning two" in full_text

    window.screw_warnings_toggle.setChecked(True)
    assert window.selected_screw_warning.isVisible() is True
    assert window.selected_screw_warning.text() == full_text


def test_details_is_collapsed_and_holds_the_metric_rows(ui_main_window):
    window = ui_main_window
    assert window.details_group.is_collapsed is True
    for widget in (
        window.selected_screw_convergence,
        window.selected_screw_craniocaudal,
        window.selected_screw_endplate,
        window.selected_screw_alignment,
        window.selected_screw_hu,
        window.selected_screw_source,
        window.selected_screw_body_hu,
        window.selected_screw_wall,
        window.selected_screw_facet,
        window.selected_screw_heary,
        window.selected_screw_trajectory,
        window.selected_screw_pedicle,
    ):
        assert window.details_group.isAncestorOf(widget)


def test_details_endplate_row_names_the_neighbour_reference(ui_main_window):
    from src.models.screw import Screw

    window = ui_main_window
    screw = Screw(
        entry_point=(0.0, 0.0, 0.0),
        target_point=(0.0, -40.0, 0.0),
        diameter=6.0,
        vertebra_level="L1",
        side="right",
        grade="B",
        metrics={
            "endplate_angle_deg": -3.2,
            "endplate_reference": "neighbours",
            "endplate_reference_levels": "T12, L3",
        },
    )
    window._tool_ctrl.screw_tool.add_screw(screw)
    window._tool_ctrl._add_screw_to_list(screw)
    window.screw_list_widget.setCurrentRow(0)

    assert "T12, L3" in window.selected_screw_endplate.text()
    assert "T12, L3" in window.selected_screw_endplate.toolTip()


def test_screw_mpr_controls_show_only_while_active(ui_main_window, qtbot):
    window = ui_main_window
    window.show()
    qtbot.waitExposed(window)
    window.show_step("Review")
    assert window.screw_mpr_controls.isVisible() is False

    # Move off Review so the inactive -> active edge below has somewhere to
    # switch away from -- otherwise "already on Review" would pass trivially.
    window.show_step("Plan")
    window._screw_mpr_ctrl._active = True
    window.refresh_mode_indicators()
    assert window.screw_mpr_controls.isVisible() is True
    assert window.screw_axis_mpr_btn.isVisible() is False
    assert window.standard_mpr_btn.isVisible() is True
    # Screw MPR turning on switches the panel to Review.
    assert window.step_panel.current_step == "Review"

    window._screw_mpr_ctrl._active = False
    window.refresh_mode_indicators()
    assert window.screw_mpr_controls.isVisible() is False
    assert window.screw_axis_mpr_btn.isVisible() is True


@pytest.mark.parametrize(
    "row, count, expected",
    [
        (-1, 0, "No screws"),
        (-1, 1, "1 screw"),
        (-1, 3, "3 screws"),
        (0, 3, "Screw 1 of 3"),
        (2, 3, "Screw 3 of 3"),
    ],
)
def test_format_screw_counter_reports_the_count_when_nothing_is_selected(
    row, count, expected
):
    """Rows can exist with no selection (a loaded plan before it selects one,
    a cleared selection) -- the header must show the count, not claim there
    are no screws while the table and workflow bar disagree."""
    assert main_window_module.MainWindow._format_screw_counter(row, count) == expected


def test_edit_menu_starts_the_edit_modes(monkeypatch, qtbot, isolated_qsettings):
    """The Edit menu's actions call ScrewEditController.start/cancel.

    The controller is created inside MainWindow.__init__ and its signals are
    connected there too, so the class methods must be patched *before*
    construction -- patching the instance afterwards would miss a direct
    (non-lambda) ``.connect(self._screw_edit_ctrl.cancel)`` binding, which
    captures that exact bound-method object rather than looking it up again
    at call time.
    """
    from src.controllers.screw_edit_controller import ScrewEditController

    calls = []
    monkeypatch.setattr(
        ScrewEditController, "start", lambda self, mode: calls.append(mode)
    )
    monkeypatch.setattr(
        ScrewEditController, "cancel", lambda self, *a, **k: calls.append("cancel")
    )
    monkeypatch.setattr(main_window_module, "MPRViewer", DummyMPRViewer)
    monkeypatch.setattr(main_window_module, "Viewer3D", DummyViewer3D)
    QApplication.instance().setProperty("themeName", "graphite_blue")
    window = main_window_module.MainWindow()
    qtbot.addWidget(window)

    window._screw_edit_move_entry_action.trigger()
    window._screw_edit_move_tip_action.trigger()
    window._screw_edit_move_whole_action.trigger()
    window._screw_edit_cancel_action.trigger()

    assert calls == ["entry", "tip", "move", "cancel"]


def test_selecting_a_screw_shows_the_review_step(ui_main_window):
    from src.models.screw import Screw

    window = ui_main_window
    window.show_step("Study")
    screw = Screw(
        entry_point=(0.0, 0.0, 0.0),
        target_point=(0.0, -40.0, 0.0),
        diameter=6.0,
    )
    window._tool_ctrl.screw_tool.add_screw(screw)
    window._tool_ctrl._add_screw_to_list(screw)
    window.screw_list_widget.setCurrentRow(0)

    assert window.step_panel.current_step == "Review"
    # show_step must also mark the workflow bar's Review button active, not
    # only switch the visible page.
    from src.ui.workflow_bar import STEP_NAMES

    review_index = STEP_NAMES.index("Review")
    assert window.workflow_bar.buttons[review_index].property("active") == "true"


def test_plan_table_counts_warnings_per_row(ui_main_window):
    from src.models.screw import Screw
    from src.ui.screw_plan_table import WARNINGS_COLUMN

    window = ui_main_window
    screw = Screw(
        entry_point=(0.0, 0.0, 0.0),
        target_point=(0.0, -40.0, 0.0),
        diameter=6.0,
        vertebra_level="L4",
        side="right",
        grade="N/A",
        warnings=["a", "b"],
    )
    window.screw_list_widget.addScrewRow(screw, 0)

    item = window.screw_list_widget.item(0, WARNINGS_COLUMN)
    assert item.text() == "3"


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
    assert table.rowText(0) == "1 · L3 · Left · -- · 6.5 · 40.0 · Grade A · "
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
    assert table.rowText(0) == "1 · L5 · Left · -- · 7.5 · 50.0 · Grade C · "


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


def _press(window, pane):
    """Send a real left press to *pane* so the app-level focus filter sees it."""
    from PyQt6.QtCore import QEvent, QPointF, Qt
    from PyQt6.QtGui import QMouseEvent

    event = QMouseEvent(
        QEvent.Type.MouseButtonPress,
        QPointF(4.0, 4.0),
        Qt.MouseButton.LeftButton,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )
    QApplication.instance().notify(pane, event)


def test_ctrl_m_follows_the_pane_the_pointer_is_working_in(ui_main_window):
    """Panning or dragging in a pane has to count as being in it.

    ``crosshair_moved`` fires only from ``MPRViewer._on_left_click``, past the
    pan, measure and screw-pick branches, so every other way of working in a
    pane left the focus behind.  Maximising also used to pin the focus to the
    maximised pane, so 3D stayed the Ctrl+M target after a restore no matter
    what the user did next.
    """
    window = ui_main_window

    window.toggle_maximized_view("3d")
    assert window._maximized_view == "3d"
    window.toggle_maximized_view("3d")
    assert window._maximized_view is None

    _press(window, window.sagittal_viewer)
    window._maximize_view_action.trigger()

    assert window._maximized_view == "sagittal"


def test_a_press_inside_a_pane_child_still_names_the_pane(ui_main_window):
    """The press lands on a VTK render window, not on the pane itself."""
    from PyQt6.QtWidgets import QWidget

    window = ui_main_window
    child = QWidget(window.coronal_viewer)

    _press(window, child)
    window._maximize_view_action.trigger()

    assert window._maximized_view == "coronal"


def test_a_press_outside_every_pane_leaves_the_focus_alone(ui_main_window):
    window = ui_main_window
    _press(window, window.coronal_viewer)

    _press(window, window.screw_list_widget)
    window._maximize_view_action.trigger()

    assert window._maximized_view == "coronal"


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
def test_restoring_a_maximized_layout_repaints_every_re_shown_pane(
    render_recording_main_window,
):
    # Re-showing a hidden VTK pane does not repaint its surface, so the 3D
    # view kept a stale, full-size copy of the maximised axial slice.
    window = render_recording_main_window
    panes = _pane_map(window)

    window.set_view_layout("maximize:axial")
    baseline = {name: pane.render_requests for name, pane in panes.items()}

    window.set_view_layout("planning")

    for name, pane in panes.items():
        assert pane.render_requests > baseline[name], name


def test_maximizing_does_not_repaint_the_panes_it_hides(
    render_recording_main_window,
):
    window = render_recording_main_window
    panes = _pane_map(window)
    baseline = {name: pane.render_requests for name, pane in panes.items()}

    window.set_view_layout("maximize:axial")

    assert panes["axial"].render_requests > baseline["axial"]
    for name in ("sagittal", "coronal", "3d"):
        assert panes[name].render_requests == baseline[name], name


def test_theme_change_repaints_the_mpr_pan_icon(render_recording_main_window):
    window = render_recording_main_window

    window.apply_theme("soft_light", persist=False)

    expected = THEMES["soft_light"]["viewer_foreground"]
    for viewer in window._get_mpr_viewers():
        assert viewer.pan_icon_colors[-1] == expected


def test_screw_plan_table_headers_stay_short_with_units_in_the_tooltips(
    ui_main_window,
):
    from PyQt6.QtWidgets import QHeaderView

    from src.ui.screw_plan_table import GRADE_COLUMN, MINIMUM_SECTION_WIDTH_PX

    table = ui_main_window.screw_list_widget
    header = table.horizontalHeader()

    titles = [
        table.horizontalHeaderItem(column).text()
        for column in range(table.columnCount())
    ]
    assert titles == ["#", "Level", "Side", "Pedicle", "Ø", "Len", "Grade", "⚠"]

    assert table.horizontalHeaderItem(3).toolTip() == "Measured pedicle width (mm)"
    assert table.horizontalHeaderItem(4).toolTip() == "Screw diameter (mm)"
    assert table.horizontalHeaderItem(5).toolTip() == "Screw length (mm)"
    assert all(
        table.horizontalHeaderItem(column).toolTip()
        for column in range(table.columnCount())
    )

    from src.ui.screw_plan_table import WARNINGS_COLUMN

    fit = QHeaderView.ResizeMode.ResizeToContents
    stretch = QHeaderView.ResizeMode.Stretch
    assert [
        header.sectionResizeMode(c)
        for c in (0, 3, 4, 5, GRADE_COLUMN, WARNINGS_COLUMN)
    ] == [fit, fit, fit, fit, fit, fit]
    assert [header.sectionResizeMode(c) for c in (1, 2)] == [
        stretch,
        stretch,
    ]
    assert header.minimumSectionSize() == MINIMUM_SECTION_WIDTH_PX


def test_grade_chip_column_is_wide_enough_for_its_text(ui_main_window):
    from src.models.screw import Screw
    from src.ui.screw_plan_table import GRADE_COLUMN

    table = ui_main_window.screw_list_widget
    table.addScrewRow(
        Screw(
            entry_point=(0.0, 0.0, 0.0),
            target_point=(0.0, -40.0, 0.0),
            diameter=6.5,
            vertebra_level="L4",
            side="left",
            grade="B",
        ),
        0,
    )

    chip = table.item(0, GRADE_COLUMN)
    assert chip.text() == "Grade B"
    text_width = table.fontMetrics().horizontalAdvance(chip.text())
    assert table.columnWidth(GRADE_COLUMN) >= text_width
    # The values themselves are unchanged by the shorter headers.
    assert table.rowText(0) == "1 · L4 · Left · -- · 6.5 · 40.0 · Grade B · "


def test_narrow_screws_render_red_in_3d_and_mpr(ui_main_window):
    from src.controllers.tool_controller import screw_display_color
    from src.models.screw import Screw
    from src.utils.constants import COLOR_SCREW, COLOR_SCREW_BREACH

    plain = Screw(entry_point=(0.0, 0.0, 0.0), target_point=(0.0, -40.0, 0.0))
    narrow = Screw(
        entry_point=(0.0, 0.0, 0.0),
        target_point=(0.0, -40.0, 0.0),
        metrics={"narrow_pedicle": True, "pedicle_width_mm": 4.5},
    )

    assert screw_display_color(plain) == pytest.approx(
        tuple(v / 255.0 for v in COLOR_SCREW)
    )
    assert screw_display_color(narrow) == pytest.approx(
        tuple(v / 255.0 for v in COLOR_SCREW_BREACH)
    )
    # A medial breach on a pedicle that is not narrow keeps the normal colour;
    # the grade chip and the warning carry that finding.
    breaching = Screw(
        entry_point=(0.0, 0.0, 0.0),
        target_point=(0.0, -40.0, 0.0),
        metrics={"narrow_pedicle": False, "medial_breach_mm": 1.2},
    )
    assert screw_display_color(breaching) == pytest.approx(
        tuple(v / 255.0 for v in COLOR_SCREW)
    )


def test_screw_plan_table_shows_the_pedicle_width_and_chips_the_narrow_ones(
    ui_main_window,
):
    from PyQt6.QtGui import QColor

    from src.models.screw import Screw
    from src.ui.screw_plan_table import PEDICLE_COLUMN
    from src.ui.styles import get_theme

    table = ui_main_window.screw_list_widget
    table.addScrewRow(
        Screw(
            entry_point=(0.0, 0.0, 0.0), target_point=(0.0, -40.0, 0.0),
            diameter=4.0, vertebra_level="T11", side="left", grade="B",
            metrics={"narrow_pedicle": True, "pedicle_width_mm": 4.5},
        ),
        0,
    )
    table.addScrewRow(
        Screw(
            entry_point=(0.0, 0.0, 0.0), target_point=(0.0, -40.0, 0.0),
            diameter=6.5, vertebra_level="L4", side="right", grade="A",
            metrics={"narrow_pedicle": False, "pedicle_width_mm": 9.2},
        ),
        1,
    )

    assert table.item(0, PEDICLE_COLUMN).text() == "4.5 mm"
    assert table.item(1, PEDICLE_COLUMN).text() == "9.2 mm"
    palette = get_theme("light")
    ui_main_window.apply_theme("light")
    assert table.item(0, PEDICLE_COLUMN).background().color() == QColor(
        palette["grade_d"]
    )
    assert table.item(0, PEDICLE_COLUMN).foreground().color() == QColor(
        palette["grade_text"]
    )
    assert table.item(1, PEDICLE_COLUMN).background().color() != QColor(
        palette["grade_d"]
    )
    assert table.rowText(0) == "1 · T11 · Left · 4.5 mm · 4.0 · 40.0 · Grade B · "


def test_cockpit_shows_the_pedicle_row_and_the_narrow_legend(ui_main_window):
    from src.models.screw import Screw

    window = ui_main_window
    assert "narrow pedicle" in window.screw_narrow_legend.text().lower()

    narrow = Screw(
        entry_point=(0.0, 30.0, 0.0), target_point=(0.0, -10.0, 0.0),
        metrics={"narrow_pedicle": True, "pedicle_width_mm": 4.5},
    )
    index = window._tool_ctrl.add_existing_screw(narrow, select=True)
    window.update_selected_screw_inspector(index, narrow, False)
    assert window.selected_screw_pedicle.text() == "4.5 mm · narrow"
    assert "color:" in window.selected_screw_pedicle.styleSheet()

    plain = Screw(
        entry_point=(0.0, 30.0, 0.0), target_point=(0.0, -10.0, 0.0),
        metrics={"pedicle_width_mm": 9.2},
    )
    window.update_selected_screw_inspector(index, plain, False)
    assert window.selected_screw_pedicle.text() == "9.2 mm"
    assert window.selected_screw_pedicle.styleSheet() == ""

    window.update_selected_screw_inspector(-1, None, False)
    assert window.selected_screw_pedicle.text() == "--"


def test_the_cockpit_says_untrusted_rather_than_narrow_for_a_flagged_width(
    ui_main_window,
):
    """An out-of-band width is flagged in either direction, so "narrow" lies.

    The level is still marked -- same red, same policy -- but the row says why
    it is marked, and the number carries a "?" instead of being quoted as a
    measurement.
    """
    from src.models.screw import Screw

    window = ui_main_window
    wide = Screw(
        entry_point=(0.0, 30.0, 0.0), target_point=(0.0, -10.0, 0.0),
        metrics={
            "narrow_pedicle": True,
            "width_uncertain": True,
            "pedicle_width_mm": 24.0,
        },
    )
    index = window._tool_ctrl.add_existing_screw(wide, select=True)
    window.update_selected_screw_inspector(index, wide, False)

    assert window.selected_screw_pedicle.text() == "24.0 mm? · width not trusted"
    assert "color:" in window.selected_screw_pedicle.styleSheet()


def test_the_narrow_warning_is_listed_first_in_the_cockpit(ui_main_window):
    from src.models.screw import Screw

    window = ui_main_window
    screw = Screw(
        entry_point=(0.0, 30.0, 0.0), target_point=(0.0, -10.0, 0.0),
        grade="B", breach_distance=1.5,
        warnings=[
            "Narrow pedicle (4.5 mm): 4.0 mm screw is 89 % of the width — "
            "verify the measurement or accept a lateral (in-out-in) breach",
            "Breach distance 1.5 mm (grade B)",
        ],
        metrics={"narrow_pedicle": True, "pedicle_width_mm": 4.5},
    )
    index = window._tool_ctrl.add_existing_screw(screw, select=True)

    window.update_selected_screw_inspector(index, screw, False)

    lines = window.selected_screw_warning.text().splitlines()
    assert lines[0].startswith("Narrow pedicle (4.5 mm)")
