"""Focused MainWindow tests for the final fix-wave batch C findings.

``isolated_qsettings``/``ui_main_window`` here are thin shims over
``tests.conftest`` (kept so old imports keep working).
"""

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("pytestqt")

from PyQt6.QtWidgets import QApplication

import src.ui.main_window as main_window_module
from src.controllers.screw_mpr_controller import SCREW_MPR_CONTROLS_HELP
from src.core.planner_config import PlannerConfig
from src.models.screw import Screw
from tests.conftest import make_isolated_qsettings, make_ui_main_window
from tests.test_ui_integration import DummyMPRViewer, DummyViewer3D


@pytest.fixture
def isolated_qsettings(tmp_path, monkeypatch):
    """Thin shim over tests.conftest (kept so old imports keep working)."""
    return make_isolated_qsettings(tmp_path, monkeypatch)


@pytest.fixture
def ui_main_window(monkeypatch, qtbot, isolated_qsettings):
    """Thin shim over tests.conftest (kept so old imports keep working)."""
    return make_ui_main_window(monkeypatch, qtbot, isolated_qsettings)


# ---------------------------------------------------------------------------
# M2 -- Trajectory row in the Selected Screw inspector
# ---------------------------------------------------------------------------


def test_inspector_shows_traditional_trajectory_label(ui_main_window):
    window = ui_main_window
    screw = Screw(
        entry_point=(0.0, 0.0, 0.0),
        target_point=(0.0, -45.0, 8.0),
        diameter=6.5,
        vertebra_level="L3",
        side="left",
        metrics={"trajectory_type": "traditional"},
    )
    window._tool_ctrl.screw_tool.add_screw(screw)
    window._tool_ctrl._add_screw_to_list(screw)

    window.screw_list_widget.setCurrentRow(0)

    assert window.selected_screw_trajectory.text() == "Traditional"


def test_inspector_shows_cbt_trajectory_label_with_cranial_angle(ui_main_window):
    window = ui_main_window
    screw = Screw(
        entry_point=(0.0, 0.0, 0.0),
        target_point=(0.0, -45.0, 8.0),
        diameter=5.5,
        vertebra_level="L3",
        side="right",
        metrics={"trajectory_type": "cbt", "cbt_cranial_angle_deg": 22.5},
    )
    window._tool_ctrl.screw_tool.add_screw(screw)
    window._tool_ctrl._add_screw_to_list(screw)

    window.screw_list_widget.setCurrentRow(0)

    assert window.selected_screw_trajectory.text() == "CBT (cranial 22.5°)"


def test_inspector_trajectory_row_is_dash_when_absent(ui_main_window):
    window = ui_main_window
    screw = Screw(
        entry_point=(0.0, 0.0, 0.0),
        target_point=(0.0, -45.0, 8.0),
        diameter=6.0,
        vertebra_level="L3",
        side="left",
    )
    window._tool_ctrl.screw_tool.add_screw(screw)
    window._tool_ctrl._add_screw_to_list(screw)

    window.screw_list_widget.setCurrentRow(0)

    assert window.selected_screw_trajectory.text() == "—"


# ---------------------------------------------------------------------------
# M6 -- Planner settings should not persist unreachable values
# ---------------------------------------------------------------------------


def test_min_convergence_deg_is_not_persisted(ui_main_window):
    assert "min_convergence_deg" not in main_window_module.MainWindow.PLANNER_SETTINGS_KEYS


def test_planner_settings_round_trip_never_pins_min_convergence(
    ui_main_window, isolated_qsettings
):
    """min_convergence_deg has no editable widget, so a stale persisted value
    must never override the dataclass default on the next load."""
    window = ui_main_window

    settings = isolated_qsettings("SNUBH", "PedicleScrewSimulator")
    settings.beginGroup("planner")
    settings.setValue("min_convergence_deg", -20.0)
    settings.endGroup()
    settings.sync()

    window.load_planner_settings()

    assert window.planner_config().min_convergence_deg == pytest.approx(
        PlannerConfig().min_convergence_deg
    )


def test_planner_settings_persist_only_the_three_slider_weights(
    ui_main_window, isolated_qsettings
):
    window = ui_main_window
    window.save_planner_settings()

    settings = isolated_qsettings("SNUBH", "PedicleScrewSimulator")
    settings.beginGroup("planner")
    try:
        for name in ("length", "endplate", "centering"):
            assert settings.value(f"weights/{name}") is None
        for name in ("safety", "density", "rod"):
            assert settings.value(f"weights/{name}") is not None
    finally:
        settings.endGroup()


# ---------------------------------------------------------------------------
# M7 -- Unified QSettings scope with a one-time legacy migration
# ---------------------------------------------------------------------------


def test_app_settings_helper_targets_the_unified_org_and_app(
    monkeypatch, isolated_qsettings
):
    calls = []

    def spy(*args):
        calls.append(args)
        return isolated_qsettings(*args)

    monkeypatch.setattr(main_window_module, "QSettings", spy)

    main_window_module.app_settings()

    assert ("SNUBH", "PedicleScrewSimulator") in calls


def test_app_settings_only_checks_the_legacy_scope_once_per_process(
    monkeypatch, isolated_qsettings
):
    """The migration guard must stop app_settings() from re-opening the
    legacy scope on every call until a theme happens to get written."""
    calls = []

    def spy(*args):
        calls.append(args)
        return isolated_qsettings(*args)

    monkeypatch.setattr(main_window_module, "QSettings", spy)

    main_window_module.app_settings()
    main_window_module.app_settings()
    main_window_module.app_settings()

    legacy_calls = [c for c in calls if c == ("ScrewFixation", "PedicleScrewPlanner")]
    assert len(legacy_calls) == 1


def test_theme_and_planner_settings_share_one_scope(ui_main_window, isolated_qsettings):
    window = ui_main_window
    window.apply_theme("graphite_mint", persist=True)

    unified = isolated_qsettings("SNUBH", "PedicleScrewSimulator")
    assert unified.value("appearance/theme") == "graphite_mint"


def test_theme_change_recolours_the_orientation_markers(ui_main_window):
    window = ui_main_window
    before = {
        viewer.plane: viewer.orientation_refresh_count
        for viewer in window._get_mpr_viewers()
    }

    window.apply_theme("graphite_mint", persist=False)

    for viewer in window._get_mpr_viewers():
        assert viewer.orientation_refresh_count == before[viewer.plane] + 1


def test_legacy_theme_and_geometry_are_migrated_into_the_unified_scope(
    monkeypatch, qtbot, isolated_qsettings
):
    monkeypatch.setattr(main_window_module, "MPRViewer", DummyMPRViewer)
    monkeypatch.setattr(main_window_module, "Viewer3D", DummyViewer3D)

    legacy = isolated_qsettings("ScrewFixation", "PedicleScrewPlanner")
    legacy.setValue("appearance/theme", "graphite_mint")
    legacy.setValue("geometry", b"legacy-geometry-bytes")
    legacy.setValue("windowState", b"legacy-window-state-bytes")
    legacy.sync()

    QApplication.instance().setProperty("themeName", None)
    window = main_window_module.MainWindow()
    qtbot.addWidget(window)
    try:
        unified = isolated_qsettings("SNUBH", "PedicleScrewSimulator")
        assert unified.value("appearance/theme") == "graphite_mint"
        assert bytes(unified.value("geometry")) == b"legacy-geometry-bytes"
        assert bytes(unified.value("windowState")) == b"legacy-window-state-bytes"
    finally:
        window.close()
        window.deleteLater()


# ---------------------------------------------------------------------------
# closeEvent -- cancel a running auto-placement optimiser instead of blocking
# ---------------------------------------------------------------------------


class _FakeAutoPlacementThread:
    """Stand-in for a still-running auto-placement optimiser thread."""

    def __init__(self):
        self.cancel_called = False

    def isRunning(self):
        return True

    def request_cancel(self):
        self.cancel_called = True


def test_close_event_cancels_running_auto_placement_instead_of_blocking(
    ui_main_window,
):
    from PyQt6.QtGui import QCloseEvent

    window = ui_main_window
    ctrl = window._auto_placement_ctrl
    ctrl._thread = _FakeAutoPlacementThread()

    try:
        event = QCloseEvent()
        window.closeEvent(event)

        assert ctrl._thread.cancel_called is True
        assert not event.isAccepted()
        assert (
            window.statusbar.currentMessage()
            == "Cancelling planning — close again once it stops"
        )
    finally:
        # The stub always reports itself as running; clear it so the
        # window's own teardown close() doesn't hit closeEvent's
        # "still running" path again during qtbot cleanup.
        ctrl._thread = None


# ---------------------------------------------------------------------------
# W5 -- Screw MPR interaction controls
# ---------------------------------------------------------------------------


def test_screw_mpr_rotation_control_has_spec_range_and_starts_disabled(
    ui_main_window,
):
    window = ui_main_window

    assert window.screw_axis_rotation_spin.minimum() == -180
    assert window.screw_axis_rotation_spin.maximum() == 180
    assert window.screw_axis_rotation_spin.singleStep() == 5
    assert window.screw_axis_rotation_spin.value() == 0
    assert window.screw_axis_rotation_spin.isEnabled() is False
    assert window.screw_mpr_reset_btn.isEnabled() is False
    assert "right-hand rule" in window.screw_axis_rotation_spin.toolTip()


def test_rotation_spin_and_reset_button_drive_the_screw_mpr_controller(
    ui_main_window,
):
    window = ui_main_window
    controller = window._screw_mpr_ctrl

    window.screw_axis_rotation_spin.setValue(20)
    assert controller.rotation_deg == pytest.approx(20.0)

    # Qt drops click() on a disabled button, and the button is disabled
    # until Screw MPR is entered; enable it so the wiring is exercised.
    window.screw_mpr_reset_btn.setEnabled(True)
    window.screw_mpr_reset_btn.click()
    assert controller.rotation_deg == 0.0
    assert window.screw_axis_rotation_spin.value() == 0
    assert window.screw_axis_position_slider.value() == 50


def test_screw_mpr_handlers_are_registered_on_the_mpr_viewers(ui_main_window):
    window = ui_main_window
    controller = window._screw_mpr_ctrl

    for viewer in window._get_mpr_viewers():
        assert viewer.custom_scroll_handler == controller.handle_scroll
    assert window.coronal_viewer.custom_rotate_handler == (
        controller.handle_rotate_drag
    )
    assert window.axial_viewer.custom_rotate_handler is None
    assert window.sagittal_viewer.custom_rotate_handler is None


def test_help_menu_exposes_the_screw_mpr_gesture_map(ui_main_window):
    window = ui_main_window

    assert window._screw_mpr_help_action.text() == "Screw MPR controls"
    assert "Shift + wheel" in SCREW_MPR_CONTROLS_HELP
    assert "Middle-button drag" in SCREW_MPR_CONTROLS_HELP


# ---------------------------------------------------------------------------
# W8 -- Endplate-parallel option and cockpit row
# ---------------------------------------------------------------------------


def test_endplate_widgets_start_from_the_planner_defaults(ui_main_window):
    window = ui_main_window
    defaults = PlannerConfig()

    assert window.plan_endplate_parallel_check.isChecked() is defaults.endplate_parallel
    assert window.plan_endplate_tolerance_spin.value() == pytest.approx(
        defaults.endplate_tolerance_deg
    )


def test_endplate_widgets_feed_the_planner_config(ui_main_window):
    window = ui_main_window
    window.plan_endplate_parallel_check.setChecked(False)
    window.plan_endplate_tolerance_spin.setValue(6.5)

    config = window.planner_config()

    assert config.endplate_parallel is False
    assert config.endplate_tolerance_deg == pytest.approx(6.5)


def test_endplate_settings_round_trip_through_qsettings(ui_main_window, isolated_qsettings):
    window = ui_main_window
    window.plan_endplate_parallel_check.setChecked(False)
    window.plan_endplate_tolerance_spin.setValue(4.0)
    window.save_planner_settings()

    # Both widgets are wired (Step 5) to autosave on every change, so setting
    # them through the normal setters here would immediately overwrite the
    # (False, 4.0) just persisted above, defeating the round trip this test
    # means to check. Block signals to leave the widgets showing stale
    # values without touching QSettings, then prove load_planner_settings()
    # pulls the persisted values back in regardless.
    window.plan_endplate_parallel_check.blockSignals(True)
    window.plan_endplate_parallel_check.setChecked(True)
    window.plan_endplate_parallel_check.blockSignals(False)
    window.plan_endplate_tolerance_spin.blockSignals(True)
    window.plan_endplate_tolerance_spin.setValue(10.0)
    window.plan_endplate_tolerance_spin.blockSignals(False)
    window.load_planner_settings()

    assert window.plan_endplate_parallel_check.isChecked() is False
    assert window.plan_endplate_tolerance_spin.value() == pytest.approx(4.0)
    assert window.planner_config().endplate_parallel is False


def test_reset_defaults_restores_the_endplate_option(ui_main_window, isolated_qsettings):
    window = ui_main_window
    window.plan_endplate_parallel_check.setChecked(False)
    window.plan_endplate_tolerance_spin.setValue(1.0)

    window.reset_planner_settings()

    assert window.plan_endplate_parallel_check.isChecked() is True
    assert window.plan_endplate_tolerance_spin.value() == pytest.approx(10.0)


def test_cockpit_shows_the_endplate_angle(ui_main_window):
    window = ui_main_window
    screw = Screw(
        entry_point=(0.0, 0.0, 0.0),
        target_point=(0.0, -45.0, 8.0),
        diameter=6.0,
        vertebra_level="L3",
        side="left",
        metrics={"endplate_angle_deg": 2.4},
    )
    window._tool_ctrl.screw_tool.add_screw(screw)
    window._tool_ctrl._add_screw_to_list(screw)

    window.screw_list_widget.setCurrentRow(0)

    assert window.selected_screw_endplate.text() == "+2.4°"


def test_cockpit_endplate_row_is_dash_when_unmeasured(ui_main_window):
    window = ui_main_window
    screw = Screw(
        entry_point=(0.0, 0.0, 0.0),
        target_point=(0.0, -45.0, 8.0),
        diameter=6.0,
        vertebra_level="L3",
        side="left",
    )
    window._tool_ctrl.screw_tool.add_screw(screw)
    window._tool_ctrl._add_screw_to_list(screw)

    window.screw_list_widget.setCurrentRow(0)

    assert window.selected_screw_endplate.text() == "--"


# ---------------------------------------------------------------------------
# W9 -- Optimizer-by-default migration and the cockpit Alignment row
# ---------------------------------------------------------------------------


@pytest.fixture
def planner_window_factory(monkeypatch, qtbot, isolated_qsettings):
    """Build MainWindows over one isolated settings store, viewers stubbed.

    Separate from ``ui_main_window`` on purpose: the migration only runs once
    per settings store, so a test about it must control when the *first* window
    is built.
    """
    monkeypatch.setattr(main_window_module, "MPRViewer", DummyMPRViewer)
    monkeypatch.setattr(main_window_module, "Viewer3D", DummyViewer3D)
    QApplication.instance().setProperty("themeName", "soft_light")

    def build():
        window = main_window_module.MainWindow()
        qtbot.addWidget(window)
        return window

    return isolated_qsettings, build


def test_persisted_legacy_mode_migrates_to_optimizer_once(planner_window_factory):
    isolated, build = planner_window_factory
    settings = isolated("SNUBH", "PedicleScrewSimulator")
    settings.beginGroup("planner")
    settings.setValue("mode", "legacy")
    settings.endGroup()
    settings.sync()

    migrated = build()

    assert migrated.plan_mode_combo.currentData() == "optimizer"
    assert (
        migrated.statusbar.currentMessage()
        == main_window_module.PLANNER_MODE_MIGRATION_MESSAGE
    )

    # A user who chooses Legacy after the migration is respected.
    migrated.plan_mode_combo.setCurrentIndex(migrated.plan_mode_combo.findData("legacy"))
    respected = build()

    assert respected.plan_mode_combo.currentData() == "legacy"
    assert (
        respected.statusbar.currentMessage()
        != main_window_module.PLANNER_MODE_MIGRATION_MESSAGE
    )


def test_optimizer_mode_is_left_alone_by_the_migration(planner_window_factory):
    _isolated, build = planner_window_factory

    window = build()

    assert window.plan_mode_combo.currentData() == "optimizer"
    assert window.statusbar.currentMessage() != main_window_module.PLANNER_MODE_MIGRATION_MESSAGE


def test_inspector_shows_the_construct_alignment_row(ui_main_window):
    window = ui_main_window
    screw = Screw(
        entry_point=(0.0, 0.0, 0.0),
        target_point=(0.0, -45.0, 8.0),
        diameter=6.0,
        vertebra_level="L3",
        side="left",
        metrics={"rod_misalignment_mm": 1.24, "convergence_deviation_deg": -2.13},
    )
    index = window._tool_ctrl.add_existing_screw(screw, select=True)
    # Written after the add: the tool re-measures these over the screws it
    # holds, and a lone screw is trivially on its own rod line.  This test is
    # about how the row renders the numbers, not about how they are measured.
    screw.metrics.update(
        {"rod_misalignment_mm": 1.24, "convergence_deviation_deg": -2.13}
    )
    window.update_selected_screw_inspector(index, screw, False)

    assert window.selected_screw_alignment.text() == "rod 1.2 mm · conv -2.1°"


def test_inspector_alignment_row_defaults_to_a_dash(ui_main_window):
    window = ui_main_window
    screw = Screw(entry_point=(0.0, 0.0, 0.0), target_point=(0.0, -45.0, 8.0))
    index = window._tool_ctrl.add_existing_screw(screw, select=True)
    window.update_selected_screw_inspector(index, screw, False)

    assert window.selected_screw_alignment.text() == "--"

    window.update_selected_screw_inspector(-1, None, False)

    assert window.selected_screw_alignment.text() == "--"


# ---------------------------------------------------------------------------
# The cockpit survives a plan file it did not write
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad", ["n/a", "", None, True, [1.0]])
def test_inspector_alignment_row_dashes_on_an_unreadable_value(ui_main_window, bad):
    """A hand-edited plan JSON must not crash the inspector on selection."""
    window = ui_main_window
    screw = Screw(
        entry_point=(0.0, 0.0, 0.0),
        target_point=(0.0, -45.0, 8.0),
        diameter=6.0,
        metrics={"rod_misalignment_mm": bad, "convergence_deviation_deg": bad},
    )
    index = window._tool_ctrl.add_existing_screw(screw, select=True)
    window.update_selected_screw_inspector(index, screw, False)

    assert window.selected_screw_alignment.text() == "--"


def test_inspector_alignment_row_keeps_the_half_it_can_read(ui_main_window):
    window = ui_main_window
    screw = Screw(
        entry_point=(0.0, 0.0, 0.0),
        target_point=(0.0, -45.0, 8.0),
        diameter=6.0,
        metrics={"rod_misalignment_mm": 1.24, "convergence_deviation_deg": "n/a"},
    )
    index = window._tool_ctrl.add_existing_screw(screw, select=True)
    window.update_selected_screw_inspector(index, screw, False)

    assert window.selected_screw_alignment.text() == "rod 1.2 mm"


# ---------------------------------------------------------------------------
# The narrow-pedicle colour follows the palette
# ---------------------------------------------------------------------------


def test_the_pedicle_row_is_repainted_when_the_theme_changes(ui_main_window):
    """An inline colour is not in the stylesheet, so nothing else repaints it."""
    from src.ui.styles import THEMES

    window = ui_main_window
    narrow = Screw(
        entry_point=(0.0, 30.0, 0.0),
        target_point=(0.0, -10.0, 0.0),
        metrics={"narrow_pedicle": True, "pedicle_width_mm": 4.5},
    )
    index = window._tool_ctrl.add_existing_screw(narrow, select=True)
    window.update_selected_screw_inspector(index, narrow, False)
    assert THEMES["soft_light"]["danger"] in window.selected_screw_pedicle.styleSheet()

    window.apply_theme("graphite_blue", persist=False)

    assert THEMES["graphite_blue"]["danger"] in window.selected_screw_pedicle.styleSheet()


def test_a_theme_change_does_not_colour_a_normal_pedicle(ui_main_window):
    window = ui_main_window
    plain = Screw(
        entry_point=(0.0, 30.0, 0.0),
        target_point=(0.0, -10.0, 0.0),
        metrics={"pedicle_width_mm": 9.2},
    )
    index = window._tool_ctrl.add_existing_screw(plain, select=True)
    window.update_selected_screw_inspector(index, plain, False)

    window.apply_theme("graphite_blue", persist=False)

    assert window.selected_screw_pedicle.styleSheet() == ""


# ---------------------------------------------------------------------------
# The screw tool is judged by the parameters the panel shows
# ---------------------------------------------------------------------------


def test_the_panel_parameters_reach_the_screw_tool(ui_main_window):
    """A manual drag and an auto screw must meet the same thresholds."""
    window = ui_main_window
    tool = window._tool_ctrl.screw_tool

    window.plan_wall_clearance_spin.setValue(1.5)
    window.plan_narrow_pedicle_spin.setValue(6.5)

    assert tool._wall_clearance_mm == pytest.approx(1.5)
    assert tool._narrow_pedicle_mm == pytest.approx(6.5)
    assert tool._width_bound_disagreement_mm == pytest.approx(
        window.planner_config().width_bound_disagreement_mm
    )


def test_resetting_the_planner_defaults_resets_the_screw_tool(ui_main_window):
    from src.core.planner_config import PlannerConfig

    window = ui_main_window
    window.plan_wall_clearance_spin.setValue(1.5)

    window.reset_planner_settings()

    assert window._tool_ctrl.screw_tool._wall_clearance_mm == pytest.approx(
        PlannerConfig().wall_clearance_mm
    )


# ---------------------------------------------------------------------------
# The persisted 1.0 mm wall clearance is retired once
# ---------------------------------------------------------------------------


def _persisted_planner(isolated):
    """A fresh QSettings handle inside the planner group of the shared store."""
    settings = isolated("SNUBH", "PedicleScrewSimulator")
    settings.beginGroup("planner")
    return settings


def _persist_planner(isolated, **values):
    settings = _persisted_planner(isolated)
    for key, value in values.items():
        settings.setValue(key, value)
    settings.endGroup()
    settings.sync()


def _clearance_flag_is_set(isolated):
    settings = _persisted_planner(isolated)
    raw = settings.value(main_window_module._PLANNER_CLEARANCE_MIGRATION_KEY)
    settings.endGroup()
    return str(raw).strip().lower() in ("true", "1", "yes")


def _persisted_clearance(isolated):
    settings = _persisted_planner(isolated)
    raw = settings.value("wall_clearance_mm")
    settings.endGroup()
    return float(raw)


def test_a_persisted_legacy_wall_clearance_is_reset_to_the_new_default(
    planner_window_factory,
):
    """1.0 mm was the old default; kept, it starves the optimiser of solutions."""
    isolated, build = planner_window_factory
    _persist_planner(isolated, wall_clearance_mm=1.0)

    migrated = build()

    assert migrated.plan_wall_clearance_spin.value() == pytest.approx(0.0)
    assert (
        main_window_module.PLANNER_CLEARANCE_MIGRATION_MESSAGE
        in migrated.statusbar.currentMessage()
    )
    assert _persisted_clearance(isolated) == pytest.approx(0.0)
    assert _clearance_flag_is_set(isolated)


def test_a_hand_picked_wall_clearance_is_left_alone_but_still_flagged(
    planner_window_factory,
):
    isolated, build = planner_window_factory
    _persist_planner(isolated, wall_clearance_mm=0.5)

    window = build()

    assert window.plan_wall_clearance_spin.value() == pytest.approx(0.5)
    assert (
        main_window_module.PLANNER_CLEARANCE_MIGRATION_MESSAGE
        not in window.statusbar.currentMessage()
    )
    assert _persisted_clearance(isolated) == pytest.approx(0.5)
    # Set either way, so the check never runs a second time.
    assert _clearance_flag_is_set(isolated)


def test_the_clearance_migration_does_not_run_twice(planner_window_factory):
    isolated, build = planner_window_factory
    _persist_planner(
        isolated,
        wall_clearance_mm=1.0,
        **{main_window_module._PLANNER_CLEARANCE_MIGRATION_KEY: True},
    )

    window = build()

    assert window.plan_wall_clearance_spin.value() == pytest.approx(1.0)
    assert (
        main_window_module.PLANNER_CLEARANCE_MIGRATION_MESSAGE
        not in window.statusbar.currentMessage()
    )
    assert _persisted_clearance(isolated) == pytest.approx(1.0)


def test_an_unreadable_store_does_not_burn_the_clearance_migration(
    planner_window_factory,
):
    """One malformed key must not retire the migration with the 1.0 mm intact.

    ``load_planner_settings`` empties its parsed mapping on the first numeric
    key it cannot read, so the persisted clearance never reaches the check.
    Writing the flag there marked the migration done forever while the value
    it exists to retire sat untouched in the store.
    """
    isolated, build = planner_window_factory
    _persist_planner(
        isolated, wall_clearance_mm=1.0, pedicle_fill_ratio="not-a-number"
    )

    window = build()

    assert _persisted_clearance(isolated) == pytest.approx(1.0)
    assert not _clearance_flag_is_set(isolated)
    assert (
        main_window_module.PLANNER_CLEARANCE_MIGRATION_MESSAGE
        not in window.statusbar.currentMessage()
    )


def test_the_migration_still_runs_once_the_store_is_readable_again(
    planner_window_factory,
):
    isolated, build = planner_window_factory
    _persist_planner(
        isolated, wall_clearance_mm=1.0, pedicle_fill_ratio="not-a-number"
    )
    build()
    _persist_planner(isolated, pedicle_fill_ratio=0.8)

    repaired = build()

    assert repaired.plan_wall_clearance_spin.value() == pytest.approx(0.0)
    assert _persisted_clearance(isolated) == pytest.approx(0.0)
    assert _clearance_flag_is_set(isolated)
    assert (
        main_window_module.PLANNER_CLEARANCE_MIGRATION_MESSAGE
        in repaired.statusbar.currentMessage()
    )


def test_an_empty_store_still_flags_the_clearance_migration(planner_window_factory):
    """Nothing persisted is a completed run: a 1.0 mm picked later is a choice."""
    isolated, build = planner_window_factory

    window = build()

    assert window.plan_wall_clearance_spin.value() == pytest.approx(0.0)
    assert _clearance_flag_is_set(isolated)


def test_a_wall_clearance_chosen_after_the_migration_is_respected(
    planner_window_factory,
):
    isolated, build = planner_window_factory
    _persist_planner(isolated, wall_clearance_mm=1.0)

    migrated = build()
    assert migrated.plan_wall_clearance_spin.value() == pytest.approx(0.0)

    # The user wants the buffer back; the migration must not take it again.
    migrated.plan_wall_clearance_spin.setValue(1.0)
    respected = build()

    assert respected.plan_wall_clearance_spin.value() == pytest.approx(1.0)
    assert (
        main_window_module.PLANNER_CLEARANCE_MIGRATION_MESSAGE
        not in respected.statusbar.currentMessage()
    )


def test_both_migration_messages_reach_the_status_bar_together(
    planner_window_factory,
):
    isolated, build = planner_window_factory
    _persist_planner(isolated, mode="legacy", wall_clearance_mm=1.0)

    migrated = build()

    message = migrated.statusbar.currentMessage()
    assert main_window_module.PLANNER_MODE_MIGRATION_MESSAGE in message
    assert main_window_module.PLANNER_CLEARANCE_MIGRATION_MESSAGE in message
