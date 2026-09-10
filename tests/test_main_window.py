"""Focused MainWindow tests for the final fix-wave batch C findings.

Builds its own minimal ``ui_main_window``/``isolated_qsettings`` fixtures
(mirroring the ones in ``tests/test_ui_integration.py``) rather than
importing them: pytest fixtures re-exported via a plain import shadow their
own name in every consuming test function, which ruff's pyflakes-derived
F811/F401 checks (correctly, if unhelpfully) flag as redefinitions.
"""

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("pytestqt")

from PyQt6.QtCore import QSettings
from PyQt6.QtWidgets import QApplication

import src.ui.main_window as main_window_module
from src.controllers.screw_mpr_controller import SCREW_MPR_CONTROLS_HELP
from src.core.planner_config import PlannerConfig
from src.models.screw import Screw
from tests.test_ui_integration import DummyMPRViewer, DummyViewer3D


@pytest.fixture
def isolated_qsettings(tmp_path, monkeypatch):
    """Redirect MainWindow's QSettings into temp INI files.

    Mirrors ``tests.test_ui_integration.isolated_qsettings``.
    """
    directory = tmp_path / "qsettings"
    directory.mkdir(parents=True, exist_ok=True)

    def factory(organization="Default", application="App", *_args, **_kwargs):
        path = directory / f"{organization}-{application}.ini"
        return QSettings(str(path), QSettings.Format.IniFormat)

    monkeypatch.setattr(main_window_module, "QSettings", factory)
    # app_settings() only checks the legacy scope once per process; reset
    # that guard so each test's migration behavior is independent.
    monkeypatch.setattr(main_window_module, "_migrated", False)
    return factory


@pytest.fixture
def ui_main_window(monkeypatch, qtbot, isolated_qsettings):
    """Build MainWindow with lightweight viewer stubs.

    Mirrors ``tests.test_ui_integration.ui_main_window``.
    """
    monkeypatch.setattr(main_window_module, "MPRViewer", DummyMPRViewer)
    monkeypatch.setattr(main_window_module, "Viewer3D", DummyViewer3D)
    QApplication.instance().setProperty("themeName", "soft_light")

    window = main_window_module.MainWindow()
    qtbot.addWidget(window)
    return window


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
