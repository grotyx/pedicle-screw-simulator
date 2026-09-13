"""Shared Qt fixtures for UI workflow tests.

Canonical home of ``isolated_qsettings`` and ``ui_main_window``. The five
UI test modules used to copy these bodies; they now keep thin shims that
delegate here so behaviour stays identical in one place.
"""

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def make_isolated_qsettings(tmp_path, monkeypatch):
    """Redirect MainWindow QSettings into temp INI files."""
    from PyQt6.QtCore import QSettings

    import src.ui.main_window as main_window_module

    directory = tmp_path / "qsettings"
    directory.mkdir(parents=True, exist_ok=True)

    def factory(organization="Default", application="App", *_args, **_kwargs):
        path = directory / f"{organization}-{application}.ini"
        return QSettings(str(path), QSettings.Format.IniFormat)

    monkeypatch.setattr(main_window_module, "QSettings", factory)
    # app_settings() only checks the legacy scope once per process; reset
    # that guard so each test's migration behaviour is independent.
    monkeypatch.setattr(main_window_module, "_migrated", False)
    return factory


def make_ui_main_window(
    monkeypatch, qtbot, isolated_qsettings, theme_name="soft_light"
):
    """Build MainWindow with lightweight viewer stubs."""
    from PyQt6.QtWidgets import QApplication

    import src.ui.main_window as main_window_module
    from tests.test_ui_integration import DummyMPRViewer, DummyViewer3D

    monkeypatch.setattr(main_window_module, "MPRViewer", DummyMPRViewer)
    monkeypatch.setattr(main_window_module, "Viewer3D", DummyViewer3D)
    QApplication.instance().setProperty("themeName", theme_name)

    window = main_window_module.MainWindow()
    qtbot.addWidget(window)
    return window


@pytest.fixture
def isolated_qsettings(tmp_path, monkeypatch):
    """Redirect MainWindow QSettings into temp INI files."""
    return make_isolated_qsettings(tmp_path, monkeypatch)


@pytest.fixture
def ui_main_window(monkeypatch, qtbot, isolated_qsettings):
    """Build MainWindow with lightweight viewer stubs."""
    return make_ui_main_window(monkeypatch, qtbot, isolated_qsettings)
