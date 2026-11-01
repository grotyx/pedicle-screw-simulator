"""Help > Check for Updates against a local file:// server and a fake install."""

import json
import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("pytestqt")

from PyQt6.QtWidgets import QMainWindow, QMessageBox, QStatusBar

import src.controllers.update_controller as controller_module
from src.core import app_update
from tests.test_app_update import _fake_dist


class _Window(QMainWindow):
    def __init__(self):
        super().__init__()
        self.statusbar = QStatusBar()
        self.setStatusBar(self.statusbar)


@pytest.fixture
def server(tmp_path, monkeypatch):
    """An installed 0.2.5 and a server publishing 0.2.6 with only the app layer changed."""
    install = _fake_dist(tmp_path / "install")
    app_update.build_layers(install, "0.2.5", tmp_path / "old-out")
    published = tmp_path / "server"
    manifest = app_update.build_layers(
        _fake_dist(tmp_path / "new", app_text="app-2"), "0.2.6", published
    )
    (published / "latest").mkdir()
    (published / app_update.MANIFEST_NAME).rename(published / "latest" / app_update.MANIFEST_NAME)
    monkeypatch.setenv("PSS_UPDATE_URL", published.as_uri())
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "localappdata"))
    monkeypatch.delenv("PSS_NO_UPDATE", raising=False)
    return install, manifest


def test_update_downloads_changed_layers_and_waits_for_close(server, qtbot, monkeypatch):
    install, manifest = server
    window = _Window()
    qtbot.addWidget(window)
    asked = []

    def question(_parent, _title, text, *_args, **_kwargs):
        asked.append(text)
        # Yes to downloading, No to restarting now.
        return QMessageBox.StandardButton.No if "Restart" in text else QMessageBox.StandardButton.Yes

    monkeypatch.setattr(QMessageBox, "question", staticmethod(question))
    launched = []
    monkeypatch.setattr(controller_module.subprocess, "Popen", lambda args, **k: launched.append(args))
    ctrl = controller_module.UpdateController(window, install_dir=install)

    ctrl.check()
    qtbot.waitUntil(lambda: ctrl.pending_script is not None, timeout=10000)

    assert "0.2.6 is available" in asked[0] and "(app)" in asked[0]
    staging = ctrl.pending_script.parent
    plan = json.loads((staging / "apply.json").read_text())
    assert plan["version"] == "0.2.6"
    assert (staging / "files" / "PedicleScrewSimulator.exe").read_text() == "app-2"
    # Nothing in the install changes until the app closes.
    assert (install / "PedicleScrewSimulator.exe").read_text() == "app-1"
    assert "close" in window.statusbar.currentMessage()

    ctrl.launch_pending()
    assert launched and launched[0][-1] == str(staging / "apply_update.ps1")
    assert ctrl.pending_script is None


def test_latest_version_says_so_and_downloads_nothing(server, qtbot, monkeypatch):
    install, manifest = server
    app_update.build_layers(_fake_dist(install.parent / "same"), "0.2.6", install.parent / "same-out")
    (install / app_update.MANIFEST_NAME).write_text(
        (install.parent / "same" / app_update.MANIFEST_NAME).read_text()
    )
    window = _Window()
    qtbot.addWidget(window)
    infos = []
    monkeypatch.setattr(QMessageBox, "information", staticmethod(lambda *a, **k: infos.append(a[2])))
    ctrl = controller_module.UpdateController(window, install_dir=install)

    ctrl.check()
    qtbot.waitUntil(lambda: bool(infos), timeout=10000)

    assert "latest version" in infos[0]
    assert ctrl.pending_script is None


def test_source_runs_do_not_update(qtbot, monkeypatch, tmp_path):
    window = _Window()
    qtbot.addWidget(window)
    infos = []
    monkeypatch.setattr(QMessageBox, "information", staticmethod(lambda *a, **k: infos.append(a[2])))
    ctrl = controller_module.UpdateController(window, install_dir=tmp_path)  # no components.json

    assert not ctrl.enabled
    ctrl.check(silent=True)
    assert infos == []
    ctrl.check()
    assert "installed Windows app" in infos[0]
