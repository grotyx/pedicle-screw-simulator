"""Focused SegmentationController tests for the final fix-wave batch C findings.

Builds its own minimal ``ui_main_window``/``isolated_qsettings`` fixtures
(mirroring the ones in ``tests/test_ui_integration.py``) rather than
importing them: pytest fixtures re-exported via a plain import shadow their
own name in every consuming test function, which ruff's pyflakes-derived
F811/F401 checks (correctly, if unhelpfully) flag as redefinitions.
"""

import logging
import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("pytestqt")
pytest.importorskip("SimpleITK")

import SimpleITK as sitk
from PyQt6.QtCore import QSettings
from PyQt6.QtWidgets import QApplication

import src.ui.main_window as main_window_module
from src.core.totalseg_integration import SegmentationRunResult
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
    # app_settings() only checks the legacy scope once per process; reset
    # that guard so each test's migration behavior is independent.
    monkeypatch.setattr(main_window_module, "_migrated", False)
    return factory


@pytest.fixture
def ui_main_window(monkeypatch, qtbot, isolated_qsettings):
    """Build MainWindow with lightweight viewer stubs."""
    monkeypatch.setattr(main_window_module, "MPRViewer", DummyMPRViewer)
    monkeypatch.setattr(main_window_module, "Viewer3D", DummyViewer3D)
    QApplication.instance().setProperty("themeName", "soft_light")

    window = main_window_module.MainWindow()
    qtbot.addWidget(window)
    return window


def _create_test_image():
    image = sitk.Image([16, 16, 12], sitk.sitkInt16)
    image = image + 250
    image.SetSpacing((1.0, 1.0, 1.2))
    image.SetOrigin((0.5, 1.0, 2.0))
    return image


def _write_mask(image, path):
    mask = sitk.Cast(image > 0, sitk.sitkUInt8)
    mask.CopyInformation(image)
    sitk.WriteImage(mask, str(path))


class _ProgressStub:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


# ---------------------------------------------------------------------------
# M11 -- reset_state() logs when the thread wait times out
# ---------------------------------------------------------------------------


class _StubThread:
    """A segmentation thread that never actually stops within the wait()."""

    def __init__(self):
        self.cancel_calls = 0

    def request_cancel(self):
        self.cancel_calls += 1

    def isRunning(self):
        return True

    def wait(self, _msecs):
        return False


def test_reset_state_warns_and_still_purges_when_wait_times_out(
    ui_main_window, caplog
):
    window = ui_main_window
    ctrl = window._seg_ctrl
    ctrl._segmentation_thread = _StubThread()
    ctrl._active_work_dir = "C:/fake/run/dir"

    purge_calls = []
    ctrl.workspace.purge = lambda: purge_calls.append(True)

    try:
        with caplog.at_level(logging.WARNING):
            ctrl.reset_state()

        assert purge_calls == [True]
        assert any(
            "did not stop" in record.message and "C:/fake/run/dir" in record.message
            for record in caplog.records
        )
    finally:
        # The stub always reports itself as running; clear it so MainWindow's
        # own teardown close() doesn't hit closeEvent's blocking "still
        # running" QMessageBox (see tests.test_ui_integration._FakeSegmentationThread).
        ctrl._segmentation_thread = None


def test_reset_state_does_not_warn_when_thread_stops_in_time(ui_main_window, caplog):
    window = ui_main_window
    ctrl = window._seg_ctrl

    class _QuickStopThread(_StubThread):
        def wait(self, _msecs):
            return True

    ctrl._segmentation_thread = _QuickStopThread()

    try:
        with caplog.at_level(logging.WARNING):
            ctrl.reset_state()

        assert not any("did not stop" in record.message for record in caplog.records)
    finally:
        ctrl._segmentation_thread = None


# ---------------------------------------------------------------------------
# M12 -- wait cursor and status messages around the GUI-thread resample/regrade
# ---------------------------------------------------------------------------


def test_finished_shows_wait_cursor_and_status_during_resample_and_regrade(
    ui_main_window, tmp_path, monkeypatch
):
    window = ui_main_window
    image = _create_test_image()
    window._on_dicom_loaded(
        image=image,
        metadata={"series_id": "SERIES-CURSOR", "num_slices": image.GetSize()[2]},
        progress=_ProgressStub(),
    )

    mask_path = tmp_path / "mask_cursor.nii.gz"
    _write_mask(image, mask_path)

    messages = []
    monkeypatch.setattr(
        window.statusbar, "showMessage", lambda msg, *a, **kw: messages.append(msg)
    )

    cursor_events = []
    monkeypatch.setattr(
        QApplication,
        "setOverrideCursor",
        staticmethod(lambda *a, **kw: cursor_events.append("set")),
    )
    monkeypatch.setattr(
        QApplication,
        "restoreOverrideCursor",
        staticmethod(lambda *a, **kw: cursor_events.append("restore")),
    )

    window._on_segmentation_finished(
        SegmentationRunResult(
            success=True,
            method="totalsegmentator",
            mask_path=str(mask_path),
            message="ok",
        )
    )

    assert "Resampling pedicle mask…" in messages
    assert "Re-grading screws…" in messages
    assert cursor_events.count("set") == 2
    assert cursor_events.count("restore") == 2
    # Both calls run synchronously on the GUI thread (no worker hand-off);
    # a "set" is always immediately paired with a "restore" before the next.
    for index in range(0, len(cursor_events), 2):
        assert cursor_events[index : index + 2] == ["set", "restore"]
