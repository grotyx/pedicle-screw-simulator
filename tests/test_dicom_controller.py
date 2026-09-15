"""Lifecycle tests for DICOM loading and application shutdown."""

import logging

from src.controllers.dicom_controller import DicomController, DicomLoadThread
from src.core.dicom_loader import DicomLoader
from src.ui.main_window import MainWindow


class _StatusBar:
    def __init__(self):
        self.message = ""

    def showMessage(self, message):
        self.message = message


class _Window:
    def __init__(self):
        self.statusbar = _StatusBar()


class _Thread:
    def __init__(self, running):
        self._running = running

    def isRunning(self):
        return self._running


class _Event:
    def __init__(self):
        self.ignored = False

    def ignore(self):
        self.ignored = True


def test_is_running_reflects_load_thread_state():
    controller = DicomController(object(), _Window())
    assert controller.is_running is False

    controller._load_thread = _Thread(True)
    assert controller.is_running is True

    controller._load_thread = _Thread(False)
    assert controller.is_running is False


def test_scan_directory_returns_empty_list_when_gdcm_returns_none(monkeypatch):
    class _Reader:
        def GetGDCMSeriesIDs(self, _directory):
            return None

    monkeypatch.setattr(
        "src.core.dicom_loader.sitk.ImageSeriesReader",
        lambda: _Reader(),
    )

    assert DicomLoader().scan_directory("empty") == []


def test_open_folder_ignores_duplicate_load(monkeypatch):
    window = _Window()
    controller = DicomController(object(), window)
    controller._load_thread = _Thread(True)

    def unexpected_dialog(*_args, **_kwargs):
        raise AssertionError("folder dialog must not open during an active load")

    monkeypatch.setattr(
        "src.controllers.dicom_controller.QFileDialog.getExistingDirectory",
        unexpected_dialog,
    )

    controller.open_folder()

    assert "already" in window.statusbar.message.lower()


class _SegController:
    def __init__(self, running):
        self.is_running = running


def test_open_folder_blocks_while_segmentation_is_running(monkeypatch):
    window = _Window()
    window._seg_ctrl = _SegController(running=True)
    controller = DicomController(object(), window)

    def unexpected_dialog(*_args, **_kwargs):
        raise AssertionError("folder dialog must not open during segmentation")

    monkeypatch.setattr(
        "src.controllers.dicom_controller.QFileDialog.getExistingDirectory",
        unexpected_dialog,
    )

    controller.open_folder()

    assert "segmentation" in window.statusbar.message.lower()


def test_close_event_blocks_while_dicom_is_loading(monkeypatch):
    warnings = []

    class _Controller:
        is_running = True

    class _MainWindowStub:
        _dicom_ctrl = _Controller()

    monkeypatch.setattr(
        "src.ui.main_window.QMessageBox.warning",
        lambda *args: warnings.append(args),
    )
    event = _Event()

    MainWindow.closeEvent(_MainWindowStub(), event)

    assert event.ignored is True
    assert warnings


def test_series_picker_prioritizes_largest_ct_series(monkeypatch):
    window = _Window()
    controller = DicomController(object(), window)
    captured = {}

    def choose_first(_parent, _title, _label, options, current, _editable):
        captured["options"] = options
        captured["current"] = current
        return options[0], True

    monkeypatch.setattr(
        "src.controllers.dicom_controller.QInputDialog.getItem",
        choose_first,
    )
    selected = controller._select_series_id(
        ["LOCALIZER", "CT-VOLUME"],
        [
            {
                "series_id": "LOCALIZER",
                "modality": "CT",
                "description": "Topogram",
                "num_files": 1,
            },
            {
                "series_id": "CT-VOLUME",
                "modality": "CT",
                "description": "Spine 0.5 mm",
                "num_files": 537,
            },
        ],
    )

    assert selected == "CT-VOLUME"
    assert "537 slices" in captured["options"][0]
    assert captured["current"] == 0


def test_load_error_is_logged_not_printed(caplog, capsys, qapp):
    thread = DicomLoadThread(directory="Z:/does/not/exist")
    errors = []
    thread.error.connect(errors.append)
    with caplog.at_level(logging.ERROR, logger="src.controllers.dicom_controller"):
        thread.run()
    assert errors
    assert "DICOM Load Error" not in capsys.readouterr().out


def test_load_error_from_a_raised_exception_is_logged_not_printed(
    monkeypatch, caplog, capsys, qapp
):
    """DicomLoadThread.run() must route unexpected exceptions through the
    logger (with the traceback at debug level) instead of printing to
    stdout, so patient-bearing tracebacks never land in shipped logs
    unfiltered."""

    def _raise(*_args, **_kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(DicomLoader, "scan_directory", _raise)

    thread = DicomLoadThread(directory="Z:/does/not/exist")
    errors = []
    thread.error.connect(errors.append)
    with caplog.at_level(logging.DEBUG, logger="src.controllers.dicom_controller"):
        thread.run()

    assert errors == ["boom"]
    out = capsys.readouterr().out
    assert "DICOM Load Error" not in out
    assert "boom" not in out
    messages = [record.getMessage() for record in caplog.records]
    assert any("DICOM load failed" in message and "boom" in message for message in messages)
    assert any("Traceback" in message for message in messages)


def test_cancel_before_pixel_read_reports_cancelled(qapp):
    """A cancel set before load_series runs must surface as a cancel, not a load."""
    thread = DicomLoadThread(directory="any", series_id="SERIES")
    thread.request_cancel()
    errors = []
    finished = []
    thread.error.connect(errors.append)
    thread.finished.connect(lambda *a: finished.append(a))
    thread.run()
    assert errors == ["DICOM load cancelled"]
    assert finished == []


def test_cancelled_result_is_discarded_without_touching_viewers(qapp):
    """A late finished() after cancel must not set a volume or info text."""
    from src.ui.job_dialog import JobDialog

    calls = {"volume": 0}

    class _VM:
        def set_volume(self, _image):
            calls["volume"] += 1

    class _Label:
        def __init__(self):
            self.text = ""

        def setText(self, value):
            self.text = value

    class _Window:
        def __init__(self):
            self.statusbar = _StatusBar()
            self.info_label = _Label()
            self._current_series_id = None

        def reset_workspace(self):
            return None

    window = _Window()
    controller = DicomController(_VM(), window)
    thread = DicomLoadThread(directory="any", series_id="SERIES")
    thread.request_cancel()
    controller._load_thread = thread
    progress = JobDialog("Loading DICOM...", None)
    controller._on_loaded(object(), {"series_id": "SERIES"}, progress)

    assert calls["volume"] == 0
    assert window.info_label.text == ""
    assert "cancelled" in window.statusbar.message.lower()
    progress.deleteLater()


def test_progress_reports_to_log_tail_and_freezes_on_cancel(qapp):
    """Thread progress must use set_log, keeping the static label.

    Wiring progress to setLabelText repainted mark_cancelling's "Cancelling
    load…" back to a progress message; the log tail leaves the label alone.
    """
    from src.ui.job_dialog import JobDialog

    dialog = JobDialog("Loading", None, on_cancel=lambda: None)
    try:
        label_before = dialog.labelText()
        dialog.set_log("Scanning DICOM directory...")
        assert dialog.labelText() == label_before
        assert "Scanning" in dialog._log_label.text()

        dialog.mark_cancelling("Cancelling load…")
        # The controller's progress slot skips the log write while
        # cancelling; mirror that guard here.
        if not dialog.is_cancelling:
            dialog.set_log("Loading series: X")
        assert dialog.labelText() == "Cancelling load…"
    finally:
        dialog.close_cleanly()
        dialog.deleteLater()


def test_dicom_dialog_is_window_modal(qapp):
    """The DICOM dialog is window-modal so it cannot widen a mid-load race."""
    from PyQt6.QtCore import Qt

    from src.ui.job_dialog import JobDialog

    dialog = JobDialog(
        "Loading DICOM...", None, on_cancel=lambda: None, window_modal=True
    )
    try:
        assert dialog.windowModality() == Qt.WindowModality.WindowModal
    finally:
        dialog.close_cleanly()
        dialog.deleteLater()


def test_job_dialog_close_does_not_reenter_cancel(qapp):
    """Closing the shared dialog must not fire the cancel callback."""
    from src.ui.job_dialog import JobDialog

    fired = []
    dialog = JobDialog("job", None, on_cancel=lambda: fired.append(True))
    dialog.close_cleanly()
    assert fired == []
    dialog.deleteLater()


def test_job_dialog_reports_progress_and_log(qapp):
    """Determinate percentages and the log tail must reach the widgets."""
    from src.ui.job_dialog import JobDialog

    dialog = JobDialog("job", None)
    dialog.set_progress(3, 10)
    assert (dialog.minimum(), dialog.maximum(), dialog.value()) == (0, 10, 3)
    assert dialog._log_label.text() == ""
    dialog.set_log("L4 left (3/10)")
    assert dialog._log_label.text() == "L4 left (3/10)"
    dialog.set_progress(0, 0)
    assert (dialog.minimum(), dialog.maximum()) == (0, 0)
    dialog.mark_cancelling("Cancelling…")
    assert dialog.is_cancelling is True
    assert dialog.labelText() == "Cancelling…"
    dialog.close_cleanly()
    dialog.deleteLater()


def test_job_dialog_frozen_build_has_no_cancel_button(qapp):
    """The packaged segmentation build degrades to a log-only dialog."""
    from src.ui.job_dialog import JobDialog

    dialog = JobDialog("job", None, cancellable=False)
    assert dialog.findChild(object) is not None  # log tail still present
    assert dialog._log_label.text() == ""
    dialog.close_cleanly()
    dialog.deleteLater()
