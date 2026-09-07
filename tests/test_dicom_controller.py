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
