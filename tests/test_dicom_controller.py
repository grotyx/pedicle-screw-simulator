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

    assert len(errors) == 1 and "RuntimeError" in errors[0]
    out = capsys.readouterr().out
    assert "DICOM Load Error" not in out
    assert "boom" not in out
    messages = [record.getMessage() for record in caplog.records]
    assert any("DICOM load failed" in m and "RuntimeError" in m for m in messages)
    # the frames carry source lines (code), never the exception message
    assert not any("boom" in m for m in messages if "Frames" not in m)
    assert any("Frames" in m for m in messages)


_PATIENT_FOLDERS = [
    r"D:\PACS\HONG_GILDONG\IM1",
    r"\\pacs\export\KIMMINSU\IM0001",
    "Z:/PACS export/Kim Minsu 1234/CT",
    "/mnt/pacs/PARKJISOO/study/IM2",
    r"D:\PACS\Hong Gildong\nested\IM1",
]
_NAME_FRAGMENTS = ["GILDONG", "KIMMINSU", "Minsu", "PARKJISOO", "Gildong", "PACS", "pacs"]


def _assert_no_names(*texts):
    for text in texts:
        for fragment in _NAME_FRAGMENTS:
            assert fragment not in text, (fragment, text)


def test_load_error_log_and_dialog_text_do_not_contain_patient_folders(
    monkeypatch, caplog, qapp
):
    def _raise(_self, directory):
        raise RuntimeError(f"Could not read {directory}/IM0001.dcm: bad header")

    monkeypatch.setattr(DicomLoader, "scan_directory", _raise)
    for folder in _PATIENT_FOLDERS:
        caplog.clear()
        thread = DicomLoadThread(directory=folder)
        errors = []
        thread.error.connect(errors.append)
        with caplog.at_level(logging.DEBUG, logger="src.controllers.dicom_controller"):
            thread.run()

        logged = " ".join(r.getMessage() for r in caplog.records)
        assert "RuntimeError" in logged and "RuntimeError" in errors[0]
        _assert_no_names(logged, errors[0])


def test_scan_and_summary_failures_do_not_leak_patient_folders(monkeypatch, caplog):
    dialogs = []
    monkeypatch.setattr(
        "src.controllers.dicom_controller.QMessageBox.critical",
        lambda _parent, title, text: dialogs.append(text),
    )

    class _Win(_Window):
        _seg_ctrl = None

    for folder in _PATIENT_FOLDERS:
        monkeypatch.setattr(
            "src.controllers.dicom_controller.QFileDialog.getExistingDirectory",
            lambda *a, _f=folder, **k: _f,
        )

        def _raise(_self, directory):
            raise OSError(f"cannot open {directory}/IM0001.dcm")

        monkeypatch.setattr(DicomLoader, "scan_directory", _raise)
        caplog.clear()
        with caplog.at_level(logging.DEBUG, logger="src.controllers.dicom_controller"):
            DicomController(object(), _Win()).open_folder()

        logged = " ".join(r.getMessage() for r in caplog.records)
        assert "OSError" in logged and "OSError" in dialogs[-1]
        _assert_no_names(logged, dialogs[-1])

    # series-summary failure (multi-series folder) is logged without the message
    monkeypatch.setattr(DicomLoader, "scan_directory", lambda _s, d: ["A", "B"])

    def _bad_summaries(_self, directory):
        raise OSError(f"cannot open {directory}/IM0001.dcm")

    monkeypatch.setattr(DicomLoader, "get_series_summaries", _bad_summaries)
    monkeypatch.setattr(DicomController, "_select_series_id", lambda *a: None)
    caplog.clear()
    with caplog.at_level(logging.DEBUG, logger="src.controllers.dicom_controller"):
        DicomController(object(), _Win()).open_folder()
    logged = " ".join(r.getMessage() for r in caplog.records)
    assert "OSError" in logged
    _assert_no_names(logged)


class _Label:
    def __init__(self):
        self.text = ""

    def setText(self, text):
        self.text = text


class _Progress:
    def close(self):
        pass


class _LoadedWindow(_Window):
    def __init__(self, reset):
        super().__init__()
        self.info_label = _Label()
        self._reset = reset

    def reset_workspace(self):
        self._reset()


class _VM:
    def set_volume(self, _image):
        pass


def test_initial_render_is_scheduled_even_when_reset_workspace_raises(monkeypatch):
    scheduled = []
    monkeypatch.setattr(
        "src.controllers.dicom_controller.QTimer.singleShot",
        lambda ms, fn: scheduled.append(fn),
    )
    monkeypatch.setattr(
        "src.controllers.dicom_controller.QMessageBox.critical", lambda *a: None
    )

    def _boom():
        raise RuntimeError("reset failed")

    controller = DicomController(_VM(), _LoadedWindow(_boom))
    controller._on_loaded(object(), {"num_slices": 3}, _Progress())

    assert controller._coordinated_initial_render in scheduled


def test_geometry_warnings_are_shown_without_blocking_the_load(monkeypatch):
    monkeypatch.setattr(
        "src.controllers.dicom_controller.QTimer.singleShot", lambda ms, fn: None
    )
    window = _LoadedWindow(lambda: None)
    controller = DicomController(_VM(), window)
    controller._on_loaded(
        object(),
        {"num_slices": 3, "geometry_warnings": ["Gantry tilt of 20.0\u00b0 detected"]},
        _Progress(),
    )

    assert "Loaded 3 slices" in window.statusbar.message
    assert "Gantry tilt" in window.statusbar.message
    assert "Gantry tilt" in window.info_label.text
