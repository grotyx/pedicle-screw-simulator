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

import src.controllers.segmentation_controller as seg_controller_module
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


# ---------------------------------------------------------------------------
# F4 -- a 60 s heartbeat keeps this instance's workspace off the stale purge
# ---------------------------------------------------------------------------


def test_running_a_segmentation_starts_a_60s_workspace_heartbeat(
    ui_main_window, monkeypatch, tmp_path
):
    """TotalSegmentator emits nothing for the whole subprocess run.

    Without a timer the lock's last heartbeat is the "Running TotalSegmentator"
    message, so a CPU run longer than the purge threshold lets the next
    instance's startup delete the directory it is writing into.
    """
    from src.core.totalseg_integration import SegmentationWorkspace

    window = ui_main_window
    ctrl = window._seg_ctrl
    ctrl.workspace = SegmentationWorkspace(root=str(tmp_path))
    window._on_dicom_loaded(
        image=_create_test_image(),
        metadata={"series_id": "SERIES-HEARTBEAT", "num_slices": 12},
        progress=_ProgressStub(),
    )

    class _NoopThread:
        def __init__(self, **_kwargs):
            self.progress = _Signal()
            self.finished = _Signal()
            self.error = _Signal()
            self.cancelled = _Signal()

        def start(self):
            return None

        def isRunning(self):
            return False

        def request_cancel(self):
            return None

    monkeypatch.setattr(
        seg_controller_module, "AutoSegmentationThread", _NoopThread
    )

    try:
        ctrl.run()

        timer = ctrl._heartbeat_timer
        assert timer is not None and timer.isActive()
        assert timer.interval() == 60_000
        assert ctrl.HEARTBEAT_INTERVAL_MS == 60_000
    finally:
        ctrl._segmentation_thread = None
        ctrl._close_progress_dialog()


class _Signal:
    """Minimal signal stand-in: connect only, never emitted in these tests."""

    def connect(self, _slot):
        return None


def test_heartbeat_refreshes_the_lock_of_every_live_workspace(
    ui_main_window, tmp_path
):
    from pathlib import Path

    from src.core.totalseg_integration import SegmentationWorkspace

    ctrl = ui_main_window._seg_ctrl
    ctrl.workspace = SegmentationWorkspace(root=str(tmp_path))
    created = ctrl.workspace.create()
    lock = Path(created) / SegmentationWorkspace.LOCK_NAME
    old = os.path.getmtime(lock) - 500
    os.utime(lock, (old, old))

    ctrl._start_heartbeat()
    try:
        ctrl._on_heartbeat()
        assert os.path.getmtime(lock) > old
        assert ctrl._heartbeat_timer.isActive()
    finally:
        ctrl._stop_heartbeat()


def test_heartbeat_stops_once_no_workspace_is_left(ui_main_window, tmp_path):
    from src.core.totalseg_integration import SegmentationWorkspace

    ctrl = ui_main_window._seg_ctrl
    ctrl.workspace = SegmentationWorkspace(root=str(tmp_path))
    ctrl.workspace.create()
    ctrl._start_heartbeat()

    ctrl.workspace.purge()
    ctrl._on_heartbeat()

    assert ctrl._heartbeat_timer is None or not ctrl._heartbeat_timer.isActive()


def test_reset_state_stops_the_heartbeat(ui_main_window, tmp_path):
    from src.core.totalseg_integration import SegmentationWorkspace

    ctrl = ui_main_window._seg_ctrl
    ctrl.workspace = SegmentationWorkspace(root=str(tmp_path))
    ctrl.workspace.create()
    ctrl._start_heartbeat()

    ctrl.reset_state()

    assert ctrl._heartbeat_timer is None or not ctrl._heartbeat_timer.isActive()


# ---------------------------------------------------------------------------
# F7 -- the threshold fallback must re-grade too, and commit mask+method together
# ---------------------------------------------------------------------------


def _finish_run(window, ctrl, mask_path, method, monkeypatch):
    """Drive `_on_finished` for one completed run with the dialogs silenced."""
    monkeypatch.setattr(
        seg_controller_module.QMessageBox, "warning", lambda *a, **k: None
    )
    ctrl._on_finished(
        SegmentationRunResult(
            success=True,
            method=method,
            mask_path=str(mask_path),
            message="ok",
        )
    )


def test_threshold_fallback_regrades_the_existing_screws(
    ui_main_window, tmp_path, monkeypatch
):
    """The viewers just swapped in a mask with no vertebra labels.

    Leaving the old grades on screen would present measurements taken against
    a mask that is no longer the one being displayed.
    """
    window = ui_main_window
    ctrl = window._seg_ctrl
    image = _create_test_image()
    window._on_dicom_loaded(
        image=image,
        metadata={"series_id": "SERIES-FALLBACK", "num_slices": image.GetSize()[2]},
        progress=_ProgressStub(),
    )
    mask_path = tmp_path / "fallback_mask.nii.gz"
    _write_mask(image, mask_path)

    regrades = []
    monkeypatch.setattr(
        window._tool_ctrl, "regrade_all", lambda: regrades.append(True)
    )

    _finish_run(window, ctrl, mask_path, "threshold_fallback", monkeypatch)

    assert regrades == [True]
    assert window._tool_ctrl.screw_tool.grader is None


def test_a_failed_overlay_never_leaves_the_method_behind_the_mask_path(
    ui_main_window, tmp_path, monkeypatch
):
    """Mask path and method are one fact; a half-applied result mis-routes planning.

    With the mask path advanced to a threshold run but the method still reading
    "totalsegmentator", `run_planning` would happily plan screws on a mask that
    has no vertebra labels at all.
    """
    window = ui_main_window
    ctrl = window._seg_ctrl
    image = _create_test_image()
    window._on_dicom_loaded(
        image=image,
        metadata={"series_id": "SERIES-PARTIAL", "num_slices": image.GetSize()[2]},
        progress=_ProgressStub(),
    )
    first_mask = tmp_path / "first_mask.nii.gz"
    _write_mask(image, first_mask)
    _finish_run(window, ctrl, first_mask, "totalsegmentator", monkeypatch)
    assert ctrl._last_segmentation_method == "totalsegmentator"

    def _boom(_detected):
        raise RuntimeError("overlay bookkeeping failed")

    monkeypatch.setattr(window, "update_vertebra_level_checks", _boom)
    fallback_mask = tmp_path / "second_mask.nii.gz"
    _write_mask(image, fallback_mask)

    _finish_run(window, ctrl, fallback_mask, "threshold_fallback", monkeypatch)

    if ctrl._last_segmentation_mask_path == str(fallback_mask):
        assert ctrl._last_segmentation_method == "threshold_fallback"
    else:
        assert ctrl._last_segmentation_method == "totalsegmentator"


# ---------------------------------------------------------------------------
# F1 (controller side) -- a mask a fraction of a voxel off the CT still grades
# ---------------------------------------------------------------------------


def test_a_drifted_mask_is_resampled_onto_the_ct_grid_before_grading(
    ui_main_window, tmp_path, monkeypatch
):
    """A float32 NIfTI origin round-trip must not cost every screw its grade."""
    window = ui_main_window
    ctrl = window._seg_ctrl
    image = _create_test_image()
    window._on_dicom_loaded(
        image=image,
        metadata={"series_id": "SERIES-DRIFT", "num_slices": image.GetSize()[2]},
        progress=_ProgressStub(),
    )

    drifted = sitk.Cast(image > 0, sitk.sitkUInt8)
    drifted.CopyInformation(image)
    origin = list(image.GetOrigin())
    origin[2] += 0.4 * image.GetSpacing()[2]      # well past any grid tolerance
    drifted.SetOrigin(origin)
    mask_path = tmp_path / "drifted_mask.nii.gz"
    sitk.WriteImage(drifted, str(mask_path))

    _finish_run(window, ctrl, mask_path, "totalsegmentator", monkeypatch)

    assert window._tool_ctrl.screw_tool.grader is not None


def test_grading_is_skipped_when_no_ct_is_loaded(ui_main_window, tmp_path):
    ctrl = ui_main_window._seg_ctrl
    mask = sitk.Cast(_create_test_image() > 0, sitk.sitkUInt8)
    assert ctrl._build_grader(mask) is None


# ---------------------------------------------------------------------------
# W2 -- the plan file records how the mask behind its screws was produced
# ---------------------------------------------------------------------------


def _refined_result(mask_path, notes, raw_mask_path="raw.nii.gz"):
    return SegmentationRunResult(
        success=True,
        method="totalsegmentator",
        mask_path=str(mask_path),
        message="ok",
        raw_mask_path=raw_mask_path,
        refinement_notes=list(notes),
    )


def test_mask_refinement_metadata_is_empty_before_any_run(ui_main_window):
    ctrl = ui_main_window._seg_ctrl

    assert ctrl.mask_refinement_metadata() == {"enabled": False, "ct_guided": False}


def test_mask_refinement_metadata_reports_a_ct_guided_run(
    ui_main_window, tmp_path, monkeypatch
):
    from src.core.mask_refinement import CT_GUIDED_NOTE

    window = ui_main_window
    ctrl = window._seg_ctrl
    image = _create_test_image()
    window._on_dicom_loaded(
        image=image,
        metadata={"series_id": "SERIES-META", "num_slices": image.GetSize()[2]},
        progress=_ProgressStub(),
    )
    mask_path = tmp_path / "refined.nii.gz"
    _write_mask(image, mask_path)
    monkeypatch.setattr(
        seg_controller_module.QMessageBox, "warning", lambda *a, **k: None
    )

    ctrl._on_finished(_refined_result(mask_path, [CT_GUIDED_NOTE]))

    assert ctrl.mask_refinement_metadata() == {"enabled": True, "ct_guided": True}


def test_mask_refinement_metadata_reports_an_antialias_only_run(
    ui_main_window, tmp_path, monkeypatch
):
    from src.core.mask_refinement import ANTIALIAS_ONLY_NOTE, NO_CT_NOTE

    window = ui_main_window
    ctrl = window._seg_ctrl
    image = _create_test_image()
    window._on_dicom_loaded(
        image=image,
        metadata={"series_id": "SERIES-AA", "num_slices": image.GetSize()[2]},
        progress=_ProgressStub(),
    )
    mask_path = tmp_path / "aa.nii.gz"
    _write_mask(image, mask_path)
    monkeypatch.setattr(
        seg_controller_module.QMessageBox, "warning", lambda *a, **k: None
    )

    ctrl._on_finished(_refined_result(mask_path, [ANTIALIAS_ONLY_NOTE, NO_CT_NOTE]))

    assert ctrl.mask_refinement_metadata() == {"enabled": True, "ct_guided": False}


def test_reset_state_forgets_the_previous_refinement(
    ui_main_window, tmp_path, monkeypatch
):
    from src.core.mask_refinement import CT_GUIDED_NOTE

    window = ui_main_window
    ctrl = window._seg_ctrl
    image = _create_test_image()
    window._on_dicom_loaded(
        image=image,
        metadata={"series_id": "SERIES-RESET", "num_slices": image.GetSize()[2]},
        progress=_ProgressStub(),
    )
    mask_path = tmp_path / "refined.nii.gz"
    _write_mask(image, mask_path)
    monkeypatch.setattr(
        seg_controller_module.QMessageBox, "warning", lambda *a, **k: None
    )
    ctrl._on_finished(_refined_result(mask_path, [CT_GUIDED_NOTE]))

    ctrl.reset_state()

    assert ctrl.mask_refinement_metadata() == {"enabled": False, "ct_guided": False}


# ---------------------------------------------------------------------------
# W2 -- the refinement checkbox, its persistence, and the status text
# ---------------------------------------------------------------------------


def test_the_refine_checkbox_is_on_by_default(ui_main_window):
    window = ui_main_window

    assert window.seg_refine_check.text() == "Refine boundaries against CT"
    assert window.seg_refine_check.isChecked() is True


def test_unchecking_refine_is_persisted_and_restored(ui_main_window):
    window = ui_main_window
    window.seg_refine_check.setChecked(False)

    window.load_segmentation_settings()

    assert window.seg_refine_check.isChecked() is False

    window.seg_refine_check.setChecked(True)
    window.load_segmentation_settings()
    assert window.seg_refine_check.isChecked() is True


def test_the_checkbox_decides_whether_the_worker_refines(
    ui_main_window, monkeypatch, tmp_path
):
    from src.core.totalseg_integration import SegmentationWorkspace

    window = ui_main_window
    ctrl = window._seg_ctrl
    ctrl.workspace = SegmentationWorkspace(root=str(tmp_path))
    image = _create_test_image()
    window._on_dicom_loaded(
        image=image,
        metadata={"series_id": "SERIES-REFINE", "num_slices": image.GetSize()[2]},
        progress=_ProgressStub(),
    )
    captured = {}

    class _CapturingThread:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.progress = _Signal()
            self.finished = _Signal()
            self.error = _Signal()
            self.cancelled = _Signal()

        def start(self):
            return None

        def isRunning(self):
            return False

        def request_cancel(self):
            return None

    monkeypatch.setattr(
        seg_controller_module, "AutoSegmentationThread", _CapturingThread
    )

    window.seg_refine_check.setChecked(False)
    try:
        ctrl.run()
        assert captured["refine"] is False
        ctrl._segmentation_thread = None
        ctrl._close_progress_dialog()

        window.seg_refine_check.setChecked(True)
        ctrl.run()
        assert captured["refine"] is True
    finally:
        ctrl._segmentation_thread = None
        ctrl._close_progress_dialog()
        ctrl._stop_heartbeat()


def test_the_worker_thread_forwards_refine_to_the_core(qtbot, monkeypatch, tmp_path):
    """`qtbot` only guarantees a QApplication exists for the QThread."""
    captured = {}

    def _fake_run(**kwargs):
        captured.update(kwargs)
        return SegmentationRunResult(
            success=True,
            method="totalsegmentator",
            mask_path="mask.nii.gz",
            message="ok",
        )

    monkeypatch.setattr(
        seg_controller_module, "run_segmentation_with_fallback", _fake_run
    )
    thread = seg_controller_module.AutoSegmentationThread(
        sitk_image=_create_test_image(),
        task="total",
        device="cpu",
        work_dir=str(tmp_path),
        refine=False,
    )

    thread.run()

    assert captured["refine"] is False


def _status_after(window, ctrl, tmp_path, monkeypatch, raw_mask_path, notes, name):
    image = _create_test_image()
    window._on_dicom_loaded(
        image=image,
        metadata={"series_id": "SERIES-STATUS", "num_slices": image.GetSize()[2]},
        progress=_ProgressStub(),
    )
    mask_path = tmp_path / name
    _write_mask(image, mask_path)
    monkeypatch.setattr(
        seg_controller_module.QMessageBox, "warning", lambda *a, **k: None
    )
    ctrl._on_finished(
        SegmentationRunResult(
            success=True,
            method="totalsegmentator",
            mask_path=str(mask_path),
            message="ok",
            raw_mask_path=raw_mask_path,
            refinement_notes=list(notes),
        )
    )
    return window.seg_status_label.text()


def test_status_label_names_the_refinement_state(
    ui_main_window, tmp_path, monkeypatch
):
    from src.core.mask_refinement import ANTIALIAS_ONLY_NOTE, CT_GUIDED_NOTE

    window = ui_main_window
    ctrl = window._seg_ctrl

    assert "Refined (CT-guided)" in _status_after(
        window, ctrl, tmp_path, monkeypatch, "raw.nii.gz", [CT_GUIDED_NOTE], "a.nii.gz"
    )
    assert "Refined (anti-alias only)" in _status_after(
        window, ctrl, tmp_path, monkeypatch, "raw.nii.gz",
        [ANTIALIAS_ONLY_NOTE], "b.nii.gz"
    )
    assert "Raw mask" in _status_after(
        window, ctrl, tmp_path, monkeypatch, None, [], "c.nii.gz"
    )


# ---------------------------------------------------------------------------
# Re-segmenting drops the pedicle analyses measured on the previous mask
# ---------------------------------------------------------------------------


def test_finished_drops_the_analyses_measured_on_the_old_mask(
    ui_main_window, tmp_path
):
    """One row may not mix a new mask's grade with an old mask's pedicle width.

    Boundary refinement moves exactly the walls the isthmus width is measured
    between, so re-running segmentation on the same study invalidates every
    analysis.  Keeping them re-graded each screw against the new mask while
    re-deriving its width, narrow verdict and endplate angle from the old one.
    """
    window = ui_main_window
    image = _create_test_image()
    window._on_dicom_loaded(
        image=image,
        metadata={"series_id": "SERIES-REGISTRY", "num_slices": image.GetSize()[2]},
        progress=_ProgressStub(),
    )
    screw_tool = window._tool_ctrl.screw_tool
    screw_tool.set_analysis_by_level({29: object()})
    assert screw_tool._analysis_by_level

    mask_path = tmp_path / "mask_registry.nii.gz"
    _write_mask(image, mask_path)
    window._on_segmentation_finished(
        SegmentationRunResult(
            success=True,
            method="totalsegmentator",
            mask_path=str(mask_path),
            message="ok",
        )
    )

    assert screw_tool._analysis_by_level == {}
