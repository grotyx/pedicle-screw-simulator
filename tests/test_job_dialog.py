"""JobDialog unit tests: cancel wiring, close, frozen, progress/log."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("pytestqt")

from src.ui.job_dialog import JobDialog, make_job_dialog


def test_cancel_wiring_calls_on_cancel_once_then_disconnects(qapp):
    fired = []
    dialog = JobDialog("job", None, on_cancel=lambda: fired.append(True))
    try:
        dialog.canceled.emit()
        dialog.canceled.emit()
        assert fired == [True]
    finally:
        dialog.deleteLater()


def test_close_cleanly_does_not_reenter_cancel(qapp):
    fired = []
    dialog = JobDialog("job", None, on_cancel=lambda: fired.append(True))
    try:
        dialog.close_cleanly()
        dialog.canceled.emit()
        assert fired == []
    finally:
        dialog.deleteLater()


def test_frozen_no_cancel_branch_has_log_tail_but_no_cancel_button(qapp):
    from PyQt6.QtWidgets import QPushButton

    dialog = JobDialog("job", None, cancellable=False)
    try:
        assert dialog.findChild(object) is not None
        buttons = dialog.findChildren(QPushButton)
        assert all("cancel" not in button.text().lower() for button in buttons)
        dialog.set_log("tail")
        assert dialog._log_label.text() == "tail"
        dialog.canceled.emit()  # no connection: must not raise
        dialog.close_cleanly()
    finally:
        dialog.deleteLater()


def test_progress_and_log_and_cancelling_freeze(qapp):
    dialog = JobDialog("job", None)
    try:
        assert (dialog.minimum(), dialog.maximum()) == (0, 0)
        dialog.set_progress(3, 10)
        assert (dialog.minimum(), dialog.maximum(), dialog.value()) == (0, 10, 3)
        dialog.set_log("L4 left (3/10)")
        assert dialog._log_label.text() == "L4 left (3/10)"
        dialog.set_progress(0, 0)
        assert (dialog.minimum(), dialog.maximum()) == (0, 0)
        assert dialog.is_cancelling is False
        dialog.mark_cancelling("Cancelling…")
        assert dialog.is_cancelling is True
        assert dialog.labelText() == "Cancelling…"
    finally:
        dialog.close_cleanly()
        dialog.deleteLater()


def test_window_modal_flag_sets_window_modality(qapp):
    from PyQt6.QtCore import Qt

    dialog = JobDialog("job", None, window_modal=True)
    try:
        assert (
            dialog.windowModality() == Qt.WindowModality.WindowModal
        )
    finally:
        dialog.close_cleanly()
        dialog.deleteLater()

    dialog = JobDialog("job", None)
    try:
        assert dialog.windowModality() != Qt.WindowModality.WindowModal
    finally:
        dialog.close_cleanly()
        dialog.deleteLater()


def test_make_job_dialog_builds_a_job_dialog(qapp):
    dialog = make_job_dialog("job", None)
    try:
        assert isinstance(dialog, JobDialog)
    finally:
        dialog.close_cleanly()
        dialog.deleteLater()
