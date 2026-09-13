"""Shared background-job dialog: one cancellable progress surface.

DICOM loading, TotalSegmentator segmentation, automatic screw planning,
and STL export used to build their own ``QProgressDialog`` inline, each
with slightly different modality, cancel wiring, and teardown. The three
``_close_progress_dialog`` copies (DICOM implicit, segmentation, planning)
all exist because ``QProgressDialog.close()`` emits ``canceled()``: the
``canceled`` connection must be dropped before closing or every normal
completion looks like a user cancellation.

This module is the single copy of that dance. It does not run the job;
it only owns the dialog surface:

* indeterminate (``0, 0``) or determinate (``0, total``) range,
* an optional one-line log tail under the label,
* a cancel button that degrades to a disabled "Close this window to stop"
  note when cancellation is unavailable (frozen segmentation build),
* teardown that disconnects ``canceled`` before closing.

Jobs keep their own threads and their own cooperative cancel flags; the
dialog only reports. A caller that needs per-level percentages wires its
thread's progress signal to :meth:`set_progress`; callers without a
meaningful percentage (planning cost varies too much per pedicle) stay
indeterminate and only update the label.
"""

from __future__ import annotations

from typing import Callable, Optional

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QLabel, QProgressDialog, QVBoxLayout, QWidget


class JobDialog(QProgressDialog):
    """A cancellable progress dialog with a one-line log tail."""

    def __init__(
        self,
        label: str,
        parent: Optional[QWidget] = None,
        *,
        cancel_text: str = "Cancel",
        cancellable: bool = True,
        on_cancel: Optional[Callable[[], None]] = None,
        minimum_duration_ms: int = 0,
        window_modal: bool = False,
    ) -> None:
        super().__init__(label, cancel_text, 0, 0, parent)
        self.setWindowModality(
            Qt.WindowModality.WindowModal if window_modal else Qt.WindowModality.NonModal
        )
        self.setMinimumDuration(minimum_duration_ms)
        self.setRange(0, 0)
        self._log_label = QLabel("", self)
        self._log_label.setObjectName("jobDialogLog")
        self._log_label.setWordWrap(True)
        layout = self.layout()
        if isinstance(layout, QVBoxLayout):
            layout.addWidget(self._log_label)
        self._on_cancel_callback = on_cancel
        self._cancelling = False
        if not cancellable:
            self.setCancelButton(None)
        elif on_cancel is not None:
            self.canceled.connect(self._handle_canceled)

    def _handle_canceled(self) -> None:
        callback = self._on_cancel_callback
        self._on_cancel_callback = None
        try:
            self.canceled.disconnect(self._handle_canceled)
        except TypeError:  # pragma: no cover - never connected
            pass
        if callback is not None:
            callback()

    def set_log(self, text: str) -> None:
        """Show a one-line tail (source path, level name) under the label."""
        self._log_label.setText(str(text))

    def set_progress(self, value: int, total: int) -> None:
        """Switch to a determinate bar; ``total <= 0`` restores indeterminate."""
        if total > 0:
            self.setRange(0, int(total))
            self.setValue(int(value))
        else:
            self.setRange(0, 0)

    def mark_cancelling(self, text: str = "Cancelling…") -> None:
        """Freeze the label on the cancel note so queued progress cannot repaint it."""
        self._cancelling = True
        self.setLabelText(str(text))

    @property
    def is_cancelling(self) -> bool:
        return self._cancelling

    def close_cleanly(self) -> None:
        """Close without re-entering the cancel path.

        ``QProgressDialog.close()`` emits ``canceled()``; the connection is
        dropped first so a normal completion never looks like a cancellation.
        Mirrors the three per-controller ``_close_progress_dialog`` copies
        this helper replaces.
        """
        self._on_cancel_callback = None
        try:
            self.canceled.disconnect(self._handle_canceled)
        except TypeError:  # pragma: no cover - never connected
            pass
        self.close()


def make_job_dialog(
    label: str,
    parent: Optional[QWidget] = None,
    **kwargs,
) -> JobDialog:
    """Build a :class:`JobDialog`; kept as a function for stub-friendly tests."""
    return JobDialog(label, parent, **kwargs)
