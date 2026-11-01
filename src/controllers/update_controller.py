"""Help > Check for Updates, plus the quiet check shortly after start-up.

Only the installed Windows build updates itself: it is the one that ships a
``components.json`` beside the executable. Changed layers are downloaded in
the background, then swapped in by a PowerShell script once the app closes.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

from PyQt6.QtCore import QThread, QTimer, pyqtSignal
from PyQt6.QtWidgets import QMessageBox, QProgressDialog

from src import __version__
from src.core import app_update

logger = logging.getLogger(__name__)

STARTUP_CHECK_DELAY_MS = 8000
_MB = 1024 * 1024


def installed_app_dir() -> Optional[Path]:
    """The installed Windows app folder, or None for source runs and other platforms."""
    if getattr(sys, "frozen", False) and sys.platform == "win32":
        return Path(sys.executable).resolve().parent
    return None


def staging_root() -> Path:
    base = os.environ.get("LOCALAPPDATA") or str(Path.home())
    return Path(base) / "PedicleScrewSimulator" / "updates"


class UpdateCheckThread(QThread):
    checked = pyqtSignal(object, object)  # UpdatePlan | None, remote manifest
    failed = pyqtSignal(str)

    def __init__(self, local: dict):
        super().__init__()
        self._local = local

    def run(self):
        try:
            remote = app_update.fetch_manifest(app_update.update_url())
            self.checked.emit(app_update.plan_update(self._local, remote), remote)
        except Exception as exc:
            logger.warning("Update check failed: %s", exc)
            self.failed.emit(str(exc))


class UpdateDownloadThread(QThread):
    progress = pyqtSignal(int)  # bytes since the last signal
    staged = pyqtSignal(object)  # staging folder
    failed = pyqtSignal(str)

    def __init__(self, plan, local: dict, remote: dict, staging: Path):
        super().__init__()
        self._plan, self._local, self._remote, self._staging = plan, local, remote, staging
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def run(self):
        try:
            base = app_update.update_url()
            archives = [
                app_update.download_layer(
                    base, self._remote["layers"][name], self._staging / "download",
                    self.progress.emit, lambda: self._cancel,
                )
                for name in self._plan.layers
            ]
            self.staged.emit(app_update.stage_update(
                self._plan, self._local, self._remote, archives, self._staging
            ))
        except Exception as exc:
            logger.warning("Update download failed: %s", exc)
            self.failed.emit(str(exc))


class UpdateController:
    """Checks the server, downloads changed layers and installs them on exit."""

    def __init__(self, main_window, install_dir: Optional[Path] = None):
        self._window = main_window
        self._install_dir = install_dir if install_dir is not None else installed_app_dir()
        self._local = (
            app_update.load_local_manifest(self._install_dir) if self._install_dir else None
        )
        self._thread: Optional[QThread] = None
        self._progress: Optional[QProgressDialog] = None
        self._done = 0
        self.pending_script: Optional[Path] = None

    @property
    def enabled(self) -> bool:
        return self._local is not None and os.environ.get("PSS_NO_UPDATE") != "1"

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.isRunning()

    def schedule_startup_check(self) -> None:
        if self.enabled:
            QTimer.singleShot(STARTUP_CHECK_DELAY_MS, lambda: self.check(silent=True))

    def check(self, silent: bool = False) -> None:
        if self.is_running or self.pending_script is not None:
            return
        if not self.enabled:
            if not silent:
                QMessageBox.information(
                    self._window, "Check for Updates",
                    "Automatic updates are available in the installed Windows app only.",
                )
            return
        thread = UpdateCheckThread(self._local)
        thread.checked.connect(lambda plan, remote: self._on_checked(plan, remote, silent))
        thread.failed.connect(lambda message: self._on_check_failed(message, silent))
        self._start(thread)

    def _start(self, thread: QThread) -> None:
        thread.finished.connect(self._release_thread)
        self._thread = thread
        thread.start()

    def _release_thread(self) -> None:
        if self._thread is not None and not self._thread.isRunning():
            self._thread.deleteLater()
            self._thread = None

    def _on_check_failed(self, message: str, silent: bool) -> None:
        if not silent:
            QMessageBox.warning(self._window, "Check for Updates", message)

    def _on_checked(self, plan, remote: dict, silent: bool) -> None:
        if plan is None:
            if not silent:
                QMessageBox.information(
                    self._window, "Check for Updates",
                    f"{app_update.PRODUCT} {__version__} is the latest version.",
                )
            return
        megabytes = max(1, round(plan.download_bytes / _MB))
        answer = QMessageBox.question(
            self._window, "Update Available",
            f"{app_update.PRODUCT} {plan.version} is available (you have {__version__}).\n\n"
            f"Download {megabytes} MB now? Only the parts that changed "
            f"({', '.join(plan.layers)}) are downloaded; it installs when you close the app.",
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        self._download(plan, remote, megabytes)

    def _download(self, plan, remote: dict, megabytes: int) -> None:
        thread = UpdateDownloadThread(plan, self._local, remote, staging_root() / plan.version)
        progress = QProgressDialog("Downloading the update...", "Cancel", 0, megabytes, self._window)
        progress.setWindowTitle("Update")
        progress.setMinimumDuration(0)
        progress.canceled.connect(thread.cancel)
        self._progress, self._done = progress, 0
        thread.progress.connect(self._on_progress)
        thread.staged.connect(self._on_staged)
        thread.failed.connect(self._on_download_failed)
        self._start(thread)

    def _on_progress(self, delta: int) -> None:
        self._done += delta
        if self._progress is not None:
            self._progress.setValue(min(self._progress.maximum(), self._done // _MB))

    def _close_progress(self) -> None:
        if self._progress is not None:
            self._progress.close()
            self._progress = None

    def _on_download_failed(self, message: str) -> None:
        self._close_progress()
        QMessageBox.warning(self._window, "Update", f"The update was not installed.\n\n{message}")

    def _on_staged(self, staging: Path) -> None:
        self._close_progress()
        self.pending_script = app_update.write_apply_script(
            staging, self._install_dir, os.getpid(), Path(sys.executable)
        )
        answer = QMessageBox.question(
            self._window, "Update Ready",
            "The update is downloaded. Restart now to install it?\n\n"
            "Choose No to install it when you next close the app.",
        )
        if answer == QMessageBox.StandardButton.Yes:
            self._window.close()
        else:
            self._window.statusbar.showMessage("The update installs when you close the app")

    def launch_pending(self) -> None:
        """Start the apply script; called as the main window closes."""
        if self.pending_script is None:
            return
        flags = getattr(subprocess, "DETACHED_PROCESS", 0) | getattr(
            subprocess, "CREATE_NEW_PROCESS_GROUP", 0
        )
        subprocess.Popen(
            ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass",
             "-WindowStyle", "Hidden", "-File", str(self.pending_script)],
            creationflags=flags, close_fds=True,
        )
        self.pending_script = None
