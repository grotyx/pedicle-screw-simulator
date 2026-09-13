"""DICOM loading controller.

Extracted from MainWindow to reduce God Class complexity.
Handles DICOM folder scanning, series selection, threaded loading,
and coordinated initial rendering after volume load.
"""

import logging
import time
from typing import Optional

from PyQt6.QtCore import QThread, QTimer, pyqtSignal
from PyQt6.QtWidgets import (
    QFileDialog,
    QInputDialog,
    QMessageBox,
)

from src.core.dicom_loader import DicomLoader

logger = logging.getLogger(__name__)


class DicomLoadThread(QThread):
    """Thread for loading DICOM data without blocking UI."""

    finished = pyqtSignal(object, dict)  # (image, metadata)
    error = pyqtSignal(str)
    progress = pyqtSignal(str)

    def __init__(self, directory: str, series_id: Optional[str] = None):
        super().__init__()
        self.directory = directory
        self.series_id = series_id
        self._cancel_requested = False

    def request_cancel(self) -> None:
        """Ask the load to stop at the next cooperative checkpoint (GUI safe)."""
        self._cancel_requested = True

    @property
    def cancel_requested(self) -> bool:
        return self._cancel_requested

    def run(self):
        try:
            loader = DicomLoader()

            target_series = self.series_id
            if target_series is None:
                self.progress.emit("Scanning DICOM directory...")
                series_ids = loader.scan_directory(self.directory)
                if not series_ids:
                    self.error.emit("No DICOM series found in directory")
                    return
                summaries = loader.get_series_summaries(self.directory)
                if summaries:
                    target_series = max(
                        summaries,
                        key=lambda item: (
                            item.get("modality") == "CT",
                            int(item.get("num_files", 0)),
                        ),
                    )["series_id"]
                else:
                    target_series = series_ids[0]

            short_id = target_series[:16] + "..." if len(target_series) > 19 else target_series
            self.progress.emit(f"Loading series: {short_id}")
            if self._cancel_requested:
                self.error.emit("DICOM load cancelled")
                return
            image = loader.load_series(self.directory, series_id=target_series)

            if self._cancel_requested:
                self.error.emit("DICOM load cancelled")
                return
            metadata = loader.get_metadata()
            metadata["series_id"] = target_series
            self.progress.emit("Complete")
            self.finished.emit(image, metadata)

        except Exception as e:
            import traceback
            logger.error("DICOM load failed: %s", e)
            logger.debug("Traceback:\n%s", traceback.format_exc())
            self.error.emit(str(e))


class DicomController:
    """Manages DICOM folder loading, series selection, and volume setup."""

    def __init__(self, volume_manager, main_window):
        self._vm = volume_manager
        self._window = main_window
        self._load_thread = None

    @property
    def is_running(self) -> bool:
        return self._load_thread is not None and self._load_thread.isRunning()

    def open_folder(self):
        """Open a DICOM folder dialog and load the data."""
        if self.is_running:
            self._window.statusbar.showMessage("DICOM loading is already running")
            return

        seg_ctrl = getattr(self._window, "_seg_ctrl", None)
        if seg_ctrl is not None and seg_ctrl.is_running:
            self._window.statusbar.showMessage(
                "Cancel or wait for the running segmentation before loading another study"
            )
            return

        folder = QFileDialog.getExistingDirectory(
            self._window,
            "Select DICOM Folder",
            "",
            QFileDialog.Option.ShowDirsOnly,
        )

        if not folder:
            return

        # Scan series for optional selection (multi-series folders)
        try:
            self._window.statusbar.showMessage("Scanning DICOM series...")
            scan_loader = DicomLoader()
            series_ids = scan_loader.scan_directory(folder)
        except Exception as e:
            QMessageBox.critical(
                self._window, "Error", f"Failed to scan DICOM folder: {e}"
            )
            self._window.statusbar.showMessage("Scan failed")
            return

        if not series_ids:
            QMessageBox.warning(
                self._window,
                "No DICOM",
                "No DICOM series found in selected folder.",
            )
            self._window.statusbar.showMessage("No DICOM series found")
            return

        selected_series_id = series_ids[0]
        if len(series_ids) > 1:
            try:
                summaries = scan_loader.get_series_summaries(folder)
            except Exception as exc:
                logger.warning("Failed to read DICOM series summaries: %s", exc)
                summaries = []
            selected_series_id = self._select_series_id(series_ids, summaries)
            if selected_series_id is None:
                self._window.statusbar.showMessage("Load canceled")
                return

        # Show progress dialog. Cancellable now: the load thread checks a
        # cooperative flag between scan and read, so cancelling before the
        # pixel read starts avoids the volume ever reaching the viewers.
        # A cancel that lands mid-read still finishes the read (SimpleITK
        # offers no interrupt) and its result is then discarded.
        # Local import: src.ui.__init__ imports MainWindow, which imports
        # the controllers -- a module-level import here would loop.
        from src.ui.job_dialog import JobDialog

        self._load_thread = DicomLoadThread(folder, series_id=selected_series_id)
        progress = JobDialog(
            "Loading DICOM...",
            self._window,
            on_cancel=self._on_load_cancel_requested,
        )
        progress.set_log(folder)
        self._load_progress = progress
        progress.show()

        # Load in thread
        self._load_thread.finished.connect(
            lambda img, meta: self._on_loaded(img, meta, progress)
        )
        self._load_thread.error.connect(
            lambda err: self._on_error(err, progress)
        )
        self._load_thread.progress.connect(progress.setLabelText)
        self._load_thread.start()

    def _on_load_cancel_requested(self) -> None:
        """Ask the load thread to stop; its late result is discarded."""
        thread = self._load_thread
        dialog = getattr(self, "_load_progress", None)
        if thread is None:
            if dialog is not None:
                dialog.close_cleanly()
                self._load_progress = None
            return
        request = getattr(thread, "request_cancel", None)
        if callable(request):
            request()
        if dialog is not None:
            dialog.mark_cancelling("Cancelling load…")
        self._window.statusbar.showMessage("Cancelling DICOM load…")

    def _select_series_id(self, series_ids, summaries=None):
        """Show a picker dialog for multi-series DICOM folders."""
        summary_by_id = {
            summary.get("series_id"): summary for summary in (summaries or [])
        }
        ordered_ids = sorted(
            series_ids,
            key=lambda series_id: (
                summary_by_id.get(series_id, {}).get("modality") == "CT",
                int(summary_by_id.get(series_id, {}).get("num_files", 0)),
            ),
            reverse=True,
        )
        display_map = {}
        options = []
        for index, series_id in enumerate(ordered_ids, start=1):
            summary = summary_by_id.get(series_id, {})
            modality = summary.get("modality", "Unknown")
            description = summary.get("description", "Unnamed series")
            num_files = int(summary.get("num_files", 0))
            series_number = summary.get("series_number", "?")
            short_id = series_id if len(series_id) <= 24 else series_id[:21] + "..."
            label = (
                f"{index}. {modality} | {num_files} slices | {description} | "
                f"Series {series_number} | {short_id}"
            )
            display_map[label] = series_id
            options.append(label)

        selected_label, ok = QInputDialog.getItem(
            self._window,
            "Select DICOM Series",
            "Series Instance UID:",
            options,
            0,
            False,
        )
        if not ok or not selected_label:
            return None
        return display_map[selected_label]

    def _on_loaded(self, image, metadata, progress):
        """Handle successful DICOM loading."""
        close = getattr(progress, "close_cleanly", None)
        if callable(close):
            close()
        else:  # pragma: no cover - legacy stub dialogs in tests
            progress.close()
        self._load_progress = None

        if getattr(self._load_thread, "cancel_requested", False):
            logger.info("Discarding DICOM result after user cancel")
            self._window.statusbar.showMessage("DICOM load cancelled")
            self._release_load_thread()
            return

        try:
            # Set volume in manager
            t0 = time.perf_counter()
            logger.info("_on_loaded: setting volume...")
            self._vm.set_volume(image)
            logger.info("_on_loaded: set_volume done (%.3fs)", time.perf_counter() - t0)

            logger.info("_on_loaded: clearing previous state...")
            self._window._current_series_id = metadata.get("series_id")
            self._window.reset_workspace()
            logger.info("_on_loaded: state cleared, updating UI...")

            # Update info label: geometry only, never patient identifiers.
            # DicomLoader._extract_metadata strips PHI (name, ID, birth/study
            # dates, accession, institution), so metadata.get('patient_name')
            # is always 'Unknown' -- requesting it here would show a row that
            # can never hold a value.
            series_id = metadata.get("series_id", "Unknown")
            if isinstance(series_id, str) and len(series_id) > 36:
                series_id = series_id[:33] + "..."
            info_text = (
                f"Modality: {metadata.get('modality', 'Unknown')}\n"
                f"Series: {series_id}\n"
                f"Size: {metadata.get('size', 'Unknown')}\n"
                f"Spacing: {metadata.get('spacing', 'Unknown')}\n"
                f"Slices: {metadata.get('num_slices', 'Unknown')}"
            )
            if metadata.get("orientation_normalized"):
                info_text += "\nOrientation: normalised to LPS"
                if metadata.get("resampled"):
                    info_text += " (oblique volume resampled)"
            self._window.info_label.setText(info_text)

            self._window.statusbar.showMessage(
                f"Loaded {metadata.get('num_slices', 0)} slices"
            )

            # Schedule coordinated initial render for all viewers.
            # Individual per-viewer QTimers cause Cocoa drawRect busy-loops
            # on macOS -- see _coordinated_initial_render() docstring.
            QTimer.singleShot(100, self._coordinated_initial_render)

            logger.info("_on_loaded: COMPLETE (%.3fs total)", time.perf_counter() - t0)
        except Exception as e:
            import traceback
            error_msg = f"{str(e)}\n\n{traceback.format_exc()}"
            print(f"Volume Processing Error: {error_msg}")
            QMessageBox.critical(
                self._window,
                "Error",
                f"Failed to process volume: {str(e)}",
            )
        finally:
            self._release_load_thread()

    def _on_error(self, error, progress):
        """Handle DICOM loading error."""
        close = getattr(progress, "close_cleanly", None)
        if callable(close):
            close()
        else:  # pragma: no cover - legacy stub dialogs in tests
            progress.close()
        self._load_progress = None
        if str(error) == "DICOM load cancelled":
            self._window.statusbar.showMessage("DICOM load cancelled")
            self._release_load_thread()
            return
        QMessageBox.critical(
            self._window, "Error", f"Failed to load DICOM: {error}"
        )
        self._window.statusbar.showMessage("Load failed")
        self._release_load_thread()

    def _release_load_thread(self):
        """Release a completed worker without destroying a running QThread."""
        thread = self._load_thread
        if thread is None:
            return
        if thread.isRunning():
            QTimer.singleShot(50, self._release_load_thread)
            return
        thread.deleteLater()
        self._load_thread = None

    def _coordinated_initial_render(self):
        """Render all viewers sequentially in one event loop callback.

        Lifts render guards and renders each viewer. MPR viewers render
        2D slices (fast). The 3D viewer does a coarse Phase 1 render,
        then schedules a fine Phase 2 render via QTimer after 500ms.
        """
        logger.info("_coordinated_initial_render: lifting all render guards")

        mpr_viewers = self._window._get_mpr_viewers()

        # Render MPR viewers first (fast: 2D slices)
        for v in mpr_viewers:
            v._deferred_initial_render()

        # Render 3D viewer Phase 1 (CPU ray cast with coarse sampling).
        # Phase 2 fires via QTimer 500ms later.
        self._window.viewer_3d._deferred_render_phase1()

        logger.info("_coordinated_initial_render: all renders done")
