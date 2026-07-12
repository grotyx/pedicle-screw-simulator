"""DICOM loading controller.

Extracted from MainWindow to reduce God Class complexity.
Handles DICOM folder scanning, series selection, threaded loading,
and coordinated initial rendering after volume load.
"""

import time
import logging
from PyQt6.QtWidgets import (
    QFileDialog,
    QMessageBox,
    QProgressDialog,
    QInputDialog,
)
from PyQt6.QtCore import QThread, QTimer, pyqtSignal, Qt
from typing import Optional

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
            image = loader.load_series(self.directory, series_id=target_series)

            metadata = loader.get_metadata()
            metadata["series_id"] = target_series
            self.progress.emit("Complete")
            self.finished.emit(image, metadata)

        except Exception as e:
            import traceback
            error_msg = f"{str(e)}\n\n{traceback.format_exc()}"
            print(f"DICOM Load Error: {error_msg}")  # Console output for debugging
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

        # Show progress dialog
        progress = QProgressDialog("Loading DICOM...", None, 0, 0, self._window)
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(0)
        progress.show()

        # Load in thread
        self._load_thread = DicomLoadThread(folder, series_id=selected_series_id)
        self._load_thread.finished.connect(
            lambda img, meta: self._on_loaded(img, meta, progress)
        )
        self._load_thread.error.connect(
            lambda err: self._on_error(err, progress)
        )
        self._load_thread.progress.connect(progress.setLabelText)
        self._load_thread.start()

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
        progress.close()

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

            # Update info label
            series_id = metadata.get("series_id", "Unknown")
            if isinstance(series_id, str) and len(series_id) > 36:
                series_id = series_id[:33] + "..."
            info_text = (
                f"Patient: {metadata.get('patient_name', 'Unknown')}\n"
                f"Study Date: {metadata.get('study_date', 'Unknown')}\n"
                f"Modality: {metadata.get('modality', 'Unknown')}\n"
                f"Series: {series_id}\n"
                f"Size: {metadata.get('size', 'Unknown')}\n"
                f"Spacing: {metadata.get('spacing', 'Unknown')}\n"
                f"Slices: {metadata.get('num_slices', 'Unknown')}"
            )
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
        progress.close()
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
