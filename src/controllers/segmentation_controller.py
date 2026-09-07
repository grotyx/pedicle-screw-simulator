"""Segmentation controller.

Extracted from MainWindow to reduce God Class complexity.
Handles TotalSegmentator auto-segmentation workflow: running the
background thread, applying results to viewers, label filtering,
and visibility toggling.
"""

import logging
import sys
import traceback
from PyQt6.QtWidgets import QMessageBox, QProgressDialog
from PyQt6.QtCore import QThread, pyqtSignal, Qt
from typing import Dict, Optional

from src.core.totalseg_integration import (
    SPINE_ROI_SUBSET,
    ProcessHolder,
    SegmentationCancelled,
    SegmentationWorkspace,
    preferred_segmentation_device,
    run_segmentation_with_fallback,
)
from src.core.totalseg_labels import (
    format_label_display,
    format_selected_label_text,
    get_segmentation_label_map,
)

logger = logging.getLogger(__name__)


class AutoSegmentationThread(QThread):
    """Thread for running auto-segmentation without blocking UI."""

    finished = pyqtSignal(object)  # SegmentationRunResult
    error = pyqtSignal(str)
    progress = pyqtSignal(str)
    cancelled = pyqtSignal()

    def __init__(self, sitk_image, task: str, device: str, work_dir: str,
                 roi_subset=None, fast=False, force_split=False):
        super().__init__()
        self.sitk_image = sitk_image
        self.task = task
        self.device = device
        self.work_dir = work_dir
        self.roi_subset = roi_subset
        self.fast = fast
        self.force_split = force_split
        self._holder = ProcessHolder()

    def request_cancel(self) -> None:
        """Terminate the running segmentation subprocess (GUI thread safe)."""
        self._holder.terminate()

    def _on_progress(self, message: str) -> None:
        """Refresh the workspace heartbeat lock, then forward progress."""
        SegmentationWorkspace.touch(self.work_dir)
        self.progress.emit(message)

    def run(self):
        try:
            result = run_segmentation_with_fallback(
                image=self.sitk_image,
                work_dir=self.work_dir,
                task=self.task,
                device=self.device,
                roi_subset=self.roi_subset,
                fast=self.fast,
                force_split=self.force_split,
                progress_callback=self._on_progress,
                process_holder=self._holder,
            )
            self.progress.emit("Segmentation completed.")
            self.finished.emit(result)
        except SegmentationCancelled:
            logger.info("Auto segmentation cancelled by user")
            self.cancelled.emit()
            return
        except Exception as e:
            logger.error("Auto segmentation failed: %s", e)
            logger.debug("Traceback:\n%s", traceback.format_exc())
            self.error.emit(str(e))


class SegmentationController:
    """Manages auto-segmentation workflow and overlay rendering."""

    def __init__(self, volume_manager, main_window):
        self._vm = volume_manager
        self._window = main_window
        self.workspace = SegmentationWorkspace()
        self._segmentation_thread: Optional[AutoSegmentationThread] = None
        self._segmentation_progress: Optional[QProgressDialog] = None
        self._last_segmentation_mask_path: Optional[str] = None
        self._last_segmentation_method: str = "totalsegmentator"
        self._segmentation_label_map: Dict[int, str] = {}
        self._updating_segmentation_label_ui = False
        self._vertebrae_isolated: bool = False
        self._last_vtk_mask = None
        self._detected_vertebra_labels: list[int] = []

    @property
    def is_running(self) -> bool:
        """Return True if a segmentation thread is currently active."""
        return (
            self._segmentation_thread is not None
            and self._segmentation_thread.isRunning()
        )

    def run(self):
        """Run TotalSegmentator with fallback in background thread."""
        if self._vm.get_vtk_image() is None:
            QMessageBox.warning(
                self._window,
                "No Volume Loaded",
                "Load a DICOM volume before running segmentation.",
            )
            return

        if self.is_running:
            self._window.statusbar.showMessage(
                "Auto segmentation is already running"
            )
            return

        sitk_image = self._vm.get_sitk_image()
        if sitk_image is None:
            QMessageBox.warning(
                self._window,
                "No Volume Loaded",
                "SimpleITK volume is not available.",
            )
            return

        ui_task = self._window.seg_task_combo.currentData()
        device = preferred_segmentation_device()
        device_index = self._window.seg_device_combo.findData(device)
        if device_index >= 0:
            self._window.seg_device_combo.setCurrentIndex(device_index)
        fast = self._window.seg_quality_combo.currentData() == "fast"
        force_split = device == "cpu" and not fast

        # "spine_only" uses the 'total' task with roi_subset filter
        if ui_task == "spine_only":
            task = "total"
            roi_subset = list(SPINE_ROI_SUBSET)
        else:
            task = ui_task
            roi_subset = None

        self._segmentation_progress = QProgressDialog(
            "Preparing auto segmentation...",
            "Cancel",
            0,
            0,
            self._window,
        )
        self._segmentation_progress.setWindowModality(
            Qt.WindowModality.WindowModal
        )
        self._segmentation_progress.setMinimumDuration(0)
        self._segmentation_progress.canceled.connect(self._on_cancel_requested)
        if getattr(sys, "frozen", False):
            # The packaged build runs nnU-Net in-process and cannot interrupt it.
            self._segmentation_progress.setCancelButton(None)
            self._window.seg_status_label.setToolTip(
                "Cancellation is not available in the packaged build"
            )
        self._segmentation_progress.show()

        self._window.seg_run_btn.setEnabled(False)
        self._window.seg_status_label.setText("Running segmentation...")
        self._window.statusbar.showMessage("Auto segmentation started")

        work_dir = self.workspace.create()
        self._segmentation_thread = AutoSegmentationThread(
            sitk_image=sitk_image,
            task=task,
            device=device,
            work_dir=work_dir,
            roi_subset=roi_subset,
            fast=fast,
            force_split=force_split,
        )
        self._segmentation_thread.progress.connect(self._on_progress)
        self._segmentation_thread.finished.connect(self._on_finished)
        self._segmentation_thread.error.connect(self._on_error)
        self._segmentation_thread.cancelled.connect(self._on_cancelled)
        self._segmentation_thread.start()

    def _on_cancel_requested(self):
        """Ask the running segmentation thread to terminate its subprocess."""
        if self._segmentation_thread is not None:
            self._window.seg_status_label.setText("Cancelling segmentation...")
            self._segmentation_thread.request_cancel()

    def _on_cancelled(self):
        """Reset UI state after the segmentation run was cancelled."""
        if self._segmentation_progress is not None:
            self._segmentation_progress.close()
            self._segmentation_progress = None
        self._window.seg_run_btn.setEnabled(True)
        self._segmentation_thread = None
        self.workspace.purge()
        self._window.seg_status_label.setText("Segmentation cancelled")
        self._window.statusbar.showMessage("Auto segmentation cancelled")

    def _on_progress(self, message: str):
        """Update UI while segmentation is running."""
        if self._segmentation_progress is not None:
            self._segmentation_progress.setLabelText(message)
        self._window.statusbar.showMessage(message)

    def _on_finished(self, result):
        """Handle completed segmentation run."""
        if self._segmentation_progress is not None:
            self._segmentation_progress.close()
            self._segmentation_progress = None
        self._window.seg_run_btn.setEnabled(True)
        self._segmentation_thread = None

        logger.info(
            "_on_finished: method=%s, mask=%s, msg=%s",
            result.method, result.mask_path, result.message,
        )

        try:
            from src.utils.vtk_helpers import sitk_to_vtk
            import SimpleITK as sitk
            import numpy as np

            mask_image = sitk.ReadImage(result.mask_path)
            mask_arr = sitk.GetArrayFromImage(mask_image)
            unique_labels = np.unique(mask_arr)
            logger.info(
                "_on_finished: mask size=%s, spacing=%s, dtype=%s, "
                "unique_labels=%s (count=%d)",
                mask_image.GetSize(),
                mask_image.GetSpacing(),
                mask_arr.dtype,
                unique_labels[:20].tolist(),
                len(unique_labels),
            )

            from src.core.vertebral_mesh import VERTEBRA_LABELS
            detected = sorted(
                int(label) for label in unique_labels if int(label) in VERTEBRA_LABELS
            )
            del mask_arr, unique_labels

            vtk_mask = sitk_to_vtk(mask_image)
            self._last_vtk_mask = vtk_mask
            self._detected_vertebra_labels = detected
            label_value = int(self._window.seg_label_spin.value())
            logger.info("_on_finished: label_value=%d", label_value)
            for viewer in self._window._get_mpr_viewers():
                viewer.set_segmentation_mask(vtk_mask, label_value=label_value)
            self._window.viewer_3d.set_segmentation_mask(
                vtk_mask, label_value=label_value
            )
            self._window.viewer_3d.set_vertebral_mesh(vtk_mask, labels=detected)
            self._last_segmentation_mask_path = result.mask_path
            from src.core.screw_grading import ScrewGrader
            if result.method == "totalsegmentator":
                try:
                    grader = ScrewGrader(mask_image, self._vm.get_sitk_image())
                except ValueError:
                    logger.warning(
                        "Mask and CT grids differ; manual screws will not be graded.",
                        exc_info=True,
                    )
                    grader = None
                self._window._tool_ctrl.screw_tool.set_grader(grader)
                self._window._tool_ctrl.regrade_all()
            else:
                self._window._tool_ctrl.screw_tool.set_grader(None)
            self._window.update_vertebra_level_checks(detected)

            self.update_visibility()
            self._last_segmentation_method = result.method
            self.refresh_label_options(method_override=result.method)

            if result.method != "totalsegmentator":
                self._window.auto_screw_plan_btn.setEnabled(False)
                self._window.seg_status_label.setText(
                    "Threshold mask ready · planning unavailable"
                )
            else:
                self._window.seg_status_label.setText(
                    f"Segmentation ready · {len(detected)} vertebrae detected"
                )
            self._window.vertebra_isolate_btn.setText("Isolate Vertebrae")
            self._window.vertebra_isolate_btn.setEnabled(True)
            self._window.statusbar.showMessage(result.message)

            if result.geometry_warnings:
                warning_text = "\n".join(
                    f"- {item}" for item in result.geometry_warnings[:5]
                )
                if len(result.geometry_warnings) > 5:
                    warning_text += (
                        f"\n- ... (+{len(result.geometry_warnings) - 5} more)"
                    )
                QMessageBox.warning(
                    self._window,
                    "Geometry Warning",
                    "Segmentation mask geometry does not fully match "
                    "the input volume.\n\n"
                    f"{warning_text}",
                )

            if result.method != "totalsegmentator":
                QMessageBox.warning(
                    self._window,
                    "Fallback Used",
                    result.message,
                )
        except Exception as e:
            self._window.seg_status_label.setText(
                "Segmentation completed, but overlay rendering failed."
            )
            QMessageBox.warning(
                self._window,
                "Overlay Error",
                f"Segmentation output was generated but could not be "
                f"rendered: {e}",
            )

    def _on_error(self, error: str):
        """Handle segmentation failure."""
        if self._segmentation_progress is not None:
            self._segmentation_progress.close()
            self._segmentation_progress = None
        self._window.seg_run_btn.setEnabled(True)
        self._segmentation_thread = None
        self._window.seg_status_label.setText("Segmentation failed")
        QMessageBox.critical(
            self._window,
            "Segmentation Error",
            f"Failed to run auto segmentation: {error}",
        )
        self._window.statusbar.showMessage("Auto segmentation failed")

    def isolate_vertebrae(self) -> bool:
        """Isolate vertebral bodies: mask MPR views and hide 3D volume."""
        if self._last_segmentation_mask_path is None:
            QMessageBox.warning(
                self._window,
                "No Segmentation",
                "Run auto segmentation first.",
            )
            return False

        try:
            from src.utils.vtk_helpers import sitk_to_vtk
            import SimpleITK as sitk

            mask_image = sitk.ReadImage(self._last_segmentation_mask_path)
            vtk_mask = sitk_to_vtk(mask_image)

            self._vm.set_vertebral_mask(vtk_mask)
            vertebral_only = self._vm.get_vertebral_only_image()
            if vertebral_only is None:
                return False

            for viewer in self._window._get_mpr_viewers():
                viewer.set_reslice_input(vertebral_only)

            if self._window.viewer_3d is not None:
                self._window.viewer_3d.set_volume_visible(False)

            # Hide segmentation overlay (redundant when masked)
            for viewer in self._window._get_mpr_viewers():
                viewer.set_segmentation_visible(False)

            self._vertebrae_isolated = True
            self._window.vertebra_isolate_btn.setText("Restore Full Volume")
            self._window.vertebra_isolate_btn.setEnabled(True)
            self._window.vertebra_restore_btn.setEnabled(True)
            self._window.statusbar.showMessage("Vertebral bodies isolated")
            return True
        except Exception as e:
            logger.exception("Failed to isolate vertebrae")
            QMessageBox.warning(
                self._window, "Error", f"Failed to isolate vertebrae: {e}"
            )
            return False

    def ensure_mpr_vertebrae_isolated(self) -> bool:
        """Ensure MPR viewers use the segmentation-masked CT volume."""
        if self._vertebrae_isolated:
            return True
        return self.isolate_vertebrae()

    def restore_full_volume(self):
        """Restore original CT volume in all views."""
        for viewer in self._window._get_mpr_viewers():
            viewer.restore_original_input()

        if self._window.viewer_3d is not None:
            self._window.viewer_3d.set_volume_visible(True)

        # Restore segmentation visibility based on checkbox state
        self.update_visibility()

        self._vm.clear_vertebral_mask()
        self._vertebrae_isolated = False
        self._window.vertebra_isolate_btn.setText("Isolate Vertebrae")
        self._window.vertebra_isolate_btn.setEnabled(
            self._last_segmentation_mask_path is not None
        )
        self._window.vertebra_restore_btn.setEnabled(False)
        self._window.statusbar.showMessage("Full volume restored")

    def toggle_vertebrae_isolation(self) -> None:
        """Toggle the single visible isolation button between both states."""
        if self._vertebrae_isolated:
            self.restore_full_volume()
        else:
            self.isolate_vertebrae()

    def show_selected_vertebrae(self) -> None:
        """Show only the selected segmented vertebrae in the 3D viewport."""
        if self._last_vtk_mask is None:
            self._window.statusbar.showMessage("Run segmentation first")
            return
        selected_labels = sorted(
            int(item.data(Qt.ItemDataRole.UserRole))
            for item in self._window.vertebra_display_list.selectedItems()
        )
        if not selected_labels:
            self._window.statusbar.showMessage(
                "Select at least one vertebra for 3D display"
            )
            return
        self._show_vertebral_mesh_only(selected_labels)

    def show_all_vertebrae(self) -> None:
        """Show all detected segmented vertebrae without the CT volume."""
        labels = list(self._detected_vertebra_labels)
        if not labels:
            labels = [
                int(
                    self._window.vertebra_display_list.item(index).data(
                        Qt.ItemDataRole.UserRole
                    )
                )
                for index in range(self._window.vertebra_display_list.count())
            ]
        if self._last_vtk_mask is None or not labels:
            self._window.statusbar.showMessage("Run segmentation first")
            return
        self._show_vertebral_mesh_only(labels)

    def show_vertebrae_by_labels(self, labels: list[int]) -> None:
        """Apply the compact 3D vertebra multi-selection immediately."""
        if self._last_vtk_mask is None:
            self._window.statusbar.showMessage("Run segmentation first")
            return
        if labels:
            self._show_vertebral_mesh_only(sorted(set(labels)))
            return

        self._window.viewer_3d.clear_vertebral_mesh()
        self._window.viewer_3d.set_volume_visible(True)
        self._window.statusbar.showMessage("No 3D vertebrae selected")

    def _show_vertebral_mesh_only(self, labels: list[int]) -> None:
        self._window.viewer_3d.set_vertebral_mesh(
            self._last_vtk_mask,
            labels=sorted(labels),
        )
        self._window.viewer_3d.set_volume_visible(False)
        names = [
            self._window.vertebra_display_list.item(index).text()
            for index in range(self._window.vertebra_display_list.count())
            if int(
                self._window.vertebra_display_list.item(index).data(
                    Qt.ItemDataRole.UserRole
                )
            ) in labels
        ]
        self._window.statusbar.showMessage(
            f"3D vertebrae: {', '.join(names)}"
        )

    def clear_overlay(self):
        """Clear segmentation overlays from 2D and 3D viewers."""
        if self._vertebrae_isolated:
            self.restore_full_volume()
        for viewer in self._window._get_mpr_viewers():
            viewer.clear_segmentation_mask()
        if self._window.viewer_3d is None:
            return
        self._window.viewer_3d.clear_vertebral_mesh()
        self._window.viewer_3d.clear_segmentation_mask()
        self._last_segmentation_mask_path = None
        self._window.clear_vertebra_display_options()
        self._window.seg_status_label.setText("Ready for automatic segmentation")
        self._window.vertebra_isolate_btn.setText("Isolate Vertebrae")
        self._window.vertebra_isolate_btn.setEnabled(False)
        self._last_segmentation_method = "totalsegmentator"
        self._last_vtk_mask = None
        self._detected_vertebra_labels = []
        self.refresh_label_options()
        self._window.statusbar.showMessage("Segmentation overlay cleared")

    def apply_label_filter(self, _value=None):
        """Apply label filter to existing segmentation overlays."""
        label_value = int(self._window.seg_label_spin.value())
        self._sync_label_ui(label_value)
        if self._last_segmentation_mask_path is None:
            return
        for viewer in self._window._get_mpr_viewers():
            viewer.set_segmentation_label(label_value)
        if self._window.viewer_3d is not None:
            self._window.viewer_3d.set_segmentation_label(label_value)
        self.update_visibility()

    def on_task_changed(self, _index=None):
        """Refresh label-name map when task changes."""
        self.refresh_label_options()

    def on_label_combo_changed(self, _index=None):
        """Set numeric label ID from selected anatomy name."""
        if self._updating_segmentation_label_ui:
            return

        selected_label = self._window.seg_label_combo.currentData()
        if selected_label is None:
            selected_label = 0

        self._window.seg_label_spin.setValue(int(selected_label))

    def refresh_label_options(self, method_override: Optional[str] = None):
        """Rebuild label-name dropdown for current task/method."""
        ui_task = str(self._window.seg_task_combo.currentData() or "total")
        task = "total" if ui_task == "spine_only" else ui_task
        method = method_override or self._last_segmentation_method
        self._segmentation_label_map = get_segmentation_label_map(
            task=task, method=method
        )

        self._updating_segmentation_label_ui = True
        try:
            self._window.seg_label_combo.clear()
            self._window.seg_label_combo.addItem("0: all labels", 0)
            for label_id in sorted(self._segmentation_label_map.keys()):
                label_name = self._segmentation_label_map[label_id]
                self._window.seg_label_combo.addItem(
                    format_label_display(label_id, label_name),
                    label_id,
                )
        finally:
            self._updating_segmentation_label_ui = False

        current_label = int(self._window.seg_label_spin.value())
        if (
            current_label > 0
            and current_label not in self._segmentation_label_map
        ):
            self._window.seg_label_spin.setValue(0)
            return
        self._sync_label_ui(current_label)

    def _sync_label_ui(self, label_value: int):
        """Synchronize label spin/combo/info text to one selected label ID."""
        if self._updating_segmentation_label_ui:
            return

        self._updating_segmentation_label_ui = True
        try:
            combo_index = self._window.seg_label_combo.findData(label_value)
            if combo_index >= 0:
                self._window.seg_label_combo.setCurrentIndex(combo_index)
            else:
                self._window.seg_label_combo.setCurrentIndex(0)

            self._window.seg_label_info.setText(
                format_selected_label_text(
                    label_value, self._segmentation_label_map
                )
            )
        finally:
            self._updating_segmentation_label_ui = False

    def update_visibility(self, _state=None):
        """Toggle segmentation overlay visibility in 2D and 3D viewers."""
        visible_2d = self._window.seg_show_2d_check.isChecked()
        visible_3d = self._window.seg_show_3d_check.isChecked()
        for viewer in self._window._get_mpr_viewers():
            viewer.set_segmentation_visible(visible_2d)
        if self._window.viewer_3d is not None:
            self._window.viewer_3d.set_segmentation_visible(visible_3d)

    def reset_state(self):
        """Reset segmentation state for a new DICOM load."""
        if self._vertebrae_isolated:
            self.restore_full_volume()
        self._last_segmentation_mask_path = None
        self._last_segmentation_method = "totalsegmentator"
        self._last_vtk_mask = None
        self._detected_vertebra_labels = []
        if hasattr(self._window, "_tool_ctrl"):
            self._window._tool_ctrl.screw_tool.set_grader(None)
        self._window.clear_vertebra_display_options()
        self._window.seg_status_label.setText("Ready for automatic segmentation")
        self._window.vertebra_isolate_btn.setText("Isolate Vertebrae")
        self._window.vertebra_isolate_btn.setEnabled(False)
        self._window.seg_label_spin.setValue(0)
        self._window.seg_show_2d_check.setChecked(True)
        self._window.seg_show_3d_check.setChecked(True)
        self.refresh_label_options()
        self.workspace.purge()
