"""Segmentation controller.

Extracted from MainWindow to reduce God Class complexity.
Handles TotalSegmentator auto-segmentation workflow: running the
background thread, applying results to viewers, label filtering,
and visibility toggling.
"""

import logging
import os
import sys
import traceback
from typing import Dict, Optional

import numpy as np
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import QApplication, QMessageBox, QProgressDialog

from src.core.subregion_segmentation import (
    find_subregion_model,
    resample_to_reference,
)
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
                 roi_subset=None, fast=False, force_split=False,
                 subregion_model=None):
        super().__init__()
        self.sitk_image = sitk_image
        self.task = task
        self.device = device
        self.work_dir = work_dir
        self.roi_subset = roi_subset
        self.fast = fast
        self.force_split = force_split
        self.subregion_model = subregion_model
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
                subregion_model=self.subregion_model,
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
        self._active_work_dir: Optional[str] = None
        self._last_segmentation_mask_path: Optional[str] = None
        self._last_segmentation_method: str = "totalsegmentator"
        self._segmentation_label_map: Dict[int, str] = {}
        self._updating_segmentation_label_ui = False
        self._vertebrae_isolated: bool = False
        self._last_vtk_mask = None
        self._detected_vertebra_labels: list[int] = []
        # (z, y, x) boolean pedicle mask from the optional subregion model,
        # already resampled onto the vertebra mask grid.
        self._last_pedicle_mask: Optional[np.ndarray] = None
        # Why the requested subregion model could not be resolved. Set before
        # the run starts, so the runner never sees the model and cannot report
        # the reason itself.
        self._pending_subregion_message: str = ""

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
        if getattr(sys, "frozen", False):
            # The packaged build runs nnU-Net in-process and cannot interrupt it.
            self._segmentation_progress.setCancelButton(None)
            self._window.seg_status_label.setToolTip(
                "Cancellation is not available in the packaged build"
            )
        else:
            self._segmentation_progress.canceled.connect(
                self._on_cancel_requested
            )
        self._segmentation_progress.show()

        self._window.seg_run_btn.setEnabled(False)
        self._window.seg_status_label.setText("Running segmentation...")
        self._window.statusbar.showMessage("Auto segmentation started")

        self._pending_subregion_message = ""
        work_dir = self.workspace.create()
        self._active_work_dir = work_dir
        self._segmentation_thread = AutoSegmentationThread(
            sitk_image=sitk_image,
            task=task,
            device=device,
            work_dir=work_dir,
            roi_subset=roi_subset,
            fast=fast,
            force_split=force_split,
            subregion_model=self._resolve_subregion_model(),
        )
        self._segmentation_thread.progress.connect(self._on_progress)
        self._segmentation_thread.finished.connect(self._on_finished)
        self._segmentation_thread.error.connect(self._on_error)
        self._segmentation_thread.cancelled.connect(self._on_cancelled)
        self._segmentation_thread.start()

    def _resolve_subregion_model(self):
        """Locate the optional pedicle subregion model requested in the UI.

        Returns ``None`` when the feature is switched off or no usable model
        is found; the run then proceeds as a plain TotalSegmentator pass. When
        the user did ask for the model, the reason it could not be resolved is
        recorded in ``_pending_subregion_message`` so the finished run can say
        so instead of silently behaving like a plain TotalSegmentator pass.
        """
        if not self._window.seg_use_subregion_check.isChecked():
            return None
        explicit_dir = self._window.seg_subregion_dir_edit.text().strip() or None
        try:
            model = find_subregion_model(explicit_dir)
        except Exception as exc:
            logger.warning(
                "Could not resolve the pedicle subregion model", exc_info=True
            )
            self._pending_subregion_message = (
                f"{self.PEDICLE_FAILURE_PREFIX}{exc}"
            )
            return None
        if model is None:
            searched = (
                explicit_dir
                or os.environ.get("PSS_SUBREGION_MODEL_DIR")
                or "PSS_SUBREGION_MODEL_DIR (unset)"
            )
            self._pending_subregion_message = (
                f"{self.PEDICLE_FAILURE_PREFIX}"
                f"no dataset.json found in {searched}"
            )
        return model

    def _close_progress_dialog(self):
        """Close the progress dialog without re-entering the cancel path.

        ``QProgressDialog.close()`` emits ``canceled()``, so the connection has
        to be dropped first or every normal completion would look like a user
        cancellation and terminate the (already finished) run.
        """
        dialog = self._segmentation_progress
        if dialog is None:
            return
        self._segmentation_progress = None
        try:
            dialog.canceled.disconnect(self._on_cancel_requested)
        except TypeError:
            # Never connected (frozen build); nothing to detach.
            pass
        dialog.close()

    def _on_cancel_requested(self):
        """Ask the running segmentation thread to terminate its subprocess."""
        if self._segmentation_thread is None or self._segmentation_progress is None:
            return
        self._window.seg_status_label.setText("Cancelling segmentation...")
        self._segmentation_thread.request_cancel()

    def _on_cancelled(self):
        """Reset UI state after the segmentation run was cancelled."""
        self._close_progress_dialog()
        self._window.seg_run_btn.setEnabled(True)
        self._segmentation_thread = None
        # Only this run's directory: earlier runs may still own the mask that
        # `_last_segmentation_mask_path` points at.
        self.workspace.remove(self._active_work_dir)
        self._active_work_dir = None
        self._window.seg_status_label.setText("Segmentation cancelled")
        self._window.statusbar.showMessage("Auto segmentation cancelled")

    def _on_progress(self, message: str):
        """Update UI while segmentation is running."""
        if self._segmentation_progress is not None:
            self._segmentation_progress.setLabelText(message)
        self._window.statusbar.showMessage(message)

    def _on_finished(self, result):
        """Handle completed segmentation run."""
        self._close_progress_dialog()
        self._window.seg_run_btn.setEnabled(True)
        self._segmentation_thread = None

        logger.info(
            "_on_finished: method=%s, mask=%s, msg=%s",
            result.method, result.mask_path, result.message,
        )

        try:
            import numpy as np
            import SimpleITK as sitk

            from src.utils.vtk_helpers import sitk_to_vtk

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
            self._window.statusbar.showMessage("Resampling pedicle mask…")
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
            try:
                self._last_pedicle_mask = self._load_pedicle_mask(result, mask_image)
            finally:
                QApplication.restoreOverrideCursor()
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
                self._window.statusbar.showMessage("Re-grading screws…")
                QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
                try:
                    self._window._tool_ctrl.regrade_all()
                finally:
                    QApplication.restoreOverrideCursor()
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
                    + self._pedicle_status_suffix(result)
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

    PEDICLE_FAILURE_PREFIX = "Pedicle model unavailable: "

    def _load_pedicle_mask(self, result, mask_image):
        """Return the stage-2 pedicle mask as a bool array on the mask grid.

        Returns ``None`` when the subregion stage did not run, failed, or its
        output cannot be read: the pedicle mask is an optional refinement and
        must never break the segmentation overlay.
        """
        if not getattr(result, "subregion_mask_path", None):
            return None
        pedicle_label = (result.subregion_labels or {}).get("pedicle")
        if pedicle_label is None:
            logger.warning(
                "Subregion model produced no 'pedicle' label; ignoring its mask"
            )
            return None
        try:
            import SimpleITK as sitk

            subregion = resample_to_reference(
                sitk.ReadImage(result.subregion_mask_path), mask_image
            )
            return sitk.GetArrayFromImage(subregion) == int(pedicle_label)
        except Exception:
            logger.warning("Could not read the pedicle subregion mask", exc_info=True)
            return None

    def _pedicle_status_suffix(self, result) -> str:
        """Status-label tail describing the optional pedicle model's outcome."""
        if self._last_pedicle_mask is not None:
            if self._last_pedicle_mask.any():
                return " · pedicle model used"
            return " · pedicle model ran but found no pedicle voxels"
        message = getattr(result, "subregion_message", "") or ""
        if not message and not getattr(result, "subregion_mask_path", None):
            # The stage never ran because the model itself was unresolvable.
            message = self._pending_subregion_message
        if message:
            reason = message
            if reason.startswith(self.PEDICLE_FAILURE_PREFIX):
                reason = reason[len(self.PEDICLE_FAILURE_PREFIX):]
            return f" · pedicle model unavailable: {reason}"
        if getattr(result, "subregion_mask_path", None):
            # The model ran but its output was unusable (missing label / read error).
            return " · pedicle model unavailable: mask could not be read"
        return ""

    def _on_error(self, error: str):
        """Handle segmentation failure."""
        self._close_progress_dialog()
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
            import SimpleITK as sitk

            from src.utils.vtk_helpers import sitk_to_vtk

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
        self._last_pedicle_mask = None
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
        if (
            self._segmentation_thread is not None
            and self._segmentation_thread.isRunning()
        ):
            self._segmentation_thread.request_cancel()
            if not self._segmentation_thread.wait(5000):
                logger.warning(
                    "Segmentation thread did not stop within 5s while "
                    "resetting state; purging its workspace directory "
                    "anyway (run dir: %s)",
                    self._active_work_dir,
                )
        if self._vertebrae_isolated:
            self.restore_full_volume()
        self._last_segmentation_mask_path = None
        self._last_segmentation_method = "totalsegmentator"
        self._last_pedicle_mask = None
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
