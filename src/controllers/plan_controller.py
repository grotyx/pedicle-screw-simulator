"""Plan save/load/export controller.

Extracted from MainWindow to reduce God Class complexity.
Handles plan serialization (JSON), CSV export, STL export,
and applying loaded plans back to the tool state.
"""

import logging
from typing import List, Optional

from PyQt6.QtWidgets import QApplication, QFileDialog, QMessageBox

from src.controllers.tool_controller import screw_display_color
from src.utils.planning_io import (
    deserialize_plan,
    export_screws_csv,
    load_plan_json,
    save_plan_json,
    serialize_plan,
)
from src.utils.stl_export import export_bone_stl

logger = logging.getLogger(__name__)


class PlanController:
    """Manages plan save, load, and export operations."""

    def __init__(self, volume_manager, main_window):
        self._vm = volume_manager
        self._window = main_window

    def save_dialog(self):
        """Save current screw/measurement plan to JSON."""
        path, _ = QFileDialog.getSaveFileName(
            self._window,
            "Save Plan JSON",
            "",
            "JSON Files (*.json)",
        )
        if not path:
            return

        tool_ctrl = self._window._tool_ctrl
        screws = tool_ctrl.screw_tool.get_screws()
        measurements = tool_ctrl.measurement_tool.get_measurements()
        planes = self._build_measurement_planes(
            tool_ctrl._measurement_entries, len(measurements)
        )

        try:
            payload = serialize_plan(
                series_id=self._window._current_series_id,
                screws=screws,
                measurements=measurements,
                measurement_planes=planes,
                metadata={
                    "mask_refinement": (
                        self._window._seg_ctrl.mask_refinement_metadata()
                    )
                },
            )
            save_plan_json(path, payload)
            self._window.statusbar.showMessage(
                f"Plan saved ({len(screws)} screws, "
                f"{len(measurements)} measurements)"
            )
        except Exception as e:
            QMessageBox.critical(
                self._window, "Save Error", f"Failed to save plan: {e}"
            )

    def load_dialog(self):
        """Load screw/measurement plan from JSON and render overlays."""
        if self._vm.get_vtk_image() is None:
            QMessageBox.warning(
                self._window,
                "No Volume Loaded",
                "Load a DICOM volume before loading a plan.",
            )
            return

        path, _ = QFileDialog.getOpenFileName(
            self._window,
            "Load Plan JSON",
            "",
            "JSON Files (*.json)",
        )
        if not path:
            return

        try:
            payload = load_plan_json(path)
            parsed = deserialize_plan(payload)
            plan_series_id = parsed.get("series_id")
            if (
                isinstance(plan_series_id, str)
                and isinstance(self._window._current_series_id, str)
                and plan_series_id
                and self._window._current_series_id
                and plan_series_id != self._window._current_series_id
            ):
                answer = QMessageBox.question(
                    self._window,
                    "Series Mismatch",
                    "Plan series ID differs from current DICOM series. "
                    "Continue loading?",
                    QMessageBox.StandardButton.Yes
                    | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.No,
                )
                if answer != QMessageBox.StandardButton.Yes:
                    self._window.statusbar.showMessage("Plan load canceled")
                    return

            self._apply_loaded_plan(
                screws=parsed["screws"],
                measurements=parsed["measurements"],
                planes=parsed["measurement_planes"],
            )
            self._window.statusbar.showMessage(
                f"Plan loaded ({len(parsed['screws'])} screws, "
                f"{len(parsed['measurements'])} measurements)"
            )
        except Exception as e:
            QMessageBox.critical(
                self._window, "Load Error", f"Failed to load plan: {e}"
            )

    def export_csv_dialog(self):
        """Export current screws to CSV file."""
        screws = self._window._tool_ctrl.screw_tool.get_screws()
        if not screws:
            self._window.statusbar.showMessage(
                "No screws available for export"
            )
            return

        path, _ = QFileDialog.getSaveFileName(
            self._window,
            "Export Screws CSV",
            "",
            "CSV Files (*.csv)",
        )
        if not path:
            return

        try:
            export_screws_csv(path, screws)
            self._window.statusbar.showMessage(f"Exported screws CSV: {path}")
        except Exception as e:
            QMessageBox.critical(
                self._window, "Export Error", f"Failed to export CSV: {e}"
            )

    def export_stl_dialog(self):
        """Export bone surface as STL file."""
        from src.ui.job_dialog import JobDialog

        vtk_image = self._vm.get_vtk_image()
        if vtk_image is None:
            QMessageBox.warning(
                self._window,
                "No Volume Loaded",
                "Load a DICOM volume before exporting STL.",
            )
            return

        path, _ = QFileDialog.getSaveFileName(
            self._window,
            "Export Bone STL",
            "",
            "STL Files (*.stl)",
        )
        if not path:
            return

        dialog = JobDialog(
            "Exporting STL...",
            self._window,
            cancellable=False,
        )
        dialog.set_log(path)
        dialog.show()
        try:
            self._window.statusbar.showMessage("Exporting STL...")
            QApplication.processEvents()
            export_bone_stl(vtk_image, path)
            self._window.statusbar.showMessage(f"STL exported: {path}")
        except Exception as e:
            QMessageBox.critical(
                self._window, "Export Error", f"Failed to export STL: {e}"
            )
        finally:
            dialog.close_cleanly()

    def _apply_loaded_plan(self, screws, measurements, planes):
        """Replace current tool data with loaded plan contents."""
        tool_ctrl = self._window._tool_ctrl
        tool_ctrl.clear_screws()
        tool_ctrl.clear_measurements()
        tool_ctrl.screw_tool.cancel()
        tool_ctrl.measurement_tool.cancel()
        tool_ctrl._active_measure_plane = None

        for screw_index, screw in enumerate(screws):
            tool_ctrl.screw_tool.add_screw(screw)
            actor = self._window.viewer_3d.add_screw(
                screw.entry_point,
                screw.target_point,
                radius=screw.diameter / 2.0,
                color=screw_display_color(screw),
                screw_id=screw_index,
            )
            tool_ctrl._screw_actors.append(actor)
            tool_ctrl._add_screw_to_list(screw)
            tool_ctrl._add_screw_to_mpr(screw_index, screw)

        for measurement, plane in zip(measurements, planes, strict=True):
            tool_ctrl.measurement_tool.add_measurement(measurement)
            measurement_id = tool_ctrl._next_measurement_id
            tool_ctrl._next_measurement_id += 1
            label = (
                measurement.label
                or tool_ctrl._format_measurement_result(measurement)
            )
            tool_ctrl._measurement_entries.append(
                {
                    "id": measurement_id,
                    "plane": plane,
                    "mode": measurement.mode,
                    "label": label,
                }
            )
            viewer = tool_ctrl._get_viewer_by_plane(plane)
            if viewer is not None:
                viewer.add_measurement(
                    measurement_id=measurement_id,
                    points=measurement.points,
                    label=label,
                )
            self._window.viewer_3d.add_measurement(
                measurement_id=measurement_id,
                points=measurement.points,
                label=label,
            )

        tool_ctrl._refresh_measurement_list()

        # Plans saved by an older HU-heuristic grader (or saved before a
        # grader was ever attached) carry stale grades. Re-grade now that
        # the screws are attached to the tool controller so the inspector
        # and overlays reflect the current segmentation-based grader.
        #
        # Unconditionally: with no grader attached there is nothing behind a
        # stored "A", and `regrade_all` marks such screws N/A. Skipping the
        # call in that case is exactly the case where the saved grade is least
        # trustworthy, and it was being displayed as current.
        tool_ctrl.regrade_all()

        # Match automatic planning: land on the first screw so the Review
        # header, inspector and 3D highlight show it instead of "No screws".
        if screws:
            self._window.screw_list_widget.setCurrentRow(0)

    @staticmethod
    def _build_measurement_planes(entries, count: int) -> List[Optional[str]]:
        """Build per-measurement plane list aligned to measurement order."""
        planes: List[Optional[str]] = [None] * count
        for index, entry in enumerate(entries[:count]):
            planes[index] = entry.get("plane")
        return planes
