"""Plan save/load/export controller.

Extracted from MainWindow to reduce God Class complexity.
Handles plan serialization (JSON), CSV export, STL export,
and applying loaded plans back to the tool state.
"""

import logging
import os
from datetime import datetime
from typing import List, Optional

from PyQt6.QtCore import QStandardPaths
from PyQt6.QtWidgets import QApplication, QFileDialog, QMessageBox

from src.controllers.tool_controller import screw_display_color
from src.core.plan_metrics import (
    _pedicle_centers,
    compare_screw_lists,
    screw_geometry_problem,
)
from src.utils.planning_io import (
    deserialize_plan,
    export_screws_csv,
    load_plan_json,
    save_plan_json,
    serialize_plan,
    series_uid_digest,
    write_text_atomic,
)
from src.utils.report_export import ReportContext, build_report_html, capture_png
from src.utils.stl_export import export_bone_stl

logger = logging.getLogger(__name__)

#: QSettings key holding the folder the user last saved or loaded a plan in.
_PLAN_DIR_KEY = "plan/last_dir"


class PlanController:
    """Manages plan save, load, and export operations."""

    def __init__(self, volume_manager, main_window):
        self._vm = volume_manager
        self._window = main_window
        #: The loaded reference plan, kept apart from the editable plan: never
        #: graded, saved, exported or selectable.  ``None`` when there is none.
        self._reference: Optional[dict] = None
        self.reference_dialog = None

    def _start_dir(self) -> str:
        """Folder file dialogs open in: last plan folder, else Documents.

        Never the working directory, which is the git checkout when the app
        is started from the run scripts; plans are patient-identifiable.
        """
        last = self._window._settings.value(_PLAN_DIR_KEY)
        if isinstance(last, str) and os.path.isdir(last):
            return last
        return QStandardPaths.writableLocation(
            QStandardPaths.StandardLocation.DocumentsLocation
        )

    def _remember_dir(self, path: str) -> None:
        self._window._settings.setValue(_PLAN_DIR_KEY, os.path.dirname(path))

    def save_dialog(self):
        """Save current screw/measurement plan to JSON."""
        path, _ = QFileDialog.getSaveFileName(
            self._window,
            "Save Plan JSON",
            self._start_dir(),
            "JSON Files (*.json)",
        )
        if not path:
            return
        self._remember_dir(path)

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
                measurement_screw_aligned=self._build_measurement_screw_aligned(
                    tool_ctrl._measurement_entries, len(measurements)
                ),
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
            self._start_dir(),
            "JSON Files (*.json)",
        )
        if not path:
            return
        self._remember_dir(path)

        try:
            payload = load_plan_json(path)
            parsed = deserialize_plan(payload)
            plan_digest = parsed.get("series_uid_sha256")
            current_digest = series_uid_digest(self._window._current_series_id)
            if plan_digest and current_digest and plan_digest != current_digest:
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
                screw_aligned=parsed["measurement_screw_aligned"],
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
            self._start_dir(),
            "CSV Files (*.csv)",
        )
        if not path:
            return
        self._remember_dir(path)

        try:
            export_screws_csv(path, screws)
            self._window.statusbar.showMessage(f"Exported screws CSV: {path}")
        except Exception as e:
            QMessageBox.critical(
                self._window, "Export Error", f"Failed to export CSV: {e}"
            )

    def export_report_dialog(self):
        """Write a self-contained HTML planning report (no patient identifiers)."""
        vtk_image = self._vm.get_vtk_image()
        if vtk_image is None:
            QMessageBox.warning(
                self._window,
                "No Volume Loaded",
                "Load a DICOM volume before exporting a report.",
            )
            return

        path, _ = QFileDialog.getSaveFileName(
            self._window,
            "Export Planning Report",
            self._start_dir(),
            "HTML Files (*.html)",
        )
        if not path:
            return
        self._remember_dir(path)

        win = self._window
        images = {}
        # The report lists only the working plan, so its 3D image must not
        # show the reference ghosts or the rotation guide (a control) either.
        hide_reference = self._reference is not None and self._reference["visible"]
        if hide_reference:
            win.viewer_3d.set_reference_screws_visible(False)
        win.viewer_3d.set_guides_visible(False)
        try:
            for title, viewer in (
                ("Axial", win.axial_viewer),
                ("Sagittal", win.sagittal_viewer),
                ("Coronal", win.coronal_viewer),
                ("3D", win.viewer_3d),
            ):
                widget = getattr(viewer, "vtk_widget", None)
                images[title] = (
                    capture_png(widget.GetRenderWindow()) if widget is not None else None
                )
        finally:
            win.viewer_3d.set_guides_visible(True)
            if hide_reference:
                win.viewer_3d.set_reference_screws_visible(True)
        try:
            html = build_report_html(
                ReportContext(
                    screws=win._tool_ctrl.screw_tool.get_screws(),
                    planner=win.planner_config(),
                    spacing=tuple(vtk_image.GetSpacing()),
                    size=tuple(vtk_image.GetDimensions()),
                    images=images,
                )
            )
            write_text_atomic(path, html)
            win.statusbar.showMessage(f"Exported planning report: {path}")
        except Exception as e:
            QMessageBox.critical(
                win, "Export Error", f"Failed to export report: {e}"
            )

    # --- reference plan ----------------------------------------------------

    def load_reference_dialog(self):
        """Open a plan JSON as the reference to compare the working plan with."""
        if self._vm.get_vtk_image() is None:
            QMessageBox.warning(
                self._window,
                "No Volume Loaded",
                "Load a DICOM volume before loading a reference plan.",
            )
            return
        path, _ = QFileDialog.getOpenFileName(
            self._window,
            "Load Reference Plan JSON",
            self._start_dir(),
            "JSON Files (*.json)",
        )
        if not path:
            return
        self._remember_dir(path)
        try:
            payload = load_plan_json(path)
            parsed = deserialize_plan(payload)
        except Exception as e:
            QMessageBox.critical(
                self._window, "Load Error", f"Failed to load reference plan: {e}"
            )
            return

        plan_digest = parsed.get("series_uid_sha256")
        current_digest = series_uid_digest(self._window._current_series_id)
        if plan_digest and current_digest and plan_digest != current_digest:
            # Unlike loading a plan to edit, there is no "continue anyway":
            # numbers from another series' coordinates would look meaningful.
            QMessageBox.warning(
                self._window,
                "Different Series",
                "The reference plan was made on a different CT series and "
                "cannot be compared with this one.",
            )
            return

        screws = parsed["screws"]
        self._reference = {
            "screws": screws,
            "centers": self._reference_centers(payload, len(screws)),
            "frame_verified": bool(plan_digest and current_digest),
            "visible": True,
        }
        self._window.viewer_3d.set_reference_screws(
            [
                (screw.entry_point, screw.target_point, screw.diameter / 2.0)
                for screw in screws
                if screw_geometry_problem(screw) is None
            ]
        )
        self._window.viewer_3d.set_reference_screws_visible(True)
        self._window.refresh_reference_controls(True)
        self._window.statusbar.showMessage(
            f"Reference plan loaded ({len(screws)} screws)"
        )
        self.compare_reference()

    @staticmethod
    def _reference_centers(payload, count: int):
        centers = _pedicle_centers(payload)
        return centers if len(centers) == count else [None] * count

    def compare_reference(self):
        """Compare the working plan as it is now and show the result."""
        if self._reference is None:
            return
        from src.ui.reference_comparison_dialog import ReferenceComparisonDialog

        current = self._window._tool_ctrl.screw_tool.get_screws()
        reference = self._reference["screws"]
        result = compare_screw_lists(
            current, reference, reference_pedicle_centers=self._reference["centers"]
        )
        if self.reference_dialog is None:
            self.reference_dialog = ReferenceComparisonDialog(
                self._window, on_recompare=self.compare_reference
            )
        self.reference_dialog.set_result(
            result,
            current,
            reference,
            self._reference["frame_verified"],
            computed_at=datetime.now().strftime("%H:%M:%S"),
        )
        self.reference_dialog.show()
        self.reference_dialog.raise_()

    def set_reference_visible(self, visible: bool) -> None:
        if self._reference is None:
            return
        self._reference["visible"] = bool(visible)
        self._window.viewer_3d.set_reference_screws_visible(visible)

    def clear_reference(self):
        """Drop the reference plan, its ghosts and its comparison window."""
        self._reference = None
        if self.reference_dialog is not None:
            self.reference_dialog.close()
        self._window.viewer_3d.set_reference_screws([])
        self._window.refresh_reference_controls(False)

    def export_stl_dialog(self):
        """Export bone surface as STL file."""
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
            self._start_dir(),
            "STL Files (*.stl)",
        )
        if not path:
            return
        self._remember_dir(path)

        try:
            self._window.statusbar.showMessage("Exporting STL...")
            QApplication.processEvents()
            export_bone_stl(vtk_image, path)
            self._window.statusbar.showMessage(f"STL exported: {path}")
        except Exception as e:
            QMessageBox.critical(
                self._window, "Export Error", f"Failed to export STL: {e}"
            )

    def _apply_loaded_plan(self, screws, measurements, planes, screw_aligned=None):
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

        aligned = screw_aligned or [False] * len(measurements)
        for measurement, plane, is_aligned in zip(
            measurements, planes, aligned, strict=True
        ):
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
                    "screw_aligned": bool(is_aligned),
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

    @staticmethod
    def _build_measurement_screw_aligned(entries, count: int) -> List[bool]:
        """Per-measurement screw-view flags, False where there is no entry."""
        flags = [False] * count
        for index, entry in enumerate(entries[:count]):
            flags[index] = bool(entry.get("screw_aligned"))
        return flags
