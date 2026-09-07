"""Auto pedicle screw placement controller.

Orchestrates segmentation mask -> pedicle analysis -> immediately editable
screw trajectories.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, TYPE_CHECKING

from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtWidgets import QMessageBox

from src.core.pedicle_analyzer import PedicleAnalyzer, VERTEBRA_LABELS
from src.core.auto_screw_planner import AutoScrewPlanner, PlannedScrew
from src.core.vertebra import PedicleAnalysisResult
from src.models.screw import Screw

if TYPE_CHECKING:
    from src.core.volume_manager import VolumeManager

logger = logging.getLogger(__name__)

# Grade -> (R, G, B) float color mapping for 3D preview.
GRADE_COLORS: Dict[str, tuple] = {
    "A": (0.2, 0.8, 0.2),   # Green
    "B": (0.9, 0.9, 0.1),   # Yellow
    "C": (1.0, 0.5, 0.0),   # Orange
    "D": (1.0, 0.2, 0.2),   # Red
    "E": (0.6, 0.0, 0.0),   # Dark red
}


class _PlanningThread(QThread):
    """Background thread for pedicle analysis + screw planning."""

    finished = pyqtSignal(list)   # List[PlannedScrew]
    error = pyqtSignal(str)
    progress = pyqtSignal(str)

    def __init__(self, mask_image, ct_image, labels: List[int]):
        super().__init__()
        self._mask = mask_image
        self._ct = ct_image
        self._labels = labels

    def run(self):
        try:
            import SimpleITK as sitk

            self.progress.emit("Analyzing vertebral pedicles...")
            analyzer = PedicleAnalyzer(self._mask, self._ct)
            analyses = analyzer.analyze_all(labels=self._labels)

            successful = [a for a in analyses if a.success]
            self.progress.emit(
                f"Analyzed {len(analyses)} vertebrae, "
                f"{len(successful)} with pedicle data. Planning screws..."
            )

            planner = AutoScrewPlanner(self._ct, self._mask)
            planned = planner.plan_all(analyses, sides="both")

            self.progress.emit(
                f"Planned {len(planned)} screw trajectories."
            )
            self.finished.emit(planned)
        except Exception as exc:
            import traceback
            logger.error("Planning failed: %s\n%s", exc, traceback.format_exc())
            self.error.emit(str(exc))


class AutoPlacementController:
    """Manages the auto pedicle screw planning workflow."""

    def __init__(self, volume_manager: VolumeManager, main_window):
        self._vm = volume_manager
        self._window = main_window
        self._thread: Optional[_PlanningThread] = None
        self._planned_screws: List[PlannedScrew] = []
        self._preview_actors: list = []  # VTK actors for 3D preview
        self._preview_overlay_ids: List[int] = []
        self._next_preview_overlay_id = 10000

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.isRunning()

    @property
    def planned_screws(self) -> List[PlannedScrew]:
        return list(self._planned_screws)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_planning(self):
        """Execute auto pedicle screw planning for selected vertebrae."""
        seg_ctrl = self._window._seg_ctrl
        mask_path = seg_ctrl._last_segmentation_mask_path
        if mask_path is None:
            QMessageBox.warning(
                self._window,
                "No Segmentation",
                "Run auto segmentation first to generate a vertebral mask.",
            )
            return

        ct_image = self._vm.get_sitk_image()
        if ct_image is None:
            QMessageBox.warning(
                self._window,
                "No CT Volume",
                "Load a DICOM volume before planning screws.",
            )
            return

        if self.is_running:
            self._window.statusbar.showMessage(
                "Auto screw planning is already running"
            )
            return

        # Gather selected labels from UI checkboxes.
        selected_labels = self._get_selected_labels()
        if not selected_labels:
            QMessageBox.warning(
                self._window,
                "No Vertebrae Selected",
                "Select at least one vertebra for auto screw planning.",
            )
            return

        if not seg_ctrl.ensure_mpr_vertebrae_isolated():
            self._window.statusbar.showMessage(
                "Unable to display segmented CT for screw planning"
            )
            return

        import SimpleITK as sitk
        mask_image = sitk.ReadImage(mask_path)

        self._window.auto_screw_plan_btn.setEnabled(False)
        self._window.auto_screw_status.setText("Planning...")
        self._window.statusbar.showMessage("Auto screw planning started")

        self._thread = _PlanningThread(mask_image, ct_image, selected_labels)
        self._thread.progress.connect(self._on_progress)
        self._thread.finished.connect(self._on_finished)
        self._thread.error.connect(self._on_error)
        self._thread.start()

    def accept_all(self):
        """Accept all planned screws and convert to manual Screw objects."""
        if not self._planned_screws:
            return
        self._accept_screws(list(range(len(self._planned_screws))))

    def accept_selected(self):
        """Accept only the selected rows in the result table."""
        table = self._window.auto_screw_table
        selected_rows = sorted({idx.row() for idx in table.selectedIndexes()})
        if not selected_rows:
            self._window.statusbar.showMessage(
                "Select rows in the table to accept"
            )
            return
        self._accept_screws(selected_rows)

    def reject_selected(self):
        """Remove selected planned screws from the preview."""
        table = self._window.auto_screw_table
        selected_rows = sorted(
            {idx.row() for idx in table.selectedIndexes()}, reverse=True
        )
        if not selected_rows:
            self._window.statusbar.showMessage(
                "Select rows in the table to reject"
            )
            return

        for row in selected_rows:
            if 0 <= row < len(self._planned_screws):
                self._remove_preview_at(row)
                self._planned_screws.pop(row)

        self._refresh_table()
        self._window.statusbar.showMessage("Selected screws rejected")

    def clear_plan(self):
        """Clear all planned screws and preview actors."""
        self._clear_preview()
        self._planned_screws.clear()
        self._refresh_table()
        self._window.auto_screw_status.setText("No auto plan")
        self._window.statusbar.showMessage("Auto screw plan cleared")

    def reset_state(self):
        """Reset controller state for a new DICOM load."""
        self._clear_preview()
        self._planned_screws.clear()
        if hasattr(self._window, 'auto_screw_table'):
            self._refresh_table()
        if hasattr(self._window, 'auto_screw_status'):
            self._window.auto_screw_status.setText("No auto plan")

    # ------------------------------------------------------------------
    # Private: planning callbacks
    # ------------------------------------------------------------------

    def _on_progress(self, message: str):
        self._window.auto_screw_status.setText(message)
        self._window.statusbar.showMessage(message)

    def _on_finished(self, planned: List[PlannedScrew]):
        self._thread = None
        self._refresh_plan_button()

        self._clear_preview()
        self._planned_screws = []

        if not planned:
            self._window.auto_screw_status.setText(
                "Planning complete: no valid trajectories found"
            )
            self._window.statusbar.showMessage(
                "Auto planning: no valid screw trajectories"
            )
            self._refresh_table()
            return

        first_new_index = len(
            self._window._tool_ctrl.screw_tool.get_screws()
        )
        for offset, planned_screw in enumerate(planned):
            self._window._tool_ctrl.add_existing_screw(
                planned_screw_to_screw(planned_screw),
                select=offset == 0,
            )
        self._window.screw_list_widget.setCurrentRow(first_new_index)
        self._refresh_table()

        grade_counts: Dict[str, int] = {}
        for ps in planned:
            grade_counts[ps.gertzbein_grade] = (
                grade_counts.get(ps.gertzbein_grade, 0) + 1
            )
        grade_summary = ", ".join(
            f"{g}:{n}" for g, n in sorted(grade_counts.items())
        )
        status = (
            f"Added {len(planned)} editable screws. "
            f"Grades: {grade_summary}. Select a screw and drag it directly."
        )
        self._window.auto_screw_status.setText(status)
        self._window.statusbar.showMessage(status)

    def _on_error(self, error: str):
        self._thread = None
        self._refresh_plan_button()
        self._window.auto_screw_status.setText("Planning failed")
        QMessageBox.critical(
            self._window,
            "Auto Planning Error",
            f"Auto screw planning failed: {error}",
        )

    # ------------------------------------------------------------------
    # Private: helpers
    # ------------------------------------------------------------------

    def _get_selected_labels(self) -> List[int]:
        """Return labels shared by 3D vertebra visibility and planning."""
        labels = []
        checkboxes = getattr(self._window, '_vertebra_level_checks', {})
        for label_id, checkbox in checkboxes.items():
            if checkbox.isChecked():
                labels.append(label_id)
        return sorted(labels)

    def _refresh_plan_button(self) -> None:
        """Restore Plan button state from shared visible vertebra selection."""
        updater = getattr(
            self._window,
            "_on_vertebra_level_selection_changed",
            None,
        )
        if callable(updater):
            updater(apply_to_view=False)
        else:
            self._window.auto_screw_plan_btn.setEnabled(True)

    def _accept_screws(self, rows: List[int]):
        """Convert planned screws at given rows to manual screws."""
        tool_ctrl = self._window._tool_ctrl
        accepted = []

        for row in sorted(rows, reverse=True):
            if row < 0 or row >= len(self._planned_screws):
                continue
            ps = self._planned_screws[row]
            screw = planned_screw_to_screw(ps)
            tool_ctrl.screw_tool.add_screw(screw)
            tool_ctrl._add_screw_to_list(screw)

            self._remove_preview_at(row)

            # Add permanent screw actor.
            entry = tuple(ps.entry_lps.tolist())
            target = tuple(ps.target_lps.tolist())
            actor = self._window.viewer_3d.add_screw(
                entry, target,
                radius=ps.diameter_mm / 2.0,
            )
            tool_ctrl._screw_actors.append(actor)

            # Add permanent screw overlay to MPR viewers
            screw_id = len(tool_ctrl.screw_tool.get_screws()) - 1
            tool_ctrl._add_screw_to_mpr(screw_id, screw)

            accepted.append(ps.vertebra_name)
            self._planned_screws.pop(row)

        self._refresh_table()
        self._window.statusbar.showMessage(
            f"Accepted screws for: {', '.join(accepted)}"
        )

    def _allocate_preview_overlay_id(self) -> int:
        """Return a stable overlay ID that is never derived from a table row."""
        preview_id = self._next_preview_overlay_id
        self._next_preview_overlay_id += 1
        return preview_id

    def _remove_preview_at(self, row: int) -> None:
        """Remove one preview from both 3D and MPR viewers."""
        if 0 <= row < len(self._preview_actors):
            actor = self._preview_actors.pop(row)
            if actor is not None and self._window.viewer_3d is not None:
                self._window.viewer_3d.remove_screw(actor)

        if 0 <= row < len(self._preview_overlay_ids):
            preview_id = self._preview_overlay_ids.pop(row)
            for viewer in self._window._get_mpr_viewers():
                viewer.remove_screw_overlay(preview_id)

    def _clear_preview(self):
        """Remove all preview actors from 3D and 2D viewers."""
        if self._window.viewer_3d is not None:
            for actor in self._preview_actors:
                if actor is not None:
                    self._window.viewer_3d.remove_screw(actor)
        self._preview_actors.clear()

        for preview_id in self._preview_overlay_ids:
            for viewer in self._window._get_mpr_viewers():
                viewer.remove_screw_overlay(preview_id)
        self._preview_overlay_ids.clear()

    def _refresh_table(self):
        """Rebuild the result table from current planned screws."""
        table = self._window.auto_screw_table
        table.setRowCount(len(self._planned_screws))

        for row, ps in enumerate(self._planned_screws):
            from PyQt6.QtWidgets import QTableWidgetItem
            table.setItem(row, 0, QTableWidgetItem(ps.vertebra_name))
            table.setItem(row, 1, QTableWidgetItem(ps.side.capitalize()))
            table.setItem(row, 2, QTableWidgetItem(f"{ps.length_mm:.1f}"))
            table.setItem(row, 3, QTableWidgetItem(f"{ps.diameter_mm:.1f}"))
            table.setItem(row, 4, QTableWidgetItem(ps.gertzbein_grade))
            table.setItem(
                row, 5, QTableWidgetItem(f"{ps.confidence:.0%}")
            )


def planned_screw_to_screw(ps: PlannedScrew) -> Screw:
    """Convert a PlannedScrew from the auto planner to a manual Screw model."""
    return Screw(
        entry_point=tuple(ps.entry_lps.tolist()),
        target_point=tuple(ps.target_lps.tolist()),
        diameter=ps.diameter_mm,
        vertebra_level=ps.vertebra_name,
        side=ps.side,
        grade=ps.gertzbein_grade,
        breach_distance=float(ps.breach_mm),
        mean_hu=float(ps.mean_bone_density),
        min_hu=float(ps.min_bone_density),
        warnings=list(ps.warnings),
        source="auto",
    )
