"""Auto pedicle screw placement controller.

Orchestrates segmentation mask -> pedicle analysis -> immediately editable
screw trajectories.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, TYPE_CHECKING

from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtWidgets import QMessageBox

from src.core.pedicle_analyzer import PedicleAnalyzer
from src.core.auto_screw_planner import AutoScrewPlanner, PlannedScrew
from src.core.planner_config import PlannerConfig
from src.models.screw import Screw

if TYPE_CHECKING:
    from src.core.volume_manager import VolumeManager

logger = logging.getLogger(__name__)


class _PlanningThread(QThread):
    """Background thread for pedicle analysis + screw planning."""

    finished = pyqtSignal(list)   # List[PlannedScrew]
    error = pyqtSignal(str)
    progress = pyqtSignal(str)

    def __init__(
        self,
        mask_image,
        ct_image,
        labels: List[int],
        config: Optional[PlannerConfig] = None,
    ):
        super().__init__()
        self._mask = mask_image
        self._ct = ct_image
        self._labels = labels
        self._config = config

    def run(self):
        try:
            self.progress.emit("Analyzing vertebral pedicles...")
            analyzer = PedicleAnalyzer(self._mask, self._ct)
            analyses = analyzer.analyze_all(labels=self._labels)

            successful = [a for a in analyses if a.success]
            self.progress.emit(
                f"Analyzed {len(analyses)} vertebrae, "
                f"{len(successful)} with pedicle data. Planning screws..."
            )

            planner = AutoScrewPlanner(self._ct, self._mask, config=self._config)
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
        self._last_planned: List[PlannedScrew] = []

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.isRunning()

    @property
    def last_planned(self) -> List[PlannedScrew]:
        """Read-only copy of the trajectories produced by the last run."""
        return list(self._last_planned)

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

        if getattr(seg_ctrl, "_last_segmentation_method", None) != "totalsegmentator":
            QMessageBox.warning(
                self._window,
                "Planning Unavailable",
                "Automatic planning needs a TotalSegmentator vertebra mask. "
                "The current mask is a threshold fallback without vertebra labels.",
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

        self._thread = _PlanningThread(
            mask_image,
            ct_image,
            selected_labels,
            config=self._window.planner_config(),
        )
        self._thread.progress.connect(self._on_progress)
        self._thread.finished.connect(self._on_finished)
        self._thread.error.connect(self._on_error)
        self._thread.start()

    def reset_state(self):
        """Reset controller state for a new DICOM load."""
        self._last_planned = []
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

        self._last_planned = list(planned)

        if not planned:
            self._window.auto_screw_status.setText(
                "Planning complete: no valid trajectories found"
            )
            self._window.statusbar.showMessage(
                "Auto planning: no valid screw trajectories"
            )
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
        metrics=dict(ps.metrics),
    )
