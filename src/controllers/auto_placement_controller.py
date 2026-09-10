"""Auto pedicle screw placement controller.

Orchestrates segmentation mask -> pedicle analysis -> immediately editable
screw trajectories.
"""

from __future__ import annotations

import logging
import threading
from functools import partial
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import QMessageBox, QProgressDialog

from src.core.auto_screw_planner import (
    AutoScrewPlanner,
    PlannedScrew,
    is_width_uncertain_reason,
)
from src.core.pedicle_analyzer import PedicleAnalyzer
from src.core.planner_config import PlannerConfig
from src.models.screw import Screw

if TYPE_CHECKING:
    from src.core.volume_manager import VolumeManager

logger = logging.getLogger(__name__)

#: How many dropped sides the status note names before it starts counting.
_MAX_NAMED_DROPPED_SIDES = 4


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
        pedicle_mask=None,
    ):
        super().__init__()
        self._mask = mask_image
        self._ct = ct_image
        self._labels = labels
        self._config = config
        # Optional (z, y, x) boolean pedicle mask from the subregion model.
        self._pedicle_mask = pedicle_mask
        self._cancel = threading.Event()
        #: ``(vertebra name, side, reason)`` per side the planner dropped.
        self.skipped_sides: List[Tuple[str, str, str]] = []
        #: Whether the planner stopped early on :meth:`request_cancel`.
        self.cancelled = False

    def request_cancel(self) -> None:
        """Ask the planner to stop at the next (level, side) (GUI thread safe)."""
        self._cancel.set()

    def run(self):
        try:
            self.progress.emit("Analyzing vertebral pedicles...")
            analyzer = PedicleAnalyzer(
                self._mask, self._ct, pedicle_mask=self._pedicle_mask
            )
            analyses = analyzer.analyze_all(labels=self._labels)

            successful = [a for a in analyses if a.success]
            self.progress.emit(
                f"Analyzed {len(analyses)} vertebrae, "
                f"{len(successful)} with pedicle data. Planning screws..."
            )

            planner = AutoScrewPlanner(self._ct, self._mask, config=self._config)
            planned = planner.plan_all(
                analyses,
                sides="both",
                progress=self.progress.emit,
                cancel=self._cancel.is_set,
            )
            self.skipped_sides = list(getattr(planner, "skipped_sides", ()) or ())
            self.cancelled = bool(getattr(planner, "last_run_cancelled", False))

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
        self._progress_dialog: Optional[QProgressDialog] = None
        #: Bumped by :meth:`run_planning` and :meth:`reset_state`.  A thread
        #: carries the generation it was started for, so a result that arrives
        #: after the study changed — or after a newer run replaced it — can be
        #: recognised and dropped instead of landing in the current plan.
        self._run_generation = 0
        #: Set once the user has asked to cancel, so a progress message already
        #: queued from the worker cannot overwrite the "Cancelling" label.
        self._cancel_requested = False

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
        try:
            mask_image = sitk.ReadImage(mask_path)
        except Exception as exc:
            # The workspace holding the mask can disappear underneath us — a
            # second instance's startup purge, a temp cleaner, a manual delete.
            # Raising here would surface as an unhandled exception inside a Qt
            # slot, so say what happened and what to do about it.
            logger.warning(
                "Could not read the segmentation mask %s", mask_path, exc_info=True
            )
            QMessageBox.warning(
                self._window,
                "Segmentation Mask Unavailable",
                "The segmentation mask this plan needs could not be read:\n"
                f"{mask_path}\n\n"
                "Its temporary workspace may have been removed. Re-run auto "
                "segmentation before planning screws.\n\n"
                f"Details: {exc}",
            )
            self._window.statusbar.showMessage(
                "Segmentation mask could not be read — re-run auto segmentation"
            )
            return

        self._window.auto_screw_plan_btn.setEnabled(False)
        self._window.auto_screw_status.setText("Planning...")
        self._window.statusbar.showMessage("Auto screw planning started")
        self._cancel_requested = False

        # Modeless: planning is a background job, and the surgeon may want to
        # keep reading the study while it runs.  Indeterminate because the cost
        # of a pedicle varies too much for a percentage to mean anything.
        self._progress_dialog = QProgressDialog(
            "Planning screw trajectories...",
            "Cancel",
            0,
            0,
            self._window,
        )
        self._progress_dialog.setWindowModality(Qt.WindowModality.NonModal)
        self._progress_dialog.setMinimumDuration(0)
        self._progress_dialog.setRange(0, 0)
        self._progress_dialog.canceled.connect(self._on_cancel_requested)
        self._progress_dialog.show()

        # Every run gets its own token: `finished` is emitted (queued) before
        # `isRunning()` goes False, so a second Plan click can install thread B
        # while thread A's result is still in the event queue.  The token is
        # bound into the connection, so each slot knows which run it is being
        # handed a result for instead of trusting `self._thread`.
        self._run_generation += 1
        # The thread being replaced may still be grinding through pedicles —
        # `isRunning()` goes False the moment `run` returns, which is after the
        # planner has finished but before Qt reaps it, and a discarded run must
        # not keep competing with the one that replaced it for the CPU.
        self.request_cancel()
        thread = _PlanningThread(
            mask_image,
            ct_image,
            selected_labels,
            config=self._window.planner_config(),
            pedicle_mask=getattr(seg_ctrl, "_last_pedicle_mask", None),
        )
        thread.generation = self._run_generation
        self._thread = thread
        thread.progress.connect(partial(self._on_progress, thread))
        thread.finished.connect(partial(self._on_finished, thread))
        thread.error.connect(partial(self._on_error, thread))
        thread.start()

    def request_cancel(self) -> None:
        """Ask a running planning run to stop at the next (level, side).

        Public so the main window can call it while closing rather than refusing
        to close, or leaving the user to kill the process.
        """
        if self._thread is not None:
            self._thread.request_cancel()

    def reset_state(self):
        """Reset controller state for a new DICOM load.

        A run still in flight belongs to the study being replaced: it is asked
        to stop, its dialog goes away with the study it was measuring, and the
        generation bump makes its eventual result stale, so it cannot inject the
        previous study's screws into the new plan.
        """
        self._run_generation += 1
        self.request_cancel()
        self._close_progress_dialog()
        self._cancel_requested = False
        self._last_planned = []
        if hasattr(self._window, 'auto_screw_status'):
            self._window.auto_screw_status.setText("No auto plan")

    # ------------------------------------------------------------------
    # Private: planning callbacks
    # ------------------------------------------------------------------

    def _close_progress_dialog(self):
        """Close the planning dialog without re-entering the cancel path.

        ``QProgressDialog.close()`` emits ``canceled()``, so the connection has
        to be dropped first or every normal completion would look like a user
        cancellation.  Mirrors ``SegmentationController._close_progress_dialog``.
        """
        dialog = self._progress_dialog
        if dialog is None:
            return
        self._progress_dialog = None
        try:
            dialog.canceled.disconnect(self._on_cancel_requested)
        except TypeError:   # pragma: no cover - never connected
            pass
        dialog.close()

    def _on_cancel_requested(self):
        """Ask the running planning thread to stop at the next (level, side)."""
        if self._thread is None or self._progress_dialog is None:
            return
        self._cancel_requested = True
        self._window.auto_screw_status.setText("Cancelling planning...")
        self._thread.request_cancel()

    def _on_progress(self, thread, message: str):
        # A superseded run keeps emitting until it notices the cancel; its
        # messages must not paint over the new study's (or the new run's) UI.
        if self._is_stale(thread):
            return
        # Once a cancel is in, the label belongs to it: the worker finishes the
        # pedicle it is on and its already-queued message would otherwise put
        # "Planning L4 right…" back over "Cancelling planning...".  The status
        # bar still tracks the run winding down.
        if not self._cancel_requested:
            self._window.auto_screw_status.setText(message)
        self._window.statusbar.showMessage(message)

    def _is_stale(self, thread) -> bool:
        """Whether ``thread`` is no longer the run this controller is waiting on.

        Both halves matter: the generation catches a run whose study has been
        replaced by :meth:`reset_state`, and the identity check catches a run
        that a newer ``run_planning`` superseded.  An unknown emitter (``None``,
        or a thread that was never stamped) is stale by construction — it can
        never be proved to belong to the current run.
        """
        if thread is not self._thread:
            return True
        return getattr(thread, "generation", None) != self._run_generation

    def _discard_stale(self, thread, what: str) -> None:
        """Drop a result from a superseded run, leaving the new study's UI alone."""
        logger.info("Discarding %s from a superseded planning run", what)
        if thread is self._thread:
            self._thread = None
        self._refresh_plan_button()

    def _on_finished(self, thread, planned: List[PlannedScrew]):
        if self._is_stale(thread):
            self._discard_stale(thread, f"{len(planned)} screws")
            return
        self._close_progress_dialog()
        self._thread = None
        self._refresh_plan_button()

        cancelled = bool(getattr(thread, "cancelled", False))
        skipped = list(getattr(thread, "skipped_sides", None) or ())
        note = _dropped_sides_note(skipped)
        if skipped:
            logger.info("Planner dropped %d side(s): %s", len(skipped), skipped)

        self._last_planned = list(planned)

        if not planned:
            if cancelled:
                status = bar = "Planning cancelled — 0 screws kept"
            else:
                status = "Planning complete: no valid trajectories found"
                bar = "Auto planning: no valid screw trajectories"
            self._window.auto_screw_status.setText(status + note)
            self._window.statusbar.showMessage(bar + note)
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

        if cancelled:
            # A partial construct: report what was kept, not a success summary.
            status = f"Planning cancelled — {len(planned)} screws kept"
        else:
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
            rod = _rod_misalignment_by_side(planned)
            # Only the sides that actually have screws: "R 0.0 mm" for a side
            # with none reads as a perfectly aligned rod that does not exist.
            measured = [
                f"{initial} {rod[side]:.1f} mm"
                for side, initial in (("left", "L"), ("right", "R"))
                if side in rod
            ]
            if measured:
                status += " Rod misalignment " + " / ".join(measured)
        status += note
        self._window.auto_screw_status.setText(status)
        self._window.statusbar.showMessage(status)

    def _on_error(self, thread, error: str):
        if self._is_stale(thread):
            self._discard_stale(thread, f"error {error!r}")
            return
        self._close_progress_dialog()
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


def _dropped_side_text(name: str, side: str, reason: str) -> str:
    """One dropped side, as the status line words it.

    Normally just the level and the side: the reason is in the log and the
    surgeon can see the level for themselves.  A width the analyser could not
    trust is the exception -- "L3 left" alone reads as "that pedicle is too
    small", which is the opposite of what happened -- so that reason is spelled
    out.  It already begins with the side word, so the level name prefixes it
    into a sentence.
    """
    if is_width_uncertain_reason(side, reason):
        return f"{name} {reason}"
    return f"{name} {side}"


def _dropped_sides_note(skipped: List[Tuple[str, str, str]]) -> str:
    """Name the sides the planner could not place a screw on.

    Every planner path (legacy, optimiser, CBT) records the sides it dropped,
    so this covers a missing pedicle or a rejected trajectory just as much as an
    infeasible cortical-bone corridor.  A shorter construct than the one that
    was requested must never reach the surgeon unannounced.

    A whole-spine run can drop a dozen sides, which would push the rest of the
    status line off the label, so only the first
    :data:`_MAX_NAMED_DROPPED_SIDES` are named and the remainder are counted.
    The full list is logged by the caller either way.
    """
    if not skipped:
        return ""
    named = skipped[:_MAX_NAMED_DROPPED_SIDES]
    sides = ", ".join(
        _dropped_side_text(name, side, reason) for name, side, reason in named
    )
    remaining = len(skipped) - len(named)
    if remaining:
        sides += f" and {remaining} more"
    return f" No screw planned: {sides}"


def _rod_misalignment_by_side(planned: List[PlannedScrew]) -> Dict[str, float]:
    """Per-side rod misalignment recorded by the optimiser, if it ran.

    Legacy planning leaves the metric out, in which case the caller omits the
    readout entirely rather than reporting a misleading 0.0 mm.
    """
    rod: Dict[str, float] = {}
    for ps in planned:
        value = ps.metrics.get("rod_misalignment_mm")
        if value is None:
            continue
        try:
            rod[ps.side] = float(value)
        except (TypeError, ValueError):
            logger.warning("Ignoring unreadable rod misalignment %r", value)
    return rod


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
