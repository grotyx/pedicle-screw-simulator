"""Tools > Extract Screws from Post-op CT.

Finds the implanted screws in the loaded volume and adds them to the plan as
ordinary editable screws, so they are graded, drawn and listed like any other.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtWidgets import QMessageBox

from src.core.screw_extraction import extract_screws
from src.models.screw import Screw
from src.utils.constants import MAX_SCREW_DIAMETER, MIN_SCREW_DIAMETER

if TYPE_CHECKING:
    from src.core.volume_manager import VolumeManager

logger = logging.getLogger(__name__)


def side_from_segmentation(grader, entry, target, centres=None) -> str:
    """``"left"``/``"right"`` from the vertebra the screw runs through, else ``""``.

    The planner's rule: a head on the +x (patient left, LPS) side of the
    vertebra's centre is a left screw.
    """
    if grader is None:
        return ""
    label = grader.detect_label(entry, target)
    if label is None:
        return ""
    # ``centres`` caches each label's x centre across one extraction run.
    centres = {} if centres is None else centres
    if label not in centres:
        columns = np.nonzero(grader.label_array() == label)[2]
        mask = grader.mask_image()
        centres[label] = (
            mask.GetOrigin()[0] + mask.GetSpacing()[0] * float(columns.mean())
        )
    centre_x = centres[label]
    return "left" if entry[0] > centre_x else "right"


#: Shown under an extracted screw's grade, which the post-op CT itself skews.
POSTOP_GRADE_WARNING = (
    "Grade is indicative only: metal artefact distorts the post-op "
    "segmentation and blooming widens the measured screw"
)


class ScrewExtractionThread(QThread):
    """Runs :func:`extract_screws` off the GUI thread (0.4-7 s on real CTs)."""

    extracted = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, ct_image):
        super().__init__()
        self._ct_image = ct_image

    def run(self):
        try:
            self.extracted.emit(extract_screws(self._ct_image))
        except Exception as exc:
            logger.exception("Screw extraction failed")
            self.failed.emit(f"{exc}.")


class ScrewExtractionController:
    """Runs screw extraction on the loaded CT and adds the result to the plan."""

    def __init__(self, volume_manager: VolumeManager, main_window):
        self._vm = volume_manager
        self._window = main_window
        self._thread = None

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.isRunning()

    def run_extraction(self) -> None:
        if self.is_running:
            self._window.statusbar.showMessage("Screw extraction is already running")
            return
        ct_image = self._vm.get_sitk_image()
        if ct_image is None:
            QMessageBox.warning(
                self._window, "No CT Volume", "Load a post-operative CT first."
            )
            return

        self._window.statusbar.showMessage("Extracting screws from the CT...")
        thread = ScrewExtractionThread(ct_image)
        thread.extracted.connect(lambda result: self._on_extracted(ct_image, result))
        thread.failed.connect(self._on_failed)
        thread.finished.connect(self._release_thread)
        self._thread = thread
        thread.start()

    def _release_thread(self) -> None:
        if self._thread is not None and not self._thread.isRunning():
            self._thread.deleteLater()
            self._thread = None

    def _on_failed(self, message: str) -> None:
        self._window.statusbar.showMessage("Screw extraction failed")
        QMessageBox.warning(self._window, "Cannot Extract Screws", message)

    def _on_extracted(self, ct_image, result) -> None:
        if self._vm.get_sitk_image() is not ct_image:
            # Another study was loaded meanwhile: these screws belong to the old one.
            self._window.statusbar.showMessage("Screw extraction discarded: the CT changed")
            return
        if not result.screws:
            self._window.statusbar.showMessage("Screw extraction: no screws found")
            QMessageBox.warning(
                self._window,
                "No Screws Found",
                f"No implanted screws were found in this CT.\n\n{result.reason}.",
            )
            return

        tool_ctrl = self._window._tool_ctrl
        screw_tool = tool_ctrl.screw_tool
        before = tool_ctrl.snapshot_screws()
        centres = {}
        for offset, found in enumerate(result.screws):
            screw = Screw(
                entry_point=found.entry_lps,
                target_point=found.target_lps,
                # Metal blooming widens the shaft; keep the model inside the
                # sizes the app can edit and say what was measured.
                diameter=min(
                    max(round(found.diameter_mm, 1), MIN_SCREW_DIAMETER),
                    MAX_SCREW_DIAMETER,
                ),
                side=side_from_segmentation(
                    screw_tool.grader, found.entry_lps, found.target_lps, centres
                ),
                source="extracted",
                warnings=[
                    f"Extracted from post-op CT ({found.reason}, confidence "
                    f"{found.confidence:.1f}, measured {found.diameter_mm:.1f} mm "
                    "wide); check against the images",
                    POSTOP_GRADE_WARNING,
                ],
            )
            screw_tool.evaluate_screw(screw)
            tool_ctrl.add_existing_screw(screw, select=offset == 0)
        tool_ctrl.record_screw_edit("extract screws", before)
        logger.info("Screw extraction: %s", result.reason)
        self._window.statusbar.showMessage(
            f"Extracted {len(result.screws)} screws from the CT — "
            "check each against the images"
        )
