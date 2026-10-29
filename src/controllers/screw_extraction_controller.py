"""Tools > Extract Screws from Post-op CT.

Finds the implanted screws in the loaded volume and adds them to the plan as
ordinary editable screws, so they are graded, drawn and listed like any other.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication, QMessageBox

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


class ScrewExtractionController:
    """Runs screw extraction on the loaded CT and adds the result to the plan."""

    def __init__(self, volume_manager: VolumeManager, main_window):
        self._vm = volume_manager
        self._window = main_window

    def run_extraction(self) -> None:
        ct_image = self._vm.get_sitk_image()
        if ct_image is None:
            QMessageBox.warning(
                self._window, "No CT Volume", "Load a post-operative CT first."
            )
            return

        # Synchronous: ~0.4 s on a 512 x 512 x 300 volume at 0.7 mm, ~1 s at 0.4 mm.
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            result = extract_screws(ct_image)
        except Exception as exc:
            logger.exception("Screw extraction failed")
            QMessageBox.warning(
                self._window, "Cannot Extract Screws", f"{exc}."
            )
            return
        finally:
            QApplication.restoreOverrideCursor()

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
                    "wide); check against the images"
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
