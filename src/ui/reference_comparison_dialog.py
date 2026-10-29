"""Read-only table comparing the working plan with a reference plan.

The numbers come from :func:`src.core.plan_metrics.compare_screw_lists`; this
module only lays them out.  It is a separate, resizable window so the ten
metric columns never widen the Review page.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from src.core.plan_metrics import ScrewListComparison

COLUMNS = (
    "Status", "Current #", "Reference #", "Level", "Side",
    "Head Δ (mm)", "Tip Δ (mm)", "Axis (°)", "Conv Δ (°)", "CC Δ (°)",
    "Length Δ (mm)", "Diameter Δ (mm)", "Dice",
)

FRAME_VERIFIED_TEXT = "Same CT series as the reference plan."
FRAME_UNVERIFIED_TEXT = (
    "Coordinate frame unverified: one of the plans has no series ID. "
    "Check that the amber reference screws line up with the anatomy."
)
NO_PAIRS_TEXT = "No comparable screws."


def _identity(screw) -> tuple:
    return (str(screw.vertebra_level or "") or "—", str(screw.side or "") or "—")


def comparison_rows(
    result: ScrewListComparison, current: Sequence, reference: Sequence
) -> List[tuple]:
    """One tuple of display strings per table row, matched pairs first."""
    rows: List[tuple] = []
    for (c, r), comp in zip(result.pairs, result.comparisons, strict=True):
        rows.append((
            "matched", str(c + 1), str(r + 1), comp.level or "—", comp.side or "—",
            f"{comp.head_mad_mm:.1f}", f"{comp.tip_mad_mm:.1f}",
            f"{comp.axis_angle_deg:.1f}", f"{comp.convergence_delta_deg:+.1f}",
            f"{comp.craniocaudal_delta_deg:+.1f}", f"{comp.length_delta_mm:+.1f}",
            f"{comp.diameter_delta_mm:+.2f}", f"{comp.dice:.2f}",
        ))
    blank = ("",) * 8

    def leftover(status: str, index: int, screws: Sequence, is_current: bool) -> tuple:
        number = str(index + 1)
        numbers = (number, "") if is_current else ("", number)
        return (status, *numbers, *_identity(screws[index]), *blank)

    for index in result.unmatched_current:
        rows.append(leftover("only in current plan", index, current, True))
    for index in result.unmatched_reference:
        rows.append(leftover("only in reference", index, reference, False))
    for index in result.unassigned_current:
        rows.append(leftover("no level/side (current)", index, current, True))
    for index in result.unassigned_reference:
        rows.append(leftover("no level/side (reference)", index, reference, False))
    for index, reason in result.invalid_current:
        rows.append(leftover(f"invalid (current): {reason}", index, current, True))
    for index, reason in result.invalid_reference:
        rows.append(leftover(f"invalid (reference): {reason}", index, reference, False))
    return rows


def summary_text(result: ScrewListComparison) -> str:
    """Pair count and means; spread and agreement limits only from two pairs on."""
    s = result.summary
    if s.n_matched == 0:
        return NO_PAIRS_TEXT
    spread = s.n_matched >= 2

    def mean(value: float, sd: float, unit: str) -> str:
        return f"{value:.1f} ± {sd:.1f}{unit}" if spread else f"{value:.1f}{unit}"

    lines = [
        f"{s.n_matched} matched pair{'s' if s.n_matched != 1 else ''}; "
        f"{s.n_unmatched_pred} current and {s.n_unmatched_ref} reference screws unmatched.",
        f"Head Δ {mean(s.head_mad_mean, s.head_mad_sd, ' mm')} · "
        f"Tip Δ {mean(s.tip_mad_mean, s.tip_mad_sd, ' mm')} · "
        f"Axis {mean(s.axis_angle_mean, s.axis_angle_sd, '°')} · Dice {s.dice_mean:.2f}",
    ]
    if spread:
        lines.append(
            f"Length bias {s.length_bias:+.1f} mm (95% LoA {s.length_loa[0]:+.1f} to "
            f"{s.length_loa[1]:+.1f}) · Diameter bias {s.diameter_bias:+.2f} mm "
            f"(95% LoA {s.diameter_loa[0]:+.2f} to {s.diameter_loa[1]:+.2f})"
        )
    return "\n".join(lines)


class ReferenceComparisonDialog(QDialog):
    """Non-modal window; ``Recompare`` asks the owner for a fresh snapshot."""

    def __init__(self, parent=None, on_recompare=None):
        super().__init__(parent)
        self.setWindowTitle("Compare with reference plan")
        self.resize(980, 420)
        layout = QVBoxLayout(self)

        self.frame_label = QLabel()
        self.frame_label.setWordWrap(True)
        layout.addWidget(self.frame_label)

        self.table = QTableWidget(0, len(COLUMNS))
        self.table.setHorizontalHeaderLabels(COLUMNS)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        layout.addWidget(self.table, 1)

        self.summary_label = QLabel()
        self.summary_label.setWordWrap(True)
        self.summary_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.summary_label)

        note = QLabel(
            "Δ = current − reference. The reference is a plan you chose, not a "
            "validated ground truth. Results are a snapshot: press Recompare after "
            "editing screws."
        )
        note.setWordWrap(True)
        layout.addWidget(note)

        buttons = QHBoxLayout()
        buttons.addStretch(1)
        self.recompare_btn = QPushButton("Recompare")
        if on_recompare is not None:
            self.recompare_btn.clicked.connect(on_recompare)
        buttons.addWidget(self.recompare_btn)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.close)
        buttons.addWidget(close_btn)
        layout.addLayout(buttons)

    def set_result(
        self,
        result: ScrewListComparison,
        current: Sequence,
        reference: Sequence,
        frame_verified: bool,
        computed_at: Optional[str] = None,
    ) -> None:
        self.frame_label.setText(FRAME_VERIFIED_TEXT if frame_verified else FRAME_UNVERIFIED_TEXT)
        rows = comparison_rows(result, current, reference)
        self.table.setRowCount(len(rows))
        for row, values in enumerate(rows):
            for column, value in enumerate(values):
                self.table.setItem(row, column, QTableWidgetItem(value))
        self.table.resizeColumnsToContents()
        summary = summary_text(result)
        if computed_at:
            summary = f"{summary}\nComputed at {computed_at}."
        self.summary_label.setText(summary)
