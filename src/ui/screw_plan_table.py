"""Tabular screw plan list with theme-coloured Gertzbein grade chips."""

from typing import Optional

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QHeaderView,
    QTableWidget,
    QTableWidgetItem,
)

from .styles import DEFAULT_THEME, get_theme

#: Column headers, in display order.
COLUMN_TITLES = (
    "#",
    "Level",
    "Side",
    "Ø (mm)",
    "Length (mm)",
    "Grade",
    "Source",
)

#: Index of the column that carries the coloured grade chip.
GRADE_COLUMN = 5

#: Columns whose numeric content reads better right-aligned.
_NUMERIC_COLUMNS = (0, 3, 4)

#: Gertzbein grade -> palette token. D and E share the danger colour; anything
#: else (including "N/A" before segmentation has run) falls back to grey.
GRADE_TOKENS = {
    "A": "grade_a",
    "B": "grade_b",
    "C": "grade_c",
    "D": "grade_d",
    "E": "grade_d",
}


def screw_row_cells(index: int, screw) -> tuple[str, ...]:
    """Return the seven display strings for one screw, in column order."""
    level = getattr(screw, "vertebra_level", None) or "Manual"
    side_value = getattr(screw, "side", None)
    side = str(side_value).capitalize() if side_value else "--"
    source = (
        "Auto"
        if str(getattr(screw, "source", "manual")) == "auto"
        else "Manual"
    )
    return (
        str(index + 1),
        str(level),
        side,
        f"{float(screw.diameter):.1f}",
        f"{float(screw.length):.1f}",
        f"Grade {screw.grade}",
        source,
    )


class ScrewPlanTable(QTableWidget):
    """A drop-in replacement for the old multi-line screw ``QListWidget``.

    ``currentRowChanged(int)`` mirrors ``QListWidget``'s signal of the same
    name (including the ``-1`` emitted when the selection is cleared) so the
    three controllers that listen to it keep working unchanged.
    """

    currentRowChanged = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(0, len(COLUMN_TITLES), parent)
        self.setObjectName("screwPlanTable")
        self.setHorizontalHeaderLabels(list(COLUMN_TITLES))
        self.verticalHeader().setVisible(False)
        self.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.setSelectionMode(
            QAbstractItemView.SelectionMode.SingleSelection
        )
        self.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.setAlternatingRowColors(True)
        self.setShowGrid(False)
        self.setWordWrap(False)
        self.setVerticalScrollMode(
            QAbstractItemView.ScrollMode.ScrollPerPixel
        )
        self.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOn
        )
        self.setMinimumHeight(160)
        self.setMaximumHeight(220)

        header = self.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)

        self._theme_name = DEFAULT_THEME
        self._last_row = -1
        self.currentCellChanged.connect(self._on_current_cell_changed)

    # -- QListWidget-compatible surface ------------------------------------

    def count(self) -> int:
        """Number of screw rows (``QListWidget.count`` equivalent)."""
        return self.rowCount()

    def clear(self) -> None:
        """Drop every row but keep the header labels."""
        self.clearContents()
        self.setRowCount(0)
        self._sync_current_row()

    def setCurrentRow(self, row: int) -> None:
        """Select one row, or clear the selection for an out-of-range index."""
        if row is None or int(row) < 0 or int(row) >= self.rowCount():
            self.setCurrentCell(-1, -1)
        else:
            self.setCurrentCell(int(row), 0)

    # -- Screw-specific API ------------------------------------------------

    def addScrewRow(self, screw, index: Optional[int] = None) -> int:
        """Append one screw and return the row it landed on."""
        row = self.rowCount()
        self.insertRow(row)
        self._fill_row(row, row if index is None else int(index), screw)
        return row

    def updateScrewRow(self, index: int, screw) -> None:
        """Re-render one existing row after its screw changed."""
        row = int(index)
        if 0 <= row < self.rowCount():
            self._fill_row(row, row, screw)

    def rowText(self, index: int) -> str:
        """Return one row's cells joined with ' · ', or '' when out of range."""
        row = int(index)
        if not 0 <= row < self.rowCount():
            return ""
        return " · ".join(
            (self.item(row, column).text() if self.item(row, column) else "")
            for column in range(self.columnCount())
        )

    def gradeColor(self, grade) -> str:
        """Return the palette colour this table paints for one grade."""
        token = GRADE_TOKENS.get(str(grade).strip().upper(), "grade_na")
        return get_theme(self._theme_name)[token]

    def scrollToRow(self, index: int) -> None:
        """Centre one row in the viewport, if it exists."""
        item = self.item(int(index), 0)
        if item is not None:
            self.scrollToItem(
                item,
                QAbstractItemView.ScrollHint.PositionAtCenter,
            )

    def apply_theme(self, theme_name: str) -> None:
        """Repaint every grade chip after a palette change."""
        self._theme_name = str(theme_name)
        for row in range(self.rowCount()):
            chip = self.item(row, GRADE_COLUMN)
            if chip is not None:
                self._paint_grade_chip(row, self._grade_of_row(row))

    # -- Internals ---------------------------------------------------------

    def _grade_of_row(self, row: int) -> str:
        chip = self.item(row, GRADE_COLUMN)
        if chip is None:
            return "N/A"
        return chip.text().removeprefix("Grade").strip() or "N/A"

    def _fill_row(self, row: int, index: int, screw) -> None:
        for column, text in enumerate(screw_row_cells(index, screw)):
            item = QTableWidgetItem(text)
            item.setFlags(
                Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable
            )
            if column in _NUMERIC_COLUMNS:
                item.setTextAlignment(
                    Qt.AlignmentFlag.AlignRight
                    | Qt.AlignmentFlag.AlignVCenter
                )
            else:
                item.setTextAlignment(
                    Qt.AlignmentFlag.AlignLeft
                    | Qt.AlignmentFlag.AlignVCenter
                )
            self.setItem(row, column, item)
        self._paint_grade_chip(row, str(screw.grade))

    def _paint_grade_chip(self, row: int, grade: str) -> None:
        chip = self.item(row, GRADE_COLUMN)
        if chip is None:
            return
        palette = get_theme(self._theme_name)
        token = GRADE_TOKENS.get(str(grade).strip().upper(), "grade_na")
        chip.setBackground(QColor(palette[token]))
        chip.setForeground(QColor(palette["grade_text"]))
        chip.setTextAlignment(Qt.AlignmentFlag.AlignCenter)

    def _on_current_cell_changed(self, *_args) -> None:
        self._sync_current_row()

    def _sync_current_row(self) -> None:
        row = self.currentRow()
        if row != self._last_row:
            self._last_row = row
            self.currentRowChanged.emit(row)
