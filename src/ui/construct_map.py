"""Level-by-level summary of the planned construct for the Review step."""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import QFrame, QGridLayout, QLabel, QVBoxLayout, QWidget

from src.core.pedicle_analyzer import VERTEBRA_LABELS

MANUAL_LEVEL = "Manual"
_CRANIAL_RANK = {name: -label for label, name in VERTEBRA_LABELS.items()}
_GRADES = {"A", "B", "C", "D", "E"}


def level_order_key(level: str) -> tuple:
    """Cranial first; unknown names after known ones; Manual last."""
    if level == MANUAL_LEVEL:
        return (2, 0, "")
    if level in _CRANIAL_RANK:
        return (0, _CRANIAL_RANK[level], "")
    return (1, 0, level)


class _Chip(QLabel):
    clicked = pyqtSignal(int)

    def __init__(self, index: int, text: str, grade: str, parent=None):
        super().__init__(text, parent)
        self.setObjectName("constructChip")
        self.setProperty("screwIndex", index)
        self.setProperty("grade", grade)
        self.setProperty("selected", "false")
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit(int(self.property("screwIndex")))
        super().mousePressEvent(event)


class ConstructMap(QWidget):
    """Right | vertebra | Left grid of grade chips, one row per planned level.

    Right sits on the viewer's left, matching the radiological MPR views.
    """

    screw_clicked = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("constructMap")
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        self._grid_host = QWidget(self)
        outer.addWidget(self._grid_host)
        self._grid = QGridLayout(self._grid_host)
        self._grid.setContentsMargins(0, 0, 0, 0)
        self._grid.setHorizontalSpacing(6)
        self._grid.setVerticalSpacing(4)
        self._rows: List[Tuple[str, List[int], List[int], List[int]]] = []
        self._chips: List[_Chip] = []
        self._selected = -1
        self.set_screws([])

    def rows(self) -> List[Tuple[str, List[int], List[int], List[int]]]:
        return [(lv, list(r), list(lf), list(u)) for lv, r, lf, u in self._rows]

    def set_screws(self, screws: Sequence) -> None:
        grouped: Dict[str, Tuple[List[int], List[int], List[int]]] = {}
        for index, screw in enumerate(screws):
            level = str(getattr(screw, "vertebra_level", "") or "") or MANUAL_LEVEL
            side = str(getattr(screw, "side", "") or "").lower()
            right, left, unsided = grouped.setdefault(level, ([], [], []))
            {"right": right, "left": left}.get(side, unsided).append(index)
        self._rows = [(lv, *grouped[lv]) for lv in sorted(grouped, key=level_order_key)]
        self._rebuild(list(screws))

    def set_selected(self, index: int) -> None:
        self._selected = int(index)
        for chip in self._chips:
            value = "true" if chip.property("screwIndex") == self._selected else "false"
            if chip.property("selected") != value:
                chip.setProperty("selected", value)
                chip.style().unpolish(chip)
                chip.style().polish(chip)

    def _rebuild(self, screws: list) -> None:
        while self._grid.count():
            item = self._grid.takeAt(0)
            widget = item.widget()
            if widget is not None:
                # Detach first so findChildren() never sees the stale chips
                # while deleteLater() waits for the event loop.
                widget.setParent(None)
                widget.deleteLater()
        self._chips = []
        if not self._rows:
            empty = QLabel("No screws planned yet", self._grid_host)
            empty.setObjectName("constructEmptyText")
            self._grid.addWidget(empty, 0, 0, 1, 4)
            return
        for column, text in ((1, "Right"), (3, "Left")):
            head = QLabel(text, self._grid_host)
            head.setObjectName("constructSideHeader")
            head.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self._grid.addWidget(head, 0, column)
        for row, (level, right, left, unsided) in enumerate(self._rows, start=1):
            name = QLabel(level, self._grid_host)
            name.setObjectName("constructLevel")
            name.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            self._grid.addWidget(name, row, 0)
            if unsided:
                # No known laterality: keep it in the middle, never under Right.
                self._grid.addWidget(
                    self._cell(unsided, screws, Qt.AlignmentFlag.AlignHCenter), row, 2
                )
            else:
                bar = QFrame(self._grid_host)
                bar.setObjectName("constructBar")
                bar.setFixedWidth(36)
                self._grid.addWidget(bar, row, 2)
            for column, indices, align in (
                (1, right, Qt.AlignmentFlag.AlignRight),
                (3, left, Qt.AlignmentFlag.AlignLeft),
            ):
                self._grid.addWidget(self._cell(indices, screws, align), row, column)
        self._grid.setColumnStretch(1, 1)
        self._grid.setColumnStretch(3, 1)
        self.set_selected(self._selected)

    def _cell(self, indices: List[int], screws: list, align) -> QWidget:
        cell = QWidget(self._grid_host)
        layout = QVBoxLayout(cell)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        if not indices:
            empty = QLabel(cell)
            empty.setObjectName("constructEmpty")
            layout.addWidget(empty, 0, align)
            return cell
        for index in indices:
            screw = screws[index]
            grade = str(getattr(screw, "grade", "") or "").strip().upper()
            grade = grade if grade in _GRADES else "NA"
            shown = grade if grade != "NA" else "N/A"
            text = f"{shown}  {float(screw.diameter):g}×{float(screw.length):g}"
            chip = _Chip(index, text, grade, cell)
            chip.clicked.connect(self.screw_clicked)
            self._chips.append(chip)
            layout.addWidget(chip, 0, align)
        return cell
