"""ScrewPlanTable layout: the Side column must stay readable when narrow."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("pytestqt")

from PyQt6.QtWidgets import QApplication

from src.models.screw import Screw
from src.ui.screw_plan_table import ScrewPlanTable


def test_side_column_is_wide_enough_for_its_text_at_the_default_panel_width(qtbot):
    table = ScrewPlanTable()
    qtbot.addWidget(table)
    table.resize(470, 300)
    screw = Screw(
        entry_point=(0.0, 0.0, 0.0),
        target_point=(0.0, 10.0, 0.0),
        diameter=6.5,
        vertebra_level="L4",
        side="right",
    )
    table.addScrewRow(screw)
    table.show()
    QApplication.processEvents()

    text = table.item(0, 2).text()
    assert text == "R"
    needed = table.fontMetrics().horizontalAdvance(text)
    assert table.columnWidth(2) >= needed


def test_side_cell_is_one_letter_and_unknown_side_is_dashes():
    from src.ui.screw_plan_table import screw_row_cells

    def cells(side):
        return screw_row_cells(0, Screw(entry_point=(0.0, 0.0, 0.0), target_point=(0.0, 10.0, 0.0), side=side))

    assert cells("left")[2] == "L"
    assert cells("Right")[2] == "R"
    assert cells("")[2] == "--"
