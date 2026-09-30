import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from types import SimpleNamespace as S

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication, QLabel

from src.ui.construct_map import ConstructMap, level_order_key


def screw(level, side, grade="A", d=6.5, length=45.0):
    return S(vertebra_level=level, side=side, grade=grade, diameter=d, length=length)


def test_levels_run_cranial_to_caudal_with_manual_last():
    levels = ["S1", "Manual", "L4", "T12", "L5"]
    assert sorted(levels, key=level_order_key) == ["T12", "L4", "L5", "S1", "Manual"]


def test_rows_group_screws_by_level_and_side(qtbot):
    m = ConstructMap()
    qtbot.addWidget(m)
    m.set_screws(
        [
            screw("L5", "left"),
            screw("L4", "right", "B"),
            screw("L4", "left"),
            screw("", "", "A"),
            screw("L4", "left", "C"),
        ]
    )
    assert m.rows() == [("L4", [1], [2, 4], []), ("L5", [], [0], []), ("Manual", [], [], [3])]


def test_unsided_screw_is_never_labelled_right(qtbot):
    m = ConstructMap()
    qtbot.addWidget(m)
    m.set_screws([screw("L3", "", "B")])
    assert m.rows() == [("L3", [], [], [0])]
    assert m.findChildren(QLabel, "constructChip")[0].property("screwIndex") == 0


def test_chip_text_and_grade_property(qtbot):
    m = ConstructMap()
    qtbot.addWidget(m)
    m.set_screws([screw("L4", "right", "B", 6.5, 45)])
    chips = m.findChildren(QLabel, "constructChip")
    assert [c.text() for c in chips] == ["B  6.5×45"]
    assert chips[0].property("grade") == "B"


def test_clicking_a_chip_emits_its_screw_index(qtbot):
    m = ConstructMap()
    qtbot.addWidget(m)
    m.show()
    QApplication.processEvents()
    m.set_screws([screw("L4", "right"), screw("L4", "left", "C")])
    QApplication.processEvents()
    chip = [c for c in m.findChildren(QLabel, "constructChip") if c.property("screwIndex") == 1][0]
    with qtbot.waitSignal(m.screw_clicked) as sig:
        qtbot.mouseClick(chip, Qt.MouseButton.LeftButton)
    assert sig.args == [1]


def test_selection_marks_one_chip(qtbot):
    m = ConstructMap()
    qtbot.addWidget(m)
    m.set_screws([screw("L4", "right"), screw("L4", "left")])
    m.set_selected(1)
    sel = [
        c.property("screwIndex")
        for c in m.findChildren(QLabel, "constructChip")
        if c.property("selected") == "true"
    ]
    assert sel == [1]


def test_empty_state_text(qtbot):
    m = ConstructMap()
    qtbot.addWidget(m)
    m.set_screws([])
    assert any(lbl.text() == "No screws planned yet" for lbl in m.findChildren(QLabel))


def test_rebuild_leaves_no_stale_chips(qtbot):
    m = ConstructMap()
    qtbot.addWidget(m)
    m.set_screws([screw("L4", "right", "A", 6.5, 45)])
    m.set_screws([screw("L4", "right", "A", 7.5, 45)])
    assert [c.text() for c in m.findChildren(QLabel, "constructChip")] == ["A  7.5×45"]


def test_unknown_grade_uses_na(qtbot):
    m = ConstructMap()
    qtbot.addWidget(m)
    m.set_screws([screw("L3", "left", "N/A")])
    assert m.findChildren(QLabel, "constructChip")[0].property("grade") == "NA"
