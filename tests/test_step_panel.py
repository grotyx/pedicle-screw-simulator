import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("pytestqt")

from PyQt6.QtWidgets import QLabel, QWidget

from src.ui.step_panel import StepPanel


def test_page_header_shows_step_number_title_and_hint(qtbot):
    panel = StepPanel()
    qtbot.addWidget(panel)
    page = QWidget()
    panel.add_page("Plan", page, title="Plan screws", subtitle="Auto plan, then adjust.")
    header = panel.header("Plan")
    assert header.objectName() == "stepHeader"
    texts = {label.objectName(): label.text() for label in header.findChildren(QLabel)}
    assert texts["stepHeaderEyebrow"] == "Step 3 of 4"
    assert texts["stepHeaderTitle"] == "Plan screws"
    assert texts["stepHeaderSubtitle"] == "Auto plan, then adjust."
    assert header.findChild(QLabel, "stepHeaderSubtitle").wordWrap()
    assert panel.page("Plan") is page
    assert panel.scroll_area("Plan").widget() is page


def test_page_without_title_has_no_header(qtbot):
    panel = StepPanel()
    qtbot.addWidget(panel)
    panel.add_page("Study", QWidget())
    assert panel.header("Study") is None


def test_show_step_switches_to_the_header_container(qtbot):
    panel = StepPanel()
    qtbot.addWidget(panel)
    panel.add_page("Study", QWidget(), title="Load a study")
    panel.add_page("Plan", QWidget(), title="Plan screws")
    panel.show_step("Plan")
    assert panel.stack.currentWidget().objectName() == "stepPageContainerPlan"


def test_unscrollable_page_with_header_keeps_page_and_no_scroll_area(qtbot):
    panel = StepPanel()
    qtbot.addWidget(panel)
    page = QWidget()
    panel.add_page("Review", page, scrollable=False, title="Review screws")
    assert panel.scroll_area("Review") is None
    assert panel.page("Review") is page
    eyebrow = panel.header("Review").findChild(QLabel, "stepHeaderEyebrow")
    assert eyebrow.text() == "Step 4 of 4"


def test_warning_summary_shows_per_category_counts_and_groups_lines(qtbot):
    from src.ui.step_panel import WarningSummary

    summary = WarningSummary()
    qtbot.addWidget(summary)
    lines = [
        "Estimated breach 1.2 mm — verify on CT.",
        "PCA axis unreliable; used anatomical fallback direction",
        "Pedicle width uncertain – verify diameter",
    ]
    summary.set_lines(lines, {"safety": 1, "image": 2, "info": 0})
    assert summary.toggle.text() == "⚠ 1 safety · 2 image"
    assert summary.toggle.property("warningCategory") == "safety"
    assert summary.label.text().splitlines()[0] == "Safety"
    assert "Image" in summary.label.text().splitlines()

    summary.set_lines(
        ["No estimated breach — CT review is still required."],
        {"safety": 0, "image": 0, "info": 0},
    )
    assert summary.toggle.text() == "✓ No warnings"
    assert summary.toggle.property("warningCategory") == "none"

    summary.set_lines(["x"], {"safety": 0, "image": 0, "info": 1})
    assert summary.toggle.text() == "⚠ 1 info"
    assert summary.toggle.property("warningCategory") == "info"
