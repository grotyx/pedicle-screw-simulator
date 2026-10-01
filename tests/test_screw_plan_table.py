"""ScrewPlanTable layout: the Side column must stay readable when narrow."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("pytestqt")

from PyQt6.QtCore import Qt
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


# -- Warning categories ------------------------------------------------------


def _known_warnings():
    """Every warning template the planner, grader and loader can produce.

    Built from the real constants and builder functions, so a reworded
    warning is caught here rather than silently changing bucket.
    """
    from src.core.auto_screw_planner import (
        OPTIMIZER_FALLBACK_WARNING,
        UNCONTAINED_NARROW_WARNING,
        WIDTH_UNCERTAIN_SCREW_WARNING,
        narrow_pedicle_warning,
    )
    from src.core.breach_classification import medial_breach_warning
    from src.core.dicom_loader import check_slice_geometry
    from src.core.pedicle_analyzer import (
        NO_REFERENCE_AIM_LEGACY,
        NO_REFERENCE_AIM_OPTIMIZER,
        endplate_fit_warning,
        endplate_neighbour_reference_warning,
        endplate_no_reference_warning,
    )
    from src.core.trajectory_optimizer import endplate_band_relaxed_warning
    from src.utils.constants import CBT_CONTRAINDICATION_NOTE

    # Uneven spacing and a tilted scan axis, so both loader warnings fire.
    loader = check_slice_geometry(
        [(0.0, 0.0, 0.0), (0.0, 5.0, 1.0), (0.0, 20.0, 3.0)],
        (1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
    )
    assert len(loader) == 2

    safety = [
        narrow_pedicle_warning(4.5, 4.0),
        medial_breach_warning(1.2),
        UNCONTAINED_NARROW_WARNING,
        "Breach distance 1.2 mm (grade B)",
        "Cortical clearance 0.5 mm below 1 mm",
        "Facet violation grade 2: facet joint violated",
        "Estimated breach 1.2 mm — verify on CT.",
    ]
    image = [
        WIDTH_UNCERTAIN_SCREW_WARNING,
        endplate_fit_warning(2.0),
        endplate_neighbour_reference_warning(2.0, ["T12", "L3"]),
        endplate_neighbour_reference_warning(None, ["T12"]),
        endplate_no_reference_warning(4.0, NO_REFERENCE_AIM_LEGACY),
        endplate_no_reference_warning(4.0, NO_REFERENCE_AIM_OPTIMIZER),
        f"Upper endplate unavailable; {NO_REFERENCE_AIM_LEGACY}",
        endplate_band_relaxed_warning(10.0),
        "PCA axis unreliable; used anatomical fallback direction",
        "Not graded: run segmentation first",
        "Not graded: trajectory does not pass through a segmented vertebra",
        "Grade N/A — run segmentation to grade this screw.",
        *loader,
    ]
    info = [
        "Diameter reduced from 6.5 to 5.5 mm for cortical containment",
        "Entry moved 2 mm laterally to protect the medial wall",
        CBT_CONTRAINDICATION_NOTE,
        OPTIMIZER_FALLBACK_WARNING,
        "High convergence angle 32.0° — verify on CT",
        "Trajectory HU 100 below 123 HU — loosening risk (consider larger diameter, augmentation, or CBT)",
        "Vertebral body HU 120 suggests osteoporosis (<132 HU)",
        "Vertebral body HU 135 suggests low bone density (<141 HU)",
        "Trajectory/body HU ratio 0.80 below 1.0",
        "No estimated breach — CT review is still required.",
        "A wording nobody has classified yet",
    ]
    return safety, image, info


@pytest.mark.parametrize("category", ["safety", "image", "info"])
def test_every_known_warning_lands_in_its_category(category):
    from src.ui.screw_plan_table import classify_warning

    safety, image, info = _known_warnings()
    expected = {"safety": safety, "image": image, "info": info}[category]
    wrong = [w for w in expected if classify_warning(w) != category]
    assert wrong == []


def test_unknown_warning_text_is_info():
    from src.ui.screw_plan_table import classify_warning

    assert classify_warning("") == "info"
    assert classify_warning("Something new") == "info"


def _screw(**kwargs):
    base = dict(
        entry_point=(0.0, 0.0, 0.0),
        target_point=(0.0, -40.0, 0.0),
        diameter=6.0,
        vertebra_level="L4",
        side="right",
        grade="A",
    )
    base.update(kwargs)
    return Screw(**base)


def test_warning_counts_split_by_category_and_skip_the_no_breach_line():
    from src.ui.screw_plan_table import (
        screw_warning_count,
        screw_warning_counts,
        worst_warning_category,
    )

    clean = _screw()
    assert screw_warning_counts(clean) == {"safety": 0, "image": 0, "info": 0}
    assert screw_warning_count(clean) == 0
    assert worst_warning_category(screw_warning_counts(clean)) is None

    screw = _screw(
        breach_distance=1.2,
        warnings=[
            "PCA axis unreliable; used anatomical fallback direction",
            "Diameter reduced from 6.5 to 5.5 mm for cortical containment",
            "Entry moved 2 mm laterally to protect the medial wall",
        ],
    )
    counts = screw_warning_counts(screw)
    assert counts == {"safety": 1, "image": 1, "info": 2}
    assert screw_warning_count(screw) == 4
    assert worst_warning_category(counts) == "safety"
    assert worst_warning_category({"safety": 0, "image": 2, "info": 1}) == "image"
    assert worst_warning_category({"safety": 0, "image": 0, "info": 3}) == "info"


def test_grouped_warning_text_has_headings_in_severity_order():
    from src.ui.screw_plan_table import grouped_warning_text

    text = grouped_warning_text(
        [
            "Diameter reduced from 6.5 to 5.5 mm for cortical containment",
            "Pedicle width uncertain – verify diameter",
            "Estimated breach 1.2 mm — verify on CT.",
            "PCA axis unreliable; used anatomical fallback direction",
        ]
    )
    assert text.splitlines() == [
        "Safety",
        "• Estimated breach 1.2 mm — verify on CT.",
        "Image",
        "• Pedicle width uncertain – verify diameter",
        "• PCA axis unreliable; used anatomical fallback direction",
        "Info",
        "• Diameter reduced from 6.5 to 5.5 mm for cortical containment",
    ]
    assert grouped_warning_text([]) == ""


def test_warning_cell_is_a_chip_coloured_by_the_worst_category(qtbot):
    from src.ui.screw_plan_table import WARNINGS_COLUMN
    from src.ui.styles import get_theme

    table = ScrewPlanTable()
    qtbot.addWidget(table)
    table.apply_theme("soft_light")
    palette = get_theme("soft_light")
    cases = [
        (_screw(breach_distance=1.2, warnings=["Diameter reduced from 6.5 to 5.5 mm"]), "2", "danger"),
        (_screw(warnings=["PCA axis unreliable; used anatomical fallback direction"]), "1", "warning"),
        (_screw(warnings=["Entry moved 2 mm laterally to protect the medial wall"]), "1", "grade_na"),
    ]
    for screw, _text, _token in cases:
        table.addScrewRow(screw)
    for row, (_unused, text, token) in enumerate(cases):
        item = table.item(row, WARNINGS_COLUMN)
        assert item.text() == text
        assert item.background().color().name().lower() == palette[token].lower()
        assert item.foreground().color().name().lower() == palette["grade_text"].lower()

    clean = table.addScrewRow(_screw())
    item = table.item(clean, WARNINGS_COLUMN)
    assert item.text() == ""
    assert item.background().style() == Qt.BrushStyle.NoBrush

    # A theme change repaints the chip from the stored category.
    table.apply_theme("graphite_blue")
    dark = get_theme("graphite_blue")
    assert table.item(0, WARNINGS_COLUMN).background().color().name().lower() == dark["danger"].lower()


def test_warning_tooltip_groups_lines_under_category_headings(qtbot):
    from src.ui.screw_plan_table import WARNINGS_COLUMN

    table = ScrewPlanTable()
    qtbot.addWidget(table)
    table.addScrewRow(
        _screw(
            breach_distance=1.2,
            warnings=["PCA axis unreliable; used anatomical fallback direction"],
        )
    )
    tip = table.item(0, WARNINGS_COLUMN).toolTip().splitlines()
    assert tip[0] == "Safety"
    assert "• Estimated breach 1.2 mm — verify on CT." in tip
    assert tip.index("Image") > tip.index("Safety")
    assert "• PCA axis unreliable; used anatomical fallback direction" in tip
