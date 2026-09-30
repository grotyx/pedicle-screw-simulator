"""Typed input for the screw diameter spin box."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("pytestqt")

from src.ui.spin_boxes import DiameterSpinBox
from src.utils.constants import MAX_SCREW_DIAMETER, MIN_SCREW_DIAMETER


@pytest.fixture
def spin(qtbot):
    box = DiameterSpinBox()
    box.setRange(MIN_SCREW_DIAMETER, MAX_SCREW_DIAMETER)
    box.setDecimals(1)
    box.setSingleStep(0.5)
    box.setValue(6.0)
    qtbot.addWidget(box)
    return box


@pytest.mark.parametrize("typed, expected", [
    ("7.5", 7.5),
    ("5,5", 5.5),
    ("6.5", 6.5),
    ("4.5", 4.5),
    ("65", 6.5),    # digit-only shortcut
    ("5", 5.0),
])
@pytest.mark.parametrize("via_line_edit", [True, False])
def test_typed_diameter_is_the_value_typed(qtbot, spin, typed, expected, via_line_edit):
    target = spin.lineEdit() if via_line_edit else spin
    qtbot.keyClicks(target, typed)
    assert spin.value() == pytest.approx(expected)
