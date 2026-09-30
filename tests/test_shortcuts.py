# ruff: noqa: F811
"""Unmodified single-key shortcuts (R = Reset Camera) must not steal typing.

Qt runs a ShortcutOverride event on the focused widget before it triggers a
shortcut; a text field that accepts it keeps the key.  Offscreen there is no
active window, so a real key press never reaches the shortcut map and would
pass vacuously -- this checks the override decision itself.
"""

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("pytestqt")

from PyQt6.QtCore import QEvent, Qt
from PyQt6.QtGui import QKeyEvent
from PyQt6.QtWidgets import QApplication, QPushButton

from tests.test_main_window_ui import (  # noqa: F401  (fixtures)
    isolated_qsettings,
    ui_main_window,
)


def _override_accepted(widget, key=Qt.Key.Key_R, text="r") -> bool:
    event = QKeyEvent(
        QEvent.Type.ShortcutOverride, key, Qt.KeyboardModifier.NoModifier, text
    )
    event.ignore()
    QApplication.sendEvent(widget, event)
    return event.isAccepted()


def test_r_shortcut_is_bound_to_reset_camera(ui_main_window):
    shortcuts = {
        action.shortcut().toString(): action.text()
        for action in ui_main_window.findChildren(type(ui_main_window._cancel_screw_edit_action))
        if not action.shortcut().isEmpty()
    }
    assert shortcuts.get("R") == "Reset Camera"


@pytest.mark.parametrize("field", ["diameter_spin", "length_spin"])
def test_r_typed_in_a_spin_box_is_kept_by_the_field(ui_main_window, field):
    spin = getattr(ui_main_window, field)
    assert _override_accepted(spin.lineEdit())


def test_delete_typed_in_a_spin_box_is_kept_by_the_field(ui_main_window):
    field = ui_main_window.diameter_spin.lineEdit()
    assert _override_accepted(field, Qt.Key.Key_Delete, "")


def test_override_check_is_not_vacuous():
    assert not _override_accepted(QPushButton())
