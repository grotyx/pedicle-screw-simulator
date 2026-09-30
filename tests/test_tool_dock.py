import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtGui import QAction, QActionGroup  # noqa: E402
from PyQt6.QtWidgets import QFrame, QWidget  # noqa: E402

from src.ui.tool_dock import ToolDock  # noqa: E402


def _actions(owner):
    group = QActionGroup(owner)
    group.setExclusive(True)
    a = QAction("Select", owner)
    a.setCheckable(True)
    group.addAction(a)
    a.setChecked(True)
    b = QAction("Add Screw", owner)
    b.setCheckable(True)
    group.addAction(b)
    c = QAction("Fit", owner)
    return a, b, c


def test_dock_mirrors_actions_and_separators(qtbot):
    owner = QWidget()
    qtbot.addWidget(owner)
    a, b, c = _actions(owner)
    dock = ToolDock([a, b, None, c])
    qtbot.addWidget(dock)
    assert dock.objectName() == "toolDock"
    assert dock.property("viewerOverlay") == "true"
    assert [btn.defaultAction() for btn in dock.buttons()] == [a, b, c]
    assert len(dock.findChildren(QFrame, "toolDockSeparator")) == 1


def test_dock_buttons_follow_action_state(qtbot):
    owner = QWidget()
    qtbot.addWidget(owner)
    a, b, c = _actions(owner)
    dock = ToolDock([a, b, c])
    qtbot.addWidget(dock)
    dock.buttons()[1].click()
    assert b.isChecked() and not a.isChecked()
    c.setEnabled(False)
    assert not dock.buttons()[2].isEnabled()


def test_dock_survives_reparenting(qtbot):
    owner = QWidget()
    qtbot.addWidget(owner)
    host1, host2 = QWidget(), QWidget()
    qtbot.addWidget(host1)
    qtbot.addWidget(host2)
    a, b, c = _actions(owner)
    dock = ToolDock([a, b, c])
    dock.setParent(host1)
    dock.setParent(host2)
    dock.buttons()[1].click()
    assert b.isChecked()


def test_separator_is_a_flat_one_pixel_rule(qtbot):
    owner = QWidget()
    qtbot.addWidget(owner)
    a, b, c = _actions(owner)
    dock = ToolDock([a, None, b])
    qtbot.addWidget(dock)
    separator = dock.findChildren(QFrame, "toolDockSeparator")[0]
    assert separator.frameShape() == QFrame.Shape.NoFrame
    assert separator.maximumWidth() == 1
