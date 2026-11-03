import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtCore import QPoint  # noqa: E402
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
    assert separator.maximumHeight() == 1


def test_dock_stays_inside_its_parent_and_snaps_to_edges(qtbot):
    owner = QWidget()
    qtbot.addWidget(owner)
    host = QWidget()
    host.resize(900, 900)
    qtbot.addWidget(host)
    dock = ToolDock(list(_actions(owner)), host)
    margin = ToolDock.MARGIN

    assert dock.place(QPoint(-50, -50)) == QPoint(margin, margin)
    right = host.width() - dock.width() - margin
    assert dock.place(QPoint(right - 10, 200)).x() == right  # snapped
    assert dock.place(QPoint(300, 200)) == QPoint(300, 200)  # free in the middle
    assert dock.place(QPoint(5000, 5000)) == QPoint(right, host.height() - dock.height() - margin)


def test_grip_detaches_throttles_and_drops_back_on_strip(qtbot, monkeypatch):
    from PyQt6.QtCore import QEvent, QPointF, Qt
    from PyQt6.QtGui import QMouseEvent
    from PyQt6.QtWidgets import QHBoxLayout

    host = QWidget()
    host.resize(1000, 800)
    qtbot.addWidget(host)
    layout = QHBoxLayout(host)
    strip, views = QWidget(host), QWidget(host)
    layout.addWidget(strip)
    layout.addWidget(views, 1)
    dock = ToolDock(list(_actions(host)), views)
    dock.configure_attachment(strip, views)
    host.show()
    qtbot.waitUntil(host.isVisible)

    def event(kind, global_pos, button=Qt.MouseButton.NoButton):
        mouse = QMouseEvent(kind, QPointF(dock._grip.mapFromGlobal(global_pos)),
                            QPointF(global_pos), button, Qt.MouseButton.LeftButton,
                            Qt.KeyboardModifier.NoModifier)
        dock.eventFilter(dock._grip, mouse)

    start = dock._grip.mapToGlobal(dock._grip.rect().center())
    event(QEvent.Type.MouseButtonPress, start, Qt.MouseButton.LeftButton)
    event(QEvent.Type.MouseMove, start + QPoint(150, 50))
    assert not dock.is_attached
    positions = []
    place = dock.place
    monkeypatch.setattr(dock, 'place', lambda pos: (positions.append(pos), place(pos))[1])
    monkeypatch.setattr(dock, 'adjustSize', lambda: pytest.fail('adjustSize during drag'))
    for x in range(160, 300):
        event(QEvent.Type.MouseMove, start + QPoint(x, 50))
    assert not positions
    qtbot.waitUntil(lambda: len(positions) == 1)
    final = start + QPoint(320, 75)
    event(QEvent.Type.MouseButtonRelease, final, Qt.MouseButton.LeftButton)
    assert len(positions) == 2
    assert not dock.is_attached
    event(QEvent.Type.MouseButtonPress, final, Qt.MouseButton.LeftButton)
    target = strip.mapToGlobal(strip.rect().center())
    event(QEvent.Type.MouseButtonRelease, target, Qt.MouseButton.LeftButton)
    assert dock.is_attached
    assert dock.parentWidget() is strip
    dock.set_attached(False)
    event(QEvent.Type.MouseButtonDblClick, final, Qt.MouseButton.LeftButton)
    assert dock.is_attached
