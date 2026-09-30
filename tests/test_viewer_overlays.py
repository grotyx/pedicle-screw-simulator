"""Overlay hooks shared by MPRViewer and Viewer3D: promote button, thumbnail
mode, attach_overlay and theme-driven (colour-free) overlay styling."""

import inspect
import os
import re
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("pytestqt")

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication, QFrame, QWidget

from src.ui import mpr_viewer, viewer_3d


@pytest.fixture(params=["axial", "3d"])
def viewer(request, qtbot, monkeypatch):
    """A real viewer built without a GL context (plain QWidget for VTK)."""
    monkeypatch.setattr(mpr_viewer, "create_vtk_widget", lambda parent: QWidget(parent))
    monkeypatch.setattr(viewer_3d, "create_vtk_widget", lambda parent: QWidget(parent))
    monkeypatch.setattr(mpr_viewer.MPRViewer, "_setup_vtk_pipeline", lambda self: None)
    monkeypatch.setattr(mpr_viewer.MPRViewer, "_setup_interactor", lambda self: None)
    monkeypatch.setattr(viewer_3d.Viewer3D, "_setup_vtk_pipeline", lambda self: None)
    volume_manager = SimpleNamespace(add_observer=lambda *_args: None)
    if request.param == "3d":
        v = viewer_3d.Viewer3D(volume_manager)
    else:
        v = mpr_viewer.MPRViewer(request.param, volume_manager)
    qtbot.addWidget(v)
    v.resize(400, 300)
    v.show()
    QApplication.processEvents()
    return v, request.param


def _overlays(v):
    return [
        w for w in v.findChildren(QWidget) if w.property("viewerOverlay") == "true"
    ]


def test_promote_button_only_in_thumbnail_mode(viewer):
    v, _ = viewer
    assert v.promote_button.objectName() == "promoteViewButton"
    assert not v.promote_button.isVisible() and not v.is_thumbnail()
    v.set_thumbnail(True)
    assert v.promote_button.isVisible() and v.is_thumbnail()
    v.set_thumbnail(False)
    assert not v.promote_button.isVisible() and not v.is_thumbnail()


def test_promote_emits_view_name(viewer, qtbot):
    v, name = viewer
    v.set_thumbnail(True)
    with qtbot.waitSignal(v.promote_requested) as sig:
        v.promote_button.click()
    assert sig.args == [name]


def test_thumbnail_hides_floating_controls_and_restores_them(viewer):
    v, _ = viewer
    overlays = _overlays(v)
    assert overlays
    before = {w: w.isVisible() for w in overlays}
    assert any(before.values())
    v.set_thumbnail(True)
    assert not any(w.isVisible() for w in overlays)
    v.set_thumbnail(False)
    assert {w: w.isVisible() for w in overlays} == before


def test_attach_overlay_places_widget_over_vtk_without_moving_vtk(viewer):
    v, _ = viewer
    vtk_parent = v.vtk_widget.parent()
    dock = QFrame()
    v.attach_overlay(dock, Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignHCenter)
    assert dock.parent() is v.viewport_container
    assert v.vtk_widget.parent() is vtk_parent
    assert dock.isVisible()


def test_attach_overlay_moves_widget_out_of_the_previous_layout(viewer, qtbot):
    v, _ = viewer
    other = QWidget()
    qtbot.addWidget(other)
    from PyQt6.QtWidgets import QGridLayout

    grid = QGridLayout(other)
    dock = QFrame(other)
    grid.addWidget(dock, 0, 0)
    v.attach_overlay(dock, Qt.AlignmentFlag.AlignBottom)
    assert grid.indexOf(dock) == -1
    assert v.viewport_container.layout().indexOf(dock) != -1


def test_overlay_code_has_no_literal_colours():
    # Header label / readout keep their own inline colours; overlays must not.
    for mod in (mpr_viewer, viewer_3d):
        src = inspect.getsource(mod)
        calls = re.findall(r"self\.(\w+)\.setStyleSheet\(((?:[^()]|\([^()]*\))*)\)", src)
        for name, call in calls:
            if name in ("label", "info_label"):
                continue
            assert not re.search(r"rgba?\(|#[0-9a-fA-F]{3,6}|white", call), (
                mod.__name__,
                name,
            )
