"""Tests for the 3D pane header's Vertebrae / Full CT isolation toggle."""

import os
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("pytestqt")

from PyQt6.QtCore import Qt


def _build_viewer(qtbot, monkeypatch):
    from PyQt6.QtWidgets import QWidget

    from src.ui import viewer_3d

    monkeypatch.setattr(
        viewer_3d,
        "create_vtk_widget",
        lambda parent: QWidget(parent),
    )
    monkeypatch.setattr(
        viewer_3d.Viewer3D,
        "_setup_vtk_pipeline",
        lambda self: None,
    )
    volume_manager = SimpleNamespace(add_observer=lambda *_args: None)

    viewer = viewer_3d.Viewer3D(volume_manager)
    qtbot.addWidget(viewer)
    return viewer


def test_header_toggle_emits_only_on_user_clicks_and_reflects_state(qtbot, monkeypatch):
    viewer = _build_viewer(qtbot, monkeypatch)

    assert viewer.header_bar.objectName() == "viewerHeaderBar"
    assert viewer.header_bar.layout().indexOf(viewer.label) == 0

    viewer.set_isolation_state(True, True)
    assert viewer.show_vertebrae_btn.isEnabled()
    assert viewer.show_full_ct_btn.isEnabled()
    assert viewer.show_vertebrae_btn.isChecked() is True
    assert viewer.show_full_ct_btn.isChecked() is False

    emitted = []
    viewer.isolation_requested.connect(emitted.append)

    # set_isolation_state must never emit -- it only reflects a controller's
    # actual state, and a failed isolate calling it to snap the toggle back
    # must not look like a fresh user request.
    viewer.set_isolation_state(False, True)
    assert emitted == []
    assert viewer.show_full_ct_btn.isChecked() is True

    qtbot.mouseClick(viewer.show_vertebrae_btn, Qt.MouseButton.LeftButton)
    assert emitted == [True]


def test_header_toggle_is_disabled_until_available(qtbot, monkeypatch):
    viewer = _build_viewer(qtbot, monkeypatch)

    # Initial state: unavailable, Full CT checked.
    assert viewer.show_vertebrae_btn.isEnabled() is False
    assert viewer.show_full_ct_btn.isEnabled() is False
    assert viewer.show_full_ct_btn.isChecked() is True
    assert viewer.show_vertebrae_btn.isChecked() is False

    viewer.set_isolation_state(False, True)
    assert viewer.show_vertebrae_btn.isEnabled() is True
    assert viewer.show_full_ct_btn.isEnabled() is True


def test_full_ct_click_emits_a_single_restore_request(qtbot, monkeypatch):
    viewer = _build_viewer(qtbot, monkeypatch)
    viewer.set_isolation_state(True, True)

    emitted = []
    viewer.isolation_requested.connect(emitted.append)

    qtbot.mouseClick(viewer.show_full_ct_btn, Qt.MouseButton.LeftButton)

    assert emitted == [False]
    assert viewer.show_full_ct_btn.isChecked() is True
    assert viewer.show_vertebrae_btn.isChecked() is False


def test_toggle_sequence_emits_one_request_per_click(qtbot, monkeypatch):
    viewer = _build_viewer(qtbot, monkeypatch)
    viewer.set_isolation_state(False, True)

    emitted = []
    viewer.isolation_requested.connect(emitted.append)

    qtbot.mouseClick(viewer.show_vertebrae_btn, Qt.MouseButton.LeftButton)
    qtbot.mouseClick(viewer.show_full_ct_btn, Qt.MouseButton.LeftButton)
    qtbot.mouseClick(viewer.show_vertebrae_btn, Qt.MouseButton.LeftButton)

    assert emitted == [True, False, True]
