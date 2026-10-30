"""Tools > Extract Screws from Post-op CT, end to end on a synthetic phantom.

Builds its own ``ui_main_window`` fixture, as ``tests/test_main_window.py``
does, rather than importing one (see that module's docstring for why).
"""

import os

import numpy as np
import pytest
import SimpleITK as sitk

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("pytestqt")

from PyQt6.QtCore import QSettings
from PyQt6.QtWidgets import QApplication

import src.controllers.screw_extraction_controller as controller_module
import src.ui.main_window as main_window_module
from tests.test_screw_extraction import LEFT_HEAD, RIGHT_HEAD, _grid, _phantom, _screw
from tests.test_ui_integration import DummyMPRViewer, DummyViewer3D


@pytest.fixture
def ui_main_window(monkeypatch, qtbot, tmp_path):
    directory = tmp_path / "qsettings"
    directory.mkdir()

    def factory(organization="Default", application="App", *_args, **_kwargs):
        return QSettings(
            str(directory / f"{organization}-{application}.ini"),
            QSettings.Format.IniFormat,
        )

    monkeypatch.setattr(main_window_module, "QSettings", factory)
    monkeypatch.setattr(main_window_module, "_migrated", False)
    monkeypatch.setattr(main_window_module, "MPRViewer", DummyMPRViewer)
    monkeypatch.setattr(main_window_module, "Viewer3D", DummyViewer3D)
    QApplication.instance().setProperty("themeName", "soft_light")
    window = main_window_module.MainWindow()
    qtbot.addWidget(window)
    return window


def _extract(window, qtbot):
    """Trigger the menu action and wait for the worker thread to deliver."""
    window._extract_screws_action.trigger()
    qtbot.waitUntil(lambda: not window._screw_extraction_ctrl.is_running, timeout=30000)
    QApplication.processEvents()


def test_menu_action_adds_extracted_screws_to_the_plan(ui_main_window, qtbot, monkeypatch):
    window = ui_main_window
    points = _grid()
    image = _phantom([_screw(points, LEFT_HEAD), _screw(points, RIGHT_HEAD)])
    monkeypatch.setattr(window.volume_manager, "get_sitk_image", lambda: image)

    _extract(window, qtbot)

    screws = window._tool_ctrl.screw_tool.get_screws()
    assert len(screws) == 2
    assert {s.source for s in screws} == {"extracted"}
    # No segmentation: nothing to grade against, and no level or side to guess.
    assert all(s.grade == "N/A" for s in screws)
    assert all(s.vertebra_level == "" and s.side == "" for s in screws)
    assert window.screw_list_widget.count() == 2
    assert "2 screws" in window.statusbar.currentMessage()


def test_menu_action_warns_when_no_screw_is_found(ui_main_window, qtbot, monkeypatch):
    window = ui_main_window
    empty = _phantom([])
    monkeypatch.setattr(window.volume_manager, "get_sitk_image", lambda: empty)
    warnings = []
    monkeypatch.setattr(
        controller_module.QMessageBox,
        "warning",
        lambda *args, **kwargs: warnings.append(args),
    )

    _extract(window, qtbot)

    assert window._tool_ctrl.screw_tool.get_screws() == []
    assert len(warnings) == 1


def test_side_comes_from_the_vertebra_the_screw_is_in():
    # One vertebra label centred at x = 50 mm; LPS +x is the patient's left.
    mask = np.zeros((20, 20, 100), dtype=np.uint8)
    mask[:, :, 30:70] = 22
    mask_image = sitk.GetImageFromArray(mask)

    class Grader:
        def detect_label(self, entry, target):
            return 22

        def label_array(self):
            return mask

        def mask_image(self):
            return mask_image

    side_of = controller_module.side_from_segmentation
    assert side_of(Grader(), (60.0, 5.0, 5.0), (52.0, 15.0, 5.0)) == "left"
    assert side_of(Grader(), (40.0, 5.0, 5.0), (48.0, 15.0, 5.0)) == "right"
    assert side_of(None, (60.0, 5.0, 5.0), (52.0, 15.0, 5.0)) == ""


def test_extraction_is_one_undo_step_and_reads_as_extracted(ui_main_window, qtbot, monkeypatch):
    window = ui_main_window
    points = _grid()
    image = _phantom([_screw(points, LEFT_HEAD), _screw(points, RIGHT_HEAD)])
    monkeypatch.setattr(window.volume_manager, "get_sitk_image", lambda: image)

    _extract(window, qtbot)
    window.screw_list_widget.setCurrentRow(0)
    assert "Extracted" in window.selected_screw_source.text()

    window._undo_action.trigger()

    assert window._tool_ctrl.screw_tool.get_screws() == []
    assert not window._undo_action.isEnabled()


def test_a_ct_the_extractor_cannot_read_warns_instead_of_raising(ui_main_window, qtbot, monkeypatch):
    window = ui_main_window
    image = _phantom([])
    image.SetDirection((0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0))
    monkeypatch.setattr(window.volume_manager, "get_sitk_image", lambda: image)
    warnings = []
    monkeypatch.setattr(
        controller_module.QMessageBox,
        "warning",
        lambda *args, **kwargs: warnings.append(args),
    )

    _extract(window, qtbot)

    assert window._tool_ctrl.screw_tool.get_screws() == []
    assert len(warnings) == 1 and "direction" in warnings[0][2]


def test_an_extraction_failure_warns_instead_of_escaping_the_slot(ui_main_window, qtbot, monkeypatch):
    window = ui_main_window
    empty = _phantom([])
    monkeypatch.setattr(window.volume_manager, "get_sitk_image", lambda: empty)

    def boom(_image):
        raise MemoryError("out of memory")

    monkeypatch.setattr(controller_module, "extract_screws", boom)
    warnings = []
    monkeypatch.setattr(
        controller_module.QMessageBox, "warning", lambda *args, **kwargs: warnings.append(args)
    )

    _extract(window, qtbot)

    assert len(warnings) == 1
    assert window._tool_ctrl.screw_tool.get_screws() == []


def test_a_bloomed_diameter_is_clamped_to_the_supported_range(ui_main_window, qtbot, monkeypatch):
    from src.core.screw_extraction import ExtractedScrew, ScrewExtraction
    from src.utils.constants import MAX_SCREW_DIAMETER

    window = ui_main_window
    empty = _phantom([])
    monkeypatch.setattr(window.volume_manager, "get_sitk_image", lambda: empty)
    found = ExtractedScrew((0.0, 60.0, 110.0), (0.0, 20.0, 110.0), 8.8, 40.0, 900, 0.9, "head at the wider end")
    monkeypatch.setattr(
        controller_module, "extract_screws", lambda _image: ScrewExtraction([found], "1 screw(s) found")
    )

    _extract(window, qtbot)

    screw = window._tool_ctrl.screw_tool.get_screws()[0]
    assert screw.diameter == MAX_SCREW_DIAMETER
    assert any("8.8 mm" in w for w in screw.warnings)


def test_extracted_screws_say_their_grade_is_only_indicative(ui_main_window, qtbot, monkeypatch):
    window = ui_main_window
    points = _grid()
    image = _phantom([_screw(points, LEFT_HEAD)])
    monkeypatch.setattr(window.volume_manager, "get_sitk_image", lambda: image)

    _extract(window, qtbot)

    screw = window._tool_ctrl.screw_tool.get_screws()[0]
    assert controller_module.POSTOP_GRADE_WARNING in screw.warnings


def test_a_result_for_a_ct_that_was_replaced_meanwhile_is_dropped(ui_main_window, qtbot, monkeypatch):
    window = ui_main_window
    points = _grid()
    images = [_phantom([_screw(points, LEFT_HEAD)])]
    monkeypatch.setattr(window.volume_manager, "get_sitk_image", lambda: images[-1])
    real = controller_module.extract_screws

    def slow_extract(image):
        images.append(_phantom([]))   # another study loads while extraction runs
        return real(image)

    monkeypatch.setattr(controller_module, "extract_screws", slow_extract)

    _extract(window, qtbot)

    assert window._tool_ctrl.screw_tool.get_screws() == []
    assert "discarded" in window.statusbar.currentMessage()
