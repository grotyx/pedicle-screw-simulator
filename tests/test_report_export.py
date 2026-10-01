"""One-click HTML planning report: content, privacy and the File-menu action."""

import os
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("pytestqt")
pytest.importorskip("SimpleITK")

import SimpleITK as sitk
from PyQt6.QtCore import QSettings
from PyQt6.QtWidgets import QApplication

import src.controllers.plan_controller as plan_controller_module
import src.ui.main_window as main_window_module
from src import __version__
from src.core.planner_config import PlannerConfig
from src.models.screw import Screw
from src.utils.report_export import (
    BANNER,
    IMAGE_UNAVAILABLE,
    ReportContext,
    build_report_html,
    capture_png,
)
from tests.test_ui_integration import DummyMPRViewer, DummyViewer3D

SERIES_UID = "1.2.826.0.1.3680043.8.498.77777"
PATIENT = "DOE^SYNTHETIC"


def _screw(level, side, grade="A", source="auto", breach=0.0, diameter=5.5):
    screw = Screw(
        entry_point=(0.0, 0.0, 0.0),
        target_point=(0.0, 40.0, 0.0),
        diameter=diameter,
        vertebra_level=level,
        side=side,
        grade=grade,
        breach_distance=breach,
        source=source,
        metrics={"medial_breach_mm": 1.5, "trajectory_mean_hu": 321.0},
    )
    screw.warnings = ["synthetic <b>warning</b>"]
    return screw


def _context(**kw):
    screws = [
        _screw("L5", "left", "B", breach=1.2),
        _screw("L4", "right", "A"),
        _screw("L4", "left", "C", source="manual", breach=2.5),
        _screw("T12", "right", "A"),
    ]
    return ReportContext(
        screws=screws,
        planner=PlannerConfig(
            mode="legacy",
            trajectory="cbt",
            accepted_breach_grade="B",
            narrow_pedicle_mm=4.5,
            place_uncontained_narrow=True,
        ),
        spacing=(0.5, 0.5, 1.25),
        size=(512, 512, 300),
        images={"Axial": b"\x89PNG-fake", "3D": None},
        generated_at="2026-10-01 09:30",
        **kw,
    )


def test_report_content():
    html = build_report_html(_context())
    assert BANNER in html
    assert f"Version {__version__}" in html and "2026-10-01 09:30" in html
    for text in ("legacy", "cbt", "4.5", "Place narrow screws even if not contained"):
        assert text in html
    assert "0.500 × 0.500 × 1.250" in html and "512 × 512 × 300" in html
    # One table row per screw (header + 4) in the screw table, values from the helpers.
    screw_table = html.split("<h2>Screws</h2>")[1].split("<h2>Construct map</h2>")[0]
    assert screw_table.count("<tr>") == 5
    assert "5.5 × 40.0" in screw_table and "Grade C" in screw_table
    assert "2.5" in screw_table and "1.5" in screw_table and "321" in screw_table
    assert "manual" in screw_table
    # Warnings are escaped, never injected as markup.
    assert "synthetic &lt;b&gt;warning&lt;/b&gt;" in screw_table and "<b>warning" not in html
    assert html.count("data:image/png;base64,") == 1
    assert html.count(IMAGE_UNAVAILABLE) == 1  # the 3D pane


def test_construct_grid_is_cranial_first_right_then_left():
    html = build_report_html(_context())
    grid = html.split('<table class="construct">')[1].split("</table>")[0]
    assert grid.index("T12") < grid.index("L4") < grid.index("L5")
    l4 = grid.split("L4</td>")[1].split("</tr>")[0]
    assert l4.index("#2 A") < l4.index("#3 C")  # right cell before left cell


def test_no_images_says_unavailable():
    assert IMAGE_UNAVAILABLE in build_report_html(ReportContext())


def test_capture_failure_is_not_fatal():
    assert capture_png(object()) is None


def test_report_has_no_identifiers():
    html = build_report_html(_context())
    for forbidden in (SERIES_UID, PATIENT, str(Path.home()), "Patient"):
        assert forbidden not in html


# ---------------------------------------------------------------- File menu


@pytest.fixture
def window(tmp_path, monkeypatch, qtbot):
    directory = tmp_path / "qsettings"
    directory.mkdir()

    def factory(organization="Default", application="App", *_a, **_k):
        return QSettings(
            str(directory / f"{organization}-{application}.ini"),
            QSettings.Format.IniFormat,
        )

    monkeypatch.setattr(main_window_module, "QSettings", factory)
    monkeypatch.setattr(main_window_module, "_migrated", False)
    monkeypatch.setattr(main_window_module, "MPRViewer", DummyMPRViewer)
    monkeypatch.setattr(main_window_module, "Viewer3D", DummyViewer3D)
    QApplication.instance().setProperty("themeName", "soft_light")
    win = main_window_module.MainWindow()
    qtbot.addWidget(win)
    return win


class _Progress:
    def close(self):
        return None


def _report_action(win):
    return next(
        a for a in win.menuBar().actions()[0].menu().actions() if "Report" in a.text()
    )


def test_menu_action_disabled_without_volume_then_writes_report(
    window, monkeypatch, tmp_path
):
    action = _report_action(window)
    file_menu = window.menuBar().actions()[0].menu()
    file_menu.aboutToShow.emit()
    assert not action.isEnabled()

    image = sitk.Image([16, 16, 12], sitk.sitkInt16)
    image.SetSpacing((1.0, 1.0, 1.2))
    window._on_dicom_loaded(
        image=image,
        metadata={
            "series_id": SERIES_UID,
            "patient_name": PATIENT,
            "num_slices": 12,
        },
        progress=_Progress(),
    )
    window._tool_ctrl.screw_tool.add_screw(_screw("L4", "right"))
    file_menu.aboutToShow.emit()
    assert action.isEnabled()

    target = tmp_path / "out" / "report.html"
    monkeypatch.setattr(
        plan_controller_module.QFileDialog,
        "getSaveFileName",
        staticmethod(lambda *a, **k: (str(target), "HTML Files (*.html)")),
    )
    action.trigger()

    html = target.read_text(encoding="utf-8")
    assert BANNER in html and "L4" in html and "1.200" in html
    for forbidden in (SERIES_UID, PATIENT, str(tmp_path), str(Path.home())):
        assert forbidden not in html
    assert IMAGE_UNAVAILABLE in html  # stub viewers have no render window
