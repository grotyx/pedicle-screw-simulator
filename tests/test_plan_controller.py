"""PlanController tests for the code-review fix wave (F8).

``isolated_qsettings``/``ui_main_window`` here are thin shims over
``tests.conftest`` (kept so old imports keep working).
"""

import json
import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("pytestqt")
pytest.importorskip("SimpleITK")

import SimpleITK as sitk

import src.controllers.plan_controller as plan_controller_module
from src.utils.constants import COLOR_SCREW, COLOR_SCREW_BREACH
from tests.conftest import make_isolated_qsettings, make_ui_main_window


@pytest.fixture
def isolated_qsettings(tmp_path, monkeypatch):
    """Thin shim over tests.conftest (kept so old imports keep working)."""
    return make_isolated_qsettings(tmp_path, monkeypatch)


@pytest.fixture
def ui_main_window(monkeypatch, qtbot, isolated_qsettings):
    """Thin shim over tests.conftest (kept so old imports keep working)."""
    return make_ui_main_window(monkeypatch, qtbot, isolated_qsettings)


class _ProgressStub:
    def close(self):
        return None


def _create_test_image():
    image = sitk.Image([16, 16, 12], sitk.sitkInt16)
    image = image + 250
    image.SetSpacing((1.0, 1.0, 1.2))
    return image


def _load_volume(window):
    """`load_dialog` refuses (modally) to run without a volume."""
    image = _create_test_image()
    window._on_dicom_loaded(
        image=image,
        metadata={"series_id": "SERIES-V1", "num_slices": image.GetSize()[2]},
        progress=_ProgressStub(),
    )


def _v1_plan_payload(grade="A"):
    """A plan as the removed HU-heuristic grader used to write them."""
    return {
        "version": 1,
        "series_id": "SERIES-V1",
        "screws": [
            {
                "entry_point": [10.0, 20.0, 30.0],
                "target_point": [10.0, -10.0, 30.0],
                "length": 30.0,
                "diameter": 6.0,
                "vertebra_level": "L4",
                "side": "left",
                "grade": grade,
                "breach_distance": 0.0,
                "mean_hu": 420.0,
                "min_hu": 210.0,
                "warnings": [],
                "source": "manual",
                "metrics": {},
            }
        ],
        "measurements": [],
    }


def _plan_payload_with_metrics(metrics):
    """A plan payload for one screw carrying the given ``metrics`` dict."""
    payload = _v1_plan_payload()
    payload["screws"][0]["metrics"] = metrics
    return payload


def _load_plan(window, monkeypatch, tmp_path, payload):
    """Drive ``load_dialog`` against a plan file, with the dialogs silenced."""
    _load_volume(window)
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(
        plan_controller_module.QMessageBox, "warning", lambda *a, **k: None
    )
    monkeypatch.setattr(
        plan_controller_module.QFileDialog,
        "getOpenFileName",
        staticmethod(lambda *a, **k: (str(path), "JSON Files (*.json)")),
    )
    monkeypatch.setattr(
        plan_controller_module.QMessageBox, "critical", lambda *a, **k: None
    )
    monkeypatch.setattr(
        plan_controller_module.QMessageBox,
        "question",
        lambda *a, **k: plan_controller_module.QMessageBox.StandardButton.Yes,
    )
    window._plan_ctrl.load_dialog()
    return path


def test_loading_a_v1_plan_without_a_grader_shows_na_not_the_saved_grade(
    ui_main_window, monkeypatch, tmp_path
):
    """A grade nothing can vouch for must not be displayed as current.

    v1 plans were graded by an HU heuristic that no longer exists; with no
    segmentation loaded there is no grader, and the stored "A" would otherwise
    be shown as if it had just been measured.
    """
    window = ui_main_window
    assert window._tool_ctrl.screw_tool.grader is None

    _load_plan(window, monkeypatch, tmp_path, _v1_plan_payload(grade="A"))

    screws = window._tool_ctrl.screw_tool.get_screws()
    assert len(screws) == 1
    assert screws[0].grade == "N/A"


def test_loading_a_plan_regrades_even_when_no_grader_is_attached(
    ui_main_window, monkeypatch, tmp_path
):
    window = ui_main_window
    regrades = []
    monkeypatch.setattr(
        window._tool_ctrl, "regrade_all", lambda: regrades.append(True)
    )

    _load_plan(window, monkeypatch, tmp_path, _v1_plan_payload())

    assert regrades == [True]


def test_the_screw_list_row_reflects_the_regraded_screw(
    ui_main_window, monkeypatch, tmp_path
):
    """The list row is built before the re-grade, so it must be refreshed."""
    window = ui_main_window

    _load_plan(window, monkeypatch, tmp_path, _v1_plan_payload(grade="A"))

    assert window.screw_list_widget.count() == 1
    assert "Grade N/A" in window.screw_list_widget.rowText(0)


def test_saved_plans_record_the_mask_refinement_settings(
    ui_main_window, monkeypatch, tmp_path
):
    """A plan is only as good as the mask its screws were measured against."""
    from src.core.mask_refinement import CT_GUIDED_NOTE

    window = ui_main_window
    _load_volume(window)
    window._seg_ctrl._last_raw_mask_path = "raw.nii.gz"
    window._seg_ctrl._last_refinement_notes = [CT_GUIDED_NOTE]

    path = tmp_path / "saved.json"
    monkeypatch.setattr(
        plan_controller_module.QFileDialog,
        "getSaveFileName",
        staticmethod(lambda *a, **k: (str(path), "JSON Files (*.json)")),
    )

    window._plan_ctrl.save_dialog()

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["metadata"]["mask_refinement"] == {
        "enabled": True,
        "ct_guided": True,
    }


def test_loading_a_narrow_pedicle_screw_colours_the_3d_screw_red_on_first_draw(
    ui_main_window, monkeypatch, tmp_path
):
    """The 3D screw must be red the instant it is drawn, not only after regrade.

    ``_apply_loaded_plan`` used to add the 3D screw with no ``color=`` at all,
    relying on the later unconditional ``regrade_all()`` call to repaint it.
    Stubbing out ``regrade_all`` here isolates the first draw: if the initial
    ``add_screw`` call is still colourless, this catches it even though the
    final on-screen state would otherwise self-heal.
    """
    window = ui_main_window
    monkeypatch.setattr(window._tool_ctrl, "regrade_all", lambda: None)

    payload = _plan_payload_with_metrics({"narrow_pedicle": True})
    _load_plan(window, monkeypatch, tmp_path, payload)

    expected = tuple(value / 255.0 for value in COLOR_SCREW_BREACH)
    assert window.viewer_3d.screws[0].color == expected


def test_loading_a_normal_screw_colours_the_3d_screw_with_the_default_colour(
    ui_main_window, monkeypatch, tmp_path
):
    window = ui_main_window
    monkeypatch.setattr(window._tool_ctrl, "regrade_all", lambda: None)

    payload = _plan_payload_with_metrics({"narrow_pedicle": False})
    _load_plan(window, monkeypatch, tmp_path, payload)

    expected = tuple(value / 255.0 for value in COLOR_SCREW)
    assert window.viewer_3d.screws[0].color == expected


def test_export_stl_runs_without_a_job_dialog(ui_main_window, monkeypatch, tmp_path):
    """STL export is synchronous: no dialog surface while the GUI is blocked."""
    import src.controllers.plan_controller as plan_module

    window = ui_main_window
    window._on_dicom_loaded(
        image=_create_test_image(),
        metadata={"series_id": "SERIES-STL", "num_slices": 12},
        progress=_ProgressStub(),
    )
    path = tmp_path / "bone.stl"
    monkeypatch.setattr(
        plan_module.QFileDialog,
        "getSaveFileName",
        staticmethod(lambda *a, **k: (str(path), "STL Files (*.stl)")),
    )
    created = []
    real_dialog = plan_module.JobDialog if hasattr(plan_module, "JobDialog") else None
    assert real_dialog is None

    def _boom(*_a, **_k):
        created.append(True)
        raise AssertionError("STL export must not build a JobDialog")

    monkeypatch.setattr("src.ui.job_dialog.JobDialog", _boom)
    calls = []
    monkeypatch.setattr(
        plan_module, "export_bone_stl", lambda image, out: calls.append(out)
    )

    window._plan_ctrl.export_stl_dialog()

    assert created == []
    assert calls == [str(path)]
    assert window.statusbar.currentMessage() == f"STL exported: {path}"
