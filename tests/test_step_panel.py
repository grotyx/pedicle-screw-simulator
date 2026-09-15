"""Step-panel builder tests: each builder returns page + refs for controls."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("pytestqt")

from src.ui.step_panel import (
    STEP_NAMES,
    StepPanel,
    build_plan_page,
    build_review_page,
    build_segment_page,
    build_study_page,
)


def test_study_builder_returns_page_and_refs(qtbot):
    page, refs = build_study_page()
    qtbot.addWidget(page)

    for name in (
        "open_dicom_btn",
        "info_label",
        "window_slider",
        "level_slider",
        "_btn_bone",
        "_btn_soft",
        "tf_preset_combo",
        "opacity_slider",
        "info_group",
        "wl_group",
        "view_group",
    ):
        assert name in refs
        assert page.isAncestorOf(refs[name])


def test_segment_builder_returns_page_and_refs(qtbot):
    page, refs = build_segment_page()
    qtbot.addWidget(page)

    for name in (
        "seg_run_btn",
        "seg_refine_check",
        "seg_status_label",
        "vertebra_isolate_btn",
        "isolation_hint_label",
        "vertebra_level_container",
        "seg_advanced_panel",
        "seg_task_combo",
        "seg_device_combo",
        "seg_quality_combo",
        "seg_label_spin",
        "seg_label_combo",
        "seg_label_info",
        "seg_show_2d_check",
        "seg_show_3d_check",
        "seg_clear_btn",
        "vertebra_restore_btn",
        "vertebra_display_list",
        "vertebra_show_selected_btn",
        "vertebra_show_all_btn",
        "seg_use_subregion_check",
        "seg_subregion_dir_edit",
        "seg_subregion_browse_btn",
        "seg_advanced_toggle",
    ):
        assert name in refs
    for name in (
        "seg_run_btn",
        "seg_status_label",
        "vertebra_isolate_btn",
        "isolation_hint_label",
        "seg_advanced_toggle",
        "seg_advanced_panel",
    ):
        assert page.isAncestorOf(refs[name])
    # Level grid lives in its container until the Plan page adopts it.
    assert refs["vertebra_level_container"].isAncestorOf(
        refs["_vertebra_level_placeholder"]
    )


def test_plan_builder_returns_page_and_refs(qtbot):
    _segment_page, segment_refs = build_segment_page()
    qtbot.addWidget(_segment_page)
    page, refs = build_plan_page(segment_refs["vertebra_level_container"])
    qtbot.addWidget(page)

    for name in (
        "vertebra_select_all_btn",
        "vertebra_clear_all_btn",
        "workspace_mode_combo",
        "auto_screw_review_notice",
        "auto_screw_plan_btn",
        "auto_screw_status",
        "_btn_clear_screws",
        "plan_mode_combo",
        "plan_trajectory_combo",
        "plan_fill_ratio_spin",
        "plan_wall_clearance_spin",
        "plan_anterior_margin_spin",
        "plan_max_convergence_spin",
        "plan_hu_threshold_spin",
        "plan_narrow_pedicle_spin",
        "plan_narrow_lateral_spin",
        "plan_weight_safety",
        "plan_weight_density",
        "plan_weight_rod",
        "plan_endplate_parallel_check",
        "plan_endplate_tolerance_spin",
        "plan_reset_defaults_btn",
        "length_spin",
        "diameter_spin",
    ):
        assert name in refs
    assert page.isAncestorOf(refs["auto_screw_plan_btn"])
    assert page.isAncestorOf(segment_refs["vertebra_level_container"])


def test_review_builder_returns_page_and_refs(qtbot):
    page, refs = build_review_page("soft_light")
    qtbot.addWidget(page)

    for name in (
        "review_title_label",
        "selected_screw_counter",
        "selected_screw_title",
        "selected_screw_grade",
        "selected_screw_diameter",
        "selected_screw_length",
        "screw_warnings_toggle",
        "selected_screw_warning",
        "screw_list_widget",
        "screw_filter_edit",
        "screw_previous_btn",
        "screw_next_btn",
        "screw_axis_mpr_btn",
        "standard_mpr_btn",
        "screw_edit_btn",
        "_screw_edit_move_entry_action",
        "_screw_edit_move_tip_action",
        "_screw_edit_move_whole_action",
        "_screw_edit_cancel_action",
        "remove_screw_btn",
        "screw_axis_position_label",
        "screw_axis_position_slider",
        "screw_axis_rotation_spin",
        "screw_mpr_reset_btn",
        "screw_mpr_controls",
        "selected_screw_metrics",
        "screw_narrow_legend",
        "screw_drag_hint",
        "measure_mode_combo",
        "measure_finish_btn",
        "measure_clear_btn",
        "measurement_list_widget",
        "show_measurement_btn",
        "edit_measurement_btn",
        "remove_measurement_btn",
    ):
        assert name in refs
    assert page.isAncestorOf(refs["screw_list_widget"])
    assert page.isAncestorOf(refs["screw_edit_btn"])
    assert page.isAncestorOf(refs["remove_screw_btn"])


def test_step_panel_registers_all_four_pages(qtbot):
    panel = StepPanel()
    qtbot.addWidget(panel)
    study_page, _study = build_study_page()
    segment_page, segment_refs = build_segment_page()
    plan_page, _plan = build_plan_page(
        segment_refs["vertebra_level_container"]
    )
    review_page, _review = build_review_page("soft_light")

    for name, page in (
        ("Study", study_page),
        ("Segment", segment_page),
        ("Plan", plan_page),
        ("Review", review_page),
    ):
        panel.add_page(name, page)

    assert STEP_NAMES == ("Study", "Segment", "Plan", "Review")
    for name in STEP_NAMES:
        assert panel.page(name) is not None

    panel.show_step("Plan")
    assert panel.current_step == "Plan"
    panel.show_step("Review")
    assert panel.current_step == "Review"
