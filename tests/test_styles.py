"""Tests for selectable application theme palettes."""

from src.ui.styles import (
    DEFAULT_THEME,
    THEME_LABELS,
    THEMES,
    get_theme,
    load_stylesheet,
)

REQUIRED_TOKENS = {
    "bg_primary",
    "bg_secondary",
    "bg_tertiary",
    "bg_toolbar",
    "accent",
    "accent_hover",
    "accent_pressed",
    "accent_dim",
    "implant",
    "text_primary",
    "text_secondary",
    "text_disabled",
    "border",
    "border_light",
    "danger",
    "success",
    "warning",
    "button_text",
    "secondary_action_text",
    "warning_bg",
    "warning_text",
    "danger_bg",
    "danger_border",
    "danger_text",
    "viewer_header",
    "viewer_readout",
    "viewer_bg",
    "viewer_foreground",
    "viewer_separator",
    "grade_a",
    "grade_b",
    "grade_c",
    "grade_d",
    "grade_na",
    "grade_text",
    "table_alternate",
    "table_header",
    "scrollbar_bg",
    "scrollbar_handle",
    "scrollbar_hover",
}


def test_graphite_blue_is_default_and_all_visual_options_exist():
    assert DEFAULT_THEME == "graphite_blue"
    assert list(THEME_LABELS) == [
        "soft_light",
        "graphite_blue",
        "graphite_mint",
    ]


def test_all_themes_supply_the_same_required_semantic_tokens():
    assert set(THEMES) == set(THEME_LABELS)
    for palette in THEMES.values():
        assert REQUIRED_TOKENS <= set(palette)


def test_unknown_theme_falls_back_to_the_default():
    assert get_theme("missing") == THEMES["graphite_blue"]


def test_each_theme_renders_its_distinct_primary_surface():
    assert "#F3F5F7" in load_stylesheet("soft_light")
    assert "#0D0F12" in load_stylesheet("graphite_blue")
    assert "#10110F" in load_stylesheet("graphite_mint")


def test_generated_stylesheets_do_not_use_gradients():
    for theme_name in THEMES:
        stylesheet = load_stylesheet(theme_name).lower()
        assert "gradient" not in stylesheet


def test_font_stack_leads_with_a_windows_native_face():
    stylesheet = load_stylesheet("graphite_blue")
    assert (
        'font-family: "Segoe UI", "SF Pro Text", "Avenir Next", '
        '"Helvetica Neue", Arial;'
    ) in stylesheet
    assert "font-size: 13px;" in stylesheet


def test_every_theme_keeps_dark_viewports_behind_a_visible_seam():
    for theme_name, palette in THEMES.items():
        stylesheet = load_stylesheet(theme_name)
        assert palette["viewer_bg"] == "#050A0F"
        assert f'background-color: {palette["viewer_bg"]};' in stylesheet
        assert f'border: 1px solid {palette["viewer_separator"]};' in stylesheet
        assert palette["viewer_separator"] != palette["viewer_bg"]


def test_grade_chip_colours_are_distinct_within_every_theme():
    for palette in THEMES.values():
        chips = [
            palette["grade_a"],
            palette["grade_b"],
            palette["grade_c"],
            palette["grade_d"],
            palette["grade_na"],
        ]
        assert len(set(chips)) == 5
