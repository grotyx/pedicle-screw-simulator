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
    "table_alternate",
    "table_header",
    "scrollbar_bg",
    "scrollbar_handle",
    "scrollbar_hover",
}


def test_soft_light_is_default_and_all_visual_options_exist():
    assert DEFAULT_THEME == "soft_light"
    assert list(THEME_LABELS) == [
        "soft_light",
        "graphite_blue",
        "graphite_mint",
    ]


def test_all_themes_supply_the_same_required_semantic_tokens():
    assert set(THEMES) == set(THEME_LABELS)
    for palette in THEMES.values():
        assert REQUIRED_TOKENS <= set(palette)


def test_unknown_theme_falls_back_to_soft_light():
    assert get_theme("missing") == THEMES["soft_light"]


def test_each_theme_renders_its_distinct_primary_surface():
    assert "#F3F5F7" in load_stylesheet("soft_light")
    assert "#0D0F12" in load_stylesheet("graphite_blue")
    assert "#10110F" in load_stylesheet("graphite_mint")


def test_generated_stylesheets_do_not_use_gradients():
    for theme_name in THEMES:
        stylesheet = load_stylesheet(theme_name).lower()
        assert "gradient" not in stylesheet
