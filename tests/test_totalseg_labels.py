"""
Tests for TotalSegmentator label mapping helpers.
"""

import os
import sys
import types

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.core.totalseg_labels import (
    _normalize_label_map,
    format_label_display,
    format_selected_label_text,
    get_segmentation_label_map,
    get_totalseg_task_label_map,
)


class TestTotalsegLabelHelpers:
    """Coverage for label map normalization/formatting."""

    def test_normalize_label_map_accepts_int_and_numeric_string_keys(self):
        normalized = _normalize_label_map(
            {
                1: "spleen",
                "2": "kidney_right",
                0: "background",
                "abc": "invalid",
            }
        )

        assert normalized == {
            1: "spleen",
            2: "kidney_right",
        }

    def test_get_totalseg_task_label_map_uses_imported_class_map(self, monkeypatch):
        fake_pkg = types.ModuleType("totalsegmentator")
        fake_map_module = types.ModuleType("totalsegmentator.map_to_binary")
        fake_map_module.class_map = {
            "body": {
                1: "body_trunc",
                2: "body_extremities",
            }
        }

        monkeypatch.setitem(sys.modules, "totalsegmentator", fake_pkg)
        monkeypatch.setitem(sys.modules, "totalsegmentator.map_to_binary", fake_map_module)

        result = get_totalseg_task_label_map("body")

        assert result == {
            1: "body_trunc",
            2: "body_extremities",
        }

    def test_get_segmentation_label_map_returns_fallback_map(self):
        result = get_segmentation_label_map(task="total", method="threshold_fallback")
        assert result == {1: "threshold_bone"}

    def test_format_helpers_return_readable_text(self):
        label_map = {
            1: "vertebrae_L4",
            2: "vertebrae_L5",
        }

        assert format_label_display(1, "vertebrae_L4") == "1: vertebrae L4"
        assert format_selected_label_text(0, label_map) == "Selected: All labels"
        assert format_selected_label_text(2, label_map) == "Selected: vertebrae L5 (ID 2)"
        assert format_selected_label_text(999, label_map) == "Selected: Unknown label ID 999"
