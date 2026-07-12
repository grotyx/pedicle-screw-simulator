"""
Tests for volume scale assessment helpers.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.core.volume_scale import (
    LARGE_REFERENCE_SIZE,
    LARGE_VOLUME_MIN_VOXELS,
    MEDIUM_VOLUME_MIN_VOXELS,
    XL_VOLUME_MIN_VOXELS,
    assess_volume_scale,
    estimate_voxel_count,
    format_scale_summary,
)


class TestVolumeScale:
    """Coverage for volume scale categorization."""

    def test_estimate_voxel_count(self):
        assert estimate_voxel_count((512, 512, 700)) == 183_500_800

    def test_assess_small_volume(self):
        result = assess_volume_scale((256, 256, 100))
        assert result.tier == "small"
        assert result.meets_large_criteria is False

    def test_assess_medium_volume_boundary(self):
        result = assess_volume_scale((MEDIUM_VOLUME_MIN_VOXELS, 1, 1))
        assert result.tier == "medium"
        assert result.meets_large_criteria is False

    def test_assess_large_volume_boundary(self):
        result = assess_volume_scale((LARGE_VOLUME_MIN_VOXELS, 1, 1))
        assert result.tier == "large"
        assert result.meets_large_criteria is True

    def test_assess_xl_volume_boundary(self):
        result = assess_volume_scale((XL_VOLUME_MIN_VOXELS, 1, 1))
        assert result.tier == "xl"
        assert result.meets_large_criteria is True

    def test_large_reference_is_classified_large(self):
        result = assess_volume_scale(LARGE_REFERENCE_SIZE)
        assert result.meets_large_criteria is True
        assert result.tier in {"large", "xl"}

    def test_invalid_size_raises(self):
        with pytest.raises(ValueError):
            estimate_voxel_count((512, 512))
        with pytest.raises(ValueError):
            estimate_voxel_count((512, 512, 0))

    def test_format_scale_summary_contains_key_fields(self):
        result = assess_volume_scale((512, 512, 700))
        summary = format_scale_summary(result)
        assert "size=512x512x700" in summary
        assert "tier=" in summary
        assert "large=True" in summary
