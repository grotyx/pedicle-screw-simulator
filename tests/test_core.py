"""
Tests for core module (DicomLoader, VolumeManager, CoordinateSystem)
"""

import pytest
import numpy as np
import importlib

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.core.coordinate_system import CoordinateSystem
from src.utils.vtk_helpers import world_to_volume_index, volume_index_to_world


class TestCoordinateSystem:
    """Tests for coordinate system transformations."""

    def test_lps_to_ras_origin(self):
        """Test LPS to RAS at origin."""
        point = (0, 0, 0)
        result = CoordinateSystem.lps_to_ras(point)
        assert result == (0, 0, 0)

    def test_lps_to_ras_positive(self):
        """Test LPS to RAS with positive values."""
        point = (10, 20, 30)
        result = CoordinateSystem.lps_to_ras(point)
        # X and Y are negated, Z stays same
        assert result == (-10, -20, 30)

    def test_lps_to_ras_negative(self):
        """Test LPS to RAS with negative values."""
        point = (-10, -20, -30)
        result = CoordinateSystem.lps_to_ras(point)
        assert result == (10, 20, -30)

    def test_ras_to_lps_inverse(self):
        """Test that RAS to LPS is inverse of LPS to RAS."""
        original = (15, 25, 35)
        ras = CoordinateSystem.lps_to_ras(original)
        back = CoordinateSystem.ras_to_lps(ras)
        assert back == original

    def test_self_inverse_transform(self):
        """Test that applying transform twice returns to original."""
        point = (100, 200, 300)
        once = CoordinateSystem.lps_to_ras(point)
        twice = CoordinateSystem.lps_to_ras(once)
        # Since transform is self-inverse (multiply by -1 twice = original)
        assert twice == point

    def test_image_to_world_identity(self):
        """Test image to world with identity direction."""
        index = (10, 20, 30)
        origin = (0, 0, 0)
        spacing = (1, 1, 1)

        result = CoordinateSystem.image_to_world(index, origin, spacing)
        assert result == (10, 20, 30)

    def test_image_to_world_with_spacing(self):
        """Test image to world with non-unit spacing."""
        index = (10, 20, 30)
        origin = (0, 0, 0)
        spacing = (0.5, 0.5, 2.0)

        result = CoordinateSystem.image_to_world(index, origin, spacing)
        assert result == (5.0, 10.0, 60.0)

    def test_image_to_world_with_origin(self):
        """Test image to world with non-zero origin."""
        index = (10, 20, 30)
        origin = (100, 200, 300)
        spacing = (1, 1, 1)

        result = CoordinateSystem.image_to_world(index, origin, spacing)
        assert result == (110, 220, 330)

    def test_world_to_image_inverse(self):
        """Test that world_to_image is inverse of image_to_world."""
        original_index = (10, 20, 30)
        origin = (5, 10, 15)
        spacing = (0.5, 0.5, 1.0)

        world = CoordinateSystem.image_to_world(original_index, origin, spacing)
        back_index = CoordinateSystem.world_to_image(world, origin, spacing)
        assert back_index == original_index

    def test_get_plane_normal_axial(self):
        """Test axial plane normal."""
        normal = CoordinateSystem.get_plane_normal("axial")
        assert normal == (0, 0, 1)

    def test_get_plane_normal_sagittal(self):
        """Test sagittal plane normal."""
        normal = CoordinateSystem.get_plane_normal("sagittal")
        assert normal == (1, 0, 0)

    def test_get_plane_normal_coronal(self):
        """Test coronal plane normal."""
        normal = CoordinateSystem.get_plane_normal("coronal")
        assert normal == (0, 1, 0)


class TestVTKHelpers:
    """Tests for VTK helper functions."""

    def test_constants_exist(self):
        """Test that constants are defined."""
        from src.utils.constants import (
            LPS_TO_RAS_MATRIX,
            HU_BONE_OPTIMAL,
            DEFAULT_WINDOW_CENTER,
            DEFAULT_WINDOW_WIDTH
        )

        assert LPS_TO_RAS_MATRIX is not None
        assert HU_BONE_OPTIMAL == 400
        assert DEFAULT_WINDOW_CENTER == 400
        assert DEFAULT_WINDOW_WIDTH == 1500

    def test_lps_to_ras_matrix_shape(self):
        """Test LPS to RAS matrix has correct shape."""
        from src.utils.constants import LPS_TO_RAS_MATRIX

        assert LPS_TO_RAS_MATRIX.shape == (4, 4)

    def test_lps_to_ras_matrix_values(self):
        """Test LPS to RAS matrix has correct values."""
        from src.utils.constants import LPS_TO_RAS_MATRIX

        expected = np.array([
            [-1,  0,  0,  0],
            [ 0, -1,  0,  0],
            [ 0,  0,  1,  0],
            [ 0,  0,  0,  1]
        ])

        np.testing.assert_array_equal(LPS_TO_RAS_MATRIX, expected)

    def test_world_index_roundtrip_with_direction(self):
        """Test world/index conversions for non-identity direction."""
        direction = (
            0, -1, 0,
            1, 0, 0,
            0, 0, 1,
        )
        origin = (10.0, 20.0, 30.0)
        spacing = (2.0, 3.0, 4.0)
        original_index = (2, 1, 3)

        world = volume_index_to_world(
            original_index,
            origin=origin,
            spacing=spacing,
            direction=direction,
        )
        roundtrip_index = world_to_volume_index(
            world,
            origin=origin,
            spacing=spacing,
            direction=direction,
        )

        assert roundtrip_index == original_index


class TestCoreModule:
    """Tests for core package imports."""

    def test_core_import_without_eager_heavy_dependencies(self):
        """Importing src.core should not eagerly require SimpleITK."""
        module = importlib.import_module("src.core")
        assert hasattr(module, "CoordinateSystem")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
