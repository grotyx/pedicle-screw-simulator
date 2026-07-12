"""
Tests for sitk_to_vtk() conversion in src/utils/vtk_helpers.py

Phase 0 TDD: These tests define the expected behavior BEFORE modifying
the function. All tests should pass with both the current implementation
and the optimized (2-copy) implementation.
"""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import SimpleITK as sitk
import vtk
from vtk.util import numpy_support

from src.utils.vtk_helpers import sitk_to_vtk


def _make_sitk_image(
    array: np.ndarray,
    spacing=(1.0, 1.0, 1.0),
    origin=(0.0, 0.0, 0.0),
) -> sitk.Image:
    """Helper: create SimpleITK image from numpy array (z,y,x ordering)."""
    image = sitk.GetImageFromArray(array)
    image.SetSpacing(spacing)
    image.SetOrigin(origin)
    return image


class TestSitkToVtkDimensions:
    """Verify VTK image dimensions match SimpleITK image size."""

    def test_small_cube(self):
        arr = np.zeros((10, 20, 30), dtype=np.int16)  # z=10, y=20, x=30
        img = _make_sitk_image(arr)
        vtk_img = sitk_to_vtk(img)
        assert vtk_img.GetDimensions() == (30, 20, 10)

    def test_asymmetric_volume(self):
        arr = np.zeros((5, 64, 128), dtype=np.int16)
        img = _make_sitk_image(arr)
        vtk_img = sitk_to_vtk(img)
        assert vtk_img.GetDimensions() == (128, 64, 5)

    def test_single_slice(self):
        arr = np.zeros((1, 32, 32), dtype=np.int16)
        img = _make_sitk_image(arr)
        vtk_img = sitk_to_vtk(img)
        assert vtk_img.GetDimensions() == (32, 32, 1)


class TestSitkToVtkSpacing:
    """Verify VTK image spacing matches SimpleITK spacing."""

    def test_unit_spacing(self):
        arr = np.zeros((4, 4, 4), dtype=np.int16)
        img = _make_sitk_image(arr, spacing=(1.0, 1.0, 1.0))
        vtk_img = sitk_to_vtk(img)
        assert vtk_img.GetSpacing() == pytest.approx((1.0, 1.0, 1.0))

    def test_anisotropic_spacing(self):
        arr = np.zeros((4, 4, 4), dtype=np.int16)
        img = _make_sitk_image(arr, spacing=(0.488, 0.488, 2.5))
        vtk_img = sitk_to_vtk(img)
        assert vtk_img.GetSpacing() == pytest.approx((0.488, 0.488, 2.5))

    def test_fine_spacing(self):
        arr = np.zeros((4, 4, 4), dtype=np.int16)
        img = _make_sitk_image(arr, spacing=(0.3125, 0.3125, 0.625))
        vtk_img = sitk_to_vtk(img)
        assert vtk_img.GetSpacing() == pytest.approx((0.3125, 0.3125, 0.625))


class TestSitkToVtkOrigin:
    """Verify VTK image origin matches SimpleITK origin."""

    def test_zero_origin(self):
        arr = np.zeros((4, 4, 4), dtype=np.int16)
        img = _make_sitk_image(arr, origin=(0.0, 0.0, 0.0))
        vtk_img = sitk_to_vtk(img)
        assert vtk_img.GetOrigin() == pytest.approx((0.0, 0.0, 0.0))

    def test_nonzero_origin(self):
        arr = np.zeros((4, 4, 4), dtype=np.int16)
        img = _make_sitk_image(arr, origin=(-125.0, -200.5, 33.7))
        vtk_img = sitk_to_vtk(img)
        assert vtk_img.GetOrigin() == pytest.approx((-125.0, -200.5, 33.7))


class TestSitkToVtkVoxelValues:
    """Verify voxel values survive the conversion at multiple coordinates."""

    def test_corner_values_int16(self):
        """Check that values at 8 corners of a volume are preserved."""
        rng = np.random.RandomState(42)
        arr = rng.randint(-1000, 3000, size=(8, 16, 32), dtype=np.int16)
        img = _make_sitk_image(arr, spacing=(0.5, 0.5, 1.0))
        vtk_img = sitk_to_vtk(img)

        # Test corners: (x, y, z) in VTK indexing
        test_points = [
            (0, 0, 0),
            (31, 0, 0),
            (0, 15, 0),
            (0, 0, 7),
            (31, 15, 7),
            (15, 8, 4),  # center-ish
        ]
        for x, y, z in test_points:
            vtk_val = vtk_img.GetScalarComponentAsFloat(x, y, z, 0)
            # SimpleITK array is (z, y, x)
            sitk_val = float(arr[z, y, x])
            assert vtk_val == sitk_val, (
                f"Mismatch at VTK({x},{y},{z}): vtk={vtk_val}, sitk={sitk_val}"
            )

    def test_known_pattern(self):
        """Use a deterministic pattern to verify coordinate mapping."""
        # Each voxel stores its flattened (x,y,z) index
        shape_zyx = (4, 6, 8)  # z=4, y=6, x=8
        arr = np.zeros(shape_zyx, dtype=np.int16)
        for z in range(shape_zyx[0]):
            for y in range(shape_zyx[1]):
                for x in range(shape_zyx[2]):
                    arr[z, y, x] = x * 100 + y * 10 + z
        img = _make_sitk_image(arr)
        vtk_img = sitk_to_vtk(img)

        for z in range(shape_zyx[0]):
            for y in range(shape_zyx[1]):
                for x in range(shape_zyx[2]):
                    expected = x * 100 + y * 10 + z
                    actual = vtk_img.GetScalarComponentAsFloat(x, y, z, 0)
                    assert actual == expected, (
                        f"Pattern mismatch at ({x},{y},{z}): "
                        f"expected={expected}, actual={actual}"
                    )


class TestSitkToVtkDtype:
    """Verify correct VTK array type for different numpy dtypes."""

    def _get_vtk_scalar_type(self, vtk_img):
        return vtk_img.GetScalarType()

    def test_int16(self):
        arr = np.zeros((4, 4, 4), dtype=np.int16)
        img = _make_sitk_image(arr)
        vtk_img = sitk_to_vtk(img)
        assert self._get_vtk_scalar_type(vtk_img) == vtk.VTK_SHORT

    def test_uint8(self):
        arr = np.zeros((4, 4, 4), dtype=np.uint8)
        img = _make_sitk_image(arr)
        vtk_img = sitk_to_vtk(img)
        assert self._get_vtk_scalar_type(vtk_img) == vtk.VTK_UNSIGNED_CHAR

    def test_float32(self):
        arr = np.zeros((4, 4, 4), dtype=np.float32)
        img = _make_sitk_image(arr)
        vtk_img = sitk_to_vtk(img)
        assert self._get_vtk_scalar_type(vtk_img) == vtk.VTK_FLOAT

    def test_int32(self):
        arr = np.zeros((4, 4, 4), dtype=np.int32)
        img = _make_sitk_image(arr)
        vtk_img = sitk_to_vtk(img)
        assert self._get_vtk_scalar_type(vtk_img) == vtk.VTK_INT

    def test_float64(self):
        arr = np.zeros((4, 4, 4), dtype=np.float64)
        img = _make_sitk_image(arr)
        vtk_img = sitk_to_vtk(img)
        assert self._get_vtk_scalar_type(vtk_img) == vtk.VTK_DOUBLE


class TestSitkToVtkScalarCount:
    """Verify that the total number of scalars matches the voxel count."""

    def test_scalar_count(self):
        arr = np.zeros((10, 20, 30), dtype=np.int16)
        img = _make_sitk_image(arr)
        vtk_img = sitk_to_vtk(img)
        scalars = vtk_img.GetPointData().GetScalars()
        assert scalars.GetNumberOfTuples() == 10 * 20 * 30

    def test_number_of_points(self):
        arr = np.zeros((5, 10, 15), dtype=np.int16)
        img = _make_sitk_image(arr)
        vtk_img = sitk_to_vtk(img)
        assert vtk_img.GetNumberOfPoints() == 5 * 10 * 15


class TestSitkToVtkRealisticCT:
    """Test with realistic CT-like data (int16, typical HU range)."""

    def test_ct_hu_range_preserved(self):
        """Typical spine CT: int16, HU from -1000 (air) to +3000 (dense bone)."""
        rng = np.random.RandomState(123)
        arr = rng.randint(-1000, 3001, size=(32, 64, 64), dtype=np.int16)
        img = _make_sitk_image(arr, spacing=(0.488, 0.488, 2.0), origin=(-150.0, -200.0, -50.0))
        vtk_img = sitk_to_vtk(img)

        assert vtk_img.GetDimensions() == (64, 64, 32)
        assert vtk_img.GetSpacing() == pytest.approx((0.488, 0.488, 2.0))
        assert vtk_img.GetOrigin() == pytest.approx((-150.0, -200.0, -50.0))
        assert vtk_img.GetScalarType() == vtk.VTK_SHORT

        # Spot-check HU values at several random locations
        for _ in range(20):
            x = rng.randint(0, 64)
            y = rng.randint(0, 64)
            z = rng.randint(0, 32)
            expected = float(arr[z, y, x])
            actual = vtk_img.GetScalarComponentAsFloat(x, y, z, 0)
            assert actual == expected, f"HU mismatch at ({x},{y},{z})"


class TestDownsampleVtkImage:
    """Tests for downsample_vtk_image() using vtkImageShrink3D."""

    def test_import_exists(self):
        from src.utils.vtk_helpers import downsample_vtk_image
        assert callable(downsample_vtk_image)

    def test_dimensions_halved(self):
        from src.utils.vtk_helpers import downsample_vtk_image
        arr = np.zeros((10, 20, 30), dtype=np.int16)
        img = _make_sitk_image(arr)
        vtk_img = sitk_to_vtk(img)
        result = downsample_vtk_image(vtk_img, (2, 2, 2))
        dims = result.GetDimensions()
        assert dims == (15, 10, 5)

    def test_dimensions_with_asymmetric_shrink(self):
        from src.utils.vtk_helpers import downsample_vtk_image
        arr = np.zeros((8, 16, 32), dtype=np.int16)
        img = _make_sitk_image(arr)
        vtk_img = sitk_to_vtk(img)
        result = downsample_vtk_image(vtk_img, (2, 2, 1))
        dims = result.GetDimensions()
        assert dims == (16, 8, 8)

    def test_output_is_vtk_image_data(self):
        from src.utils.vtk_helpers import downsample_vtk_image
        arr = np.zeros((4, 8, 16), dtype=np.int16)
        img = _make_sitk_image(arr)
        vtk_img = sitk_to_vtk(img)
        result = downsample_vtk_image(vtk_img, (2, 2, 2))
        assert isinstance(result, vtk.vtkImageData)

    def test_source_has_vtkImageShrink3D(self):
        """downsample_vtk_image must use vtkImageShrink3D internally."""
        import inspect
        from src.utils import vtk_helpers
        source = inspect.getsource(vtk_helpers)
        assert "vtkImageShrink3D" in source


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
