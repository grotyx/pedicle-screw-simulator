"""
Tests for VolumeManager (Phase 1b).

Tests the refactored VolumeManager that:
- No longer has _generate_bone_surface() or SurfaceGenerationThread dependencies
- Supports transfer function preset management
- Preserves all coordinate/geometry APIs
"""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import SimpleITK as sitk
import vtk

from src.core.volume_manager import VolumeManager
from src.utils.constants import TRANSFER_FUNCTION_PRESETS


def _make_test_volume(
    size=(32, 32, 16),
    spacing=(1.0, 1.0, 2.0),
    origin=(0.0, 0.0, 0.0),
    fill_value=0,
) -> sitk.Image:
    """Create a test SimpleITK volume."""
    arr = np.full(
        (size[2], size[1], size[0]), fill_value, dtype=np.int16
    )  # z, y, x
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing(spacing)
    img.SetOrigin(origin)
    return img


class TestSetVolume:
    """Test VolumeManager.set_volume() stores and caches correctly."""

    def test_stores_vtk_image(self):
        vm = VolumeManager()
        img = _make_test_volume()
        vm.set_volume(img)
        assert vm.get_vtk_image() is not None

    def test_stores_sitk_image(self):
        vm = VolumeManager()
        img = _make_test_volume()
        vm.set_volume(img)
        assert vm.get_sitk_image() is img

    def test_caches_dimensions(self):
        vm = VolumeManager()
        img = _make_test_volume(size=(64, 48, 32))
        vm.set_volume(img)
        assert vm.dimensions == (64, 48, 32)

    def test_caches_spacing(self):
        vm = VolumeManager()
        img = _make_test_volume(spacing=(0.5, 0.5, 1.5))
        vm.set_volume(img)
        assert vm.spacing == pytest.approx((0.5, 0.5, 1.5))

    def test_caches_origin(self):
        vm = VolumeManager()
        img = _make_test_volume(origin=(-100.0, -200.0, 50.0))
        vm.set_volume(img)
        assert vm.origin == pytest.approx((-100.0, -200.0, 50.0))

    def test_center_computed_correctly(self):
        vm = VolumeManager()
        # size=(32,32,16), spacing=(1,1,2), origin=(0,0,0)
        # bounds: x=[0,31], y=[0,31], z=[0,30]
        img = _make_test_volume(size=(32, 32, 16), spacing=(1.0, 1.0, 2.0))
        vm.set_volume(img)
        cx, cy, cz = vm.center
        assert cx == pytest.approx(15.5)
        assert cy == pytest.approx(15.5)
        assert cz == pytest.approx(15.0)

    def test_slice_positions_initialized_to_center(self):
        vm = VolumeManager()
        img = _make_test_volume(size=(32, 32, 16), spacing=(1.0, 1.0, 2.0))
        vm.set_volume(img)
        # axial = center_z, sagittal = center_x, coronal = center_y
        assert vm.get_slice_position("axial") == pytest.approx(vm.center[2])
        assert vm.get_slice_position("sagittal") == pytest.approx(vm.center[0])
        assert vm.get_slice_position("coronal") == pytest.approx(vm.center[1])


class TestObservers:
    """Test VolumeManager observer notification system."""

    def test_volume_loaded_notified(self):
        events = []
        vm = VolumeManager()
        vm.add_observer("test", lambda event, **kw: events.append(event))
        vm.set_volume(_make_test_volume())
        assert "volume_loaded" in events

    def test_slice_changed_notified(self):
        events = []
        vm = VolumeManager()
        vm.set_volume(_make_test_volume())
        vm.add_observer("test", lambda event, **kw: events.append(event))
        vm.set_slice_position("axial", 5.0)
        assert "slice_changed" in events

    def test_remove_observer(self):
        events = []
        vm = VolumeManager()
        vm.add_observer("test", lambda event, **kw: events.append(event))
        vm.remove_observer("test")
        vm.set_volume(_make_test_volume())
        assert len(events) == 0


class TestSlicePositions:
    """Test slice position management and clamping."""

    def test_clamp_axial_to_bounds(self):
        vm = VolumeManager()
        vm.set_volume(_make_test_volume(size=(32, 32, 16), spacing=(1, 1, 2)))
        # z bounds: [0, 30] for 16 slices at spacing 2
        vm.set_slice_position("axial", 9999.0)
        assert vm.get_slice_position("axial") <= 30.0

        vm.set_slice_position("axial", -9999.0)
        assert vm.get_slice_position("axial") >= 0.0

    def test_clamp_sagittal_to_bounds(self):
        vm = VolumeManager()
        vm.set_volume(_make_test_volume(size=(32, 32, 16), spacing=(1, 1, 2)))
        vm.set_slice_position("sagittal", 9999.0)
        assert vm.get_slice_position("sagittal") <= 31.0

    def test_invalid_plane_raises(self):
        vm = VolumeManager()
        vm.set_volume(_make_test_volume())
        with pytest.raises(ValueError, match="Unknown plane"):
            vm.set_slice_position("oblique", 0.0)


class TestTransferFunctionPreset:
    """Test transfer function preset management (Phase 1b new feature)."""

    def test_default_preset_is_bone(self):
        vm = VolumeManager()
        assert vm.transfer_function_preset == "Bone"

    def test_set_valid_preset(self):
        vm = VolumeManager()
        vm.set_transfer_function_preset("Soft Tissue")
        assert vm.transfer_function_preset == "Soft Tissue"

    def test_set_invalid_preset_raises(self):
        vm = VolumeManager()
        with pytest.raises(ValueError, match="Unknown transfer function preset"):
            vm.set_transfer_function_preset("NonExistent")

    def test_preset_change_notifies_observers(self):
        events = []
        vm = VolumeManager()
        vm.add_observer("test", lambda event, **kw: events.append(event))
        vm.set_transfer_function_preset("MIP (Maximum Intensity)")
        assert "transfer_function_changed" in events

    def test_all_presets_accessible(self):
        for name in TRANSFER_FUNCTION_PRESETS:
            vm = VolumeManager()
            vm.set_transfer_function_preset(name)
            assert vm.transfer_function_preset == name


class TestCoordinateConversion:
    """Test world ↔ index coordinate conversion."""

    def test_world_to_index_center(self):
        vm = VolumeManager()
        vm.set_volume(_make_test_volume(size=(32, 32, 16), spacing=(1, 1, 2)))
        i, j, k = vm.world_to_index(15.5, 15.5, 15.0)
        assert (i, j, k) == (16, 16, 8)  # rounded

    def test_index_to_world_roundtrip(self):
        vm = VolumeManager()
        vm.set_volume(
            _make_test_volume(
                size=(64, 64, 32), spacing=(0.5, 0.5, 1.0), origin=(-16.0, -16.0, 0.0)
            )
        )
        x, y, z = vm.index_to_world(10, 20, 5)
        i, j, k = vm.world_to_index(x, y, z)
        assert (i, j, k) == (10, 20, 5)

    def test_index_clamped_to_valid_range(self):
        vm = VolumeManager()
        vm.set_volume(_make_test_volume(size=(32, 32, 16)))
        i, j, k = vm.world_to_index(-999.0, -999.0, -999.0)
        assert i == 0 and j == 0 and k == 0

        i, j, k = vm.world_to_index(9999.0, 9999.0, 9999.0)
        assert i == 31 and j == 31 and k == 15


class TestNoSurfaceGeneration:
    """Verify that surface generation methods have been removed (Phase 1b)."""

    def test_no_bone_surface_attribute(self):
        vm = VolumeManager()
        assert not hasattr(vm, "_bone_surface") or vm._bone_surface is None

    def test_no_surface_pending_attribute(self):
        vm = VolumeManager()
        assert not hasattr(vm, "_surface_generation_pending")

    def test_no_generate_bone_surface_method(self):
        vm = VolumeManager()
        assert not hasattr(vm, "_generate_bone_surface")

    def test_no_get_bone_surface_method(self):
        vm = VolumeManager()
        assert not hasattr(vm, "get_bone_surface")

    def test_no_set_bone_threshold_method(self):
        vm = VolumeManager()
        assert not hasattr(vm, "set_bone_threshold")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
