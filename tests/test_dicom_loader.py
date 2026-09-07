import numpy as np
import SimpleITK as sitk

from src.core.dicom_loader import normalize_orientation


def _ramp_image(direction):
    arr = np.arange(4 * 5 * 6, dtype=np.int16).reshape(4, 5, 6)  # (z, y, x)
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing((1.0, 2.0, 3.0))
    img.SetOrigin((10.0, 20.0, 30.0))
    img.SetDirection(direction)
    return img


def test_identity_is_unchanged():
    img = _ramp_image((1, 0, 0, 0, 1, 0, 0, 0, 1))
    out, info = normalize_orientation(img)
    assert out is img
    assert info["orientation_normalized"] is False


def test_flipped_direction_is_normalised():
    img = _ramp_image((-1, 0, 0, 0, -1, 0, 0, 0, 1))
    out, info = normalize_orientation(img)
    assert np.allclose(out.GetDirection(), (1, 0, 0, 0, 1, 0, 0, 0, 1))
    assert info["orientation_normalized"] is True and info["resampled"] is False
    # physical content preserved: the voxel at original index (2,3,1) keeps its value at the same LPS point
    point = img.TransformIndexToPhysicalPoint((2, 3, 1))
    new_index = out.TransformPhysicalPointToIndex(point)
    assert out[new_index] == img[(2, 3, 1)]


def test_oblique_direction_is_resampled():
    c, s = np.cos(np.radians(10)), np.sin(np.radians(10))
    img = _ramp_image((1, 0, 0, 0, c, -s, 0, s, c))
    out, info = normalize_orientation(img)
    assert np.allclose(out.GetDirection(), (1, 0, 0, 0, 1, 0, 0, 0, 1))
    assert info["resampled"] is True
    assert out.GetSpacing() == img.GetSpacing()


def test_transform_direction_left_multiplies_only():
    from src.core.coordinate_system import CoordinateSystem
    d = (1, 0, 0, 0, 0, -1, 0, 1, 0)
    out = np.array(CoordinateSystem.transform_direction(d)).reshape(3, 3)
    expected = np.diag([-1, -1, 1]) @ np.array(d).reshape(3, 3)
    assert np.allclose(out, expected)
