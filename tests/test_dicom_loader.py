import numpy as np
import SimpleITK as sitk

from src.core.dicom_loader import DicomLoader, check_slice_geometry, normalize_orientation


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


def test_tilted_sagittal_direction_permutes_spacing_by_dominant_axis():
    # Image axis 0 is dominantly world Z, axis 1 is dominantly world X,
    # axis 2 is dominantly world Y (a slightly tilted sagittal acquisition).
    col0 = np.array([0.0, 0.17, 0.98])
    col1 = np.array([1.0, 0.0, 0.0])
    col2 = np.array([0.0, 0.98, -0.17])
    col0 = col0 / np.linalg.norm(col0)
    col1 = col1 / np.linalg.norm(col1)
    col2 = col2 / np.linalg.norm(col2)
    direction = (
        col0[0], col1[0], col2[0],
        col0[1], col1[1], col2[1],
        col0[2], col1[2], col2[2],
    )
    size = (4, 5, 6)
    spacing = (0.5, 0.5, 3.0)
    img = sitk.Image(size, sitk.sitkInt16)
    img.SetSpacing(spacing)
    img.SetOrigin((0.0, 0.0, 0.0))
    img.SetDirection(direction)

    out, info = normalize_orientation(img)

    assert info["resampled"] is True
    assert np.allclose(out.GetDirection(), (1, 0, 0, 0, 1, 0, 0, 0, 1))

    # Dominant world axis per image axis: axis0->Z, axis1->X, axis2->Y.
    out_spacing = out.GetSpacing()
    assert np.isclose(out_spacing[0], spacing[1])  # world X <- image axis 1
    assert np.isclose(out_spacing[1], spacing[2])  # world Y <- image axis 2
    assert np.isclose(out_spacing[2], spacing[0])  # world Z <- image axis 0

    # Output size must cover the true world-space bounding box of the input
    # volume using the (correctly permuted) per-world-axis spacing, not be
    # 6x oversampled/undersampled from using the wrong spacing on an axis.
    out_size = out.GetSize()
    corners = []
    for i in (0, size[0] - 1):
        for j in (0, size[1] - 1):
            for k in (0, size[2] - 1):
                corners.append(img.TransformContinuousIndexToPhysicalPoint((float(i), float(j), float(k))))
    corners = np.asarray(corners)
    lo, hi = corners.min(axis=0), corners.max(axis=0)
    for a in range(3):
        expected_size = int(np.ceil((hi[a] - lo[a]) / out_spacing[a])) + 1
        assert out_size[a] == expected_size


def test_45_degree_z_rotation_produces_valid_axis_permutation():
    # A 45-degree rotation about Z makes image axes 0 and 1 EQUALLY dominant
    # on world axis 0 (and world axis 1): independent per-column argmax maps
    # both to the same world axis, leaving another world axis with spacing
    # 0.0 and raising OverflowError computing new_size. The assignment must
    # instead be a genuine permutation so every world axis gets spacing.
    c = s = np.cos(np.radians(45))
    direction = (c, -s, 0.0, s, c, 0.0, 0.0, 0.0, 1.0)
    size = (4, 5, 6)
    spacing = (0.5, 0.6, 0.7)
    img = sitk.Image(size, sitk.sitkInt16)
    img.SetSpacing(spacing)
    img.SetOrigin((0.0, 0.0, 0.0))
    img.SetDirection(direction)

    out, info = normalize_orientation(img)

    assert info["resampled"] is True
    assert np.allclose(out.GetDirection(), (1, 0, 0, 0, 1, 0, 0, 0, 1))

    out_spacing = out.GetSpacing()
    out_size = out.GetSize()
    assert all(v > 0 and np.isfinite(v) for v in out_spacing)
    assert all(v > 0 for v in out_size)
    # Each world axis must receive exactly one of the input spacing values
    # (a permutation), not a 0.0 default from an unclaimed axis.
    assert sorted(round(v, 6) for v in out_spacing) == sorted(round(v, 6) for v in spacing)


def test_transform_direction_left_multiplies_only():
    from src.core.coordinate_system import CoordinateSystem
    d = (1, 0, 0, 0, 0, -1, 0, 1, 0)
    out = np.array(CoordinateSystem.transform_direction(d)).reshape(3, 3)
    expected = np.diag([-1, -1, 1]) @ np.array(d).reshape(3, 3)
    assert np.allclose(out, expected)


def test_unsigned_oblique_series_background_is_not_bone():
    arr = np.full((4, 5, 6), 500, dtype=np.uint16)
    img = sitk.GetImageFromArray(arr)
    c, s = np.cos(np.radians(30)), np.sin(np.radians(30))
    img.SetDirection((c, -s, 0.0, s, c, 0.0, 0.0, 0.0, 1.0))

    out, info = normalize_orientation(img)

    assert info["resampled"]
    assert out.GetPixelID() in (sitk.sitkInt16, sitk.sitkFloat32, sitk.sitkInt32)
    assert sitk.GetArrayViewFromImage(out).min() == -1000


def test_empty_optional_numeric_tags_do_not_drop_metadata(monkeypatch):
    import pydicom

    ds = pydicom.Dataset()
    ds.Modality = "CT"
    ds.SliceThickness = None  # empty Type-2 tag
    ds.KVP = ""
    monkeypatch.setattr("src.core.dicom_loader.pydicom.dcmread", lambda *a, **k: ds)
    loader = DicomLoader()
    loader._file_names = ["a.dcm", "b.dcm"]

    loader._extract_metadata(None)
    meta = loader.get_metadata()

    assert "error" not in meta
    assert meta["modality"] == "CT"
    assert meta["num_slices"] == 2
    assert meta["slice_thickness"] == 0.0
    assert meta["kvp"] == 0.0


def test_no_series_error_does_not_contain_the_folder_path(tmp_path):
    folder = tmp_path / "Patient Kim Minsu"
    folder.mkdir()
    try:
        DicomLoader().load_series(str(folder))
    except ValueError as exc:
        assert "Minsu" not in str(exc)
        assert str(tmp_path) not in str(exc)
    else:
        raise AssertionError("expected ValueError")


_AXIAL = (1, 0, 0, 0, 1, 0)


def _stack(zs, shift=0.0):
    return [(shift * i, 0.0, float(z)) for i, z in enumerate(zs)]


def test_uniform_axial_stack_has_no_geometry_warnings():
    assert check_slice_geometry(_stack([0, 1, 2, 3, 4]), _AXIAL) == []


def test_missing_slice_reports_non_uniform_spacing():
    warnings = check_slice_geometry(_stack([0, 1, 2, 4, 5]), _AXIAL)
    assert len(warnings) == 1
    assert "not uniform" in warnings[0]
    assert "min 1.00 mm" in warnings[0] and "max 2.00 mm" in warnings[0]


def test_gantry_tilt_is_reported():
    # 20 degree shear: in-plane shift of tan(20) mm per 1 mm slice
    warnings = check_slice_geometry(_stack(range(6), shift=np.tan(np.radians(20))), _AXIAL)
    assert len(warnings) == 1
    assert "Gantry tilt of 20.0" in warnings[0]


def test_geometry_check_ignores_degenerate_input():
    assert check_slice_geometry([], _AXIAL) == []
    assert check_slice_geometry([(0, 0, 0)], _AXIAL) == []
    assert check_slice_geometry(_stack([0, 1, 2]), (0, 0, 0, 0, 0, 0)) == []
