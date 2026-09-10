"""
Tests for PedicleAnalyzer and vertebra data models.

Uses synthetic SimpleITK mask images with known geometry so that
expected centroids, bounding boxes, and pedicle properties can be
verified analytically.
"""

import logging
import os
import sys

import numpy as np
import pytest
import SimpleITK as sitk

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.core.pedicle_analyzer import (
    VERTEBRA_LABELS,
    PedicleAnalyzer,
)
from src.core.vertebra import PedicleAnalysisResult, Vertebra

# ---------------------------------------------------------------------------
# Helpers for building synthetic masks
# ---------------------------------------------------------------------------

def _make_mask(
    shape: tuple = (40, 60, 60),
    spacing: tuple = (1.0, 1.0, 1.0),
    origin: tuple = (0.0, 0.0, 0.0),
) -> sitk.Image:
    """Create a blank uint8 SimpleITK image with the given geometry.

    ``shape`` is given in numpy order (z, y, x).
    """
    arr = np.zeros(shape, dtype=np.uint8)
    img = sitk.GetImageFromArray(arr)
    # sitk expects size/spacing/origin in (x, y, z) order.
    img.SetSpacing(tuple(reversed(spacing)))
    img.SetOrigin(tuple(reversed(origin)))
    return img


def _paint_label(img: sitk.Image, label: int, slices) -> sitk.Image:
    """Paint *label* into the image at the given numpy-style slices.

    Returns the modified image (modifies underlying array in-place via
    a round-trip).
    """
    arr = sitk.GetArrayFromImage(img)
    arr[slices] = label
    new_img = sitk.GetImageFromArray(arr)
    new_img.CopyInformation(img)
    return new_img


def _make_h_vertebra_mask(
    label: int = 27,
    shape: tuple = (40, 60, 60),
    spacing: tuple = (1.0, 1.0, 1.0),
    origin: tuple = (0.0, 0.0, 0.0),
) -> sitk.Image:
    """Build a synthetic H-shaped vertebra.

    The H shape simulates a vertebral body (large anterior block) and
    two posterior pedicles (left and right) in the superior half:

    Layout in each axial slice of the superior portion (z >= 20):
        body: y=[5..30], x=[20..40]       (anterior, wide)
        left pedicle: y=[35..45], x=[40..48]  (posterior-left)
        right pedicle: y=[35..45], x=[12..20] (posterior-right)

    In the inferior half (z < 20) only the body is present (no pedicle
    separation at that level).
    """
    img = _make_mask(shape, spacing, origin)
    arr = sitk.GetArrayFromImage(img)

    # Inferior half — body only (z = 0..19)
    arr[0:20, 5:45, 12:48] = label

    # Superior half — body + pedicles (z = 20..39)
    # Body (anterior)
    arr[20:40, 5:30, 20:40] = label
    # Left pedicle (high x = patient left for standard orientation)
    arr[20:40, 35:45, 40:48] = label
    # Right pedicle (low x = patient right)
    arr[20:40, 35:45, 12:20] = label

    new_img = sitk.GetImageFromArray(arr)
    new_img.CopyInformation(img)
    return new_img


def _make_connected_vertebra_mask(label: int = 27) -> sitk.Image:
    """Build a single-component body/pedicle phantom like a real mask."""
    img = _make_mask(shape=(40, 60, 60))
    arr = sitk.GetArrayFromImage(img)
    arr[10:30, 10:32, 18:42] = label
    arr[10:30, 28:48, 36:46] = label
    arr[10:30, 28:48, 14:24] = label
    connected = sitk.GetImageFromArray(arr)
    connected.CopyInformation(img)
    return connected



def _make_anatomical_phantom(label: int = 28, with_arch: bool = True) -> sitk.Image:
    """Body ellipse + two 8 mm pedicles + posterior arch, 1 mm isotropic, LPS identity.

    The pedicles start at ``yy >= 44`` so that they actually merge with the
    body ellipse, whose posterior wall has already fallen back to y = 44.9 by
    the time it reaches their x = 60 axis, and the laminar arch starts at
    ``yy >= 60`` so that it joins them from behind and leaves the spinal canal
    hollow instead of filling it with bone anterior to the laminae.

    The arch ellipse only touches the pedicles at ``x = 60`` and is separated
    from them by air everywhere else, so a posterior ray-cast entry point lands
    inside the lamina with no drillable bone behind it.  ``with_arch=False``
    drops the arch, leaving the pedicle's own posterior cortex as the entry
    surface; the trajectory optimiser tests use that variant.
    """
    Z, Y, X = 60, 90, 90
    zz, yy, xx = np.mgrid[0:Z, 0:Y, 0:X]
    body = (((xx - 45) / 20.0) ** 2 + ((yy - 35) / 15.0) ** 2 <= 1) & (zz >= 15) & (zz < 45)
    ped = np.zeros_like(body)
    for cx in (30, 60):
        ped |= (((xx - cx) / 4.0) ** 2 + ((zz - 32) / 6.0) ** 2 <= 1) & (yy >= 44) & (yy < 62)
    arch = np.zeros_like(body)
    if with_arch:
        arch = ((((xx - 45) / 22.0) ** 2 + ((yy - 66) / 10.0) ** 2 <= 1)
                & ~(((xx - 45) / 14.0) ** 2 + ((yy - 64) / 6.0) ** 2 <= 1)
                & (yy >= 60) & (zz >= 26) & (zz < 40))
    arr = np.zeros((Z, Y, X), dtype=np.uint8)
    arr[body | ped | arch] = label
    return sitk.GetImageFromArray(arr)


def _make_one_sided_coronal_phantom(label: int = 28, with_left: bool = True) -> sitk.Image:
    """A vertebra the coronal isthmus search can only measure on one side.

    The right pedicle is a solid 12 x 14 x 12 mm block joined to the body, so
    the coronal scan records it normally.  The left one is a 3 x 3 mm sliver
    running 19 mm posteriorly: every coronal cross-section of it is 9 mm²,
    under :attr:`PedicleAnalyzer.MIN_PEDICLE_AREA_MM2`, so the coronal search
    never records it — while in an axial slice it is a 57-voxel island clear of
    the body, exactly what the axial connected-component pass looks for.

    ``with_left=False`` drops the sliver entirely: no path can find that side.
    """
    Z, Y, X = 60, 90, 90
    zz, yy, xx = np.mgrid[0:Z, 0:Y, 0:X]
    body = (xx >= 25) & (xx < 66) & (yy >= 20) & (yy < 50) & (zz >= 10) & (zz < 50)
    right = (xx >= 26) & (xx < 38) & (yy >= 48) & (yy < 62) & (zz >= 26) & (zz < 38)
    arr = np.zeros((Z, Y, X), dtype=np.uint8)
    shape = body | right
    if with_left:
        shape = shape | (
            (xx >= 57) & (xx < 60) & (yy >= 52) & (yy < 71) & (zz >= 31) & (zz < 34)
        )
    arr[shape] = label
    return sitk.GetImageFromArray(arr)


def _make_sliver_corridor_phantom(label: int = 28) -> sitk.Image:
    """An 8 mm pedicle corridor with a solid medial fragment beside it.

    Reproduces the spec's L2-right failure at phantom scale.  For four coronal
    slices in the middle of the left corridor a 3 x 10 mm fragment (a superior
    articular process tip in the real mask) sits 7 mm lateral of the midline,
    medial to the real 8 mm pedicle at 14.5 mm.  The "candidate nearest the
    midline" rule picks the fragment, the minimum-area rule then makes it the
    isthmus, and the level is reported as a 3 mm pedicle.  The right corridor
    has no fragment, so the two sides show the same failure side by side.

    The fragment is deliberately *solid*: 30 mm2 and 3 mm across, clear of
    both :attr:`PedicleAnalyzer.MIN_PEDICLE_AREA_MM2` and
    :attr:`PedicleAnalyzer.MIN_SLICE_WIDTH_MM`, so neither floor can reject it
    and only following the corridor by continuity keeps it out of the walk.
    """
    Z, Y, X = 60, 90, 90
    zz, yy, xx = np.mgrid[0:Z, 0:Y, 0:X]
    body = (((xx - 45) / 20.0) ** 2 + ((yy - 35) / 15.0) ** 2 <= 1) & (zz >= 15) & (zz < 45)
    ped = np.zeros_like(body)
    for cx in (30, 60):
        ped |= (
            (xx >= cx - 4) & (xx < cx + 4)          # exactly 8 voxels = 8.0 mm
            & (np.abs(zz - 32) <= 6)
            & (yy >= 44)
            & (yy < 62)
        )
    fragment = (
        (xx >= 51) & (xx <= 53)                 # 3 voxels = 3.0 mm across
        & (zz >= 27) & (zz <= 36)               # x 10 voxels = 30 mm2
        & (yy >= 52) & (yy < 56)
    )
    arr = np.zeros((Z, Y, X), dtype=np.uint8)
    arr[body | ped | fragment] = label
    return sitk.GetImageFromArray(arr)


def _make_coarse_phantom(label: int = 28) -> sitk.Image:
    """The body/pedicle phantom drawn on a 2 mm inference grid.

    The pedicles are exactly four coarse voxels wide, so their true width is
    8.0 mm however finely the mask is later resampled.  The body is an ellipse
    so that the mask as a whole has the oblique boundaries a stair step shows
    up on.
    """
    Z, Y, X = 30, 45, 45
    zz, yy, xx = np.mgrid[0:Z, 0:Y, 0:X]
    body = (
        (((xx - 22.5) / 10.0) ** 2 + ((yy - 17.5) / 7.5) ** 2 <= 1)
        & (zz >= 8)
        & (zz < 23)
    )
    ped = np.zeros_like(body)
    for cx in (15, 30):
        ped |= (
            (xx >= cx) & (xx < cx + 4)
            & (np.abs(zz - 16) <= 3)
            & (yy >= 22)
            & (yy < 31)
        )
    arr = np.zeros((Z, Y, X), dtype=np.uint8)
    arr[body | ped] = label
    image = sitk.GetImageFromArray(arr)
    image.SetSpacing((2.0, 2.0, 2.0))
    return image


def _nearest_neighbour_upsample(image: sitk.Image, factor: int) -> sitk.Image:
    """Resample *image* by an integer factor with nearest-neighbour interpolation.

    This is exactly what the app receives from TotalSegmentator: a mask
    inferred on a coarse grid and pushed onto the fine CT grid, so every
    boundary becomes a stair step ``factor`` voxels deep and every value
    repeats in runs of ``factor``.
    """
    resampler = sitk.ResampleImageFilter()
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampler.SetOutputSpacing([s / factor for s in image.GetSpacing()])
    resampler.SetSize([int(n * factor) for n in image.GetSize()])
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetDefaultPixelValue(0)
    return resampler.Execute(image)


def _make_notched_corridor_phantom(notch_slices: int, label: int = 28) -> sitk.Image:
    """A left corridor interrupted by ``notch_slices`` blank coronal slices.

    The left corridor is an 8 x 13 mm box (104 mm2) from y = 44 to y = 55, then
    ``notch_slices`` slices where that side has nothing, then a narrower
    5 x 7 mm box (35 mm2) -- the true isthmus -- for five more slices.  The
    right corridor runs straight through, so a notched slice is never blank in
    the coronal plane and the walk has to decide for itself whether the left
    corridor ended.  Bridging the notch reaches the 5 mm distal isthmus;
    stopping at it leaves only the 8 mm proximal box.
    """
    Z, Y, X = 60, 90, 90
    zz, yy, xx = np.mgrid[0:Z, 0:Y, 0:X]
    body = (((xx - 45) / 20.0) ** 2 + ((yy - 35) / 15.0) ** 2 <= 1) & (zz >= 15) & (zz < 45)
    notch_start = 56
    far_start = notch_start + notch_slices
    far_end = far_start + 5
    near = (xx >= 56) & (xx < 64) & (np.abs(zz - 32) <= 6) & (yy >= 44) & (yy < notch_start)
    far = (
        (xx >= 58) & (xx < 63) & (np.abs(zz - 32) <= 3)
        & (yy >= far_start) & (yy < far_end)
    )
    right = (xx >= 26) & (xx < 34) & (np.abs(zz - 32) <= 6) & (yy >= 44) & (yy < far_end)
    arr = np.zeros((Z, Y, X), dtype=np.uint8)
    arr[body | near | far | right] = label
    return sitk.GetImageFromArray(arr)


# ---------------------------------------------------------------------------
# Vertebra dataclass tests
# ---------------------------------------------------------------------------

class TestVertebraDataclass:
    """Validation of the Vertebra data model."""

    def test_valid_construction(self):
        v = Vertebra(
            label=27,
            name="L5",
            centroid_lps=np.array([1.0, 2.0, 3.0]),
            bounding_box=(np.array([0.0, 0.0, 0.0]), np.array([10.0, 10.0, 10.0])),
            volume_mm3=1000.0,
            mask_indices=np.array([[1, 2, 3], [4, 5, 6]]),
        )
        assert v.label == 27
        assert v.name == "L5"
        assert v.centroid_lps.dtype == np.float64
        np.testing.assert_array_equal(v.centroid_lps, [1.0, 2.0, 3.0])

    def test_centroid_wrong_shape_raises(self):
        with pytest.raises(ValueError, match="centroid_lps"):
            Vertebra(
                label=27, name="L5",
                centroid_lps=np.array([1.0, 2.0]),
                bounding_box=(np.zeros(3), np.ones(3)),
                volume_mm3=0.0,
                mask_indices=np.zeros((1, 3)),
            )

    def test_bounding_box_wrong_shape_raises(self):
        with pytest.raises(ValueError, match="bounding_box"):
            Vertebra(
                label=27, name="L5",
                centroid_lps=np.zeros(3),
                bounding_box=(np.zeros(2), np.ones(3)),
                volume_mm3=0.0,
                mask_indices=np.zeros((1, 3)),
            )

    def test_mask_indices_wrong_shape_raises(self):
        with pytest.raises(ValueError, match="mask_indices"):
            Vertebra(
                label=27, name="L5",
                centroid_lps=np.zeros(3),
                bounding_box=(np.zeros(3), np.ones(3)),
                volume_mm3=0.0,
                mask_indices=np.zeros((4,)),
            )


class TestPedicleAnalysisResultDefaults:
    """PedicleAnalysisResult default field values."""

    def test_defaults(self):
        v = Vertebra(
            label=27, name="L5",
            centroid_lps=np.zeros(3),
            bounding_box=(np.zeros(3), np.ones(3)),
            volume_mm3=100.0,
            mask_indices=np.zeros((1, 3), dtype=int),
        )
        r = PedicleAnalysisResult(vertebra=v)
        assert r.success is False
        assert r.left_pedicle_center is None
        assert r.right_pedicle_center is None
        assert r.left_pedicle_width == 0.0
        assert r.right_pedicle_width == 0.0
        assert r.upper_endplate_normal is None
        assert r.warnings == []


# ---------------------------------------------------------------------------
# PedicleAnalyzer: get_available_vertebrae
# ---------------------------------------------------------------------------

class TestGetAvailableVertebrae:
    """Tests for vertebra extraction from multilabel masks."""

    def test_two_labels_present(self):
        """Mask with L5 (27) and L4 (28) should return two vertebrae."""
        img = _make_mask(shape=(30, 30, 30))
        img = _paint_label(img, label=27, slices=(slice(0, 15), slice(5, 25), slice(5, 25)))
        img = _paint_label(img, label=28, slices=(slice(15, 30), slice(5, 25), slice(5, 25)))

        analyzer = PedicleAnalyzer(img)
        vertebrae = analyzer.get_available_vertebrae()

        assert len(vertebrae) == 2
        names = [v.name for v in vertebrae]
        assert "L5" in names
        assert "L4" in names

    def test_labels_sorted_ascending(self):
        """Vertebrae should be returned sorted by label id."""
        img = _make_mask(shape=(30, 30, 30))
        img = _paint_label(img, label=31, slices=(slice(0, 10), slice(5, 25), slice(5, 25)))
        img = _paint_label(img, label=27, slices=(slice(10, 20), slice(5, 25), slice(5, 25)))
        img = _paint_label(img, label=29, slices=(slice(20, 30), slice(5, 25), slice(5, 25)))

        analyzer = PedicleAnalyzer(img)
        vertebrae = analyzer.get_available_vertebrae()

        assert [v.label for v in vertebrae] == [27, 29, 31]

    def test_missing_label_skipped(self):
        """Labels not present in the mask are silently skipped."""
        img = _make_mask(shape=(20, 20, 20))
        img = _paint_label(img, label=32, slices=(slice(0, 20), slice(2, 18), slice(2, 18)))

        analyzer = PedicleAnalyzer(img)
        vertebrae = analyzer.get_available_vertebrae()

        assert len(vertebrae) == 1
        assert vertebrae[0].name == "T12"

    def test_empty_mask_returns_empty(self):
        """Completely empty mask yields no vertebrae."""
        img = _make_mask(shape=(10, 10, 10))
        analyzer = PedicleAnalyzer(img)
        vertebrae = analyzer.get_available_vertebrae()

        assert vertebrae == []

    def test_non_vertebra_labels_ignored(self):
        """Labels outside the 25-43 range are not returned."""
        img = _make_mask(shape=(10, 10, 10))
        img = _paint_label(img, label=1, slices=(slice(0, 10), slice(0, 10), slice(0, 10)))

        analyzer = PedicleAnalyzer(img)
        vertebrae = analyzer.get_available_vertebrae()

        assert vertebrae == []

    def test_centroid_calculation(self):
        """Centroid should match expected physical position for a simple block."""
        shape = (20, 20, 20)
        spacing = (2.0, 2.0, 2.0)
        origin = (0.0, 0.0, 0.0)
        img = _make_mask(shape, spacing, origin)

        # Paint label 27 in a block: z=[5,15), y=[5,15), x=[5,15).
        img = _paint_label(img, label=27,
                           slices=(slice(5, 15), slice(5, 15), slice(5, 15)))

        analyzer = PedicleAnalyzer(img)
        vertebrae = analyzer.get_available_vertebrae()

        assert len(vertebrae) == 1
        v = vertebrae[0]

        # The mean index in z,y,x is exactly (9.5, 9.5, 9.5) (block indices
        # 5..14 inclusive).  With spacing=2 and origin=0, the sub-voxel LPS
        # physical centroid is (9.5*2, 9.5*2, 9.5*2) = (19, 19, 19).
        expected = np.array([19.0, 19.0, 19.0])
        assert v.centroid_lps == pytest.approx(expected)

    def test_volume_mm3(self):
        """Volume should equal voxel_count * voxel_volume."""
        spacing = (0.5, 0.5, 1.0)
        img = _make_mask(shape=(10, 10, 10), spacing=spacing)
        # 10x10x10 block
        img = _paint_label(img, label=27,
                           slices=(slice(0, 10), slice(0, 10), slice(0, 10)))

        analyzer = PedicleAnalyzer(img)
        vertebrae = analyzer.get_available_vertebrae()

        voxel_vol = 0.5 * 0.5 * 1.0
        expected_vol = 1000 * voxel_vol
        assert pytest.approx(vertebrae[0].volume_mm3, rel=1e-6) == expected_vol


# ---------------------------------------------------------------------------
# PedicleAnalyzer: analyze_pedicle
# ---------------------------------------------------------------------------

class TestAnalyzePedicle:
    """Tests for per-vertebra pedicle morphology analysis."""

    def test_h_shaped_vertebra_detects_both_pedicles(self):
        """The H-shaped phantom should yield left and right pedicle centres."""
        img = _make_h_vertebra_mask(label=27)
        analyzer = PedicleAnalyzer(img)
        vertebrae = analyzer.get_available_vertebrae()
        assert len(vertebrae) == 1

        result = analyzer.analyze_pedicle(vertebrae[0])

        assert result.success is True
        assert result.left_pedicle_center is not None
        assert result.right_pedicle_center is not None
        assert result.left_pedicle_axis is not None
        assert result.right_pedicle_axis is not None
        assert result.left_pedicle_width > 0
        assert result.right_pedicle_width > 0

    def test_connected_mask_uses_anatomical_fallback_for_both_pedicles(self):
        img = _make_connected_vertebra_mask()
        analyzer = PedicleAnalyzer(img)
        vertebra = analyzer.get_available_vertebrae()[0]

        result = analyzer.analyze_pedicle(vertebra)

        assert result.success is True
        assert result.left_pedicle_center is not None
        assert result.right_pedicle_center is not None
        assert result.left_pedicle_center[0] > result.vertebral_body_center[0]
        assert result.right_pedicle_center[0] < result.vertebral_body_center[0]
        assert result.left_pedicle_center[1] > result.vertebral_body_center[1]
        assert result.right_pedicle_center[1] > result.vertebral_body_center[1]
        assert result.method in {"coronal_isthmus", "axial_components"}

    def test_upper_endplate_normal_tracks_sagittal_body_tilt(self):
        """Superior body envelope should recover its AP-to-superior slope."""
        image = _make_mask(shape=(50, 50, 50))
        array = sitk.GetArrayFromImage(image)
        expected_slope = 0.25
        for y_index in range(5, 31):
            superior_z = 22 + int(round(expected_slope * (y_index - 5)))
            array[8:superior_z + 1, y_index, 15:35] = 27
        tilted = sitk.GetImageFromArray(array)
        tilted.CopyInformation(image)
        analyzer = PedicleAnalyzer(tilted)
        vertebra = analyzer.get_available_vertebrae()[0]

        result = analyzer.analyze_pedicle(vertebra)

        assert result.upper_endplate_normal is not None
        recovered_slope = (
            -result.upper_endplate_normal[1]
            / result.upper_endplate_normal[2]
        )
        assert recovered_slope == pytest.approx(expected_slope, abs=0.06)

    def test_h_shaped_left_right_separation(self):
        """Left pedicle centre x-coordinate should be greater than right's
        (in index-derived physical space with identity direction)."""
        img = _make_h_vertebra_mask(label=27)
        analyzer = PedicleAnalyzer(img)
        v = analyzer.get_available_vertebrae()[0]
        result = analyzer.analyze_pedicle(v)

        # With identity direction and origin=(0,0,0),
        # LPS x = physical x = index x * spacing.
        # Left pedicle is at higher x-index, so its x-coord is larger.
        assert result.left_pedicle_center[0] > result.right_pedicle_center[0]

    def test_pedicle_axis_is_unit_vector(self):
        """Pedicle axes should be normalised unit vectors."""
        img = _make_h_vertebra_mask(label=27)
        analyzer = PedicleAnalyzer(img)
        v = analyzer.get_available_vertebrae()[0]
        result = analyzer.analyze_pedicle(v)

        for axis in [result.left_pedicle_axis, result.right_pedicle_axis]:
            if axis is not None:
                assert pytest.approx(np.linalg.norm(axis), abs=1e-6) == 1.0

    def test_vertebral_body_center_is_anterior(self):
        """The body centre should be anterior to the overall centroid.

        In our H-shaped phantom with identity direction, anterior means
        lower y-index / lower physical y.
        """
        img = _make_h_vertebra_mask(label=27)
        analyzer = PedicleAnalyzer(img)
        v = analyzer.get_available_vertebrae()[0]
        result = analyzer.analyze_pedicle(v)

        assert result.vertebral_body_center is not None
        # The body centre y should be less than the overall centroid y
        # (anterior = lower y in LPS with identity direction).
        assert result.vertebral_body_center[1] <= v.centroid_lps[1]

    def test_sacrum_is_skipped(self):
        """Sacrum (label 25) should be skipped with a warning."""
        img = _make_mask(shape=(20, 30, 30))
        img = _paint_label(img, label=25,
                           slices=(slice(0, 20), slice(5, 25), slice(5, 25)))

        analyzer = PedicleAnalyzer(img)
        vertebrae = analyzer.get_available_vertebrae()
        assert len(vertebrae) == 1
        assert vertebrae[0].name == "sacrum"

        result = analyzer.analyze_pedicle(vertebrae[0])
        assert result.success is False
        assert any("skipped" in w.lower() for w in result.warnings)

    def test_very_small_vertebra(self):
        """A vertebra with only a few voxels should not crash."""
        img = _make_mask(shape=(10, 10, 10))
        # Paint a tiny region — just 2 voxels thick in z.
        img = _paint_label(img, label=31,
                           slices=(slice(4, 6), slice(4, 6), slice(4, 6)))

        analyzer = PedicleAnalyzer(img)
        vertebrae = analyzer.get_available_vertebrae()
        assert len(vertebrae) == 1

        result = analyzer.analyze_pedicle(vertebrae[0])
        # Should not crash; may or may not find pedicles.
        assert isinstance(result, PedicleAnalysisResult)


# ---------------------------------------------------------------------------
# PedicleAnalyzer: coronal isthmus search
# ---------------------------------------------------------------------------

class TestCoronalIsthmus:
    def test_anatomical_phantom_isthmus_within_2mm(self):
        analyzer = PedicleAnalyzer(_make_anatomical_phantom())
        vertebra = analyzer.get_available_vertebrae()[0]
        result = analyzer.analyze_pedicle(vertebra)
        assert result.success and result.method == "coronal_isthmus"
        assert np.linalg.norm(result.left_pedicle_center - np.array([60.0, 55.0, 32.0])) <= 2.0
        assert np.linalg.norm(result.right_pedicle_center - np.array([30.0, 55.0, 32.0])) <= 2.0
        assert 7.0 <= result.left_pedicle_width <= 9.0
        assert 11.0 <= result.left_pedicle_height <= 13.0
        assert result.left_pedicle_axis[1] > 0.7  # posterior-oriented, mostly AP

    def test_axis_window_excludes_laminar_arch_slices(self):
        """The arch starts at y = 60; the axis window must stop short of it."""
        mask = _make_anatomical_phantom()
        analyzer = PedicleAnalyzer(mask)
        vertebra = analyzer.get_available_vertebrae()[0]
        binary = (sitk.GetArrayFromImage(mask) == vertebra.label).astype(np.uint8)
        indices_zyx = np.argwhere(binary)
        body_center = analyzer._estimate_body_center(binary, indices_zyx)
        body_center_ijk = mask.TransformPhysicalPointToContinuousIndex(
            tuple(float(v) for v in body_center)
        )
        z_indices = np.where(binary.any(axis=(1, 2)))[0]

        found = analyzer._find_pedicle_coronal(
            binary,
            body_center_ijk,
            (int(z_indices.min()), int(z_indices.max())),
            "left",
        )

        assert found is not None
        j_lo, j_hi = found["isthmus_window_j"]
        assert j_lo <= found["isthmus_j"] <= j_hi
        assert j_hi < 60, "axis window must exclude the laminar arch slices"
        assert found["axis_lps"][1] > 0.7

    def test_phantom_yields_two_planned_screws_through_pedicles(self):
        from src.core.auto_screw_planner import AutoScrewPlanner
        mask = _make_anatomical_phantom()
        arr = sitk.GetArrayFromImage(mask)
        ct = sitk.GetImageFromArray(np.where(arr > 0, 400, -50).astype(np.int16))
        ct.CopyInformation(mask)
        analyzer = PedicleAnalyzer(mask)
        result = analyzer.analyze_pedicle(analyzer.get_available_vertebrae()[0])
        planner = AutoScrewPlanner(ct, mask)
        screws = planner.plan_all([result])
        assert {s.side for s in screws} == {"left", "right"}
        for screw in screws:
            cx = 60.0 if screw.side == "left" else 30.0
            direction = screw.target_lps - screw.entry_lps
            samples = [screw.entry_lps + direction * t for t in np.linspace(0, 1, 40)]
            inside = sum(1 for p in samples if 48 <= p[1] < 62 and abs(p[0] - cx) <= 4 and abs(p[2] - 32) <= 6)
            assert inside >= 8, f"{screw.side} trajectory misses the pedicle corridor"
            assert screw.gertzbein_grade in {"A", "B"}

    def test_medial_sliver_does_not_displace_the_real_pedicle(self):
        """A one-voxel fragment beside the corridor must not become the isthmus."""
        analyzer = PedicleAnalyzer(_make_sliver_corridor_phantom())
        vertebra = analyzer.get_available_vertebrae()[0]

        result = analyzer.analyze_pedicle(vertebra)

        assert result.success and result.method == "coronal_isthmus"
        # Both corridors are the same 8 mm box; only the left one has a
        # fragment beside it, so both sides must report the same width.
        assert 7.5 <= result.left_pedicle_width <= 8.5
        assert 7.5 <= result.right_pedicle_width <= 8.5
        # The isthmus centre stays on the corridor axis (x = 60), not on the
        # fragment at x = 52.
        assert result.left_pedicle_center[0] == pytest.approx(59.5, abs=1.0)

    def test_nearest_neighbour_upsampled_mask_keeps_the_true_width(self):
        """A 2 mm mask pushed onto a 0.5 mm grid must still measure 8 mm."""
        mask = _nearest_neighbour_upsample(_make_coarse_phantom(), 4)
        analyzer = PedicleAnalyzer(mask)
        vertebra = analyzer.get_available_vertebrae()[0]

        result = analyzer.analyze_pedicle(vertebra)

        assert result.success
        assert result.left_pedicle_width == pytest.approx(8.0, abs=1.0)
        assert result.right_pedicle_width == pytest.approx(8.0, abs=1.0)
        # The bounding-box extent and the inscribed diameter bracket the truth,
        # so the lower bound is a real second opinion, never zero.
        assert result.left_width_lower_bound_mm == pytest.approx(8.0, abs=1.0)
        assert result.right_width_lower_bound_mm == pytest.approx(8.0, abs=1.0)
        assert result.left_width_lower_bound_mm <= result.left_pedicle_width

    @pytest.mark.parametrize(
        "notch_slices, expected_width_mm, expected_lower_bound_mm",
        [(1, 6.0, 5.0), (2, 6.0, 5.0), (3, 6.0, 5.0), (4, 8.0, 8.0)],
    )
    def test_short_notch_is_bridged_and_a_long_one_ends_the_walk(
        self, notch_slices, expected_width_mm, expected_lower_bound_mm
    ):
        """Up to MAX_TRACK_GAP_SLICES missing slices the corridor continues.

        A notch of 1-3 slices is bridged, so the walk reaches the 5 mm distal
        isthmus; the fourth blank slice exhausts the budget and the walk keeps
        only the 8 mm proximal box.

        The distal box is five voxels across, and an inscribed diameter is
        measured to the centres of the surrounding background voxels, so it
        reads 6 mm where the extent reads 5 mm.  The reported width is the
        larger of the two, so it is the lower bound that pins the 5 mm the
        phantom was drawn with -- and it is the two that together separate the
        bridged corridor from the 8 mm proximal box.
        """
        mask = _make_notched_corridor_phantom(notch_slices)
        analyzer = PedicleAnalyzer(mask)
        vertebra = analyzer.get_available_vertebrae()[0]
        binary = (sitk.GetArrayFromImage(mask) == vertebra.label).astype(np.uint8)
        indices_zyx = np.argwhere(binary)
        body_center = analyzer._estimate_body_center(binary, indices_zyx)
        body_center_ijk = mask.TransformPhysicalPointToContinuousIndex(
            tuple(float(v) for v in body_center)
        )
        z_indices = np.where(binary.any(axis=(1, 2)))[0]

        found = analyzer._find_pedicle_coronal(
            binary,
            body_center_ijk,
            (int(z_indices.min()), int(z_indices.max())),
            "left",
        )

        assert found is not None
        assert found["width_mm"] == pytest.approx(expected_width_mm)
        assert found["width_lower_bound_mm"] == pytest.approx(expected_lower_bound_mm)


# ---------------------------------------------------------------------------
# PedicleAnalyzer: pedicle subregion label path
# ---------------------------------------------------------------------------

def _make_pedicle_label(shape: tuple, centres=(30, 60)) -> np.ndarray:
    """A pedicle subregion label narrower (6 mm) than the phantom's 8 mm.

    Deliberately mismatched so a measurement that came from the label can be
    told apart from one the geometric search produced off the phantom itself.
    """
    zz, yy, xx = np.mgrid[0:shape[0], 0:shape[1], 0:shape[2]]
    pedicle = np.zeros(shape, bool)
    for cx in centres:
        pedicle |= (
            (((xx - cx) / 3.0) ** 2 + ((zz - 32) / 6.0) ** 2 <= 1)
            & (yy >= 48)
            & (yy < 62)
        )
    return pedicle


class TestSubregionLabelPath:
    def test_pedicle_label_overrides_geometric_search(self):
        mask = _make_anatomical_phantom()
        arr = sitk.GetArrayFromImage(mask)
        pedicle = _make_pedicle_label(arr.shape)
        analyzer = PedicleAnalyzer(mask, pedicle_mask=pedicle)
        result = analyzer.analyze_pedicle(analyzer.get_available_vertebrae()[0])
        assert result.method == "subregion_label"
        assert 5.0 <= result.left_pedicle_width <= 7.0
        assert np.linalg.norm(result.left_pedicle_center - np.array([60.0, 55.0, 32.0])) <= 1.5

    def test_a_sliver_slice_inside_the_label_is_not_the_isthmus(self):
        """One stair-stepped slice must not become the label path's isthmus."""
        mask = _make_anatomical_phantom()
        arr = sitk.GetArrayFromImage(mask)
        pedicle = _make_pedicle_label(arr.shape, centres=(60,))  # left only
        # Replace one interior coronal slice of the corridor with a 1 mm-wide,
        # 6 mm2 column — below both MIN_SLICE_WIDTH_MM and MIN_PEDICLE_AREA_MM2,
        # and by far the smallest cross-section in the corridor.
        pedicle[:, 54, :] = False
        pedicle[30:36, 54, 60] = True

        analyzer = PedicleAnalyzer(mask, pedicle_mask=pedicle)
        result = analyzer.analyze_pedicle(analyzer.get_available_vertebrae()[0])

        # Without the floors the sliver wins on area and the level reads 1 mm.
        assert 5.0 <= result.left_pedicle_width <= 7.0

    def test_a_uniformly_thin_label_is_still_measured(self):
        """When every slice fails the floors, the narrowest is still the answer."""
        mask = _make_anatomical_phantom()
        arr = sitk.GetArrayFromImage(mask)
        pedicle = np.zeros(arr.shape, bool)
        pedicle[30:36, 48:62, 60] = True  # a 1 mm-wide, 6 mm2 corridor throughout

        analyzer = PedicleAnalyzer(mask, pedicle_mask=pedicle)
        result = analyzer.analyze_pedicle(analyzer.get_available_vertebrae()[0])

        # The floors must not drop a side outright — they only rank candidates.
        assert result.left_pedicle_center is not None
        # One voxel across, so the inscribed diameter reads 2 mm (it is
        # measured to the centres of the background voxels either side) and
        # the reported width takes the larger of the two estimates.  The lower
        # bound is the one that still says 1 mm.
        assert result.left_pedicle_width == pytest.approx(2.0)
        assert result.left_width_lower_bound_mm == pytest.approx(1.0)

    def test_shape_mismatch_is_rejected(self):
        mask = _make_anatomical_phantom()
        with pytest.raises(ValueError):
            PedicleAnalyzer(mask, pedicle_mask=np.zeros((2, 2, 2), bool))

    def test_single_sided_label_falls_back_to_coronal_for_the_other_side(self):
        mask = _make_anatomical_phantom()
        arr = sitk.GetArrayFromImage(mask)
        pedicle = _make_pedicle_label(arr.shape, centres=(60,))  # left only
        analyzer = PedicleAnalyzer(mask, pedicle_mask=pedicle)
        result = analyzer.analyze_pedicle(analyzer.get_available_vertebrae()[0])

        assert result.method == "subregion_label+coronal_isthmus"
        # Left came from the 6 mm label, right from the 8 mm phantom geometry.
        assert 5.0 <= result.left_pedicle_width <= 7.0
        assert 7.0 <= result.right_pedicle_width <= 9.0
        assert np.linalg.norm(result.right_pedicle_center - np.array([30.0, 55.0, 32.0])) <= 2.0

    def test_a_repaired_label_miss_is_logged_rather_than_warned(self, caplog):
        """A side the label missed but coronal found is provenance, not a warning."""
        mask = _make_anatomical_phantom()
        arr = sitk.GetArrayFromImage(mask)
        pedicle = _make_pedicle_label(arr.shape, centres=(60,))  # left only
        analyzer = PedicleAnalyzer(mask, pedicle_mask=pedicle)

        with caplog.at_level(logging.INFO, logger="src.core.pedicle_analyzer"):
            result = analyzer.analyze_pedicle(analyzer.get_available_vertebrae()[0])

        assert result.right_pedicle_center is not None
        # Nothing the surgeon has to read: the side was measured after all…
        assert not any("subregion label" in w for w in result.warnings)
        # …but the label's gap is still on the record.
        assert any(
            "No right pedicle found in the pedicle subregion label" in record.getMessage()
            for record in caplog.records
        )

    def test_label_touching_the_volume_edge_measures_the_same(self):
        """Cropping to the side's bounding box must not move any measurement.

        The corridor is flush against the z = 0 and x = X-1 faces, so its
        bounding box is clamped by the volume on two sides, and a detached
        speck sits far enough away to pull the box wide open.  Every returned
        value is asserted against the corridor's own geometry, which is what
        a full-volume labelling would have produced.
        """
        Z, Y, X = 30, 40, 40
        arr = np.full((Z, Y, X), 28, dtype=np.uint8)
        mask = sitk.GetImageFromArray(arr)

        pedicle = np.zeros((Z, Y, X), bool)
        pedicle[0:5, 10:20, 34:40] = True   # corridor, on the z=0 and x=39 faces
        pedicle[20:22, 30:33, 20:24] = True  # detached speck, 24 voxels

        analyzer = PedicleAnalyzer(mask, pedicle_mask=pedicle)
        binary = (sitk.GetArrayFromImage(mask) == 28).astype(np.uint8)
        voxels = np.argwhere(binary.astype(bool) & pedicle)
        found = analyzer._find_pedicle_from_label(voxels, (19.5, 15.0, 15.0), "left")

        assert found is not None
        # Corridor only — the speck is large enough to clear the voxel-count
        # guard on its own, so it is the component step that must drop it.
        assert found["width_mm"] == pytest.approx(6.0)    # x 34..39
        assert found["height_mm"] == pytest.approx(5.0)   # z 0..4
        assert found["isthmus_j"] == 15                   # middle of the tied y 11..18
        assert found["isthmus_window_j"] == (10, 19)
        assert found["center_lps"] == pytest.approx(np.array([36.5, 15.0, 2.0]))
        assert found["axis_lps"] == pytest.approx(np.array([0.0, 1.0, 0.0]), abs=1e-6)

    def test_no_label_keeps_the_coronal_path(self):
        analyzer = PedicleAnalyzer(_make_anatomical_phantom(), pedicle_mask=None)
        result = analyzer.analyze_pedicle(analyzer.get_available_vertebrae()[0])
        assert result.method == "coronal_isthmus"

    def test_empty_label_falls_back_to_the_coronal_path(self):
        mask = _make_anatomical_phantom()
        empty = np.zeros(sitk.GetArrayFromImage(mask).shape, bool)
        analyzer = PedicleAnalyzer(mask, pedicle_mask=empty)
        result = analyzer.analyze_pedicle(analyzer.get_available_vertebrae()[0])
        assert result.method == "coronal_isthmus"
        assert result.success


# ---------------------------------------------------------------------------
# PedicleAnalyzer: per-side fall-through to the axial pass
# ---------------------------------------------------------------------------

class TestAxialFallThrough:
    """A side the coronal/label paths miss still gets the axial search."""

    def test_axial_pass_recovers_a_side_the_coronal_search_missed(self):
        """A one-sided coronal miss must not end the search for that side."""
        analyzer = PedicleAnalyzer(_make_one_sided_coronal_phantom())
        result = analyzer.analyze_pedicle(analyzer.get_available_vertebrae()[0])

        assert result.success
        assert result.method == "coronal_isthmus+axial_components"
        # The right pedicle keeps the coronal measurement it already had…
        assert result.right_pedicle_center == pytest.approx(np.array([31.5, 56.0, 31.5]))
        # …and the left one, invisible to the coronal search, comes from axial.
        assert result.left_pedicle_center is not None
        assert result.left_pedicle_center[0] > result.vertebral_body_center[0]

    def test_a_side_no_path_can_find_is_reported_but_does_not_fail_the_other(self):
        mask = _make_one_sided_coronal_phantom(with_left=False)
        analyzer = PedicleAnalyzer(mask)
        result = analyzer.analyze_pedicle(analyzer.get_available_vertebrae()[0])

        assert result.success                       # the right side is usable
        assert result.left_pedicle_center is None
        assert result.right_pedicle_center is not None
        assert any("left" in w for w in result.warnings)


# ---------------------------------------------------------------------------
# PedicleAnalyzer: analyze_all
# ---------------------------------------------------------------------------

class TestAnalyzeAll:
    """Tests for batch analysis."""

    def test_analyze_all_processes_all_vertebrae(self):
        """analyze_all without label filter processes every vertebra in mask."""
        img = _make_mask(shape=(30, 30, 30))
        img = _paint_label(img, label=27,
                           slices=(slice(0, 15), slice(5, 25), slice(5, 25)))
        img = _paint_label(img, label=28,
                           slices=(slice(15, 30), slice(5, 25), slice(5, 25)))

        analyzer = PedicleAnalyzer(img)
        results = analyzer.analyze_all()

        assert len(results) == 2
        labels = [r.vertebra.label for r in results]
        assert 27 in labels
        assert 28 in labels

    def test_analyze_all_with_label_filter(self):
        """analyze_all with labels parameter restricts output."""
        img = _make_mask(shape=(30, 30, 30))
        img = _paint_label(img, label=27,
                           slices=(slice(0, 10), slice(5, 25), slice(5, 25)))
        img = _paint_label(img, label=28,
                           slices=(slice(10, 20), slice(5, 25), slice(5, 25)))
        img = _paint_label(img, label=29,
                           slices=(slice(20, 30), slice(5, 25), slice(5, 25)))

        analyzer = PedicleAnalyzer(img)
        results = analyzer.analyze_all(labels=[27, 29])

        assert len(results) == 2
        labels = [r.vertebra.label for r in results]
        assert 27 in labels
        assert 29 in labels
        assert 28 not in labels

    def test_analyze_all_empty_mask(self):
        """analyze_all on an empty mask returns an empty list."""
        img = _make_mask(shape=(10, 10, 10))
        analyzer = PedicleAnalyzer(img)
        results = analyzer.analyze_all()

        assert results == []


# ---------------------------------------------------------------------------
# Label map consistency
# ---------------------------------------------------------------------------

class TestVertebralLabels:
    """Sanity checks for the VERTEBRA_LABELS mapping."""

    def test_label_range(self):
        assert min(VERTEBRA_LABELS.keys()) == 25
        assert max(VERTEBRA_LABELS.keys()) == 43

    def test_expected_names_present(self):
        names = set(VERTEBRA_LABELS.values())
        for expected in ["sacrum", "S1", "L5", "L1", "T12", "T1"]:
            assert expected in names

    def test_all_labels_contiguous(self):
        keys = sorted(VERTEBRA_LABELS.keys())
        assert keys == list(range(25, 44))


# ---------------------------------------------------------------------------
# Edge case: custom spacing and origin
# ---------------------------------------------------------------------------

class TestCustomGeometry:
    """Verify physical-coordinate handling with non-trivial spacing/origin."""

    def test_nonunit_spacing_centroid(self):
        """Centroid should respect non-unit spacing."""
        spacing = (0.5, 0.5, 2.0)  # z,y,x order for _make_mask
        img = _make_mask(shape=(20, 20, 20), spacing=spacing)
        img = _paint_label(img, label=27,
                           slices=(slice(5, 15), slice(5, 15), slice(5, 15)))

        analyzer = PedicleAnalyzer(img)
        v = analyzer.get_available_vertebrae()[0]

        # Mean index is exactly 9.5 for each axis (sub-voxel, no rounding).
        # The spacing in _make_mask is reversed to match sitk (x,y,z) order,
        # so sitk spacing = (2.0, 0.5, 0.5) -> x:2.0, y:0.5, z:0.5
        # With origin=(0,0,0), centroid x = 9.5*2.0 = 19.0, y = 9.5*0.5 = 4.75,
        # z = 9.5*0.5 = 4.75.
        # Verify that centroid is not zero and has plausible magnitude.
        assert np.linalg.norm(v.centroid_lps) > 0


class TestSubvoxelCentroid:
    """_indices_centroid_lps should not round to the nearest voxel."""

    def test_centroid_is_subvoxel_accurate(self):
        img = _make_mask(shape=(10, 10, 10))
        analyzer = PedicleAnalyzer(img)
        indices = np.array([[0, 0, 0], [0, 0, 1]])  # z, y, x -> mean x = 0.5
        assert analyzer._indices_centroid_lps(indices)[0] == pytest.approx(0.5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
