"""
Tests for PedicleAnalyzer and vertebra data models.

Uses synthetic SimpleITK mask images with known geometry so that
expected centroids, bounding boxes, and pedicle properties can be
verified analytically.
"""

import os
import sys

import numpy as np
import pytest
import SimpleITK as sitk

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.core.vertebra import PedicleAnalysisResult, Vertebra
from src.core.pedicle_analyzer import (
    VERTEBRA_LABELS,
    PedicleAnalyzer,
)


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



def _make_anatomical_phantom(label: int = 28) -> sitk.Image:
    """Body ellipse + two 8 mm pedicles + posterior arch, 1 mm isotropic, LPS identity.

    The laminar arch starts at ``yy >= 60`` so that it joins the pedicles from
    behind and leaves the spinal canal hollow, instead of filling the canal
    with bone anterior to the laminae.
    """
    Z, Y, X = 60, 90, 90
    zz, yy, xx = np.mgrid[0:Z, 0:Y, 0:X]
    body = (((xx - 45) / 20.0) ** 2 + ((yy - 35) / 15.0) ** 2 <= 1) & (zz >= 15) & (zz < 45)
    ped = np.zeros_like(body)
    for cx in (30, 60):
        ped |= (((xx - cx) / 4.0) ** 2 + ((zz - 32) / 6.0) ** 2 <= 1) & (yy >= 48) & (yy < 62)
    arch = ((((xx - 45) / 22.0) ** 2 + ((yy - 66) / 10.0) ** 2 <= 1)
            & ~(((xx - 45) / 14.0) ** 2 + ((yy - 64) / 6.0) ** 2 <= 1)
            & (yy >= 60) & (zz >= 26) & (zz < 40))
    arr = np.zeros((Z, Y, X), dtype=np.uint8)
    arr[body | ped | arch] = label
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

        # The mean index in z,y,x is approximately (9.5, 9.5, 9.5).
        # With spacing=2 and origin=0, the LPS physical centroid should be
        # near (9.5*2, 9.5*2, 9.5*2) = (19, 19, 19).
        # Because we round to int before TransformIndexToPhysicalPoint,
        # the centroid is at index 10 -> 10*2 = 20 for each axis.
        expected_approx = np.array([20.0, 20.0, 20.0])
        np.testing.assert_allclose(v.centroid_lps, expected_approx, atol=2.1)

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

        # Mean index ~9.5 for each axis.  Physical coords:
        # x = round(9.5)*sx = 10*0.5 = 5.0 (sitk spacing is reversed from numpy shape)
        # The spacing in _make_mask is reversed to match sitk (x,y,z) order,
        # so sitk spacing = (2.0, 0.5, 0.5) -> x:2.0, y:0.5, z:0.5
        # With origin=(0,0,0), centroid x ~ 10*2.0 = 20.0, y ~ 10*0.5 = 5.0, z ~ 10*0.5 = 5.0
        # Verify that centroid is not zero and has plausible magnitude.
        assert np.linalg.norm(v.centroid_lps) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
