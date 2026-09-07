import numpy as np
import pytest
import SimpleITK as sitk

from src.core.screw_grading import GradeResult, ScrewGrader


def _cube_mask(label=28, size=60, lo=20, hi=40, spacing=(1.0, 1.0, 1.0)):
    arr = np.zeros((size, size, size), dtype=np.uint8)
    arr[lo:hi, lo:hi, lo:hi] = label
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing(spacing)
    return img


def _ct_like(mask, inside_hu=350, outside_hu=-50):
    arr = sitk.GetArrayFromImage(mask)
    ct = sitk.GetImageFromArray(np.where(arr > 0, inside_hu, outside_hu).astype(np.int16))
    ct.CopyInformation(mask)
    return ct


def _asymmetric_mask(label=28, spacing=(0.5, 1.0, 2.0)):
    """A label with a different extent and voxel size on every axis.

    Array indices are ``(z, y, x)``; ``spacing`` is SimpleITK's ``(x, y, z)``.
    The label covers x index 10..49 (5.0-24.5 mm), y index 22..29 (22-29 mm)
    and z index 20..39 (40-78 mm), so an inverted index order or a reversed
    sampling order changes every distance it produces.
    """
    arr = np.zeros((60, 60, 60), dtype=np.uint8)
    arr[20:40, 22:30, 10:50] = label
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing(spacing)
    return img


class TestGradeFromBreach:
    @pytest.mark.parametrize("breach,grade", [(0.0, "A"), (1.9, "B"), (2.0, "C"), (3.9, "C"), (4.0, "D"), (5.9, "D"), (6.0, "E"), (25.0, "E")])
    def test_thresholds(self, breach, grade):
        assert ScrewGrader.grade_from_breach(breach) == grade


class TestScrewGrader:
    def test_contained_screw_is_grade_a_with_wall_distance(self):
        mask = _cube_mask()
        grader = ScrewGrader(mask, _ct_like(mask))
        result = grader.grade(entry=(30.0, 35.0, 30.0), target=(30.0, 25.0, 30.0), diameter_mm=6.0)
        assert isinstance(result, GradeResult)
        assert result.grade == "A"
        assert result.breach_mm == 0.0
        # screw ends are 5 mm from the y walls; radial surface is 7 mm from the x/z walls -> min ~5 mm
        assert 4.0 <= result.min_wall_mm <= 6.0
        assert result.label == 28
        assert result.mean_hu == pytest.approx(350.0)

    def test_fully_outside_bone_is_grade_e(self):
        mask = _cube_mask()
        grader = ScrewGrader(mask)
        result = grader.grade(entry=(5.0, 5.0, 30.0), target=(5.0, 50.0, 30.0), diameter_mm=6.5, label=28)
        assert result.grade == "E"
        assert result.breach_mm >= 6.0

    def test_partial_breach_measures_distance(self):
        mask = _cube_mask()  # label occupies x in [20, 40)
        grader = ScrewGrader(mask)
        # centreline in the last inside voxel column (x=39); radius 3 -> surface at x=42, ~3 mm outside
        result = grader.grade(entry=(39.0, 35.0, 30.0), target=(39.0, 25.0, 30.0), diameter_mm=6.0, label=28)
        assert result.grade in {"B", "C"}
        assert 2.0 <= result.breach_mm <= 3.5

    def test_detect_label_picks_dominant_vertebra(self):
        arr = np.zeros((60, 60, 60), dtype=np.uint8)
        arr[20:40, 20:40, 20:40] = 28
        arr[42:58, 20:40, 20:40] = 29
        mask = sitk.GetImageFromArray(arr)
        grader = ScrewGrader(mask)
        assert grader.detect_label((30.0, 38.0, 30.0), (30.0, 22.0, 30.0)) == 28
        assert grader.detect_label((30.0, 38.0, 50.0), (30.0, 22.0, 50.0)) == 29
        assert grader.detect_label((5.0, 5.0, 5.0), (6.0, 6.0, 6.0)) is None

    def test_grade_without_label_returns_none(self):
        grader = ScrewGrader(_cube_mask())
        assert grader.grade((5.0, 5.0, 5.0), (6.0, 6.0, 6.0), 6.0) is None

    def test_anisotropic_spacing_uses_physical_distance(self):
        mask = _cube_mask(spacing=(0.5, 0.5, 2.0))  # label x,y in [10,20) mm, z in [40,80) mm
        grader = ScrewGrader(mask)
        # screw ends 3 mm from the y walls; radius 2 -> surface 3 mm from the x walls; z is 20 mm away
        result = grader.grade(entry=(15.0, 17.0, 60.0), target=(15.0, 13.0, 60.0), diameter_mm=4.0, label=28)
        assert result.grade == "A"
        assert result.min_wall_mm == pytest.approx(3.0, abs=0.6)

    def test_asymmetric_label_wall_distance_is_axis_specific(self):
        mask = _asymmetric_mask()
        grader = ScrewGrader(mask)
        # Screw runs along +x from x=6.0 mm (3 voxels of 0.5 mm inside the x
        # wall) to x=16.0 mm at y=26.0 mm, z=60.0 mm; the 1 mm radius stays
        # well inside on y (3 mm to the wall) and z (18 mm to the wall).
        result = grader.grade(entry=(6.0, 26.0, 60.0), target=(16.0, 26.0, 60.0), diameter_mm=2.0, label=28)
        assert result.breach_mm == 0.0
        assert result.min_wall_mm == pytest.approx(1.5, abs=1e-6)

    def test_asymmetric_label_breach_is_axis_specific(self):
        mask = _asymmetric_mask()
        grader = ScrewGrader(mask)
        # Centreline on the last inside z slice (z=78 mm); the 4 mm radius puts
        # the surface at z=82 mm, two 2 mm voxels beyond the z wall, while the
        # widest y excursion (y=30 mm) is only one 1 mm voxel outside.
        result = grader.grade(entry=(6.0, 26.0, 78.0), target=(16.0, 26.0, 78.0), diameter_mm=8.0, label=28)
        assert result.breach_mm == pytest.approx(4.0, abs=1e-6)
        assert result.grade == "D"
        assert result.min_wall_mm == 0.0

    def test_sample_outside_the_cropped_maps_scores_the_crop_margin(self):
        grader = ScrewGrader(_cube_mask(), crop_margin_mm=3.0)
        # The label spans index 20..39, so a 3 mm margin crops to 17..42 and the
        # whole screw sits outside the distance maps.
        result = grader.grade(entry=(5.0, 5.0, 5.0), target=(5.0, 5.0, 15.0), diameter_mm=2.0, label=28)
        assert result.breach_mm == grader.crop_margin_mm == 3.0
        assert result.grade == "C"

    def test_ct_image_with_a_different_grid_is_rejected(self):
        mask = _cube_mask()
        rescaled = _ct_like(mask)
        rescaled.SetSpacing((1.0, 1.0, 2.0))
        with pytest.raises(ValueError, match="same grid"):
            ScrewGrader(mask, rescaled)

        shifted = _ct_like(mask)
        shifted.SetOrigin((3.0, 0.0, 0.0))
        with pytest.raises(ValueError, match="same grid"):
            ScrewGrader(mask, shifted)

        resized = sitk.GetImageFromArray(np.zeros((30, 60, 60), dtype=np.int16))
        with pytest.raises(ValueError, match="same grid"):
            ScrewGrader(mask, resized)

    def test_crop_margin_mm_is_exposed(self):
        grader = ScrewGrader(_cube_mask(), crop_margin_mm=8.0)
        assert grader.crop_margin_mm == 8.0
