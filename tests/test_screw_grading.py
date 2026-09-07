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

    def test_crop_margin_mm_is_exposed(self):
        grader = ScrewGrader(_cube_mask(), crop_margin_mm=8.0)
        assert grader.crop_margin_mm == 8.0
