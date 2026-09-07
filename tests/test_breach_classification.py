import numpy as np
import pytest
import SimpleITK as sitk

from src.core.breach_classification import facet_violation_grade, heary_direction
from src.core.screw_grading import ScrewGrader


@pytest.mark.parametrize("offset,side,expected", [
    ((-1.0, 0, 0), "left", "medial"), ((1.0, 0, 0), "left", "lateral"),
    ((1.0, 0, 0), "right", "medial"), ((-1.0, 0, 0), "right", "lateral"),
    ((0, -1.0, 0), "left", "anterior"), ((0, 1.0, 0), "left", "posterior"),
    ((0, 0, 1.0), "left", "superior"), ((0, 0, -1.0), "left", "inferior")])
def test_heary_direction(offset, side, expected):
    c = np.array([10.0, 10.0, 10.0])
    assert heary_direction(c + np.asarray(offset), c, side) == expected


def test_heary_direction_uses_the_dominant_axis():
    c = np.array([10.0, 10.0, 10.0])
    # 3 mm medial, 1 mm superior -> medial dominates
    assert heary_direction(c + np.array([-3.0, 0.0, 1.0]), c, "left") == "medial"
    assert heary_direction(c + np.array([-1.0, 0.0, 3.0]), c, "left") == "superior"


def test_heary_direction_is_none_for_a_coincident_point():
    c = np.array([10.0, 10.0, 10.0])
    assert heary_direction(c, c, "left") == "none"
    assert heary_direction(c + np.array([1e-12, 0.0, 0.0]), c, "left") == "none"


def _two_level(spacing=(1.0, 1.0, 1.0), gap=2):
    """L4 (28) below L3 (29) with ``gap`` empty slices between them.

    Array indices are ``(z, y, x)``; L4 covers z 10..33 and L3 starts at
    ``34 + gap``. Both span y 10..49 and x 15..44.
    """
    arr = np.zeros((80, 60, 60), np.uint8)
    arr[10:34, 10:50, 15:45] = 28
    arr[34 + gap:74, 10:50, 15:45] = 29
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing(spacing)
    return img


def test_facet_grade_0_when_far_from_cephalad_level():
    grader = ScrewGrader(_two_level())
    grade, text = facet_violation_grade(grader, (30.0, 48.0, 20.0), (30.0, 12.0, 20.0), 6.0, 28)
    assert grade == 0
    assert text == "no facet contact"


def test_facet_grade_3_when_head_penetrates_cephalad_level():
    grader = ScrewGrader(_two_level())
    grade, text = facet_violation_grade(grader, (30.0, 48.0, 38.0), (30.0, 12.0, 20.0), 6.0, 28)
    assert grade == 3 and "penetrates" in text


def test_facet_grade_1_when_the_head_abuts_the_cephalad_level():
    # 8 empty slices: L4 ends at z 33, L3 starts at z 42. A 6 mm screw centred
    # on z 38 reaches z 41 -> exactly 1 mm from the cephalad label.
    grader = ScrewGrader(_two_level(gap=8))
    grade, text = facet_violation_grade(grader, (30.0, 48.0, 38.0), (30.0, 12.0, 38.0), 6.0, 28)
    assert grade == 1
    assert text == "screw abuts the cephalad facet"


def test_facet_grade_2_when_penetration_is_under_a_millimetre():
    # 0.5 mm voxels: L3 starts at index 36 (z = 18.0 mm). A 1 mm screw centred
    # on z = 17.5 mm reaches the first cephalad voxel only, 0.5 mm deep.
    grader = ScrewGrader(_two_level(spacing=(0.5, 0.5, 0.5)))
    grade, text = facet_violation_grade(grader, (15.0, 24.0, 17.5), (15.0, 6.0, 17.5), 1.0, 28)
    assert grade == 2
    assert text == "screw enters the cephalad facet (<1 mm)"


def test_facet_grade_0_when_the_cephalad_label_is_absent():
    arr = np.zeros((80, 60, 60), np.uint8)
    arr[10:34, 10:50, 15:45] = 28
    grader = ScrewGrader(sitk.GetImageFromArray(arr))
    grade, text = facet_violation_grade(grader, (30.0, 48.0, 38.0), (30.0, 12.0, 20.0), 6.0, 28)
    assert (grade, text) == (0, "no cephalad vertebra segmented")


def test_facet_grade_0_for_the_topmost_labelled_vertebra():
    arr = np.zeros((80, 60, 60), np.uint8)
    arr[10:34, 10:50, 15:45] = 43        # T1: no cephalad label exists
    grader = ScrewGrader(sitk.GetImageFromArray(arr))
    grade, text = facet_violation_grade(grader, (30.0, 48.0, 20.0), (30.0, 12.0, 20.0), 6.0, 43)
    assert (grade, text) == (0, "no cephalad vertebra segmented")


def test_facet_grade_only_considers_the_proximal_screw():
    """A screw whose distal end lies in the cephalad label still grades 0.

    Only the proximal 30 % near the entry point can violate a facet joint.
    """
    grader = ScrewGrader(_two_level())
    # Entry deep inside L4, target inside L3: the head is 20+ mm from L3.
    grade, _ = facet_violation_grade(grader, (30.0, 48.0, 12.0), (30.0, 12.0, 45.0), 6.0, 28)
    assert grade == 0


def test_facet_grade_handles_a_degenerate_trajectory():
    grader = ScrewGrader(_two_level())
    grade, _ = facet_violation_grade(grader, (30.0, 48.0, 20.0), (30.0, 48.0, 20.0), 6.0, 28)
    assert grade == 0
