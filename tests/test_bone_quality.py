import numpy as np
import pytest
import SimpleITK as sitk

from src.core.bone_quality import BoneQualityMetrics, assess_bone_quality, vertebral_body_hu
from src.core.screw_grading import ScrewGrader


def _phantom(body_hu=250, pedicle_hu=400):
    arr = np.zeros((60, 80, 60), np.uint8)          # z, y, x
    arr[20:40, 10:40, 20:40] = 28                    # body: y 10..39
    arr[26:34, 40:56, 26:34] = 28                    # pedicle-like bar: y 40..55, 8 mm square
    hu = np.full(arr.shape, -50, np.int16)
    hu[20:40, 10:40, 20:40] = body_hu
    hu[26:34, 40:56, 26:34] = pedicle_hu
    mask = sitk.GetImageFromArray(arr)
    ct = sitk.GetImageFromArray(hu)
    return ct, mask


ENTRY = (30.0, 54.0, 30.0)
TARGET = (30.0, 12.0, 30.0)
BODY_CENTRE = (30.0, 25.0, 30.0)
ISTHMUS_CENTRE = (30.0, 48.0, 30.0)


class TestVertebralBodyHu:
    def test_body_hu_uses_trabecular_roi(self):
        ct, mask = _phantom(body_hu=250)
        assert vertebral_body_hu(ct, mask, 28, body_center_lps=BODY_CENTRE) == pytest.approx(250.0)

    def test_roi_excludes_other_labels_and_background(self):
        # A ring of a different label around the body must not drag the mean down.
        ct, mask = _phantom(body_hu=250)
        arr = sitk.GetArrayFromImage(mask)
        hu = sitk.GetArrayFromImage(ct)
        arr[20:40, 10:40, 20:26] = 29    # overlaps the ellipsoid (x 22..38)
        hu[20:40, 10:40, 20:26] = -900
        mask = sitk.GetImageFromArray(arr)
        ct = sitk.GetImageFromArray(hu)
        assert vertebral_body_hu(ct, mask, 28, body_center_lps=BODY_CENTRE) == pytest.approx(250.0)

    def test_returns_none_when_roi_too_small(self):
        ct, mask = _phantom()
        assert vertebral_body_hu(ct, mask, 28, body_center_lps=(5.0, 70.0, 5.0)) is None

    def test_returns_none_for_absent_label(self):
        ct, mask = _phantom()
        assert vertebral_body_hu(ct, mask, 99, body_center_lps=BODY_CENTRE) is None

    def test_radii_are_physical_millimetres(self):
        ct, mask = _phantom()
        big = vertebral_body_hu(ct, mask, 28, BODY_CENTRE, radii_mm=(8.0, 8.0, 6.0))
        assert big == pytest.approx(250.0)
        # A 1 mm ball holds fewer than 20 voxels at 1 mm spacing -> no value.
        assert vertebral_body_hu(ct, mask, 28, BODY_CENTRE, radii_mm=(1.0, 1.0, 1.0)) is None


class TestAssessBoneQuality:
    def test_metrics_and_no_warnings_for_good_bone(self):
        ct, mask = _phantom(body_hu=250, pedicle_hu=400)
        grader = ScrewGrader(mask, ct)
        m = assess_bone_quality(grader, ENTRY, TARGET, 4.0, 28,
                                body_center_lps=BODY_CENTRE, isthmus_center_lps=ISTHMUS_CENTRE)
        assert isinstance(m, BoneQualityMetrics)
        assert m.pedicle_mean_hu == pytest.approx(400.0, abs=30)
        assert m.body_mean_hu == pytest.approx(250.0)
        assert m.trajectory_mean_hu > 250
        assert m.trajectory_min_hu == pytest.approx(250.0)
        assert m.trajectory_body_ratio == pytest.approx(m.trajectory_mean_hu / 250.0)
        assert m.warnings == []

    def test_low_hu_raises_expected_warnings(self):
        ct, mask = _phantom(body_hu=100, pedicle_hu=110)
        grader = ScrewGrader(mask, ct)
        m = assess_bone_quality(grader, ENTRY, TARGET, 4.0, 28, body_center_lps=BODY_CENTRE)
        text = "\n".join(m.warnings)
        assert "Trajectory HU" in text and "loosening" in text
        assert "osteoporosis" in text

    def test_loosening_warning_wording(self):
        ct, mask = _phantom(body_hu=100, pedicle_hu=100)
        grader = ScrewGrader(mask, ct)
        m = assess_bone_quality(grader, ENTRY, TARGET, 4.0, 28)
        assert m.warnings == [
            "Trajectory HU 100 below 123 HU — loosening risk "
            "(consider larger diameter, augmentation, or CBT)"
        ]

    def test_osteoporosis_warning_wording(self):
        ct, mask = _phantom(body_hu=100, pedicle_hu=400)
        grader = ScrewGrader(mask, ct)
        m = assess_bone_quality(grader, ENTRY, TARGET, 4.0, 28, body_center_lps=BODY_CENTRE)
        assert "Vertebral body HU 100 suggests osteoporosis (<132 HU)" in m.warnings
        assert not any("low bone density" in w for w in m.warnings)

    def test_low_bmd_warning_only_when_not_osteoporotic(self):
        ct, mask = _phantom(body_hu=135, pedicle_hu=135)
        grader = ScrewGrader(mask, ct)
        m = assess_bone_quality(grader, ENTRY, TARGET, 4.0, 28, body_center_lps=BODY_CENTRE)
        assert m.warnings == ["Vertebral body HU 135 suggests low bone density (<141 HU)"]

    def test_ratio_warning_when_trajectory_weaker_than_body(self):
        ct, mask = _phantom(body_hu=300, pedicle_hu=150)
        grader = ScrewGrader(mask, ct)
        m = assess_bone_quality(grader, ENTRY, TARGET, 4.0, 28, body_center_lps=BODY_CENTRE)
        assert m.trajectory_body_ratio < 1.0
        assert m.warnings == [
            f"Trajectory/body HU ratio {m.trajectory_body_ratio:.2f} below 1.0"
        ]

    def test_custom_trajectory_threshold_is_used(self):
        ct, mask = _phantom(body_hu=250, pedicle_hu=400)
        grader = ScrewGrader(mask, ct)
        m = assess_bone_quality(grader, ENTRY, TARGET, 4.0, 28, trajectory_threshold=500.0)
        assert any(w.startswith("Trajectory HU") and "below 500 HU" in w for w in m.warnings)

    def test_pedicle_mean_is_none_without_isthmus_centre(self):
        ct, mask = _phantom()
        m = assess_bone_quality(ScrewGrader(mask, ct), ENTRY, TARGET, 4.0, 28,
                                body_center_lps=BODY_CENTRE)
        assert m.pedicle_mean_hu is None

    def test_pedicle_mean_uses_only_samples_near_the_isthmus(self):
        # Pedicle bar is far denser than the body, so restricting to a 10 mm
        # ball around the isthmus must beat the whole-trajectory mean.
        ct, mask = _phantom(body_hu=250, pedicle_hu=800)
        m = assess_bone_quality(ScrewGrader(mask, ct), ENTRY, TARGET, 4.0, 28,
                                isthmus_center_lps=ISTHMUS_CENTRE)
        assert m.pedicle_mean_hu > m.trajectory_mean_hu
        assert m.pedicle_mean_hu == pytest.approx(800.0, abs=60)

    def test_body_metrics_are_none_without_a_ct(self):
        _, mask = _phantom()
        m = assess_bone_quality(ScrewGrader(mask), ENTRY, TARGET, 4.0, 28,
                                body_center_lps=BODY_CENTRE, isthmus_center_lps=ISTHMUS_CENTRE)
        assert m.trajectory_mean_hu is None
        assert m.trajectory_min_hu is None
        assert m.pedicle_mean_hu is None
        assert m.body_mean_hu is None
        assert m.trajectory_body_ratio is None
        assert m.warnings == []

    def test_trajectory_entirely_outside_the_volume_yields_no_metrics(self):
        ct, mask = _phantom()
        m = assess_bone_quality(ScrewGrader(mask, ct), (-90.0, -90.0, -90.0), (-80.0, -90.0, -90.0),
                                4.0, 28)
        assert m.trajectory_mean_hu is None
        assert m.trajectory_min_hu is None
        assert m.warnings == []
