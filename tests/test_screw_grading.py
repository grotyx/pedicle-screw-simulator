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


class TestBreachPoint:
    def test_contained_screw_has_no_breach_point(self):
        grader = ScrewGrader(_cube_mask())
        result = grader.grade(entry=(30.0, 35.0, 30.0), target=(30.0, 25.0, 30.0), diameter_mm=6.0)
        assert result.breach_mm == 0.0
        assert result.breach_point_lps is None
        assert result.breach_centre_lps is None

    def test_breach_point_is_the_worst_sample_and_its_centreline_point(self):
        grader = ScrewGrader(_cube_mask())  # label occupies x in [20, 40)
        entry, target = (39.0, 35.0, 30.0), (39.0, 25.0, 30.0)
        result = grader.grade(entry=entry, target=target, diameter_mm=6.0, label=28)
        assert result.breach_mm > 0.0
        point = np.asarray(result.breach_point_lps)
        centre = np.asarray(result.breach_centre_lps)
        # The deepest sample sits one radius beyond the +x wall of the label.
        assert point[0] == pytest.approx(42.0)
        assert centre[0] == pytest.approx(39.0)
        assert centre[2] == pytest.approx(30.0)
        # The centreline point belongs to the trajectory and matches the sample.
        assert np.linalg.norm(point - centre) == pytest.approx(3.0)
        assert 25.0 <= centre[1] <= 35.0
        assert isinstance(result.breach_point_lps, tuple)
        assert isinstance(result.breach_centre_lps, tuple)

    def test_breach_point_is_one_of_the_cylinder_samples(self):
        grader = ScrewGrader(_asymmetric_mask())
        entry, target = (6.0, 26.0, 78.0), (16.0, 26.0, 78.0)
        result = grader.grade(entry=entry, target=target, diameter_mm=8.0, label=28)
        points = grader.cylinder_points(entry, target, 8.0)
        assert np.isclose(points, np.asarray(result.breach_point_lps)).all(axis=1).any()
        centres = grader.cylinder_points(entry, target, 0.0)
        assert np.isclose(centres, np.asarray(result.breach_centre_lps)).all(axis=1).any()

    def test_breach_point_without_radial_samples_is_the_centreline_point(self):
        grader = ScrewGrader(_cube_mask(), radial_samples=0)
        result = grader.grade(entry=(5.0, 30.0, 30.0), target=(15.0, 30.0, 30.0), diameter_mm=6.0, label=28)
        assert result.breach_mm > 0.0
        assert result.breach_point_lps == result.breach_centre_lps


class TestDistancesAtPoints:
    def test_distances_match_the_label_geometry(self):
        grader = ScrewGrader(_cube_mask())  # label occupies index 20..39 on every axis
        points = np.array([[30.0, 30.0, 30.0], [42.0, 30.0, 30.0], [39.0, 30.0, 30.0]])
        d_out, d_in = grader.distances_at_points(points, 28)
        assert d_out == pytest.approx([0.0, 3.0, 0.0])
        assert d_in[0] == pytest.approx(10.0)
        assert d_in[1] == 0.0
        assert d_in[2] == pytest.approx(1.0)

    def test_distances_for_an_absent_label_are_the_crop_margin(self):
        grader = ScrewGrader(_cube_mask(), crop_margin_mm=7.0)
        d_out, d_in = grader.distances_at_points(np.array([[30.0, 30.0, 30.0]]), 29)
        assert d_out == pytest.approx([7.0])
        assert d_in == pytest.approx([0.0])

    def test_distances_accept_an_empty_array(self):
        grader = ScrewGrader(_cube_mask())
        d_out, d_in = grader.distances_at_points(np.empty((0, 3)), 28)
        assert d_out.shape == (0,) and d_in.shape == (0,)

    def test_distances_agree_with_the_grade_breach(self):
        grader = ScrewGrader(_cube_mask())
        entry, target = (39.0, 35.0, 30.0), (39.0, 25.0, 30.0)
        result = grader.grade(entry=entry, target=target, diameter_mm=6.0, label=28)
        d_out, _ = grader.distances_at_points(grader.cylinder_points(entry, target, 6.0), 28)
        assert float(d_out.max()) == pytest.approx(result.breach_mm)


class TestSamplingHelpers:
    def test_cylinder_points_shape_and_ordering(self):
        grader = ScrewGrader(_cube_mask(), sample_step_mm=1.0, radial_samples=8)
        entry, target = (30.0, 35.0, 30.0), (30.0, 25.0, 30.0)
        points = grader.cylinder_points(entry, target, diameter_mm=6.0)
        assert isinstance(points, np.ndarray)
        assert points.dtype == np.float64
        # 10 mm at 1 mm steps -> 11 centres, each with 1 + 8 samples
        assert points.shape == (11 * 9, 3)
        # The first sample of each block is the centreline point itself.
        centres = points[::9]
        assert centres[0] == pytest.approx(np.asarray(entry))
        assert centres[-1] == pytest.approx(np.asarray(target))
        assert centres[:, 1] == pytest.approx(np.arange(35.0, 24.9, -1.0))
        # Every radial sample sits exactly one radius from its centre.
        for block in range(11):
            centre = points[block * 9]
            radial = points[block * 9 + 1: block * 9 + 9]
            assert np.linalg.norm(radial - centre, axis=1) == pytest.approx(3.0)

    def test_cylinder_points_radial_samples_are_perpendicular(self):
        grader = ScrewGrader(_cube_mask(), sample_step_mm=2.0)
        points = grader.cylinder_points((30.0, 35.0, 30.0), (30.0, 25.0, 30.0), 6.0)
        # Trajectory runs along -y, so every radial offset must keep y constant.
        assert points[1:9, 1] == pytest.approx(35.0)

    def test_cylinder_points_honours_radial_samples_setting(self):
        grader = ScrewGrader(_cube_mask(), sample_step_mm=1.0, radial_samples=4)
        points = grader.cylinder_points((30.0, 35.0, 30.0), (30.0, 25.0, 30.0), 6.0)
        assert points.shape == (11 * 5, 3)

    def test_cylinder_points_for_a_degenerate_trajectory(self):
        grader = ScrewGrader(_cube_mask())
        points = grader.cylinder_points((30.0, 30.0, 30.0), (30.0, 30.0, 30.0), 6.0)
        assert points.ndim == 2 and points.shape[1] == 3
        assert points == pytest.approx(np.full(points.shape, 30.0))

    def test_hu_at_points_matches_the_ct_and_is_nan_outside(self):
        mask = _cube_mask()
        grader = ScrewGrader(mask, _ct_like(mask, inside_hu=350, outside_hu=-50))
        hu = grader.hu_at_points(np.array([
            [30.0, 30.0, 30.0],     # inside the label
            [5.0, 5.0, 5.0],        # inside the volume, outside the label
            [-10.0, 30.0, 30.0],    # outside the volume
            [30.0, 30.0, 999.0],    # outside the volume
        ]))
        assert hu.shape == (4,)
        assert hu[0] == pytest.approx(350.0)
        assert hu[1] == pytest.approx(-50.0)
        assert np.isnan(hu[2]) and np.isnan(hu[3])

    def test_hu_at_points_is_all_nan_without_a_ct(self):
        grader = ScrewGrader(_cube_mask())
        hu = grader.hu_at_points(np.array([[30.0, 30.0, 30.0], [31.0, 30.0, 30.0]]))
        assert hu.shape == (2,)
        assert np.all(np.isnan(hu))

    def test_hu_at_points_accepts_an_empty_array(self):
        mask = _cube_mask()
        grader = ScrewGrader(mask, _ct_like(mask))
        assert grader.hu_at_points(np.empty((0, 3))).shape == (0,)

    def test_hu_at_points_matches_simpleitk_index_rounding(self):
        mask = _cube_mask(spacing=(0.5, 1.0, 2.0))
        ct_arr = np.arange(60 ** 3, dtype=np.int32).reshape(60, 60, 60)
        ct = sitk.GetImageFromArray(ct_arr)
        ct.CopyInformation(mask)
        grader = ScrewGrader(mask, ct)
        rng = np.random.default_rng(7)
        points = rng.uniform(-5.0, 40.0, size=(500, 3))
        expected = []
        for point in points:
            idx = mask.TransformPhysicalPointToIndex([float(v) for v in point])
            size = mask.GetSize()
            inside = all(0 <= idx[a] < size[a] for a in range(3))
            expected.append(float(ct_arr[idx[2], idx[1], idx[0]]) if inside else np.nan)
        np.testing.assert_array_equal(grader.hu_at_points(points), np.asarray(expected))

    def test_label_array_and_mask_image_expose_the_segmentation(self):
        mask = _cube_mask()
        grader = ScrewGrader(mask)
        labels = grader.label_array()
        assert labels.shape == (60, 60, 60)          # (z, y, x)
        assert int(labels[30, 30, 30]) == 28
        assert int(labels[5, 5, 5]) == 0
        assert grader.mask_image() is mask

    def test_non_identity_direction_is_rejected(self):
        mask = _cube_mask()
        mask.SetDirection((-1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0))
        with pytest.raises(ValueError, match="identity"):
            ScrewGrader(mask)


class TestEvaluateBatch:
    def test_batch_matches_single_evaluation(self):
        mask = _cube_mask(); ct = _ct_like(mask)
        grader = ScrewGrader(mask, ct)
        entries = np.array([[30.0, 38.0, 30.0], [39.0, 38.0, 30.0], [5.0, 5.0, 30.0]])
        targets = np.array([[30.0, 22.0, 30.0], [39.0, 22.0, 30.0], [5.0, 50.0, 30.0]])
        batch = grader.evaluate_batch(entries, targets, 6.0, 28)
        for i in range(3):
            single = grader.grade(entries[i], targets[i], 6.0, label=28)
            assert batch.breach_mm[i] == pytest.approx(single.breach_mm, abs=0.51)
            assert batch.min_wall_mm[i] == pytest.approx(single.min_wall_mm, abs=0.51)
        assert np.isfinite(batch.mean_hu[0])

    def test_batch_shapes_and_dtypes(self):
        mask = _cube_mask()
        grader = ScrewGrader(mask, _ct_like(mask))
        entries = np.array([[30.0, 38.0, 30.0], [39.0, 38.0, 30.0]])
        targets = np.array([[30.0, 22.0, 30.0], [39.0, 22.0, 30.0]])
        batch = grader.evaluate_batch(entries, targets, 6.0, 28)
        for values in (batch.breach_mm, batch.min_wall_mm, batch.mean_hu, batch.min_hu):
            assert values.shape == (2,)
            assert values.dtype == np.float64

    def test_min_wall_is_zero_where_the_screw_breaches(self):
        grader = ScrewGrader(_cube_mask())
        entries = np.array([[5.0, 5.0, 30.0]])
        targets = np.array([[5.0, 50.0, 30.0]])
        batch = grader.evaluate_batch(entries, targets, 6.0, 28)
        assert batch.breach_mm[0] > 0.0
        assert batch.min_wall_mm[0] == 0.0

    def test_hu_is_nan_without_a_ct(self):
        grader = ScrewGrader(_cube_mask())
        entries = np.array([[30.0, 38.0, 30.0]])
        targets = np.array([[30.0, 22.0, 30.0]])
        batch = grader.evaluate_batch(entries, targets, 6.0, 28)
        assert np.isnan(batch.mean_hu[0])
        assert np.isnan(batch.min_hu[0])

    def test_empty_batch_returns_empty_arrays(self):
        mask = _cube_mask()
        grader = ScrewGrader(mask, _ct_like(mask))
        batch = grader.evaluate_batch(np.empty((0, 3)), np.empty((0, 3)), 6.0, 28)
        assert batch.breach_mm.shape == (0,)
        assert batch.min_wall_mm.shape == (0,)
        assert batch.mean_hu.shape == (0,)
        assert batch.min_hu.shape == (0,)

    def test_chunking_does_not_change_the_result(self, monkeypatch):
        import src.core.screw_grading as screw_grading

        mask = _cube_mask()
        grader = ScrewGrader(mask, _ct_like(mask))
        rng = np.random.default_rng(3)
        entries = rng.uniform(18.0, 42.0, size=(7, 3))
        targets = rng.uniform(18.0, 42.0, size=(7, 3))
        whole = grader.evaluate_batch(entries, targets, 6.0, 28)
        monkeypatch.setattr(screw_grading, "MAX_BATCH_SAMPLE_POINTS", 500)
        chunked = grader.evaluate_batch(entries, targets, 6.0, 28)
        np.testing.assert_allclose(chunked.breach_mm, whole.breach_mm)
        np.testing.assert_allclose(chunked.min_wall_mm, whole.min_wall_mm)
        np.testing.assert_allclose(chunked.mean_hu, whole.mean_hu)
        np.testing.assert_allclose(chunked.min_hu, whole.min_hu)

    def test_batch_matches_single_evaluation_on_an_anisotropic_grid(self):
        mask = _asymmetric_mask()
        grader = ScrewGrader(mask, _ct_like(mask))
        entries = np.array([[6.0, 26.0, 50.0], [12.0, 26.0, 44.0], [20.0, 40.0, 60.0]])
        targets = np.array([[22.0, 26.0, 50.0], [12.0, 26.0, 74.0], [20.0, 10.0, 60.0]])
        batch = grader.evaluate_batch(entries, targets, 4.0, 28)
        for i in range(3):
            single = grader.grade(entries[i], targets[i], 4.0, label=28)
            assert batch.breach_mm[i] == pytest.approx(single.breach_mm, abs=0.51)
            assert batch.min_wall_mm[i] == pytest.approx(single.min_wall_mm, abs=0.51)
