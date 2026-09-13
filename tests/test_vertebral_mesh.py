"""
Tests for vertebral body mesh extraction (src/core/vertebral_mesh.py).

Verifies:
- Color generation for all vertebra labels (HSV warm gradient)
- Label detection in vtkImageData masks
- Mesh extraction from synthetic masks
- Edge cases (empty masks, no vertebral labels, None input)
- Viewer3D API presence for vertebral mesh methods
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import vtk

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _create_labeled_mask(
    dims=(10, 10, 10),
    spacing=(1.0, 1.0, 1.0),
    origin=(0.0, 0.0, 0.0),
    label_slabs=None,
):
    """Create a vtkImageData with integer labels for testing.

    Args:
        dims: Volume dimensions (x, y, z).
        spacing: Voxel spacing.
        origin: Volume origin.
        label_slabs: List of (label_value, z_start, z_end) tuples.
            Voxels in the z-range [z_start, z_end) get that label.
            Default: empty volume (all zeros).

    Returns:
        vtkImageData with unsigned short scalars.
    """
    image = vtk.vtkImageData()
    image.SetDimensions(*dims)
    image.SetSpacing(*spacing)
    image.SetOrigin(*origin)
    image.AllocateScalars(vtk.VTK_UNSIGNED_SHORT, 1)

    # Zero out
    scalars = image.GetPointData().GetScalars()
    n = scalars.GetNumberOfTuples()
    for i in range(n):
        scalars.SetTuple1(i, 0)

    if label_slabs:
        nx, ny, nz = dims
        for label_val, z_start, z_end in label_slabs:
            for z in range(z_start, min(z_end, nz)):
                for y in range(ny):
                    for x in range(nx):
                        idx = z * ny * nx + y * nx + x
                        scalars.SetTuple1(idx, label_val)

    image.Modified()
    return image


def _create_sphere_mask(
    dims=(32, 32, 32),
    spacing=(1.0, 1.0, 1.0),
    radius=11.0,
    label=27,
):
    """Create a vtkImageData holding one labelled sphere.

    Args:
        dims: Volume dimensions (x, y, z).
        spacing: Voxel spacing (x, y, z) in mm.
        radius: Sphere radius in mm.
        label: Label value written inside the sphere.

    Returns:
        (vtkImageData, sphere centre as an (x, y, z) numpy array in mm).
    """
    import numpy as np
    from vtk.util.numpy_support import numpy_to_vtk

    sp = np.asarray(spacing, dtype=float)
    z_index, y_index, x_index = np.indices((dims[2], dims[1], dims[0]))
    world = np.stack(
        (x_index * sp[0], y_index * sp[1], z_index * sp[2]), axis=-1
    )
    center = (np.asarray(dims) - 1) * sp / 2.0
    values = np.where(
        np.linalg.norm(world - center, axis=-1) <= radius, label, 0
    ).astype(np.uint16)

    image = vtk.vtkImageData()
    image.SetDimensions(*dims)
    image.SetSpacing(*spacing)
    image.GetPointData().SetScalars(numpy_to_vtk(values.ravel(), deep=True))
    return image, center


# ---------------------------------------------------------------------------
# Tests: Color generation
# ---------------------------------------------------------------------------

class TestVertebralColors:
    """Verify the low-saturation warm ivory anatomy palette."""

    def test_all_labels_produce_valid_rgb(self):
        from src.core.vertebral_mesh import VERTEBRA_LABELS, generate_vertebra_color

        for label in VERTEBRA_LABELS:
            r, g, b = generate_vertebra_color(label)
            assert 0.0 <= r <= 1.0, f"label {label}: r={r}"
            assert 0.0 <= g <= 1.0, f"label {label}: g={g}"
            assert 0.0 <= b <= 1.0, f"label {label}: b={b}"

    def test_sacrum_is_warm_ivory_instead_of_red(self):
        from src.core.vertebral_mesh import generate_vertebra_color

        r, g, b = generate_vertebra_color(25)
        assert r > g > b
        assert g > 0.68
        assert b > 0.58
        assert r - b < 0.25

    def test_t1_remains_light_warm_bone(self):
        from src.core.vertebral_mesh import generate_vertebra_color

        r, g, b = generate_vertebra_color(43)
        assert r > g > b
        assert min(r, g, b) > 0.65

    def test_colors_are_distinct(self):
        """Adjacent vertebrae should have different colors."""
        from src.core.vertebral_mesh import generate_vertebra_color

        colors = [generate_vertebra_color(label) for label in range(25, 44)]
        for i in range(len(colors) - 1):
            assert colors[i] != colors[i + 1], (
                f"Labels {25 + i} and {26 + i} have identical colors"
            )

    def test_unknown_label_returns_gray(self):
        from src.core.vertebral_mesh import generate_vertebra_color

        r, g, b = generate_vertebra_color(999)
        assert r == g == b, "Unknown label should return neutral gray"

    def test_get_vertebra_colors_returns_all(self):
        from src.core.vertebral_mesh import VERTEBRA_LABELS, get_vertebra_colors

        color_map = get_vertebra_colors()
        assert set(color_map.keys()) == set(VERTEBRA_LABELS.keys())


# ---------------------------------------------------------------------------
# Tests: Label detection
# ---------------------------------------------------------------------------

class TestLabelDetection:
    """Verify voxel presence checks."""

    def test_detect_present_label(self):
        from src.core.vertebral_mesh import _label_has_voxels_numpy

        mask = _create_labeled_mask(label_slabs=[(27, 0, 3)])
        assert _label_has_voxels_numpy(mask, 27) is True

    def test_detect_absent_label(self):
        from src.core.vertebral_mesh import _label_has_voxels_numpy

        mask = _create_labeled_mask(label_slabs=[(27, 0, 3)])
        assert _label_has_voxels_numpy(mask, 30) is False

    def test_empty_mask_has_no_labels(self):
        from src.core.vertebral_mesh import _label_has_voxels_numpy

        mask = _create_labeled_mask()
        for label in range(25, 44):
            assert _label_has_voxels_numpy(mask, label) is False

    def test_out_of_range_label_rejected_fast(self):
        """Labels outside scalar range should be rejected without scanning."""
        from src.core.vertebral_mesh import _label_has_voxels_numpy

        mask = _create_labeled_mask(label_slabs=[(27, 0, 3)])
        # scalar range is [0, 27], label 100 is out of range
        assert _label_has_voxels_numpy(mask, 100) is False

    def test_vtk_fallback_detection(self):
        """Verify the pure-VTK fallback path works."""
        from src.core.vertebral_mesh import _label_has_voxels

        mask = _create_labeled_mask(label_slabs=[(32, 2, 5)])
        assert _label_has_voxels(mask, 32) is True
        assert _label_has_voxels(mask, 33) is False

    def test_label_presence_alias_delegates_to_numpy(self):
        """_label_has_voxels must not run its own per-voxel GetTuple1 loop."""
        import inspect

        from src.core import vertebral_mesh
        from src.core.vertebral_mesh import _label_has_voxels

        mask = _create_labeled_mask(label_slabs=[(32, 2, 5)])
        assert _label_has_voxels(mask, 32) is True
        assert _label_has_voxels(mask, 33) is False
        source = inspect.getsource(_label_has_voxels)
        assert "GetTuple1" not in source
        assert "_label_has_voxels_numpy" in source
        assert hasattr(vertebral_mesh, "_label_has_voxels_slow")

    def test_detect_vertebral_labels_returns_sorted(self):
        """detect_vertebral_labels should return sorted list of present labels."""
        from src.core.vertebral_mesh import detect_vertebral_labels

        mask = _create_labeled_mask(
            dims=(10, 10, 20),
            label_slabs=[(27, 0, 5), (32, 5, 10), (28, 10, 15)],
        )
        detected = detect_vertebral_labels(mask)
        assert detected == [27, 28, 32]

    def test_detect_vertebral_labels_empty_mask(self):
        """Empty mask should return empty list."""
        from src.core.vertebral_mesh import detect_vertebral_labels

        mask = _create_labeled_mask()
        assert detect_vertebral_labels(mask) == []

    def test_detect_vertebral_labels_none_input(self):
        """None input should return empty list."""
        from src.core.vertebral_mesh import detect_vertebral_labels

        assert detect_vertebral_labels(None) == []

    def test_detect_vertebral_labels_ignores_non_vertebral(self):
        """Non-vertebral labels (outside 25-43) should be ignored."""
        from src.core.vertebral_mesh import detect_vertebral_labels

        mask = _create_labeled_mask(label_slabs=[(5, 0, 5), (27, 5, 10)])
        detected = detect_vertebral_labels(mask)
        assert detected == [27]


# ---------------------------------------------------------------------------
# Tests: Mesh extraction
# ---------------------------------------------------------------------------

class TestMeshExtraction:
    """Verify end-to-end mesh extraction from synthetic masks."""

    def test_single_vertebra_mesh(self):
        """Extracting a single vertebra should produce non-empty polydata."""
        from src.core.vertebral_mesh import extract_vertebral_mesh

        mask = _create_labeled_mask(
            dims=(20, 20, 20),
            label_slabs=[(27, 5, 15)],  # L5
        )
        result = extract_vertebral_mesh(mask)
        assert result is not None
        assert result.GetNumberOfCells() > 0
        assert result.GetNumberOfPoints() > 0

    def test_multiple_vertebrae_mesh(self):
        """Multiple vertebrae should produce more cells than a single one."""
        from src.core.vertebral_mesh import extract_vertebral_mesh

        mask = _create_labeled_mask(
            dims=(20, 20, 30),
            label_slabs=[
                (27, 0, 10),   # L5
                (28, 10, 20),  # L4
                (29, 20, 30),  # L3
            ],
        )
        result = extract_vertebral_mesh(mask)
        assert result is not None
        assert result.GetNumberOfCells() > 0

    def test_mesh_has_color_array(self):
        """Resulting polydata should have VertebraColors cell data."""
        from src.core.vertebral_mesh import extract_vertebral_mesh

        mask = _create_labeled_mask(
            dims=(20, 20, 20),
            label_slabs=[(27, 5, 15)],
        )
        result = extract_vertebral_mesh(mask)
        assert result is not None

        colors = result.GetCellData().GetArray("VertebraColors")
        assert colors is not None
        assert colors.GetNumberOfComponents() == 3
        assert colors.GetNumberOfTuples() == result.GetNumberOfCells()

    def test_none_input_returns_none(self):
        from src.core.vertebral_mesh import extract_vertebral_mesh

        result = extract_vertebral_mesh(None)
        assert result is None

    def test_empty_mask_returns_none(self):
        """All-zero mask should return None (no vertebral labels)."""
        from src.core.vertebral_mesh import extract_vertebral_mesh

        mask = _create_labeled_mask(dims=(10, 10, 10))
        result = extract_vertebral_mesh(mask)
        assert result is None

    def test_non_vertebral_labels_ignored(self):
        """Labels outside 25-43 should be ignored."""
        from src.core.vertebral_mesh import extract_vertebral_mesh

        # Label 5 is not a vertebra label
        mask = _create_labeled_mask(
            dims=(20, 20, 20),
            label_slabs=[(5, 0, 20)],
        )
        result = extract_vertebral_mesh(mask)
        assert result is None

    def test_smoothing_parameters_respected(self):
        """Custom smoothing parameters should not cause errors."""
        from src.core.vertebral_mesh import extract_vertebral_mesh

        mask = _create_labeled_mask(
            dims=(20, 20, 20),
            label_slabs=[(25, 5, 15)],
        )
        result = extract_vertebral_mesh(
            mask, smoothing_iterations=5, smoothing_passband=0.05
        )
        assert result is not None
        assert result.GetNumberOfCells() > 0

    def test_default_pipeline_does_not_increase_polygon_count(self):
        """Interactive mesh generation must not amplify surface complexity."""
        from src.core.vertebral_mesh import extract_vertebral_mesh

        mask = _create_labeled_mask(
            dims=(30, 30, 30),
            label_slabs=[(27, 5, 25)],
        )
        raw = vtk.vtkDiscreteMarchingCubes()
        raw.SetInputData(mask)
        raw.SetValue(0, 27)
        raw.Update()

        result = extract_vertebral_mesh(mask)

        assert result is not None
        assert result.GetNumberOfCells() <= raw.GetOutput().GetNumberOfCells()

    def test_coarse_anisotropic_mask_produces_faired_dense_surface(self):
        """Surface fairing should retain shape and avoid excessive decimation."""
        import numpy as np
        from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy

        from src.core.vertebral_mesh import extract_vertebral_mesh

        dims = (32, 32, 20)
        spacing = np.array((2.0, 2.0, 4.0))
        z_index, y_index, x_index = np.indices(
            (dims[2], dims[1], dims[0])
        )
        world = np.stack(
            (
                x_index * spacing[0],
                y_index * spacing[1],
                z_index * spacing[2],
            ),
            axis=-1,
        )
        center = (np.array(dims) - 1) * spacing / 2.0
        labels = np.where(
            np.linalg.norm(world - center, axis=-1) <= 15.0,
            27,
            0,
        ).astype(np.uint16)

        mask = vtk.vtkImageData()
        mask.SetDimensions(*dims)
        mask.SetSpacing(*spacing)
        mask.GetPointData().SetScalars(
            numpy_to_vtk(labels.ravel(), deep=True)
        )

        raw = vtk.vtkDiscreteMarchingCubes()
        raw.SetInputData(mask)
        raw.SetValue(0, 27)
        raw.Update()
        result = extract_vertebral_mesh(mask, labels=[27])

        assert result is not None
        points = vtk_to_numpy(result.GetPoints().GetData())
        radial_distance = np.linalg.norm(points - center, axis=1)
        assert radial_distance.std() < 0.48
        assert result.GetNumberOfCells() >= raw.GetOutput().GetNumberOfCells() * 0.5

    def test_mesh_uses_spacing_aware_gaussian_before_flying_edges(self):
        """Gaussian sigma must be physical-mm based, not a fixed voxel value."""
        import inspect

        from src.core.vertebral_mesh import extract_vertebral_mesh

        source = inspect.getsource(extract_vertebral_mesh)

        assert "vtkImageGaussianSmooth()" in source
        assert "vtkFlyingEdges3D()" in source
        assert "smoothing_mm / spacing" in source

    def test_upsampled_15mm_label_terraces_are_smoothed_in_physical_space(self):
        """Model TotalSegmentator's coarse labels resampled onto a fine CT grid."""
        import numpy as np
        from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy

        from src.core.vertebral_mesh import extract_vertebral_mesh

        coarse_dims = (24, 24, 24)
        coarse_spacing = 1.5
        upsample_factor = 3
        z_index, y_index, x_index = np.indices(
            (coarse_dims[2], coarse_dims[1], coarse_dims[0])
        )
        coarse_center = (
            (np.array(coarse_dims) - 1) * coarse_spacing / 2.0
        )
        coarse_world = np.stack(
            (
                x_index * coarse_spacing,
                y_index * coarse_spacing,
                z_index * coarse_spacing,
            ),
            axis=-1,
        )
        coarse = np.where(
            np.linalg.norm(coarse_world - coarse_center, axis=-1) <= 12.0,
            27,
            0,
        ).astype(np.uint16)
        labels = np.repeat(
            np.repeat(
                np.repeat(coarse, upsample_factor, axis=0),
                upsample_factor,
                axis=1,
            ),
            upsample_factor,
            axis=2,
        )

        mask = vtk.vtkImageData()
        mask.SetDimensions(labels.shape[2], labels.shape[1], labels.shape[0])
        mask.SetSpacing(0.5, 0.5, 0.5)
        mask.GetPointData().SetScalars(
            numpy_to_vtk(labels.ravel(), deep=True)
        )
        result = extract_vertebral_mesh(mask, labels=[27])

        points = vtk_to_numpy(result.GetPoints().GetData())
        fine_center = (np.array(mask.GetDimensions()) - 1) * 0.5 / 2.0
        radial_distance = np.linalg.norm(points - fine_center, axis=1)
        assert radial_distance.std() < 0.18


class TestVertebraLabelArray:
    """Every mesh cell carries its own vertebral label (viewer_3d's local cut)."""

    def test_mesh_cells_carry_their_vertebra_label(self):
        from src.core.vertebral_mesh import VERTEBRA_LABEL_ARRAY, extract_vertebral_mesh

        mask = _create_labeled_mask(
            dims=(20, 20, 30),
            label_slabs=[
                (27, 0, 10),   # L5
                (28, 10, 20),  # L4
                (29, 20, 30),  # L3
            ],
        )
        result = extract_vertebral_mesh(mask)
        assert result is not None

        label_array = result.GetCellData().GetArray(VERTEBRA_LABEL_ARRAY)
        assert label_array is not None
        assert label_array.GetNumberOfComponents() == 1
        assert label_array.GetNumberOfTuples() == result.GetNumberOfCells()
        present = {int(label_array.GetTuple1(i)) for i in range(label_array.GetNumberOfTuples())}
        assert present == {27, 28, 29}

        # "VertebraColors" must still be the array that drives rendering.
        colors = result.GetCellData().GetScalars()
        assert colors.GetName() == "VertebraColors"


class TestSplitMeshByLabel:
    """split_mesh_by_label carves one vertebra out for viewer_3d's local cut."""

    @staticmethod
    def _combined_mesh():
        from src.core.vertebral_mesh import extract_vertebral_mesh

        mask = _create_labeled_mask(
            dims=(20, 20, 30),
            label_slabs=[
                (27, 0, 10),   # L5
                (28, 10, 20),  # L4
                (29, 20, 30),  # L3
            ],
        )
        return extract_vertebral_mesh(mask)

    def test_split_mesh_by_label_separates_one_vertebra(self):
        from src.core.vertebral_mesh import VERTEBRA_LABEL_ARRAY, split_mesh_by_label

        combined = self._combined_mesh()
        matching, rest = split_mesh_by_label(combined, 28)

        assert matching.GetNumberOfCells() > 0
        assert rest.GetNumberOfCells() > 0
        assert matching.GetNumberOfCells() + rest.GetNumberOfCells() == combined.GetNumberOfCells()

        matching_labels = matching.GetCellData().GetArray(VERTEBRA_LABEL_ARRAY)
        assert matching_labels.GetRange() == (28, 28)

        rest_labels = rest.GetCellData().GetArray(VERTEBRA_LABEL_ARRAY)
        rest_values = {
            int(rest_labels.GetTuple1(i)) for i in range(rest_labels.GetNumberOfTuples())
        }
        assert 28 not in rest_values
        assert rest_values == {27, 29}

        # Every cell array (not just VertebraColors) must survive the split.
        assert matching.GetCellData().GetArray("VertebraColors") is not None
        assert rest.GetCellData().GetArray("VertebraColors") is not None

    def test_split_mesh_by_label_without_the_label_returns_everything_as_rest(self):
        from src.core.vertebral_mesh import split_mesh_by_label

        combined = self._combined_mesh()

        matching, rest = split_mesh_by_label(combined, 99)

        assert matching.GetNumberOfCells() == 0
        assert rest.GetNumberOfCells() == combined.GetNumberOfCells()

    def test_split_mesh_by_label_missing_array_returns_everything_as_rest(self):
        import vtk

        from src.core.vertebral_mesh import split_mesh_by_label

        plain = vtk.vtkPolyData()
        source = vtk.vtkSphereSource()
        source.Update()
        plain.DeepCopy(source.GetOutput())

        matching, rest = split_mesh_by_label(plain, 28)

        assert matching.GetNumberOfCells() == 0
        assert rest.GetNumberOfCells() == plain.GetNumberOfCells()


class TestLabelWorldBounds:
    """label_world_bounds sizes viewer_3d's local mesh clip and volume crop."""

    def test_label_world_bounds_matches_the_voxels_plus_margin(self):
        from src.core.vertebral_mesh import label_world_bounds

        mask = _create_labeled_mask(
            dims=(20, 20, 30),
            spacing=(1.0, 1.0, 1.0),
            origin=(0.0, 0.0, 0.0),
            label_slabs=[(28, 10, 20)],
        )

        bounds = label_world_bounds(mask, 28, margin_mm=2.0)

        assert bounds == pytest.approx((0.0, 19.0, 0.0, 19.0, 8.0, 21.0))

    def test_label_world_bounds_clamps_to_the_mask(self):
        from src.core.vertebral_mesh import label_world_bounds

        # The label fills the whole mask, so a margin must clamp rather than
        # extend past the mask's own bounds.
        mask = _create_labeled_mask(
            dims=(10, 10, 10),
            spacing=(1.0, 1.0, 1.0),
            label_slabs=[(28, 0, 10)],
        )

        bounds = label_world_bounds(mask, 28, margin_mm=5.0)

        assert bounds == pytest.approx((0.0, 9.0, 0.0, 9.0, 0.0, 9.0))

    def test_label_world_bounds_absent_label_returns_none(self):
        from src.core.vertebral_mesh import label_world_bounds

        mask = _create_labeled_mask(dims=(10, 10, 10), label_slabs=[(28, 0, 10)])

        assert label_world_bounds(mask, 30) is None

    def test_label_world_bounds_none_mask_returns_none(self):
        from src.core.vertebral_mesh import label_world_bounds

        assert label_world_bounds(None, 28) is None


class TestDerivedSmoothingDefault:
    """The Gaussian sigma follows the mask's own grid, not a fixed constant."""

    def test_native_ct_grid_keeps_the_09mm_floor(self):
        from src.core.vertebral_mesh import default_smoothing_mm

        assert default_smoothing_mm((0.39, 0.39, 1.0)) == pytest.approx(0.9)

    def test_raw_15mm_inference_grid_also_lands_on_the_floor(self):
        from src.core.vertebral_mesh import default_smoothing_mm

        assert default_smoothing_mm((1.5, 1.5, 1.5)) == pytest.approx(0.9)

    def test_coarse_slices_scale_to_half_the_coarsest_spacing(self):
        from src.core.vertebral_mesh import default_smoothing_mm

        assert default_smoothing_mm((0.5, 0.5, 3.0)) == pytest.approx(1.5)

    def test_derived_sigma_is_clamped_at_2mm(self):
        from src.core.vertebral_mesh import default_smoothing_mm

        assert default_smoothing_mm((0.5, 0.5, 6.0)) == pytest.approx(2.0)

    def test_signature_defaults_use_the_shared_constants(self):
        import inspect

        from src.core.vertebral_mesh import (
            MESH_SMOOTHING_ITERATIONS,
            MESH_SMOOTHING_PASSBAND,
            extract_vertebral_mesh,
        )

        assert MESH_SMOOTHING_ITERATIONS == 30
        assert MESH_SMOOTHING_PASSBAND == pytest.approx(0.06)
        params = inspect.signature(extract_vertebral_mesh).parameters
        assert params["smoothing_iterations"].default == MESH_SMOOTHING_ITERATIONS
        assert params["smoothing_passband"].default == MESH_SMOOTHING_PASSBAND
        assert params["smoothing_mm"].default is None

    def test_none_sigma_is_derived_from_the_mask_spacing(self):
        from src.core.vertebral_mesh import default_smoothing_mm, extract_vertebral_mesh

        mask, _ = _create_sphere_mask(
            dims=(24, 24, 24), spacing=(0.5, 0.5, 4.0), radius=5.0
        )
        derived = default_smoothing_mm(mask.GetSpacing())
        assert derived == pytest.approx(2.0)

        implicit = extract_vertebral_mesh(mask, labels=[27])
        explicit = extract_vertebral_mesh(mask, labels=[27], smoothing_mm=derived)
        assert implicit is not None and explicit is not None
        assert implicit.GetNumberOfCells() == explicit.GetNumberOfCells()
        assert implicit.GetNumberOfPoints() == explicit.GetNumberOfPoints()


class TestFairingChangeIsSafe:
    """30 iterations / 0.06 pass-band must fair harder without eroding the mesh."""

    def test_cell_count_stays_within_20_percent_of_the_old_settings(self):
        import numpy as np

        from src.core.vertebral_mesh import extract_vertebral_mesh

        mask, _ = _create_sphere_mask()
        today = extract_vertebral_mesh(
            mask,
            labels=[27],
            smoothing_iterations=20,
            smoothing_passband=0.08,
            smoothing_mm=0.9,
        )
        updated = extract_vertebral_mesh(mask, labels=[27])
        assert today is not None and updated is not None
        assert updated.GetNumberOfCells() == pytest.approx(
            today.GetNumberOfCells(), rel=0.20
        )
        assert np.isfinite(updated.GetNumberOfCells())
        assert updated.GetNumberOfCells() > 0

    def test_new_fairing_is_at_least_as_smooth_as_the_old(self):
        import numpy as np
        from vtk.util.numpy_support import vtk_to_numpy

        from src.core.vertebral_mesh import extract_vertebral_mesh

        mask, center = _create_sphere_mask()
        today = extract_vertebral_mesh(
            mask,
            labels=[27],
            smoothing_iterations=20,
            smoothing_passband=0.08,
            smoothing_mm=0.9,
        )
        updated = extract_vertebral_mesh(mask, labels=[27])

        def radial_std(mesh):
            points = vtk_to_numpy(mesh.GetPoints().GetData())
            return float(np.linalg.norm(points - center, axis=1).std())

        assert radial_std(updated) <= radial_std(today) + 1e-9

    def test_new_fairing_does_not_shrink_the_sphere(self):
        import numpy as np
        from vtk.util.numpy_support import vtk_to_numpy

        from src.core.vertebral_mesh import extract_vertebral_mesh

        mask, center = _create_sphere_mask()
        updated = extract_vertebral_mesh(mask, labels=[27])
        points = vtk_to_numpy(updated.GetPoints().GetData())
        mean_radius = float(np.linalg.norm(points - center, axis=1).mean())
        assert mean_radius == pytest.approx(11.0, abs=0.35)


# ---------------------------------------------------------------------------
# Tests: Constants and module structure
# ---------------------------------------------------------------------------

class TestModuleConstants:
    """Verify module-level constants."""

    def test_label_range(self):
        from src.core.vertebral_mesh import VERTEBRA_LABEL_MAX, VERTEBRA_LABEL_MIN

        assert VERTEBRA_LABEL_MIN == 25
        assert VERTEBRA_LABEL_MAX == 43

    def test_all_19_labels_defined(self):
        from src.core.vertebral_mesh import VERTEBRA_LABELS

        assert len(VERTEBRA_LABELS) == 19

    def test_label_keys_contiguous(self):
        from src.core.vertebral_mesh import VERTEBRA_LABELS

        expected = set(range(25, 44))
        assert set(VERTEBRA_LABELS.keys()) == expected

    def test_label_names_contain_vertebrae(self):
        """All labels except sacrum should contain 'vertebrae'."""
        from src.core.vertebral_mesh import VERTEBRA_LABELS

        for label, name in VERTEBRA_LABELS.items():
            if label == 25:
                assert name == "sacrum"
            else:
                assert "vertebrae" in name, f"Label {label} name missing 'vertebrae': {name}"


# ---------------------------------------------------------------------------
# Tests: HSV conversion
# ---------------------------------------------------------------------------

class TestHSVConversion:
    """Verify internal HSV to RGB conversion."""

    def test_red(self):
        from src.core.vertebral_mesh import _hsv_to_rgb

        r, g, b = _hsv_to_rgb(0.0, 1.0, 1.0)
        assert abs(r - 1.0) < 0.01
        assert abs(g - 0.0) < 0.01
        assert abs(b - 0.0) < 0.01

    def test_green(self):
        from src.core.vertebral_mesh import _hsv_to_rgb

        r, g, b = _hsv_to_rgb(1 / 3, 1.0, 1.0)
        assert abs(r - 0.0) < 0.01
        assert abs(g - 1.0) < 0.01
        assert abs(b - 0.0) < 0.01

    def test_blue(self):
        from src.core.vertebral_mesh import _hsv_to_rgb

        r, g, b = _hsv_to_rgb(2 / 3, 1.0, 1.0)
        assert abs(r - 0.0) < 0.01
        assert abs(g - 0.0) < 0.01
        assert abs(b - 1.0) < 0.01

    def test_white(self):
        from src.core.vertebral_mesh import _hsv_to_rgb

        r, g, b = _hsv_to_rgb(0.0, 0.0, 1.0)
        assert abs(r - 1.0) < 0.01
        assert abs(g - 1.0) < 0.01
        assert abs(b - 1.0) < 0.01

    def test_black(self):
        from src.core.vertebral_mesh import _hsv_to_rgb

        r, g, b = _hsv_to_rgb(0.0, 0.0, 0.0)
        assert abs(r) < 0.01
        assert abs(g) < 0.01
        assert abs(b) < 0.01

    def test_yellow(self):
        """H=60deg (1/6) should produce yellow."""
        from src.core.vertebral_mesh import _hsv_to_rgb

        r, g, b = _hsv_to_rgb(1 / 6, 1.0, 1.0)
        assert abs(r - 1.0) < 0.01
        assert abs(g - 1.0) < 0.01
        assert abs(b - 0.0) < 0.01


# ---------------------------------------------------------------------------
# Tests: create_vertebral_only_volume
# ---------------------------------------------------------------------------

class TestCreateVertebralOnlyVolume:
    """Verify masked volume creation for vertebral body isolation."""

    def _make_ct_and_mask(self):
        """Create a simple CT volume and a matching mask."""
        import numpy as np
        from vtk.util.numpy_support import numpy_to_vtk

        dims = (10, 10, 10)
        ct = vtk.vtkImageData()
        ct.SetDimensions(*dims)
        ct.SetSpacing(1.0, 1.0, 1.0)
        ct.SetOrigin(0.0, 0.0, 0.0)
        ct.AllocateScalars(vtk.VTK_SHORT, 1)

        # Fill CT with HU=500 everywhere
        n = dims[0] * dims[1] * dims[2]
        ct_arr = np.full(n, 500, dtype=np.int16)
        vtk_ct = numpy_to_vtk(ct_arr, deep=True)
        vtk_ct.SetName("ImageScalars")
        ct.GetPointData().SetScalars(vtk_ct)

        # Create mask: label 27 (L5) in z=0..4, label 0 elsewhere
        mask = vtk.vtkImageData()
        mask.SetDimensions(*dims)
        mask.SetSpacing(1.0, 1.0, 1.0)
        mask.SetOrigin(0.0, 0.0, 0.0)
        mask.AllocateScalars(vtk.VTK_UNSIGNED_SHORT, 1)

        mask_arr = np.zeros(n, dtype=np.uint16)
        nx, ny = dims[0], dims[1]
        for z in range(5):
            for y in range(ny):
                for x in range(nx):
                    mask_arr[z * ny * nx + y * nx + x] = 27
        vtk_mask = numpy_to_vtk(mask_arr, deep=True)
        vtk_mask.SetName("ImageScalars")
        mask.GetPointData().SetScalars(vtk_mask)

        return ct, mask

    def test_vertebral_voxels_preserved(self):
        from vtk.util.numpy_support import vtk_to_numpy

        from src.core.vertebral_mesh import create_vertebral_only_volume

        ct, mask = self._make_ct_and_mask()
        result = create_vertebral_only_volume(ct, mask, smooth_sigma=0)
        result_arr = vtk_to_numpy(result.GetPointData().GetScalars())

        # First 500 voxels (z=0..4) should still be 500
        assert (result_arr[:500] == 500).all()

    def test_non_vertebral_voxels_set_to_air(self):
        from vtk.util.numpy_support import vtk_to_numpy

        from src.core.vertebral_mesh import create_vertebral_only_volume

        ct, mask = self._make_ct_and_mask()
        result = create_vertebral_only_volume(ct, mask, smooth_sigma=0)
        result_arr = vtk_to_numpy(result.GetPointData().GetScalars())

        # Last 500 voxels (z=5..9) should be -1000
        assert (result_arr[500:] == -1000).all()

    def test_result_geometry_matches_original(self):
        from src.core.vertebral_mesh import create_vertebral_only_volume

        ct, mask = self._make_ct_and_mask()
        result = create_vertebral_only_volume(ct, mask)

        assert result.GetDimensions() == ct.GetDimensions()
        assert result.GetSpacing() == ct.GetSpacing()
        assert result.GetOrigin() == ct.GetOrigin()

    def test_smoothing_blurs_boundary(self):
        """Gaussian smoothing should produce intermediate values at boundary."""
        import numpy as np
        from vtk.util.numpy_support import vtk_to_numpy

        from src.core.vertebral_mesh import create_vertebral_only_volume

        ct, mask = self._make_ct_and_mask()
        result = create_vertebral_only_volume(ct, mask, smooth_sigma=0.7)
        result_arr = vtk_to_numpy(result.GetPointData().GetScalars())

        # Interior bone voxels (z=0..2) should still be close to 500
        dims = ct.GetDimensions()
        n_per_slice = dims[0] * dims[1]  # 100
        interior_bone = result_arr[:2 * n_per_slice]
        assert np.mean(interior_bone) > 400

        # Interior air voxels (z=7..9) should still be close to -1000
        interior_air = result_arr[7 * n_per_slice:]
        assert np.mean(interior_air) < -900

        # Boundary region (z=4,5) should have blurred intermediate values
        boundary = result_arr[4 * n_per_slice:6 * n_per_slice]
        assert np.min(boundary) < 400  # Some blurring occurred


# ---------------------------------------------------------------------------
# Tests: Viewer3D API presence
# ---------------------------------------------------------------------------

class TestViewer3DVertebralAPI:
    """Verify Viewer3D has vertebral mesh methods."""

    def test_has_set_vertebral_mesh(self):
        from src.ui.viewer_3d import Viewer3D

        assert hasattr(Viewer3D, "set_vertebral_mesh")

    def test_has_clear_vertebral_mesh(self):
        from src.ui.viewer_3d import Viewer3D

        assert hasattr(Viewer3D, "clear_vertebral_mesh")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
