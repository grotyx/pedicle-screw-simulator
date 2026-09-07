"""
Vertebral body 3D mesh extraction from TotalSegmentator segmentation masks.

Extracts individual vertebral labels (T1-S1 + sacrum, labels 25-43) from a
multilabel segmentation mask and produces a combined colored vtkPolyData mesh.
Each vertebra receives a distinct low-saturation warm ivory color.

Pipeline per label:
  1. Crop each label to a spacing-aware padded region
  2. Smooth a float binary field using a physical 0.9 mm Gaussian
  3. Extract a sub-voxel surface with vtkFlyingEdges3D
  4. Light surface fairing and moderate topology-preserving decimation
  5. Point-normal recomputation and Phong rendering
  6. Per-cell color assignment via vtkUnsignedCharArray

All extracted surfaces are combined into one vtkAppendPolyData output.
"""

import logging
import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import vtk

logger = logging.getLogger(__name__)

# TotalSegmentator vertebral body labels (T1 through sacrum)
VERTEBRA_LABEL_MIN = 25
VERTEBRA_LABEL_MAX = 43

# Ordered from caudal (sacrum) to cranial (T1)
VERTEBRA_LABELS: Dict[int, str] = {
    25: "sacrum",
    26: "vertebrae_S1",
    27: "vertebrae_L5",
    28: "vertebrae_L4",
    29: "vertebrae_L3",
    30: "vertebrae_L2",
    31: "vertebrae_L1",
    32: "vertebrae_T12",
    33: "vertebrae_T11",
    34: "vertebrae_T10",
    35: "vertebrae_T9",
    36: "vertebrae_T8",
    37: "vertebrae_T7",
    38: "vertebrae_T6",
    39: "vertebrae_T5",
    40: "vertebrae_T4",
    41: "vertebrae_T3",
    42: "vertebrae_T2",
    43: "vertebrae_T1",
}


def _hsv_to_rgb(h: float, s: float, v: float) -> Tuple[float, float, float]:
    """Convert HSV (all 0-1 range) to RGB (0-1 range).

    Args:
        h: Hue in [0, 1].
        s: Saturation in [0, 1].
        v: Value in [0, 1].

    Returns:
        Tuple of (r, g, b) each in [0, 1].
    """
    if s == 0.0:
        return (v, v, v)

    h6 = h * 6.0
    i = int(h6)
    f = h6 - i
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))

    if i == 0:
        return (v, t, p)
    elif i == 1:
        return (q, v, p)
    elif i == 2:
        return (p, v, t)
    elif i == 3:
        return (p, q, v)
    elif i == 4:
        return (t, p, v)
    else:
        return (v, p, q)


def generate_vertebra_color(label: int) -> Tuple[float, float, float]:
    """Generate a warm ivory/peach color for a vertebra label.

    Args:
        label: TotalSegmentator label ID (25-43).

    Returns:
        RGB tuple (0-1 range).
    """
    if label not in VERTEBRA_LABELS:
        return (0.7, 0.7, 0.7)

    # Normalize position: 0.0 (label 25, sacrum) -> 1.0 (label 43, T1)
    total_range = VERTEBRA_LABEL_MAX - VERTEBRA_LABEL_MIN
    if total_range == 0:
        t = 0.0
    else:
        t = (label - VERTEBRA_LABEL_MIN) / total_range

    caudal = (0.88, 0.76, 0.64)
    cranial = (0.97, 0.89, 0.78)
    return tuple(
        caudal[channel] + t * (cranial[channel] - caudal[channel])
        for channel in range(3)
    )


def get_vertebra_colors() -> Dict[int, Tuple[float, float, float]]:
    """Return the full color map for all vertebral labels.

    Returns:
        Dict mapping label ID to RGB tuple (0-1 range).
    """
    return {label: generate_vertebra_color(label) for label in VERTEBRA_LABELS}


def _label_has_voxels(mask_image: vtk.vtkImageData, label: int) -> bool:
    """Check whether a specific label value exists in the mask.

    Scans the scalar array for at least one voxel matching `label`.
    Uses VTK scalar range first as a fast rejection test.

    Args:
        mask_image: Multilabel segmentation mask.
        label: Label value to check.

    Returns:
        True if at least one voxel has this label value.
    """
    scalar_range = mask_image.GetScalarRange()
    if label < scalar_range[0] or label > scalar_range[1]:
        return False

    scalars = mask_image.GetPointData().GetScalars()
    if scalars is None:
        return False

    n_tuples = scalars.GetNumberOfTuples()
    for i in range(n_tuples):
        if int(scalars.GetTuple1(i)) == label:
            return True
    return False


def _label_has_voxels_numpy(mask_image: vtk.vtkImageData, label: int) -> bool:
    """Fast numpy-based check for voxel presence.

    Falls back to VTK iteration if numpy import fails.

    Args:
        mask_image: Multilabel segmentation mask.
        label: Label value to check.

    Returns:
        True if at least one voxel has this label value.
    """
    scalar_range = mask_image.GetScalarRange()
    if label < scalar_range[0] or label > scalar_range[1]:
        return False

    try:
        from vtk.util.numpy_support import vtk_to_numpy
        scalars = mask_image.GetPointData().GetScalars()
        if scalars is None:
            return False
        arr = vtk_to_numpy(scalars)
        return bool((arr == label).any())
    except ImportError:
        return _label_has_voxels(mask_image, label)


def create_vertebral_only_volume(
    original_vtk: vtk.vtkImageData,
    mask_vtk: vtk.vtkImageData,
    smooth_sigma: float = 0.7,
) -> vtk.vtkImageData:
    """Create a CT volume showing only vertebral body regions.

    Copies original CT values for voxels where the mask label is in the
    vertebral range (25-43). All other voxels are set to -1000 (air HU).
    Gaussian smoothing is applied to eliminate staircase artifacts at the
    bone-air boundary.

    Args:
        original_vtk: Original CT vtkImageData.
        mask_vtk: Multilabel segmentation mask vtkImageData.
        smooth_sigma: Gaussian smoothing sigma in voxels (0 to disable).

    Returns:
        New vtkImageData with non-vertebral voxels set to air and smoothed.
    """
    import numpy as np
    from scipy.ndimage import gaussian_filter
    from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy

    ct_scalars = original_vtk.GetPointData().GetScalars()
    mask_scalars = mask_vtk.GetPointData().GetScalars()
    ct_arr = vtk_to_numpy(ct_scalars).copy().astype(np.float32)
    mask_arr = vtk_to_numpy(mask_scalars)

    # Build boolean mask: True where label is a vertebral body (25-43)
    vertebral_mask = (mask_arr >= VERTEBRA_LABEL_MIN) & (mask_arr <= VERTEBRA_LABEL_MAX)
    ct_arr[~vertebral_mask] = -1000

    # Gaussian smoothing to eliminate staircase artifacts at boundaries
    if smooth_sigma > 0:
        dims = original_vtk.GetDimensions()  # (x, y, z)
        vol = ct_arr.reshape(dims[2], dims[1], dims[0])  # (z, y, x)
        vol = gaussian_filter(vol, sigma=smooth_sigma)
        ct_arr = vol.ravel().astype(np.int16)
    else:
        ct_arr = ct_arr.astype(np.int16)

    result = vtk.vtkImageData()
    result.DeepCopy(original_vtk)
    vtk_arr = numpy_to_vtk(ct_arr, deep=True)
    vtk_arr.SetName("ImageScalars")
    result.GetPointData().SetScalars(vtk_arr)

    logger.info(
        "create_vertebral_only_volume: %d / %d voxels retained, sigma=%.1f",
        int(vertebral_mask.sum()),
        len(ct_arr),
        smooth_sigma,
    )
    return result


def detect_vertebral_labels(mask_image: vtk.vtkImageData) -> List[int]:
    """Detect which vertebral labels (25-43) are present in a segmentation mask.

    Args:
        mask_image: Multilabel segmentation mask (vtkImageData).

    Returns:
        Sorted list of label IDs that have at least one voxel in the mask.
    """
    if mask_image is None:
        return []
    return sorted(
        label
        for label in VERTEBRA_LABELS
        if _label_has_voxels_numpy(mask_image, label)
    )


def _find_label_vois(
    mask_image: vtk.vtkImageData,
    labels: List[int],
    padding_xyz: Tuple[int, int, int],
) -> Dict[int, Tuple[int, int, int, int, int, int]]:
    """Locate requested labels once and return padded VTK extents."""
    from scipy.ndimage import find_objects
    from vtk.util.numpy_support import vtk_to_numpy

    scalars = mask_image.GetPointData().GetScalars()
    if scalars is None:
        return {}
    dimensions = mask_image.GetDimensions()
    volume = vtk_to_numpy(scalars).reshape(
        dimensions[2], dimensions[1], dimensions[0]
    )
    objects = find_objects(volume, max_label=max(labels))
    extent = mask_image.GetExtent()
    padding_x, padding_y, padding_z = padding_xyz
    vois = {}
    for label in labels:
        slices = objects[label - 1] if label - 1 < len(objects) else None
        if slices is None:
            continue
        z_slice, y_slice, x_slice = slices
        vois[label] = (
            max(extent[0], extent[0] + x_slice.start - padding_x),
            min(extent[1], extent[0] + x_slice.stop - 1 + padding_x),
            max(extent[2], extent[2] + y_slice.start - padding_y),
            min(extent[3], extent[2] + y_slice.stop - 1 + padding_y),
            max(extent[4], extent[4] + z_slice.start - padding_z),
            min(extent[5], extent[4] + z_slice.stop - 1 + padding_z),
        )
    return vois


def extract_vertebral_mesh(
    mask_image: vtk.vtkImageData,
    smoothing_iterations: int = 20,
    smoothing_passband: float = 0.08,
    target_reduction: float = 0.50,
    labels: Optional[List[int]] = None,
    smoothing_mm: float = 0.9,
) -> Optional[vtk.vtkPolyData]:
    """Extract colored vertebral body mesh from a multilabel segmentation mask.

    TotalSegmentator labels may retain their 1.5 mm inference grid after being
    resampled onto a finer CT. Each label is therefore smoothed in physical
    millimetres before surface extraction instead of using a fixed voxel sigma.

    All surfaces are combined via vtkAppendPolyData.

    Args:
        mask_image: vtkImageData with integer labels from TotalSegmentator.
        smoothing_iterations: Number of Windowed Sinc iterations.
        smoothing_passband: Windowed Sinc passband parameter.
        target_reduction: Requested fraction of polygons to remove.
        labels: Optional pre-detected vertebral labels, avoiding another scan.
        smoothing_mm: Gaussian standard deviation in physical millimetres.

    Returns:
        Combined colored vtkPolyData, or None if no vertebral labels found.
    """
    if mask_image is None:
        logger.warning("extract_vertebral_mesh: mask_image is None")
        return None

    if labels is None:
        requested_labels = detect_vertebral_labels(mask_image)
    else:
        requested_labels = sorted(
            {int(label) for label in labels if int(label) in VERTEBRA_LABELS}
        )

    if not requested_labels:
        logger.info("extract_vertebral_mesh: no vertebral labels found in mask")
        return None

    reduction = min(max(float(target_reduction), 0.0), 0.95)
    iterations = max(int(smoothing_iterations), 0)
    passband = min(max(float(smoothing_passband), 0.001), 1.0)
    smoothing_mm = min(max(float(smoothing_mm), 0.0), 2.0)
    spacing = np.asarray(mask_image.GetSpacing(), dtype=float)
    sigma_voxels = tuple(
        float(value) for value in smoothing_mm / spacing
    )
    padding_xyz = tuple(
        int(math.ceil(sigma * 2.5)) + 2
        for sigma in sigma_voxels
    )
    label_vois = _find_label_vois(
        mask_image, requested_labels, padding_xyz
    )

    appender = vtk.vtkAppendPolyData()
    labels_found: List[int] = []

    for label in requested_labels:
        voi = label_vois.get(label)
        if voi is None:
            continue
        logger.debug(
            "extract_vertebral_mesh: processing label %d (%s)",
            label,
            VERTEBRA_LABELS[label],
        )

        crop = vtk.vtkExtractVOI()
        crop.SetInputData(mask_image)
        crop.SetVOI(*voi)

        if hasattr(vtk, "vtkImageBinaryThreshold"):
            threshold = vtk.vtkImageBinaryThreshold()
            threshold.SetInputConnection(crop.GetOutputPort())
            threshold.SetLowerThreshold(float(label))
            threshold.SetUpperThreshold(float(label))
            threshold.SetInValue(1.0)
            threshold.SetOutValue(0.0)
            threshold.SetReplaceIn(True)
            threshold.SetReplaceOut(True)
            threshold.SetOutputScalarTypeToFloat()
        else:
            threshold = vtk.vtkImageThreshold()
            threshold.SetInputConnection(crop.GetOutputPort())
            threshold.ThresholdBetween(label, label)
            threshold.SetInValue(1.0)
            threshold.SetOutValue(0.0)
            threshold.SetOutputScalarTypeToFloat()

        gaussian = vtk.vtkImageGaussianSmooth()
        gaussian.SetInputConnection(threshold.GetOutputPort())
        gaussian.SetStandardDeviations(*sigma_voxels)
        gaussian.SetRadiusFactors(2.5, 2.5, 2.5)
        gaussian.SetDimensionality(3)

        extractor = vtk.vtkFlyingEdges3D()
        extractor.SetInputConnection(gaussian.GetOutputPort())
        extractor.SetValue(0, 0.5)
        extractor.ComputeNormalsOff()
        extractor.ComputeGradientsOff()
        extractor.Update()

        surface = extractor.GetOutput()
        n_cells = surface.GetNumberOfCells()
        if n_cells == 0:
            logger.debug(
                "extract_vertebral_mesh: label %d produced 0 cells, skipping",
                label,
            )
            continue

        smoother = vtk.vtkWindowedSincPolyDataFilter()
        smoother.SetInputData(surface)
        smoother.SetNumberOfIterations(iterations)
        smoother.SetPassBand(passband)
        smoother.BoundarySmoothingOff()
        smoother.FeatureEdgeSmoothingOff()
        smoother.NonManifoldSmoothingOn()
        smoother.NormalizeCoordinatesOn()
        smoother.Update()

        decimator = vtk.vtkDecimatePro()
        decimator.SetInputConnection(smoother.GetOutputPort())
        decimator.SetTargetReduction(reduction)
        decimator.PreserveTopologyOn()
        decimator.Update()

        # Recompute normals after smoothing for proper lighting
        normals = vtk.vtkPolyDataNormals()
        normals.SetInputConnection(decimator.GetOutputPort())
        normals.ComputePointNormalsOn()
        normals.ComputeCellNormalsOff()
        normals.SplittingOff()
        normals.Update()

        smoothed_with_normals = normals.GetOutput()

        # Step 3: Assign per-cell color
        r, g, b = generate_vertebra_color(label)
        r_byte = int(r * 255)
        g_byte = int(g * 255)
        b_byte = int(b * 255)

        colors = vtk.vtkUnsignedCharArray()
        colors.SetNumberOfComponents(3)
        colors.SetName("VertebraColors")
        n_out_cells = smoothed_with_normals.GetNumberOfCells()
        colors.SetNumberOfTuples(n_out_cells)
        colors.FillComponent(0, r_byte)
        colors.FillComponent(1, g_byte)
        colors.FillComponent(2, b_byte)

        smoothed_with_normals.GetCellData().SetScalars(colors)
        appender.AddInputData(smoothed_with_normals)
        labels_found.append(label)

        logger.debug(
            "extract_vertebral_mesh: label %d -> %d cells, color=(%.2f, %.2f, %.2f)",
            label,
            n_out_cells,
            r,
            g,
            b,
        )

    if not labels_found:
        logger.info("extract_vertebral_mesh: no vertebral labels found in mask")
        return None

    appender.Update()
    result = vtk.vtkPolyData()
    result.DeepCopy(appender.GetOutput())

    logger.info(
        "extract_vertebral_mesh: %d vertebrae extracted (%s), total cells=%d",
        len(labels_found),
        ", ".join(VERTEBRA_LABELS.get(label, str(label)) for label in labels_found),
        result.GetNumberOfCells(),
    )

    return result
