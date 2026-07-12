"""
VTK helper functions for coordinate transforms and data conversion
"""

from __future__ import annotations

import numpy as np
from typing import Tuple, Optional

try:
    import vtk
    from vtk.util import numpy_support
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    vtk = None
    numpy_support = None

try:
    import SimpleITK as sitk
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    sitk = None

from .constants import LPS_TO_RAS_MATRIX


def sitk_to_vtk(sitk_image: sitk.Image) -> vtk.vtkImageData:
    """
    Convert SimpleITK image to VTK vtkImageData.

    Memory copies:
      1. sitk.GetArrayFromImage() → numpy (z, y, x)  [copy 1]
      2. numpy_to_vtk(deep=True) → VTK array          [copy 2]

    The intermediate step uses np.asfortranarray() which re-layouts the (z,y,x)
    C-contiguous array into Fortran order. In Fortran order the first axis
    varies fastest, so (z,y,x) Fortran == (x,y,z) C — exactly what VTK needs.

    Args:
        sitk_image: SimpleITK Image object

    Returns:
        vtk.vtkImageData with correct dimensions, spacing, origin

    Raises:
        TypeError: if the image has an unsupported numpy dtype
    """
    if sitk is None:
        raise ImportError("SimpleITK is required for sitk_to_vtk conversion")
    if vtk is None or numpy_support is None:
        raise ImportError("VTK is required for sitk_to_vtk conversion")

    # Copy 1: SimpleITK → numpy (z, y, x), C-contiguous
    numpy_array = sitk.GetArrayFromImage(sitk_image)

    # Key insight: VTK's flat array layout for SetDimensions(nx, ny, nz) indexes
    # as  offset = x + nx*(y + ny*z)  →  x varies fastest.
    # C-order ravel of a (z, y, x) array:
    #   offset = ix + nx*(iy + ny*iz)  →  identical to VTK's layout.
    # So NO transpose is needed — just C-order ravel.

    # Metadata
    size = sitk_image.GetSize()        # (x, y, z)
    spacing = sitk_image.GetSpacing()  # (x, y, z)
    origin = sitk_image.GetOrigin()    # (x, y, z)

    # Strict dtype → VTK type mapping (no silent fallback)
    dtype_to_vtk = {
        np.int8: vtk.VTK_CHAR,
        np.uint8: vtk.VTK_UNSIGNED_CHAR,
        np.int16: vtk.VTK_SHORT,
        np.uint16: vtk.VTK_UNSIGNED_SHORT,
        np.int32: vtk.VTK_INT,
        np.uint32: vtk.VTK_UNSIGNED_INT,
        np.float32: vtk.VTK_FLOAT,
        np.float64: vtk.VTK_DOUBLE,
    }
    vtk_type = dtype_to_vtk.get(numpy_array.dtype.type)
    if vtk_type is None:
        raise TypeError(
            f"Unsupported numpy dtype for VTK conversion: {numpy_array.dtype}"
        )

    # C-order ravel is a view (no copy) when array is already C-contiguous
    flat_array = numpy_array.ravel()
    # Copy 2: numpy → VTK (deep copy for memory safety)
    vtk_array = numpy_support.numpy_to_vtk(
        flat_array,
        deep=True,
        array_type=vtk_type,
    )

    # Assemble VTK image
    vtk_image = vtk.vtkImageData()
    vtk_image.SetDimensions(size[0], size[1], size[2])
    vtk_image.SetSpacing(spacing[0], spacing[1], spacing[2])
    vtk_image.SetOrigin(origin[0], origin[1], origin[2])
    vtk_image.GetPointData().SetScalars(vtk_array)

    return vtk_image


def downsample_vtk_image(
    vtk_image: vtk.vtkImageData,
    shrink_factors: Tuple[int, int, int] = (2, 2, 2),
) -> vtk.vtkImageData:
    """
    Downsample a vtkImageData by integer factors using vtkImageShrink3D.

    This reduces the 3D texture size for faster GPU upload on first render.
    For a 512x512x300 volume with (2,2,2) shrink: 156MB → ~19.5MB.

    Args:
        vtk_image: Input vtkImageData
        shrink_factors: (x, y, z) integer shrink factors

    Returns:
        Downsampled vtkImageData with updated spacing
    """
    if vtk is None:
        raise ImportError("VTK is required for downsample_vtk_image")

    shrink = vtk.vtkImageShrink3D()
    shrink.SetInputData(vtk_image)
    shrink.SetShrinkFactors(shrink_factors[0], shrink_factors[1], shrink_factors[2])
    shrink.AveragingOn()  # Average voxels (better quality than subsampling)
    shrink.Update()
    return shrink.GetOutput()


def lps_to_ras_transform(point: Tuple[float, float, float]) -> Tuple[float, float, float]:
    """
    Transform a point from DICOM LPS to VTK RAS coordinate system.

    LPS (DICOM): Left, Posterior, Superior
    RAS (VTK):   Right, Anterior, Superior

    Transform: X_ras = -X_lps, Y_ras = -Y_lps, Z_ras = Z_lps

    .. deprecated:: 3.0
        Use ``CoordinateSystem.lps_to_ras()`` instead.

    Args:
        point: (x, y, z) in LPS coordinates

    Returns:
        (x, y, z) in RAS coordinates
    """
    from ..core.coordinate_system import CoordinateSystem
    return CoordinateSystem.lps_to_ras(point)


def ras_to_lps_transform(point: Tuple[float, float, float]) -> Tuple[float, float, float]:
    """
    Transform a point from VTK RAS to DICOM LPS coordinate system.

    .. deprecated:: 3.0
        Use ``CoordinateSystem.ras_to_lps()`` instead.

    Args:
        point: (x, y, z) in RAS coordinates

    Returns:
        (x, y, z) in LPS coordinates
    """
    from ..core.coordinate_system import CoordinateSystem
    return CoordinateSystem.ras_to_lps(point)


def get_vtk_transform_matrix(
    direction_matrix: Tuple[float, ...],
    origin: Tuple[float, float, float],
    spacing: Tuple[float, float, float]
) -> vtk.vtkMatrix4x4:
    """
    Create a VTK transformation matrix from DICOM image geometry.

    Args:
        direction_matrix: 9-element tuple of direction cosines
        origin: (x, y, z) image origin
        spacing: (x, y, z) voxel spacing

    Returns:
        vtk.vtkMatrix4x4: 4x4 transformation matrix
    """
    if vtk is None:
        raise ImportError("VTK is required to build VTK transform matrices")

    matrix = vtk.vtkMatrix4x4()
    matrix.Identity()

    # Set direction with spacing
    for i in range(3):
        for j in range(3):
            matrix.SetElement(i, j, direction_matrix[i * 3 + j] * spacing[j])

    # Set origin
    for i in range(3):
        matrix.SetElement(i, 3, origin[i])

    return matrix


def create_reslice_axes(
    plane: str,
    center: Tuple[float, float, float]
) -> vtk.vtkMatrix4x4:
    """
    Create reslice axes matrix for MPR plane.

    Args:
        plane: One of 'axial', 'sagittal', 'coronal'
        center: (x, y, z) center point for the slice

    Returns:
        vtk.vtkMatrix4x4: Reslice axes matrix
    """
    if vtk is None:
        raise ImportError("VTK is required to create reslice axes")

    from .constants import (
        AXIAL_DIRECTION_COSINES,
        CORONAL_DIRECTION_COSINES,
        SAGITTAL_DIRECTION_COSINES
    )

    axes = vtk.vtkMatrix4x4()
    axes.Identity()

    if plane == 'axial':
        cosines = AXIAL_DIRECTION_COSINES
    elif plane == 'coronal':
        cosines = CORONAL_DIRECTION_COSINES
    elif plane == 'sagittal':
        cosines = SAGITTAL_DIRECTION_COSINES
    else:
        raise ValueError(f"Unknown plane: {plane}")

    # Set direction cosines
    for i in range(3):
        for j in range(3):
            axes.SetElement(i, j, cosines[i * 3 + j])

    # Set center position
    axes.SetElement(0, 3, center[0])
    axes.SetElement(1, 3, center[1])
    axes.SetElement(2, 3, center[2])

    return axes


def world_to_volume_index(
    point: Tuple[float, float, float],
    origin: Tuple[float, float, float],
    spacing: Tuple[float, float, float],
    direction: Optional[Tuple[float, ...]] = None
) -> Tuple[int, int, int]:
    """
    Convert world coordinates to volume voxel indices.

    .. deprecated:: 3.0
        Use ``CoordinateSystem.world_to_image()`` instead.

    Args:
        point: (x, y, z) world coordinates
        origin: Volume origin
        spacing: Volume spacing
        direction: Direction cosines (default: identity)

    Returns:
        (i, j, k) voxel indices
    """
    from ..core.coordinate_system import CoordinateSystem
    return CoordinateSystem.world_to_image(point, origin, spacing, direction)


def volume_index_to_world(
    index: Tuple[int, int, int],
    origin: Tuple[float, float, float],
    spacing: Tuple[float, float, float],
    direction: Optional[Tuple[float, ...]] = None
) -> Tuple[float, float, float]:
    """
    Convert volume voxel indices to world coordinates.

    .. deprecated:: 3.0
        Use ``CoordinateSystem.image_to_world()`` instead.

    Args:
        index: (i, j, k) voxel indices
        origin: Volume origin
        spacing: Volume spacing
        direction: Direction cosines (default: identity)

    Returns:
        (x, y, z) world coordinates
    """
    from ..core.coordinate_system import CoordinateSystem
    return CoordinateSystem.image_to_world(index, origin, spacing, direction)
