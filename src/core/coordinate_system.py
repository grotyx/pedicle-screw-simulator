"""
Coordinate System - Handles DICOM LPS to VTK RAS transformations

CRITICAL DOCUMENTATION:

DICOM Coordinate System (LPS):
- L: Patient's Left (positive X)
- P: Patient's Posterior (positive Y)
- S: Patient's Superior (positive Z)

VTK/ITK Coordinate System (RAS):
- R: Patient's Right (positive X)
- A: Patient's Anterior (positive Y)
- S: Patient's Superior (positive Z)

Conversion:
- X_ras = -X_lps (flip left/right)
- Y_ras = -Y_lps (flip anterior/posterior)
- Z_ras = Z_lps (same superior direction)

Matrix Form:
    | -1   0   0   0 |
    |  0  -1   0   0 |
    |  0   0   1   0 |
    |  0   0   0   1 |
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

try:
    import vtk
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    vtk = None


class CoordinateSystem:
    """
    Handles coordinate system transformations between DICOM and VTK.

    This class provides utilities for:
    - LPS <-> RAS coordinate conversion
    - Image space <-> World space conversion
    - Direction matrix handling
    """

    # Transform matrix: LPS to RAS
    LPS_TO_RAS = np.array([
        [-1,  0,  0,  0],
        [ 0, -1,  0,  0],
        [ 0,  0,  1,  0],
        [ 0,  0,  0,  1]
    ], dtype=np.float64)

    # Transform matrix: RAS to LPS (same transform, it's self-inverse)
    RAS_TO_LPS = LPS_TO_RAS.copy()

    @staticmethod
    def lps_to_ras(point: Tuple[float, float, float]) -> Tuple[float, float, float]:
        """
        Convert a point from DICOM LPS to VTK RAS coordinates.

        Args:
            point: (x, y, z) in LPS coordinates

        Returns:
            (x, y, z) in RAS coordinates
        """
        return (-point[0], -point[1], point[2])

    @staticmethod
    def ras_to_lps(point: Tuple[float, float, float]) -> Tuple[float, float, float]:
        """
        Convert a point from VTK RAS to DICOM LPS coordinates.

        Args:
            point: (x, y, z) in RAS coordinates

        Returns:
            (x, y, z) in LPS coordinates
        """
        return (-point[0], -point[1], point[2])

    @staticmethod
    def transform_direction(
        direction: Tuple[float, ...],
        from_system: str = "lps",
        to_system: str = "ras"
    ) -> Tuple[float, ...]:
        """
        Transform a 3x3 direction matrix between coordinate systems.

        Args:
            direction: 9-element tuple of direction cosines (row-major)
            from_system: Source coordinate system ('lps' or 'ras')
            to_system: Target coordinate system ('lps' or 'ras')

        Returns:
            Transformed direction matrix as 9-element tuple
        """
        if from_system == to_system:
            return direction

        # Convert to numpy array
        dir_matrix = np.array(direction).reshape(3, 3)

        # Apply transformation
        transform = CoordinateSystem.LPS_TO_RAS[:3, :3]
        transformed = transform @ dir_matrix

        return tuple(transformed.flatten())

    @staticmethod
    def create_vtk_transform(
        origin: Tuple[float, float, float],
        spacing: Tuple[float, float, float],
        direction: Tuple[float, ...],
        apply_lps_to_ras: bool = True
    ) -> vtk.vtkTransform:
        """
        Create a VTK transform from image geometry.

        This creates a transform that converts from image index coordinates
        to world coordinates, optionally applying LPS->RAS conversion.

        Args:
            origin: Image origin in world coordinates
            spacing: Voxel spacing
            direction: 9-element direction cosine matrix
            apply_lps_to_ras: Whether to include LPS->RAS transform

        Returns:
            vtk.vtkTransform: Complete transformation
        """
        if vtk is None:
            raise ImportError("VTK is required to create VTK transforms")

        # Build 4x4 matrix
        matrix = vtk.vtkMatrix4x4()
        matrix.Identity()

        # Set direction with spacing
        for i in range(3):
            for j in range(3):
                matrix.SetElement(i, j, direction[i * 3 + j] * spacing[j])

        # Set origin
        for i in range(3):
            matrix.SetElement(i, 3, origin[i])

        # Create transform
        transform = vtk.vtkTransform()

        if apply_lps_to_ras:
            # Apply LPS->RAS first
            lps_ras_matrix = vtk.vtkMatrix4x4()
            for i in range(4):
                for j in range(4):
                    lps_ras_matrix.SetElement(
                        i, j, CoordinateSystem.LPS_TO_RAS[i, j]
                    )
            transform.Concatenate(lps_ras_matrix)

        transform.Concatenate(matrix)

        return transform

    @staticmethod
    def image_to_world(
        index: Tuple[int, int, int],
        origin: Tuple[float, float, float],
        spacing: Tuple[float, float, float],
        direction: Optional[Tuple[float, ...]] = None
    ) -> Tuple[float, float, float]:
        """
        Convert image index to world coordinates.

        Args:
            index: (i, j, k) voxel indices
            origin: Image origin
            spacing: Voxel spacing
            direction: Optional direction matrix (default: identity)

        Returns:
            (x, y, z) world coordinates
        """
        if direction is None:
            # Simple case: identity direction
            x = origin[0] + index[0] * spacing[0]
            y = origin[1] + index[1] * spacing[1]
            z = origin[2] + index[2] * spacing[2]
            return (x, y, z)

        # General case with direction matrix
        dir_matrix = np.array(direction).reshape(3, 3)
        idx = np.array(index, dtype=np.float64)
        spaced_idx = idx * np.array(spacing)
        world = dir_matrix @ spaced_idx + np.array(origin)

        return tuple(world)

    @staticmethod
    def world_to_image(
        point: Tuple[float, float, float],
        origin: Tuple[float, float, float],
        spacing: Tuple[float, float, float],
        direction: Optional[Tuple[float, ...]] = None
    ) -> Tuple[int, int, int]:
        """
        Convert world coordinates to image index.

        Args:
            point: (x, y, z) world coordinates
            origin: Image origin
            spacing: Voxel spacing
            direction: Optional direction matrix (default: identity)

        Returns:
            (i, j, k) voxel indices (rounded to nearest integer)
        """
        if direction is None:
            # Simple case: identity direction
            i = int(round((point[0] - origin[0]) / spacing[0]))
            j = int(round((point[1] - origin[1]) / spacing[1]))
            k = int(round((point[2] - origin[2]) / spacing[2]))
            return (i, j, k)

        # General case with direction matrix
        dir_matrix = np.array(direction).reshape(3, 3)
        inv_dir = np.linalg.inv(dir_matrix)
        offset = np.array(point) - np.array(origin)
        idx = inv_dir @ offset / np.array(spacing)

        return tuple(int(round(x)) for x in idx)

    @staticmethod
    def get_plane_normal(plane: str) -> Tuple[float, float, float]:
        """
        Get the normal vector for an anatomical plane.

        Args:
            plane: 'axial', 'sagittal', or 'coronal'

        Returns:
            Normal vector (x, y, z)
        """
        normals = {
            "axial": (0, 0, 1),      # Points superior (S)
            "sagittal": (1, 0, 0),   # Points left (L) in LPS
            "coronal": (0, 1, 0),    # Points posterior (P) in LPS
        }
        return normals.get(plane, (0, 0, 1))

    @staticmethod
    def get_plane_axes(plane: str) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
        """
        Get the row and column direction vectors for an anatomical plane.

        These define how the 2D slice maps to 3D space.

        Args:
            plane: 'axial', 'sagittal', or 'coronal'

        Returns:
            (row_direction, column_direction) each as (x, y, z)
        """
        if plane == "axial":
            # Looking down from head
            # Row: left-right (L), Column: anterior-posterior (P)
            return ((1, 0, 0), (0, 1, 0))

        elif plane == "sagittal":
            # Looking from patient's left side
            # Row: anterior-posterior (P), Column: superior-inferior (S)
            return ((0, 1, 0), (0, 0, 1))

        elif plane == "coronal":
            # Looking from front
            # Row: left-right (L), Column: superior-inferior (S)
            return ((1, 0, 0), (0, 0, 1))

        else:
            raise ValueError(f"Unknown plane: {plane}")
