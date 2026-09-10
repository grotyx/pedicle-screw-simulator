"""Geometry primitives for standard and screw-aligned MPR views."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import vtk

Point3D = Tuple[float, float, float]


@dataclass(frozen=True)
class ScrewSliceProjection:
    """A screw expressed in one slice-local coordinate system."""

    entry_slice: Point3D
    target_slice: Point3D
    intersects: bool
    intersection_slice: Optional[Point3D]
    distance_mm: float


@dataclass(frozen=True)
class ScrewMPRAxes:
    """Three orthogonal reslice matrices aligned with one screw."""

    oblique_axial: vtk.vtkMatrix4x4
    oblique_sagittal: vtk.vtkMatrix4x4
    cross_section: vtk.vtkMatrix4x4
    cross_section_center: Point3D
    distance_from_entry_mm: float

    @property
    def long_axis_1(self) -> vtk.vtkMatrix4x4:
        """Backward-compatible alias for the oblique axial plane."""
        return self.oblique_axial

    @property
    def long_axis_2(self) -> vtk.vtkMatrix4x4:
        """Backward-compatible alias for the oblique sagittal plane."""
        return self.oblique_sagittal


def _matrix_to_numpy(matrix: vtk.vtkMatrix4x4) -> np.ndarray:
    return np.array(
        [
            [matrix.GetElement(row, column) for column in range(4)]
            for row in range(4)
        ],
        dtype=float,
    )


def _transform_point(point: Point3D, matrix: np.ndarray) -> Point3D:
    transformed = matrix @ np.array((*point, 1.0), dtype=float)
    if abs(transformed[3]) > 1e-12:
        transformed = transformed / transformed[3]
    return tuple(float(value) for value in transformed[:3])


def slice_to_world(point: Point3D, axes: vtk.vtkMatrix4x4) -> Point3D:
    """Transform a slice-local display point into DICOM world coordinates."""
    return _transform_point(point, _matrix_to_numpy(axes))


def world_to_slice(point: Point3D, axes: vtk.vtkMatrix4x4) -> Point3D:
    """Transform a DICOM world point into slice-local display coordinates."""
    try:
        inverse = np.linalg.inv(_matrix_to_numpy(axes))
    except np.linalg.LinAlgError as exc:
        raise ValueError("reslice axes matrix is not invertible") from exc
    return _transform_point(point, inverse)


# (negative-direction letter, positive-direction letter) per LPS axis.
_LPS_AXIS_LETTERS = (("R", "L"), ("A", "P"), ("I", "S"))

# A column whose dominant component falls below this is only approximately
# aligned with a patient axis, so its letters are marked with a suffix.
ORIENTATION_OBLIQUE_THRESHOLD = 0.7
ORIENTATION_OBLIQUE_SUFFIX = "'"


def _axis_letters(column: np.ndarray) -> Tuple[str, str]:
    """Return (letter along +column, letter along -column) for one axis."""
    norm = float(np.linalg.norm(column))
    if norm <= 1e-9:
        return ("?", "?")
    unit = column / norm
    axis = int(np.argmax(np.abs(unit)))
    negative_letter, positive_letter = _LPS_AXIS_LETTERS[axis]
    if unit[axis] < 0.0:
        positive_letter, negative_letter = negative_letter, positive_letter
    if abs(unit[axis]) < ORIENTATION_OBLIQUE_THRESHOLD:
        positive_letter += ORIENTATION_OBLIQUE_SUFFIX
        negative_letter += ORIENTATION_OBLIQUE_SUFFIX
    return positive_letter, negative_letter


def orientation_letters(axes: vtk.vtkMatrix4x4) -> Dict[str, str]:
    """Name the patient direction at each edge of a resliced image.

    Column 0 of a reslice matrix points along screen-right and column 1
    along screen-up, so the letters follow any plane -- standard or
    screw-aligned -- without a per-plane table.  A column that is only
    approximately aligned with a patient axis (dominant component below
    ``ORIENTATION_OBLIQUE_THRESHOLD``) is suffixed with ``'``.

    Args:
        axes: Reslice matrix mapping slice-local points into LPS world mm.

    Returns:
        ``{"top": ..., "bottom": ..., "left": ..., "right": ...}``.
    """
    matrix = _matrix_to_numpy(axes)
    right_letter, left_letter = _axis_letters(matrix[:3, 0])
    top_letter, bottom_letter = _axis_letters(matrix[:3, 1])
    return {
        "top": top_letter,
        "bottom": bottom_letter,
        "left": left_letter,
        "right": right_letter,
    }


def project_screw_to_slice(
    entry: Point3D,
    target: Point3D,
    axes: vtk.vtkMatrix4x4,
    diameter: float,
) -> ScrewSliceProjection:
    """Project a finite screw segment into one MPR slice coordinate system."""
    entry_slice = world_to_slice(entry, axes)
    target_slice = world_to_slice(target, axes)
    entry_distance = float(entry_slice[2])
    target_distance = float(target_slice[2])

    if entry_distance * target_distance <= 0.0:
        distance_mm = 0.0
    else:
        distance_mm = min(abs(entry_distance), abs(target_distance))

    radius = max(float(diameter), 0.0) / 2.0
    intersects = distance_mm <= radius
    intersection_slice: Optional[Point3D] = None

    delta_distance = target_distance - entry_distance
    if entry_distance * target_distance <= 0.0 and abs(delta_distance) > 1e-12:
        fraction = -entry_distance / delta_distance
        intersection_slice = (
            entry_slice[0] + fraction * (target_slice[0] - entry_slice[0]),
            entry_slice[1] + fraction * (target_slice[1] - entry_slice[1]),
            0.0,
        )
    elif intersects:
        if abs(delta_distance) <= 1e-12:
            fraction = 0.5
        else:
            fraction = 0.0 if abs(entry_distance) <= abs(target_distance) else 1.0
        intersection_slice = (
            entry_slice[0] + fraction * (target_slice[0] - entry_slice[0]),
            entry_slice[1] + fraction * (target_slice[1] - entry_slice[1]),
            0.0,
        )

    return ScrewSliceProjection(
        entry_slice=entry_slice,
        target_slice=target_slice,
        intersects=intersects,
        intersection_slice=intersection_slice,
        distance_mm=distance_mm,
    )


def _make_reslice_axes(
    x_axis: np.ndarray,
    y_axis: np.ndarray,
    normal: np.ndarray,
    center: np.ndarray,
) -> vtk.vtkMatrix4x4:
    axes = vtk.vtkMatrix4x4()
    axes.Identity()
    for row in range(3):
        axes.SetElement(row, 0, float(x_axis[row]))
        axes.SetElement(row, 1, float(y_axis[row]))
        axes.SetElement(row, 2, float(normal[row]))
        axes.SetElement(row, 3, float(center[row]))
    return axes


def build_screw_mpr_axes(
    entry: Point3D,
    target: Point3D,
    position_fraction: float = 0.5,
) -> ScrewMPRAxes:
    """Build two long-axis and one perpendicular reslice matrix for a screw."""
    entry_array = np.asarray(entry, dtype=float)
    target_array = np.asarray(target, dtype=float)
    trajectory = target_array - entry_array
    length = float(np.linalg.norm(trajectory))
    if length <= 1e-9:
        raise ValueError("cannot build screw MPR axes for a zero-length screw")

    screw_axis = trajectory / length
    patient_superior = np.array((0.0, 0.0, 1.0), dtype=float)
    superior_axis = patient_superior - np.dot(
        patient_superior,
        screw_axis,
    ) * screw_axis
    if np.linalg.norm(superior_axis) <= 1e-6:
        patient_left = np.array((1.0, 0.0, 0.0), dtype=float)
        superior_axis = patient_left - np.dot(
            patient_left,
            screw_axis,
        ) * screw_axis
    superior_axis = superior_axis / np.linalg.norm(superior_axis)
    transverse_axis = np.cross(superior_axis, screw_axis)
    transverse_axis = transverse_axis / np.linalg.norm(transverse_axis)

    fraction = min(max(float(position_fraction), 0.0), 1.0)
    midpoint = (entry_array + target_array) / 2.0
    cross_section_center = entry_array + fraction * trajectory

    oblique_axial = _make_reslice_axes(
        transverse_axis,
        screw_axis,
        -superior_axis,
        midpoint,
    )
    oblique_sagittal = _make_reslice_axes(
        -screw_axis,
        superior_axis,
        transverse_axis,
        midpoint,
    )
    cross_section = _make_reslice_axes(
        transverse_axis,
        superior_axis,
        screw_axis,
        cross_section_center,
    )

    return ScrewMPRAxes(
        oblique_axial=oblique_axial,
        oblique_sagittal=oblique_sagittal,
        cross_section=cross_section,
        cross_section_center=tuple(float(value) for value in cross_section_center),
        distance_from_entry_mm=fraction * length,
    )
