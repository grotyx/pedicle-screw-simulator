"""Direct tests for STL export over a synthetic volume.

Regression scope: src/utils/stl_export.py writes a readable binary STL for
a synthetic bone-like sphere. Round-trips through vtkSTLReader.
"""

import os

import pytest
import vtk
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("vtk")

import numpy as np

from src.utils.stl_export import export_bone_stl


def _sphere_image(dims=(32, 32, 32), radius=10.0, bone_hu=800.0):
    z_index, y_index, x_index = np.indices((dims[2], dims[1], dims[0]))
    world = np.stack(
        (x_index.astype(float), y_index.astype(float), z_index.astype(float)),
        axis=-1,
    )
    center = (np.asarray(dims, dtype=float) - 1.0) / 2.0
    values = np.where(
        np.linalg.norm(world - center, axis=-1) <= radius, bone_hu, -1000.0
    ).astype(np.float32)

    image = vtk.vtkImageData()
    image.SetDimensions(*dims)
    image.SetSpacing(1.0, 1.0, 1.0)
    image.SetOrigin(0.0, 0.0, 0.0)
    image.GetPointData().SetScalars(numpy_to_vtk(values.ravel(), deep=True))
    return image


def test_export_bone_stl_writes_readable_mesh(tmp_path):
    out_path = tmp_path / "bone.stl"

    export_bone_stl(_sphere_image(), str(out_path))

    assert out_path.stat().st_size > 0
    reader = vtk.vtkSTLReader()
    reader.SetFileName(str(out_path))
    reader.Update()
    mesh = reader.GetOutput()
    assert mesh.GetNumberOfPoints() > 0
    assert mesh.GetNumberOfCells() > 0


def test_export_bone_stl_threshold_above_bone_writes_empty_mesh(tmp_path):
    out_path = tmp_path / "empty.stl"

    export_bone_stl(_sphere_image(bone_hu=100.0), str(out_path), threshold=400.0)

    reader = vtk.vtkSTLReader()
    reader.SetFileName(str(out_path))
    reader.Update()
    assert reader.GetOutput().GetNumberOfCells() == 0


def test_export_bone_stl_is_binary(tmp_path):
    out_path = tmp_path / "bone.stl"

    export_bone_stl(_sphere_image(), str(out_path))

    with open(out_path, "rb") as handle:
        assert handle.read(5) != b"solid"


def test_export_bone_stl_mesh_roughly_matches_sphere(tmp_path):
    out_path = tmp_path / "bone.stl"

    export_bone_stl(_sphere_image(radius=10.0), str(out_path), reduction=0.0)

    reader = vtk.vtkSTLReader()
    reader.SetFileName(str(out_path))
    reader.Update()
    points = vtk_to_numpy(reader.GetOutput().GetPoints().GetData())
    center = np.array([15.5, 15.5, 15.5])
    mean_radius = float(np.linalg.norm(points - center, axis=1).mean())
    assert mean_radius == pytest.approx(10.0, abs=2.0)
