"""
STL Export utility for bone surface mesh.

This is the ONLY place where FlyingEdges3D surface extraction is used.
It is for export only, NOT for interactive 3D viewing.
"""

import vtk
from typing import Optional


def export_bone_stl(
    vtk_image: vtk.vtkImageData,
    output_path: str,
    threshold: float = 400.0,
    reduction: float = 0.5,
) -> None:
    """
    Extract bone surface from CT volume and export as STL file.

    Args:
        vtk_image: CT volume as vtkImageData
        output_path: File path for STL output
        threshold: HU threshold for bone (default 400)
        reduction: Decimation target reduction (0.0–1.0, default 0.5)
    """
    flying_edges = vtk.vtkFlyingEdges3D()
    flying_edges.SetInputData(vtk_image)
    flying_edges.SetValue(0, threshold)
    flying_edges.ComputeNormalsOn()
    flying_edges.ComputeGradientsOff()

    decimate = vtk.vtkDecimatePro()
    decimate.SetInputConnection(flying_edges.GetOutputPort())
    decimate.SetTargetReduction(reduction)
    decimate.PreserveTopologyOn()

    smoother = vtk.vtkSmoothPolyDataFilter()
    smoother.SetInputConnection(decimate.GetOutputPort())
    smoother.SetNumberOfIterations(20)
    smoother.SetRelaxationFactor(0.1)
    smoother.FeatureEdgeSmoothingOff()

    normals = vtk.vtkPolyDataNormals()
    normals.SetInputConnection(smoother.GetOutputPort())
    normals.ComputePointNormalsOn()
    normals.SplittingOff()

    writer = vtk.vtkSTLWriter()
    writer.SetInputConnection(normals.GetOutputPort())
    writer.SetFileName(output_path)
    writer.SetFileTypeToBinary()
    writer.Write()
