"""Synthetic physical-grid regressions for canonical segmentation masks."""

import numpy as np
import pytest
import SimpleITK as sitk
from vtk.util.numpy_support import vtk_to_numpy

from src.core.vertebral_mesh import create_vertebral_only_volume
from src.utils.vtk_helpers import sitk_to_vtk


@pytest.mark.parametrize('geometry', ['origin', 'spacing', 'size', 'direction'])
def test_canonical_mask_preserves_physical_label(geometry):
    from src.core.mask_alignment import align_mask_to_ct

    ct = sitk.Image([9, 9, 9], sitk.sitkInt16)
    mask = sitk.Image([9, 9, 9], sitk.sitkUInt8)
    index = (4, 4, 4)
    if geometry == 'origin':
        mask.SetOrigin((1, 0, 0))
        index = (3, 4, 4)
    elif geometry == 'spacing':
        mask.SetSpacing((2, 2, 2))
        index = (2, 2, 2)
    elif geometry == 'size':
        mask = sitk.Image([6, 6, 6], sitk.sitkUInt8)
    else:
        mask.SetDirection((-1, 0, 0, 0, -1, 0, 0, 0, 1))
        mask.SetOrigin((8, 8, 0))
    mask[index] = 28
    aligned = align_mask_to_ct(mask, ct)
    assert aligned.GetSize() == ct.GetSize()
    assert aligned.GetOrigin() == ct.GetOrigin()
    assert aligned.GetSpacing() == ct.GetSpacing()
    assert aligned.GetDirection() == ct.GetDirection()
    assert aligned[4, 4, 4] == 28
    assert set(np.unique(sitk.GetArrayFromImage(aligned))) == {0, 28}
    isolated = create_vertebral_only_volume(sitk_to_vtk(ct + 500), sitk_to_vtk(aligned), 0)
    assert vtk_to_numpy(isolated.GetPointData().GetScalars()).reshape(9, 9, 9)[4, 4, 4] == 500


def test_canonical_mask_rejects_lost_foreground():
    from src.core.mask_alignment import align_mask_to_ct

    ct = sitk.Image([4, 4, 4], sitk.sitkInt16)
    mask = sitk.Image([4, 4, 4], sitk.sitkUInt8)
    mask[1, 1, 1] = 28
    mask.SetOrigin((100, 0, 0))
    with pytest.raises(ValueError, match='overlap'):
        align_mask_to_ct(mask, ct)


@pytest.mark.parametrize('geometry', ['origin', 'spacing', 'size', 'direction', 'extent'])
def test_isolation_rejects_noncanonical_grid(geometry):
    ct = sitk_to_vtk(sitk.Image([4, 4, 4], sitk.sitkInt16))
    mask = sitk_to_vtk(sitk.Image([4, 4, 4], sitk.sitkUInt8))
    if geometry == 'origin':
        mask.SetOrigin(100, 0, 0)
    elif geometry == 'spacing':
        mask.SetSpacing(2, 1, 1)
    elif geometry == 'size':
        mask.SetDimensions(3, 4, 4)
    elif geometry == 'extent':
        mask.SetExtent(1, 4, 0, 3, 0, 3)
    else:
        mask.SetDirectionMatrix(-1, 0, 0, 0, -1, 0, 0, 0, 1)
    with pytest.raises(ValueError, match='grid'):
        create_vertebral_only_volume(ct, mask, 0)
