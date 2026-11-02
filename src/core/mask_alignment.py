"""Canonical label maps on the normalized CT's identity-LPS grid."""

import numpy as np
import SimpleITK as sitk

from .screw_grading import resample_mask_to_ct

_IDENTITY = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)


def align_mask_to_ct(mask: sitk.Image, ct: sitk.Image) -> sitk.Image:
    """Align categorical labels before any viewer or computation consumes them.

    The CT is already normalized by the DICOM loader. Reject unsupported grids
    and foreground that disappears entirely, rather than silently isolating air.
    Empty masks remain valid results (for example an empty threshold result).
    """
    for image in (mask, ct):
        if image is None or image.GetDimension() != 3 or image.GetNumberOfComponentsPerPixel() != 1:
            raise ValueError('Mask and CT must be scalar 3D images')
        if not all(image.GetSize()) or not np.isfinite((*image.GetOrigin(), *image.GetSpacing(), *image.GetDirection())).all():
            raise ValueError('Mask and CT must have finite, nonempty grids')
    if not np.allclose(ct.GetDirection(), _IDENTITY, atol=1e-6, rtol=0):
        raise ValueError('CT grid must be normalized to identity LPS')
    if all(getattr(mask, key)() == getattr(ct, key)() for key in
           ('GetSize', 'GetOrigin', 'GetSpacing', 'GetDirection')):
        return mask
    aligned = resample_mask_to_ct(mask, ct)
    if np.any(sitk.GetArrayViewFromImage(mask)) and not np.any(sitk.GetArrayViewFromImage(aligned)):
        raise ValueError('Segmentation foreground has no overlap with the CT grid')
    return aligned


def load_aligned_mask(path: str, ct: sitk.Image) -> sitk.Image:
    """Read and canonicalize a run's mask, including its downstream planning file."""
    mask = sitk.ReadImage(path)
    aligned = align_mask_to_ct(mask, ct)
    if aligned is not mask:
        sitk.WriteImage(aligned, path)
    return aligned
