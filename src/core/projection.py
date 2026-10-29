"""Virtual AP / lateral radiographs: parallel projections of the CT volume.

Pure numpy: no Qt or VTK.  The CT is reduced along the volume axis closest to
the patient A-P (AP view) or L-R (lateral view) axis, either as a ray-sum
(mean attenuation, like a digitally reconstructed radiograph) or as a maximum
intensity projection.  Images are laid out radiologically, as the MPR views
are: AP has patient right on the viewer's left, lateral has anterior on the
left, and superior is up in both.
"""

from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np

VIEWS = ("ap", "lateral")
MODES = ("raysum", "mip")
# Direction cosines further than this from the nearest patient axis make the
# view labels approximate; the window says so instead of hiding it.
OBLIQUE_TOLERANCE_DEG = 2.0
AIR_HU = -1000

VIEW_LETTERS = {
    "ap": {"top": "S", "bottom": "I", "left": "R", "right": "L"},
    "lateral": {"top": "S", "bottom": "I", "left": "A", "right": "P"},
}
# World (LPS) axis that runs along the image columns; the beam runs along
# the other horizontal axis.
_COLUMN_AXIS = {"ap": 0, "lateral": 1}


@dataclass(frozen=True)
class Projection:
    """A 2D projection and the geometry that maps LPS points onto it.

    ``image`` is (rows, cols), float 0..1, row 0 the most superior.  Pixel
    (col, row) is centred on world ``col_origin + col * col_spacing`` along
    the column axis and ``row_origin - row * row_spacing`` along S.
    """

    image: np.ndarray
    view: str
    mode: str
    col_axis: int
    col_origin: float
    col_spacing: float
    row_origin: float
    row_spacing: float
    oblique_deg: float

    @property
    def width_mm(self) -> float:
        return self.image.shape[1] * self.col_spacing

    @property
    def height_mm(self) -> float:
        return self.image.shape[0] * self.row_spacing

    @property
    def oblique(self) -> bool:
        return self.oblique_deg > OBLIQUE_TOLERANCE_DEG


def _axis_assignment(direction: Sequence[float]):
    """Map each world axis to (image axis, sign) and return the worst tilt."""
    matrix = np.asarray(direction, dtype=np.float64).reshape(3, 3)
    # Greedy claim of the largest |cosine| keeps the mapping a permutation
    # even for 45-degree obliques (see normalize_orientation).
    pairs = sorted(
        ((abs(matrix[w, a]), w, a) for w in range(3) for a in range(3)),
        reverse=True,
    )
    image_axis = [None, None, None]
    for _magnitude, w, a in pairs:
        if image_axis[w] is None and a not in image_axis:
            image_axis[w] = a
    signs = [1.0 if matrix[w, image_axis[w]] >= 0 else -1.0 for w in range(3)]
    cosines = [abs(matrix[w, image_axis[w]]) for w in range(3)]
    tilt = float(np.degrees(np.arccos(min(1.0, min(cosines)))))
    return image_axis, signs, tilt


def project_volume(
    array_zyx: np.ndarray,
    spacing: Sequence[float],
    origin: Sequence[float],
    direction: Sequence[float],
    view: str,
    mode: str,
) -> Projection:
    """Project a SimpleITK-ordered (z, y, x) CT array for ``view``/``mode``."""
    if view not in VIEWS:
        raise ValueError(f"Unknown projection view: {view!r}")
    if mode not in MODES:
        raise ValueError(f"Unknown projection mode: {mode!r}")

    image_axis, signs, tilt = _axis_assignment(direction)
    # numpy axis (2 - a) holds image axis a; reorder to world (x, y, z) and
    # flip so every index grows with its LPS coordinate.
    volume = np.transpose(array_zyx, [2 - a for a in image_axis])
    lo = []
    world_spacing = []
    for w in range(3):
        step = float(spacing[image_axis[w]])
        world_spacing.append(step)
        if signs[w] < 0:
            volume = np.flip(volume, axis=w)
            lo.append(float(origin[w]) - (volume.shape[w] - 1) * step)
        else:
            lo.append(float(origin[w]))

    col_axis = _COLUMN_AXIS[view]
    beam_axis = 1 - col_axis
    if mode == "mip":
        flat = volume.max(axis=beam_axis).astype(np.float32)
    else:
        # Air floor so padding below -1000 HU cannot read as negative mass.
        if volume.dtype.kind != "u":
            volume = np.maximum(volume, volume.dtype.type(AIR_HU))
        flat = volume.mean(axis=beam_axis, dtype=np.float32)
    # flat is (column axis, z): transpose to rows and put superior on top.
    image = flat.T[::-1]
    low, high = float(image.min()), float(image.max())
    image = (image - low) / (high - low) if high > low else np.zeros_like(image)

    return Projection(
        image=np.ascontiguousarray(image, dtype=np.float32),
        view=view,
        mode=mode,
        col_axis=col_axis,
        col_origin=lo[col_axis],
        col_spacing=world_spacing[col_axis],
        row_origin=lo[2] + (volume.shape[2] - 1) * world_spacing[2],
        row_spacing=world_spacing[2],
        oblique_deg=tilt,
    )


def project_point(
    projection: Projection, point: Sequence[float]
) -> Tuple[float, float]:
    """Continuous (col, row) pixel position of one LPS point on ``projection``."""
    col = (float(point[projection.col_axis]) - projection.col_origin) / projection.col_spacing
    row = (projection.row_origin - float(point[2])) / projection.row_spacing
    return col, row
