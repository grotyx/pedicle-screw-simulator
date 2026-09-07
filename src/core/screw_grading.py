"""Mask-based Gertzbein-Robbins grading shared by the planner and manual tool."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import SimpleITK as sitk
from scipy import ndimage as ndi

from .screw_geometry import unit_trajectory

Point3 = Sequence[float]

#: LPS axis-aligned direction matrix. ``dicom_loader.normalize_orientation``
#: guarantees it for every volume the app loads, and the vectorised index
#: transform below relies on it (a point maps to an index by origin/spacing
#: alone, with no rotation).
_IDENTITY_DIRECTION = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)


@dataclass
class GradeResult:
    grade: str
    breach_mm: float
    min_wall_mm: float
    label: int
    mean_hu: Optional[float]
    min_hu: Optional[float]
    #: The cylinder sample that breached furthest, and the centreline point it
    #: belongs to (LPS mm). Both ``None`` when the screw is contained.
    breach_point_lps: Optional[Tuple[float, float, float]] = None
    breach_centre_lps: Optional[Tuple[float, float, float]] = None


class _DistanceMaps:
    """Cropped distance maps for one label (all values in mm)."""

    def __init__(self, outside: np.ndarray, inside: np.ndarray, crop_min_zyx: np.ndarray):
        self.outside = outside      # distance to nearest label voxel (0 inside)
        self.inside = inside        # distance to nearest non-label voxel (0 outside)
        self.crop_min = crop_min_zyx
        self.shape = np.asarray(outside.shape)


class ScrewGrader:
    """Grade a screw trajectory against a segmentation mask via distance maps.

    Distances are measured between voxel centres of the discretised mask, so a
    reported breach or wall distance carries up to about half a voxel of
    optimistic bias with respect to the underlying anatomical surface.

    ``ct_image``, when given, is sampled with the mask's own index transform and
    must therefore lie on the same grid as ``mask_image``.

    ``mask_image`` must have an identity direction matrix (LPS axis-aligned);
    ``dicom_loader.normalize_orientation`` produces one for every loaded volume.
    """

    def __init__(
        self,
        mask_image: sitk.Image,
        ct_image: Optional[sitk.Image] = None,
        sample_step_mm: float = 0.5,
        radial_samples: int = 8,
        crop_margin_mm: float = 12.0,
    ) -> None:
        if not np.allclose(mask_image.GetDirection(), _IDENTITY_DIRECTION, atol=1e-6):
            raise ValueError(
                "mask_image must have an identity (LPS axis-aligned) direction matrix; "
                f"got {tuple(float(v) for v in mask_image.GetDirection())}"
            )
        self._mask = mask_image
        self._mask_array = sitk.GetArrayFromImage(mask_image)  # (z, y, x)
        self._ct = ct_image
        if self._ct is not None:
            self._require_same_grid(mask_image, self._ct)
        self._ct_array = sitk.GetArrayFromImage(self._ct) if self._ct is not None else None
        self._step = float(sample_step_mm)
        self._radial = int(radial_samples)
        self._crop_margin = float(crop_margin_mm)
        sx, sy, sz = mask_image.GetSpacing()
        self._sampling_zyx = (float(sz), float(sy), float(sx))
        self._origin = np.asarray(mask_image.GetOrigin(), dtype=np.float64)      # x, y, z
        self._spacing = np.asarray(mask_image.GetSpacing(), dtype=np.float64)    # x, y, z
        self._size = np.asarray(mask_image.GetSize(), dtype=np.int64)            # x, y, z
        self._maps: Dict[int, Optional[_DistanceMaps]] = {}

    @property
    def crop_margin_mm(self) -> float:
        return self._crop_margin

    # ------------------------------------------------------------------ access
    def mask_image(self) -> sitk.Image:
        """The segmentation this grader was built from."""
        return self._mask

    def ct_image(self) -> Optional[sitk.Image]:
        """The CT this grader samples HU from, or ``None`` if it was built without one."""
        return self._ct

    def label_array(self) -> np.ndarray:
        """The segmentation as a read-only ``(z, y, x)`` array."""
        view = self._mask_array.view()
        view.flags.writeable = False
        return view

    # ------------------------------------------------------------------ labels
    def detect_label(self, entry: Point3, target: Point3) -> Optional[int]:
        from .pedicle_analyzer import VERTEBRA_LABELS  # local import: avoids cycle

        points = np.asarray(self._centreline(entry, target), dtype=np.float64)
        idx_zyx, inside = self._indices(points)
        sel = idx_zyx[inside]
        values = self._mask_array[sel[:, 0], sel[:, 1], sel[:, 2]]

        counts: Dict[int, int] = {}
        for raw in values:                       # ordered along the trajectory: ties keep the first label seen
            value = int(raw)
            if value in VERTEBRA_LABELS:
                counts[value] = counts.get(value, 0) + 1
        if not counts:
            return None
        return max(counts, key=counts.get)

    # ----------------------------------------------------------------- sampling
    def cylinder_points(self, entry: Point3, target: Point3, diameter_mm: float) -> np.ndarray:
        """Sample points filling the screw cylinder, as an ``(N, 3)`` LPS array.

        Points are ordered by centreline step; each step contributes its centre
        followed by ``radial_samples`` points one radius out, perpendicular to
        the trajectory. A degenerate (zero-length) trajectory has no defined
        perpendicular plane, so only the centreline points are returned.
        """
        centres = np.asarray(self._centreline(entry, target), dtype=np.float64)
        direction = np.asarray(unit_trajectory(entry, target), dtype=np.float64)
        if not direction.any():
            return centres
        offsets = np.asarray(
            self._radial_offsets(direction, float(diameter_mm) / 2.0), dtype=np.float64
        ).reshape(-1, 3)
        if offsets.shape[0] == 0:
            return centres
        samples = np.empty((centres.shape[0], offsets.shape[0] + 1, 3), dtype=np.float64)
        samples[:, 0, :] = centres
        samples[:, 1:, :] = centres[:, None, :] + offsets[None, :, :]
        return samples.reshape(-1, 3)

    def hu_at_points(self, points: np.ndarray) -> np.ndarray:
        """HU at each ``(N, 3)`` LPS point; ``NaN`` where there is no CT value."""
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        values = np.full(pts.shape[0], np.nan, dtype=np.float64)
        if self._ct_array is None or pts.shape[0] == 0:
            return values
        idx_zyx, inside = self._indices(pts)
        sel = idx_zyx[inside]
        values[inside] = self._ct_array[sel[:, 0], sel[:, 1], sel[:, 2]]
        return values

    def distances_at_points(self, points: np.ndarray, label: int) -> Tuple[np.ndarray, np.ndarray]:
        """Distances (mm) from each ``(N, 3)`` LPS point to ``label``.

        Returns ``(d_out, d_in)``: distance to the nearest voxel of the label
        (0 inside it) and, for points inside, the distance to the nearest voxel
        outside it (0 elsewhere). Points beyond the label's cropped
        neighbourhood — and every point when the label is absent from the mask —
        score ``crop_margin_mm`` outside and 0 mm inside.
        """
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        maps = self._maps_for(int(label))
        if maps is None:
            return (
                np.full(pts.shape[0], self._crop_margin, dtype=np.float64),
                np.zeros(pts.shape[0], dtype=np.float64),
            )
        idx_zyx, inside = self._indices(pts)
        return self._lookup(maps, idx_zyx, inside)

    # ------------------------------------------------------------------ grading
    def grade(
        self,
        entry: Point3,
        target: Point3,
        diameter_mm: float,
        label: Optional[int] = None,
    ) -> Optional[GradeResult]:
        if label is None:
            label = self.detect_label(entry, target)
        if label is None:
            return None
        maps = self._maps_for(label)
        if maps is None:
            return None

        if not np.asarray(unit_trajectory(entry, target), dtype=np.float64).any():
            return None

        points = self.cylinder_points(entry, target, diameter_mm)
        idx_zyx, inside = self._indices(points)
        d_out, d_in = self._lookup(maps, idx_zyx, inside)

        breach = float(d_out.max()) if d_out.size else 0.0
        breach_point: Optional[Tuple[float, float, float]] = None
        breach_centre: Optional[Tuple[float, float, float]] = None
        if breach > 0.0:
            # ``cylinder_points`` emits a fixed-size block per centreline step,
            # so integer division recovers the step the worst sample came from.
            centres = np.asarray(self._centreline(entry, target), dtype=np.float64)
            per_step = max(1, points.shape[0] // centres.shape[0])
            worst = int(np.argmax(d_out))
            breach_point = tuple(float(v) for v in points[worst])
            breach_centre = tuple(float(v) for v in centres[worst // per_step])

        on_surface = d_out == 0.0
        min_wall = float(d_in[on_surface].min()) if on_surface.any() else math.inf

        if self._ct_array is None:
            hu_samples = np.empty(0, dtype=np.float64)
        else:
            sel = idx_zyx[inside]
            hu_samples = self._ct_array[sel[:, 0], sel[:, 1], sel[:, 2]].astype(np.float64)

        if breach > 0.0:
            min_wall = 0.0
        if math.isinf(min_wall):
            min_wall = 0.0
        return GradeResult(
            grade=self.grade_from_breach(breach),
            breach_mm=float(breach),
            min_wall_mm=float(min_wall),
            label=int(label),
            mean_hu=float(np.mean(hu_samples)) if hu_samples.size else None,
            min_hu=float(np.min(hu_samples)) if hu_samples.size else None,
            breach_point_lps=breach_point,
            breach_centre_lps=breach_centre,
        )

    @staticmethod
    def grade_from_breach(breach_mm: float) -> str:
        if breach_mm <= 0.0:
            return "A"
        if breach_mm < 2.0:
            return "B"
        if breach_mm < 4.0:
            return "C"
        if breach_mm < 6.0:
            return "D"
        return "E"

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _require_same_grid(mask_image: sitk.Image, ct_image: sitk.Image, tolerance: float = 1e-4) -> None:
        """Reject a CT that is not voxel-aligned with the mask (it is sampled by mask index)."""
        matches = (
            tuple(ct_image.GetSize()) == tuple(mask_image.GetSize())
            and np.allclose(ct_image.GetSpacing(), mask_image.GetSpacing(), rtol=0.0, atol=tolerance)
            and np.allclose(ct_image.GetOrigin(), mask_image.GetOrigin(), rtol=0.0, atol=tolerance)
            and np.allclose(ct_image.GetDirection(), mask_image.GetDirection(), rtol=0.0, atol=tolerance)
        )
        if not matches:
            raise ValueError("ct_image and mask_image must share the same grid")

    def _centreline(self, entry: Point3, target: Point3) -> List[np.ndarray]:
        start = np.asarray(entry, dtype=np.float64)
        end = np.asarray(target, dtype=np.float64)
        length = float(np.linalg.norm(end - start))
        n_steps = max(1, int(round(length / self._step)))
        return [start + (end - start) * (i / n_steps) for i in range(n_steps + 1)]

    def _radial_offsets(self, direction: np.ndarray, radius: float) -> List[np.ndarray]:
        abs_d = np.abs(direction)
        aux = np.zeros(3)
        aux[int(np.argmin(abs_d))] = 1.0
        u = np.cross(direction, aux)
        u /= np.linalg.norm(u)
        v = np.cross(direction, u)
        v /= np.linalg.norm(v)
        return [
            radius * (math.cos(2 * math.pi * k / self._radial) * u + math.sin(2 * math.pi * k / self._radial) * v)
            for k in range(self._radial)
        ]

    def _indices(self, points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Vectorised ``TransformPhysicalPointToIndex`` plus a bounds test.

        Returns ``(indices_zyx, inside)``: an ``(N, 3)`` int array of array
        indices and an ``(N,)`` bool mask of the points that land in the volume.
        Index values for out-of-volume points are clamped and must not be used.
        ITK rounds half-integers up, hence ``floor(x + 0.5)``; the identity
        direction enforced in ``__init__`` is what makes the axis-wise form
        valid. Indices are clipped before the integer cast so that huge or
        non-finite coordinates cannot overflow.
        """
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        with np.errstate(invalid="ignore"):
            continuous = (pts - self._origin) / self._spacing
        continuous = np.where(np.isfinite(continuous), continuous, -1.0)
        clipped = np.clip(np.floor(continuous + 0.5), -1.0, self._size.astype(np.float64))
        idx_xyz = clipped.astype(np.int64)
        inside = np.all((idx_xyz >= 0) & (idx_xyz < self._size), axis=1)
        return idx_xyz[:, ::-1], inside

    def _maps_for(self, label: int) -> Optional[_DistanceMaps]:
        if label in self._maps:
            return self._maps[label]
        binary = self._mask_array == label
        if not binary.any():
            self._maps[label] = None
            return None
        coords = np.argwhere(binary)
        pad = np.ceil(self._crop_margin / np.asarray(self._sampling_zyx)).astype(int)
        lo = np.maximum(coords.min(axis=0) - pad, 0)
        hi = np.minimum(coords.max(axis=0) + pad + 1, np.asarray(binary.shape))
        crop = binary[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
        outside = ndi.distance_transform_edt(~crop, sampling=self._sampling_zyx)
        inside = ndi.distance_transform_edt(crop, sampling=self._sampling_zyx)
        maps = _DistanceMaps(outside.astype(np.float32), inside.astype(np.float32), lo)
        self._maps[label] = maps
        return maps

    def _lookup(
        self, maps: _DistanceMaps, idx_zyx: np.ndarray, inside: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Outside/inside distances per sample; samples off the cropped maps
        score ``crop_margin_mm`` outside and 0 mm inside."""
        d_out = np.full(idx_zyx.shape[0], self._crop_margin, dtype=np.float64)
        d_in = np.zeros(idx_zyx.shape[0], dtype=np.float64)
        if idx_zyx.shape[0] == 0:
            return d_out, d_in
        local = idx_zyx - maps.crop_min
        on_map = inside & np.all((local >= 0) & (local < maps.shape), axis=1)
        sel = local[on_map]
        d_out[on_map] = maps.outside[sel[:, 0], sel[:, 1], sel[:, 2]]
        d_in[on_map] = maps.inside[sel[:, 0], sel[:, 1], sel[:, 2]]
        return d_out, d_in
