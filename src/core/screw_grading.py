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


@dataclass
class GradeResult:
    grade: str
    breach_mm: float
    min_wall_mm: float
    label: int
    mean_hu: Optional[float]
    min_hu: Optional[float]


class _DistanceMaps:
    """Cropped distance maps for one label (all values in mm)."""

    def __init__(self, outside: np.ndarray, inside: np.ndarray, crop_min_zyx: np.ndarray):
        self.outside = outside      # distance to nearest label voxel (0 inside)
        self.inside = inside        # distance to nearest non-label voxel (0 outside)
        self.crop_min = crop_min_zyx
        self.shape = np.asarray(outside.shape)


class ScrewGrader:
    """Grade a screw trajectory against a segmentation mask via distance maps."""

    def __init__(
        self,
        mask_image: sitk.Image,
        ct_image: Optional[sitk.Image] = None,
        sample_step_mm: float = 0.5,
        radial_samples: int = 8,
        crop_margin_mm: float = 12.0,
    ) -> None:
        self._mask = mask_image
        self._mask_array = sitk.GetArrayFromImage(mask_image)  # (z, y, x)
        self._ct = ct_image
        self._ct_array = sitk.GetArrayFromImage(ct_image) if ct_image is not None else None
        self._step = float(sample_step_mm)
        self._radial = int(radial_samples)
        self._crop_margin = float(crop_margin_mm)
        sx, sy, sz = mask_image.GetSpacing()
        self._sampling_zyx = (float(sz), float(sy), float(sx))
        self._maps: Dict[int, Optional[_DistanceMaps]] = {}

    @property
    def crop_margin_mm(self) -> float:
        return self._crop_margin

    # ------------------------------------------------------------------ labels
    def detect_label(self, entry: Point3, target: Point3) -> Optional[int]:
        from .pedicle_analyzer import VERTEBRA_LABELS  # local import: avoids cycle

        counts: Dict[int, int] = {}
        for point in self._centreline(entry, target):
            idx = self._to_index(point)
            if idx is None:
                continue
            value = int(self._mask_array[idx[2], idx[1], idx[0]])
            if value in VERTEBRA_LABELS:
                counts[value] = counts.get(value, 0) + 1
        if not counts:
            return None
        return max(counts, key=counts.get)

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

        direction = np.asarray(unit_trajectory(entry, target), dtype=np.float64)
        if not direction.any():
            return None
        radius = float(diameter_mm) / 2.0
        offsets = self._radial_offsets(direction, radius)

        breach = 0.0
        min_wall = math.inf
        hu_samples: List[float] = []
        for centre in self._centreline(entry, target):
            for point in (centre, *[centre + o for o in offsets]):
                d_out, d_in = self._lookup(maps, point)
                breach = max(breach, d_out)
                if d_out == 0.0:
                    min_wall = min(min_wall, d_in)
                hu = self._hu_at(point)
                if hu is not None:
                    hu_samples.append(hu)

        if breach > 0.0:
            min_wall = 0.0
        if math.isinf(min_wall):
            min_wall = 0.0
        return GradeResult(
            grade=self.grade_from_breach(breach),
            breach_mm=float(breach),
            min_wall_mm=float(min_wall),
            label=int(label),
            mean_hu=float(np.mean(hu_samples)) if hu_samples else None,
            min_hu=float(np.min(hu_samples)) if hu_samples else None,
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

    def _to_index(self, point: np.ndarray) -> Optional[Tuple[int, int, int]]:
        try:
            idx = self._mask.TransformPhysicalPointToIndex([float(point[0]), float(point[1]), float(point[2])])
        except RuntimeError:
            return None
        size = self._mask.GetSize()
        if not (0 <= idx[0] < size[0] and 0 <= idx[1] < size[1] and 0 <= idx[2] < size[2]):
            return None
        return int(idx[0]), int(idx[1]), int(idx[2])

    def _hu_at(self, point: np.ndarray) -> Optional[float]:
        if self._ct_array is None:
            return None
        idx = self._to_index(point)
        if idx is None:
            return None
        return float(self._ct_array[idx[2], idx[1], idx[0]])

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

    def _lookup(self, maps: _DistanceMaps, point: np.ndarray) -> Tuple[float, float]:
        idx = self._to_index(point)
        if idx is None:
            return self._crop_margin, 0.0
        local = np.array([idx[2], idx[1], idx[0]]) - maps.crop_min
        if np.any(local < 0) or np.any(local >= maps.shape):
            return self._crop_margin, 0.0
        z, y, x = (int(v) for v in local)
        return float(maps.outside[z, y, x]), float(maps.inside[z, y, x])
