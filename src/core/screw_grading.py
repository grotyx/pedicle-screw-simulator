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


#: Ceiling on the number of cylinder samples ``evaluate_batch`` materialises in
#: one pass. Larger batches are split into candidate chunks so that peak memory
#: stays bounded no matter how many candidates an optimiser throws at it.
MAX_BATCH_SAMPLE_POINTS = 2_000_000

#: Shortest candidate ``evaluate_batch`` will rank. A shorter trajectory has no
#: well-defined direction, so its cylinder collapses towards a single point that
#: sits comfortably inside the vertebra and scores as a flawless screw — the one
#: thing an optimiser must never be handed. Such candidates are reported as
#: unrankable instead (see :meth:`ScrewGrader.evaluate_batch`).
MIN_BATCH_CANDIDATE_LENGTH_MM = 1.0

#: Grid-equality tolerances used by :meth:`ScrewGrader.grids_match`.
#:
#: An origin is compared against a *fraction of a voxel* rather than an absolute
#: millimetre count: NIfTI stores the origin as float32, so a CT whose table
#: position puts |z| past 2048 mm round-trips through a saved mask with up to
#: 1.22e-4 mm of drift — real misalignment starts far above that, and a fixed
#: absolute tolerance either rejects the round-trip or accepts a shifted grid on
#: a fine volume.  Spacing is relative for the same reason; the direction cosines
#: are already dimensionless.
GRID_ORIGIN_VOXEL_FRACTION = 0.01
GRID_SPACING_RTOL = 1e-3
GRID_DIRECTION_ATOL = 1e-3


def resample_mask_to_ct(mask_image: sitk.Image, ct_image: sitk.Image) -> sitk.Image:
    """``mask_image`` resampled onto ``ct_image``'s grid, nearest neighbour.

    Labels are categorical, so the interpolation must never blend them, and
    voxels the mask does not reach become background (0).  The pixel type is
    preserved so the result is still a label map.
    """
    return sitk.Resample(
        mask_image,
        ct_image,
        sitk.Transform(),
        sitk.sitkNearestNeighbor,
        0,
        mask_image.GetPixelID(),
    )


@dataclass
class BatchResult:
    """Per-candidate measurements from :meth:`ScrewGrader.evaluate_batch`.

    Every array is ``(C,)`` and indexed by candidate. ``min_wall_mm`` is 0 for a
    candidate that breaches, and the HU arrays are ``NaN`` for a candidate with
    no sample inside the CT (including every candidate when there is no CT) and
    for a candidate too short to rank.
    """

    breach_mm: np.ndarray
    min_wall_mm: np.ndarray
    mean_hu: np.ndarray
    min_hu: np.ndarray


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
        self._present: Dict[int, bool] = {}

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

    def has_label(self, label: int) -> bool:
        """Whether ``label`` appears anywhere in the segmentation.

        Cached per label, on the same assumption the distance-map cache makes:
        a grader's mask never changes, so a new segmentation means a new
        grader.  Callers on the drag path ask this once per frame, and the
        answer is otherwise a full-volume scan every time.
        """
        key = int(label)
        present = self._present.get(key)
        if present is None:
            present = bool((self._mask_array == key).any())
            self._present[key] = present
        return present

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

    def evaluate_batch(
        self,
        entries: np.ndarray,
        targets: np.ndarray,
        diameter_mm: float,
        label: int,
    ) -> BatchResult:
        """Grade many candidate trajectories at once.

        ``entries`` and ``targets`` are ``(C, 3)`` LPS arrays. Every candidate is
        sampled with the same number of centreline steps, taken from the longest
        candidate, so a short candidate is sampled more finely than
        :meth:`grade` would sample it; because the reductions are a max and a
        min, the extra samples can only refine the answer, and the two agree to
        within roughly the sampling step. Unlike :meth:`grade` this never
        returns ``None``: a candidate whose samples all miss the label's cropped
        neighbourhood simply scores ``crop_margin_mm`` of breach.

        A candidate shorter than ``MIN_BATCH_CANDIDATE_LENGTH_MM`` is unrankable
        and is reported as the worst possible screw — ``crop_margin_mm`` of
        breach, no wall, ``NaN`` HU — rather than as the flawless one its
        collapsed cylinder would otherwise measure as.
        """
        starts = np.asarray(entries, dtype=np.float64).reshape(-1, 3)
        ends = np.asarray(targets, dtype=np.float64).reshape(-1, 3)
        if starts.shape != ends.shape:
            raise ValueError(
                f"entries and targets must have the same shape; got {starts.shape} and {ends.shape}"
            )
        count = starts.shape[0]
        if count == 0:
            return BatchResult(*(np.empty(0, dtype=np.float64) for _ in range(4)))

        deltas = ends - starts
        lengths = np.linalg.norm(deltas, axis=1)
        n_samples = int(math.ceil(float(lengths.max()) / self._step)) + 1
        per_step = 1 + max(0, self._radial)

        # Unit trajectories, with degenerate (zero-length) candidates left at
        # zero so the frame construction stays branch-free; whatever they sample
        # is discarded by the unrankable override below.
        directions = np.zeros_like(deltas)
        movable = lengths > 1e-12
        directions[movable] = deltas[movable] / lengths[movable, None]
        offsets = self._batch_radial_offsets(directions, float(diameter_mm) / 2.0)

        chunk = max(1, MAX_BATCH_SAMPLE_POINTS // (n_samples * per_step))
        parts: List[Tuple[np.ndarray, ...]] = [
            self._evaluate_slice(
                starts[lo:lo + chunk], deltas[lo:lo + chunk], offsets[lo:lo + chunk], n_samples, label
            )
            for lo in range(0, count, chunk)
        ]
        result = BatchResult(*(np.concatenate(values) for values in zip(*parts, strict=True)))

        unrankable = lengths < MIN_BATCH_CANDIDATE_LENGTH_MM
        if unrankable.any():
            result.breach_mm[unrankable] = self._crop_margin
            result.min_wall_mm[unrankable] = 0.0
            result.mean_hu[unrankable] = np.nan
            result.min_hu[unrankable] = np.nan
        return result

    def _batch_radial_offsets(self, directions: np.ndarray, radius: float) -> np.ndarray:
        """``(C, radial_samples, 3)`` perpendicular offsets, one frame per direction.

        The frame construction mirrors :meth:`_radial_offsets` — auxiliary axis
        at the smallest direction component, then two cross products — so a
        batch and a single evaluation sample the same points around a centre.
        """
        count = directions.shape[0]
        if self._radial <= 0:
            return np.zeros((count, 0, 3), dtype=np.float64)
        aux = np.zeros_like(directions)
        aux[np.arange(count), np.argmin(np.abs(directions), axis=1)] = 1.0
        u = self._normalise_rows(np.cross(directions, aux))
        v = self._normalise_rows(np.cross(directions, u))
        angles = 2.0 * math.pi * np.arange(self._radial, dtype=np.float64) / self._radial
        return radius * (
            np.cos(angles)[None, :, None] * u[:, None, :]
            + np.sin(angles)[None, :, None] * v[:, None, :]
        )

    @staticmethod
    def _normalise_rows(vectors: np.ndarray) -> np.ndarray:
        """Unit-length rows, leaving zero-length rows (degenerate frames) at zero."""
        norms = np.linalg.norm(vectors, axis=1)
        nonzero = norms > 0.0
        out = np.zeros_like(vectors)
        out[nonzero] = vectors[nonzero] / norms[nonzero, None]
        return out

    def _evaluate_slice(
        self,
        starts: np.ndarray,
        deltas: np.ndarray,
        offsets: np.ndarray,
        n_samples: int,
        label: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """One chunk of :meth:`evaluate_batch`, reduced along its sample axes."""
        count = starts.shape[0]
        per_step = 1 + offsets.shape[1]
        t = np.linspace(0.0, 1.0, n_samples)
        centres = starts[:, None, :] + deltas[:, None, :] * t[None, :, None]   # (C, S, 3)
        points = np.empty((count, n_samples, per_step, 3), dtype=np.float64)
        points[:, :, 0, :] = centres
        if offsets.shape[1]:
            points[:, :, 1:, :] = centres[:, :, None, :] + offsets[:, None, :, :]
        flat = points.reshape(-1, 3)

        d_out, d_in = self.distances_at_points(flat, label)
        d_out = d_out.reshape(count, -1)
        d_in = d_in.reshape(count, -1)
        breach = d_out.max(axis=1)
        min_wall = np.where(d_out == 0.0, d_in, np.inf).min(axis=1)
        min_wall = np.where(np.isfinite(min_wall) & (breach <= 0.0), min_wall, 0.0)

        hu = self.hu_at_points(flat).reshape(count, -1)
        sampled = np.isfinite(hu)
        counts = sampled.sum(axis=1)
        mean_hu = np.full(count, np.nan, dtype=np.float64)
        np.divide(
            np.where(sampled, hu, 0.0).sum(axis=1), counts, out=mean_hu, where=counts > 0
        )
        min_hu = np.where(
            counts > 0, np.where(sampled, hu, np.inf).min(axis=1), np.nan
        )
        return breach, min_wall, mean_hu, min_hu

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
    def grids_match(ct_image: sitk.Image, mask_image: sitk.Image) -> bool:
        """Whether the two volumes are voxel-aligned closely enough to share indices.

        Sizes must be equal; spacing is compared relatively
        (:data:`GRID_SPACING_RTOL`), the direction cosines absolutely
        (:data:`GRID_DIRECTION_ATOL`), and the origins within
        :data:`GRID_ORIGIN_VOXEL_FRACTION` of the *smaller* voxel — a
        sub-hundredth-of-a-voxel offset cannot move a sample into a different
        voxel, whereas an absolute millimetre tolerance is simultaneously too
        tight for a float32 origin round-trip and too loose for a fine volume.

        Callers that can repair a mismatch should resample the mask with
        :func:`resample_mask_to_ct` rather than fail.
        """
        if tuple(ct_image.GetSize()) != tuple(mask_image.GetSize()):
            return False
        ct_spacing = np.asarray(ct_image.GetSpacing(), dtype=np.float64)
        mask_spacing = np.asarray(mask_image.GetSpacing(), dtype=np.float64)
        if not np.allclose(ct_spacing, mask_spacing, rtol=GRID_SPACING_RTOL, atol=0.0):
            return False
        if not np.allclose(
            ct_image.GetDirection(),
            mask_image.GetDirection(),
            rtol=0.0,
            atol=GRID_DIRECTION_ATOL,
        ):
            return False
        voxel = float(min(mask_spacing.min(), ct_spacing.min()))
        return bool(
            np.allclose(
                ct_image.GetOrigin(),
                mask_image.GetOrigin(),
                rtol=0.0,
                atol=GRID_ORIGIN_VOXEL_FRACTION * voxel,
            )
        )

    @classmethod
    def _require_same_grid(cls, mask_image: sitk.Image, ct_image: sitk.Image) -> None:
        """Reject a CT that is not voxel-aligned with the mask (it is sampled by mask index)."""
        if not cls.grids_match(ct_image, mask_image):
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
