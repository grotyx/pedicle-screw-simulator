"""
Pedicle morphology analyzer for vertebral segmentation masks.

Takes a TotalSegmentator multilabel mask (SimpleITK image) and produces
per-vertebra pedicle geometry: isthmus centre, axis direction, minimum
width, and vertebral body centre.  All output coordinates are in the
DICOM **LPS** coordinate system.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import SimpleITK as sitk
from scipy import ndimage as ndi

from .vertebra import PedicleAnalysisResult, Vertebra

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# TotalSegmentator vertebrae label map (task="total", v2)
# ---------------------------------------------------------------------------
VERTEBRA_LABELS: Dict[int, str] = {
    25: "sacrum",
    26: "S1",
    27: "L5",
    28: "L4",
    29: "L3",
    30: "L2",
    31: "L1",
    32: "T12",
    33: "T11",
    34: "T10",
    35: "T9",
    36: "T8",
    37: "T7",
    38: "T6",
    39: "T5",
    40: "T4",
    41: "T3",
    42: "T2",
    43: "T1",
}

# Labels for which pedicle analysis is not meaningful.
_SKIP_PEDICLE_LABELS: frozenset = frozenset({25})  # sacrum


class PedicleAnalyzer:
    """Analyze vertebral pedicle morphology from segmentation masks.

    Parameters
    ----------
    mask_image : sitk.Image
        TotalSegmentator multilabel mask.  Voxel values correspond to
        label ids defined in ``VERTEBRA_LABELS``.
    ct_image : sitk.Image, optional
        Original CT volume for Hounsfield-unit sampling.  Not used in
        the current implementation but reserved for future density-aware
        analysis.
    """

    def __init__(
        self,
        mask_image: sitk.Image,
        ct_image: Optional[sitk.Image] = None,
    ) -> None:
        self._mask_image = mask_image
        self._ct_image = ct_image

        # Pre-compute numpy array and geometry once.
        # sitk.GetArrayFromImage returns shape (z, y, x).
        self._mask_array: np.ndarray = sitk.GetArrayFromImage(mask_image)
        self._spacing: Tuple[float, float, float] = mask_image.GetSpacing()  # (sx, sy, sz)
        self._origin: Tuple[float, float, float] = mask_image.GetOrigin()

    # ------------------------------------------------------------------
    # Index <-> physical helpers
    # ------------------------------------------------------------------

    def _ijk_to_lps(self, i: int, j: int, k: int) -> np.ndarray:
        """Convert *ijk* voxel index to LPS physical coordinates.

        Uses SimpleITK's ``TransformIndexToPhysicalPoint`` which
        accounts for origin, spacing **and** direction cosines.
        """
        # sitk index order: (x=i, y=j, z=k)
        pt = self._mask_image.TransformIndexToPhysicalPoint((int(i), int(j), int(k)))
        return np.array(pt, dtype=np.float64)

    def _indices_centroid_lps(self, indices_zyx: np.ndarray) -> np.ndarray:
        """Compute the physical-space centroid of a set of voxels.

        Parameters
        ----------
        indices_zyx : ndarray, shape (N, 3)
            Voxel indices in *(z, y, x)* order — the native order from
            ``np.argwhere`` on the sitk array.
        """
        mean_zyx = indices_zyx.mean(axis=0)
        # Convert to ijk (x, y, z) for sitk
        x, y, z = mean_zyx[2], mean_zyx[1], mean_zyx[0]
        return self._ijk_to_lps(int(round(x)), int(round(y)), int(round(z)))

    def _indices_to_lps(self, indices_zyx: np.ndarray) -> np.ndarray:
        """Convert an array of z-y-x indices to physical LPS coordinates."""
        indices_ijk = indices_zyx[:, ::-1].astype(np.float64, copy=False)
        scaled = indices_ijk * np.asarray(self._spacing, dtype=np.float64)
        direction = np.asarray(
            self._mask_image.GetDirection(),
            dtype=np.float64,
        ).reshape(3, 3)
        origin = np.asarray(self._origin, dtype=np.float64)
        return scaled @ direction.T + origin

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_available_vertebrae(self) -> List[Vertebra]:
        """Extract all vertebrae present in the mask.

        Returns a list of ``Vertebra`` objects sorted by label id
        (ascending).  Labels not found in the mask are silently skipped.
        """
        unique_labels = set(np.unique(self._mask_array).astype(int))
        vertebrae: List[Vertebra] = []

        for label_id in sorted(VERTEBRA_LABELS.keys()):
            if label_id not in unique_labels:
                continue

            name = VERTEBRA_LABELS[label_id]
            indices_zyx = np.argwhere(self._mask_array == label_id)  # (N, 3) z,y,x

            if indices_zyx.shape[0] == 0:
                continue

            # Build ijk index array (x, y, z) for the Vertebra dataclass.
            ijk = indices_zyx[:, ::-1].copy()  # flip to (x, y, z)

            # Centroid in LPS
            centroid = self._indices_centroid_lps(indices_zyx)

            # Bounding box in LPS
            min_zyx = indices_zyx.min(axis=0)
            max_zyx = indices_zyx.max(axis=0)
            bb_min = self._ijk_to_lps(int(min_zyx[2]), int(min_zyx[1]), int(min_zyx[0]))
            bb_max = self._ijk_to_lps(int(max_zyx[2]), int(max_zyx[1]), int(max_zyx[0]))
            # Ensure min <= max per axis after physical transform.
            bb_lo = np.minimum(bb_min, bb_max)
            bb_hi = np.maximum(bb_min, bb_max)

            # Volume (mm^3) = voxel_count * voxel_volume
            sx, sy, sz = self._spacing
            voxel_volume = sx * sy * sz
            volume_mm3 = float(indices_zyx.shape[0]) * voxel_volume

            vertebrae.append(
                Vertebra(
                    label=label_id,
                    name=name,
                    centroid_lps=centroid,
                    bounding_box=(bb_lo, bb_hi),
                    volume_mm3=volume_mm3,
                    mask_indices=ijk,
                )
            )

        return vertebrae

    # ------------------------------------------------------------------
    # Pedicle analysis
    # ------------------------------------------------------------------

    def analyze_pedicle(self, vertebra: Vertebra) -> PedicleAnalysisResult:
        """Analyze pedicle morphology for a single vertebra.

        Algorithm overview
        ------------------
        1. Extract the binary mask for this vertebra.
        2. Restrict to the **superior half** of the axial slice range
           (pedicles are located in the upper portion of each vertebra).
        3. For each axial slice in that range, run connected-component
           analysis and identify left / right pedicle regions by their
           x-coordinate relative to the vertebra centroid.
        4. Locate the *pedicle isthmus* — the slice with the minimum
           pedicle cross-sectional area.
        5. Compute the pedicle axis via PCA on the pedicle voxels near
           the isthmus.
        6. Measure minimum transverse pedicle width at the isthmus.
        7. Estimate the vertebral body centre from the anterior portion
           of the vertebra.
        """
        result = PedicleAnalysisResult(vertebra=vertebra)

        # Skip labels where pedicle analysis is not applicable.
        if vertebra.label in _SKIP_PEDICLE_LABELS:
            result.warnings.append(
                f"Pedicle analysis skipped for {vertebra.name} (not applicable)"
            )
            return result

        binary = (self._mask_array == vertebra.label).astype(np.uint8)

        # Determine the axial (z) range.
        z_indices = np.where(binary.any(axis=(1, 2)))[0]
        if z_indices.size == 0:
            result.warnings.append("No voxels found for vertebra mask")
            return result

        z_min, z_max = int(z_indices.min()), int(z_indices.max())

        if z_max - z_min < 1:
            result.warnings.append("Vertebra too thin in z-direction for pedicle analysis")
            return result

        # Collect left/right pedicle voxels across the full vertebra height.
        left_voxels_zyx: List[np.ndarray] = []
        right_voxels_zyx: List[np.ndarray] = []
        left_areas: Dict[int, float] = {}
        right_areas: Dict[int, float] = {}

        # Vertebra centroid in index space (x).
        indices_zyx = np.argwhere(binary)
        centroid_x_idx = float(indices_zyx[:, 2].mean())
        body_center = self._estimate_body_center(binary, indices_zyx)
        result.vertebral_body_center = body_center
        if body_center is not None:
            result.upper_endplate_normal = self._estimate_upper_endplate_normal(
                indices_zyx,
                body_center,
            )

        for z in range(z_min, z_max + 1):
            axial_slice = binary[z]
            if axial_slice.sum() == 0:
                continue

            labeled, n_components = ndi.label(axial_slice)
            if n_components < 2:
                # Need at least 2 components to separate body from pedicle.
                continue

            # Collect per-component info.
            components = []
            for comp_id in range(1, n_components + 1):
                comp_mask = labeled == comp_id
                comp_coords = np.argwhere(comp_mask)  # (N, 2) -> (y, x)
                area = comp_coords.shape[0]
                cx = float(comp_coords[:, 1].mean())
                components.append(
                    {"id": comp_id, "mask": comp_mask, "coords": comp_coords,
                     "area": area, "cx": cx}
                )

            # Sort by area descending — largest is typically the vertebral body.
            components.sort(key=lambda c: c["area"], reverse=True)

            # The vertebral body is the largest component whose centroid is
            # close to the overall centroid.  Pedicles are the smaller
            # components on each side.
            body_candidate = components[0]
            pedicle_candidates = components[1:]

            for pc in pedicle_candidates:
                # Minimum pedicle size guard (avoid noise).
                if pc["area"] < 3:
                    continue
                yx = pc["coords"]  # (N, 2) y,x
                zyx = np.column_stack([np.full(yx.shape[0], z), yx])  # (N, 3) z,y,x

                # Determine left/right using LPS coordinates
                # In LPS, +X = patient Left, -X = patient Right
                pc_center_lps = self._ijk_to_lps(
                    int(round(pc["cx"])),
                    int(round(float(yx[:, 0].mean()))),
                    z,
                )
                centroid_lps = self._ijk_to_lps(
                    int(round(centroid_x_idx)),
                    int(round(float(indices_zyx[:, 1].mean()))),
                    z,
                )
                if pc_center_lps[0] > centroid_lps[0]:
                    # +X in LPS = patient Left
                    left_voxels_zyx.append(zyx)
                    left_areas[z] = left_areas.get(z, 0) + pc["area"]
                else:
                    right_voxels_zyx.append(zyx)
                    right_areas[z] = right_areas.get(z, 0) + pc["area"]

        # Analyze each side independently.
        sx, sy, _ = self._spacing
        voxel_area_mm2 = sx * sy

        for side, voxels_list, areas_dict, set_center, set_axis, set_width in [
            ("left", left_voxels_zyx, left_areas,
             "_left_center", "_left_axis", "_left_width"),
            ("right", right_voxels_zyx, right_areas,
             "_right_center", "_right_axis", "_right_width"),
        ]:
            if not voxels_list:
                result.warnings.append(f"No {side} pedicle detected")
                continue

            all_voxels = np.vstack(voxels_list)  # (M, 3) z,y,x

            # --- Isthmus: slice with minimum cross-section area ---
            if areas_dict:
                isthmus_z = min(areas_dict, key=areas_dict.get)
            else:
                isthmus_z = int(np.median(all_voxels[:, 0]))

            # Isthmus centre in LPS — use 3-slice region around minimum
            isthmus_mask = np.abs(all_voxels[:, 0] - isthmus_z) <= 1
            isthmus_voxels = all_voxels[isthmus_mask]
            if isthmus_voxels.shape[0] == 0:
                isthmus_voxels = all_voxels

            isthmus_center = self._indices_centroid_lps(isthmus_voxels)

            # --- Pedicle axis via PCA ---
            # Use voxels within a few slices of the isthmus.
            margin = max(3, (z_max - z_min) // 3)
            near_mask = np.abs(all_voxels[:, 0] - isthmus_z) <= margin
            pca_voxels = all_voxels[near_mask]
            if pca_voxels.shape[0] < 3:
                pca_voxels = all_voxels

            axis = self._compute_pedicle_axis(pca_voxels)

            # --- Minimum transverse width ---
            width = self._measure_pedicle_width(isthmus_voxels, voxel_area_mm2)

            # Store results.
            if side == "left":
                result.left_pedicle_center = isthmus_center
                result.left_pedicle_axis = axis
                result.left_pedicle_width = width
            else:
                result.right_pedicle_center = isthmus_center
                result.right_pedicle_axis = axis
                result.right_pedicle_width = width

        if body_center is not None:
            crop_min = indices_zyx.min(axis=0)
            crop_max = indices_zyx.max(axis=0) + 1
            cropped_binary = binary[
                crop_min[0]:crop_max[0],
                crop_min[1]:crop_max[1],
                crop_min[2]:crop_max[2],
            ]
            cropped_distance = ndi.distance_transform_edt(
                cropped_binary,
                sampling=(self._spacing[2], self._spacing[1], self._spacing[0]),
            )
            local_indices = indices_zyx - crop_min
            clearances = cropped_distance[
                local_indices[:, 0],
                local_indices[:, 1],
                local_indices[:, 2],
            ]
            points_lps = self._indices_to_lps(indices_zyx)
            for side in ("left", "right"):
                fallback = self._estimate_connected_pedicle(
                    vertebra,
                    side,
                    points_lps,
                    clearances,
                    body_center,
                )
                if fallback is None:
                    continue
                center, axis, width = fallback
                if side == "left":
                    result.left_pedicle_center = center
                    result.left_pedicle_axis = axis
                    result.left_pedicle_width = width
                else:
                    result.right_pedicle_center = center
                    result.right_pedicle_axis = axis
                    result.right_pedicle_width = width
                result.warnings.append(
                    f"{side} pedicle estimated from 3D anatomical corridor"
                )

        # Mark success when at least one pedicle was found.
        if result.left_pedicle_center is not None or result.right_pedicle_center is not None:
            result.success = True

        return result

    def _estimate_connected_pedicle(
        self,
        vertebra: Vertebra,
        side: str,
        points_lps: np.ndarray,
        clearances: np.ndarray,
        body_center: np.ndarray,
    ) -> Optional[Tuple[np.ndarray, np.ndarray, float]]:
        """Estimate a pedicle corridor when the vertebra is one component."""
        bounds_min, bounds_max = vertebra.bounding_box
        extent = np.maximum(bounds_max - bounds_min, 1.0)
        side_sign = 1.0 if side == "left" else -1.0
        expected = np.array(
            [
                body_center[0] + side_sign * 0.28 * extent[0],
                body_center[1] + 0.30 * extent[1],
                body_center[2] + 0.05 * extent[2],
            ],
            dtype=np.float64,
        )
        radii = np.maximum(extent * np.array([0.22, 0.20, 0.30]), 3.0)
        offsets = (points_lps - expected) / radii
        side_mask = (
            points_lps[:, 0] > body_center[0]
            if side == "left"
            else points_lps[:, 0] < body_center[0]
        )
        posterior_mask = points_lps[:, 1] > body_center[1]
        candidate_mask = (
            side_mask
            & posterior_mask
            & (np.abs(offsets[:, 0]) <= 1.0)
            & (np.abs(offsets[:, 1]) <= 1.0)
            & (np.abs(offsets[:, 2]) <= 1.0)
        )
        candidate_indices = np.flatnonzero(candidate_mask)
        if candidate_indices.size == 0:
            return None

        candidate_clearances = clearances[candidate_indices]
        normalized_distance = np.linalg.norm(offsets[candidate_indices], axis=1)
        max_clearance = max(float(np.max(candidate_clearances)), 1.0)
        scores = (
            candidate_clearances
            - 0.4 * max_clearance * normalized_distance
        )
        best = candidate_indices[int(np.argmax(scores))]
        center = points_lps[best]
        clearance = float(clearances[best])
        axis = center - body_center
        axis_norm = float(np.linalg.norm(axis))
        if axis_norm <= 1e-9 or clearance <= 0.0:
            return None
        return center.copy(), axis / axis_norm, 2.0 * clearance

    def analyze_all(
        self, labels: Optional[List[int]] = None
    ) -> List[PedicleAnalysisResult]:
        """Analyze pedicles for all (or selected) vertebrae.

        Parameters
        ----------
        labels : list of int, optional
            Restrict analysis to the specified label ids.  When *None*,
            all vertebrae present in the mask are analyzed.
        """
        vertebrae = self.get_available_vertebrae()

        if labels is not None:
            label_set = set(labels)
            vertebrae = [v for v in vertebrae if v.label in label_set]

        results: List[PedicleAnalysisResult] = []
        for v in vertebrae:
            try:
                results.append(self.analyze_pedicle(v))
            except Exception as exc:  # noqa: BLE001
                logger.warning("Pedicle analysis failed for %s: %s", v.name, exc)
                failed = PedicleAnalysisResult(vertebra=v)
                failed.warnings.append(f"Analysis error: {exc}")
                results.append(failed)

        return results

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _compute_pedicle_axis(self, voxels_zyx: np.ndarray) -> np.ndarray:
        """Compute pedicle axis direction from voxel coordinates via PCA.

        Returns the first principal component as a unit vector in LPS
        coordinates.
        """
        # Convert index-space voxels to physical coordinates.
        n = voxels_zyx.shape[0]
        if n < 2:
            return np.array([1.0, 0.0, 0.0])

        # Sub-sample for efficiency when there are many voxels.
        max_samples = 2000
        if n > max_samples:
            rng = np.random.default_rng(42)
            idx = rng.choice(n, max_samples, replace=False)
            voxels_zyx = voxels_zyx[idx]

        pts = np.array(
            [
                self._ijk_to_lps(int(row[2]), int(row[1]), int(row[0]))
                for row in voxels_zyx
            ]
        )  # (N, 3)

        # Centre the points.
        centred = pts - pts.mean(axis=0)

        # PCA via SVD.
        _, _, vh = np.linalg.svd(centred, full_matrices=False)
        axis = vh[0]  # first principal component

        # Normalise (should already be unit, but be safe).
        norm = np.linalg.norm(axis)
        if norm < 1e-12:
            return np.array([1.0, 0.0, 0.0])
        return axis / norm

    @staticmethod
    def _measure_pedicle_width(
        isthmus_voxels_zyx: np.ndarray,
        voxel_area_mm2: float,
    ) -> float:
        """Estimate minimum transverse pedicle width at the isthmus.

        Uses a simple approach: take the 2-D cross-section of the
        pedicle in the isthmus slice, compute its area, and approximate
        the width as ``area / height`` where height is the bounding-box
        extent in the y (AP) direction.  Falls back to
        ``sqrt(area)`` when the extent is degenerate.
        """
        if isthmus_voxels_zyx.shape[0] == 0:
            return 0.0

        yx = isthmus_voxels_zyx[:, 1:]  # (N, 2) y, x
        area_mm2 = float(yx.shape[0]) * voxel_area_mm2

        y_extent = float(yx[:, 0].max() - yx[:, 0].min() + 1)
        x_extent = float(yx[:, 1].max() - yx[:, 1].min() + 1)

        if y_extent <= 0 or x_extent <= 0:
            return float(np.sqrt(area_mm2))

        # Width is the shorter bounding-box dimension in physical units.
        # This approximates the minimum transverse pedicle width.
        width_candidates = [
            area_mm2 / (y_extent * np.sqrt(voxel_area_mm2)),
            area_mm2 / (x_extent * np.sqrt(voxel_area_mm2)),
        ]
        return min(width_candidates)

    def _estimate_body_center(
        self, binary: np.ndarray, indices_zyx: np.ndarray
    ) -> Optional[np.ndarray]:
        """Estimate the vertebral body centre (anterior 2/3).

        In LPS, the **P** direction corresponds to +Y.  The vertebral
        body is *anterior* (lower Y in LPS / lower y-index for standard
        orientation).  We take the anterior 2/3 of the y-range and
        compute the centroid.
        """
        if indices_zyx.shape[0] == 0:
            return None

        y_vals = indices_zyx[:, 1]
        y_min, y_max = int(y_vals.min()), int(y_vals.max())
        y_range = y_max - y_min
        if y_range == 0:
            return self._indices_centroid_lps(indices_zyx)

        # Anterior 2/3 — lower y-indices.
        y_cutoff = y_min + int(round(y_range * 2.0 / 3.0))
        anterior_mask = indices_zyx[:, 1] <= y_cutoff
        anterior_voxels = indices_zyx[anterior_mask]

        if anterior_voxels.shape[0] == 0:
            return self._indices_centroid_lps(indices_zyx)

        return self._indices_centroid_lps(anterior_voxels)

    def _estimate_upper_endplate_normal(
        self,
        indices_zyx: np.ndarray,
        body_center: np.ndarray,
    ) -> Optional[np.ndarray]:
        """Fit the superior anterior-body envelope as an LPS plane."""
        if indices_zyx.shape[0] < 12:
            return None

        points = self._indices_to_lps(indices_zyx)
        extents = np.ptp(points, axis=0)
        if extents[0] < 4.0 or extents[1] < 4.0 or extents[2] < 2.0:
            return None

        body_mask = (
            (points[:, 1] <= body_center[1] + 0.15 * extents[1])
            & (np.abs(points[:, 0] - body_center[0]) <= 0.45 * extents[0])
        )
        body_points = points[body_mask]
        if body_points.shape[0] < 12:
            return None

        grid_size = max(2.0, 2.0 * min(self._spacing[0], self._spacing[1]))
        grid_cells = np.floor(body_points[:, :2] / grid_size).astype(np.int64)
        _, inverse = np.unique(grid_cells, axis=0, return_inverse=True)
        envelope = []
        for cell_index in range(int(inverse.max()) + 1):
            cell_points = body_points[inverse == cell_index]
            envelope.append(cell_points[int(np.argmax(cell_points[:, 2]))])
        samples = np.asarray(envelope, dtype=np.float64)
        if samples.shape[0] < 6:
            return None

        retained = np.ones(samples.shape[0], dtype=bool)
        coefficients = None
        for _ in range(3):
            fit_points = samples[retained]
            if fit_points.shape[0] < 6:
                return None
            design = np.column_stack(
                [fit_points[:, 0], fit_points[:, 1], np.ones(fit_points.shape[0])]
            )
            coefficients, *_ = np.linalg.lstsq(
                design,
                fit_points[:, 2],
                rcond=None,
            )
            predicted = (
                samples[:, 0] * coefficients[0]
                + samples[:, 1] * coefficients[1]
                + coefficients[2]
            )
            residuals = np.abs(samples[:, 2] - predicted)
            median = float(np.median(residuals[retained]))
            mad = float(np.median(np.abs(residuals[retained] - median)))
            threshold = max(1.5 * self._spacing[2], median + 3.0 * 1.4826 * mad)
            updated = residuals <= threshold
            if updated.sum() < 6 or np.array_equal(updated, retained):
                break
            retained = updated

        if coefficients is None:
            return None
        normal = np.array(
            [-coefficients[0], -coefficients[1], 1.0],
            dtype=np.float64,
        )
        norm = float(np.linalg.norm(normal))
        if norm <= 1e-9:
            return None
        normal /= norm
        if normal[2] < 0.0:
            normal = -normal
        return normal
