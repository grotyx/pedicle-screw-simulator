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
    pedicle_mask : ndarray, optional
        Boolean ``(z, y, x)`` pedicle subregion label on the *same grid*
        as ``mask_image``.  When supplied it is measured directly instead
        of hunting for the isthmus geometrically; see
        :meth:`_find_pedicle_from_label`.
    """

    # -- Coronal isthmus search parameters -------------------------------
    # Half-width of the midline band that separates body/lamina slices
    # from pedicle-zone slices.
    MIDLINE_BAND_MM: float = 3.0
    # Smallest coronal cross-section accepted as a pedicle.
    MIN_PEDICLE_AREA_MM2: float = 10.0
    # Accepted lateral offset of a pedicle centroid from the midline.
    MIN_LATERAL_MM: float = 3.0
    MAX_LATERAL_MM: float = 40.0
    # Tolerance on the vertebra z-range when accepting a candidate.
    Z_RANGE_MARGIN_MM: float = 5.0
    # Below this |Y| the fitted axis is considered unreliable.
    MIN_AXIS_AP_COMPONENT: float = 0.3
    # A record still counts as pedicle (rather than flaring lamina) while its
    # area stays within this multiple of the isthmus area.  The pedicle/lamina
    # step is large (the anatomical phantom jumps 73 -> 106 mm2 the moment the
    # arch joins), so a quarter over the isthmus separates them with margin
    # while still spanning a real pedicle's gentle funnel.
    PEDICLE_WINDOW_AREA_RATIO: float = 1.25
    # Fewest window records needed to fit an axis through their centroids.
    MIN_AXIS_WINDOW_RECORDS: int = 3
    # Fewest labelled voxels on one side before that side is measured.  Below
    # this the label is a speck of leakage rather than a pedicle.
    MIN_LABEL_SIDE_VOXELS: int = 20

    def __init__(
        self,
        mask_image: sitk.Image,
        ct_image: Optional[sitk.Image] = None,
        pedicle_mask: Optional[np.ndarray] = None,
    ) -> None:
        self._mask_image = mask_image
        self._ct_image = ct_image

        # Pre-compute numpy array and geometry once.
        # sitk.GetArrayFromImage returns shape (z, y, x).
        self._mask_array: np.ndarray = sitk.GetArrayFromImage(mask_image)
        self._spacing: Tuple[float, float, float] = mask_image.GetSpacing()  # (sx, sy, sz)
        self._origin: Tuple[float, float, float] = mask_image.GetOrigin()

        if pedicle_mask is not None:
            pedicle_mask = np.asarray(pedicle_mask)
            if pedicle_mask.shape != self._mask_array.shape:
                raise ValueError(
                    "pedicle_mask shape "
                    f"{pedicle_mask.shape} does not match the vertebra mask "
                    f"{self._mask_array.shape}; both must be (z, y, x) on the "
                    "same grid"
                )
            pedicle_mask = pedicle_mask.astype(bool, copy=False)
        self._pedicle_mask: Optional[np.ndarray] = pedicle_mask

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

    def _continuous_ijk_to_lps(self, i: float, j: float, k: float) -> np.ndarray:
        """Convert a *continuous* ijk index to LPS physical coordinates."""
        pt = self._mask_image.TransformContinuousIndexToPhysicalPoint(
            (float(i), float(j), float(k))
        )
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
        return self._continuous_ijk_to_lps(mean_zyx[2], mean_zyx[1], mean_zyx[0])

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
        1. Extract the binary mask for this vertebra and estimate the
           vertebral body centre from its anterior portion.
        2. **Preferred** — pedicle subregion label
           (:meth:`_find_pedicle_from_label`): when a ``pedicle_mask``
           was supplied, measure the isthmus inside the labelled corridor
           instead of inferring where it is.
        3. **Primary** — coronal isthmus search
           (:meth:`_find_pedicle_coronal`): walk coronal slices
           posteriorly from the body centre, isolate the pedicle zone
           between the posterior body wall and the lamina, and take the
           narrowest cross-section on each side.  Runs only for the sides
           the label did not supply.
        4. **Fallback** — axial connected-component search: label each
           axial slice and treat the smaller lateral components as
           pedicles.  Used only when neither of the above finds a side
           (no body centre, or a mask with no pedicle zone).

        ``result.method`` records which path produced the pedicle data.
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
        body_center_ijk: Optional[Tuple[float, float, float]] = None
        if body_center is not None:
            result.upper_endplate_normal = self._estimate_upper_endplate_normal(
                indices_zyx,
                body_center,
            )
            body_center_ijk = self._mask_image.TransformPhysicalPointToContinuousIndex(
                tuple(float(v) for v in body_center)
            )

        # --- Preferred: measure the pedicle subregion label directly --------
        # Warnings from both searches are held back rather than recorded
        # immediately: if the axial fallback then succeeds these would only
        # mislead the UI.
        label_warnings: List[str] = []
        label_sides: List[str] = []
        if body_center_ijk is not None and self._pedicle_mask is not None:
            for side in ("left", "right"):
                found = self._find_pedicle_from_label(binary, body_center_ijk, side)
                if found is None:
                    label_warnings.append(
                        f"No {side} pedicle found in the pedicle subregion label"
                    )
                    continue
                label_sides.append(side)
                self._record_side(result, side, found)

        # --- Primary: coronal cross-section isthmus search ------------------
        coronal_warnings: List[str] = []
        coronal_found = False
        if body_center_ijk is not None:
            for side in ("left", "right"):
                if side in label_sides:
                    continue
                found = self._find_pedicle_coronal(
                    binary,
                    body_center_ijk,
                    (z_min, z_max),
                    side,
                )
                if found is None:
                    coronal_warnings.append(
                        f"No {side} pedicle found by coronal isthmus search"
                    )
                    continue
                coronal_found = True
                self._record_side(result, side, found)

        if label_sides or coronal_found:
            result.warnings.extend(label_warnings)
            result.warnings.extend(coronal_warnings)
            # Name the paths that actually produced the recorded geometry.
            if not label_sides:
                result.method = "coronal_isthmus"
            elif coronal_found:
                result.method = "subregion_label+coronal_isthmus"
            else:
                result.method = "subregion_label"
            result.success = True
            return result

        # --- Fallback: axial connected-component search ---------------------
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

        # Mark success when at least one pedicle was found.
        if result.left_pedicle_center is not None or result.right_pedicle_center is not None:
            result.method = "axial_components"
            result.success = True
        else:
            result.warnings.extend(label_warnings)
            result.warnings.extend(coronal_warnings)

        return result

    @staticmethod
    def _record_side(
        result: PedicleAnalysisResult,
        side: str,
        found: Dict[str, object],
    ) -> None:
        """Copy one ``_find_pedicle_*`` record onto *result* for *side*."""
        if side == "left":
            result.left_pedicle_center = found["center_lps"]
            result.left_pedicle_axis = found["axis_lps"]
            result.left_pedicle_width = found["width_mm"]
            result.left_pedicle_height = found["height_mm"]
        else:
            result.right_pedicle_center = found["center_lps"]
            result.right_pedicle_axis = found["axis_lps"]
            result.right_pedicle_width = found["width_mm"]
            result.right_pedicle_height = found["height_mm"]

    def _find_pedicle_from_label(
        self,
        binary: np.ndarray,
        body_center_ijk: Tuple[float, float, float],
        side: str,
    ) -> Optional[Dict[str, object]]:
        """Measure one pedicle inside the supplied pedicle subregion label.

        The label already says which voxels are pedicle, so there is no
        posterior-wall / spinolaminar bracketing to do: the labelled
        corridor is intersected with this vertebra, split at the body
        centre, reduced to its largest connected component on this side
        (which drops leakage specks), and its narrowest coronal slice is
        the isthmus.

        The two end slices are ignored when picking that isthmus — they
        taper into the body and the lamina, so they are routinely the
        smallest cross-sections in the corridor without being the
        anatomical isthmus.  As in :meth:`_find_pedicle_coronal`, a
        corridor of uniform calibre ties across many slices, and the
        middle of the tied run is taken.

        Parameters
        ----------
        binary:
            Vertebra mask as a ``(z, y, x)`` array.
        body_center_ijk:
            Vertebral body centre as a continuous ``(i, j, k)`` index.
        side:
            ``"left"`` (+X in LPS) or ``"right"`` (-X in LPS).

        Returns
        -------
        dict or None
            The same keys as :meth:`_find_pedicle_coronal`.  ``None``
            when no label was supplied, or when this side holds too few
            labelled voxels to measure.
        """
        if self._pedicle_mask is None:
            return None

        sx, _, sz = self._spacing
        cx = float(body_center_ijk[0])
        voxels = np.argwhere(binary.astype(bool) & self._pedicle_mask)  # z, y, x
        if voxels.shape[0] == 0:
            return None

        lateral = (voxels[:, 2] - cx) * (1.0 if side == "left" else -1.0)
        side_voxels = voxels[lateral > 0]
        if side_voxels.shape[0] < self.MIN_LABEL_SIDE_VOXELS:
            return None

        # Largest connected component on this side (drops stray voxels).
        sub = np.zeros(binary.shape, dtype=bool)
        sub[side_voxels[:, 0], side_voxels[:, 1], side_voxels[:, 2]] = True
        labeled, n_components = ndi.label(sub)
        if n_components > 1:
            sizes = ndi.sum(sub, labeled, index=range(1, n_components + 1))
            sub = labeled == (int(np.argmax(sizes)) + 1)
            side_voxels = np.argwhere(sub)
            if side_voxels.shape[0] < self.MIN_LABEL_SIDE_VOXELS:
                return None

        js, counts = np.unique(side_voxels[:, 1], return_counts=True)
        areas = counts.astype(np.float64) * sx * sz
        # Ignore the end slices that taper into body / lamina.
        interior = slice(1, -1) if js.size > 2 else slice(None)
        interior_js = js[interior]
        interior_areas = areas[interior]
        min_area = float(interior_areas.min())
        tied = np.flatnonzero(interior_areas == min_area)
        isthmus_j = int(interior_js[tied[tied.size // 2]])

        coords = side_voxels[side_voxels[:, 1] == isthmus_j]
        # Outer voxel-boundary extents, as in _find_pedicle_coronal.
        width_mm = float(coords[:, 2].max() - coords[:, 2].min() + 1) * sx
        height_mm = float(coords[:, 0].max() - coords[:, 0].min() + 1) * sz
        center = self._continuous_ijk_to_lps(
            float(coords[:, 2].mean()),
            float(isthmus_j),
            float(coords[:, 0].mean()),
        )

        axis = self._compute_pedicle_axis(side_voxels)
        if abs(float(axis[1])) < self.MIN_AXIS_AP_COMPONENT:
            # A cloud whose principal axis is not AP enough to trust — fall
            # back on body centre -> isthmus, as the coronal search does.
            body_center_lps = self._continuous_ijk_to_lps(*body_center_ijk)
            fallback = center - body_center_lps
            norm = float(np.linalg.norm(fallback))
            if norm > 1e-9:
                axis = fallback / norm
        if axis[1] < 0:
            axis = -axis

        return {
            "center_lps": center,
            "axis_lps": axis,
            "width_mm": width_mm,
            "height_mm": height_mm,
            "isthmus_j": isthmus_j,
            # The axis is fitted over the whole labelled corridor, so the
            # window it covers is that corridor's full coronal span.
            "isthmus_window_j": (int(js[0]), int(js[-1])),
        }

    def _find_pedicle_coronal(
        self,
        binary: np.ndarray,
        body_center_ijk: Tuple[float, float, float],
        z_range: Tuple[int, int],
        side: str,
    ) -> Optional[Dict[str, object]]:
        """Locate one pedicle by scanning coronal cross-sections.

        The scan walks posteriorly from the vertebral body centre.  It
        leaves the body at the first slice behind the posterior body
        wall where nothing reaches the midline, and stops at the
        spinolaminar junction — the first slice after that where
        something reaches the midline again.  Everything in between is
        recorded, which is the pedicle *and*, while the laminae stay
        clear of the midline, the anterior part of the laminar arch:
        the recorded run is not pedicle-only.

        The two measurements are taken from different subsets of it:

        * the **isthmus** is the single narrowest recorded slice, and
          supplies the centre, width and height;
        * the **axis** is fitted only over the contiguous run of records
          around the isthmus whose area stays within
          ``PEDICLE_WINDOW_AREA_RATIO`` of the minimum, which drops the
          laminar slices as soon as the cross-section starts to flare.

        Parameters
        ----------
        binary:
            Vertebra mask as a ``(z, y, x)`` array.
        body_center_ijk:
            Vertebral body centre as a continuous ``(i, j, k)`` index.
        z_range:
            Inclusive ``(z_min, z_max)`` index range of the vertebra.
        side:
            ``"left"`` (+X in LPS) or ``"right"`` (-X in LPS).

        Returns
        -------
        dict or None
            Keys ``center_lps``, ``axis_lps`` (posterior-oriented),
            ``width_mm``, ``height_mm``, ``isthmus_j`` and
            ``isthmus_window_j`` (the inclusive ``(j_lo, j_hi)`` slice
            range the axis was fitted over); ``None`` when fewer than
            two pedicle cross-sections were found.
        """
        sx, _, sz = self._spacing
        cx = float(body_center_ijk[0])
        j_start = int(round(body_center_ijk[1]))
        lateral_sign = 1.0 if side == "left" else -1.0
        band = max(1, int(round(self.MIDLINE_BAND_MM / sx)))
        records: List[Tuple[int, float, np.ndarray]] = []
        body_ended = False

        for j in range(j_start, binary.shape[1]):
            coronal = binary[:, j, :]  # (z, x)
            if not coronal.any():
                if body_ended and records:
                    break
                continue

            labeled, n_components = ndi.label(coronal)
            band_center = int(round(cx))
            lo = max(0, band_center - band)
            hi = min(coronal.shape[1], band_center + band + 1)
            midline_ids = set(np.unique(labeled[:, lo:hi])) - {0}
            if not body_ended:
                if midline_ids:
                    continue          # still inside the vertebral body
                body_ended = True     # posterior body wall passed
            elif midline_ids:
                break                 # lamina / spinous process reached

            best: Optional[Tuple[float, np.ndarray]] = None
            for comp_id in range(1, n_components + 1):
                coords = np.argwhere(labeled == comp_id)  # (n, 2) -> z, x
                if coords.shape[0] * sx * sz < self.MIN_PEDICLE_AREA_MM2:
                    continue
                lateral_mm = (float(coords[:, 1].mean()) - cx) * sx * lateral_sign
                if not self.MIN_LATERAL_MM <= lateral_mm <= self.MAX_LATERAL_MM:
                    continue
                z_mean = float(coords[:, 0].mean())
                margin = self.Z_RANGE_MARGIN_MM / sz
                if not z_range[0] - margin <= z_mean <= z_range[1] + margin:
                    continue
                # The pedicle is the candidate nearest the midline; anything
                # further lateral is transverse process or facet.
                if best is None or lateral_mm < best[0]:
                    best = (lateral_mm, coords)
            if best is not None:
                records.append((j, best[1].shape[0] * sx * sz, best[1]))

        if len(records) < 2:
            return None

        # Isthmus = narrowest cross-section.  A corridor of uniform calibre
        # ties across several slices, so take the middle of the tied
        # minimum-area records (which need not be contiguous).
        min_area = min(area for _, area, _ in records)
        tied_indices = [i for i, record in enumerate(records) if record[1] == min_area]
        isthmus_index = tied_indices[len(tied_indices) // 2]
        isthmus_j, _, coords = records[isthmus_index]

        # Extents are outer voxel-boundary extents (max - min + 1 voxels), so
        # they over-read the underlying continuous extent by one voxel.
        width_mm = float(coords[:, 1].max() - coords[:, 1].min() + 1) * sx
        height_mm = float(coords[:, 0].max() - coords[:, 0].min() + 1) * sz
        center = self._continuous_ijk_to_lps(
            float(coords[:, 1].mean()),
            float(isthmus_j),
            float(coords[:, 0].mean()),
        )

        # Pedicle window: the contiguous run of records around the isthmus
        # whose cross-section has not yet flared into the laminar arch.
        area_limit = min_area * self.PEDICLE_WINDOW_AREA_RATIO
        lo_index = isthmus_index
        while lo_index > 0 and records[lo_index - 1][1] <= area_limit:
            lo_index -= 1
        hi_index = isthmus_index
        while hi_index + 1 < len(records) and records[hi_index + 1][1] <= area_limit:
            hi_index += 1
        window = records[lo_index:hi_index + 1]

        axis = self._fit_axis_through_centroids(window)
        if axis is None:
            # Too few slices, or a corridor whose centroids do not track the
            # AP direction — fall back on body centre -> isthmus.
            body_center_lps = self._continuous_ijk_to_lps(*body_center_ijk)
            fallback = center - body_center_lps
            norm = float(np.linalg.norm(fallback))
            axis = (
                fallback / norm if norm > 1e-9 else np.array([0.0, 1.0, 0.0])
            )
        if axis[1] < 0:
            axis = -axis

        return {
            "center_lps": center,
            "axis_lps": axis,
            "width_mm": width_mm,
            "height_mm": height_mm,
            "isthmus_j": int(isthmus_j),
            "isthmus_window_j": (int(records[lo_index][0]), int(records[hi_index][0])),
        }

    def _fit_axis_through_centroids(
        self,
        window: List[Tuple[int, float, np.ndarray]],
    ) -> Optional[np.ndarray]:
        """Fit the pedicle axis to the per-slice centroids of *window*.

        The centroid track follows the corridor's own AP course, unlike a
        PCA over the voxel cloud, which is dominated by whichever axis of
        the cross-section happens to be widest.  Returns ``None`` when the
        window is too short or the fitted direction is not AP enough to
        trust.
        """
        if len(window) < self.MIN_AXIS_WINDOW_RECORDS:
            return None

        centroids_zyx = np.array(
            [
                [
                    float(rec_coords[:, 0].mean()),
                    float(rec_j),
                    float(rec_coords[:, 1].mean()),
                ]
                for rec_j, _, rec_coords in window
            ],
            dtype=np.float64,
        )
        points = self._indices_to_lps(centroids_zyx)
        centred = points - points.mean(axis=0)
        _, _, vh = np.linalg.svd(centred, full_matrices=False)
        axis = vh[0]
        norm = float(np.linalg.norm(axis))
        if norm <= 1e-12:
            return None
        axis = axis / norm
        if abs(float(axis[1])) < self.MIN_AXIS_AP_COMPONENT:
            return None
        return axis

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
