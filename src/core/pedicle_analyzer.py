"""
Pedicle morphology analyzer for vertebral segmentation masks.

Takes a TotalSegmentator multilabel mask (SimpleITK image) and produces
per-vertebra pedicle geometry: isthmus centre, axis direction, minimum
width, and vertebral body centre.  All output coordinates are in the
DICOM **LPS** coordinate system.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Set, Tuple

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

#: Endplate plane-fit RMS residual above which the fit is called rough (mm).
#: One millimetre is a single lumbar CT voxel; 1.5 mm is the point at which the
#: fitted plane can no longer be told apart from the stair steps of the mask it
#: was fitted to, and the endplate-parallel trajectory built on it deserves a
#: second look in the sagittal view.
ENDPLATE_FIT_RMSE_WARNING_MM: float = 1.5


def endplate_fit_warning(rmse_mm: float) -> str:
    """The single wording for a rough upper-endplate fit.

    Both the analyser and :mod:`src.core.auto_screw_planner` emit this text --
    the analyser onto the analysis, the planner onto the screw that used it --
    so it lives in one place and cannot drift between them.
    """
    return (
        f"Upper endplate fit is rough (RMSE {rmse_mm:.1f} mm) — "
        "check the sagittal view"
    )


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
    # Smallest coronal cross-section accepted as a pedicle.  Raised from the
    # original 10 mm2 because a nearest-neighbour-upsampled mask routinely
    # leaves a 12-15 mm2 fragment beside the real pedicle (spec W3: a 12.5 mm2
    # sliver at 8.8 mm lateral displacing a 99 mm2 pedicle at L2 right).
    MIN_PEDICLE_AREA_MM2: float = 20.0
    # Narrowest cross-section accepted as a pedicle candidate.  A pedicle is
    # never 1-2 voxels across; a stair step is.
    MIN_SLICE_WIDTH_MM: float = 2.0
    # Furthest a corridor's cross-section centroid may move between
    # neighbouring coronal slices and still be the same pedicle.  The pedicle
    # funnels gently; a jump this size is a different structure.
    MAX_TRACK_JUMP_MM: float = 4.0
    # Slices the corridor may go untracked before the walk ends.  A stair-step
    # notch can erase one or two cross-sections without ending the pedicle.
    MAX_TRACK_GAP_SLICES: int = 3
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
    # Shortest AP travel the centroid track may be fitted over.  On a
    # CT-guided refined mask (0.39 mm coronal voxels) the area-ratio window is
    # 3-4 records = 1.2-1.6 mm, over which a single 1.0 mm slice of stair-step
    # drift swings the SVD by more than 40 degrees.  Five millimetres is short
    # enough to stay inside a real pedicle's isthmus and long enough that one
    # voxel of drift is a small angle.
    MIN_AXIS_WINDOW_MM: float = 5.0
    # Largest cranio-caudal rise the fitted axis may have per unit of AP
    # travel.  Thoracolumbar pedicles run within about 25 degrees of the axial
    # plane; 0.7 (35 degrees) keeps the margin a lordotic level needs when the
    # slices are not square to the endplate, and still refuses the fits that
    # break the planner -- the sample CT's L2-left came back at 42 degrees
    # caudal, which no L2 pedicle does.  Such a fit is discarded in favour of
    # body centre -> isthmus.
    MAX_AXIS_TILT_RATIO: float = 0.7
    # Fewest labelled voxels on one side before that side is measured.  Below
    # this the label is a speck of leakage rather than a pedicle.
    MIN_LABEL_SIDE_VOXELS: int = 20
    # Plausible minimum transverse pedicle width per level class (mm).  A
    # measurement outside its band is a segmentation or measurement artefact,
    # not a narrow pedicle: the bands are wide enough to contain every real
    # thoracolumbar pedicle and every dysplastic one worth planning around.
    LUMBAR_WIDTH_RANGE_MM: Tuple[float, float] = (5.0, 22.0)
    THORACIC_WIDTH_RANGE_MM: Tuple[float, float] = (3.5, 18.0)

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
        # Index list of the label, resolved once: intersecting it with a
        # vertebra is then a gather over the labelled voxels alone, never a
        # full-volume boolean temporary per vertebra.
        self._pedicle_voxels: Optional[np.ndarray] = (
            None if pedicle_mask is None else np.argwhere(pedicle_mask)
        )

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
           pedicles.  Runs for any side the two passes above still left
           unmeasured, not only when they found nothing at all.

        ``result.method`` records the paths that produced the pedicle
        data, ``+``-joined in the order they ran.  ``result.success`` is
        set once *either* side is measured; a side no path could find is
        left ``None`` and named in ``result.warnings``, so a caller that
        needs both must check the two sides rather than ``success``.
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
            (
                result.upper_endplate_normal,
                result.endplate_fit_rmse_mm,
            ) = self._estimate_upper_endplate_normal(
                indices_zyx,
                body_center,
            )
            if (
                result.endplate_fit_rmse_mm is not None
                and result.endplate_fit_rmse_mm > ENDPLATE_FIT_RMSE_WARNING_MM
            ):
                result.warnings.append(
                    endplate_fit_warning(result.endplate_fit_rmse_mm)
                )
            body_center_ijk = self._mask_image.TransformPhysicalPointToContinuousIndex(
                tuple(float(v) for v in body_center)
            )

        # --- Preferred: measure the pedicle subregion label directly --------
        # Warnings from both searches are held back, tagged with the side they
        # concern, rather than recorded immediately: a side a later pass finds
        # anyway would only mislead the UI.  Only the ones still unresolved at
        # the end are reported.
        held_warnings: List[Tuple[str, str]] = []
        label_sides: List[str] = []
        if body_center_ijk is not None and self._pedicle_mask is not None:
            # Intersect the label with this vertebra once, not once per side.
            all_pedicle = self._pedicle_voxels
            inside = binary[all_pedicle[:, 0], all_pedicle[:, 1], all_pedicle[:, 2]] != 0
            pedicle_voxels = all_pedicle[inside]
            for side in ("left", "right"):
                found = self._find_pedicle_from_label(
                    pedicle_voxels, body_center_ijk, side
                )
                if found is None:
                    held_warnings.append(
                        (side, f"No {side} pedicle found in the pedicle subregion label")
                    )
                    continue
                label_sides.append(side)
                self._record_side(result, side, found)

        # --- Primary: coronal cross-section isthmus search ------------------
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
                    held_warnings.append(
                        (side, f"No {side} pedicle found by coronal isthmus search")
                    )
                    continue
                coronal_found = True
                self._record_side(result, side, found)

        # Name the paths that actually produced the recorded geometry; the axial
        # pass appends itself below if it contributes a side.
        methods: List[str] = []
        if label_sides:
            methods.append("subregion_label")
        if coronal_found:
            methods.append("coronal_isthmus")

        # A side neither path recorded still gets the axial search: half a
        # construct is not an answer, and until now one recorded side ended the
        # analysis for both.
        missing_sides = [
            side
            for side in ("left", "right")
            if self._recorded_center(result, side) is None
        ]
        if not missing_sides:
            self._report_held_warnings(result, held_warnings)
            result.method = "+".join(methods)
            result.success = True
            self._apply_plausibility_gate(result, binary)
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
        axial_found = False

        for side, voxels_list, areas_dict, _set_center, _set_axis, _set_width in [
            ("left", left_voxels_zyx, left_areas,
             "_left_center", "_left_axis", "_left_width"),
            ("right", right_voxels_zyx, right_areas,
             "_right_center", "_right_axis", "_right_width"),
        ]:
            if side not in missing_sides:
                continue        # already measured by the label or coronal path
            if not voxels_list:
                held_warnings.append((side, f"No {side} pedicle detected"))
                continue
            axial_found = True

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
            if (
                abs(float(axis[1])) < self.MIN_AXIS_AP_COMPONENT
                or self._is_overtilted(axis)
            ):
                # The axial pass gathers its cloud slice by slice, so a
                # stair-stepped or obliquely drawn corridor tips its principal
                # axis out of the AP direction as readily as the coronal fit --
                # and this path never had a guard at all.  Same answer as the
                # other two: body centre -> isthmus, oriented posteriorly.
                axis = self._body_centre_axis(isthmus_center, body_center)

            # --- Minimum transverse width ---
            width = self._measure_pedicle_width(isthmus_voxels, voxel_area_mm2)

            # Store results.  This path has only the one width estimate, so it
            # is its own lower bound -- leaving the bound at its 0.0 default
            # would read as a 0 mm pedicle rather than as "not cross-checked".
            if side == "left":
                result.left_pedicle_center = isthmus_center
                result.left_pedicle_axis = axis
                result.left_pedicle_width = width
                result.left_width_lower_bound_mm = width
            else:
                result.right_pedicle_center = isthmus_center
                result.right_pedicle_axis = axis
                result.right_pedicle_width = width
                result.right_width_lower_bound_mm = width

        if axial_found:
            methods.append("axial_components")

        still_missing = self._report_held_warnings(result, held_warnings)

        # Mark success when at least one pedicle was found.
        if len(still_missing) < 2:
            result.method = "+".join(methods)
            result.success = True

        # The gate runs last on both exits: it may append to `method`, so it
        # has to see the method the detection paths settled on.
        self._apply_plausibility_gate(result, binary)
        return result

    def _report_held_warnings(
        self,
        result: PedicleAnalysisResult,
        held_warnings: List[Tuple[str, str]],
    ) -> Set[str]:
        """Split the held-back misses into UI warnings and log-only provenance.

        A side every path failed on is named in ``result.warnings`` so the UI
        can say which half of the level is gone.  A miss a *later* pass repaired
        is not the surgeon's problem — warning about it would only mislead — but
        it is still provenance: a pedicle absent from the subregion label says
        something about the label, so it goes to the log rather than nowhere.

        Returns the sides still unmeasured.
        """
        still_missing = {
            side
            for side in ("left", "right")
            if self._recorded_center(result, side) is None
        }
        for side, message in held_warnings:
            if side in still_missing:
                result.warnings.append(message)
            else:
                logger.info(
                    "%s: %s (a later pass measured the %s side anyway)",
                    result.vertebra.name,
                    message,
                    side,
                )
        return still_missing

    @staticmethod
    def _recorded_center(
        result: PedicleAnalysisResult, side: str
    ) -> Optional[np.ndarray]:
        """This side's isthmus centre, or ``None`` if no path has produced one."""
        return (
            result.left_pedicle_center if side == "left" else result.right_pedicle_center
        )

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
            result.left_width_lower_bound_mm = found["width_lower_bound_mm"]
            result.left_pedicle_height = found["height_mm"]
            result.left_pedicle_inferior_medial_lps = found["inferior_medial_lps"]
        else:
            result.right_pedicle_center = found["center_lps"]
            result.right_pedicle_axis = found["axis_lps"]
            result.right_pedicle_width = found["width_mm"]
            result.right_width_lower_bound_mm = found["width_lower_bound_mm"]
            result.right_pedicle_height = found["height_mm"]
            result.right_pedicle_inferior_medial_lps = found["inferior_medial_lps"]

    def _inferior_medial_corner_lps(
        self,
        z_indices: np.ndarray,
        x_indices: np.ndarray,
        isthmus_j: int,
        side: str,
    ) -> np.ndarray:
        """LPS corner voxel of one isthmus slice: most inferior, then most medial.

        ``z_indices`` / ``x_indices`` are the slice's voxel indices.  The
        inferior end of the pedicle is the lowest ``z`` index and the medial
        side is the ``x`` index nearest the midline, which is the smaller ``x``
        for the patient's left (+X in LPS) and the larger one for the right —
        the same index-to-LPS orientation the rest of this module assumes.
        """
        z_min = int(np.min(z_indices))
        on_floor = x_indices[z_indices == z_min]
        x_medial = int(on_floor.min() if side == "left" else on_floor.max())
        return self._ijk_to_lps(x_medial, int(isthmus_j), z_min)

    def _find_pedicle_from_label(
        self,
        pedicle_voxels: np.ndarray,
        body_center_ijk: Tuple[float, float, float],
        side: str,
    ) -> Optional[Dict[str, object]]:
        """Measure one pedicle inside the supplied pedicle subregion label.

        The label already says which voxels are pedicle, so there is no
        posterior-wall / spinolaminar bracketing to do: the labelled
        corridor (already intersected with this vertebra by the caller) is
        split at the body centre, reduced to its largest connected
        component on this side (which drops leakage specks), and its
        narrowest coronal slice is the isthmus.

        The two end slices are ignored when picking that isthmus — they
        taper into the body and the lamina, so they are routinely the
        smallest cross-sections in the corridor without being the
        anatomical isthmus.  As in :meth:`_find_pedicle_coronal`, a
        corridor of uniform calibre ties across many slices, and the
        middle of the tied run is taken.

        Parameters
        ----------
        pedicle_voxels:
            ``(N, 3)`` array of ``(z, y, x)`` indices where the pedicle
            label and this vertebra's mask overlap.
        body_center_ijk:
            Vertebral body centre as a continuous ``(i, j, k)`` index.
        side:
            ``"left"`` (+X in LPS) or ``"right"`` (-X in LPS).

        Returns
        -------
        dict or None
            The same keys as :meth:`_find_pedicle_coronal`.  ``None``
            when this side holds too few labelled voxels to measure.
        """
        if pedicle_voxels.shape[0] == 0:
            return None

        sx, _, sz = self._spacing
        cx = float(body_center_ijk[0])

        lateral = (pedicle_voxels[:, 2] - cx) * (1.0 if side == "left" else -1.0)
        side_voxels = pedicle_voxels[lateral > 0]
        if side_voxels.shape[0] < self.MIN_LABEL_SIDE_VOXELS:
            return None

        # Largest connected component on this side (drops stray voxels).
        # Labelled inside the side's own bounding box rather than a
        # full-volume array: a whole-CT boolean would be hundreds of MB per
        # side per vertebra, and a tight crop holds every labelled voxel with
        # its neighbourhood intact, so the components come out identical.
        crop_lo = side_voxels.min(axis=0)
        local = side_voxels - crop_lo
        sub = np.zeros(tuple(int(n) for n in local.max(axis=0) + 1), dtype=bool)
        sub[local[:, 0], local[:, 1], local[:, 2]] = True
        labeled, n_components = ndi.label(sub)
        if n_components > 1:
            sizes = ndi.sum(sub, labeled, index=range(1, n_components + 1))
            largest = labeled == (int(np.argmax(sizes)) + 1)
            side_voxels = np.argwhere(largest) + crop_lo
            if side_voxels.shape[0] < self.MIN_LABEL_SIDE_VOXELS:
                return None

        js, counts = np.unique(side_voxels[:, 1], return_counts=True)
        areas = counts.astype(np.float64) * sx * sz
        widths = np.array(
            [
                float(
                    side_voxels[side_voxels[:, 1] == j][:, 2].max()
                    - side_voxels[side_voxels[:, 1] == j][:, 2].min()
                    + 1
                )
                * sx
                for j in js
            ],
            dtype=np.float64,
        )
        # Ignore the end slices that taper into body / lamina.
        interior = slice(1, -1) if js.size > 2 else slice(None)
        interior_js = js[interior]
        interior_areas = areas[interior]
        interior_widths = widths[interior]
        # A stair-stepped label leaves cross-sections narrower than any real
        # pedicle inside the corridor; they are not isthmus candidates.  When
        # the floors would reject every slice the corridor is uniformly thin
        # and the narrowest slice is still the best answer available.
        solid = (interior_areas >= self.MIN_PEDICLE_AREA_MM2) & (
            interior_widths >= self.MIN_SLICE_WIDTH_MM
        )
        candidate_areas = (
            np.where(solid, interior_areas, np.inf) if solid.any() else interior_areas
        )
        min_area = float(candidate_areas.min())
        tied = np.flatnonzero(candidate_areas == min_area)
        isthmus_j = int(interior_js[tied[tied.size // 2]])

        coords = side_voxels[side_voxels[:, 1] == isthmus_j]
        # Outer voxel-boundary extents, as in _find_pedicle_coronal, taken as
        # the median over the isthmus slice and its neighbours in the labelled
        # corridor and cross-checked against the inscribed diameter.
        isthmus_pos = int(np.flatnonzero(js == isthmus_j)[0])
        neighbour_js = js[max(0, isthmus_pos - 1):isthmus_pos + 2]
        neighbour_slices = [side_voxels[side_voxels[:, 1] == j] for j in neighbour_js]
        # `widths` already holds each slice's x-extent in mm, so the width
        # median is a slice of it rather than three more scans of side_voxels.
        width_extent_mm = float(
            np.median(widths[max(0, isthmus_pos - 1):isthmus_pos + 2])
        )
        height_mm = float(
            np.median(
                [
                    float(sl[:, 0].max() - sl[:, 0].min() + 1) * sz
                    for sl in neighbour_slices
                ]
            )
        )
        width_edt_mm = self._inscribed_width_mm(coords[:, [0, 2]], (sz, sx))
        width_mm = max(width_extent_mm, width_edt_mm)
        width_lower_bound_mm = min(width_extent_mm, width_edt_mm)
        center = self._continuous_ijk_to_lps(
            float(coords[:, 2].mean()),
            float(isthmus_j),
            float(coords[:, 0].mean()),
        )

        axis = self._compute_pedicle_axis(side_voxels)
        if abs(float(axis[1])) < self.MIN_AXIS_AP_COMPONENT or self._is_overtilted(axis):
            # A cloud whose principal axis is not AP enough to trust, or one
            # that climbs out of the axial plane faster than it advances — fall
            # back on body centre -> isthmus, as the coronal search does.
            axis = self._body_centre_axis(
                center, self._continuous_ijk_to_lps(*body_center_ijk)
            )
        if axis[1] < 0:
            axis = -axis

        return {
            "center_lps": center,
            "axis_lps": axis,
            "width_mm": width_mm,
            "width_lower_bound_mm": width_lower_bound_mm,
            "height_mm": height_mm,
            "isthmus_j": isthmus_j,
            "inferior_medial_lps": self._inferior_medial_corner_lps(
                coords[:, 0], coords[:, 2], isthmus_j, side
            ),
            # The axis is fitted over the whole labelled corridor, so the
            # window it covers is that corridor's full coronal span.
            "isthmus_window_j": (int(js[0]), int(js[-1])),
        }

    @classmethod
    def _width_range_for(cls, name: str) -> Optional[Tuple[float, float]]:
        """The plausible width band for a level, or ``None`` when it is not gated.

        Only the named lumbar and thoracic levels have a band.  The sacrum is
        skipped before analysis starts, and S1's "pedicle" is the sacral ala,
        whose width has nothing to do with a thoracolumbar pedicle -- gating it
        against a lumbar band would flag every normal S1.
        """
        level = str(name).strip().upper()
        if len(level) < 2 or not level[1:].isdigit():
            return None
        number = int(level[1:])
        if level[0] == "L" and 1 <= number <= 5:
            return cls.LUMBAR_WIDTH_RANGE_MM
        if level[0] == "T" and 1 <= number <= 12:
            return cls.THORACIC_WIDTH_RANGE_MM
        return None

    def _axial_recheck_width(
        self,
        binary: np.ndarray,
        result: PedicleAnalysisResult,
        side: str,
    ) -> Optional[float]:
        """Re-measure one pedicle's width on the axial slice through its isthmus.

        The axial route cuts the corridor in a direction the coronal walk never
        does, so a corridor the coronal search reduced to a stair-step sliver
        gets a genuinely independent second opinion rather than the same number
        computed twice.  The vertebral body is the largest component of the
        slice and is dropped, exactly as the axial fallback in
        :meth:`analyze_pedicle` does; the pedicle is whichever remaining
        component is nearest the recorded isthmus centre.

        Returns ``None`` when the slice holds nothing but the body -- a pedicle
        fused to the body in the axial plane has no second opinion to give, and
        an answer measured off the body would be worse than none.
        """
        center = self._recorded_center(result, side)
        if center is None:
            return None
        index = self._mask_image.TransformPhysicalPointToContinuousIndex(
            tuple(float(v) for v in center)
        )
        z = int(round(index[2]))
        if not 0 <= z < binary.shape[0]:
            return None
        axial_slice = binary[z]
        labeled, n_components = ndi.label(axial_slice)
        if n_components < 2:
            return None
        sizes = ndi.sum(axial_slice, labeled, index=range(1, n_components + 1))
        body_id = int(np.argmax(sizes)) + 1
        target_yx = np.array([float(index[1]), float(index[0])])
        best: Optional[Tuple[float, np.ndarray]] = None
        for comp_id in range(1, n_components + 1):
            if comp_id == body_id:
                continue
            coords = np.argwhere(labeled == comp_id)  # (n, 2) -> y, x
            distance = float(np.linalg.norm(coords.mean(axis=0) - target_yx))
            if best is None or distance < best[0]:
                best = (distance, coords)
        if best is None:
            return None
        voxels_zyx = np.column_stack([np.full(best[1].shape[0], z), best[1]])
        sx, sy, _ = self._spacing
        return float(self._measure_pedicle_width(voxels_zyx, sx * sy))

    def _apply_plausibility_gate(
        self,
        result: PedicleAnalysisResult,
        binary: np.ndarray,
    ) -> None:
        """Check each measured width against its level's plausible band.

        A width outside the band is a measurement failure rather than a narrow
        pedicle, so it is re-measured axially before it is believed.  A
        plausible second opinion replaces it and is recorded in
        :attr:`~src.core.vertebra.PedicleAnalysisResult.method`; when there is
        none, the original value is kept -- it is still the best number
        available -- and the side is flagged so the planner plans it under the
        narrow-pedicle policy and marks the level as uncertain.

        A replaced width takes its lower bound with it.  The bound belongs to
        the estimate that produced it, and the axial route has only the one
        estimate, so it becomes its own bound exactly as the axial fallback in
        :meth:`analyze_pedicle` does -- leaving the coronal bound in place
        would show a reviewer a floor higher than the width it sits under.
        """
        band = self._width_range_for(result.vertebra.name)
        if band is None:
            return
        lo, hi = band
        rechecked = False
        for side in ("left", "right"):
            if self._recorded_center(result, side) is None:
                continue
            width = (
                result.left_pedicle_width
                if side == "left"
                else result.right_pedicle_width
            )
            if lo <= width <= hi:
                continue
            second = self._axial_recheck_width(binary, result, side)
            if second is not None and lo <= second <= hi:
                logger.info(
                    "%s: %s pedicle width %.1f mm re-measured axially as %.1f mm",
                    result.vertebra.name,
                    side,
                    width,
                    second,
                )
                if side == "left":
                    result.left_pedicle_width = second
                    result.left_width_lower_bound_mm = second
                else:
                    result.right_pedicle_width = second
                    result.right_width_lower_bound_mm = second
                rechecked = True
                continue
            if side not in result.width_flags:
                result.warnings.append(
                    f"{side} pedicle width {width:.1f} mm outside the expected "
                    f"{lo}–{hi} mm — verify manually"
                )
            result.width_flags[side] = "implausible"
        if rechecked and "axial_recheck" not in result.method:
            result.method = (
                f"{result.method}+axial_recheck" if result.method else "axial_recheck"
            )

    @staticmethod
    def _inscribed_width_mm(
        coords_zx: np.ndarray,
        sampling: Tuple[float, float],
    ) -> float:
        """The largest inscribed diameter of one cross-section, in mm.

        This is a conservative *floor* under the reported width, never a
        rescue.  After the one-voxel correction below, twice the largest
        inscribed radius less one in-plane voxel can never exceed the isthmus
        slice's own x-extent, so ``max(extent, this)`` is the extent on every
        cross-section wider than it is tall; the inscribed diameter only speaks
        up on a section whose bounding box is narrower than the corridor
        genuinely inside it.  The max is moreover taken against the *median*
        extent over the isthmus slice and its two neighbours, not against the
        isthmus slice's own extent, so the EDT term can win only where the
        isthmus slice is the widest of the three.  What stops a 7 mm pedicle
        being reported as 1.6 mm is that neighbourhood median, together with
        the per-slice area and width floors and the continuity tracking that
        keeps the walk on the real corridor.

        ``coords_zx`` are the component's ``(z, x)`` voxel indices and
        ``sampling`` their ``(sz, sx)`` spacing.  The component is rasterised
        into its own bounding box padded by one background voxel on every
        side, so the transform measures the distance to the real boundary
        rather than to the edge of the array.
        """
        # The transform measures to the *centre* of the nearest background
        # voxel rather than to the boundary it shares with the foreground, so
        # twice the largest radius over-reads by one voxel.  Subtracting one
        # in-plane voxel makes an odd-voxel width exact and leaves an
        # even-voxel width one voxel short -- the safe direction to err in for
        # a planner, and harmless here because the reported width is
        # max(extent, this) and the extent still wins on regular sections.
        lo = coords_zx.min(axis=0)
        local = coords_zx - lo
        section = np.zeros(
            tuple(int(n) + 2 for n in local.max(axis=0) + 1), dtype=bool
        )
        section[local[:, 0] + 1, local[:, 1] + 1] = True
        distances = ndi.distance_transform_edt(section, sampling=sampling)
        return max(0.0, 2.0 * float(distances.max()) - sampling[1])

    @staticmethod
    def _track_candidate(
        candidates: List[Tuple[float, np.ndarray, float, float]],
        previous_zx_mm: Optional[Tuple[float, float]],
        max_jump_mm: float,
    ) -> Optional[np.ndarray]:
        """Pick this coronal slice's pedicle cross-section.

        The first slice of a corridor has nothing to follow, so it still takes
        the candidate nearest the midline -- the pedicle is medial to the
        transverse process and the facet.  Every slice after that follows the
        corridor by continuity instead: the candidate whose ``(z, x)`` centroid
        is closest to the previous record's, and only while that step stays
        under ``max_jump_mm``.  A medial fragment beside the real pedicle is
        then ignored rather than preferred, which is the failure the
        nearest-midline rule produces on a stair-stepped mask.

        ``candidates`` entries are ``(lateral_mm, coords_zx, centroid_z_mm,
        centroid_x_mm)``.  Returns the chosen ``coords_zx``, or ``None`` when
        this slice has no candidate within reach.
        """
        if not candidates:
            return None
        if previous_zx_mm is None:
            return min(candidates, key=lambda item: item[0])[1]
        best: Optional[Tuple[float, np.ndarray]] = None
        for _lateral_mm, coords, z_mm, x_mm in candidates:
            jump = float(np.hypot(z_mm - previous_zx_mm[0], x_mm - previous_zx_mm[1]))
            if jump > max_jump_mm:
                continue
            if best is None or jump < best[0]:
                best = (jump, coords)
        return None if best is None else best[1]

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
        * the **axis** is fitted over the contiguous run of records
          around the isthmus whose area stays within
          ``PEDICLE_WINDOW_AREA_RATIO`` of the minimum, which drops the
          laminar slices as soon as the cross-section starts to flare —
          widened, when that run is shorter than
          ``MIN_AXIS_WINDOW_MM`` of AP travel, until it is not.

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
            ``width_mm``, ``height_mm``, ``isthmus_j``,
            ``inferior_medial_lps`` (the isthmus slice's inferior-medial
            corner voxel, the CBT entry landmark) and
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
        previous_zx_mm: Optional[Tuple[float, float]] = None
        gap = 0

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

            candidates: List[Tuple[float, np.ndarray, float, float]] = []
            for comp_id in range(1, n_components + 1):
                coords = np.argwhere(labeled == comp_id)  # (n, 2) -> z, x
                if coords.shape[0] * sx * sz < self.MIN_PEDICLE_AREA_MM2:
                    continue
                width_mm = float(
                    coords[:, 1].max() - coords[:, 1].min() + 1
                ) * sx
                if width_mm < self.MIN_SLICE_WIDTH_MM:
                    continue          # a stair-step sliver, not a pedicle
                lateral_mm = (float(coords[:, 1].mean()) - cx) * sx * lateral_sign
                if not self.MIN_LATERAL_MM <= lateral_mm <= self.MAX_LATERAL_MM:
                    continue
                z_mean = float(coords[:, 0].mean())
                margin = self.Z_RANGE_MARGIN_MM / sz
                if not z_range[0] - margin <= z_mean <= z_range[1] + margin:
                    continue
                candidates.append(
                    (
                        lateral_mm,
                        coords,
                        z_mean * sz,
                        float(coords[:, 1].mean()) * sx,
                    )
                )

            chosen = self._track_candidate(
                candidates, previous_zx_mm, self.MAX_TRACK_JUMP_MM
            )
            if chosen is None:
                # A corridor that loses its cross-section for a slice or two is
                # still a corridor; one that loses it for longer has ended.
                if records:
                    gap += 1
                    if gap > self.MAX_TRACK_GAP_SLICES:
                        break
                continue
            gap = 0
            previous_zx_mm = (
                float(chosen[:, 0].mean()) * sz,
                float(chosen[:, 1].mean()) * sx,
            )
            records.append((j, chosen.shape[0] * sx * sz, chosen))

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
        # they over-read the underlying continuous extent by one voxel.  A
        # single slice of a stair-stepped mask is not trustworthy on its own,
        # so the extents are the median over the isthmus and its neighbours in
        # the recorded corridor (fewer at a corridor end).
        neighbourhood = records[max(0, isthmus_index - 1):isthmus_index + 2]
        width_extent_mm = float(
            np.median(
                [
                    float(rec[2][:, 1].max() - rec[2][:, 1].min() + 1) * sx
                    for rec in neighbourhood
                ]
            )
        )
        height_mm = float(
            np.median(
                [
                    float(rec[2][:, 0].max() - rec[2][:, 0].min() + 1) * sz
                    for rec in neighbourhood
                ]
            )
        )
        width_edt_mm = self._inscribed_width_mm(coords, (sz, sx))
        width_mm = max(width_extent_mm, width_edt_mm)
        width_lower_bound_mm = min(width_extent_mm, width_edt_mm)
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
        # A direction fitted over a millimetre of AP travel is noise: widen the
        # window until it spans a real length, whatever the areas say.
        lo_index, hi_index = self._extend_axis_window(records, lo_index, hi_index)

        axis = self._fit_axis_through_centroids(records[lo_index:hi_index + 1])
        if axis is None:
            # Too few slices, a corridor whose centroids do not track the AP
            # direction, or one that climbs out of the axial plane faster than
            # it advances — fall back on body centre -> isthmus.
            axis = self._body_centre_axis(
                center, self._continuous_ijk_to_lps(*body_center_ijk)
            )
        if axis[1] < 0:
            axis = -axis

        return {
            "center_lps": center,
            "axis_lps": axis,
            "width_mm": width_mm,
            "width_lower_bound_mm": width_lower_bound_mm,
            "height_mm": height_mm,
            "isthmus_j": int(isthmus_j),
            "inferior_medial_lps": self._inferior_medial_corner_lps(
                coords[:, 0], coords[:, 1], isthmus_j, side
            ),
            "isthmus_window_j": (int(records[lo_index][0]), int(records[hi_index][0])),
        }

    def _extend_axis_window(
        self,
        records: List[Tuple[int, float, np.ndarray]],
        lo_index: int,
        hi_index: int,
    ) -> Tuple[int, int]:
        """Widen ``records[lo_index:hi_index + 1]`` to :attr:`MIN_AXIS_WINDOW_MM`.

        The area-ratio window says which slices are *pedicle*; it says nothing
        about whether they are enough slices to fit a direction to.  On a fine
        coronal grid they routinely are not — three 0.39 mm slices span 1.2 mm,
        and one voxel of stair-step drift across that baseline tilts the fit by
        tens of degrees.

        Slices are added alternately in front of and behind the window, over
        the recorded corridor and ignoring the area ratio (they only steady the
        direction; the isthmus, width and height are already measured), until
        the window spans ``MIN_AXIS_WINDOW_MM`` of AP travel or the corridor
        runs out.  A corridor shorter than that is used whole.
        """
        _, sy, _ = self._spacing
        last = len(records) - 1
        extend_lo = True
        while (records[hi_index][0] - records[lo_index][0] + 1) * sy < (
            self.MIN_AXIS_WINDOW_MM
        ):
            if lo_index == 0 and hi_index == last:
                break                  # the whole corridor is shorter than that
            if extend_lo and lo_index > 0:
                lo_index -= 1
            elif hi_index < last:
                hi_index += 1
            elif lo_index > 0:
                # The posterior end is exhausted; keep widening towards the
                # body.  The break above means this arm always has room.
                lo_index -= 1
            extend_lo = not extend_lo
        return lo_index, hi_index

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
        if self._is_overtilted(axis):
            return None
        return axis

    @classmethod
    def _is_overtilted(cls, axis: np.ndarray) -> bool:
        """True when *axis* rises out of the axial plane by more than
        atan(MAX_AXIS_TILT_RATIO) -- about 35 deg for the current 0.7 ratio.

        A pedicle runs very nearly axially.  A fit that climbs faster than it
        advances is an artefact of a short or drifting centroid track, and
        following it lands the entry point a centimetre off the isthmus.
        """
        return abs(float(axis[2])) > cls.MAX_AXIS_TILT_RATIO * abs(float(axis[1]))

    @staticmethod
    def _body_centre_axis(
        isthmus_center_lps: np.ndarray,
        body_center_lps: Optional[np.ndarray],
    ) -> np.ndarray:
        """Body centre -> isthmus, as a posterior-oriented unit vector.

        The direction every path falls back on when its own fit is too short,
        too lateral or too steep to trust: it cannot be more than a rough
        estimate of the corridor, but it always points into the pedicle from
        in front of it.  Degenerates to ``+Y`` when there is no body centre or
        the isthmus sits on top of it.
        """
        if body_center_lps is None:
            return np.array([0.0, 1.0, 0.0])
        direction = (
            np.asarray(isthmus_center_lps, dtype=np.float64)
            - np.asarray(body_center_lps, dtype=np.float64)
        )
        norm = float(np.linalg.norm(direction))
        if norm <= 1e-9:
            return np.array([0.0, 1.0, 0.0])
        direction = direction / norm
        return -direction if direction[1] < 0 else direction

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
    ) -> Tuple[Optional[np.ndarray], Optional[float]]:
        """Fit the superior anterior-body envelope as an LPS plane.

        Returns ``(normal, rmse_mm)``.  ``rmse_mm`` is the RMS residual of the
        final fit over the samples it retained, so it describes the plane that
        was actually returned rather than the first, outlier-polluted pass.
        Both are ``None`` when no plane could be fitted at all -- a missing fit
        is not a perfect one.
        """
        if indices_zyx.shape[0] < 12:
            return None, None

        points = self._indices_to_lps(indices_zyx)
        extents = np.ptp(points, axis=0)
        if extents[0] < 4.0 or extents[1] < 4.0 or extents[2] < 2.0:
            return None, None

        body_mask = (
            (points[:, 1] <= body_center[1] + 0.15 * extents[1])
            & (np.abs(points[:, 0] - body_center[0]) <= 0.45 * extents[0])
        )
        body_points = points[body_mask]
        if body_points.shape[0] < 12:
            return None, None

        grid_size = max(2.0, 2.0 * min(self._spacing[0], self._spacing[1]))
        grid_cells = np.floor(body_points[:, :2] / grid_size).astype(np.int64)
        _, inverse = np.unique(grid_cells, axis=0, return_inverse=True)
        envelope = []
        for cell_index in range(int(inverse.max()) + 1):
            cell_points = body_points[inverse == cell_index]
            envelope.append(cell_points[int(np.argmax(cell_points[:, 2]))])
        samples = np.asarray(envelope, dtype=np.float64)
        if samples.shape[0] < 6:
            return None, None

        retained = np.ones(samples.shape[0], dtype=bool)
        coefficients = None
        for _ in range(3):
            fit_points = samples[retained]
            if fit_points.shape[0] < 6:
                return None, None
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
            return None, None
        predicted = (
            samples[:, 0] * coefficients[0]
            + samples[:, 1] * coefficients[1]
            + coefficients[2]
        )
        residuals = samples[:, 2] - predicted
        rmse = float(np.sqrt(np.mean(residuals[retained] ** 2)))
        normal = np.array(
            [-coefficients[0], -coefficients[1], 1.0],
            dtype=np.float64,
        )
        norm = float(np.linalg.norm(normal))
        if norm <= 1e-9:
            return None, None
        normal /= norm
        if normal[2] < 0.0:
            normal = -normal
        return normal, rmse
