"""
Automatic pedicle screw trajectory planning.

Uses pedicle analysis results to compute optimal entry/target points
for pedicle screws with safety validation.  All spatial calculations
operate in the DICOM **LPS** coordinate system:

    L (+X)  --  Patient's Left
    P (+Y)  --  Patient's Posterior
    S (+Z)  --  Patient's Superior

Key directions:
    Posterior  =  +Y
    Anterior   =  -Y
    Left       =  +X
    Right      =  -X
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import SimpleITK as sitk

from .screw_geometry import convergence_angle_deg, craniocaudal_angle_deg
from .screw_grading import ScrewGrader
from .vertebra import PedicleAnalysisResult
from ..utils.constants import (
    ANTERIOR_SAFETY_MARGIN_MM,
    CORTICAL_WALL_CLEARANCE_MM,
    IMPLANT_LENGTHS_MM,
    PEDICLE_FILL_RATIO,
)

logger = logging.getLogger(__name__)

# Sacrum label -- not suitable for pedicle screw planning.
_SACRUM_LABEL = 25


@dataclass
class PlannedScrew:
    """Result of automatic screw trajectory planning."""

    vertebra_name: str          # "L4", "T12", etc.
    side: str                   # "left" or "right"
    entry_lps: np.ndarray       # Entry point in LPS (3,)
    target_lps: np.ndarray      # Target point in LPS (3,)
    length_mm: float
    diameter_mm: float          # Recommended diameter based on pedicle width
    convergence_angle: float    # Medial angulation (degrees)
    craniocaudal_angle: float   # Sagittal angulation (degrees)
    mean_bone_density: float    # Average HU along trajectory
    min_bone_density: float     # Minimum HU along trajectory
    gertzbein_grade: str        # "A", "B", "C", "D", "E"
    confidence: float           # 0.0 -- 1.0
    breach_mm: float = 0.0      # Maximum cortical breach depth (mm)
    min_wall_mm: float = 0.0    # Thinnest cortical wall clearance (mm)
    warnings: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.entry_lps = np.asarray(self.entry_lps, dtype=np.float64)
        self.target_lps = np.asarray(self.target_lps, dtype=np.float64)


class AutoScrewPlanner:
    """Plan optimal pedicle screw trajectories from pedicle analysis results.

    Algorithm
    ---------
    1. Entry point: posterior surface along pedicle axis from isthmus centre.
    2. Target point: anterior vertebral body with safety margin.
    3. Trajectory optimization: medial convergence, craniocaudal alignment.
    4. Safety validation: HU sampling along path, breach detection.
    """

    # -- Screw sizing rules --------------------------------------------------
    MIN_SCREW_DIAMETER: float = 4.0    # mm
    MAX_SCREW_DIAMETER: float = 7.5    # mm
    DIAMETER_WIDE_HEADROOM: float = 1.0  # mm before one-step upsize

    MIN_SCREW_LENGTH: float = IMPLANT_LENGTHS_MM[0]
    MAX_SCREW_LENGTH: float = IMPLANT_LENGTHS_MM[-1]
    MAX_BONE_CORRIDOR_SCAN: float = 75.0  # mm, maximum anterior ray-cast distance
    ANTERIOR_SAFETY_MARGIN: float = ANTERIOR_SAFETY_MARGIN_MM

    # -- HU thresholds -------------------------------------------------------
    BONE_HU_MIN: float = 200          # Below this = outside bone
    CORTICAL_HU: float = 800          # Cortical bone threshold

    # -- Sampling parameters -------------------------------------------------
    TRAJECTORY_SAMPLE_STEP: float = 0.5   # mm between samples
    ENTRY_BACKOFF: float = 1.0            # mm back inside bone from surface
    TARGET_LATERAL_OFFSETS: Tuple[float, ...] = (
        0.0,
        3.0,
        6.0,
        9.0,
        12.0,
        15.0,
        18.0,
    )
    MAX_CONVERGENCE_ANGLE: float = 35.0

    def __init__(
        self,
        ct_image: sitk.Image,
        mask_image: sitk.Image,
        grader: Optional[ScrewGrader] = None,
    ) -> None:
        """
        Args:
            ct_image:  Original CT volume (for HU sampling).
            mask_image:  TotalSegmentator segmentation mask.
            grader:  Optional pre-built grader (defaults to one over this pair).
        """
        if tuple(ct_image.GetSize()) != tuple(mask_image.GetSize()):
            mask_image = sitk.Resample(
                mask_image,
                ct_image,
                sitk.Transform(),
                sitk.sitkNearestNeighbor,
                0,
                mask_image.GetPixelID(),
            )
        self._ct = ct_image
        self._mask = mask_image
        self._ct_array: np.ndarray = sitk.GetArrayFromImage(ct_image)   # (z, y, x)
        self._mask_array: np.ndarray = sitk.GetArrayFromImage(mask_image)
        self._grader = grader or ScrewGrader(mask_image, ct_image)

    # =====================================================================
    # Public API
    # =====================================================================

    def plan_screw(
        self,
        analysis: PedicleAnalysisResult,
        side: str,
    ) -> Optional[PlannedScrew]:
        """Plan a single pedicle screw for one side of one vertebra.

        Returns ``None`` if planning fails (e.g. no pedicle detected,
        sacrum, pedicle too narrow).
        """
        if side not in ("left", "right"):
            raise ValueError(f"side must be 'left' or 'right', got {side!r}")

        vertebra = analysis.vertebra

        # Sacrum is not suitable for pedicle screw placement.
        if vertebra.label == _SACRUM_LABEL:
            logger.info("Skipping sacrum (label %d) for screw planning", vertebra.label)
            return None

        if not analysis.success:
            logger.info("Skipping %s: pedicle analysis unsuccessful", vertebra.name)
            return None

        # Retrieve side-specific pedicle data.
        pedicle_center, pedicle_axis, pedicle_width = self._get_side_data(analysis, side)
        if pedicle_center is None or pedicle_axis is None:
            logger.info("No %s pedicle data for %s", side, vertebra.name)
            return None

        body_center = analysis.vertebral_body_center
        if body_center is None:
            logger.info("No vertebral body centre for %s", vertebra.name)
            return None

        warnings: List[str] = []
        if analysis.upper_endplate_normal is None:
            warnings.append(
                "Upper endplate unavailable; used horizontal sagittal trajectory"
            )

        # 1. Determine optimal screw diameter.
        diameter = self._compute_diameter(pedicle_width, vertebra.name)
        if diameter is None:
            logger.info(
                "Pedicle too narrow for any screw (%s %s, width=%.1f mm)",
                vertebra.name, side, pedicle_width,
            )
            return None

        # 2. Orient pedicle axis so it points posteriorly (+Y in LPS).
        #    Validate that PCA axis is roughly AP-directed (>30% Y component).
        #    Fall back to body_center→pedicle_center direction if PCA is unreliable.
        oriented_axis = self._orient_axis_posterior(pedicle_axis)
        ap_component = abs(oriented_axis[1])
        if ap_component < 0.3:
            # PCA axis not aligned with AP direction — use anatomical fallback
            fallback_dir = pedicle_center - body_center
            fb_norm = np.linalg.norm(fallback_dir)
            if fb_norm > 1e-9:
                oriented_axis = self._orient_axis_posterior(fallback_dir / fb_norm)
                warnings.append("PCA axis unreliable; used anatomical fallback direction")

        # 3. Find entry point on posterior bone surface.
        entry = self._find_entry_point(
            pedicle_center, oriented_axis, vertebra.label,
        )
        if entry is None:
            logger.info("Entry point search failed for %s %s", vertebra.name, side)
            return None

        # 4. Find target point in anterior vertebral body.
        target = self._find_best_target(
            entry,
            oriented_axis,
            body_center,
            vertebra.label,
            diameter,
            analysis.upper_endplate_normal,
        )
        if target is None:
            logger.info("Target point search failed for %s %s", vertebra.name, side)
            return None

        # 5. Enforce length constraints.
        length = float(np.linalg.norm(target - entry))
        if length < self.MIN_SCREW_LENGTH:
            logger.info(
                "Rejecting short trajectory for %s %s: %.1f mm",
                vertebra.name,
                side,
                length,
            )
            return None

        if length > self.MAX_SCREW_LENGTH:
            direction = target - entry
            direction /= np.linalg.norm(direction)
            target = entry + direction * self.MAX_SCREW_LENGTH
            length = self.MAX_SCREW_LENGTH

        initial_diameter = diameter
        grade, breach_dist = self._evaluate_gertzbein_grade(
            entry,
            target,
            diameter,
            vertebra.label,
        )
        while grade not in {"A", "B"}:
            diameter = round(diameter - 0.5, 1)
            if diameter < self.MIN_SCREW_DIAMETER:
                logger.info(
                    "No contained screw diameter for %s %s",
                    vertebra.name,
                    side,
                )
                return None
            candidate = self._find_best_target(
                entry,
                oriented_axis,
                body_center,
                vertebra.label,
                diameter,
                analysis.upper_endplate_normal,
            )
            if candidate is None:
                return None
            target = candidate
            length = float(np.linalg.norm(target - entry))
            grade, breach_dist = self._evaluate_gertzbein_grade(
                entry,
                target,
                diameter,
                vertebra.label,
            )
        if diameter < initial_diameter:
            warnings.append(
                f"Diameter reduced from {initial_diameter:.1f} to "
                f"{diameter:.1f} mm for cortical containment"
            )

        # 6. Grade the accepted trajectory once: containment plus HU statistics.
        result = self._grader.grade(entry, target, diameter, label=vertebra.label)
        if result is None:
            grade = "E"
            breach_dist = float(self._grader.crop_margin_mm)
            min_wall = 0.0
            mean_hu = min_hu = 0.0
        else:
            grade = result.grade
            breach_dist = result.breach_mm
            min_wall = result.min_wall_mm
            mean_hu = result.mean_hu if result.mean_hu is not None else 0.0
            min_hu = result.min_hu if result.min_hu is not None else 0.0

        # 7. Calculate angles.
        convergence_angle = self._compute_convergence_angle(entry, target, side)
        craniocaudal_angle = self._compute_craniocaudal_angle(entry, target)

        if convergence_angle > 30.0:
            warnings.append(
                f"High convergence angle {convergence_angle:.1f}° — verify on CT"
            )

        # 8. Calculate confidence score.
        confidence = self._calculate_confidence(grade, mean_hu, pedicle_width, diameter)

        if breach_dist > 0:
            warnings.append(f"Breach distance {breach_dist:.1f} mm (grade {grade})")

        if 0 < min_wall < CORTICAL_WALL_CLEARANCE_MM:
            warnings.append(
                f"Cortical clearance {min_wall:.1f} mm below "
                f"{CORTICAL_WALL_CLEARANCE_MM:.0f} mm"
            )

        return PlannedScrew(
            vertebra_name=vertebra.name,
            side=side,
            entry_lps=entry,
            target_lps=target,
            length_mm=length,
            diameter_mm=diameter,
            convergence_angle=convergence_angle,
            craniocaudal_angle=craniocaudal_angle,
            mean_bone_density=mean_hu,
            min_bone_density=min_hu,
            gertzbein_grade=grade,
            confidence=confidence,
            breach_mm=breach_dist,
            min_wall_mm=min_wall,
            warnings=warnings,
        )

    def plan_all(
        self,
        analyses: List[PedicleAnalysisResult],
        sides: str = "both",
    ) -> List[PlannedScrew]:
        """Plan screws for all analyzed vertebrae.

        Parameters
        ----------
        analyses:
            List of pedicle analysis results.
        sides:
            ``"both"`` (default), ``"left"``, or ``"right"``.
        """
        if sides not in ("both", "left", "right"):
            raise ValueError(f"sides must be 'both', 'left', or 'right', got {sides!r}")

        side_list = ["left", "right"] if sides == "both" else [sides]
        results: List[PlannedScrew] = []

        for analysis in analyses:
            for side in side_list:
                planned = self.plan_screw(analysis, side)
                if planned is not None:
                    results.append(planned)

        return results

    # =====================================================================
    # Entry / target point finding
    # =====================================================================

    def _find_entry_point(
        self,
        pedicle_center: np.ndarray,
        pedicle_axis_posterior: np.ndarray,
        vertebra_label: int,
    ) -> Optional[np.ndarray]:
        """Find entry point on posterior bone surface.

        Ray-cast from pedicle centre in BOTH directions along the
        pedicle axis to find the posterior bone-air boundary.
        Then back up ``ENTRY_BACKOFF`` mm inside bone for a safe entry.
        """
        step = self.TRAJECTORY_SAMPLE_STEP
        max_distance = 50.0  # mm -- safety limit
        n_steps = int(max_distance / step)

        # Try both posterior (+axis) and anterior (-axis) directions;
        # keep the one in the posterior direction (higher Y in LPS).
        best_entry = None
        best_y = -1e9

        for direction_sign in [1.0, -1.0]:
            direction = pedicle_axis_posterior * direction_sign
            last_bone_point = None
            in_bone = False

            for i in range(n_steps):
                distance = i * step
                point = pedicle_center + direction * distance
                if self._lps_to_index(point) is None:
                    break  # outside volume
                if self._is_point_inside_mask(point, vertebra_label):
                    last_bone_point = point.copy()
                    in_bone = True
                elif in_bone:
                    # Transition from bone to air — found the surface.
                    backoff_point = last_bone_point - direction * self.ENTRY_BACKOFF
                    if backoff_point[1] > best_y:
                        best_y = backoff_point[1]
                        best_entry = backoff_point
                    break
                # If not yet in bone, keep searching (skip air gaps)

        if best_entry is not None:
            return best_entry
        return None

    def _find_target_point(
        self,
        entry: np.ndarray,
        pedicle_axis_posterior: np.ndarray,
        body_center: np.ndarray,
        vertebra_label: int,
    ) -> Optional[np.ndarray]:
        """Find target point in anterior vertebral body.

        Follow the trajectory from entry toward the body centre
        (anteriorly).  Stop when exiting the vertebra mask or reaching
        the anterior cortex, then apply the anterior safety margin.
        """
        # Direction from entry to body centre.
        direction = body_center - entry
        dir_norm = np.linalg.norm(direction)
        if dir_norm < 1e-9:
            # Fallback: anterior direction is -Y in LPS.
            direction = np.array([0.0, -1.0, 0.0])
        else:
            direction = direction / dir_norm

        step = self.TRAJECTORY_SAMPLE_STEP
        max_distance = self.MAX_BONE_CORRIDOR_SCAN
        n_steps = int(max_distance / step)

        last_inside_point = entry.copy()

        for i in range(1, n_steps):
            distance = i * step
            point = entry + direction * distance
            idx = self._lps_to_index(point)
            if idx is None:
                # Outside volume bounds.
                break

            # Check if still inside the vertebra mask.
            z_idx, y_idx, x_idx = idx[2], idx[1], idx[0]
            if not self._is_in_bounds(x_idx, y_idx, z_idx):
                break

            mask_val = self._mask_array[z_idx, y_idx, x_idx]
            if mask_val != vertebra_label:
                # Exited the vertebra.
                break

            last_inside_point = point.copy()

        # Apply anterior safety margin by pulling back from the boundary.
        total_distance = float(np.linalg.norm(last_inside_point - entry))
        safe_distance = total_distance - self.ANTERIOR_SAFETY_MARGIN
        standard_length = self._select_standard_length(safe_distance)
        if standard_length is None:
            return None
        target = entry + direction * standard_length

        return target

    def _find_best_target(
        self,
        entry: np.ndarray,
        pedicle_axis_posterior: np.ndarray,
        body_center: np.ndarray,
        vertebra_label: int,
        diameter: float,
        upper_endplate_normal: Optional[np.ndarray] = None,
    ) -> Optional[np.ndarray]:
        """Search an ipsilateral body grid and retain the safest trajectory."""
        best_target: Optional[np.ndarray] = None
        best_score: Optional[Tuple[float, float, float]] = None
        side_sign = 1.0 if entry[0] >= body_center[0] else -1.0

        target_z = self._endplate_aligned_target_z(
            entry,
            float(body_center[1]),
            upper_endplate_normal,
        )
        for lateral_offset in self.TARGET_LATERAL_OFFSETS:
            candidate_anchor = np.array(
                [
                    body_center[0] + side_sign * lateral_offset,
                    body_center[1],
                    target_z,
                ],
                dtype=np.float64,
            )
            candidate = self._find_target_point(
                entry,
                pedicle_axis_posterior,
                candidate_anchor,
                vertebra_label,
            )
            if candidate is None:
                continue
            length = float(np.linalg.norm(candidate - entry))
            if not self.MIN_SCREW_LENGTH <= length <= self.MAX_SCREW_LENGTH:
                continue
            if side_sign * (candidate[0] - body_center[0]) < 0.0:
                continue
            convergence = self._compute_convergence_angle(
                entry,
                candidate,
                "left" if side_sign > 0.0 else "right",
            )
            if convergence > self.MAX_CONVERGENCE_ANGLE or convergence < -5.0:
                # Reject over-converging and laterally diverging candidates.
                continue
            _, breach_distance = self._evaluate_gertzbein_grade(
                entry,
                candidate,
                diameter,
                vertebra_label,
            )
            mean_hu, _, _ = self._sample_hu_along_trajectory(
                entry,
                candidate,
                diameter,
                label=vertebra_label,
            )
            score = (breach_distance, -mean_hu, -length)
            if best_score is None or score < best_score:
                best_score = score
                best_target = candidate

        return best_target

    # =====================================================================
    # HU sampling
    # =====================================================================

    def _sample_hu_along_trajectory(
        self,
        entry: np.ndarray,
        target: np.ndarray,
        diameter: float,
        label: Optional[int] = None,
    ) -> Tuple[float, float, List[float]]:
        """Sample HU statistics along the screw trajectory via the grader.

        ``label`` is auto-detected from the trajectory only when the caller
        does not already know which vertebra is being instrumented.

        Returns
        -------
        mean_hu : float
        min_hu : float
        all_samples : list of float
            Always empty; retained for signature compatibility.
        """
        if label is None:
            label = self._grader.detect_label(entry, target)
        result = self._grader.grade(entry, target, diameter, label=label)
        if result is None or result.mean_hu is None:
            return 0.0, 0.0, []
        return result.mean_hu, result.min_hu, []

    # =====================================================================
    # Gertzbein-Robbins evaluation
    # =====================================================================

    def _evaluate_gertzbein_grade(
        self,
        entry: np.ndarray,
        target: np.ndarray,
        diameter: float,
        vertebra_label: int,
    ) -> Tuple[str, float]:
        """Evaluate the Gertzbein-Robbins grade via :class:`ScrewGrader`.

        Returns
        -------
        grade : str
            "A" through "E".
        breach_distance_mm : float
            Maximum distance of the screw envelope outside the vertebra mask.
        """
        result = self._grader.grade(entry, target, diameter, label=int(vertebra_label))
        if result is None:
            return "E", float(self._grader.crop_margin_mm)
        return result.grade, result.breach_mm

    # =====================================================================
    # Angle calculations
    # =====================================================================

    def _compute_convergence_angle(
        self,
        entry: np.ndarray,
        target: np.ndarray,
        side: str,
    ) -> float:
        """Signed medial convergence angle (positive = toward the midline)."""
        return convergence_angle_deg(entry, target, side)

    def _compute_craniocaudal_angle(
        self,
        entry: np.ndarray,
        target: np.ndarray,
    ) -> float:
        """Elevation of the trajectory above the axial plane, positive cranial.

        This is the module-wide convention shared with ``Screw.insertion_angle``
        and the inspector.  It differs from a sagittal-projection angle for
        converging screws: the lateral component counts toward the horizontal
        run, so a converging screw's elevation is smaller than its projection
        onto the YZ plane.
        """
        return craniocaudal_angle_deg(entry, target)

    # =====================================================================
    # Confidence scoring
    # =====================================================================

    def _calculate_confidence(
        self,
        grade: str,
        mean_hu: float,
        pedicle_width: float,
        diameter: float,
    ) -> float:
        """Calculate confidence score in range [0.0, 1.0].

        Factors:
        - Gertzbein grade (dominant): A=0.6, B=0.4, C=0.2, D=0.1, E=0.0
        - Mean HU bone density contribution: 0.0 -- 0.2
        - Pedicle width margin: 0.0 -- 0.2
        """
        grade_scores = {"A": 0.60, "B": 0.40, "C": 0.20, "D": 0.10, "E": 0.0}
        base = grade_scores.get(grade, 0.0)

        # HU contribution: 200--800 maps to 0.0--0.2.
        hu_range = self.CORTICAL_HU - self.BONE_HU_MIN
        hu_frac = max(0.0, min(1.0, (mean_hu - self.BONE_HU_MIN) / hu_range)) if hu_range > 0 else 0.0
        hu_score = hu_frac * 0.20

        # Width margin: ratio of (pedicle_width - diameter) / diameter.
        if diameter > 0 and pedicle_width > 0:
            margin_ratio = (pedicle_width - diameter) / diameter
            margin_score = max(0.0, min(1.0, margin_ratio / 0.5)) * 0.20
        else:
            margin_score = 0.0

        confidence = min(1.0, base + hu_score + margin_score)
        return round(confidence, 3)

    # =====================================================================
    # Coordinate / sampling helpers
    # =====================================================================

    def _lps_to_index(self, point_lps: np.ndarray) -> Optional[Tuple[int, int, int]]:
        """Convert LPS world coordinate to (x, y, z) image index.

        Returns ``None`` if the point is outside the image volume.
        """
        try:
            idx = self._ct.TransformPhysicalPointToIndex(
                [float(point_lps[0]), float(point_lps[1]), float(point_lps[2])]
            )
        except RuntimeError:
            return None

        size = self._ct.GetSize()  # (x, y, z)
        if not (0 <= idx[0] < size[0] and 0 <= idx[1] < size[1] and 0 <= idx[2] < size[2]):
            return None

        return (int(idx[0]), int(idx[1]), int(idx[2]))

    def _get_hu_at_lps(self, point_lps: np.ndarray) -> Optional[float]:
        """Get HU value at an LPS coordinate.

        Returns ``None`` if the point is outside the volume.
        """
        idx = self._lps_to_index(point_lps)
        if idx is None:
            return None

        x, y, z = idx
        return float(self._ct_array[z, y, x])

    def _is_in_bounds(self, x: int, y: int, z: int) -> bool:
        """Check whether array index (x, y, z) is within the mask volume."""
        shape = self._mask_array.shape  # (Z, Y, X)
        return 0 <= z < shape[0] and 0 <= y < shape[1] and 0 <= x < shape[2]

    def _is_point_inside_mask(
        self,
        point_lps: np.ndarray,
        vertebra_label: int,
    ) -> bool:
        """Check if an LPS point is inside the specified vertebra mask."""
        idx = self._lps_to_index(point_lps)
        if idx is None:
            return False

        x, y, z = idx
        if not self._is_in_bounds(x, y, z):
            return False

        return int(self._mask_array[z, y, x]) == vertebra_label

    # =====================================================================
    # Private helpers
    # =====================================================================

    @staticmethod
    def _select_standard_length(safe_length: float) -> Optional[float]:
        """Select the longest catalogue implant that fits the safe corridor."""
        candidates = [length for length in IMPLANT_LENGTHS_MM if length <= safe_length + 1e-9]
        if not candidates:
            return None
        return float(max(candidates))

    @staticmethod
    def _endplate_aligned_target_z(
        entry: np.ndarray,
        target_y: float,
        upper_endplate_normal: Optional[np.ndarray],
    ) -> float:
        """Return target Z parallel to the endplate in the sagittal YZ plane."""
        if upper_endplate_normal is None:
            return float(entry[2])
        normal = np.asarray(upper_endplate_normal, dtype=np.float64)
        if normal.shape != (3,) or abs(float(normal[2])) <= 1e-9:
            return float(entry[2])
        sagittal_slope = -float(normal[1]) / float(normal[2])
        return float(entry[2]) + sagittal_slope * (
            float(target_y) - float(entry[1])
        )

    @staticmethod
    def _get_side_data(
        analysis: PedicleAnalysisResult,
        side: str,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], float]:
        """Extract (pedicle_center, pedicle_axis, pedicle_width) for a side."""
        if side == "left":
            return (
                analysis.left_pedicle_center,
                analysis.left_pedicle_axis,
                analysis.left_pedicle_width,
            )
        else:
            return (
                analysis.right_pedicle_center,
                analysis.right_pedicle_axis,
                analysis.right_pedicle_width,
            )

    def _compute_diameter(
        self,
        pedicle_width: float,
        vertebra_name: str,
    ) -> Optional[float]:
        """Combine a level preset with the measured safe pedicle capacity.

        The corridor is limited both by the pedicle fill ratio and by a
        cortical clearance on each side.  Returns ``None`` if the pedicle
        is too narrow for the smallest available screw.
        """
        available = min(
            PEDICLE_FILL_RATIO * pedicle_width,
            pedicle_width - 2.0 * CORTICAL_WALL_CLEARANCE_MM,
        )
        if available < self.MIN_SCREW_DIAMETER:
            return None
        available = min(
            self.MAX_SCREW_DIAMETER,
            math.floor(available * 2.0) / 2.0,
        )
        preferred, automatic_maximum = self._diameter_profile_for_level(
            vertebra_name
        )
        recommendation = preferred
        if available >= preferred + self.DIAMETER_WIDE_HEADROOM:
            recommendation = min(preferred + 0.5, automatic_maximum)
        return min(available, recommendation)

    @staticmethod
    def _diameter_profile_for_level(vertebra_name: str) -> Tuple[float, float]:
        level = str(vertebra_name).strip().upper()
        if level == "S1" or level in {"L3", "L4", "L5"}:
            return 6.5, 7.0
        if level in {"L1", "L2"}:
            return 6.0, 6.5
        if level.startswith("T") and level[1:].isdigit():
            return 5.5, 6.0
        return 6.0, 6.5

    @staticmethod
    def _orient_axis_posterior(pedicle_axis: np.ndarray) -> np.ndarray:
        """Ensure the pedicle axis points in the posterior (+Y) direction.

        If the axis Y-component is negative (points anteriorly), negate it.
        """
        if pedicle_axis[1] < 0:
            return -pedicle_axis
        return pedicle_axis.copy()
