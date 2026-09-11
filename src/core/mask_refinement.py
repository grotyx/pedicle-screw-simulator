"""Label-map refinement: anti-aliasing plus CT-guided boundary correction.

TotalSegmentator infers on a 1.5 mm grid and its multilabel output is upsampled
to the CT grid by nearest neighbour, so every vertebra reaches the app with
3-4 voxel stair steps at 0.39 mm in-plane spacing. Those steps are what make
the pedicle analyser measure 1.6 mm isthmus widths and what the 3D mesh shows
as facets. This module turns that blocky label map back into a
native-resolution mask whose boundary follows the CT cortex; every downstream
consumer (MPR overlay, 3D mesh, pedicle analyser, grader) picks it up simply by
reading the refined file.

Everything here is pure numpy/scipy on cropped boxes with explicit millimetre
spacing, so it is testable without Qt, VTK or file I/O.
"""

import logging
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import SimpleITK as sitk
from scipy import ndimage as ndi

from .screw_grading import ScrewGrader

logger = logging.getLogger(__name__)

#: ``RefinementResult.notes[0]`` is always one of these two summary notes, so a
#: caller can decide "was the CT actually used?" without parsing prose. The
#: segmentation status label and the plan metadata both read it.
CT_GUIDED_NOTE = "CT-guided boundary refinement applied."
ANTIALIAS_ONLY_NOTE = "Anti-alias only (no CT guidance)."

NO_CT_NOTE = "No CT supplied: anti-alias only, boundaries are not CT-guided."
GRID_MISMATCH_NOTE = (
    "CT and mask are not on the same voxel grid: anti-alias only, "
    "boundaries are not CT-guided."
)
NO_LABELS_NOTE = "No vertebra labels present; the mask was returned unchanged."


class RefinementCancelled(RuntimeError):
    """Raised when ``should_cancel`` asks the refinement to stop.

    Deliberately *not* the segmentation pipeline's ``SegmentationCancelled``:
    this module stays free of Qt and of the subprocess plumbing, so the caller
    (``totalseg_integration._apply_mask_refinement``) translates this into the
    pipeline's own cancellation type.
    """


def _raise_if_cancelled(should_cancel: Optional[Callable[[], bool]]) -> None:
    """Abort the refinement if the caller has asked it to stop."""
    if should_cancel is not None and should_cancel():
        raise RefinementCancelled("Mask refinement cancelled")


@dataclass(frozen=True)
class RefinementConfig:
    """Tuning for :func:`refine_vertebra_mask`. All lengths are millimetres."""

    #: Physical width of the Gaussian applied to each label's binary mask.
    #: 0.75 mm is half the 1.5 mm inference voxel, i.e. just enough to erase a
    #: one-voxel stair step without eating a real 2 mm pedicle wall.
    antialias_sigma_mm: float = 0.75
    ct_guided: bool = True
    #: Floor for "this voxel is bone". 200 HU sits below cortical bone and
    #: above the densest cancellous marrow in a lumbar spine CT.
    bone_threshold_hu: float = 200.0
    #: Half-width of the metric band around the anti-aliased boundary inside
    #: which the CT gets to overrule the label.
    band_mm: float = 1.5
    #: A CT-guided result that moves a label's volume by more than this
    #: fraction is discarded in favour of the anti-aliased label.
    max_volume_change: float = 0.30
    #: Labels to refine; ``None`` means "every VERTEBRA_LABELS id present".
    labels: Optional[Sequence[int]] = None


@dataclass(frozen=True)
class LabelStats:
    """What happened to one label."""

    label: int
    raw_voxels: int
    #: Voxel count after the cross-label competition clip (an earlier label
    #: in ``requested`` order wins any voxel two labels both claim).
    refined_voxels: int
    #: ``refined_voxels / raw_voxels``: the final, post-clip ratio.
    ratio: float
    #: The ratio the volume guard actually evaluated, i.e.
    #: ``candidate.sum() / raw_voxels`` for the CT-guided candidate *before*
    #: the cross-label competition clip. When no CT-guided candidate was
    #: computed at all (no CT, grid mismatch, or the label's anti-aliased
    #: mask was empty) there is nothing distinct to report and this equals
    #: ``ratio``. Kept separate from ``ratio`` because the guard's accept/
    #: reject decision (``change > max_volume_change``) is made on the
    #: pre-clip candidate, so the two can legitimately disagree when the
    #: clip later removes voxels another label claimed first.
    candidate_ratio: float
    ct_guided_applied: bool


@dataclass(frozen=True)
class RefinementResult:
    """Refined mask plus the audit trail for it."""

    mask: sitk.Image
    per_label: Dict[int, LabelStats] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)


def _vertebra_labels() -> Dict[int, str]:
    """The TotalSegmentator vertebra label map, imported on demand.

    ``src.core.vertebral_mesh`` imports VTK at module scope; importing it here
    lazily keeps this module (and therefore the segmentation worker thread)
    free of that dependency until the names are actually needed.
    """
    from .vertebral_mesh import VERTEBRA_LABELS

    return VERTEBRA_LABELS


def spacing_zyx(image: sitk.Image) -> np.ndarray:
    """Voxel spacing in numpy axis order ``(z, y, x)``, millimetres.

    SimpleITK reports spacing as ``(x, y, z)``; every array in this module is
    ``(z, y, x)``, and mixing the two on an anisotropic 0.39/0.39/1.0 mm volume
    is exactly the kind of bug that silently smooths the wrong axis.
    """
    size_x, size_y, size_z = (float(v) for v in image.GetSpacing())
    return np.array([size_z, size_y, size_x], dtype=np.float64)


def sigma_voxels(sigma_mm: float, spacing: np.ndarray) -> Tuple[float, float, float]:
    """A physical sigma expressed in voxels per axis."""
    per_z, per_y, per_x = float(sigma_mm) / np.asarray(spacing, dtype=np.float64)
    return (float(per_z), float(per_y), float(per_x))


def refinement_status_text(
    raw_mask_path: Optional[str], notes: Sequence[str]
) -> str:
    """One-line description of how the mask now in use was produced.

    Lives here rather than in the UI so the status label, the plan metadata and
    the tests all read the same fact off the same two values.
    """
    if not raw_mask_path:
        return "Raw mask"
    if CT_GUIDED_NOTE in notes:
        return "Refined (CT-guided)"
    return "Refined (anti-alias only)"


def _padded_box(
    binary: np.ndarray, pad: Sequence[int], shape: Sequence[int]
) -> Optional[Tuple[slice, ...]]:
    """Bounding box of ``binary`` grown by ``pad`` voxels per axis, clipped."""
    found = ndi.find_objects(binary.astype(np.uint8))
    if not found or found[0] is None:
        return None
    return tuple(
        slice(max(0, span.start - grow), min(int(dim), span.stop + grow))
        for span, grow, dim in zip(found[0], pad, shape, strict=True)
    )


def _boundary_band(
    binary: np.ndarray, band_mm: float, spacing: np.ndarray
) -> np.ndarray:
    """Voxels within ``band_mm`` millimetres of the boundary of ``binary``.

    The exact metric equivalent of ``dilate(band_mm) & ~erode(band_mm)``: one
    Euclidean distance transform each way, both sampled with the voxel
    spacing, so on an anisotropic grid the band is the same *physical* width
    on every axis and therefore fewer voxels deep along the coarse one. Doing
    it with a structuring element instead would be both slower and only
    approximately metric.
    """
    inside = ndi.distance_transform_edt(binary, sampling=spacing)
    outside = ndi.distance_transform_edt(~binary, sampling=spacing)
    return (inside <= band_mm) & (outside <= band_mm)


def _fill_and_largest_component(binary: np.ndarray) -> np.ndarray:
    """Fill fully enclosed cavities, then keep only the largest 3D component.

    The fill is 3D, and that is the whole point. Every crop here is padded by
    the band width plus 3 sigma, so the spinal canal is a tube open at both
    ends of the crop: its background runs out past the top and bottom of the
    vertebra to the crop border, which means it is not a 3D hole and is left
    alone. What does get filled is a cavity with no route out, which is what
    the speckle a HU threshold punches into the band looks like.

    Filling per axial slice instead would pack the canal on every slice the
    neural arch rings shut. This mask feeds the canal-breach grader, so a
    vertebra label containing the canal is exactly the input that makes a
    breach look like it is still inside bone.
    """
    # A vertebra cut by the volume edge is safe for the same reason: the canal
    # is still open at its other end. Only a tube sealed at BOTH ends inside
    # the crop would fill, and no real vertebra is.
    filled = ndi.binary_fill_holes(binary)
    labelled, count = ndi.label(filled)
    if count <= 1:
        return filled
    sizes = np.bincount(labelled.ravel())
    sizes[0] = 0
    return labelled == int(sizes.argmax())


#: Default config singleton, so the public signature can carry a
#: ``RefinementConfig`` default without ruff's B008 (call in a default
#: argument) — the dataclass is frozen, so sharing this one instance across
#: calls is safe.
_DEFAULT_CONFIG = RefinementConfig()


def refine_vertebra_mask(
    mask: sitk.Image,
    ct: Optional[sitk.Image] = None,
    config: RefinementConfig = _DEFAULT_CONFIG,
    progress: Optional[Callable[[str], None]] = None,
    should_cancel: Optional[Callable[[], bool]] = None,
) -> RefinementResult:
    """Anti-alias every vertebra label of ``mask``, optionally CT-guided.

    The soft masks of all requested labels compete: a voxel becomes the label
    with the highest soft value, and only if that value reaches 0.5. Two
    touching vertebrae therefore share a single surface instead of overlapping
    or both claiming the gap.

    ``mask`` is returned on its own grid as ``uint8`` (TotalSegmentator's label
    ids top out at 117, so the cast is lossless), with labels outside
    ``VERTEBRA_LABELS`` copied through untouched.

    ``should_cancel`` is polled once per label at the top of each of the two
    per-label passes; when it returns True the call raises
    :class:`RefinementCancelled` and returns nothing. A vertebra takes well
    under a second, so that granularity is what makes a Cancel press land
    within the run instead of after it — and it keeps each label's work atomic,
    so no half-refined mask is ever handed back.
    """
    notes: List[str] = []
    array = sitk.GetArrayFromImage(mask)
    spacing = spacing_zyx(mask)

    present = {int(value) for value in np.unique(array) if int(value) != 0}
    if config.labels is not None:
        requested = [int(value) for value in config.labels if int(value) in present]
    else:
        requested = sorted(present & set(_vertebra_labels()))

    use_ct = bool(config.ct_guided) and ct is not None
    if config.ct_guided and ct is None:
        notes.append(NO_CT_NOTE)
    elif use_ct and not ScrewGrader.grids_match(ct, mask):
        use_ct = False
        notes.append(GRID_MISMATCH_NOTE)
        logger.info("Mask refinement: CT grid differs from mask grid; anti-alias only")

    if not requested:
        notes.insert(0, ANTIALIAS_ONLY_NOTE)
        notes.append(NO_LABELS_NOTE)
        return RefinementResult(
            mask=sitk.Cast(mask, sitk.sitkUInt8), per_label={}, notes=notes
        )

    sigma = np.asarray(sigma_voxels(config.antialias_sigma_mm, spacing))
    # A voxel this far outside a label can still be pulled in, so every crop
    # must carry that much context or the box edge would clip the boundary.
    pad_mm = config.band_mm + 3.0 * config.antialias_sigma_mm
    pad = tuple(int(np.ceil(pad_mm / step)) + 1 for step in spacing)

    box = _padded_box(np.isin(array, requested), pad, array.shape)
    sub = array[box]
    best_value = np.zeros(sub.shape, dtype=np.float32)
    best_label = np.zeros(sub.shape, dtype=np.int16)

    names = _vertebra_labels()
    raw_counts: Dict[int, int] = {}
    boxes: Dict[int, Tuple[slice, ...]] = {}
    refinable: List[int] = []
    # Every requested label gets an entry, including one that is skipped
    # below, so a caller can iterate the audit trail without a membership test.
    per_label: Dict[int, LabelStats] = {}
    for index, label in enumerate(requested, start=1):
        _raise_if_cancelled(should_cancel)
        binary = sub == label
        raw_counts[label] = int(binary.sum())
        label_box = _padded_box(binary, pad, sub.shape)
        if label_box is None:
            # Defensive: a label that reached ``requested`` has voxels, so an
            # empty box means the crop and the label disagree. Leave those
            # voxels exactly as they are rather than silently deleting them.
            notes.append(
                f"{names.get(label, str(label))}: no voxels inside the crop; "
                "left unrefined."
            )
            logger.info("Mask refinement: label %d has an empty box; skipped", label)
            per_label[label] = LabelStats(
                label=label,
                raw_voxels=raw_counts[label],
                refined_voxels=raw_counts[label],
                ratio=1.0,
                candidate_ratio=1.0,
                ct_guided_applied=False,
            )
            continue
        boxes[label] = label_box
        refinable.append(label)
        if progress is not None:
            progress(
                f"Refining {names.get(label, str(label))} "
                f"({index}/{len(requested)})..."
            )
        # Each label is smoothed inside its own crop; the running argmax is the
        # only thing shared, so cost scales with the vertebrae, not the volume.
        # ``nearest`` and not ``constant``: where the crop stops at the volume
        # face there is no data, not air. Zero padding there would shave the
        # perimeter ring off the terminal slice of a vertebra the scan cuts
        # through. Where the crop stops early the pad is >= 3 sigma of
        # background anyway, so replicating it changes nothing.
        soft = ndi.gaussian_filter(
            binary[label_box].astype(np.float32),
            sigma=sigma,
            mode="nearest",
        )
        current_value = best_value[label_box]
        current_label = best_label[label_box]
        better = soft > current_value
        best_value[label_box] = np.where(better, soft, current_value)
        best_label[label_box] = np.where(better, np.int16(label), current_label)

    antialiased = np.where(best_value >= 0.5, best_label, 0).astype(np.int16)

    ct_sub = None
    if use_ct:
        # ``.copy()``: a basic-slice view would pin the whole CT array in
        # memory for the rest of the call, which on a 512x512x292 scan is
        # 150 MB held for the sake of one vertebra-sized crop.
        ct_sub = sitk.GetArrayFromImage(ct)[box].copy()

    out_sub = np.zeros(sub.shape, dtype=np.int16)
    for label in refinable:
        # The CT-guided pass is the expensive half (two distance transforms per
        # label), so a single-vertebra run would be uncancellable if only the
        # anti-alias loop above were polled.
        _raise_if_cancelled(should_cancel)
        label_box = boxes[label]
        raw_count = max(raw_counts[label], 1)
        smooth = antialiased[label_box] == label
        smooth_ratio = int(smooth.sum()) / raw_count
        if smooth_ratio < 0.5:
            # Typically a single voxel or a one-voxel sheet, which the
            # Gaussian never lifts to 0.5 anywhere. The verdict stands; the
            # note only records what happened, without diagnosing why.
            notes.append(
                f"{names.get(label, str(label))}: anti-aliasing kept "
                f"{smooth_ratio * 100:.0f}% of its voxels."
            )
            logger.info(
                "Mask refinement: label %d shrank to %.2f under anti-aliasing",
                label,
                smooth_ratio,
            )
        refined = smooth
        ct_guided_applied = False
        # Set only when a CT-guided candidate is actually computed below;
        # otherwise there is no pre-clip candidate distinct from the final
        # ratio, so it is filled in from ``ratio`` once that is known.
        candidate_ratio = None
        if use_ct and smooth.any():
            band = _boundary_band(smooth, config.band_mm, spacing)
            # Inside the band the CT decides, but only among voxels this label
            # already wins the soft-label argmax for: bone that belongs to the
            # neighbouring vertebra must not be stolen.
            candidate = np.where(
                band,
                (ct_sub[label_box] >= config.bone_threshold_hu)
                & (best_label[label_box] == label),
                smooth,
            )
            candidate = _fill_and_largest_component(candidate)
            candidate_count = int(candidate.sum())
            # This is the exact ratio the guard below judges, before the
            # cross-label competition clip further down can shrink it -- keep
            # it so the audit trail records what actually tripped the guard.
            candidate_ratio = candidate_count / raw_count
            change = abs(candidate_count - raw_count) / raw_count
            if change > config.max_volume_change:
                notes.append(
                    f"{names.get(label, str(label))}: CT-guided step changed volume "
                    f"by {change * 100:.0f}% "
                    f"(limit {config.max_volume_change * 100:.0f}%); "
                    "kept the anti-aliased boundary."
                )
                logger.info(
                    "Mask refinement: label %d exceeded the volume guard (%.2f)",
                    label,
                    change,
                )
            else:
                refined = candidate
                ct_guided_applied = True
        # The anti-aliased labels are an argmax and so already disjoint, but a
        # CT-guided boundary is grown per label and the hole fill can cross
        # into a neighbour that was written first. Earlier labels win.
        claim = refined & (out_sub[label_box] == 0)
        region = out_sub[label_box]
        region[claim] = label
        out_sub[label_box] = region
        refined_count = int(claim.sum())
        ratio = refined_count / raw_count
        per_label[label] = LabelStats(
            label=label,
            raw_voxels=raw_counts[label],
            refined_voxels=refined_count,
            ratio=ratio,
            candidate_ratio=ratio if candidate_ratio is None else candidate_ratio,
            ct_guided_applied=ct_guided_applied,
        )

    out = array.copy()
    region = out[box]
    region[np.isin(region, refinable)] = 0
    replace = (region == 0) & (out_sub != 0)
    region[replace] = out_sub[replace]
    out[box] = region

    refined_image = sitk.GetImageFromArray(out.astype(np.uint8))
    refined_image.CopyInformation(mask)
    notes.insert(
        0,
        CT_GUIDED_NOTE
        if any(stats.ct_guided_applied for stats in per_label.values())
        else ANTIALIAS_ONLY_NOTE,
    )
    return RefinementResult(mask=refined_image, per_label=per_label, notes=notes)
