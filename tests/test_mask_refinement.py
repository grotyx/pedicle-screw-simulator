"""Tests for src/core/mask_refinement.py.

The phantom is a smooth analytic shape (a vertebral-body cylinder plus a
transverse pedicle bar) sampled twice: once on a fine 0.5 mm grid, which is the
ground truth and the CT, and once on a coarse 3.0 mm grid that is then
nearest-neighbour upsampled by exactly 6x back onto the fine grid. That
upsampled label map is the stair-stepped input TotalSegmentator actually hands
the app, one coarse voxel deep in steps. No test here touches `data/sample`.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

pytest.importorskip("SimpleITK")

import SimpleITK as sitk

from src.core.mask_refinement import (
    ANTIALIAS_ONLY_NOTE,
    CT_GUIDED_NOTE,
    GRID_MISMATCH_NOTE,
    NO_CT_NOTE,
    NO_LABELS_NOTE,
    RefinementConfig,
    refine_vertebra_mask,
    refinement_status_text,
    sigma_voxels,
    spacing_zyx,
)

FINE_SPACING_MM = 0.5
COARSE_SPACING_MM = 3.0
UPSAMPLE = 6                          # COARSE_SPACING_MM / FINE_SPACING_MM
COARSE_SHAPE = (8, 12, 12)            # (z, y, x) -> 24 x 36 x 36 mm
FINE_SHAPE = tuple(n * UPSAMPLE for n in COARSE_SHAPE)
BODY_LABEL = 28                       # vertebrae_L4
SECOND_LABEL = 29                     # vertebrae_L3
OTHER_LABEL = 90                      # not a vertebra: must pass through
BONE_HU = 400
SOFT_HU = 40


def _voxel_centres(shape, spacing):
    """(zz, yy, xx) millimetre coordinate arrays of every voxel centre."""
    axes = [(np.arange(n) + 0.5) * spacing for n in shape]
    return np.meshgrid(*axes, indexing="ij")


def _true_shape(zz, yy, xx):
    """The smooth ground truth: a body cylinder plus a transverse pedicle bar."""
    cx = cy = 18.0
    cz = 12.0
    body = ((xx - cx) ** 2 + (yy - cy) ** 2) <= 8.0 ** 2
    pedicles = (
        (((yy - (cy + 6.0)) ** 2 + (zz - cz) ** 2) <= 3.0 ** 2)
        & (np.abs(xx - cx) <= 16.0)
    )
    return body | pedicles


def truth_mask():
    """Ground-truth boolean mask on the fine grid."""
    return _true_shape(*_voxel_centres(FINE_SHAPE, FINE_SPACING_MM))


def stair_stepped_label(label=BODY_LABEL):
    """The coarse shape upsampled 6x by nearest neighbour, as a label map."""
    coarse = _true_shape(*_voxel_centres(COARSE_SHAPE, COARSE_SPACING_MM))
    fine = coarse
    for axis in range(3):
        fine = np.repeat(fine, UPSAMPLE, axis=axis)
    return fine.astype(np.uint8) * np.uint8(label)


def as_image(array, spacing=FINE_SPACING_MM):
    image = sitk.GetImageFromArray(array)
    image.SetSpacing((spacing, spacing, spacing))
    return image


def bone_ct():
    """CT with bone HU inside the true smooth shape, soft tissue outside."""
    return as_image(np.where(truth_mask(), BONE_HU, SOFT_HU).astype(np.int16))


def label_of(result, label):
    return sitk.GetArrayFromImage(result.mask) == label


def dice(a, b):
    a = np.asarray(a, dtype=bool)
    b = np.asarray(b, dtype=bool)
    return 2.0 * float((a & b).sum()) / float(a.sum() + b.sum())


def boundary_voxels(a):
    """Count of voxels of `a` that have at least one background 6-neighbour.

    A stair-stepped surface exposes more such voxels than a smooth surface of
    the same volume, so this is the stair-step proxy the spec asks for.
    """
    a = np.asarray(a, dtype=bool)
    padded = np.pad(a, 1)
    interior = (
        padded[:-2, 1:-1, 1:-1] & padded[2:, 1:-1, 1:-1]
        & padded[1:-1, :-2, 1:-1] & padded[1:-1, 2:, 1:-1]
        & padded[1:-1, 1:-1, :-2] & padded[1:-1, 1:-1, 2:]
    )
    return int((a & ~interior).sum())


def test_spacing_and_sigma_are_anisotropy_aware():
    """0.39/0.39/1.0 mm must give a smaller sigma along z, in voxels."""
    image = sitk.Image([4, 4, 4], sitk.sitkUInt8)
    image.SetSpacing((0.39, 0.39, 1.0))

    spacing = spacing_zyx(image)
    assert spacing == pytest.approx([1.0, 0.39, 0.39])

    sigma = sigma_voxels(0.75, spacing)
    assert sigma == pytest.approx((0.75, 0.75 / 0.39, 0.75 / 0.39))


def test_refinement_status_text_names_the_three_states():
    assert refinement_status_text(None, []) == "Raw mask"
    assert refinement_status_text("raw.nii.gz", [CT_GUIDED_NOTE]) == "Refined (CT-guided)"
    assert (
        refinement_status_text("raw.nii.gz", [ANTIALIAS_ONLY_NOTE, NO_CT_NOTE])
        == "Refined (anti-alias only)"
    )


# --- (e) ct=None -> anti-aliased labels only, with a note -------------------

def test_no_ct_returns_antialiased_labels_with_a_note():
    raw = stair_stepped_label()

    result = refine_vertebra_mask(as_image(raw), None)

    assert result.notes[0] == ANTIALIAS_ONLY_NOTE
    assert NO_CT_NOTE in result.notes
    stats = result.per_label[BODY_LABEL]
    assert stats.ct_guided_applied is False
    assert stats.raw_voxels == int((raw > 0).sum())
    # Anti-aliasing alone already removes stair steps and keeps the volume.
    assert boundary_voxels(label_of(result, BODY_LABEL)) < boundary_voxels(raw > 0)
    assert stats.ratio == pytest.approx(0.99, abs=0.05)


# --- (f) grid mismatch is refused with a note, not an exception -------------

def test_grid_mismatch_is_refused_with_a_note_not_an_exception():
    raw = stair_stepped_label()
    shifted = bone_ct()
    shifted.SetOrigin((5.0, 0.0, 0.0))          # far past any grid tolerance

    result = refine_vertebra_mask(as_image(raw), shifted)
    antialias_only = refine_vertebra_mask(as_image(raw), None)

    assert result.notes[0] == ANTIALIAS_ONLY_NOTE
    assert GRID_MISMATCH_NOTE in result.notes
    assert result.per_label[BODY_LABEL].ct_guided_applied is False
    assert np.array_equal(
        sitk.GetArrayFromImage(result.mask),
        sitk.GetArrayFromImage(antialias_only.mask),
    )


def test_ct_guided_off_skips_guidance_even_with_a_matching_ct():
    raw = stair_stepped_label()

    result = refine_vertebra_mask(
        as_image(raw), bone_ct(), RefinementConfig(ct_guided=False)
    )

    assert result.notes[0] == ANTIALIAS_ONLY_NOTE
    assert result.per_label[BODY_LABEL].ct_guided_applied is False


def test_non_vertebra_labels_pass_through_unchanged():
    raw = stair_stepped_label()
    raw[0] = 0
    raw[0][20:30, 20:30] = OTHER_LABEL

    result = refine_vertebra_mask(as_image(raw), None)
    out = sitk.GetArrayFromImage(result.mask)

    assert int((out == OTHER_LABEL).sum()) == int((raw == OTHER_LABEL).sum())
    assert OTHER_LABEL not in result.per_label
    assert sorted(int(v) for v in np.unique(out)) == [0, BODY_LABEL, OTHER_LABEL]


def test_a_mask_without_vertebra_labels_is_returned_unchanged():
    raw = np.zeros(FINE_SHAPE, np.uint8)
    raw[0][20:30, 20:30] = OTHER_LABEL

    result = refine_vertebra_mask(as_image(raw), None)

    assert result.notes[0] == ANTIALIAS_ONLY_NOTE
    assert NO_LABELS_NOTE in result.notes
    assert result.per_label == {}
    assert np.array_equal(sitk.GetArrayFromImage(result.mask), raw)


def test_result_mask_keeps_the_grid_and_is_uint8():
    result = refine_vertebra_mask(as_image(stair_stepped_label()), None)

    assert result.mask.GetSize() == (FINE_SHAPE[2], FINE_SHAPE[1], FINE_SHAPE[0])
    assert result.mask.GetSpacing() == pytest.approx(
        (FINE_SPACING_MM, FINE_SPACING_MM, FINE_SPACING_MM)
    )
    assert result.mask.GetPixelID() == sitk.sitkUInt8


def test_progress_callback_names_every_label():
    messages = []
    raw = stair_stepped_label()

    refine_vertebra_mask(as_image(raw), None, progress=messages.append)

    assert any("vertebrae_L4" in message for message in messages)


def two_label_phantom():
    """The phantom split at mid-height into two touching vertebrae."""
    raw = stair_stepped_label()
    half = FINE_SHAPE[0] // 2
    two = raw.copy()
    two[:half][two[:half] > 0] = BODY_LABEL
    two[half:][two[half:] > 0] = SECOND_LABEL
    return two


# --- (g) anti-alias alone must not let two labels fight over the interface ---

def test_two_labels_stay_disjoint_without_ct():
    """Anti-aliasing alone: one id per voxel and neither label melts away."""
    two = two_label_phantom()

    result = refine_vertebra_mask(as_image(two), None)
    out = sitk.GetArrayFromImage(result.mask)

    assert sorted(int(v) for v in np.unique(out)) == [0, BODY_LABEL, SECOND_LABEL]
    # A label map cannot store two ids in one voxel, so "disjoint" is the claim
    # that the two label counts add up to the whole foreground: nothing was
    # written twice and nothing was dropped in the hand-over.
    body = int((out == BODY_LABEL).sum())
    second = int((out == SECOND_LABEL).sum())
    assert body + second == int((out > 0).sum())
    for label in (BODY_LABEL, SECOND_LABEL):
        assert result.per_label[label].ratio >= 0.90


# --- (h) the volume edge is unknown, not background -------------------------

def test_a_label_cut_by_the_volume_edge_keeps_its_terminal_slice():
    """Outside the field of view is unknown, not air.

    A Gaussian padded with zeros shaves the perimeter ring off the terminal
    slice, which is exactly the slice a surgeon scrolls to at the top of a
    scan.
    """
    raw = np.zeros(FINE_SHAPE, np.uint8)
    raw[0:12, 20:52, 20:52] = BODY_LABEL

    result = refine_vertebra_mask(as_image(raw), None)
    out = label_of(result, BODY_LABEL)

    kept = int(out[0].sum()) / int((raw[0] > 0).sum())
    assert kept >= 0.95


# --- (i) a label the smoothing kernel swallows is named in the notes --------

def test_a_label_smaller_than_the_kernel_is_named_in_the_notes():
    """A one-voxel label genuinely is noise, but the audit trail must say so."""
    raw = np.zeros(FINE_SHAPE, np.uint8)
    raw[24, 36, 36] = BODY_LABEL

    result = refine_vertebra_mask(as_image(raw), None)

    assert result.per_label[BODY_LABEL].ratio < 0.5
    assert any(
        note.startswith("vertebrae_L4: anti-aliasing kept") for note in result.notes
    )


# --- (a) refined Dice beats raw Dice by >= 0.03 -----------------------------

def test_ct_guided_refinement_improves_dice_against_the_true_shape():
    raw = stair_stepped_label()
    truth = truth_mask()

    result = refine_vertebra_mask(as_image(raw), bone_ct())

    raw_dice = dice(raw > 0, truth)
    refined_dice = dice(label_of(result, BODY_LABEL), truth)
    assert result.notes[0] == CT_GUIDED_NOTE
    assert result.per_label[BODY_LABEL].ct_guided_applied is True
    assert refined_dice - raw_dice >= 0.03


# --- (b) fewer stair steps -------------------------------------------------

def test_ct_guided_refinement_removes_stair_steps():
    raw = stair_stepped_label()

    result = refine_vertebra_mask(as_image(raw), bone_ct())

    assert boundary_voxels(label_of(result, BODY_LABEL)) < boundary_voxels(raw > 0)


# --- (c) the volume guard ---------------------------------------------------

def test_all_bone_ct_trips_the_guard_and_keeps_the_antialiased_mask():
    """Every voxel bone means the band would swallow a 1.5 mm shell.

    Without the guard the label grows ~45 %, which is how a bad CT window or a
    metal artefact would silently inflate a vertebra.
    """
    raw = stair_stepped_label()
    all_bone = as_image(np.full(FINE_SHAPE, BONE_HU, np.int16))

    guarded = refine_vertebra_mask(as_image(raw), all_bone)
    antialias_only = refine_vertebra_mask(as_image(raw), None)

    assert guarded.notes[0] == ANTIALIAS_ONLY_NOTE
    assert guarded.per_label[BODY_LABEL].ct_guided_applied is False
    assert any("changed volume" in note for note in guarded.notes)
    assert np.array_equal(
        sitk.GetArrayFromImage(guarded.mask),
        sitk.GetArrayFromImage(antialias_only.mask),
    )


# --- (d) two touching labels stay disjoint ----------------------------------

def test_two_touching_labels_stay_disjoint_and_keep_their_volume():
    """The upper and lower halves of the phantom are two vertebrae.

    The CT is bone across the interface, so without the "soft-label argmax is
    this label" condition the first label's band eats ~3 voxel layers of the
    second (measured: it drops to 77 % of its raw volume).
    """
    two = two_label_phantom()
    half = FINE_SHAPE[0] // 2

    result = refine_vertebra_mask(as_image(two), bone_ct())
    out = sitk.GetArrayFromImage(result.mask)

    assert sorted(int(v) for v in np.unique(out)) == [0, BODY_LABEL, SECOND_LABEL]
    # Neither label crosses into the other's territory.
    assert int((out[half:] == BODY_LABEL).sum()) == 0
    assert int((out[:half] == SECOND_LABEL).sum()) == 0
    for label in (BODY_LABEL, SECOND_LABEL):
        assert result.per_label[label].ratio >= 0.90


def test_band_width_is_measured_in_millimetres_not_voxels():
    """A band of 0 mm leaves the anti-aliased boundary exactly as it was."""
    raw = stair_stepped_label()

    narrow = refine_vertebra_mask(
        as_image(raw), bone_ct(), RefinementConfig(band_mm=0.0)
    )
    antialias_only = refine_vertebra_mask(as_image(raw), None)

    assert np.array_equal(
        sitk.GetArrayFromImage(narrow.mask),
        sitk.GetArrayFromImage(antialias_only.mask),
    )
