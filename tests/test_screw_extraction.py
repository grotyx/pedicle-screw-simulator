"""Screw extraction from a post-operative CT, on synthetic phantoms only."""

import numpy as np
import pytest
import SimpleITK as sitk

from src.core.screw_extraction import (
    METAL_HU_THRESHOLD,
    extract_screw_near,
    extract_screws,
)

#: (x, y, z) mm; deliberately anisotropic, as a clinical axial series is.
SPACING = (0.4, 0.4, 1.0)
#: Array shape (z, y, x): 40 x 70 x 100 mm.
SHAPE = (40, 175, 250)
ORIGIN = (-20.0, 10.0, 100.0)

SCREW_D = 6.5
SCREW_L = 45.0
#: Head -> tip: anterior (LPS -y), slightly cranial and lateral-to-medial.
AXIS = np.array([0.1, -1.0, 0.15]) / np.linalg.norm([0.1, -1.0, 0.15])
LEFT_HEAD = np.array([0.0, 65.0, 115.0])
RIGHT_HEAD = np.array([40.0, 65.0, 115.0])


def _grid():
    z, y, x = np.indices(SHAPE, dtype=np.float64)
    return np.stack(
        [
            ORIGIN[0] + x * SPACING[0],
            ORIGIN[1] + y * SPACING[1],
            ORIGIN[2] + z * SPACING[2],
        ],
        axis=-1,
    )


def _cylinder(points, start, end, diameter):
    axis = end - start
    length = np.linalg.norm(axis)
    axis = axis / length
    rel = points - start
    t = rel @ axis
    radial = np.linalg.norm(rel - t[..., None] * axis, axis=-1)
    return (t >= 0) & (t <= length) & (radial <= diameter / 2.0)


def _box(points, centre, half):
    return np.all(np.abs(points - np.asarray(centre)) <= np.asarray(half), axis=-1)


def _phantom(metal_masks):
    """Bone-like block (300 HU) in soft tissue (40 HU) with 3000 HU metal."""
    hu = np.full(SHAPE, 40, dtype=np.int16)
    hu[5:35, 20:160, 20:230] = 300
    for mask in metal_masks:
        hu[mask] = 3000
    image = sitk.GetImageFromArray(hu)
    image.SetSpacing(SPACING)
    image.SetOrigin(ORIGIN)
    return image


def _screw(points, head, diameter=SCREW_D, length=SCREW_L, axis=AXIS):
    return _cylinder(points, head, head + axis * length, diameter)


def _tulip(points, head, axis=AXIS):
    # A 12 mm block sitting on the head, centred 5 mm behind it.
    return _box(points, head - axis * 5.0, (6.0, 6.0, 6.0))


def _angle_deg(a, b):
    cos = abs(float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))))
    return float(np.degrees(np.arccos(min(1.0, cos))))


def _match(results, head):
    return min(results, key=lambda s: np.linalg.norm(np.asarray(s.entry_lps) - head))


def test_two_parallel_screws_recover_axis_endpoints_and_diameter():
    points = _grid()
    image = _phantom([_screw(points, LEFT_HEAD), _screw(points, RIGHT_HEAD)])

    extraction = extract_screws(image)

    assert len(extraction.screws) == 2, extraction.reason
    for head in (LEFT_HEAD, RIGHT_HEAD):
        screw = _match(extraction.screws, head)
        tip = head + AXIS * SCREW_L
        assert np.linalg.norm(np.asarray(screw.entry_lps) - head) <= 1.5
        assert np.linalg.norm(np.asarray(screw.target_lps) - tip) <= 1.5
        direction = np.asarray(screw.target_lps) - np.asarray(screw.entry_lps)
        assert _angle_deg(direction, AXIS) <= 2.0
        assert abs(screw.diameter_mm - SCREW_D) <= 1.0
        assert abs(screw.length_mm - SCREW_L) <= 1.5
        assert screw.voxel_count > 0
        assert 0.0 < screw.confidence <= 1.0


def test_rod_bridged_screws_with_tulips_separate_into_two():
    points = _grid()
    rod = _cylinder(
        points,
        LEFT_HEAD - AXIS * 5.0 - np.array([8.0, 0.0, 0.0]),
        RIGHT_HEAD - AXIS * 5.0 + np.array([8.0, 0.0, 0.0]),
        5.5,
    )
    image = _phantom(
        [
            _screw(points, LEFT_HEAD),
            _screw(points, RIGHT_HEAD),
            _tulip(points, LEFT_HEAD),
            _tulip(points, RIGHT_HEAD),
            rod,
        ]
    )

    extraction = extract_screws(image)

    assert len(extraction.screws) == 2, extraction.reason
    for head in (LEFT_HEAD, RIGHT_HEAD):
        screw = _match(extraction.screws, head)
        tip = head + AXIS * SCREW_L
        # The entry sits where the shaft leaves the tulip, not behind it.
        assert np.linalg.norm(np.asarray(screw.entry_lps) - head) <= 2.5
        assert np.linalg.norm(np.asarray(screw.target_lps) - tip) <= 1.5
        direction = np.asarray(screw.target_lps) - np.asarray(screw.entry_lps)
        assert _angle_deg(direction, AXIS) <= 2.0
        assert abs(screw.diameter_mm - SCREW_D) <= 1.0


def test_head_is_the_wide_end_even_when_it_is_anterior():
    # Cross-section beats the dorsal fallback: the tulip end is the head even
    # in this (unphysical) screw whose head points forwards.
    points = _grid()
    head = np.array([20.0, 20.0, 115.0])
    axis = -AXIS
    image = _phantom([_screw(points, head, axis=axis), _tulip(points, head, axis=axis)])

    extraction = extract_screws(image)

    assert len(extraction.screws) == 1, extraction.reason
    screw = extraction.screws[0]
    assert np.linalg.norm(np.asarray(screw.entry_lps) - head) <= 2.5
    assert np.linalg.norm(np.asarray(screw.target_lps) - (head + axis * SCREW_L)) <= 1.5


def test_plain_cylinder_head_falls_back_to_the_dorsal_end():
    points = _grid()
    image = _phantom([_screw(points, LEFT_HEAD)])

    screw = extract_screws(image).screws[0]

    # Head is posterior: larger LPS y than the tip.
    assert screw.entry_lps[1] > screw.target_lps[1]
    assert "dorsal" in screw.reason


def test_rods_and_cross_links_are_not_taken_for_screws():
    # On real post-op CTs, headless craniocaudal rod pieces and transverse
    # connectors passed the shape check; a pedicle screw runs front to back.
    points = _grid()
    rod = _cylinder(points, np.array([70.0, 30.0, 103.0]), np.array([70.0, 30.0, 137.0]), 6.0)
    cross_link = _cylinder(
        points, np.array([-15.0, 40.0, 104.0]), np.array([25.0, 40.0, 104.0]), 6.0
    )
    screw = _screw(points, RIGHT_HEAD)
    image = _phantom([rod, cross_link, screw])

    extraction = extract_screws(image)

    assert len(extraction.screws) == 1
    found = extraction.screws[0]
    assert np.linalg.norm(np.asarray(found.entry_lps) - RIGHT_HEAD) <= 1.5


def test_small_metal_blob_is_rejected():
    points = _grid()
    clip = _box(points, (20.0, 40.0, 120.0), (1.5, 1.5, 4.0))
    image = _phantom([clip])

    extraction = extract_screws(image)

    assert extraction.screws == []
    assert extraction.reason


def test_clip_next_to_a_screw_is_ignored():
    points = _grid()
    clip = _box(points, (20.0, 20.0, 105.0), (1.5, 1.5, 4.0))
    image = _phantom([clip, _screw(points, LEFT_HEAD)])

    extraction = extract_screws(image)

    assert len(extraction.screws) == 1
    assert np.linalg.norm(np.asarray(extraction.screws[0].entry_lps) - LEFT_HEAD) <= 1.5


def test_volume_without_metal_returns_empty_with_reason():
    image = _phantom([])

    extraction = extract_screws(image)

    assert extraction.screws == []
    assert str(int(METAL_HU_THRESHOLD)) in extraction.reason


def test_threshold_is_adjustable():
    points = _grid()
    image = _phantom([_screw(points, LEFT_HEAD)])

    assert extract_screws(image, threshold_hu=3500).screws == []
    assert len(extract_screws(image, threshold_hu=2500).screws) == 1


def test_seeded_mode_picks_the_screw_nearest_the_click():
    points = _grid()
    image = _phantom([_screw(points, LEFT_HEAD), _screw(points, RIGHT_HEAD)])
    click = RIGHT_HEAD + AXIS * 20.0 + np.array([2.0, 0.0, 0.0])

    extraction = extract_screw_near(image, tuple(click))

    assert len(extraction.screws) == 1
    assert np.linalg.norm(np.asarray(extraction.screws[0].entry_lps) - RIGHT_HEAD) <= 1.5


def test_seeded_mode_far_from_any_screw_returns_empty():
    points = _grid()
    image = _phantom([_screw(points, LEFT_HEAD)])

    extraction = extract_screw_near(image, (60.0, 20.0, 105.0))

    assert extraction.screws == []
    assert extraction.reason


def test_rotated_direction_matrix_is_rejected():
    image = _phantom([])
    image.SetDirection((0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0))

    with pytest.raises(ValueError):
        extract_screws(image)
