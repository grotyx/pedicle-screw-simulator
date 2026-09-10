"""Tests for the cortical bone trajectory (CBT / mCBT) planning mode."""

import os
import sys

import numpy as np
import pytest
import SimpleITK as sitk
from scipy import ndimage as ndi

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.core.auto_screw_planner import AutoScrewPlanner
from src.core.cbt_planner import cbt_directions, cbt_entry_point, plan_cbt_screw
from src.core.pedicle_analyzer import PedicleAnalyzer
from src.core.planner_config import PlannerConfig
from src.core.screw_grading import ScrewGrader
from src.core.trajectory_optimizer import TIP_SEGMENT_MM
from src.utils.constants import CBT_CONTRAINDICATION_NOTE, CBT_DEFAULTS

LABEL = 28  # L4


def _make_tall_phantom(label: int = LABEL) -> sitk.Image:
    """A blocky L4 with a tall pedicle and a deep posterior arch, 1 mm isotropic.

    ``tests.test_pedicle_analyzer._make_anatomical_phantom`` is deliberately
    tight: its 30 mm body and 12 mm pedicles leave no room for a 25 deg cranial
    trajectory to run 30 mm of catalogue length, and it has no bone inferomedial
    to the isthmus for a CBT entry to sit on.  This phantom keeps the same
    coordinate conventions (LPS identity, midline at x = 45) and adds what CBT
    needs:

    * a 40 mm tall body box (``z`` 10..49) so the cranial trajectory stays
      inside bone all the way to a 40 mm tip;
    * pedicles as tall ellipses (14 x 22 mm, centred ``x`` = 56 / 34, ``z`` = 32)
      spanning ``y`` 44..61, whose medial edge stops short of the 6 mm midline
      band so the coronal isthmus search still brackets them;
    * a deep posterior arch block (``y`` 54..77) standing in for the
      pars/lamina, which reaches the midline (ending the isthmus search where
      the spinolaminar junction would) and carries the inferomedial CBT entry.

    The isthmus therefore lands at ``y`` = 52 with its inferior-medial corner at
    ``(56, 52, 21)``; the CBT entry seed (2 mm medial, 3 mm inferior) casts
    posteriorly into the arch.
    """
    Z, Y, X = 60, 90, 90
    zz, yy, xx = np.mgrid[0:Z, 0:Y, 0:X]
    body = (xx >= 25) & (xx < 66) & (yy >= 20) & (yy < 50) & (zz >= 10) & (zz < 50)
    ped = np.zeros_like(body)
    for cx in (34, 56):
        ped |= (
            (((xx - cx) / 7.0) ** 2 + ((zz - 32) / 11.0) ** 2 <= 1)
            & (yy >= 44)
            & (yy < 62)
        )
    arch = (
        (xx >= 23) & (xx < 68) & (yy >= 54) & (yy < 78) & (zz >= 12) & (zz < 40)
    )
    arr = np.zeros((Z, Y, X), dtype=np.uint8)
    arr[body | ped | arch] = label
    return sitk.GetImageFromArray(arr)


def _make_ct(mask: sitk.Image, cortical_hu: int = 900, cancellous_hu: int = 200) -> sitk.Image:
    """A CT on the mask grid with a 2 mm cortical shell around the bone.

    CBT trades pedicle fill for cortical purchase, so a uniform-HU phantom
    cannot tell the two trajectory families apart; the shell makes the
    difference measurable.
    """
    arr = sitk.GetArrayFromImage(mask)
    bone = arr > 0
    interior = ndi.binary_erosion(bone, iterations=2)
    hu = np.full(bone.shape, -50, dtype=np.int16)
    hu[bone] = cortical_hu
    hu[interior] = cancellous_hu
    ct = sitk.GetImageFromArray(hu)
    ct.CopyInformation(mask)
    return ct


def _setup():
    """Phantom + matching CT + pedicle analysis."""
    mask = _make_tall_phantom()
    ct = _make_ct(mask)
    analyzer = PedicleAnalyzer(mask)
    analysis = analyzer.analyze_pedicle(analyzer.get_available_vertebrae()[0])
    return ct, mask, analysis


# --------------------------------------------------------------- analysis side
def test_analysis_exposes_inferior_medial_isthmus_corner():
    _ct, _mask, analysis = _setup()
    for side in ("left", "right"):
        corner = getattr(analysis, f"{side}_pedicle_inferior_medial_lps")
        center = getattr(analysis, f"{side}_pedicle_center")
        assert corner is not None
        assert corner.shape == (3,)
        assert corner[1] == pytest.approx(center[1])          # same isthmus slice
        assert corner[2] < center[2]                          # inferior
        medial = -1.0 if side == "left" else 1.0
        assert medial * (corner[0] - center[0]) <= 0.0        # at or medial to centre


# ------------------------------------------------------------------- entry point
def test_cbt_entry_is_posterior_and_inferior_to_isthmus():
    ct, mask, analysis = _setup()
    grader = ScrewGrader(mask, ct)
    entry = cbt_entry_point(grader, analysis, "left", LABEL)
    assert entry is not None
    assert entry[1] > analysis.left_pedicle_center[1]        # posterior
    assert entry[2] < analysis.left_pedicle_center[2]        # inferior


def test_cbt_entry_sits_inside_bone():
    ct, mask, analysis = _setup()
    grader = ScrewGrader(mask, ct)
    for side in ("left", "right"):
        entry = cbt_entry_point(grader, analysis, side, LABEL)
        assert entry is not None
        d_out, _d_in = grader.distances_at_points(entry.reshape(1, 3), LABEL)
        assert d_out[0] == 0.0


def test_cbt_entry_is_at_least_one_voxel_inside_the_cortex():
    """A back-off smaller than a voxel would seat the entry in the surface layer.

    The mask below has 2 mm ``y`` voxels, so the nominal 1 mm back-off lands in
    the same voxel as the last bone sample; the entry must move a whole voxel.
    """
    arr = np.zeros((40, 40, 40), dtype=np.uint8)
    arr[10:30, 5:16, 10:30] = LABEL          # y index 5..15 -> 10..30 mm
    mask = sitk.GetImageFromArray(arr)
    mask.SetSpacing((1.0, 2.0, 1.0))
    analysis = _setup()[2]
    analysis.left_pedicle_inferior_medial_lps = np.array([20.0, 12.0, 20.0])

    entry = cbt_entry_point(ScrewGrader(mask), analysis, "left", LABEL)

    assert entry is not None
    i, j, k = mask.TransformPhysicalPointToIndex(tuple(float(v) for v in entry))
    assert arr[k, j, i] == LABEL             # inside bone…
    assert arr[k, j + 1, i] == LABEL         # …and not in the outermost layer


def test_cbt_entry_without_a_corner_returns_none():
    ct, mask, analysis = _setup()
    analysis.left_pedicle_inferior_medial_lps = None
    grader = ScrewGrader(mask, ct)
    assert cbt_entry_point(grader, analysis, "left", LABEL) is None


# ------------------------------------------------------------------ screw plan
def test_cbt_screw_angles_and_size():
    ct, mask, analysis = _setup()
    screw = plan_cbt_screw(
        ScrewGrader(mask, ct), analysis, "left", LABEL, PlannerConfig(trajectory="cbt")
    )
    assert screw is not None
    assert 18.0 <= screw.craniocaudal_angle <= 32.0
    assert -17.0 <= screw.convergence_angle <= -5.0        # lateral = negative convergence
    assert screw.diameter_mm in CBT_DEFAULTS["diameter_mm"]
    assert screw.length_mm in CBT_DEFAULTS["lengths_mm"]
    assert screw.gertzbein_grade in {"A", "B"}
    assert screw.metrics["trajectory_type"] == "cbt"
    assert CBT_CONTRAINDICATION_NOTE in screw.warnings


def test_a_cbt_screw_on_a_narrow_pedicle_is_flagged():
    """A CBT screw shares the traditional planner's narrow-pedicle policy.

    ``_finalise_screw`` decides the flag from its ``narrow`` argument alone, so
    a CBT side measured below ``narrow_pedicle_mm`` must be flagged the same
    way a traditional one is -- red in the UI, and carrying the same warning.
    """
    ct, mask, analysis = _setup()
    analysis.left_pedicle_width = 3.0  # below the default 5.0 mm threshold

    screw = plan_cbt_screw(
        ScrewGrader(mask, ct), analysis, "left", LABEL, PlannerConfig(trajectory="cbt")
    )

    assert screw is not None
    assert screw.metrics["narrow_pedicle"] is True
    assert any("narrow" in warning.lower() for warning in screw.warnings)


def test_cbt_screw_diverges_laterally_on_both_sides():
    ct, mask, analysis = _setup()
    grader = ScrewGrader(mask, ct)
    config = PlannerConfig(trajectory="cbt")
    for side, lateral_sign in (("left", 1.0), ("right", -1.0)):
        screw = plan_cbt_screw(grader, analysis, side, LABEL, config)
        assert screw is not None
        travel = screw.target_lps - screw.entry_lps
        assert lateral_sign * travel[0] > 0.0     # tip lands lateral to the entry
        assert travel[1] < 0.0                    # advancing anteriorly
        assert travel[2] > 0.0                    # advancing cranially


def test_planned_cbt_tip_clears_the_anterior_margin():
    """The tip, not just the shaft, has to respect ``anterior_margin_mm``."""
    ct, mask, analysis = _setup()
    grader = ScrewGrader(mask, ct)
    config = PlannerConfig(trajectory="cbt")        # 4 mm anterior margin

    screw = plan_cbt_screw(grader, analysis, "left", LABEL, config)

    assert screw is not None
    direction = screw.target_lps - screw.entry_lps
    direction /= np.linalg.norm(direction)
    tip = grader.evaluate_batch(
        (screw.target_lps - direction * TIP_SEGMENT_MM)[None, :],
        screw.target_lps[None, :],
        screw.diameter_mm,
        LABEL,
    )
    assert tip.breach_mm[0] <= 0.0
    assert tip.min_wall_mm[0] >= config.anterior_margin_mm - config.wall_clearance_mm


def test_the_chosen_cbt_trajectory_is_pinned():
    """Pins the winner, so narrowing which rows are graded cannot move it."""
    ct, mask, analysis = _setup()
    grader = ScrewGrader(mask, ct)

    screw = plan_cbt_screw(grader, analysis, "left", LABEL, PlannerConfig(trajectory="cbt"))

    assert screw is not None
    assert screw.entry_lps == pytest.approx(np.array([54.0, 76.0, 18.0]))
    assert screw.target_lps == pytest.approx(
        np.array([58.58078891, 38.69246822, 31.68080573])
    )
    assert (screw.diameter_mm, screw.length_mm) == (6.0, 40.0)
    assert screw.metrics["score"] == pytest.approx(0.5788617886178862)
    assert screw.metrics["cbt_cranial_angle_deg"] == pytest.approx(20.0)


def test_the_tip_batch_only_grades_shaft_feasible_candidates():
    """The anterior check must not double the cost of the whole sweep."""
    ct, mask, analysis = _setup()
    grader = ScrewGrader(mask, ct)
    graded = []
    real = grader.evaluate_batch
    grader.evaluate_batch = (                                   # type: ignore[method-assign]
        lambda e, t, d, label: (graded.append(len(e)), real(e, t, d, label))[1]
    )

    screw = plan_cbt_screw(grader, analysis, "left", LABEL, PlannerConfig(trajectory="cbt"))

    assert screw is not None
    sweep = len(cbt_directions("left")[0]) * len(CBT_DEFAULTS["lengths_mm"])
    assert max(graded) == sweep                       # the shaft pass sees them all
    assert sum(graded) < 2 * sweep * len(CBT_DEFAULTS["diameter_mm"])


def test_an_unreachable_anterior_margin_rejects_every_candidate():
    ct, mask, analysis = _setup()
    grader = ScrewGrader(mask, ct)

    lax = plan_cbt_screw(
        grader, analysis, "left", LABEL,
        PlannerConfig(trajectory="cbt", anterior_margin_mm=0.0),
    )
    strict = plan_cbt_screw(
        grader, analysis, "left", LABEL,
        PlannerConfig(trajectory="cbt", anterior_margin_mm=15.0),
    )

    assert lax is not None
    assert strict is None       # no tip in this phantom is 14 mm off every wall


# --------------------------------------------------------------- plan_all wiring
@pytest.mark.parametrize("mode", ["optimizer", "legacy"])
def test_plan_all_cbt_trajectory_marks_every_screw(mode):
    ct, mask, analysis = _setup()
    planner = AutoScrewPlanner(
        ct, mask, config=PlannerConfig(mode=mode, trajectory="cbt")
    )
    screws = planner.plan_all([analysis])
    assert len(screws) == 2
    assert {s.side for s in screws} == {"left", "right"}
    for screw in screws:
        assert screw.metrics["trajectory_type"] == "cbt"
        assert screw.craniocaudal_angle > 15.0


def test_plan_all_records_a_dropped_side_with_a_reason():
    """A side CBT cannot solve is reported, not silently missing from the plan."""
    ct, mask, analysis = _setup()
    analysis.right_pedicle_inferior_medial_lps = None
    planner = AutoScrewPlanner(ct, mask, config=PlannerConfig(trajectory="cbt"))

    screws = planner.plan_all([analysis])

    assert [s.side for s in screws] == ["left"]
    name = analysis.vertebra.name
    assert [(v, s) for v, s, _r in planner.skipped_sides] == [(name, "right")]
    assert planner.skipped_sides[0][2]          # a non-empty reason


def test_plan_all_traditional_trajectory_leaves_skipped_sides_empty():
    ct, mask, analysis = _setup()
    planner = AutoScrewPlanner(ct, mask, config=PlannerConfig(mode="legacy"))
    planner.plan_all([analysis])
    assert planner.skipped_sides == []


def test_plan_all_cbt_reports_progress_and_can_be_cancelled():
    ct, mask, analysis = _setup()
    planner = AutoScrewPlanner(ct, mask, config=PlannerConfig(trajectory="cbt"))
    messages = []

    planner.plan_all([analysis], progress=messages.append)

    name = analysis.vertebra.name
    assert messages == [
        f"Planning {name} left (1/2)…",
        f"Planning {name} right (2/2)…",
    ]

    asked = []

    def cancel():
        asked.append(True)
        return len(asked) > 1

    partial = planner.plan_all([analysis], cancel=cancel)

    assert [s.side for s in partial] == ["left"]
    assert planner.last_run_cancelled is True


def test_plan_all_traditional_trajectory_is_labelled_traditional():
    ct, mask, analysis = _setup()
    planner = AutoScrewPlanner(ct, mask, config=PlannerConfig(mode="legacy"))
    screws = planner.plan_all([analysis])
    assert screws
    for screw in screws:
        assert screw.metrics["trajectory_type"] == "traditional"
