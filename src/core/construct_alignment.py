"""Construct-level alignment metrics, and keeping them true after an edit.

Three of the numbers on a screw describe the *set* it belongs to rather than
the screw itself: how far its head sits off its side's best-fit rod line, how
far its convergence sits from its neighbours', and the RMS of that
disagreement across the side.  Move one head and all three change for every
screw on that side.

The planner stamps them once at the end of a run.  Left at that, the first
drag of an auto screw made the cockpit's Alignment row describe where the
screw used to be -- the rod offset of a trajectory the user had already
replaced.  So the same stamping runs again after every edit, from
:class:`~src.tools.screw_tool.ScrewTool`, over the :class:`~src.models.screw.Screw`
model instead of :class:`~src.core.auto_screw_planner.PlannedScrew`.

Two callers, two screw types, one implementation: the accessors below are how
the caller says where its fields live, so a dragged screw and a planned one
can never disagree about the same construct.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .trajectory_optimizer import (
    convergence_deviations_deg,
    convergence_spread_deg,
    rod_misalignment_mm,
)

#: RMS distance of one side's heads from their best-fit line, in millimetres.
ROD_KEY = "rod_misalignment_mm"

#: RMS convergence disagreement across one side, in degrees.
SPREAD_KEY = "convergence_spread_deg"

#: Signed offset of this screw's convergence from its neighbours', in degrees.
#: Absent on a screw the term excludes (sacral), which was never asked to agree.
DEVIATION_KEY = "convergence_deviation_deg"

#: Every metric this module owns, so a caller can strip them in one place.
ALIGNMENT_KEYS: Tuple[str, ...] = (ROD_KEY, SPREAD_KEY, DEVIATION_KEY)


def stamp_alignment(
    screws: Iterable[Any],
    *,
    side_of: Callable[[Any], str],
    entry_of: Callable[[Any], Sequence[float]],
    convergence_of: Callable[[Any], float],
    level_of: Callable[[Any], Optional[int]],
    metrics_of: Callable[[Any], Dict[str, Any]],
) -> None:
    """Measure each side's alignment and write it onto that side's screws.

    Each side is measured on its own: the two rods are bent independently and
    a left-side outlier says nothing about the right.  A screw whose level the
    convergence term excludes (S1) gets the side's spread but no deviation of
    its own, because it was never asked to agree.
    """
    on_sides = _group_by_side(screws, side_of)
    for members in on_sides.values():
        if not members:
            continue
        heads = np.asarray(
            [[float(v) for v in entry_of(s)] for s in members], dtype=np.float64
        )
        rod = rod_misalignment_mm(heads)

        angles = [float(convergence_of(s)) for s in members]
        levels = [level_of(s) for s in members]
        spread = convergence_spread_deg(angles, levels)
        deviations = convergence_deviations_deg(angles, levels)

        for screw, deviation in zip(members, deviations, strict=True):
            metrics = metrics_of(screw)
            metrics[ROD_KEY] = rod
            metrics[SPREAD_KEY] = spread
            if deviation is None:
                # "Not asked to agree" is not "agrees perfectly": leave no
                # number rather than a 0.0 the surgeon would read as a finding.
                metrics.pop(DEVIATION_KEY, None)
            else:
                metrics[DEVIATION_KEY] = float(deviation)


def restamp_carriers(
    screws: Iterable[Any],
    *,
    side_of: Callable[[Any], str],
    entry_of: Callable[[Any], Sequence[float]],
    convergence_of: Callable[[Any], float],
    level_of: Callable[[Any], Optional[int]],
    metrics_of: Callable[[Any], Dict[str, Any]],
) -> None:
    """Refresh these metrics on the screws that already carry them.

    Only those: carrying :data:`ROD_KEY` is what marks a screw as part of a
    construct the optimiser actually harmonised.  A legacy-mode plan and a
    hand-placed screw never had the metrics and must not acquire them here --
    :func:`~src.core.auto_screw_planner.construct_summary` reads their absence
    as "this construct was never harmonised", and inventing a rod fit for one
    that was not would be a lie in the safest-looking direction.

    The rod line is refitted over the carriers alone for the same reason: they
    are the construct that was planned, and a screw added by hand afterwards
    was never part of it.
    """
    carriers = [s for s in screws if ROD_KEY in metrics_of(s)]
    if not carriers:
        return
    stamp_alignment(
        carriers,
        side_of=side_of,
        entry_of=entry_of,
        convergence_of=convergence_of,
        level_of=level_of,
        metrics_of=metrics_of,
    )


def _group_by_side(
    screws: Iterable[Any], side_of: Callable[[Any], str]
) -> Dict[str, List[Any]]:
    """Split screws into the two rods, dropping any with no side.

    A screw with no side belongs to neither rod; including it would bend a
    line it is not on.
    """
    grouped: Dict[str, List[Any]] = {"left": [], "right": []}
    for screw in screws:
        side = str(side_of(screw) or "")
        if side in grouped:
            grouped[side].append(screw)
    return grouped
