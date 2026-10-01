from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from typing import TYPE_CHECKING, Any, Dict, Mapping, Tuple

import numpy as np

from ..utils.constants import (
    ANTERIOR_SAFETY_MARGIN_MM,
    IMPLANT_DIAMETERS_MM,
    IMPLANT_LENGTHS_MM,
    PEDICLE_FILL_RATIO,
    TRAJECTORY_HU_LOOSENING_THRESHOLD,
)
from .screw_grading import BREACH_GRADE_LIMITS_MM

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .trajectory_optimizer import OptimizerWeights

#: Planner back-ends: the candidate optimiser or the original greedy planner.
PLANNER_MODES: Tuple[str, ...] = ("optimizer", "legacy")

#: Trajectory families the planner may aim for.
TRAJECTORY_KINDS: Tuple[str, ...] = ("traditional", "cbt")

#: Gertzbein grades the user may let automatic planning accept, best first.
ACCEPTED_BREACH_GRADES: Tuple[str, ...] = ("A", "B", "C")


def _default_weights() -> "OptimizerWeights":
    """Build the default objective weights.

    Imported lazily: :mod:`.trajectory_optimizer` imports this module, so a
    module-level import here would be circular.
    """
    from .trajectory_optimizer import OptimizerWeights

    return OptimizerWeights()


@dataclass(frozen=True)
class PlannerConfig:
    pedicle_fill_ratio: float = PEDICLE_FILL_RATIO
    #: Cortical clearance the planner keeps on each side of the screw.  The
    #: default is 0 mm: a clearance that drops a whole pedicle side is a policy,
    #: not a measurement, and the surgeon-facing rule is now "place the smallest
    #: screw and mark the level".  The spin box still lets it be raised.
    wall_clearance_mm: float = 0.0
    anterior_margin_mm: float = ANTERIOR_SAFETY_MARGIN_MM
    max_convergence_deg: float = 35.0
    min_convergence_deg: float = -5.0
    #: Measured pedicle width below which a side is planned under the
    #: narrow-pedicle policy: smallest implant, medial wall protected, level
    #: marked red.
    narrow_pedicle_mm: float = 5.0
    #: Worst Gertzbein grade automatic planning may accept, one of
    #: :data:`ACCEPTED_BREACH_GRADES`.  "A" -- no cortical breach at all -- is
    #: the default; accepting a breach is the user's explicit choice.  On a
    #: narrow side only a lateral (in-out-in) breach is ever accepted, and
    #: nowhere is a medial (canal-side) one.
    accepted_breach_grade: str = "A"
    #: Let the legacy fallback place a narrow side's smallest screw even when
    #: its best trajectory is not contained (a breach beyond
    #: :attr:`accepted_breach_grade`, or any medial or craniocaudal breach).
    #: Off by default: such a side is left unplanned with a reason.
    place_uncontained_narrow: bool = False
    implant_lengths_mm: Tuple[float, ...] = IMPLANT_LENGTHS_MM
    implant_diameters_mm: Tuple[float, ...] = IMPLANT_DIAMETERS_MM
    trajectory_hu_threshold: float = TRAJECTORY_HU_LOOSENING_THRESHOLD
    #: Planning back-end, one of :data:`PLANNER_MODES`.
    mode: str = "optimizer"
    #: Trajectory family, one of :data:`TRAJECTORY_KINDS`.
    trajectory: str = "traditional"
    #: Aim the trajectory along the upper endplate.  When off, the legacy
    #: planner keeps the target at the entry height and the optimiser drops both
    #: its endplate band and its endplate objective, leaving the sagittal angle
    #: to the other objectives.
    endplate_parallel: bool = True
    #: Half-width of the hard band the optimiser holds the endplate angle inside
    #: while :attr:`endplate_parallel` is on (degrees).
    endplate_tolerance_deg: float = 10.0
    #: Objective weights consumed by the optimiser (ignored in legacy mode).
    weights: "OptimizerWeights" = field(default_factory=_default_weights)

    def __post_init__(self) -> None:
        # Consumers read the catalogues' ends as the smallest and largest
        # implant, so they are kept ascending (and free of repeats) however a
        # caller or a saved setting listed them.
        for name in ("implant_lengths_mm", "implant_diameters_mm"):
            object.__setattr__(
                self, name, tuple(sorted({float(v) for v in getattr(self, name)}))
            )

    @property
    def accepted_grades(self) -> Tuple[str, ...]:
        """Every grade from A up to :attr:`accepted_breach_grade`."""
        return ACCEPTED_BREACH_GRADES[
            : ACCEPTED_BREACH_GRADES.index(self.accepted_breach_grade) + 1
        ]

    @property
    def lateral_breach_limit_mm(self) -> float:
        """Upper edge of the accepted grade's breach range (mm).

        0 for "A"; otherwise the grade's *exclusive* limit from
        :data:`~.screw_grading.BREACH_GRADE_LIMITS_MM` -- test a breach with
        :meth:`accepts_breach`, which applies the edges exactly as grading does.
        """
        if self.accepted_breach_grade == "A":
            return 0.0
        return BREACH_GRADE_LIMITS_MM[self.accepted_breach_grade]

    def accepts_breach(self, breach_mm: Any) -> Any:
        """Whether each breach (mm) grades no worse than :attr:`accepted_breach_grade`.

        Matches :meth:`~.screw_grading.ScrewGrader.grade_from_breach` at the
        boundaries: grade A is ``breach <= 0`` and every later grade's limit is
        exclusive, so exactly 2.0 mm is a C and is refused under "Up to B".
        Works on a scalar or a NumPy array.
        """
        breach = np.asarray(breach_mm, dtype=np.float64)
        if self.accepted_breach_grade == "A":
            return breach <= 0.0
        return breach < self.lateral_breach_limit_mm

    def validate(self) -> None:
        if not 0.5 <= self.pedicle_fill_ratio <= 1.0:
            raise ValueError("pedicle_fill_ratio must be within [0.5, 1.0]")
        if not 0.0 <= self.wall_clearance_mm <= 3.0:
            raise ValueError("wall_clearance_mm must be within [0, 3]")
        if not 0.0 <= self.anterior_margin_mm <= 15.0:
            raise ValueError("anterior_margin_mm must be within [0, 15]")
        if not 0.0 <= self.endplate_tolerance_deg <= 30.0:
            raise ValueError("endplate_tolerance_deg must be within [0, 30]")
        if not -30.0 <= self.min_convergence_deg < self.max_convergence_deg <= 90.0:
            raise ValueError("convergence limits must satisfy -30 <= min < max <= 90")
        if not 3.0 <= self.narrow_pedicle_mm <= 8.0:
            raise ValueError("narrow_pedicle_mm must be within [3, 8]")
        if self.accepted_breach_grade not in ACCEPTED_BREACH_GRADES:
            raise ValueError(
                f"accepted_breach_grade must be one of {ACCEPTED_BREACH_GRADES}, "
                f"got {self.accepted_breach_grade!r}"
            )
        if not self.implant_lengths_mm or not self.implant_diameters_mm:
            raise ValueError("implant catalogues must not be empty")
        if self.implant_lengths_mm[0] <= 0.0 or self.implant_diameters_mm[0] <= 0.0:
            raise ValueError("implant catalogue sizes must be positive")
        if self.mode not in PLANNER_MODES:
            raise ValueError(f"mode must be one of {PLANNER_MODES}, got {self.mode!r}")
        if self.trajectory not in TRAJECTORY_KINDS:
            raise ValueError(
                f"trajectory must be one of {TRAJECTORY_KINDS}, got {self.trajectory!r}"
            )

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "PlannerConfig":
        names = {f.name for f in fields(cls)}
        kwargs = {k: v for k, v in data.items() if k in names}
        for key in ("implant_lengths_mm", "implant_diameters_mm"):
            if key in kwargs:
                kwargs[key] = tuple(float(v) for v in kwargs[key])
        for key in ("mode", "trajectory", "accepted_breach_grade"):
            if key in kwargs:
                kwargs[key] = str(kwargs[key])
        for key in ("endplate_parallel", "place_uncontained_narrow"):
            if key in kwargs:
                raw = kwargs[key]
                # QSettings returns "true"/"false" strings, and bool("false") is True.
                kwargs[key] = (
                    raw.strip().lower() in ("true", "1", "yes")
                    if isinstance(raw, str)
                    else bool(raw)
                )
        if "endplate_tolerance_deg" in kwargs:
            kwargs["endplate_tolerance_deg"] = float(kwargs["endplate_tolerance_deg"])
        if isinstance(kwargs.get("weights"), Mapping):
            from .trajectory_optimizer import OptimizerWeights

            allowed = {f.name for f in fields(OptimizerWeights)}
            kwargs["weights"] = OptimizerWeights(
                **{
                    name: float(value)
                    for name, value in kwargs["weights"].items()
                    if name in allowed
                }
            )
        cfg = cls(**kwargs)
        cfg.validate()
        return cfg

    def to_mapping(self) -> Dict[str, Any]:
        data = asdict(self)
        data["implant_lengths_mm"] = list(self.implant_lengths_mm)
        data["implant_diameters_mm"] = list(self.implant_diameters_mm)
        data["weights"] = asdict(self.weights)
        return data
