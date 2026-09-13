from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from typing import TYPE_CHECKING, Any, Dict, Mapping, Tuple

from ..utils.constants import (
    ANTERIOR_SAFETY_MARGIN_MM,
    IMPLANT_DIAMETERS_MM,
    IMPLANT_LENGTHS_MM,
    PEDICLE_FILL_RATIO,
    TRAJECTORY_HU_LOOSENING_THRESHOLD,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .trajectory_optimizer import OptimizerWeights

#: Planner back-ends: the candidate optimiser or the original greedy planner.
PLANNER_MODES: Tuple[str, ...] = ("optimizer", "legacy")

#: Trajectory families the planner may aim for.
TRAJECTORY_KINDS: Tuple[str, ...] = ("traditional", "cbt")


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
    #: Lateral (in-out-in) breach a narrow side may accept.  2 mm keeps the
    #: default inside a Gertzbein grade B; the literature accepts up to about
    #: 6 mm in T4-T9 with the in-out-in technique, which the range allows.
    narrow_lateral_breach_mm: float = 2.0
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
    #: Containment bar the legacy planner enforces in its diameter step-down.
    #: The default (``False``) requires grade A (zero breach), matching the
    #: optimiser's ``breach_mm <= 0`` rule and CBT's shaft-feasibility rule, so
    #: the same anatomy is held to the same bar on every path.  When ``True``
    #: the legacy loop also accepts grade B (< 2 mm breach); the accepted
    #: screw then carries a persistent "Grade B accepted" warning so a
    #: re-grade keeps the acceptance on the record.  The narrow policy is
    #: untouched: it is already on the smallest implant and slides laterally
    #: instead of stepping down, whatever this flag says.
    accept_grade_b: bool = False
    #: Conservative floor for pedicle sizing.  The analyser reports both the
    #: width and the smaller ``width_lower_bound_mm`` second opinion (extent
    #: vs. inscribed diameter); when the two disagree by more than this the
    #: measurement is cross-section dependent and the planner sizes from the
    #: bound, not the headline width.
    width_bound_disagreement_mm: float = 1.0
    #: Objective weights consumed by the optimiser (ignored in legacy mode).
    weights: "OptimizerWeights" = field(default_factory=_default_weights)

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
        if not 0.0 <= self.narrow_lateral_breach_mm <= 6.0:
            raise ValueError("narrow_lateral_breach_mm must be within [0, 6]")
        if not self.implant_lengths_mm or not self.implant_diameters_mm:
            raise ValueError("implant catalogues must not be empty")
        for key in ("implant_lengths_mm", "implant_diameters_mm"):
            values = [float(v) for v in getattr(self, key)]
            if any(b < a - 1e-9 for a, b in zip(values, values[1:], strict=False)):
                raise ValueError(f"{key} must be sorted in ascending order")
        if not 0.0 <= self.width_bound_disagreement_mm <= 5.0:
            raise ValueError("width_bound_disagreement_mm must be within [0, 5]")
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
                kwargs[key] = tuple(sorted(float(v) for v in kwargs[key]))
        for key in ("mode", "trajectory"):
            if key in kwargs:
                kwargs[key] = str(kwargs[key])
        if "endplate_parallel" in kwargs:
            raw = kwargs["endplate_parallel"]
            # QSettings returns "true"/"false" strings, and bool("false") is True.
            kwargs["endplate_parallel"] = (
                raw.strip().lower() in ("true", "1", "yes")
                if isinstance(raw, str)
                else bool(raw)
            )
        if "endplate_tolerance_deg" in kwargs:
            kwargs["endplate_tolerance_deg"] = float(kwargs["endplate_tolerance_deg"])
        if "accept_grade_b" in kwargs:
            raw = kwargs["accept_grade_b"]
            # QSettings hands bools back as "true"/"false" strings, as with
            # endplate_parallel above.
            kwargs["accept_grade_b"] = (
                raw.strip().lower() in ("true", "1", "yes")
                if isinstance(raw, str)
                else bool(raw)
            )
        if "width_bound_disagreement_mm" in kwargs:
            kwargs["width_bound_disagreement_mm"] = float(
                kwargs["width_bound_disagreement_mm"]
            )
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
