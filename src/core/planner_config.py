from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from typing import TYPE_CHECKING, Any, Dict, Mapping, Tuple

from ..utils.constants import (
    ANTERIOR_SAFETY_MARGIN_MM,
    CORTICAL_WALL_CLEARANCE_MM,
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
    wall_clearance_mm: float = CORTICAL_WALL_CLEARANCE_MM
    anterior_margin_mm: float = ANTERIOR_SAFETY_MARGIN_MM
    max_convergence_deg: float = 35.0
    min_convergence_deg: float = -5.0
    implant_lengths_mm: Tuple[float, ...] = IMPLANT_LENGTHS_MM
    implant_diameters_mm: Tuple[float, ...] = IMPLANT_DIAMETERS_MM
    trajectory_hu_threshold: float = TRAJECTORY_HU_LOOSENING_THRESHOLD
    #: Planning back-end, one of :data:`PLANNER_MODES`.
    mode: str = "optimizer"
    #: Trajectory family, one of :data:`TRAJECTORY_KINDS`.
    trajectory: str = "traditional"
    #: Objective weights consumed by the optimiser (ignored in legacy mode).
    weights: "OptimizerWeights" = field(default_factory=_default_weights)

    def validate(self) -> None:
        if not 0.5 <= self.pedicle_fill_ratio <= 1.0:
            raise ValueError("pedicle_fill_ratio must be within [0.5, 1.0]")
        if not 0.0 <= self.wall_clearance_mm <= 3.0:
            raise ValueError("wall_clearance_mm must be within [0, 3]")
        if not 0.0 <= self.anterior_margin_mm <= 15.0:
            raise ValueError("anterior_margin_mm must be within [0, 15]")
        if not -30.0 <= self.min_convergence_deg < self.max_convergence_deg <= 90.0:
            raise ValueError("convergence limits must satisfy -30 <= min < max <= 90")
        if not self.implant_lengths_mm or not self.implant_diameters_mm:
            raise ValueError("implant catalogues must not be empty")
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
        for key in ("mode", "trajectory"):
            if key in kwargs:
                kwargs[key] = str(kwargs[key])
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
