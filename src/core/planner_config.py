from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any, Dict, Mapping, Tuple

from ..utils.constants import (ANTERIOR_SAFETY_MARGIN_MM, CORTICAL_WALL_CLEARANCE_MM,
                               IMPLANT_DIAMETERS_MM, IMPLANT_LENGTHS_MM, PEDICLE_FILL_RATIO,
                               TRAJECTORY_HU_LOOSENING_THRESHOLD)


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

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "PlannerConfig":
        names = {f.name for f in fields(cls)}
        kwargs = {k: v for k, v in data.items() if k in names}
        for key in ("implant_lengths_mm", "implant_diameters_mm"):
            if key in kwargs:
                kwargs[key] = tuple(float(v) for v in kwargs[key])
        cfg = cls(**kwargs)
        cfg.validate()
        return cfg

    def to_mapping(self) -> Dict[str, Any]:
        data = asdict(self)
        data["implant_lengths_mm"] = list(self.implant_lengths_mm)
        data["implant_diameters_mm"] = list(self.implant_diameters_mm)
        return data
