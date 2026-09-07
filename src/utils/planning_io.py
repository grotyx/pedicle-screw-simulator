"""
Planning I/O helpers for saving/loading simulation plans.
"""

import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.models.measurement import Measurement
from src.models.screw import Screw

PLAN_VERSION = 3
VALID_PLANES = {"axial", "sagittal", "coronal"}


def _parse_point3(value: Any, field_name: str) -> Tuple[float, float, float]:
    """Parse a 3-element numeric point tuple."""
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"Invalid {field_name}: expected 3 numeric values")
    return (float(value[0]), float(value[1]), float(value[2]))


def _jsonable_metric(value: Any) -> Any:
    """Coerce one metric value to something ``json.dump`` can write.

    Metric values are floats, ints, strings or ``None``; numpy scalars reach
    here from the graders, and a non-finite float would otherwise be written as
    the non-standard ``NaN``/``Infinity`` literal, so both are normalised.
    """
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return float(value) if math.isfinite(value) else None
    if hasattr(value, "__index__"):          # numpy integer
        return int(value)
    try:                                     # numpy float / anything float-like
        as_float = float(value)
    except (TypeError, ValueError):
        return str(value)
    return as_float if math.isfinite(as_float) else None


def metrics_to_dict(metrics: Any) -> Dict[str, Any]:
    """JSON-safe copy of a screw's metric bundle (``{}`` when absent)."""
    if not isinstance(metrics, dict):
        return {}
    return {str(key): _jsonable_metric(value) for key, value in metrics.items()}


def _metric_number(metrics: Dict[str, Any], key: str) -> str:
    """Format one numeric metric for CSV; empty when missing or unmeasurable."""
    value = metrics.get(key)
    if value is None or isinstance(value, (bool, str)):
        return ""
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return ""


def screw_to_dict(screw: Screw) -> Dict[str, Any]:
    """Serialize Screw dataclass to plain dict."""
    return {
        "entry_point": list(screw.entry_point),
        "target_point": list(screw.target_point),
        "length": float(screw.length),
        "diameter": float(screw.diameter),
        "vertebra_level": screw.vertebra_level,
        "side": screw.side,
        "trajectory": list(screw.trajectory),
        "insertion_angle": float(screw.insertion_angle),
        "medial_angle": float(screw.medial_angle),
        "grade": screw.grade,
        "breach_distance": float(screw.breach_distance),
        "mean_hu": None if screw.mean_hu is None else float(screw.mean_hu),
        "min_hu": None if screw.min_hu is None else float(screw.min_hu),
        "warnings": list(screw.warnings),
        "source": screw.source,
        "metrics": metrics_to_dict(screw.metrics),
    }


def screw_from_dict(data: Dict[str, Any]) -> Screw:
    """Deserialize Screw dataclass from dict."""
    if not isinstance(data, dict):
        raise ValueError("Invalid screw payload: expected object")

    entry_point = _parse_point3(data.get("entry_point"), "entry_point")
    target_point = _parse_point3(data.get("target_point"), "target_point")

    screw = Screw(
        entry_point=entry_point,
        target_point=target_point,
        diameter=float(data.get("diameter", 6.0)),
    )
    screw.vertebra_level = str(data.get("vertebra_level", ""))
    screw.side = str(data.get("side", ""))
    screw.grade = str(data.get("grade", screw.grade))
    screw.breach_distance = float(
        data.get("breach_distance", screw.breach_distance)
    )

    mean_hu = data.get("mean_hu")
    min_hu = data.get("min_hu")
    screw.mean_hu = None if mean_hu is None else float(mean_hu)
    screw.min_hu = None if min_hu is None else float(min_hu)
    raw_warnings = data.get("warnings", [])
    screw.warnings = [str(w) for w in raw_warnings] if isinstance(raw_warnings, list) else []
    screw.source = str(data.get("source", "manual"))
    # Absent in v1/v2 payloads; an empty bundle simply means "not measured".
    screw.metrics = metrics_to_dict(data.get("metrics"))

    if "trajectory" in data:
        screw.trajectory = _parse_point3(data["trajectory"], "trajectory")

    # insertion_angle/medial_angle are derived from entry/target/side; recompute
    # rather than trust stale values from the payload.
    screw._recompute_geometry()

    return screw


def measurement_to_dict(measurement: Measurement) -> Dict[str, Any]:
    """Serialize Measurement dataclass to plain dict."""
    return {
        "points": [list(point) for point in measurement.points],
        "distance": float(measurement.distance),
        "mode": measurement.mode,
        "angle": None if measurement.angle is None else float(measurement.angle),
        "label": measurement.label,
    }


def measurement_from_dict(data: Dict[str, Any]) -> Measurement:
    """Deserialize Measurement dataclass from dict."""
    if not isinstance(data, dict):
        raise ValueError("Invalid measurement payload: expected object")
    if not isinstance(data.get("points"), list):
        raise ValueError("Invalid measurement points")

    points = [_parse_point3(point, "point") for point in data["points"]]
    mode = str(data.get("mode", "distance"))
    if mode not in ("distance", "path", "angle"):
        raise ValueError(f"Invalid measurement mode: {mode}")

    return Measurement(
        points=points,
        distance=float(data.get("distance", 0.0)),
        mode=mode,
        angle=None if data.get("angle") is None else float(data["angle"]),
        label=str(data.get("label", "")),
    )


def serialize_plan(
    series_id: Optional[str],
    screws: List[Screw],
    measurements: List[Measurement],
    measurement_planes: Optional[List[Optional[str]]] = None,
) -> Dict[str, Any]:
    """Build plan payload dictionary for JSON persistence."""
    planes = measurement_planes
    if planes is None:
        planes = [None] * len(measurements)
    if len(planes) != len(measurements):
        raise ValueError("Measurement plane count must match measurements")

    measurement_items = []
    for measurement, plane in zip(measurements, planes, strict=True):
        item = measurement_to_dict(measurement)
        if plane is not None and plane not in VALID_PLANES:
            raise ValueError(f"Invalid measurement plane: {plane}")
        item["plane"] = plane
        measurement_items.append(item)

    return {
        "version": PLAN_VERSION,
        "series_id": series_id,
        "screws": [screw_to_dict(screw) for screw in screws],
        "measurements": measurement_items,
    }


def deserialize_plan(
    payload: Dict[str, Any]
) -> Dict[str, Any]:
    """Parse plan payload into runtime objects."""
    if not isinstance(payload, dict):
        raise ValueError("Invalid plan payload")

    screw_items = payload.get("screws", [])
    measurement_items = payload.get("measurements", [])
    if not isinstance(screw_items, list) or not isinstance(measurement_items, list):
        raise ValueError("Invalid plan payload: screws/measurements must be lists")

    screws = [screw_from_dict(item) for item in screw_items]
    measurements: List[Measurement] = []
    planes: List[Optional[str]] = []

    for item in measurement_items:
        plane = item.get("plane")
        if plane is not None and plane not in VALID_PLANES:
            raise ValueError(f"Invalid measurement plane: {plane}")
        measurements.append(measurement_from_dict(item))
        planes.append(plane)

    return {
        "version": int(payload.get("version", PLAN_VERSION)),
        "series_id": payload.get("series_id"),
        "screws": screws,
        "measurements": measurements,
        "measurement_planes": planes,
    }


def save_plan_json(path: str, payload: Dict[str, Any]) -> None:
    """Write plan payload to JSON file."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def load_plan_json(path: str) -> Dict[str, Any]:
    """Load plan payload from JSON file."""
    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def export_screws_csv(path: str, screws: List[Screw]) -> None:
    """Export screw list to CSV file."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "index", "vertebra_level", "side", "source", "length_mm", "diameter_mm",
            "grade", "breach_distance_mm", "mean_hu", "min_hu",
            "entry_x", "entry_y", "entry_z", "target_x", "target_y", "target_z",
            "convergence_angle_deg", "craniocaudal_angle_deg",
            "trajectory_mean_hu", "pedicle_mean_hu", "body_mean_hu", "hu_ratio",
            "min_wall_mm", "heary_direction", "facet_grade", "warnings",
        ])

        for index, screw in enumerate(screws, start=1):
            metrics = screw.metrics if isinstance(screw.metrics, dict) else {}
            writer.writerow([
                index,
                screw.vertebra_level,
                screw.side,
                screw.source,
                f"{screw.length:.3f}",
                f"{screw.diameter:.3f}",
                screw.grade,
                f"{screw.breach_distance:.3f}",
                "" if screw.mean_hu is None else f"{screw.mean_hu:.3f}",
                "" if screw.min_hu is None else f"{screw.min_hu:.3f}",
                f"{screw.entry_point[0]:.3f}",
                f"{screw.entry_point[1]:.3f}",
                f"{screw.entry_point[2]:.3f}",
                f"{screw.target_point[0]:.3f}",
                f"{screw.target_point[1]:.3f}",
                f"{screw.target_point[2]:.3f}",
                f"{screw.medial_angle:.3f}",
                f"{screw.insertion_angle:.3f}",
                _metric_number(metrics, "trajectory_mean_hu"),
                _metric_number(metrics, "pedicle_mean_hu"),
                _metric_number(metrics, "body_mean_hu"),
                _metric_number(metrics, "trajectory_body_ratio"),
                _metric_number(metrics, "min_wall_mm"),
                "" if metrics.get("heary_direction") is None else str(metrics["heary_direction"]),
                "" if metrics.get("facet_grade") is None else f"{int(metrics['facet_grade'])}",
                "|".join(screw.warnings),
            ])
