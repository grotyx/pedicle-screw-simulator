"""
Planning I/O helpers for saving/loading simulation plans.
"""

import csv
import hashlib
import json
import math
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.models.measurement import Measurement
from src.models.screw import Screw

PLAN_VERSION = 3
VALID_PLANES = {"axial", "sagittal", "coronal"}
_FORMULA_LEAD = ("=", "+", "-", "@", chr(9), chr(13))


def series_uid_digest(series_uid: Optional[str]) -> Optional[str]:
    """One-way SHA-256 of a DICOM Series Instance UID (``None`` if absent).

    A plan file must not carry the raw UID: it can be looked up in PACS to
    re-link a shared plan to its patient. The digest still tells the same
    study from a different one.
    """
    if not isinstance(series_uid, str) or not series_uid:
        return None
    return hashlib.sha256(series_uid.encode("utf-8")).hexdigest()


def _finite(value: Any, field_name: str) -> float:
    """``float(value)``, rejecting NaN/Infinity (``json`` accepts both)."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"Invalid {field_name}: expected a finite number") from None
    if not math.isfinite(number):
        raise ValueError(f"Invalid {field_name}: expected a finite number")
    return number


def _parse_point3(value: Any, field_name: str) -> Tuple[float, float, float]:
    """Parse a 3-element finite numeric point tuple."""
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"Invalid {field_name}: expected 3 numeric values")
    return tuple(_finite(v, field_name) for v in value)  # type: ignore[return-value]


def _csv_text(value: Any) -> str:
    """Text cell for CSV; a leading formula character gets a ``'`` prefix."""
    text = str(value)
    return "'" + text if text.startswith(_FORMULA_LEAD) else text


def _jsonable_metric(value: Any) -> Any:
    """Coerce one metric value to something ``json.dump`` can write.

    Metric values are floats, ints, strings or ``None``; numpy scalars reach
    here from the graders, and a non-finite float would otherwise be written as
    the non-standard ``NaN``/``Infinity`` literal, so both are normalised.

    Containers recurse.  The optimiser stores ``score_components`` as a dict of
    per-term contributions, and without recursion it fell through to
    ``str(value)`` — the file then held a single-quoted Python literal that no
    JSON consumer, this app on reload included, could turn back into numbers.
    """
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _jsonable_metric(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable_metric(item) for item in value]
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


def _metric_number(metrics: Dict[str, Any], key: str, spec: str = ".3f") -> str:
    """Format one numeric metric for CSV; empty when missing or unmeasurable.

    ``spec`` is a format spec, except for ``"d"``, which rounds to a plain
    integer (grades read better as ``2`` than as ``2.000``). Anything that is
    not a number — including a string a future writer might store under a
    numeric key — yields an empty cell rather than raising mid-export.
    """
    value = metrics.get(key)
    if value is None or isinstance(value, (bool, str)):
        return ""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return ""
    if not math.isfinite(number):
        return ""
    return f"{int(number):d}" if spec == "d" else format(number, spec)


def _metric_flag(metrics: Dict[str, Any], key: str) -> str:
    """Format one boolean metric for CSV; empty when it was never recorded.

    A manually placed screw has no pedicle analysis behind it, so ``narrow`` is
    unknown rather than false -- and a spreadsheet that reads an empty cell as
    "not narrow" is at least reading it as a missing measurement, which an
    explicit ``false`` would hide.
    """
    value = metrics.get(key)
    if value is None:
        return ""
    return "true" if bool(value) else "false"


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
        "reviewed": bool(screw.reviewed),
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
        diameter=_finite(data.get("diameter", 6.0), "diameter"),
    )
    screw.vertebra_level = str(data.get("vertebra_level", ""))
    screw.side = str(data.get("side", ""))
    screw.grade = str(data.get("grade", screw.grade))
    screw.breach_distance = _finite(
        data.get("breach_distance", screw.breach_distance), "breach_distance"
    )

    mean_hu = data.get("mean_hu")
    min_hu = data.get("min_hu")
    screw.mean_hu = None if mean_hu is None else _finite(mean_hu, "mean_hu")
    screw.min_hu = None if min_hu is None else _finite(min_hu, "min_hu")
    raw_warnings = data.get("warnings", [])
    screw.warnings = [str(w) for w in raw_warnings] if isinstance(raw_warnings, list) else []
    screw.source = str(data.get("source", "manual"))
    # Absent in v1/v2 payloads; an empty bundle simply means "not measured".
    screw.metrics = metrics_to_dict(data.get("metrics"))
    # Additive key: plans saved before it existed load as not reviewed.
    screw.reviewed = data.get("reviewed") is True

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
        distance=_finite(data.get("distance", 0.0), "distance"),
        mode=mode,
        angle=None if data.get("angle") is None else _finite(data["angle"], "angle"),
        label=str(data.get("label", "")),
    )


def serialize_plan(
    series_id: Optional[str],
    screws: List[Screw],
    measurements: List[Measurement],
    measurement_planes: Optional[List[Optional[str]]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    measurement_screw_aligned: Optional[List[bool]] = None,
) -> Dict[str, Any]:
    """Build plan payload dictionary for JSON persistence.

    ``metadata`` is an additive, free-form block describing how the plan was
    produced (currently the mask-refinement settings). It never affects how
    screws or measurements are read back, so a reader that does not know a key
    simply ignores it and the plan version stays at 3.

    ``series_id`` is stored only as a SHA-256 digest (``series_uid_sha256``),
    never as the raw UID.
    """
    planes = measurement_planes
    if planes is None:
        planes = [None] * len(measurements)
    if len(planes) != len(measurements):
        raise ValueError("Measurement plane count must match measurements")

    aligned = measurement_screw_aligned or [False] * len(measurements)
    if len(aligned) != len(measurements):
        raise ValueError("Screw-view flag count must match measurements")

    measurement_items = []
    for measurement, plane, screw_aligned in zip(measurements, planes, aligned, strict=True):
        item = measurement_to_dict(measurement)
        if plane is not None and plane not in VALID_PLANES:
            raise ValueError(f"Invalid measurement plane: {plane}")
        item["plane"] = plane
        if screw_aligned:
            # Taken on the oblique Screw MPR view: ``plane`` names the pane,
            # not a standard slice the points lie on.
            item["screw_aligned"] = True
        measurement_items.append(item)

    return {
        "version": PLAN_VERSION,
        "series_uid_sha256": series_uid_digest(series_id),
        "screws": [screw_to_dict(screw) for screw in screws],
        "measurements": measurement_items,
        "metadata": dict(metadata or {}),
    }


def deserialize_plan(
    payload: Dict[str, Any]
) -> Dict[str, Any]:
    """Parse plan payload into runtime objects."""
    if not isinstance(payload, dict):
        raise ValueError("Invalid plan payload")

    try:
        version = int(payload.get("version", PLAN_VERSION))
    except (TypeError, ValueError):
        raise ValueError("Invalid plan payload: version must be an integer") from None
    if version > PLAN_VERSION:
        raise ValueError(
            f"Plan file version {version} is newer than this app supports "
            f"(up to version {PLAN_VERSION}); update the app to open it"
        )

    screw_items = payload.get("screws", [])
    measurement_items = payload.get("measurements", [])
    if not isinstance(screw_items, list) or not isinstance(measurement_items, list):
        raise ValueError("Invalid plan payload: screws/measurements must be lists")

    raw_metadata = payload.get("metadata", {})
    if raw_metadata is not None and not isinstance(raw_metadata, dict):
        raise ValueError("Invalid plan payload: metadata must be an object")

    screws = [screw_from_dict(item) for item in screw_items]
    measurements: List[Measurement] = []
    planes: List[Optional[str]] = []
    screw_aligned: List[bool] = []

    for item in measurement_items:
        if not isinstance(item, dict):
            raise ValueError("Invalid measurement payload: expected object")
        plane = item.get("plane")
        if plane is not None and plane not in VALID_PLANES:
            raise ValueError(f"Invalid measurement plane: {plane}")
        measurements.append(measurement_from_dict(item))
        planes.append(plane)
        screw_aligned.append(item.get("screw_aligned") is True)

    return {
        "version": version,
        # Older plans hold the raw UID under ``series_id``; digest it here so
        # callers compare digests either way and never see (or re-save) it.
        "series_uid_sha256": payload.get("series_uid_sha256")
        or series_uid_digest(payload.get("series_id")),
        "screws": screws,
        "measurements": measurements,
        "measurement_planes": planes,
        "measurement_screw_aligned": screw_aligned,
        "metadata": dict(raw_metadata or {}),
    }


def write_text_atomic(path: str, text: str) -> None:
    """Write ``text`` to ``path`` through a temp file, never truncating on failure."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    # Temp file + os.replace: a crash or full disk never truncates the old file.
    fd, tmp_name = tempfile.mkstemp(dir=target.parent, prefix=target.name, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, target)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise


def save_plan_json(path: str, payload: Dict[str, Any]) -> None:
    """Write plan payload to JSON file."""
    write_text_atomic(path, json.dumps(payload, indent=2))


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
            "min_wall_mm", "heary_direction", "facet_grade", "trajectory_type",
            "warnings",
            # Appended, per the plan-file contract: readers key on the header.
            "pedicle_width_mm", "narrow_pedicle", "medial_breach_mm",
            "lateral_breach_mm",
            # Appended last, per the plan-file contract: existing readers index
            # by header name, and new columns must never shift an old one.
            "endplate_angle_deg",
            # ``narrow_pedicle`` is true for an untrustworthy width as well as
            # a small one, because both are planned under the same policy.
            # This column is what tells a reader which of the two it is
            # looking at, so a 24 mm "narrow" pedicle can be read correctly.
            "width_uncertain",
            # Appended last, per the plan-file contract: which endplate this
            # screw's angle was measured against ("own", "neighbours" or
            # "none") -- without it a reader cannot tell an aligned screw from
            # one that was never measured against an endplate at all.
            "endplate_reference",
        ])

        for index, screw in enumerate(screws, start=1):
            metrics = screw.metrics if isinstance(screw.metrics, dict) else {}
            writer.writerow([
                index,
                _csv_text(screw.vertebra_level),
                _csv_text(screw.side),
                _csv_text(screw.source),
                f"{screw.length:.3f}",
                f"{screw.diameter:.3f}",
                _csv_text(screw.grade),
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
                "" if metrics.get("heary_direction") is None else _csv_text(metrics["heary_direction"]),
                _metric_number(metrics, "facet_grade", "d"),
                # CBT and traditional screws are graded on the same scale but
                # are not clinically interchangeable, so the export names the
                # family; a manually placed screw belongs to neither and the
                # cell stays empty rather than guessing "traditional".
                "" if metrics.get("trajectory_type") is None else _csv_text(metrics["trajectory_type"]),
                _csv_text("|".join(screw.warnings)),
                _metric_number(metrics, "pedicle_width_mm"),
                _metric_flag(metrics, "narrow_pedicle"),
                _metric_number(metrics, "medial_breach_mm"),
                _metric_number(metrics, "lateral_breach_mm"),
                _metric_number(metrics, "endplate_angle_deg"),
                _metric_flag(metrics, "width_uncertain"),
                _csv_text(metrics.get("endplate_reference") or ""),
            ])
