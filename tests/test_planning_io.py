"""
Tests for plan JSON/CSV persistence helpers.
"""

import csv
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.models.measurement import Measurement
from src.models.screw import Screw
from src.utils.planning_io import (
    deserialize_plan,
    export_screws_csv,
    load_plan_json,
    save_plan_json,
    serialize_plan,
)


class TestPlanningIO:
    """Roundtrip tests for plan persistence."""

    def test_plan_json_roundtrip(self, tmp_path):
        screws = [
            Screw(
                entry_point=(1.0, 2.0, 3.0),
                target_point=(1.0, 2.0, 33.0),
                diameter=6.0,
            )
        ]
        screws[0].grade = "B"

        measurements = [
            Measurement(
                points=[(0.0, 0.0, 0.0), (3.0, 4.0, 0.0)],
                distance=5.0,
                mode="distance",
                label="5.00 mm",
            )
        ]
        planes = ["axial"]

        payload = serialize_plan(
            series_id="1.2.840.test",
            screws=screws,
            measurements=measurements,
            measurement_planes=planes,
        )

        file_path = tmp_path / "plan.json"
        save_plan_json(str(file_path), payload)
        loaded_payload = load_plan_json(str(file_path))
        parsed = deserialize_plan(loaded_payload)

        assert parsed["series_id"] == "1.2.840.test"
        assert len(parsed["screws"]) == 1
        assert parsed["screws"][0].grade == "B"
        assert len(parsed["measurements"]) == 1
        assert parsed["measurements"][0].label == "5.00 mm"
        assert parsed["measurement_planes"] == ["axial"]

    def test_export_screws_csv(self, tmp_path):
        screws = [
            Screw(
                entry_point=(0.0, 0.0, 0.0),
                target_point=(0.0, 0.0, 30.0),
                diameter=5.0,
            )
        ]
        csv_path = tmp_path / "screws.csv"

        export_screws_csv(str(csv_path), screws)

        with csv_path.open("r", encoding="utf-8") as handle:
            rows = list(csv.reader(handle))

        assert rows[0][0] == "index"
        assert rows[0][4] == "length_mm"
        assert rows[1][0] == "1"
        assert rows[1][5] == "5.000"

    def test_serialize_plan_rejects_invalid_plane(self):
        measurements = [
            Measurement(
                points=[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)],
                distance=1.0,
                mode="distance",
                label="1.00 mm",
            )
        ]
        try:
            serialize_plan(
                series_id=None,
                screws=[],
                measurements=measurements,
                measurement_planes=["invalid_plane"],
            )
        except ValueError as exc:
            assert "Invalid measurement plane" in str(exc)
            return
        raise AssertionError("Expected ValueError for invalid plane")

    def test_metadata_round_trips_through_the_plan_file(self, tmp_path):
        payload = serialize_plan(
            series_id="SERIES-META",
            screws=[],
            measurements=[],
            metadata={"mask_refinement": {"enabled": True, "ct_guided": True}},
        )
        path = tmp_path / "plan.json"
        save_plan_json(str(path), payload)

        parsed = deserialize_plan(load_plan_json(str(path)))

        assert parsed["metadata"] == {
            "mask_refinement": {"enabled": True, "ct_guided": True}
        }

    def test_metadata_defaults_to_an_empty_object(self):
        payload = serialize_plan(series_id="S", screws=[], measurements=[])

        assert payload["metadata"] == {}
        assert deserialize_plan(payload)["metadata"] == {}

    def test_a_v3_plan_without_metadata_still_loads(self):
        """Every plan saved before this change has no metadata key at all."""
        parsed = deserialize_plan(
            {"version": 3, "series_id": "S", "screws": [], "measurements": []}
        )

        assert parsed["metadata"] == {}

    def test_non_object_metadata_is_rejected(self):
        with pytest.raises(ValueError, match="metadata"):
            deserialize_plan(
                {"version": 3, "series_id": "S", "screws": [], "measurements": [],
                 "metadata": ["not", "an", "object"]}
            )


def test_v2_roundtrip_preserves_metadata(tmp_path):
    from src.models.screw import Screw
    from src.utils.planning_io import (
        PLAN_VERSION,
        deserialize_plan,
        load_plan_json,
        save_plan_json,
        serialize_plan,
    )
    screw = Screw(entry_point=(1.0, 2.0, 3.0), target_point=(1.0, -30.0, 3.0), diameter=6.0,
                  vertebra_level="L4", side="left", grade="B", breach_distance=0.8,
                  mean_hu=210.0, min_hu=90.0, warnings=["note"], source="auto")
    payload = serialize_plan("series", [screw], [], [])
    assert payload["version"] == PLAN_VERSION
    path = tmp_path / "plan.json"
    save_plan_json(str(path), payload)
    parsed = deserialize_plan(load_plan_json(str(path)))
    loaded = parsed["screws"][0]
    assert loaded.mean_hu == 210.0 and loaded.min_hu == 90.0
    assert loaded.warnings == ["note"] and loaded.source == "auto"


def test_v3_roundtrip_metrics(tmp_path):
    from src.models.screw import Screw
    from src.utils.planning_io import (
        deserialize_plan,
        load_plan_json,
        save_plan_json,
        serialize_plan,
    )
    screw = Screw(entry_point=(1.0, 2.0, 3.0), target_point=(1.0, -30.0, 3.0), diameter=6.0,
                  vertebra_level="L4", side="left", grade="B", breach_distance=0.8,
                  mean_hu=210.0, min_hu=90.0, warnings=["note"], source="auto",
                  metrics={"facet_grade": 1, "heary_direction": "medial",
                           "trajectory_mean_hu": 180.5})
    payload = serialize_plan("series", [screw], [], [])
    assert payload["version"] == 3
    path = tmp_path / "plan.json"
    save_plan_json(str(path), payload)
    parsed = deserialize_plan(load_plan_json(str(path)))
    loaded = parsed["screws"][0]
    assert loaded.metrics == {"facet_grade": 1, "heary_direction": "medial",
                              "trajectory_mean_hu": 180.5}


def test_v2_payload_without_metrics_loads_empty_dict():
    from src.utils.planning_io import deserialize_plan
    payload = {"version": 2, "series_id": None, "measurements": [], "screws": [
        {"entry_point": [0, 0, 0], "target_point": [0, 0, 30], "length": 30, "diameter": 6.5,
         "vertebra_level": "L4", "side": "left", "grade": "A", "breach_distance": 0.0,
         "mean_hu": 300.0, "min_hu": 120.0, "warnings": [], "source": "auto"}]}
    parsed = deserialize_plan(payload)
    assert parsed["screws"][0].metrics == {}


def test_metrics_serialisation_is_json_safe():
    """Non-finite and non-scalar metric values must not reach the JSON file."""
    import json

    from src.models.screw import Screw
    from src.utils.planning_io import screw_from_dict, screw_to_dict

    screw = Screw(entry_point=(0.0, 0.0, 0.0), target_point=(0.0, 0.0, 30.0))
    screw.metrics = {
        "trajectory_mean_hu": float("nan"),
        "body_mean_hu": float("inf"),
        "facet_grade": 2,
        "facet_text": "no facet contact",
        "pedicle_mean_hu": None,
    }
    data = screw_to_dict(screw)
    encoded = json.dumps(data, allow_nan=False)   # raises if NaN/inf survived
    restored = screw_from_dict(json.loads(encoded))
    assert restored.metrics["trajectory_mean_hu"] is None
    assert restored.metrics["body_mean_hu"] is None
    assert restored.metrics["facet_grade"] == 2
    assert restored.metrics["facet_text"] == "no facet contact"
    assert restored.metrics["pedicle_mean_hu"] is None


def test_csv_has_metric_columns(tmp_path):
    import csv as _csv

    from src.models.screw import Screw
    from src.utils.planning_io import export_screws_csv
    screw = Screw(entry_point=(20, 30, 0), target_point=(12, -8, 0), side="left",
                  metrics={"trajectory_mean_hu": 180.5, "pedicle_mean_hu": 210.0,
                           "body_mean_hu": 150.0, "trajectory_body_ratio": 1.203,
                           "min_wall_mm": 1.25, "heary_direction": "medial",
                           "facet_grade": 2, "trajectory_type": "traditional"})
    path = tmp_path / "s.csv"
    export_screws_csv(str(path), [screw, Screw(entry_point=(0, 0, 0), target_point=(0, 0, 30))])
    with path.open("r", encoding="utf-8") as handle:
        rows = list(_csv.reader(handle))
    header = rows[0]
    for column in ("trajectory_mean_hu", "pedicle_mean_hu", "body_mean_hu", "hu_ratio",
                   "min_wall_mm", "heary_direction", "facet_grade", "trajectory_type"):
        assert column in header
    row = dict(zip(header, rows[1], strict=True))
    assert row["trajectory_mean_hu"] == "180.500"
    assert row["hu_ratio"] == "1.203"
    assert row["heary_direction"] == "medial"
    assert row["facet_grade"] == "2"
    assert row["trajectory_type"] == "traditional"
    # A screw with no metrics leaves the columns empty rather than shifting them.
    blank = dict(zip(header, rows[2], strict=True))
    assert blank["trajectory_mean_hu"] == "" and blank["facet_grade"] == ""
    assert blank["trajectory_type"] == ""


def test_csv_tolerates_non_numeric_metric_values(tmp_path):
    """A metric stored under a numeric key as text must not abort the export."""
    import csv as _csv

    from src.models.screw import Screw
    from src.utils.planning_io import export_screws_csv
    screw = Screw(entry_point=(0, 0, 0), target_point=(0, 0, 30),
                  metrics={"facet_grade": "grade two", "min_wall_mm": float("nan"),
                           "heary_direction": "medial"})
    path = tmp_path / "s.csv"
    export_screws_csv(str(path), [screw])       # must not raise
    with path.open("r", encoding="utf-8") as handle:
        rows = list(_csv.reader(handle))
    row = dict(zip(rows[0], rows[1], strict=True))
    assert row["facet_grade"] == ""
    assert row["min_wall_mm"] == ""
    assert row["heary_direction"] == "medial"


def test_v1_payload_without_metadata_loads():
    from src.utils.planning_io import deserialize_plan
    payload = {"version": 1, "series_id": None, "measurements": [], "screws": [
        {"entry_point": [0, 0, 0], "target_point": [0, 0, 30], "length": 30, "diameter": 6.5,
         "vertebra_level": "", "side": "", "grade": "A", "breach_distance": 0.0}]}
    parsed = deserialize_plan(payload)
    screw = parsed["screws"][0]
    assert screw.source == "manual" and screw.warnings == [] and screw.mean_hu is None


def test_csv_has_signed_angle_columns(tmp_path):
    from src.models.screw import Screw
    from src.utils.planning_io import export_screws_csv
    path = tmp_path / "s.csv"
    export_screws_csv(str(path), [Screw(entry_point=(20, 30, 0), target_point=(12, -8, 0), side="left")])
    header = path.read_text(encoding="utf-8").splitlines()[0]
    assert "convergence_angle_deg" in header and "craniocaudal_angle_deg" in header
    assert "mean_hu" in header and "source" in header


def test_payload_angles_are_recomputed_from_geometry():
    """Stale insertion_angle/medial_angle in a payload must be discarded and
    recomputed from entry/target/side, not trusted as-is (see screw_from_dict's
    call to screw._recompute_geometry())."""
    from src.core.screw_geometry import convergence_angle_deg, craniocaudal_angle_deg
    from src.utils.planning_io import screw_from_dict

    entry = [20.0, 30.0, 0.0]
    target = [12.0, -8.0, 0.0]

    payload_left = {
        "entry_point": entry, "target_point": target, "diameter": 6.0,
        "vertebra_level": "L4", "side": "left", "grade": "A", "breach_distance": 0.0,
        # Deliberately wrong stale values that must NOT survive deserialization.
        "insertion_angle": 168.0, "medial_angle": -90.0,
    }
    screw_left = screw_from_dict(payload_left)

    expected_medial_left = convergence_angle_deg(entry, target, "left")
    expected_craniocaudal = craniocaudal_angle_deg(entry, target)
    assert screw_left.medial_angle == pytest.approx(expected_medial_left)
    assert expected_medial_left == pytest.approx(11.9, abs=0.1)
    assert screw_left.insertion_angle == pytest.approx(expected_craniocaudal)
    assert expected_craniocaudal == pytest.approx(0.0, abs=1e-6)

    # Same geometry, opposite side: sign must flip, proving `side` is applied
    # before the recompute rather than the stale payload value being reused.
    payload_right = dict(payload_left, side="right")
    screw_right = screw_from_dict(payload_right)
    expected_medial_right = convergence_angle_deg(entry, target, "right")
    assert screw_right.medial_angle == pytest.approx(expected_medial_right)
    assert expected_medial_right == pytest.approx(-expected_medial_left)


def test_plan_payload_contains_no_patient_identifiers():
    """Plan JSON must carry only geometry/grading data, never DICOM PHI."""
    import json

    from src.models.screw import Screw
    from src.utils.planning_io import serialize_plan

    payload = serialize_plan(
        "1.2.3",
        [Screw(entry_point=(0, 0, 0), target_point=(0, 0, 30))],
        [],
        [],
    )
    text = json.dumps(payload).lower()
    for token in ("patient", "birth", "studydate", "accession"):
        assert token not in text


def test_score_components_round_trip_as_a_dict(tmp_path):
    """``score_components`` is a documented v3 field: it must stay a mapping.

    A nested dict used to fall through ``_jsonable_metric`` to ``str(value)``,
    so the plan file held a single-quoted Python literal that no JSON consumer
    — this app on reload included — could read back as numbers.
    """
    import json

    from src.models.screw import Screw
    from src.utils.planning_io import (
        deserialize_plan,
        load_plan_json,
        save_plan_json,
        serialize_plan,
    )

    screw = Screw(entry_point=(1.0, 2.0, 3.0), target_point=(1.0, -30.0, 3.0),
                  side="left", source="auto",
                  metrics={"score": 1.9,
                           "score_components": {"safety": 0.9, "density": 0.4},
                           "rod_misalignment_mm": 2.5})
    payload = serialize_plan("series", [screw], [], [])
    assert payload["version"] == 3
    # The in-memory payload must already be JSON-native, not a repr string.
    assert payload["screws"][0]["metrics"]["score_components"] == {
        "safety": 0.9, "density": 0.4,
    }

    path = tmp_path / "plan.json"
    save_plan_json(str(path), payload)
    raw = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(raw["screws"][0]["metrics"]["score_components"], dict)

    loaded = deserialize_plan(load_plan_json(str(path)))["screws"][0]
    components = loaded.metrics["score_components"]
    assert isinstance(components, dict)
    assert set(components) == {"safety", "density"}
    assert all(isinstance(value, float) for value in components.values())
    assert components["safety"] == pytest.approx(0.9)
    assert loaded.metrics["rod_misalignment_mm"] == pytest.approx(2.5)


def test_nested_metric_containers_are_normalised():
    """Numpy scalars and non-finite floats inside containers are coerced too."""
    import json

    import numpy as np

    from src.models.screw import Screw
    from src.utils.planning_io import screw_to_dict

    screw = Screw(entry_point=(0.0, 0.0, 0.0), target_point=(0.0, 0.0, 30.0))
    screw.metrics = {
        "score_components": {"safety": np.float64(0.75), "density": float("nan")},
        "candidate_scores": [np.float32(1.5), (np.int64(2), float("inf"))],
    }
    data = screw_to_dict(screw)
    json.dumps(data, allow_nan=False)          # raises if NaN/inf survived
    components = data["metrics"]["score_components"]
    assert components["safety"] == pytest.approx(0.75)
    assert components["density"] is None
    assert data["metrics"]["candidate_scores"] == [1.5, [2, None]]


def test_csv_has_trajectory_type_column(tmp_path):
    """The CSV must name the trajectory family; CBT vs traditional is clinical."""
    import csv as _csv

    from src.models.screw import Screw
    from src.utils.planning_io import export_screws_csv

    cbt = Screw(entry_point=(20, 30, 0), target_point=(12, -8, 0), side="left",
                metrics={"trajectory_type": "cbt"})
    manual = Screw(entry_point=(0, 0, 0), target_point=(0, 0, 30))
    path = tmp_path / "s.csv"
    export_screws_csv(str(path), [cbt, manual])
    with path.open("r", encoding="utf-8") as handle:
        rows = list(_csv.reader(handle))
    header = rows[0]
    assert "trajectory_type" in header
    assert dict(zip(header, rows[1], strict=True))["trajectory_type"] == "cbt"
    # Absent on a manual screw: an empty cell, not a shifted row.
    assert dict(zip(header, rows[2], strict=True))["trajectory_type"] == ""


def test_csv_carries_the_narrow_pedicle_columns(tmp_path):
    import csv as _csv

    from src.utils.planning_io import export_screws_csv

    narrow = Screw(
        entry_point=(20, 30, 0), target_point=(12, -8, 0), side="left",
        metrics={
            "pedicle_width_mm": 4.5,
            "narrow_pedicle": True,
            "medial_breach_mm": 0.0,
            "lateral_breach_mm": 1.5,
        },
    )
    manual = Screw(entry_point=(0, 0, 0), target_point=(0, 0, 30))
    path = tmp_path / "screws.csv"

    export_screws_csv(str(path), [narrow, manual])

    rows = list(_csv.reader(path.open(encoding="utf-8")))
    header = rows[0]
    assert header[-4:] == [
        "pedicle_width_mm", "narrow_pedicle", "medial_breach_mm", "lateral_breach_mm"
    ]
    row = dict(zip(header, rows[1], strict=True))
    assert row["pedicle_width_mm"] == "4.500"
    assert row["narrow_pedicle"] == "true"
    assert row["medial_breach_mm"] == "0.000"
    assert row["lateral_breach_mm"] == "1.500"

    blank = dict(zip(header, rows[2], strict=True))
    assert blank["pedicle_width_mm"] == ""
    assert blank["narrow_pedicle"] == ""


def test_plan_json_round_trips_the_narrow_metrics(tmp_path):
    from src.utils.planning_io import deserialize_plan, serialize_plan

    screw = Screw(
        entry_point=(20, 30, 0), target_point=(12, -8, 0), side="left",
        metrics={"narrow_pedicle": True, "pedicle_width_mm": 4.5,
                 "medial_breach_mm": 0.0, "lateral_breach_mm": 1.5},
    )

    parsed = deserialize_plan(serialize_plan("s", [screw], []))

    assert parsed["version"] == 3
    metrics = parsed["screws"][0].metrics
    assert metrics["narrow_pedicle"] is True
    assert metrics["pedicle_width_mm"] == 4.5
    assert metrics["lateral_breach_mm"] == 1.5
