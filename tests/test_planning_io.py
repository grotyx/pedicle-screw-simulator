"""
Tests for plan JSON/CSV persistence helpers.
"""

import csv
import os
import sys

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
        assert rows[0][1] == "length_mm"
        assert rows[1][0] == "1"
        assert rows[1][2] == "5.000"

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
        assert False, "Expected ValueError for invalid plane"
