import pytest

from src.core.planner_config import PlannerConfig


def test_defaults_match_constants():
    from src.utils import constants as c
    cfg = PlannerConfig()
    assert cfg.pedicle_fill_ratio == c.PEDICLE_FILL_RATIO
    assert cfg.anterior_margin_mm == c.ANTERIOR_SAFETY_MARGIN_MM
    assert cfg.trajectory_hu_threshold == c.TRAJECTORY_HU_LOOSENING_THRESHOLD


def test_implant_catalogues_are_sorted_on_construction():
    cfg = PlannerConfig(implant_lengths_mm=(45, 30.0, 55.0), implant_diameters_mm=(6.5, 4.0, 5.5))
    assert cfg.implant_lengths_mm == (30.0, 45.0, 55.0)
    assert cfg.implant_diameters_mm == (4.0, 5.5, 6.5)
    assert PlannerConfig.from_mapping({"implant_lengths_mm": [50, 25]}).implant_lengths_mm == (
        25.0,
        50.0,
    )


@pytest.mark.parametrize("field", ["implant_lengths_mm", "implant_diameters_mm"])
def test_validate_rejects_a_non_positive_catalogue_size(field):
    with pytest.raises(ValueError):
        PlannerConfig(**{field: (0.0, 5.0)}).validate()


def test_roundtrip_mapping_ignores_unknown_keys():
    cfg = PlannerConfig(pedicle_fill_ratio=0.7, anterior_margin_mm=5.0)
    data = cfg.to_mapping()
    data["unknown"] = 1
    assert PlannerConfig.from_mapping(data) == cfg


@pytest.mark.parametrize("field,value", [("pedicle_fill_ratio", 1.2), ("pedicle_fill_ratio", 0.2),
                                         ("wall_clearance_mm", -1.0), ("anterior_margin_mm", 30.0),
                                         ("max_convergence_deg", 91.0),
                                         ("narrow_pedicle_mm", 2.5), ("narrow_pedicle_mm", 8.5)])
def test_validate_rejects_out_of_range(field, value):
    with pytest.raises(ValueError):
        PlannerConfig(**{field: value}).validate()


class TestAcceptedBreachGrade:
    """Owner policy: no breach unless the user explicitly picks how far."""

    def test_default_accepts_no_breach(self):
        config = PlannerConfig()
        assert config.accepted_breach_grade == "A"
        assert config.accepted_grades == ("A",)
        assert not hasattr(config, "narrow_lateral_breach_mm")

    @pytest.mark.parametrize("grade,accepted", [
        ("A", ("A",)), ("B", ("A", "B")), ("C", ("A", "B", "C")),
    ])
    def test_accepted_grades_run_from_a_to_the_choice(self, grade, accepted):
        assert PlannerConfig(accepted_breach_grade=grade).accepted_grades == accepted

    @pytest.mark.parametrize("value", ["D", "E", "a", "", "2.0"])
    def test_validate_rejects_anything_but_a_b_or_c(self, value):
        with pytest.raises(ValueError, match="accepted_breach_grade"):
            PlannerConfig(accepted_breach_grade=value).validate()

    @pytest.mark.parametrize("grade", ["A", "B", "C"])
    @pytest.mark.parametrize(
        "breach", [0.0, 1e-6, 1.5, 1.999, 2.0, 2.5, 3.0, 3.999, 4.0, 4.5, 6.0]
    )
    def test_accepts_breach_matches_the_grading_boundaries(self, grade, breach):
        """"Up to B" is exactly the range graded B, strict and inclusive edges alike."""
        import numpy as np

        from src.core.screw_grading import ScrewGrader

        config = PlannerConfig(accepted_breach_grade=grade)
        expected = ScrewGrader.grade_from_breach(breach) in config.accepted_grades
        assert bool(config.accepts_breach(breach)) is expected
        # Vectorised for the optimiser's batches, with the same answer.
        assert config.accepts_breach(np.array([breach, breach])).tolist() == [expected] * 2

    def test_the_lateral_limit_is_the_upper_edge_of_the_grade(self):
        assert PlannerConfig().lateral_breach_limit_mm == 0.0
        assert PlannerConfig(accepted_breach_grade="B").lateral_breach_limit_mm == 2.0
        assert PlannerConfig(accepted_breach_grade="C").lateral_breach_limit_mm == 4.0

    def test_round_trips_through_to_mapping(self):
        config = PlannerConfig(accepted_breach_grade="C")
        assert PlannerConfig.from_mapping(config.to_mapping()) == config

    def test_an_old_lateral_cap_setting_is_ignored(self):
        """A saved 2.0 mm cap must not silently keep breach acceptance."""
        restored = PlannerConfig.from_mapping({"narrow_lateral_breach_mm": 2.0})
        assert restored.accepted_breach_grade == "A"
        assert restored == PlannerConfig()


def test_optimizer_defaults_and_weight_roundtrip():
    from src.core.trajectory_optimizer import OptimizerWeights

    cfg = PlannerConfig()
    assert cfg.mode == "optimizer"
    assert cfg.trajectory == "traditional"
    assert cfg.weights == OptimizerWeights()

    tuned = PlannerConfig(mode="legacy", trajectory="cbt",
                          weights=OptimizerWeights(safety=1.5, density=0.25, rod=0.1))
    data = tuned.to_mapping()
    assert data["weights"]["safety"] == pytest.approx(1.5)
    assert isinstance(data["weights"], dict)
    assert PlannerConfig.from_mapping(data) == tuned


@pytest.mark.parametrize("field,value", [("mode", "magic"), ("trajectory", "diagonal")])
def test_validate_rejects_unknown_enum(field, value):
    with pytest.raises(ValueError):
        PlannerConfig(**{field: value}).validate()


class TestEndplateOption:
    def test_defaults_are_on_at_ten_degrees(self):
        config = PlannerConfig()
        assert config.endplate_parallel is True
        assert config.endplate_tolerance_deg == pytest.approx(10.0)

    def test_tolerance_is_validated(self):
        PlannerConfig(endplate_tolerance_deg=0.0).validate()
        PlannerConfig(endplate_tolerance_deg=30.0).validate()
        with pytest.raises(ValueError, match="endplate_tolerance_deg"):
            PlannerConfig(endplate_tolerance_deg=30.5).validate()
        with pytest.raises(ValueError, match="endplate_tolerance_deg"):
            PlannerConfig(endplate_tolerance_deg=-1.0).validate()

    def test_from_mapping_reads_qsettings_style_booleans(self):
        """QSettings hands back "false" as a string; bool("false") is True."""
        assert PlannerConfig.from_mapping({"endplate_parallel": "false"}).endplate_parallel is False
        assert PlannerConfig.from_mapping({"endplate_parallel": "true"}).endplate_parallel is True
        assert PlannerConfig.from_mapping({"endplate_parallel": "0"}).endplate_parallel is False
        assert PlannerConfig.from_mapping({"endplate_parallel": False}).endplate_parallel is False

    def test_uncontained_narrow_is_off_by_default_and_round_trips(self):
        assert PlannerConfig().place_uncontained_narrow is False
        read = PlannerConfig.from_mapping
        assert read({"place_uncontained_narrow": "false"}).place_uncontained_narrow is False
        assert read({"place_uncontained_narrow": "true"}).place_uncontained_narrow is True
        restored = read(PlannerConfig(place_uncontained_narrow=True).to_mapping())
        assert restored.place_uncontained_narrow is True

    def test_round_trips_through_to_mapping(self):
        config = PlannerConfig(endplate_parallel=False, endplate_tolerance_deg=7.5)
        restored = PlannerConfig.from_mapping(config.to_mapping())
        assert restored.endplate_parallel is False
        assert restored.endplate_tolerance_deg == pytest.approx(7.5)
