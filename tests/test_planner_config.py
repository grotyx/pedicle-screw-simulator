import pytest

from src.core.planner_config import PlannerConfig


def test_defaults_match_constants():
    from src.utils import constants as c
    cfg = PlannerConfig()
    assert cfg.pedicle_fill_ratio == c.PEDICLE_FILL_RATIO
    assert cfg.anterior_margin_mm == c.ANTERIOR_SAFETY_MARGIN_MM
    assert cfg.trajectory_hu_threshold == c.TRAJECTORY_HU_LOOSENING_THRESHOLD


def test_roundtrip_mapping_ignores_unknown_keys():
    cfg = PlannerConfig(pedicle_fill_ratio=0.7, anterior_margin_mm=5.0)
    data = cfg.to_mapping()
    data["unknown"] = 1
    assert PlannerConfig.from_mapping(data) == cfg


@pytest.mark.parametrize("field,value", [("pedicle_fill_ratio", 1.2), ("pedicle_fill_ratio", 0.2),
                                         ("wall_clearance_mm", -1.0), ("anterior_margin_mm", 30.0),
                                         ("max_convergence_deg", 91.0),
                                         ("narrow_pedicle_mm", 2.5), ("narrow_pedicle_mm", 8.5),
                                         ("narrow_lateral_breach_mm", -0.1),
                                         ("narrow_lateral_breach_mm", 6.5)])
def test_validate_rejects_out_of_range(field, value):
    with pytest.raises(ValueError):
        PlannerConfig(**{field: value}).validate()


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

    def test_round_trips_through_to_mapping(self):
        config = PlannerConfig(endplate_parallel=False, endplate_tolerance_deg=7.5)
        restored = PlannerConfig.from_mapping(config.to_mapping())
        assert restored.endplate_parallel is False
        assert restored.endplate_tolerance_deg == pytest.approx(7.5)


class TestContainmentBar:
    def test_grade_b_opt_in_defaults_off(self):
        assert PlannerConfig().accept_grade_b is False

    def test_from_mapping_reads_qsettings_style_booleans(self):
        assert PlannerConfig.from_mapping({"accept_grade_b": "true"}).accept_grade_b is True
        assert PlannerConfig.from_mapping({"accept_grade_b": "false"}).accept_grade_b is False
        assert PlannerConfig.from_mapping({"accept_grade_b": True}).accept_grade_b is True

    def test_new_fields_round_trip(self):
        config = PlannerConfig(accept_grade_b=True, width_bound_disagreement_mm=2.0)
        restored = PlannerConfig.from_mapping(config.to_mapping())
        assert restored.accept_grade_b is True
        assert restored.width_bound_disagreement_mm == pytest.approx(2.0)

    def test_bound_disagreement_is_validated(self):
        PlannerConfig(width_bound_disagreement_mm=0.0).validate()
        PlannerConfig(width_bound_disagreement_mm=5.0).validate()
        with pytest.raises(ValueError, match="width_bound_disagreement_mm"):
            PlannerConfig(width_bound_disagreement_mm=5.5).validate()
        with pytest.raises(ValueError, match="width_bound_disagreement_mm"):
            PlannerConfig(width_bound_disagreement_mm=-0.1).validate()


class TestCatalogueHygiene:
    def test_from_mapping_sorts_unsorted_catalogues(self):
        cfg = PlannerConfig.from_mapping(
            {
                "implant_diameters_mm": [6.5, 4.0, 5.5, 4.5],
                "implant_lengths_mm": [45.0, 25.0, 35.0],
            }
        )
        assert list(cfg.implant_diameters_mm) == [4.0, 4.5, 5.5, 6.5]
        assert list(cfg.implant_lengths_mm) == [25.0, 35.0, 45.0]
        cfg.validate()

    @pytest.mark.parametrize(
        "field",
        ["implant_diameters_mm", "implant_lengths_mm"],
    )
    def test_direct_unsorted_construction_raises(self, field):
        with pytest.raises(ValueError, match="ascending"):
            PlannerConfig(**{field: (6.5, 4.0, 5.5)}).validate()

    def test_empty_catalogue_still_rejected(self):
        with pytest.raises(ValueError, match="must not be empty"):
            PlannerConfig(implant_diameters_mm=()).validate()
