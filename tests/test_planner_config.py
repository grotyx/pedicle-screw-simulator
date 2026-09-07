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
                                         ("max_convergence_deg", 91.0)])
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
