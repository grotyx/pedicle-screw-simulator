import json

import pytest

from src.core.subregion_segmentation import find_subregion_model, parse_dataset_labels


def test_parse_labels_maps_roles():
    labels = {"background": 0, "Corpus": 1, "Pedicle": 2, "Lamina": 3, "SpinousProcess": 4,
              "TransverseProcess": 5, "ArticularProcess": 6}
    roles = parse_dataset_labels({"labels": labels})
    assert roles == {"corpus": 1, "pedicle": 2, "lamina": 3, "spinous": 4, "transverse": 5, "articular": 6}


def test_parse_labels_requires_pedicle():
    with pytest.raises(ValueError):
        parse_dataset_labels({"labels": {"background": 0, "corpus": 1}})


def test_find_model_from_directory(tmp_path, monkeypatch):
    model_dir = tmp_path / "Dataset501_SpineSubregions" / "nnUNetTrainer__nnUNetPlans__3d_fullres"
    model_dir.mkdir(parents=True)
    (model_dir / "dataset.json").write_text(json.dumps({"labels": {"background": 0, "pedicle": 2, "corpus": 1}}))
    monkeypatch.delenv("PSS_SUBREGION_MODEL_DIR", raising=False)
    assert find_subregion_model() is None
    model = find_subregion_model(str(model_dir))
    assert model.dataset_id == "Dataset501_SpineSubregions"
    assert model.configuration == "3d_fullres"
    assert model.labels["pedicle"] == 2
    monkeypatch.setenv("PSS_SUBREGION_MODEL_DIR", str(model_dir))
    assert find_subregion_model().root == model_dir
