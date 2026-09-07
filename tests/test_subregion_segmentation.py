import json
import sys

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


def test_parse_labels_exclusive_assignment_prefers_pedicle_over_corpus():
    roles = parse_dataset_labels({"labels": {"vertebral_body_pedicle": 1, "corpus": 2}})
    assert roles["pedicle"] == 1
    assert roles["corpus"] == 2


def test_parse_labels_whole_token_match_avoids_prefix_collision():
    roles = parse_dataset_labels({"labels": {"Pedicle": 2, "PedicleScrewChannel": 7}})
    assert roles["pedicle"] == 2


def test_find_model_rejects_malformed_json(tmp_path, monkeypatch):
    model_dir = tmp_path / "Dataset501_SpineSubregions" / "nnUNetTrainer__nnUNetPlans__3d_fullres"
    model_dir.mkdir(parents=True)
    (model_dir / "dataset.json").write_text("{not valid json")
    monkeypatch.delenv("PSS_SUBREGION_MODEL_DIR", raising=False)
    with pytest.raises(ValueError, match="not valid JSON"):
        find_subregion_model(str(model_dir))


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


def test_build_predict_command_maps_gpu_to_cuda(tmp_path):
    from src.core.subregion_segmentation import SubregionModel, build_predict_command
    model = SubregionModel(root=tmp_path, dataset_id="Dataset501_X", configuration="3d_fullres", labels={"pedicle": 2})
    cmd = build_predict_command(model, tmp_path / "in", tmp_path / "out", "gpu:0")
    assert cmd[0].endswith("nnUNetv2_predict") or cmd[:2] == [sys.executable, "-m"]
    assert "-d" in cmd and "Dataset501_X" in cmd and "-device" in cmd and "cuda" in cmd


def test_resample_to_reference_matches_geometry():
    import numpy as np
    import SimpleITK as sitk

    from src.core.subregion_segmentation import resample_to_reference
    ref = sitk.GetImageFromArray(np.zeros((20, 20, 20), np.int16))
    ref.SetSpacing((1.0, 1.0, 1.0))
    mask = sitk.GetImageFromArray(np.ones((10, 10, 10), np.uint8))
    mask.SetSpacing((2.0, 2.0, 2.0))
    out = resample_to_reference(mask, ref)
    assert out.GetSize() == ref.GetSize() and out.GetSpacing() == ref.GetSpacing()
    assert sitk.GetArrayFromImage(out).max() == 1


def test_run_subregion_segmentation_uses_process_holder(tmp_path, monkeypatch):
    import numpy as np
    import SimpleITK as sitk

    from src.core import subregion_segmentation as ss
    model = ss.SubregionModel(root=tmp_path / "Dataset501_X" / "cfg", dataset_id="Dataset501_X",
                              configuration="3d_fullres", labels={"pedicle": 2})
    calls = {}
    class FakeProc:
        returncode = 0
        def communicate(self):
            (tmp_path / "subregion_out").mkdir(exist_ok=True)
            sitk.WriteImage(sitk.GetImageFromArray(np.zeros((4, 4, 4), np.uint8)), str(tmp_path / "subregion_out" / "spine.nii.gz"))
            return "", ""
        def poll(self): return 0
    def fake_popen(cmd, **kwargs):
        calls["cmd"] = cmd
        calls["env"] = kwargs.get("env")
        return FakeProc()
    monkeypatch.setattr(ss.subprocess, "Popen", fake_popen)
    image = sitk.GetImageFromArray(np.zeros((4, 4, 4), np.int16))
    out = ss.run_subregion_segmentation(image, model, str(tmp_path), "cpu")
    assert out.endswith("spine.nii.gz")
    assert calls["env"]["nnUNet_results"] == str(tmp_path)
    assert (tmp_path / "subregion_in" / "spine_0000.nii.gz").exists()


def test_run_subregion_segmentation_refuses_frozen_build(tmp_path, monkeypatch):
    import numpy as np
    import SimpleITK as sitk

    from src.core import subregion_segmentation as ss
    model = ss.SubregionModel(root=tmp_path / "Dataset501_X" / "cfg", dataset_id="Dataset501_X",
                              configuration="3d_fullres", labels={"pedicle": 2})
    monkeypatch.setattr(ss.sys, "frozen", True, raising=False)
    image = sitk.GetImageFromArray(np.zeros((4, 4, 4), np.int16))
    with pytest.raises(RuntimeError, match="non-frozen"):
        ss.run_subregion_segmentation(image, model, str(tmp_path), "cpu")
    assert not (tmp_path / "subregion_in").exists()


def test_run_subregion_segmentation_cancelled_before_spawn(tmp_path, monkeypatch):
    import numpy as np
    import SimpleITK as sitk

    from src.core import subregion_segmentation as ss
    model = ss.SubregionModel(root=tmp_path / "Dataset501_X" / "cfg", dataset_id="Dataset501_X",
                              configuration="3d_fullres", labels={"pedicle": 2})

    def fake_popen(cmd, **kwargs):
        raise AssertionError("Popen should not be called once the run is already cancelled")
    monkeypatch.setattr(ss.subprocess, "Popen", fake_popen)

    holder = ss.ProcessHolder()
    holder.cancelled = True
    image = sitk.GetImageFromArray(np.zeros((4, 4, 4), np.int16))
    with pytest.raises(ss.SegmentationCancelled):
        ss.run_subregion_segmentation(image, model, str(tmp_path), "cpu", process_holder=holder)


def test_run_subregion_segmentation_nonzero_returncode_raises(tmp_path, monkeypatch):
    import numpy as np
    import SimpleITK as sitk

    from src.core import subregion_segmentation as ss
    model = ss.SubregionModel(root=tmp_path / "Dataset501_X" / "cfg", dataset_id="Dataset501_X",
                              configuration="3d_fullres", labels={"pedicle": 2})

    class FakeProc:
        returncode = 1
        def communicate(self): return "", "boom"
        def poll(self): return 1
    monkeypatch.setattr(ss.subprocess, "Popen", lambda cmd, **kwargs: FakeProc())

    image = sitk.GetImageFromArray(np.zeros((4, 4, 4), np.int16))
    with pytest.raises(RuntimeError, match="Subregion model failed"):
        ss.run_subregion_segmentation(image, model, str(tmp_path), "cpu")


def test_run_subregion_segmentation_missing_output_raises(tmp_path, monkeypatch):
    import numpy as np
    import SimpleITK as sitk

    from src.core import subregion_segmentation as ss
    model = ss.SubregionModel(root=tmp_path / "Dataset501_X" / "cfg", dataset_id="Dataset501_X",
                              configuration="3d_fullres", labels={"pedicle": 2})

    class FakeProc:
        returncode = 0
        def communicate(self): return "", ""
        def poll(self): return 0
    monkeypatch.setattr(ss.subprocess, "Popen", lambda cmd, **kwargs: FakeProc())

    image = sitk.GetImageFromArray(np.zeros((4, 4, 4), np.int16))
    with pytest.raises(RuntimeError, match="produced no output"):
        ss.run_subregion_segmentation(image, model, str(tmp_path), "cpu")
