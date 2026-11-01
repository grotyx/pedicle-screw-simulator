"""Layered self-update: layer split, manifest diff, download check, apply plan."""

import hashlib
import json
import zipfile

import pytest

from src.core import app_update


def _fake_dist(root, app_text="app-1", runtime_text="rt-1", cuda_text="cuda-1"):
    (root / "_internal" / "torch" / "lib").mkdir(parents=True)
    (root / "_internal" / "vtkmodules").mkdir(parents=True)
    (root / "PedicleScrewSimulator.exe").write_text(app_text)
    (root / "_internal" / "base_library.zip").write_text("stdlib")
    (root / "_internal" / "vtkmodules" / "core.pyd").write_text(runtime_text)
    (root / "_internal" / "torch" / "lib" / "cudnn64_9.dll").write_text(cuda_text)
    return root


def test_files_split_into_cuda_runtime_and_app_layers():
    assert app_update.layer_of("_internal/torch/lib/cublas64_12.dll") == "cuda"
    assert app_update.layer_of("_internal/vtkmodules/core.pyd") == "runtime"
    assert app_update.layer_of("_internal/torch/nn/functional.py") == "runtime"
    assert app_update.layer_of("PedicleScrewSimulator.exe") == "app"
    assert app_update.layer_of("_internal/base_library.zip") == "app"
    assert app_update.layer_of("LICENSE") == "app"


def test_layer_ids_change_only_with_their_own_files(tmp_path):
    first = app_update.build_layers(_fake_dist(tmp_path / "a"), "0.2.5", tmp_path / "out-a")
    second = app_update.build_layers(_fake_dist(tmp_path / "b", app_text="app-2"), "0.2.6", tmp_path / "out-b")

    assert first["layers"]["cuda"]["id"] == second["layers"]["cuda"]["id"]
    assert first["layers"]["runtime"]["id"] == second["layers"]["runtime"]["id"]
    assert first["layers"]["app"]["id"] != second["layers"]["app"]["id"]
    # The installer ships the manifest so the installed copy knows its layers.
    shipped = json.loads((tmp_path / "b" / app_update.MANIFEST_NAME).read_text())
    assert shipped == second


def test_layer_archive_holds_its_files_and_matches_its_checksum(tmp_path):
    manifest = app_update.build_layers(_fake_dist(tmp_path / "d"), "0.2.6", tmp_path / "out")
    cuda = manifest["layers"]["cuda"]
    archive = tmp_path / "out" / cuda["archive"]

    assert hashlib.sha256(archive.read_bytes()).hexdigest() == cuda["sha256"]
    assert archive.stat().st_size == cuda["size"]
    with zipfile.ZipFile(archive) as zf:
        assert zf.namelist() == ["_internal/torch/lib/cudnn64_9.dll"]


def test_update_plan_downloads_only_the_changed_layers(tmp_path):
    old = app_update.build_layers(_fake_dist(tmp_path / "a"), "0.2.5", tmp_path / "out-a")
    new = app_update.build_layers(_fake_dist(tmp_path / "b", app_text="app-2"), "0.2.6", tmp_path / "out-b")

    plan = app_update.plan_update(old, new)

    assert plan.version == "0.2.6"
    assert plan.layers == ["app"]
    assert plan.download_bytes == new["layers"]["app"]["size"]


def test_no_update_when_the_server_is_not_newer(tmp_path):
    old = app_update.build_layers(_fake_dist(tmp_path / "a"), "0.2.6", tmp_path / "out-a")
    same = dict(old, version="0.2.6")
    older = dict(old, version="0.2.5")

    assert app_update.plan_update(old, same) is None
    assert app_update.plan_update(old, older) is None
    assert app_update.is_newer("0.2.10", "0.2.9")
    assert app_update.is_newer("0.2.6", "0.2.6rc1")
    assert not app_update.is_newer("0.2.6rc1", "0.2.6")


def test_a_download_with_the_wrong_checksum_is_rejected(tmp_path):
    manifest = app_update.build_layers(_fake_dist(tmp_path / "d"), "0.2.6", tmp_path / "out")
    layer = dict(manifest["layers"]["app"], sha256="0" * 64)
    base_url = (tmp_path / "out").as_uri()

    with pytest.raises(app_update.UpdateError, match="checksum"):
        app_update.download_layer(base_url, layer, tmp_path / "dl")
    assert not list((tmp_path / "dl").glob("*.zip"))


def test_staging_extracts_changed_layers_and_lists_files_to_remove(tmp_path):
    old = app_update.build_layers(_fake_dist(tmp_path / "a"), "0.2.5", tmp_path / "out-a")
    newdist = _fake_dist(tmp_path / "b", runtime_text="rt-2")
    (newdist / "_internal" / "vtkmodules" / "core.pyd").rename(newdist / "_internal" / "vtkmodules" / "core2.pyd")
    new = app_update.build_layers(newdist, "0.2.6", tmp_path / "out-b")
    plan = app_update.plan_update(old, new)
    archives = [app_update.download_layer((tmp_path / "out-b").as_uri(), new["layers"][n], tmp_path / "dl")
                for n in plan.layers]

    staging = app_update.stage_update(plan, old, new, archives, tmp_path / "staging")

    assert (staging / "files" / "_internal" / "vtkmodules" / "core2.pyd").read_text() == "rt-2"
    apply_plan = json.loads((staging / "apply.json").read_text())
    assert apply_plan["remove"] == ["_internal/vtkmodules/core.pyd"]
    assert apply_plan["version"] == "0.2.6"
    assert json.loads((staging / "files" / app_update.MANIFEST_NAME).read_text()) == new


def test_apply_script_waits_for_the_app_and_restores_on_failure(tmp_path):
    script = app_update.write_apply_script(tmp_path / "staging", tmp_path / "install", 4242,
                                           tmp_path / "install" / "PedicleScrewSimulator.exe")
    text = script.read_text(encoding="utf-8")

    assert "Wait-Process -Id 4242" in text
    assert "backup" in text and "catch" in text
    assert "DisplayVersion" in text


@pytest.mark.parametrize("mutate", [
    lambda m: m.update(version="0.2.6'; Remove-Item C:\\"),
    lambda m: m["layers"]["app"]["files"].append("../evil.dll"),
    lambda m: m["layers"]["app"]["files"].append("C:/Windows/evil.dll"),
    lambda m: m["layers"]["app"]["files"].append("_internal\\evil.dll"),
    lambda m: m["layers"]["app"].update(archive="../../outside.zip"),
])
def test_unsafe_manifest_is_rejected(tmp_path, mutate):
    manifest = app_update.build_layers(_fake_dist(tmp_path / "d"), "0.2.6", tmp_path / "out")
    mutate(manifest)
    with pytest.raises(app_update.UpdateError):
        app_update.validate_manifest(manifest)


def test_archive_must_hold_exactly_the_listed_files(tmp_path):
    local = app_update.build_layers(_fake_dist(tmp_path / "old"), "0.2.5", tmp_path / "old-out")
    remote = app_update.build_layers(_fake_dist(tmp_path / "new", app_text="app-2"), "0.2.6", tmp_path / "out")
    archive = tmp_path / "out" / remote["layers"]["app"]["archive"]
    with zipfile.ZipFile(archive, "a") as zf:
        zf.writestr("../escape.txt", "x")
    plan = app_update.plan_update(local, remote)
    with pytest.raises(app_update.UpdateError):
        app_update.stage_update(plan, local, remote, [archive], tmp_path / "staging")
