"""
Tests for TotalSegmentator integration fallback behavior.
"""

import os
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

sitk = pytest.importorskip("SimpleITK")

import src.core.totalseg_integration as totalseg


def _create_test_image(value: int = 300):
    """Create a small constant CT-like volume for segmentation tests."""
    image = sitk.Image([8, 8, 8], sitk.sitkInt16)
    image = image + int(value)
    return image


def test_spine_roi_subset_has_no_cervical_levels():
    from src.core.totalseg_integration import SPINE_ROI_SUBSET
    assert not any(name.startswith("vertebrae_C") for name in SPINE_ROI_SUBSET)
    assert "vertebrae_L4" in SPINE_ROI_SUBSET and "sacrum" in SPINE_ROI_SUBSET


class TestTotalSegIntegration:
    """Coverage for fallback segmentation flow."""

    def test_frozen_app_uses_in_process_python_api(self, tmp_path, monkeypatch):
        calls = []
        fake_package = types.ModuleType("totalsegmentator")
        fake_package.__path__ = []
        fake_api = types.ModuleType("totalsegmentator.python_api")

        def fake_totalsegmentator(input_path, output_path, **kwargs):
            calls.append((input_path, output_path, kwargs))
            sitk.WriteImage(
                sitk.Image([2, 2, 2], sitk.sitkUInt8),
                str(output_path),
            )

        fake_api.totalsegmentator = fake_totalsegmentator
        monkeypatch.setitem(sys.modules, "totalsegmentator", fake_package)
        monkeypatch.setitem(
            sys.modules,
            "totalsegmentator.python_api",
            fake_api,
        )
        monkeypatch.setattr(totalseg.sys, "frozen", True, raising=False)
        monkeypatch.setattr(totalseg, "_ensure_torch_shm_executable", lambda: None)

        output = totalseg.run_totalsegmentator(
            sitk.Image([2, 2, 2], sitk.sitkInt16),
            str(tmp_path),
            device="cpu",
            roi_subset=["vertebrae_L4"],
            fast=True,
            force_split=True,
        )

        assert Path(output).exists()
        assert len(calls) == 1
        input_path, output_path, kwargs = calls[0]
        assert Path(input_path).name == "input_volume.nii.gz"
        assert Path(output_path).name == "totalseg_multilabel.nii.gz"
        assert kwargs == {
            "task": "total",
            "ml": True,
            "device": "cpu",
            "roi_subset": ["vertebrae_L4"],
            "fast": True,
            "force_split": True,
            "quiet": True,
            "nr_thr_saving": 1,
        }

    def test_threshold_fallback_creates_binary_mask(self, tmp_path):
        image = _create_test_image(300)

        mask_path = totalseg.run_threshold_fallback(
            image=image,
            work_dir=str(tmp_path),
        )

        assert os.path.exists(mask_path)
        mask = sitk.ReadImage(mask_path)
        values = set(sitk.GetArrayFromImage(mask).ravel().tolist())
        assert values.issubset({0, 1})
        assert 1 in values

    def test_totalsegmentator_memory_options_are_forwarded(
        self, tmp_path, monkeypatch
    ):
        captured = []

        def _fake_popen(command, **_kwargs):
            captured.extend(command)
            input_path = command[command.index("-i") + 1]
            output_path = command[command.index("-o") + 1]
            image = sitk.ReadImage(str(input_path))
            mask = sitk.Cast(image > 0, sitk.sitkUInt8)
            mask.CopyInformation(image)
            sitk.WriteImage(mask, str(output_path))
            return types.SimpleNamespace(
                returncode=0,
                communicate=lambda: ("", ""),
                poll=lambda: 0,
                terminate=lambda: None,
            )

        fake_spec = types.SimpleNamespace(origin="/tmp/TotalSegmentator.py")
        monkeypatch.setattr(totalseg.importlib.util, "find_spec", lambda _name: fake_spec)
        monkeypatch.setattr(totalseg.subprocess, "Popen", _fake_popen)
        monkeypatch.setattr(totalseg, "_ensure_torch_shm_executable", lambda: None)

        result_path = totalseg.run_totalsegmentator(
            image=_create_test_image(),
            work_dir=str(tmp_path),
            roi_subset=["vertebrae_L4"],
            fast=True,
            force_split=True,
        )

        assert Path(result_path).exists()
        assert captured[captured.index("-rs") + 1] == "vertebrae_L4"
        assert "-f" in captured
        assert "-fs" in captured
        assert captured[captured.index("-ns") + 1] == "1"

    def test_run_segmentation_uses_fallback_when_unavailable(self, tmp_path, monkeypatch):
        image = _create_test_image(300)
        monkeypatch.setattr(totalseg, "is_totalsegmentator_available", lambda: False)

        result = totalseg.run_segmentation_with_fallback(
            image=image,
            work_dir=str(tmp_path),
            task="total",
            device="cpu",
        )

        assert result.success is True
        assert result.method == "threshold_fallback"
        assert os.path.exists(result.mask_path)
        assert "not installed" in result.message.lower()

    def test_run_segmentation_falls_back_on_system_exit(self, tmp_path, monkeypatch):
        image = _create_test_image(300)
        monkeypatch.setattr(totalseg, "is_totalsegmentator_available", lambda: True)

        def _raise_system_exit(*args, **kwargs):
            raise SystemExit("license required")

        monkeypatch.setattr(totalseg, "run_totalsegmentator", _raise_system_exit)

        result = totalseg.run_segmentation_with_fallback(
            image=image,
            work_dir=str(tmp_path),
            task="body",
            device="cpu",
        )

        assert result.success is True
        assert result.method == "threshold_fallback"
        assert os.path.exists(result.mask_path)
        assert "fallback mask" in result.message.lower()

    def test_ensure_torch_shm_executable_sets_execute_bit(self, tmp_path, monkeypatch):
        torch_pkg = tmp_path / "torch"
        torch_pkg.mkdir()
        (torch_pkg / "__init__.py").write_text("", encoding="utf-8")

        bin_dir = torch_pkg / "bin"
        bin_dir.mkdir()
        shm = bin_dir / "torch_shm_manager"
        shm.write_text("dummy", encoding="utf-8")
        shm.chmod(0o644)

        fake_torch = types.SimpleNamespace(__file__=str(torch_pkg / "__init__.py"))
        monkeypatch.setitem(sys.modules, "torch", fake_torch)

        totalseg._ensure_torch_shm_executable()

        assert os.access(shm, os.X_OK)

    def test_check_segmentation_geometry_returns_empty_when_aligned(self):
        image = _create_test_image(300)
        mask = sitk.Cast(image > 0, sitk.sitkUInt8)
        mask.CopyInformation(image)

        warnings = totalseg.check_segmentation_geometry(
            reference_image=image,
            mask_image=mask,
        )

        assert warnings == []

    def test_check_segmentation_geometry_detects_spacing_mismatch(self):
        image = _create_test_image(300)
        image.SetSpacing((1.0, 1.0, 1.0))

        mask = sitk.Cast(image > 0, sitk.sitkUInt8)
        mask.CopyInformation(image)
        mask.SetSpacing((1.0, 1.5, 1.0))

        warnings = totalseg.check_segmentation_geometry(
            reference_image=image,
            mask_image=mask,
        )

        assert any("spacing mismatch" in item for item in warnings)

    def test_run_segmentation_reports_geometry_warning_on_mismatch(
        self, tmp_path, monkeypatch
    ):
        image = _create_test_image(300)
        monkeypatch.setattr(totalseg, "is_totalsegmentator_available", lambda: True)

        def _fake_totalseg(image, work_dir, **kwargs):
            output_path = Path(work_dir) / "fake_mismatch_mask.nii.gz"
            mask = sitk.Cast(image > 0, sitk.sitkUInt8)
            mask.CopyInformation(image)
            mask.SetOrigin((9.0, 0.0, 0.0))
            sitk.WriteImage(mask, str(output_path))
            return str(output_path)

        monkeypatch.setattr(totalseg, "run_totalsegmentator", _fake_totalseg)

        result = totalseg.run_segmentation_with_fallback(
            image=image,
            work_dir=str(tmp_path),
            task="total",
            device="cpu",
        )

        assert result.method == "totalsegmentator"
        assert result.geometry_warnings
        assert "geometry check warning" in result.message.lower()

    def test_preferred_device_uses_cuda_gpu_when_available(self, monkeypatch):
        fake_torch = types.SimpleNamespace(
            cuda=types.SimpleNamespace(is_available=lambda: True)
        )
        monkeypatch.setitem(sys.modules, "torch", fake_torch)

        assert totalseg.preferred_segmentation_device() == "gpu"

    def test_preferred_device_falls_back_to_cpu_without_cuda(self, monkeypatch):
        fake_torch = types.SimpleNamespace(
            cuda=types.SimpleNamespace(is_available=lambda: False)
        )
        monkeypatch.setitem(sys.modules, "torch", fake_torch)

        assert totalseg.preferred_segmentation_device() == "cpu"

    def test_gpu_failure_retries_totalsegmentator_on_cpu(
        self,
        tmp_path,
        monkeypatch,
    ):
        image = _create_test_image(300)
        devices = []
        force_split_values = []
        monkeypatch.setattr(totalseg, "is_totalsegmentator_available", lambda: True)

        def _fake_totalseg(image, work_dir, device, force_split, **_kwargs):
            devices.append(device)
            force_split_values.append(force_split)
            if device == "gpu":
                raise RuntimeError("CUDA out of memory")
            output_path = Path(work_dir) / "cpu_retry_mask.nii.gz"
            mask = sitk.Cast(image > 0, sitk.sitkUInt8)
            mask.CopyInformation(image)
            sitk.WriteImage(mask, str(output_path))
            return str(output_path)

        monkeypatch.setattr(totalseg, "run_totalsegmentator", _fake_totalseg)

        result = totalseg.run_segmentation_with_fallback(
            image=image,
            work_dir=str(tmp_path),
            task="total",
            device="gpu",
            fast=False,
        )

        assert result.method == "totalsegmentator"
        assert devices == ["gpu", "cpu"]
        assert force_split_values == [False, True]
        assert "CPU retry" in result.message


import time

from src.core.totalseg_integration import SegmentationWorkspace


def test_workspace_create_and_purge(tmp_path):
    ws = SegmentationWorkspace(root=str(tmp_path))
    first = ws.create()
    second = ws.create()
    (Path(first) / "input_volume.nii.gz").write_bytes(b"x")
    assert Path(first).name.startswith(SegmentationWorkspace.PREFIX)
    assert Path(second).is_dir()
    ws.purge()
    assert not Path(first).exists() and not Path(second).exists()
    ws.purge()  # idempotent


def test_purge_stale_removes_only_prefixed_dirs(tmp_path):
    stale = tmp_path / (SegmentationWorkspace.PREFIX + "abc")
    stale.mkdir()
    (stale / "input_volume.nii.gz").write_bytes(b"x")
    other = tmp_path / "unrelated"
    other.mkdir()
    old = time.time() - 10
    os.utime(stale, (old, old))
    removed = SegmentationWorkspace.purge_stale(root=str(tmp_path), older_than_seconds=5)
    assert removed == 1
    assert not stale.exists() and other.exists()


def test_workspace_create_writes_lock_file(tmp_path):
    ws = SegmentationWorkspace(root=str(tmp_path))
    created = ws.create()
    assert (Path(created) / ".lock").exists()


def test_workspace_touch_updates_lock_mtime(tmp_path):
    ws = SegmentationWorkspace(root=str(tmp_path))
    created = ws.create()
    lock_path = Path(created) / ".lock"
    old = time.time() - 100
    os.utime(lock_path, (old, old))
    SegmentationWorkspace.touch(created)
    assert lock_path.stat().st_mtime > old


def test_purge_stale_keeps_dir_with_fresh_lock_heartbeat(tmp_path):
    stale = tmp_path / (SegmentationWorkspace.PREFIX + "heartbeat")
    stale.mkdir()
    old = time.time() - 10
    os.utime(stale, (old, old))
    # Directory mtime is old, but the heartbeat lock file is fresh.
    SegmentationWorkspace.touch(str(stale))

    removed = SegmentationWorkspace.purge_stale(root=str(tmp_path), older_than_seconds=5)

    assert removed == 0
    assert stale.exists()


def _can_symlink(tmp_path: Path) -> bool:
    target = tmp_path / "_symlink_target_probe"
    target.mkdir(exist_ok=True)
    link = tmp_path / "_symlink_probe"
    try:
        os.symlink(target, link, target_is_directory=True)
    except (OSError, NotImplementedError):
        return False
    finally:
        if link.is_symlink():
            link.unlink()
    return True


def test_purge_stale_skips_symlinks(tmp_path):
    if not _can_symlink(tmp_path):
        pytest.skip("Platform/user does not support creating symlinks")

    real_target = tmp_path / "real_target"
    real_target.mkdir()
    old = time.time() - 10
    os.utime(real_target, (old, old))

    link = tmp_path / (SegmentationWorkspace.PREFIX + "link")
    os.symlink(real_target, link, target_is_directory=True)

    removed = SegmentationWorkspace.purge_stale(root=str(tmp_path), older_than_seconds=5)

    assert removed == 0
    assert link.is_symlink()
    assert real_target.exists()


class _FakePopen:
    """Minimal subprocess.Popen stand-in for cancellation tests."""

    def __init__(self, command, stdout=None, stderr=None, text=None,
                 on_communicate=None, returncode=0):
        self.command = list(command)
        self.terminated = False
        self.returncode = returncode
        self._on_communicate = on_communicate
        self._exited = False

    def poll(self):
        return self.returncode if self._exited else None

    def terminate(self):
        self.terminated = True

    def communicate(self):
        if self._on_communicate is not None:
            self._on_communicate(self)
        self._exited = True
        return "", ""


def test_process_holder_terminate_marks_cancelled():
    from src.core.totalseg_integration import ProcessHolder

    class Fake:
        def __init__(self):
            self.terminated = False

        def poll(self):
            return None

        def terminate(self):
            self.terminated = True

    holder = ProcessHolder()
    holder.process = Fake()
    holder.terminate()
    assert holder.cancelled is True and holder.process.terminated is True


def test_process_holder_terminate_without_process_is_safe():
    holder = totalseg.ProcessHolder()
    holder.terminate()
    assert holder.cancelled is True
    assert holder.process is None


def test_process_holder_does_not_terminate_finished_process():
    holder = totalseg.ProcessHolder()
    proc = _FakePopen(["x"], returncode=0)
    proc._exited = True
    holder.process = proc
    holder.terminate()
    assert holder.cancelled is True
    assert proc.terminated is False


def test_run_totalsegmentator_raises_cancelled_when_holder_cancelled(
    tmp_path, monkeypatch
):
    holder = totalseg.ProcessHolder()

    def _fake_popen(command, **kwargs):
        return _FakePopen(
            command,
            on_communicate=lambda proc: holder.terminate(),
            returncode=-15,
        )

    fake_spec = types.SimpleNamespace(origin="/tmp/TotalSegmentator.py")
    monkeypatch.setattr(totalseg.importlib.util, "find_spec", lambda _n: fake_spec)
    monkeypatch.setattr(totalseg.subprocess, "Popen", _fake_popen)
    monkeypatch.setattr(totalseg, "_ensure_torch_shm_executable", lambda: None)

    with pytest.raises(totalseg.SegmentationCancelled):
        totalseg.run_totalsegmentator(
            image=_create_test_image(),
            work_dir=str(tmp_path),
            process_holder=holder,
        )

    assert holder.process.terminated is True


def test_run_totalsegmentator_registers_process_on_holder(tmp_path, monkeypatch):
    holder = totalseg.ProcessHolder()
    created = []

    def _fake_popen(command, **kwargs):
        def _write(proc):
            input_path = command[command.index("-i") + 1]
            output_path = command[command.index("-o") + 1]
            image = sitk.ReadImage(str(input_path))
            mask = sitk.Cast(image > 0, sitk.sitkUInt8)
            mask.CopyInformation(image)
            sitk.WriteImage(mask, str(output_path))

        proc = _FakePopen(command, on_communicate=_write, returncode=0)
        created.append(proc)
        return proc

    fake_spec = types.SimpleNamespace(origin="/tmp/TotalSegmentator.py")
    monkeypatch.setattr(totalseg.importlib.util, "find_spec", lambda _n: fake_spec)
    monkeypatch.setattr(totalseg.subprocess, "Popen", _fake_popen)
    monkeypatch.setattr(totalseg, "_ensure_torch_shm_executable", lambda: None)

    result_path = totalseg.run_totalsegmentator(
        image=_create_test_image(),
        work_dir=str(tmp_path),
        process_holder=holder,
    )

    assert Path(result_path).exists()
    assert holder.process is created[0]


def test_frozen_run_raises_cancelled_before_starting(tmp_path, monkeypatch):
    started = []
    fake_package = types.ModuleType("totalsegmentator")
    fake_package.__path__ = []
    fake_api = types.ModuleType("totalsegmentator.python_api")
    fake_api.totalsegmentator = lambda *a, **k: started.append(a)
    monkeypatch.setitem(sys.modules, "totalsegmentator", fake_package)
    monkeypatch.setitem(sys.modules, "totalsegmentator.python_api", fake_api)
    monkeypatch.setattr(totalseg.sys, "frozen", True, raising=False)
    monkeypatch.setattr(totalseg, "_ensure_torch_shm_executable", lambda: None)

    holder = totalseg.ProcessHolder()
    holder.terminate()

    with pytest.raises(totalseg.SegmentationCancelled):
        totalseg.run_totalsegmentator(
            image=sitk.Image([2, 2, 2], sitk.sitkInt16),
            work_dir=str(tmp_path),
            device="cpu",
            process_holder=holder,
        )

    assert started == []


def test_fallback_runner_reraises_cancel(monkeypatch, tmp_path):
    import numpy as np

    monkeypatch.setattr(totalseg, "is_totalsegmentator_available", lambda: True)

    def fake_run(**kwargs):
        raise totalseg.SegmentationCancelled("cancelled")

    monkeypatch.setattr(totalseg, "run_totalsegmentator", fake_run)
    image = sitk.GetImageFromArray(np.zeros((4, 4, 4), np.int16))
    with pytest.raises(totalseg.SegmentationCancelled):
        totalseg.run_segmentation_with_fallback(
            image=image, work_dir=str(tmp_path), device="cpu"
        )


def test_fallback_runner_forwards_process_holder(monkeypatch, tmp_path):
    holders = []
    monkeypatch.setattr(totalseg, "is_totalsegmentator_available", lambda: True)

    def _fake_totalseg(image, work_dir, process_holder=None, **_kwargs):
        holders.append(process_holder)
        output_path = Path(work_dir) / "holder_mask.nii.gz"
        mask = sitk.Cast(image > 0, sitk.sitkUInt8)
        mask.CopyInformation(image)
        sitk.WriteImage(mask, str(output_path))
        return str(output_path)

    monkeypatch.setattr(totalseg, "run_totalsegmentator", _fake_totalseg)
    holder = totalseg.ProcessHolder()

    result = totalseg.run_segmentation_with_fallback(
        image=_create_test_image(300),
        work_dir=str(tmp_path),
        device="cpu",
        process_holder=holder,
    )

    assert result.method == "totalsegmentator"
    assert holders == [holder]


def test_cancel_on_gpu_attempt_does_not_retry_on_cpu(monkeypatch, tmp_path):
    devices = []
    monkeypatch.setattr(totalseg, "is_totalsegmentator_available", lambda: True)

    def _fake_totalseg(image, work_dir, device, **_kwargs):
        devices.append(device)
        raise totalseg.SegmentationCancelled("cancelled")

    monkeypatch.setattr(totalseg, "run_totalsegmentator", _fake_totalseg)

    with pytest.raises(totalseg.SegmentationCancelled):
        totalseg.run_segmentation_with_fallback(
            image=_create_test_image(300),
            work_dir=str(tmp_path),
            device="gpu",
        )

    assert devices == ["gpu"]


def test_run_totalsegmentator_does_not_spawn_when_already_cancelled(
    tmp_path, monkeypatch
):
    spawned = []

    def _fake_popen(command, **kwargs):
        spawned.append(command)
        return _FakePopen(command)

    fake_spec = types.SimpleNamespace(origin="/tmp/TotalSegmentator.py")
    monkeypatch.setattr(totalseg.importlib.util, "find_spec", lambda _n: fake_spec)
    monkeypatch.setattr(totalseg.subprocess, "Popen", _fake_popen)
    monkeypatch.setattr(totalseg, "_ensure_torch_shm_executable", lambda: None)

    holder = totalseg.ProcessHolder()
    holder.terminate()

    with pytest.raises(totalseg.SegmentationCancelled):
        totalseg.run_totalsegmentator(
            image=_create_test_image(),
            work_dir=str(tmp_path),
            process_holder=holder,
        )

    assert spawned == []


def test_cancel_racing_the_spawn_still_terminates_process(tmp_path, monkeypatch):
    holder = totalseg.ProcessHolder()

    def _fake_popen(command, **kwargs):
        # Simulate the user cancelling after Popen returned but before the
        # holder learned about the process.
        holder.terminate()
        return _FakePopen(command, returncode=-15)

    fake_spec = types.SimpleNamespace(origin="/tmp/TotalSegmentator.py")
    monkeypatch.setattr(totalseg.importlib.util, "find_spec", lambda _n: fake_spec)
    monkeypatch.setattr(totalseg.subprocess, "Popen", _fake_popen)
    monkeypatch.setattr(totalseg, "_ensure_torch_shm_executable", lambda: None)

    with pytest.raises(totalseg.SegmentationCancelled):
        totalseg.run_totalsegmentator(
            image=_create_test_image(),
            work_dir=str(tmp_path),
            process_holder=holder,
        )

    assert holder.process.terminated is True


def test_workspace_remove_deletes_only_the_given_dir(tmp_path):
    ws = SegmentationWorkspace(root=str(tmp_path))
    first = ws.create()
    second = ws.create()

    ws.remove(second)

    assert Path(first).exists()
    assert not Path(second).exists()
    assert second not in ws._dirs

    # The still-tracked directory is purged normally afterwards.
    ws.purge()
    assert not Path(first).exists()


def test_workspace_remove_is_safe_for_unknown_or_none_paths(tmp_path):
    ws = SegmentationWorkspace(root=str(tmp_path))
    tracked = ws.create()
    stranger = tmp_path / "not_tracked"
    stranger.mkdir()

    ws.remove(None)
    ws.remove(str(stranger))
    ws.remove(str(tmp_path / "missing"))

    assert Path(tracked).exists()
    assert not stranger.exists()
    assert ws._dirs == [tracked]


# ---------------------------------------------------------------------------
# Optional subregion (pedicle) stage
# ---------------------------------------------------------------------------


def _fake_totalseg_writer(tmp_path):
    """Return a run_totalsegmentator stand-in writing a valid multilabel mask."""
    import numpy as np

    def fake_total(**kwargs):
        path = tmp_path / "totalseg_multilabel.nii.gz"
        sitk.WriteImage(
            sitk.GetImageFromArray(np.zeros((4, 4, 4), np.uint8)), str(path)
        )
        return str(path)

    return fake_total


def test_subregion_failure_does_not_fail_run(monkeypatch, tmp_path):
    import numpy as np

    from src.core import totalseg_integration as ts
    from src.core.subregion_segmentation import SubregionModel

    monkeypatch.setattr(ts, "is_totalsegmentator_available", lambda: True)
    monkeypatch.setattr(ts, "run_totalsegmentator", _fake_totalseg_writer(tmp_path))

    def fake_sub(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(ts, "run_subregion_segmentation", fake_sub)

    model = SubregionModel(
        root=tmp_path,
        dataset_id="D",
        configuration="3d_fullres",
        labels={"pedicle": 2},
    )
    image = sitk.GetImageFromArray(np.zeros((4, 4, 4), np.int16))

    result = ts.run_segmentation_with_fallback(
        image=image,
        work_dir=str(tmp_path),
        device="cpu",
        subregion_model=model,
    )

    assert result.method == "totalsegmentator"
    assert result.subregion_mask_path is None
    assert "boom" in result.subregion_message


def test_subregion_success_populates_result(monkeypatch, tmp_path):
    import numpy as np

    from src.core import totalseg_integration as ts
    from src.core.subregion_segmentation import SubregionModel

    monkeypatch.setattr(ts, "is_totalsegmentator_available", lambda: True)
    monkeypatch.setattr(ts, "run_totalsegmentator", _fake_totalseg_writer(tmp_path))

    sub_path = tmp_path / "subregion.nii.gz"
    sitk.WriteImage(
        sitk.GetImageFromArray(np.zeros((4, 4, 4), np.uint8)), str(sub_path)
    )
    captured = {}

    def fake_sub(image, model, work_dir, device, progress_callback=None,
                 process_holder=None):
        captured["device"] = device
        captured["work_dir"] = work_dir
        return str(sub_path)

    monkeypatch.setattr(ts, "run_subregion_segmentation", fake_sub)

    model = SubregionModel(
        root=tmp_path,
        dataset_id="D",
        configuration="3d_fullres",
        labels={"pedicle": 2, "corpus": 1},
    )
    image = sitk.GetImageFromArray(np.zeros((4, 4, 4), np.int16))

    result = ts.run_segmentation_with_fallback(
        image=image,
        work_dir=str(tmp_path),
        device="cpu",
        subregion_model=model,
    )

    assert result.method == "totalsegmentator"
    assert result.subregion_mask_path == str(sub_path)
    assert result.subregion_labels == {"pedicle": 2, "corpus": 1}
    assert result.subregion_message == ""
    assert captured["device"] == "cpu"
    assert captured["work_dir"] == str(tmp_path)


def test_subregion_cancellation_propagates(monkeypatch, tmp_path):
    import numpy as np

    from src.core import totalseg_integration as ts
    from src.core.subregion_segmentation import SubregionModel

    monkeypatch.setattr(ts, "is_totalsegmentator_available", lambda: True)
    monkeypatch.setattr(ts, "run_totalsegmentator", _fake_totalseg_writer(tmp_path))

    def fake_sub(*args, **kwargs):
        raise totalseg.SegmentationCancelled("cancelled by user")

    monkeypatch.setattr(ts, "run_subregion_segmentation", fake_sub)

    model = SubregionModel(
        root=tmp_path,
        dataset_id="D",
        configuration="3d_fullres",
        labels={"pedicle": 2},
    )
    image = sitk.GetImageFromArray(np.zeros((4, 4, 4), np.int16))

    with pytest.raises(totalseg.SegmentationCancelled):
        ts.run_segmentation_with_fallback(
            image=image,
            work_dir=str(tmp_path),
            device="cpu",
            subregion_model=model,
        )


def test_no_subregion_model_leaves_fields_empty(monkeypatch, tmp_path):
    import numpy as np

    from src.core import totalseg_integration as ts

    monkeypatch.setattr(ts, "is_totalsegmentator_available", lambda: True)
    monkeypatch.setattr(ts, "run_totalsegmentator", _fake_totalseg_writer(tmp_path))

    def fail(*args, **kwargs):
        raise AssertionError("subregion stage must not run without a model")

    monkeypatch.setattr(ts, "run_subregion_segmentation", fail)

    image = sitk.GetImageFromArray(np.zeros((4, 4, 4), np.int16))
    result = ts.run_segmentation_with_fallback(
        image=image, work_dir=str(tmp_path), device="cpu"
    )

    assert result.subregion_mask_path is None
    assert result.subregion_labels == {}
    assert result.subregion_message == ""


# ---------------------------------------------------------------------------
# F4 -- a workspace owned by a live process survives another instance's purge
# ---------------------------------------------------------------------------


def _dead_pid() -> int:
    """A PID that is certainly not running any more.

    A child we spawned and reaped: on POSIX the PID is free, and on Windows the
    still-open process handle keeps the id from being recycled while the object
    itself reports its exit code.
    """
    import subprocess

    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait()
    return proc.pid


def test_workspace_create_records_the_owning_pid(tmp_path):
    ws = SegmentationWorkspace(root=str(tmp_path))
    created = ws.create()
    lock = Path(created) / SegmentationWorkspace.LOCK_NAME
    assert lock.read_text(encoding="utf-8").strip() == str(os.getpid())


def test_workspace_touch_keeps_the_owning_pid(tmp_path):
    ws = SegmentationWorkspace(root=str(tmp_path))
    created = ws.create()
    SegmentationWorkspace.touch(created)
    lock = Path(created) / SegmentationWorkspace.LOCK_NAME
    assert lock.read_text(encoding="utf-8").strip() == str(os.getpid())


def test_pid_is_alive_for_the_current_process():
    assert SegmentationWorkspace._pid_is_alive(os.getpid()) is True


def test_pid_is_alive_is_false_for_a_finished_process():
    assert SegmentationWorkspace._pid_is_alive(_dead_pid()) is False


def test_pid_is_alive_rejects_nonsense_pids():
    assert SegmentationWorkspace._pid_is_alive(0) is False
    assert SegmentationWorkspace._pid_is_alive(-1) is False


def test_purge_stale_keeps_a_workspace_owned_by_a_live_process(tmp_path):
    """A long CPU run in another instance must survive this instance's startup."""
    live = tmp_path / (SegmentationWorkspace.PREFIX + "live")
    live.mkdir()
    lock = live / SegmentationWorkspace.LOCK_NAME
    lock.write_text(str(os.getpid()), encoding="utf-8")
    old = time.time() - 7200
    os.utime(lock, (old, old))
    os.utime(live, (old, old))

    removed = SegmentationWorkspace.purge_stale(
        root=str(tmp_path), older_than_seconds=3600
    )

    assert removed == 0
    assert live.exists()


def test_purge_stale_removes_a_workspace_whose_owner_is_gone(tmp_path):
    abandoned = tmp_path / (SegmentationWorkspace.PREFIX + "abandoned")
    abandoned.mkdir()
    lock = abandoned / SegmentationWorkspace.LOCK_NAME
    lock.write_text(str(_dead_pid()), encoding="utf-8")
    old = time.time() - 7200
    os.utime(lock, (old, old))
    os.utime(abandoned, (old, old))

    removed = SegmentationWorkspace.purge_stale(
        root=str(tmp_path), older_than_seconds=3600
    )

    assert removed == 1
    assert not abandoned.exists()


def test_purge_stale_falls_back_to_mtime_for_an_unreadable_lock(tmp_path):
    """A lock from an older build (no PID) still ages out normally."""
    legacy = tmp_path / (SegmentationWorkspace.PREFIX + "legacy")
    legacy.mkdir()
    lock = legacy / SegmentationWorkspace.LOCK_NAME
    lock.write_text("not-a-pid", encoding="utf-8")
    old = time.time() - 7200
    os.utime(lock, (old, old))
    os.utime(legacy, (old, old))

    removed = SegmentationWorkspace.purge_stale(
        root=str(tmp_path), older_than_seconds=3600
    )

    assert removed == 1


def test_touch_all_refreshes_every_tracked_workspace(tmp_path):
    ws = SegmentationWorkspace(root=str(tmp_path))
    first = ws.create()
    second = ws.create()
    old = time.time() - 500
    for path in (first, second):
        lock = Path(path) / SegmentationWorkspace.LOCK_NAME
        os.utime(lock, (old, old))

    assert ws.touch_all() == 2

    for path in (first, second):
        lock = Path(path) / SegmentationWorkspace.LOCK_NAME
        assert lock.stat().st_mtime > old


def test_touch_all_ignores_directories_that_are_already_gone(tmp_path):
    ws = SegmentationWorkspace(root=str(tmp_path))
    created = ws.create()
    ws.remove(created)
    assert ws.touch_all() == 0


def test_purge_stale_removes_a_workspace_past_the_absolute_age_ceiling(tmp_path):
    """A recycled PID must not pin a dead workspace forever."""
    pinned = tmp_path / (SegmentationWorkspace.PREFIX + "pinned")
    pinned.mkdir()
    lock = pinned / SegmentationWorkspace.LOCK_NAME
    lock.write_text(str(os.getpid()), encoding="utf-8")   # a live PID
    ancient = time.time() - (SegmentationWorkspace.MAX_AGE_SECONDS + 60)
    os.utime(lock, (ancient, ancient))
    os.utime(pinned, (ancient, ancient))

    removed = SegmentationWorkspace.purge_stale(
        root=str(tmp_path), older_than_seconds=3600
    )

    assert removed == 1
    assert not pinned.exists()


def test_the_absolute_age_ceiling_is_seven_days():
    assert SegmentationWorkspace.MAX_AGE_SECONDS == 604800


def test_a_live_workspace_just_under_the_ceiling_is_still_kept(tmp_path):
    live = tmp_path / (SegmentationWorkspace.PREFIX + "long_run")
    live.mkdir()
    lock = live / SegmentationWorkspace.LOCK_NAME
    lock.write_text(str(os.getpid()), encoding="utf-8")
    recent = time.time() - (SegmentationWorkspace.MAX_AGE_SECONDS - 600)
    os.utime(lock, (recent, recent))
    os.utime(live, (recent, recent))

    removed = SegmentationWorkspace.purge_stale(
        root=str(tmp_path), older_than_seconds=3600
    )

    assert removed == 0
    assert live.exists()


# ---------------------------------------------------------------------------
# W2 -- TotalSegmentator output is refined before anyone measures it
# ---------------------------------------------------------------------------


def _blocky_vertebra_mask(image):
    """A label-28 blob on `image`'s grid, in one-voxel-deep steps."""
    import numpy as np

    size_x, size_y, size_z = image.GetSize()
    array = np.zeros((size_z, size_y, size_x), np.uint8)
    array[1:-1, 1:-1, 1:-1] = 28
    array[1, 1:-1, 1:-1] = 0                       # a step on the caudal face
    mask = sitk.GetImageFromArray(array)
    mask.CopyInformation(image)
    return mask


def _fake_blocky_totalseg(tmp_path):
    def fake_total(image, work_dir, **_kwargs):
        path = Path(work_dir) / "totalseg_multilabel.nii.gz"
        sitk.WriteImage(_blocky_vertebra_mask(image), str(path))
        return str(path)

    return fake_total


def _refinable_image():
    """A CT big enough for a 0.75 mm sigma to mean something."""
    image = sitk.Image([24, 24, 20], sitk.sitkInt16)
    image = image + 400
    image.SetSpacing((0.5, 0.5, 0.5))
    return image


def test_refinement_writes_a_refined_mask_and_keeps_the_raw_one(
    tmp_path, monkeypatch
):
    from src.core import totalseg_integration as ts

    monkeypatch.setattr(ts, "is_totalsegmentator_available", lambda: True)
    monkeypatch.setattr(ts, "run_totalsegmentator", _fake_blocky_totalseg(tmp_path))

    result = ts.run_segmentation_with_fallback(
        image=_refinable_image(), work_dir=str(tmp_path), device="cpu"
    )

    assert result.method == "totalsegmentator"
    assert Path(result.mask_path).name == ts.REFINED_MASK_NAME
    assert Path(result.mask_path).exists()
    assert Path(result.raw_mask_path).name == "totalseg_multilabel.nii.gz"
    assert Path(result.raw_mask_path).exists()
    assert result.refinement_notes


def test_refine_false_leaves_the_raw_mask_in_place(tmp_path, monkeypatch):
    from src.core import totalseg_integration as ts

    monkeypatch.setattr(ts, "is_totalsegmentator_available", lambda: True)
    monkeypatch.setattr(ts, "run_totalsegmentator", _fake_blocky_totalseg(tmp_path))

    def _never(*_args, **_kwargs):
        raise AssertionError("refinement must not run when refine=False")

    from src.core import mask_refinement

    monkeypatch.setattr(mask_refinement, "refine_vertebra_mask", _never)

    result = ts.run_segmentation_with_fallback(
        image=_refinable_image(),
        work_dir=str(tmp_path),
        device="cpu",
        refine=False,
    )

    assert Path(result.mask_path).name == "totalseg_multilabel.nii.gz"
    assert result.raw_mask_path is None
    assert result.refinement_notes == []


def test_the_threshold_fallback_is_never_refined(tmp_path, monkeypatch):
    from src.core import totalseg_integration as ts

    monkeypatch.setattr(ts, "is_totalsegmentator_available", lambda: False)

    def _never(*_args, **_kwargs):
        raise AssertionError("a threshold mask has no vertebra labels to refine")

    from src.core import mask_refinement

    monkeypatch.setattr(mask_refinement, "refine_vertebra_mask", _never)

    result = ts.run_segmentation_with_fallback(
        image=_refinable_image(), work_dir=str(tmp_path), device="cpu", refine=True
    )

    assert result.method == "threshold_fallback"
    assert result.raw_mask_path is None
    assert result.refinement_notes == []


def test_a_failing_refinement_keeps_the_raw_mask_and_notes_the_reason(
    tmp_path, monkeypatch
):
    """A blocky mask still plans screws; a crashed run plans none."""
    from src.core import mask_refinement
    from src.core import totalseg_integration as ts

    monkeypatch.setattr(ts, "is_totalsegmentator_available", lambda: True)
    monkeypatch.setattr(ts, "run_totalsegmentator", _fake_blocky_totalseg(tmp_path))

    def _boom(*_args, **_kwargs):
        raise MemoryError("not enough memory for the soft mask")

    monkeypatch.setattr(mask_refinement, "refine_vertebra_mask", _boom)

    result = ts.run_segmentation_with_fallback(
        image=_refinable_image(), work_dir=str(tmp_path), device="cpu"
    )

    assert result.success is True
    assert Path(result.mask_path).name == "totalseg_multilabel.nii.gz"
    assert result.raw_mask_path is None
    assert any("not enough memory" in note for note in result.refinement_notes)


def test_refinement_progress_reaches_the_callback(tmp_path, monkeypatch):
    from src.core import totalseg_integration as ts

    monkeypatch.setattr(ts, "is_totalsegmentator_available", lambda: True)
    monkeypatch.setattr(ts, "run_totalsegmentator", _fake_blocky_totalseg(tmp_path))
    messages = []

    ts.run_segmentation_with_fallback(
        image=_refinable_image(),
        work_dir=str(tmp_path),
        device="cpu",
        progress_callback=messages.append,
    )

    assert any("Refining mask boundaries" in message for message in messages)
