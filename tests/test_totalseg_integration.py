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

        def _fake_run(command, **_kwargs):
            captured.extend(command)
            input_path = command[command.index("-i") + 1]
            output_path = command[command.index("-o") + 1]
            image = sitk.ReadImage(str(input_path))
            mask = sitk.Cast(image > 0, sitk.sitkUInt8)
            mask.CopyInformation(image)
            sitk.WriteImage(mask, str(output_path))
            return types.SimpleNamespace(returncode=0, stdout="", stderr="")

        fake_spec = types.SimpleNamespace(origin="/tmp/TotalSegmentator.py")
        monkeypatch.setattr(totalseg.importlib.util, "find_spec", lambda _name: fake_spec)
        monkeypatch.setattr(totalseg.subprocess, "run", _fake_run)
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
