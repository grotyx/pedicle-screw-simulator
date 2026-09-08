"""
TotalSegmentator integration helpers with safe fallback behavior.
"""

import importlib.util
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Dict, List, Optional

import SimpleITK as sitk

if TYPE_CHECKING:  # pragma: no cover - typing only, avoids an import cycle
    from src.core.subregion_segmentation import SubregionModel


SPINE_ROI_SUBSET = (
    "sacrum",
    "vertebrae_S1",
    "vertebrae_L5",
    "vertebrae_L4",
    "vertebrae_L3",
    "vertebrae_L2",
    "vertebrae_L1",
    "vertebrae_T12",
    "vertebrae_T11",
    "vertebrae_T10",
    "vertebrae_T9",
    "vertebrae_T8",
    "vertebrae_T7",
    "vertebrae_T6",
    "vertebrae_T5",
    "vertebrae_T4",
    "vertebrae_T3",
    "vertebrae_T2",
    "vertebrae_T1",
)


class SegmentationCancelled(RuntimeError):
    """Raised when the user cancels a running segmentation."""


class ProcessHolder:
    """Shared handle to the running TotalSegmentator subprocess.

    The worker thread stores the live ``Popen`` here so the GUI thread can
    terminate it when the user presses Cancel.
    """

    def __init__(self) -> None:
        self.process: Optional[subprocess.Popen] = None
        self.cancelled: bool = False

    def terminate(self) -> None:
        """Mark the run as cancelled and terminate the process if running.

        On Windows this kills only the TotalSegmentator process itself; the
        nnU-Net worker children it spawned are not in a job object and may
        outlive the call until they finish on their own.
        """
        self.cancelled = True
        proc = self.process
        if proc is not None and proc.poll() is None:
            proc.terminate()


class SegmentationWorkspace:
    """Owns temporary directories that hold patient volumes and masks."""

    PREFIX = "screwfix_totalseg_"
    LOCK_NAME = ".lock"

    def __init__(self, root: Optional[str] = None) -> None:
        self._root = root or tempfile.gettempdir()
        self._dirs: List[str] = []

    def create(self) -> str:
        path = tempfile.mkdtemp(prefix=self.PREFIX, dir=self._root)
        self._dirs.append(path)
        self.touch(path)
        return path

    def remove(self, path: Optional[str]) -> None:
        """Delete one workspace directory, leaving the other runs intact.

        Used when a single run is abandoned (cancelled) and the directories of
        earlier, still-referenced runs must survive.
        """
        if not path:
            return
        shutil.rmtree(path, ignore_errors=True)
        try:
            self._dirs.remove(path)
        except ValueError:
            # Not tracked by this workspace; the rmtree above was still valid.
            pass

    def purge(self) -> None:
        while self._dirs:
            shutil.rmtree(self._dirs.pop(), ignore_errors=True)

    @classmethod
    def touch(cls, path: str) -> bool:
        """Refresh the heartbeat lock of `path`, restamping the owning PID.

        Written by the process that owns the directory, so the PID recorded is
        always the one whose liveness :meth:`purge_stale` should be asking
        about. Returns whether the heartbeat landed.
        """
        lock_path = Path(path) / cls.LOCK_NAME
        try:
            lock_path.write_text(str(os.getpid()), encoding="utf-8")
        except OSError:
            # Best effort only; a missed heartbeat just risks a stale purge.
            return False
        return True

    def touch_all(self) -> int:
        """Refresh every workspace this instance still owns; return the count.

        The controller calls this on a timer: TotalSegmentator emits nothing
        between spawning the subprocess and collecting its output, and the
        finished mask keeps being read out of the directory long after the run,
        so progress messages alone leave hours-long gaps in the heartbeat.
        """
        return sum(1 for path in list(self._dirs) if self.touch(path))

    #: ``PROCESS_QUERY_LIMITED_INFORMATION`` — the least privilege that still
    #: answers "is this process running?", and one every same-session process
    #: can be opened with.
    _PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
    _STILL_ACTIVE = 259          # GetExitCodeProcess for a running process
    _ERROR_ACCESS_DENIED = 5

    @classmethod
    def _pid_is_alive(cls, pid: int) -> bool:
        """Whether `pid` is a process that is still running.

        Errs towards "alive": a PID we cannot interrogate must not cost another
        instance its live workspace.  Windows goes through ``OpenProcess`` +
        ``GetExitCodeProcess`` because ``os.kill(pid, 0)`` *terminates* the
        target there rather than probing it.
        """
        if pid <= 0:
            return False
        if os.name == "nt":
            return cls._windows_pid_is_alive(pid)
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            # Someone else's process: it exists, which is all we asked.
            return True
        except OSError:
            return False
        return True

    @classmethod
    def _windows_pid_is_alive(cls, pid: int) -> bool:
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.OpenProcess.argtypes = [
            wintypes.DWORD, wintypes.BOOL, wintypes.DWORD
        ]
        kernel32.OpenProcess.restype = wintypes.HANDLE
        kernel32.GetExitCodeProcess.argtypes = [
            wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)
        ]
        kernel32.GetExitCodeProcess.restype = wintypes.BOOL
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel32.CloseHandle.restype = wintypes.BOOL

        handle = kernel32.OpenProcess(
            cls._PROCESS_QUERY_LIMITED_INFORMATION, False, pid
        )
        if not handle:
            # Access denied means the process is there but not ours to inspect.
            return ctypes.get_last_error() == cls._ERROR_ACCESS_DENIED
        try:
            exit_code = wintypes.DWORD()
            if not kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
                return True
            return exit_code.value == cls._STILL_ACTIVE
        finally:
            kernel32.CloseHandle(handle)

    @classmethod
    def _lock_owner_pid(cls, lock_path: Path) -> Optional[int]:
        """PID recorded in a lock file, or None for a legacy/unreadable lock."""
        try:
            text = lock_path.read_text(encoding="utf-8").strip()
        except (OSError, UnicodeDecodeError):
            return None
        try:
            return int(text)
        except ValueError:
            return None

    @classmethod
    def purge_stale(cls, root: Optional[str] = None, older_than_seconds: float = 0.0) -> int:
        base = Path(root or tempfile.gettempdir())
        removed = 0
        now = time.time()
        for entry in base.glob(cls.PREFIX + "*"):
            if entry.is_symlink():
                continue
            if not entry.is_dir():
                continue
            age_mtime = entry.stat().st_mtime
            lock_path = entry / cls.LOCK_NAME
            owner_pid: Optional[int] = None
            if lock_path.exists():
                age_mtime = max(age_mtime, lock_path.stat().st_mtime)
                owner_pid = cls._lock_owner_pid(lock_path)
            if now - age_mtime < older_than_seconds:
                continue
            if owner_pid is not None and cls._pid_is_alive(owner_pid):
                # Another instance is still working in here (a CPU run can
                # easily outlive any age threshold); its data is not ours
                # to delete.
                continue
            shutil.rmtree(entry, ignore_errors=True)
            removed += 1
        return removed


@dataclass
class SegmentationRunResult:
    """Result payload for auto-segmentation runs."""
    success: bool
    method: str  # "totalsegmentator" or "threshold_fallback"
    mask_path: str
    message: str
    geometry_warnings: List[str] = field(default_factory=list)
    # Optional stage-2 spine subregion (pedicle/corpus/lamina) output. Absent
    # unless a subregion model was supplied and its run succeeded.
    subregion_mask_path: Optional[str] = None
    subregion_labels: Dict[str, int] = field(default_factory=dict)
    subregion_message: str = ""


def run_subregion_segmentation(*args, **kwargs) -> str:
    """Delegate to the subregion runner, imported lazily.

    ``src.core.subregion_segmentation`` imports this module, so the real
    function cannot be imported at module scope. Keeping this named wrapper
    here also gives tests a single attribute to patch.
    """
    from src.core.subregion_segmentation import (
        run_subregion_segmentation as _run_subregion,
    )

    return _run_subregion(*args, **kwargs)


def is_totalsegmentator_available() -> bool:
    """Return True when TotalSegmentator package is importable."""
    return importlib.util.find_spec("totalsegmentator") is not None


def preferred_segmentation_device() -> str:
    """Prefer a CUDA GPU and otherwise use the reliable CPU path."""
    try:
        import torch

        if torch.cuda.is_available():
            return "gpu"
    except Exception:
        pass
    return "cpu"


def _emit_progress(
    progress_callback: Optional[Callable[[str], None]],
    message: str
) -> None:
    """Emit progress update if callback exists."""
    if progress_callback is not None:
        progress_callback(message)


def run_totalsegmentator(
    image: sitk.Image,
    work_dir: str,
    task: str = "total",
    device: str = "cpu",
    roi_subset: Optional[List[str]] = None,
    fast: bool = False,
    force_split: bool = False,
    progress_callback: Optional[Callable[[str], None]] = None,
    process_holder: Optional[ProcessHolder] = None,
) -> str:
    """
    Run TotalSegmentator on a SimpleITK volume and return mask path.

    Output is a multilabel NIfTI mask file.

    When ``process_holder`` is supplied the spawned subprocess is published on
    it so another thread can cancel the run; a cancelled run raises
    :class:`SegmentationCancelled`.
    """
    _ensure_torch_shm_executable()

    output_root = Path(work_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    input_path = output_root / "input_volume.nii.gz"
    output_path = output_root / "totalseg_multilabel.nii.gz"

    _emit_progress(progress_callback, "Preparing NIfTI input for TotalSegmentator...")
    sitk.WriteImage(image, str(input_path))

    _emit_progress(
        progress_callback,
        (
            f"Running TotalSegmentator task '{task}'. "
            "The first run may download model weights..."
        ),
    )

    if getattr(sys, "frozen", False):
        # nnU-Net runs in-process here and cannot be interrupted once started,
        # so cancellation is only honoured before the run begins.
        if process_holder is not None and process_holder.cancelled:
            raise SegmentationCancelled("Segmentation cancelled by user")
        _run_frozen_totalsegmentator(
            input_path=input_path,
            output_path=output_path,
            task=task,
            device=device,
            roi_subset=roi_subset,
            fast=fast,
            force_split=force_split,
        )
        if not output_path.exists():
            raise RuntimeError(
                "TotalSegmentator finished but no output mask was created."
            )
        return str(output_path)

    module_spec = importlib.util.find_spec(
        "totalsegmentator.bin.TotalSegmentator"
    )
    if module_spec is None or module_spec.origin is None:
        raise RuntimeError("TotalSegmentator command module was not found.")

    command = [
        sys.executable,
        module_spec.origin,
        "-i",
        str(input_path),
        "-o",
        str(output_path),
        "-ta",
        task,
        "-ml",
        "-d",
        device,
        "-q",
        "-ns",
        "1",
    ]
    if roi_subset:
        command.extend(["-rs", *roi_subset])
    if fast:
        command.append("-f")
    if force_split:
        command.append("-fs")

    if process_holder is not None and process_holder.cancelled:
        raise SegmentationCancelled("Segmentation cancelled by user")

    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if process_holder is not None:
        process_holder.process = process
        # Cancel may have landed between the spawn and the assignment above,
        # in which case terminate() found no process to kill: retry it here.
        if process_holder.cancelled:
            process_holder.terminate()
    stdout, stderr = process.communicate()
    if process_holder is not None and process_holder.cancelled:
        raise SegmentationCancelled("Segmentation cancelled by user")
    if process.returncode != 0:
        detail = (stderr or "").strip() or (stdout or "").strip()
        raise RuntimeError(
            f"TotalSegmentator execution failed: {detail or 'unknown error'}"
        )

    if not output_path.exists():
        raise RuntimeError("TotalSegmentator finished but no output mask was created.")

    return str(output_path)


def _run_frozen_totalsegmentator(
    *,
    input_path: Path,
    output_path: Path,
    task: str,
    device: str,
    roi_subset: Optional[List[str]],
    fast: bool,
    force_split: bool,
) -> None:
    """Run the bundled Python API because frozen apps cannot spawn `-m`."""
    try:
        from totalsegmentator.python_api import totalsegmentator
    except ImportError as exc:
        raise RuntimeError(
            "The packaged TotalSegmentator runtime could not be loaded."
        ) from exc

    totalsegmentator(
        str(input_path),
        str(output_path),
        task=task,
        ml=True,
        device=device,
        roi_subset=roi_subset,
        fast=fast,
        force_split=force_split,
        quiet=True,
        nr_thr_saving=1,
    )


def _ensure_torch_shm_executable() -> None:
    """
    Ensure torch shared-memory manager binary is executable on macOS/Linux.

    Some installations can leave this file without execute permission,
    which breaks multiprocessing inference with "Permission denied".
    """
    try:
        import torch
    except Exception:
        return

    shm_path = Path(torch.__file__).resolve().parent / "bin" / "torch_shm_manager"
    if not shm_path.exists():
        return
    if os.access(shm_path, os.X_OK):
        return

    try:
        mode = shm_path.stat().st_mode
        shm_path.chmod(mode | 0o111)
    except Exception:
        # Best effort only; fallback path still protects runtime.
        return


def run_threshold_fallback(
    image: sitk.Image,
    work_dir: str,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> str:
    """
    Run a simple HU threshold fallback segmentation.

    This is not AI segmentation; it provides a safe non-blocking backup mask.
    """
    output_root = Path(work_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    output_path = output_root / "threshold_fallback_mask.nii.gz"

    _emit_progress(progress_callback, "Running threshold fallback segmentation...")
    # Conservative bone-like threshold for CT HU
    mask = sitk.BinaryThreshold(
        image,
        lowerThreshold=180,
        upperThreshold=6000,
        insideValue=1,
        outsideValue=0,
    )
    mask = sitk.Cast(mask, sitk.sitkUInt8)
    sitk.WriteImage(mask, str(output_path))
    return str(output_path)


def _geometry_close(a: float, b: float, tolerance: float = 1e-4) -> bool:
    return abs(float(a) - float(b)) <= tolerance


def check_segmentation_geometry(
    reference_image: sitk.Image,
    mask_image: sitk.Image,
    tolerance: float = 1e-4,
) -> List[str]:
    """
    Compare geometry metadata between reference image and segmentation mask.

    Returns list of warning messages when mismatch is detected.
    """
    warnings: List[str] = []

    ref_size = tuple(int(v) for v in reference_image.GetSize())
    mask_size = tuple(int(v) for v in mask_image.GetSize())
    if ref_size != mask_size:
        warnings.append(f"size mismatch: ref={ref_size}, mask={mask_size}")

    ref_spacing = tuple(float(v) for v in reference_image.GetSpacing())
    mask_spacing = tuple(float(v) for v in mask_image.GetSpacing())
    for axis, (ref_v, mask_v) in enumerate(zip(ref_spacing, mask_spacing, strict=False)):
        if not _geometry_close(ref_v, mask_v, tolerance=tolerance):
            warnings.append(
                f"spacing mismatch at axis {axis}: ref={ref_v}, mask={mask_v}"
            )

    ref_origin = tuple(float(v) for v in reference_image.GetOrigin())
    mask_origin = tuple(float(v) for v in mask_image.GetOrigin())
    for axis, (ref_v, mask_v) in enumerate(zip(ref_origin, mask_origin, strict=False)):
        if not _geometry_close(ref_v, mask_v, tolerance=tolerance):
            warnings.append(
                f"origin mismatch at axis {axis}: ref={ref_v}, mask={mask_v}"
            )

    ref_direction = tuple(float(v) for v in reference_image.GetDirection())
    mask_direction = tuple(float(v) for v in mask_image.GetDirection())
    if len(ref_direction) != len(mask_direction):
        warnings.append(
            f"direction length mismatch: ref={len(ref_direction)}, "
            f"mask={len(mask_direction)}"
        )
    else:
        for index, (ref_v, mask_v) in enumerate(zip(ref_direction, mask_direction, strict=True)):
            if not _geometry_close(ref_v, mask_v, tolerance=tolerance):
                warnings.append(
                    f"direction mismatch at index {index}: ref={ref_v}, mask={mask_v}"
                )

    return warnings


def validate_segmentation_output(
    reference_image: sitk.Image,
    mask_path: str,
) -> List[str]:
    """
    Load segmentation mask and return geometry warnings relative to input image.
    """
    mask_image = sitk.ReadImage(mask_path)
    return check_segmentation_geometry(reference_image, mask_image)


def _build_result(
    image: sitk.Image,
    method: str,
    mask_path: str,
    message: str,
    subregion_mask_path: Optional[str] = None,
    subregion_labels: Optional[Dict[str, int]] = None,
    subregion_message: str = "",
) -> SegmentationRunResult:
    geometry_warnings = validate_segmentation_output(image, mask_path)
    if geometry_warnings:
        summary = "; ".join(geometry_warnings[:2])
        if len(geometry_warnings) > 2:
            summary += f" (+{len(geometry_warnings) - 2} more)"
        message = f"{message} Geometry check warning: {summary}"

    return SegmentationRunResult(
        success=True,
        method=method,
        mask_path=mask_path,
        message=message,
        geometry_warnings=geometry_warnings,
        subregion_mask_path=subregion_mask_path,
        subregion_labels=dict(subregion_labels or {}),
        subregion_message=subregion_message,
    )


def _run_subregion_stage(
    image: sitk.Image,
    model: "SubregionModel",
    work_dir: str,
    device: str,
    progress_callback: Optional[Callable[[str], None]],
    process_holder: Optional[ProcessHolder],
):
    """Run the optional stage-2 subregion model after a TotalSegmentator run.

    Returns ``(mask_path, labels, message)``. A stage-2 failure is reported
    through ``message`` only: the caller keeps the TotalSegmentator mask and
    never degrades to the threshold fallback because of it. Cancellation is
    the one exception and propagates to abort the whole run.
    """
    try:
        mask_path = run_subregion_segmentation(
            image,
            model,
            work_dir,
            device,
            progress_callback,
            process_holder,
        )
    except SegmentationCancelled:
        raise
    except Exception as exc:
        return None, {}, f"Pedicle model unavailable: {exc}"
    return mask_path, dict(model.labels), ""


def run_segmentation_with_fallback(
    image: sitk.Image,
    work_dir: str,
    task: str = "total",
    device: str = "cpu",
    roi_subset: Optional[List[str]] = None,
    fast: bool = False,
    force_split: bool = False,
    progress_callback: Optional[Callable[[str], None]] = None,
    process_holder: Optional[ProcessHolder] = None,
    subregion_model: Optional["SubregionModel"] = None,
) -> SegmentationRunResult:
    """
    Run TotalSegmentator with fallback strategy.

    Behavior:
    1) If TotalSegmentator is available and succeeds -> use it.
    2) Otherwise fallback to threshold mask.

    When ``subregion_model`` is given, a second-stage subregion inference runs
    after a successful TotalSegmentator run. Its failure never fails the run:
    it only sets ``subregion_message`` on the result.

    A user cancellation propagates as :class:`SegmentationCancelled`; no
    threshold fallback mask is produced in that case.
    """
    if not is_totalsegmentator_available():
        # No subprocess is ever spawned on this path, so Cancel is a no-op:
        # the threshold mask is cheap and completes before the user can react.
        mask_path = run_threshold_fallback(
            image=image,
            work_dir=work_dir,
            progress_callback=progress_callback,
        )
        return _build_result(
            image=image,
            method="threshold_fallback",
            mask_path=mask_path,
            message=(
                "TotalSegmentator is not installed. "
                "Threshold fallback mask was generated instead."
            ),
        )

    attempt_devices = [device]
    if str(device).startswith("gpu"):
        attempt_devices.append("cpu")
    errors = []

    for attempt_index, attempt_device in enumerate(attempt_devices):
        try:
            mask_path = run_totalsegmentator(
                image=image,
                work_dir=work_dir,
                task=task,
                device=attempt_device,
                roi_subset=roi_subset,
                fast=fast,
                force_split=(
                    force_split
                    or (attempt_device == "cpu" and not fast)
                ),
                progress_callback=progress_callback,
                process_holder=process_holder,
            )
            message = "TotalSegmentator segmentation completed."
            if attempt_index > 0:
                message = (
                    "TotalSegmentator segmentation completed after CPU retry."
                )
            subregion_mask_path = None
            subregion_labels: Dict[str, int] = {}
            subregion_message = ""
            if subregion_model is not None:
                (
                    subregion_mask_path,
                    subregion_labels,
                    subregion_message,
                ) = _run_subregion_stage(
                    image=image,
                    model=subregion_model,
                    work_dir=work_dir,
                    device=attempt_device,
                    progress_callback=progress_callback,
                    process_holder=process_holder,
                )
            return _build_result(
                image=image,
                method="totalsegmentator",
                mask_path=mask_path,
                message=message,
                subregion_mask_path=subregion_mask_path,
                subregion_labels=subregion_labels,
                subregion_message=subregion_message,
            )
        except BaseException as exc:
            if isinstance(exc, (KeyboardInterrupt, SegmentationCancelled)):
                raise
            errors.append(f"{attempt_device}: {exc}")
            if attempt_index + 1 < len(attempt_devices):
                _emit_progress(
                    progress_callback,
                    "GPU segmentation failed; retrying on CPU...",
                )

    mask_path = run_threshold_fallback(
        image=image,
        work_dir=work_dir,
        progress_callback=progress_callback,
    )
    return _build_result(
        image=image,
        method="threshold_fallback",
        mask_path=mask_path,
        message=(
            "TotalSegmentator failed; fallback mask was generated. "
            f"Reason: {'; '.join(errors)}"
        ),
    )
