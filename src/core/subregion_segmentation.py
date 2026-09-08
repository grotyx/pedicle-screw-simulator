"""
Discovery and label parsing for the optional spine subregion nnU-Net model.

This module is Qt-free. It locates a locally-trained nnU-Net "subregion"
model (one that segments a vertebra into corpus / pedicle / lamina /
spinous / transverse / articular subregions) and parses its
``dataset.json`` to resolve anatomical roles to the label ids the model
actually uses. Label ids are never hard-coded: they are read from the
model's own ``dataset.json`` at runtime via ``SUBREGION_ROLE_PATTERNS``.

This module also runs inference with the discovered model
(``run_subregion_segmentation``) and resamples its output mask back onto the
original CT grid (``resample_to_reference``).
"""

import json
import os
import re
import shutil
import subprocess
import sys
import sysconfig
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import SimpleITK as sitk

from src.core.totalseg_integration import (
    ProcessHolder,
    SegmentationCancelled,
    _emit_progress,
)

# Role -> case-insensitive regex tried against dataset.json label names.
SUBREGION_ROLE_PATTERNS = {
    "corpus": r"corpus|body|vertebral_body",
    "pedicle": r"pedicle",
    "lamina": r"lamina",
    "spinous": r"spinous",
    "transverse": r"transverse",
    "articular": r"articular|facet",
}

# Assignment order for SUBREGION_ROLE_PATTERNS: a label name claimed by one
# role is no longer eligible for another, so this order decides who wins a
# collision (e.g. "vertebral_body_pedicle" matching both "pedicle" and
# "corpus"/"body"). "pedicle" goes first because it is the one role callers
# depend on always resolving correctly.
_ROLE_ASSIGNMENT_ORDER = ["pedicle", "lamina", "spinous", "transverse", "articular", "corpus"]


@dataclass
class SubregionModel:
    root: Path  # directory that contains dataset.json and fold checkpoints (nnUNet results layout)
    dataset_id: str  # e.g. "Dataset501_SpineSubregions" (parent folder name)
    configuration: str  # "3d_fullres" by default
    labels: Dict[str, int]  # role -> label id resolved through SUBREGION_ROLE_PATTERNS


def _resolve_label_value(value: Any) -> int:
    return int(value[0] if isinstance(value, (list, tuple)) else value)


def parse_dataset_labels(dataset_json: Dict[str, Any]) -> Dict[str, int]:
    """
    Resolve anatomical roles to label ids from an nnU-Net dataset.json.

    ``dataset_json["labels"]`` maps name -> id, or name -> [ids] for region
    labels (the first id is used in that case). Roles are assigned in
    ``_ROLE_ASSIGNMENT_ORDER``, and a label name claimed by one role is
    removed from consideration for the rest, so a name like
    "vertebral_body_pedicle" cannot satisfy both "pedicle" and "corpus".
    For each role, a whole-token match (the pattern bounded by "_", "-",
    whitespace, or the string's edges) is preferred over a bare substring
    match, so "Pedicle" is chosen over "PedicleScrewChannel" when both are
    present. Raises ValueError if there is no usable "labels" mapping, or
    if the required "pedicle" role cannot be resolved.
    """
    raw = dataset_json.get("labels")
    if not isinstance(raw, dict):
        raise ValueError("dataset.json has no 'labels' mapping")

    claimed = set()
    roles: Dict[str, int] = {}
    for role in _ROLE_ASSIGNMENT_ORDER:
        pattern = SUBREGION_ROLE_PATTERNS[role]
        whole_token = re.compile(rf"(?:^|[_\s-])(?:{pattern})(?:[_\s-]|$)", re.IGNORECASE)
        substring = re.compile(pattern, re.IGNORECASE)

        available = [(name, value) for name, value in raw.items() if name not in claimed]
        match = next((item for item in available if whole_token.search(str(item[0]))), None)
        if match is None:
            match = next((item for item in available if substring.search(str(item[0]))), None)
        if match is not None:
            name, value = match
            roles[role] = _resolve_label_value(value)
            claimed.add(name)

    if "pedicle" not in roles:
        raise ValueError("dataset.json defines no pedicle label")

    return roles


def find_subregion_model(explicit_dir: Optional[str] = None) -> Optional[SubregionModel]:
    """
    Locate a spine subregion nnU-Net model directory.

    Checked in order: ``explicit_dir``, then the ``PSS_SUBREGION_MODEL_DIR``
    environment variable. A candidate is accepted only if it is a directory
    containing a ``dataset.json`` with a "labels" mapping that resolves at
    least a "pedicle" role. Returns None if no candidate qualifies.
    """
    candidates = [explicit_dir, os.environ.get("PSS_SUBREGION_MODEL_DIR")]
    for candidate in candidates:
        if not candidate:
            continue
        root = Path(candidate)
        dataset_json_path = root / "dataset.json"
        if not dataset_json_path.is_file():
            continue

        with dataset_json_path.open("r", encoding="utf-8") as handle:
            try:
                dataset_json = json.load(handle)
            except json.JSONDecodeError as exc:
                raise ValueError(f"dataset.json is not valid JSON: {dataset_json_path}") from exc
        labels = parse_dataset_labels(dataset_json)

        configuration = "3d_fullres"
        # nnUNet results folders are named "<Trainer>__<Plans>__<configuration>";
        # split on the LAST "__" so a trainer/plans name containing its own
        # double underscore does not swallow the configuration.
        if "__" in root.name:
            configuration = root.name.rsplit("__", 1)[1]

        return SubregionModel(
            root=root,
            dataset_id=root.parent.name,
            configuration=configuration,
            labels=labels,
        )
    return None


def _find_nnunet_predict_executable() -> Optional[str]:
    """
    Locate the ``nnUNetv2_predict`` console script for the running interpreter.

    Checked in order:

    1. Next to ``sys.executable`` (``Scripts/`` on Windows, ``bin/``
       elsewhere) — where pip installs a console script for the current
       interpreter in the common case.
    2. ``sysconfig.get_path("scripts")`` — covers interpreter layouts where
       the scripts directory isn't literally ``sys.executable``'s parent
       (e.g. a user-site install, or a venv reached through a symlinked
       interpreter).
    3. ``shutil.which`` against the process ``PATH``, as a last resort only.

    The first two are tried first and preferred because a bare PATH lookup
    can resolve to an unrelated Python installation's console script (a
    different venv, or a system Python) and silently run against the wrong
    environment; PATH is only consulted once both interpreter-adjacent
    locations have come up empty, which beats falling back to the
    argv-ignoring ``-m`` module invocation.
    """
    names = ["nnUNetv2_predict.exe", "nnUNetv2_predict"] if os.name == "nt" else ["nnUNetv2_predict"]

    search_dirs = [Path(sys.executable).parent]
    scripts_dir = sysconfig.get_path("scripts")
    if scripts_dir:
        scripts_path = Path(scripts_dir)
        if scripts_path not in search_dirs:
            search_dirs.append(scripts_path)

    for directory in search_dirs:
        for name in names:
            candidate = directory / name
            if candidate.is_file():
                return str(candidate)

    return shutil.which("nnUNetv2_predict")


def _resolve_predict_folds(model_root: Path) -> List[str]:
    """
    Determine the ``-f`` fold argument(s) for ``nnUNetv2_predict`` from the
    ``fold_*`` directories actually present under ``model_root``, instead of
    hard-coding ``all`` regardless of how the model was trained.

    A ``fold_all`` directory (the model was trained once on the whole
    dataset) maps to ``["all"]``. Otherwise every ``fold_<N>`` directory
    present contributes its number, sorted ascending, so e.g. ``fold_0``,
    ``fold_1``, ``fold_3`` on disk yields ``["0", "1", "3"]``. Falls back to
    ``["all"]`` when ``model_root`` has no recognizable fold directory,
    matching the previous unconditional default.
    """
    if not model_root.is_dir():
        return ["all"]

    fold_dirs = [p.name for p in model_root.iterdir() if p.is_dir() and p.name.startswith("fold_")]
    if any(name == "fold_all" for name in fold_dirs):
        return ["all"]

    numbers = sorted(
        (name[len("fold_") :] for name in fold_dirs if name[len("fold_") :].isdigit()),
        key=int,
    )
    return numbers if numbers else ["all"]


def build_predict_command(
    model: SubregionModel, input_dir: Path, output_dir: Path, device: str
) -> List[str]:
    """
    Build the ``nnUNetv2_predict`` command line for ``model``.

    Prefers the console-script entry point installed alongside the running
    interpreter, falling back to invoking ``predict_entry_point()`` directly
    via ``-c`` with the current interpreter (NOT ``-m
    nnunetv2.inference.predict_from_raw_data``: that module's own
    ``__main__`` guard runs an argv-ignoring demo, not the CLI). That
    fallback only works with a real Python interpreter and an nnunetv2
    install on its module path; it cannot run in a frozen/packaged build,
    where ``sys.executable`` is the frozen app itself rather than a Python
    interpreter (see the ``sys.frozen`` guard in
    ``run_subregion_segmentation``, which refuses to run there at all).
    ``-f`` is derived from the ``fold_*`` directories under ``model.root``
    (see ``_resolve_predict_folds``). ``device`` values starting with "gpu"
    map to nnU-Net's "cuda" device; anything else runs on "cpu".
    """
    executable = _find_nnunet_predict_executable()
    base = (
        [executable]
        if executable
        else [
            sys.executable,
            "-c",
            "from nnunetv2.inference.predict_from_raw_data import predict_entry_point; predict_entry_point()",
        ]
    )
    return base + [
        "-i", str(input_dir),
        "-o", str(output_dir),
        "-d", model.dataset_id,
        "-c", model.configuration,
        "-f", *_resolve_predict_folds(model.root),
        "-device", "cuda" if str(device).startswith("gpu") else "cpu",
        "--disable_tta",
    ]


def run_subregion_segmentation(
    image: sitk.Image,
    model: SubregionModel,
    work_dir: str,
    device: str,
    progress_callback=None,
    process_holder: Optional[ProcessHolder] = None,
) -> str:
    """
    Run the spine subregion nnU-Net on ``image`` and return the output mask path.

    Writes the input volume to ``<work_dir>/subregion_in/spine_0000.nii.gz``
    (the ``_0000`` channel suffix nnU-Net requires), points
    ``nnUNet_results`` at the model's results folder, and runs
    ``nnUNetv2_predict`` as a subprocess. When ``process_holder`` is supplied
    the spawned process is published on it so another thread can cancel the
    run; a cancelled run raises :class:`SegmentationCancelled`.

    Raises ``RuntimeError`` immediately in a frozen build: unlike
    TotalSegmentator (which runs nnU-Net in-process there), this path always
    shells out to ``nnUNetv2_predict``, and a frozen app has no Python
    interpreter to fall back to for the "-m nnunetv2..." invocation.
    """
    if getattr(sys, "frozen", False):
        raise RuntimeError(
            "The spine subregion model requires a non-frozen Python environment with nnunetv2 installed"
        )

    input_dir = Path(work_dir) / "subregion_in"
    output_dir = Path(work_dir) / "subregion_out"
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(image, str(input_dir / "spine_0000.nii.gz"))

    env = dict(os.environ)
    env["nnUNet_results"] = str(model.root.parent.parent)
    env.setdefault("nnUNet_raw", str(Path(work_dir) / "nnunet_raw"))
    env.setdefault("nnUNet_preprocessed", str(Path(work_dir) / "nnunet_pre"))

    _emit_progress(progress_callback, "Running spine subregion model (pedicle/corpus/lamina)...")

    if process_holder is not None and process_holder.cancelled:
        raise SegmentationCancelled("Subregion segmentation cancelled by user")

    process = subprocess.Popen(
        build_predict_command(model, input_dir, output_dir, device),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )
    if process_holder is not None:
        process_holder.process = process
        # Cancel may have landed between the spawn and the assignment above,
        # in which case terminate() found no process to kill: retry it here.
        if process_holder.cancelled:
            process_holder.terminate()
    stdout, stderr = process.communicate()
    if process_holder is not None and process_holder.cancelled:
        raise SegmentationCancelled("Subregion segmentation cancelled by user")
    if process.returncode != 0:
        raise RuntimeError(f"Subregion model failed: {(stderr or stdout or '').strip()[-2000:]}")

    output = output_dir / "spine.nii.gz"
    if not output.exists():
        raise RuntimeError("Subregion model produced no output")
    return str(output)


def resample_to_reference(mask: sitk.Image, reference: sitk.Image) -> sitk.Image:
    """
    Resample ``mask`` onto ``reference``'s grid using nearest-neighbour interpolation.

    Returns ``mask`` unchanged when its geometry (size, spacing, origin,
    direction) already matches ``reference``, since resampling a label mask
    that is already on the target grid can only introduce rounding noise.
    """
    same = (
        mask.GetSize() == reference.GetSize()
        and np.allclose(mask.GetSpacing(), reference.GetSpacing())
        and np.allclose(mask.GetOrigin(), reference.GetOrigin())
        and np.allclose(mask.GetDirection(), reference.GetDirection())
    )
    if same:
        return mask
    return sitk.Resample(mask, reference, sitk.Transform(), sitk.sitkNearestNeighbor, 0, mask.GetPixelID())
