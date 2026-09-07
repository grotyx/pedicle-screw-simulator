"""
Discovery and label parsing for the optional spine subregion nnU-Net model.

This module is Qt-free. It locates a locally-trained nnU-Net "subregion"
model (one that segments a vertebra into corpus / pedicle / lamina /
spinous / transverse / articular subregions) and parses its
``dataset.json`` to resolve anatomical roles to the label ids the model
actually uses. Label ids are never hard-coded: they are read from the
model's own ``dataset.json`` at runtime via ``SUBREGION_ROLE_PATTERNS``.

Running inference with the discovered model is out of scope here; see the
segmentation runner for that.
"""

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

# Role -> case-insensitive regex tried against dataset.json label names.
# Order matters only in that "pedicle" must be present in the result; the
# patterns themselves are independent of each other.
SUBREGION_ROLE_PATTERNS = {
    "corpus": r"corpus|body|vertebral_body",
    "pedicle": r"pedicle",
    "lamina": r"lamina",
    "spinous": r"spinous",
    "transverse": r"transverse",
    "articular": r"articular|facet",
}


@dataclass
class SubregionModel:
    root: Path  # directory that contains dataset.json and fold checkpoints (nnUNet results layout)
    dataset_id: str  # e.g. "Dataset501_SpineSubregions" (parent folder name)
    configuration: str  # "3d_fullres" by default
    labels: Dict[str, int]  # role -> label id resolved through SUBREGION_ROLE_PATTERNS


def parse_dataset_labels(dataset_json: Dict[str, Any]) -> Dict[str, int]:
    """
    Resolve anatomical roles to label ids from an nnU-Net dataset.json.

    ``dataset_json["labels"]`` maps name -> id, or name -> [ids] for region
    labels (the first id is used in that case). Every role in
    ``SUBREGION_ROLE_PATTERNS`` whose pattern matches some label name is
    included in the result. Raises ValueError if there is no usable
    "labels" mapping, or if the required "pedicle" role cannot be resolved.
    """
    raw = dataset_json.get("labels")
    if not isinstance(raw, dict):
        raise ValueError("dataset.json has no 'labels' mapping")

    roles: Dict[str, int] = {}
    for role, pattern in SUBREGION_ROLE_PATTERNS.items():
        regex = re.compile(pattern, re.IGNORECASE)
        for name, value in raw.items():
            if regex.search(str(name)):
                roles[role] = int(value[0] if isinstance(value, (list, tuple)) else value)
                break

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
            dataset_json = json.load(handle)
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
