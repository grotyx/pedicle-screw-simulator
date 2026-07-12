"""
Helpers for TotalSegmentator label-id to anatomy-name mapping.
"""

from collections.abc import Mapping
from typing import Dict


def _normalize_label_map(raw_map) -> Dict[int, str]:
    """
    Normalize raw label map into {int_label_id: anatomy_name}.

    Accepts maps where keys may be int or numeric strings.
    """
    if not isinstance(raw_map, Mapping):
        return {}

    normalized: Dict[int, str] = {}
    for raw_key, raw_value in raw_map.items():
        try:
            label_id = int(raw_key)
        except (TypeError, ValueError):
            continue

        if label_id <= 0:
            continue

        label_name = str(raw_value).strip() if raw_value is not None else ""
        if not label_name:
            label_name = f"label_{label_id}"

        normalized[label_id] = label_name

    return normalized


def get_totalseg_task_label_map(task: str) -> Dict[int, str]:
    """
    Return TotalSegmentator class map for the given task.

    Returns empty map when TotalSegmentator is unavailable.
    """
    try:
        from totalsegmentator.map_to_binary import class_map
    except Exception:
        return {}

    task_map = class_map.get(task, {})
    return _normalize_label_map(task_map)


def get_segmentation_label_map(task: str, method: str) -> Dict[int, str]:
    """
    Return label map for current segmentation method/task.
    """
    if method == "threshold_fallback":
        return {1: "threshold_bone"}
    return get_totalseg_task_label_map(task)


def format_label_display(label_id: int, label_name: str) -> str:
    """Format one label for a UI dropdown entry."""
    return f"{label_id}: {label_name.replace('_', ' ')}"


def format_selected_label_text(label_value: int, label_map: Mapping[int, str]) -> str:
    """Format currently selected label state text for UI."""
    if label_value <= 0:
        return "Selected: All labels"

    label_name = label_map.get(label_value)
    if label_name is None:
        return f"Selected: Unknown label ID {label_value}"

    return f"Selected: {label_name.replace('_', ' ')} (ID {label_value})"
