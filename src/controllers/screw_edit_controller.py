"""Shared controller for click and direct-drag screw trajectory editing."""

from __future__ import annotations

import math
from typing import Optional, Tuple

Point3D = Tuple[float, float, float]


class ScrewEditController:
    """Own screw edit mode and apply MPR clicks to selected geometry."""

    MODES = {"entry", "tip", "move"}
    MIN_LENGTH_MM = 1.0

    def __init__(self, volume_manager, main_window):
        self._vm = volume_manager
        self._window = main_window
        self._mode = "idle"
        self._selected_index: Optional[int] = None
        self._drag_anchor: Optional[Point3D] = None
        self._drag_entry: Optional[Point3D] = None
        self._drag_target: Optional[Point3D] = None
        self._active_source: Optional[str] = None

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def is_active(self) -> bool:
        return self._mode != "idle"

    @property
    def active_source(self) -> Optional[str]:
        return self._active_source

    def start(self, mode: str) -> bool:
        """Start a one-click edit mode for the selected screw."""
        if mode not in self.MODES:
            raise ValueError(f"Unknown screw edit mode: {mode}")
        if self._vm.get_vtk_image() is None:
            self._window.statusbar.showMessage(
                "Load a DICOM volume before editing a screw"
            )
            self.refresh_controls()
            return False

        row = self._window.screw_list_widget.currentRow()
        if self._screw_at(row) is None:
            self._window.statusbar.showMessage("Select a screw to edit")
            self.refresh_controls()
            return False

        self._window._tool_ctrl.set_tool("navigate")
        self._selected_index = row
        self._mode = mode
        self.refresh_controls()
        labels = {
            "entry": "entry point",
            "tip": "tip point",
            "move": "new screw center",
        }
        self._window.statusbar.showMessage(
            f"Screw edit: click the {labels[mode]} in any MPR view"
        )
        return True

    def cancel(self, show_message: bool = True) -> None:
        """Cancel the pending edit without changing screw geometry."""
        was_active = self.is_active
        self._mode = "idle"
        self._selected_index = None
        self._drag_anchor = None
        self._drag_entry = None
        self._drag_target = None
        self._active_source = None
        self.refresh_controls()
        if was_active and show_message:
            self._window.statusbar.showMessage("Screw edit cancelled")

    def reset(self) -> None:
        """Reset edit state during volume or planning workspace changes."""
        self.cancel(show_message=False)

    def begin_drag(self, index: int, part: str, world_point: Point3D) -> bool:
        """Begin direct manipulation of an entry handle, tip, or shaft."""
        mode = "move" if part == "shaft" else part
        if mode not in self.MODES:
            raise ValueError(f"Unknown screw drag part: {part}")
        if self._vm.get_vtk_image() is None:
            return False
        screw = self._screw_at(index)
        if screw is None:
            return False

        self._window._tool_ctrl.set_tool("navigate")
        self._window.screw_list_widget.setCurrentRow(index)
        self._selected_index = int(index)
        self._mode = mode
        self._drag_anchor = self._point(world_point)
        self._drag_entry = self._point(screw.entry_point)
        self._drag_target = self._point(screw.target_point)
        self.refresh_controls()
        labels = {"entry": "insertion head", "tip": "tip", "move": "shaft"}
        self._window.statusbar.showMessage(
            f"Dragging screw {labels[mode]}"
        )
        return True

    def update_drag(self, world_point: Point3D, source: str = "view") -> bool:
        """Apply one continuous drag sample and refresh every linked view."""
        if (
            not self.is_active
            or self._selected_index is None
            or self._drag_anchor is None
            or self._drag_entry is None
            or self._drag_target is None
        ):
            return False

        point = self._point(world_point)
        self._active_source = str(source)
        if self._mode == "entry":
            entry_point, target_point = point, self._drag_target
        elif self._mode == "tip":
            entry_point, target_point = self._drag_entry, point
        else:
            delta = tuple(
                point[axis] - self._drag_anchor[axis] for axis in range(3)
            )
            entry_point = tuple(
                self._drag_entry[axis] + delta[axis] for axis in range(3)
            )
            target_point = tuple(
                self._drag_target[axis] + delta[axis] for axis in range(3)
            )

        if not self._apply_points(entry_point, target_point):
            return False
        updated = self._screw_at(self._selected_index)
        if str(source).strip().lower() == "3d" and updated is not None:
            self._focus_mpr_on_3d_edit(updated)
        self._window.statusbar.showMessage(
            f"Screw updated from {source}; double-click again to finish"
        )
        return True

    def _focus_mpr_on_3d_edit(self, screw) -> None:
        """Move standard MPR crosshairs to the part being edited in 3D."""
        if self._mode == "entry":
            point = screw.entry_point
        elif self._mode == "tip":
            point = screw.target_point
        else:
            point = tuple(
                (screw.entry_point[index] + screw.target_point[index]) / 2.0
                for index in range(3)
            )
        self._vm.set_crosshair_position(
            *point,
            source_plane="3d_screw_edit",
        )

    def end_drag(self) -> None:
        """Finish direct manipulation while preserving the last valid state."""
        selected_index = self._selected_index
        was_active = self.is_active
        self.cancel(show_message=False)
        if was_active and selected_index is not None:
            self._realign_screw_mpr(selected_index)

    def handle_click(
        self,
        plane: str,
        x: float,
        y: float,
        z: float,
    ) -> bool:
        """Consume one MPR click when editing and update the selected screw."""
        if not self.is_active or self._selected_index is None:
            return False

        if self._window.screw_list_widget.currentRow() != self._selected_index:
            self.cancel(show_message=False)
            return False

        screw = self._screw_at(self._selected_index)
        if screw is None:
            self.cancel(show_message=False)
            return False

        point = (float(x), float(y), float(z))
        entry_point, target_point = self._edited_points(screw, point)
        mode = self._mode
        if not self._apply_points(entry_point, target_point):
            return True
        selected_index = self._selected_index
        self.cancel(show_message=False)
        if selected_index is not None:
            self._realign_screw_mpr(selected_index)
        labels = {"entry": "Entry", "tip": "Tip", "move": "Whole screw"}
        self._window.statusbar.showMessage(
            f"{labels[mode]} updated from {plane.capitalize()} MPR"
        )
        return True

    def _realign_screw_mpr(self, index: int) -> None:
        """Apply deferred Screw MPR axes after an MPR-origin edit finishes."""
        controller = getattr(self._window, "_screw_mpr_ctrl", None)
        if controller is not None:
            controller.on_screw_updated(int(index))

    def on_screw_selection_changed(self, row: int) -> None:
        """Cancel a pending edit when the selected screw changes."""
        if self.is_active and row != self._selected_index:
            self._cancel_active_viewer_locks()
            self.cancel(show_message=False)
        self.refresh_controls()

    def _cancel_active_viewer_locks(self) -> None:
        """Release stale pointer locks without clearing a new pending click."""
        get_mpr_viewers = getattr(self._window, "_get_mpr_viewers", None)
        viewers = list(get_mpr_viewers()) if callable(get_mpr_viewers) else []
        viewer_3d = getattr(self._window, "viewer_3d", None)
        if viewer_3d is not None:
            viewers.append(viewer_3d)
        for viewer in viewers:
            if bool(getattr(viewer, "is_screw_interaction_active", False)):
                viewer.cancel_screw_interaction()

    def on_screw_removed(self, row: int) -> None:
        """Cancel or reindex edit state before a screw is removed."""
        if self._selected_index is None:
            return
        if row == self._selected_index:
            self.cancel(show_message=False)
        elif row < self._selected_index:
            self._selected_index -= 1

    def refresh_controls(self) -> None:
        """Synchronize edit button state with volume, selection, and mode."""
        row = self._window.screw_list_widget.currentRow()
        can_edit = (
            self._vm.get_vtk_image() is not None
            and self._screw_at(row) is not None
        )
        buttons = {
            "entry": getattr(self._window, "screw_edit_entry_btn", None),
            "tip": getattr(self._window, "screw_edit_tip_btn", None),
            "move": getattr(self._window, "screw_edit_move_btn", None),
        }
        for mode, button in buttons.items():
            if button is None:
                continue
            button.setEnabled(can_edit)
            button.setChecked(self._mode == mode)
        cancel_button = getattr(self._window, "screw_edit_cancel_btn", None)
        if cancel_button is not None:
            cancel_button.setEnabled(self.is_active)

    def _screw_at(self, index: int):
        screws = self._window._tool_ctrl.screw_tool.get_screws()
        if index < 0 or index >= len(screws):
            return None
        return screws[index]

    def _edited_points(self, screw, point: Point3D) -> tuple[Point3D, Point3D]:
        entry = tuple(float(value) for value in screw.entry_point)
        target = tuple(float(value) for value in screw.target_point)
        if self._mode == "entry":
            return point, target
        if self._mode == "tip":
            return entry, point

        half_vector = tuple(
            (target[index] - entry[index]) / 2.0 for index in range(3)
        )
        moved_entry = tuple(
            point[index] - half_vector[index] for index in range(3)
        )
        moved_target = tuple(
            point[index] + half_vector[index] for index in range(3)
        )
        return moved_entry, moved_target

    def _apply_points(self, entry_point: Point3D, target_point: Point3D) -> bool:
        """Validate, store, and synchronize one proposed screw geometry."""
        if self._selected_index is None:
            return False
        if not self._points_inside_volume(entry_point, target_point):
            self._window.statusbar.showMessage(
                "Screw entry and tip must remain inside the CT volume"
            )
            return False
        if self._distance(entry_point, target_point) < self.MIN_LENGTH_MM:
            self._window.statusbar.showMessage(
                "Screw length must be at least 1.0 mm"
            )
            return False

        updated = self._window._tool_ctrl.screw_tool.replace_screw(
            self._selected_index,
            entry_point=entry_point,
            target_point=target_point,
        )
        self._window._tool_ctrl.refresh_screw(self._selected_index, updated)
        return True

    @staticmethod
    def _point(values) -> Point3D:
        return tuple(float(value) for value in values)

    def _points_inside_volume(self, *points: Point3D) -> bool:
        bounds = self._vm.bounds
        if len(bounds) != 6:
            return False
        return all(
            bounds[0] <= point[0] <= bounds[1]
            and bounds[2] <= point[1] <= bounds[3]
            and bounds[4] <= point[2] <= bounds[5]
            for point in points
        )

    @staticmethod
    def _distance(entry: Point3D, target: Point3D) -> float:
        return math.sqrt(
            sum((target[index] - entry[index]) ** 2 for index in range(3))
        )
