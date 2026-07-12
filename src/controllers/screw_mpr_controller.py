"""Controller for explicit screw-aligned multiplanar reconstruction."""

from __future__ import annotations

import logging
from typing import Optional

from src.core.mpr_geometry import build_screw_mpr_axes

logger = logging.getLogger(__name__)


class ScrewMPRController:
    """Apply screw-aligned axes to the three existing MPR viewers."""

    def __init__(self, volume_manager, main_window):
        self._vm = volume_manager
        self._window = main_window
        self._active = False
        self._selected_index: Optional[int] = None
        self._position_percent = 50

    @property
    def is_active(self) -> bool:
        return self._active

    @property
    def position_percent(self) -> int:
        return self._position_percent

    def _screws(self):
        return self._window._tool_ctrl.screw_tool.get_screws()

    def _screw_at(self, row: int):
        screws = self._screws()
        if row < 0 or row >= len(screws):
            return None
        return screws[row]

    def _has_loaded_volume(self) -> bool:
        return self._vm.get_vtk_image() is not None

    def enter(self) -> bool:
        """Enter screw-axis MPR for the explicitly selected screw."""
        if not self._has_loaded_volume():
            self._window.statusbar.showMessage(
                "Load a DICOM volume before using Screw MPR"
            )
            self.refresh_controls()
            return False

        row = self._window.screw_list_widget.currentRow()
        screw = self._screw_at(row)
        if screw is None:
            self._window.statusbar.showMessage(
                "Select a screw before using Screw MPR"
            )
            self.refresh_controls()
            return False

        self._selected_index = row
        self._position_percent = 50
        try:
            self._apply_screw(screw)
        except ValueError as exc:
            logger.warning("Cannot enter Screw-axis MPR: %s", exc)
            self._selected_index = None
            self._window.statusbar.showMessage(
                "Cannot align MPR to a zero-length screw"
            )
            self.refresh_controls()
            return False

        self._active = True
        self._set_review_filter(row)
        self._window.set_view_layout("planning")
        self._set_slider_value(50)
        self.refresh_selected_screw()
        self.refresh_controls()
        self._window.statusbar.showMessage(
            f"Screw MPR active for screw #{row + 1}"
        )
        return True

    def exit(self) -> None:
        """Restore standard axial, sagittal, and coronal reslice planes."""
        was_active = self._active
        self._active = False
        self._selected_index = None
        self._position_percent = 50
        if was_active:
            for viewer in self._window._get_mpr_viewers():
                viewer.clear_custom_reslice_axes()
            self._set_review_filter(None)
        self._set_slider_value(50)
        self._window.screw_axis_position_label.setText("Position: 50%")
        self.refresh_selected_screw()
        self.refresh_controls()
        if was_active:
            self._window.statusbar.showMessage("Std MPR restored")

    def set_position(self, percent: int) -> None:
        """Move the perpendicular cross-section from entry to target."""
        self._position_percent = max(0, min(100, int(percent)))
        if not self._active or self._selected_index is None:
            self._window.screw_axis_position_label.setText(
                f"Position: {self._position_percent}%"
            )
            return

        screw = self._screw_at(self._selected_index)
        if screw is None:
            self.exit()
            return
        try:
            self._apply_screw(screw)
        except ValueError:
            self.exit()

    def on_screw_selection_changed(self, row: int) -> None:
        """Refresh controls and active axes after screw list selection changes."""
        screw = self._screw_at(row)
        if self._active:
            if screw is None:
                self.exit()
                return
            self._selected_index = row
            self._set_review_filter(row)
            try:
                self._apply_screw(screw)
            except ValueError:
                self.exit()
                return
        self.refresh_selected_screw()
        self.refresh_controls()

    def select_previous(self) -> None:
        """Select the previous screw without wrapping at the first item."""
        current = self._window.screw_list_widget.currentRow()
        if current > 0:
            self._window.screw_list_widget.setCurrentRow(current - 1)

    def select_next(self) -> None:
        """Select the next screw without wrapping at the last item."""
        current = self._window.screw_list_widget.currentRow()
        count = len(self._screws())
        if current < count - 1:
            self._window.screw_list_widget.setCurrentRow(max(current + 1, 0))

    def refresh_selected_screw(self) -> None:
        """Publish the current screw and review state to the inspector."""
        row = self._window.screw_list_widget.currentRow()
        updater = getattr(
            self._window,
            "update_selected_screw_inspector",
            None,
        )
        if callable(updater):
            updater(row, self._screw_at(row), self._active)

    def on_screw_removed(self, row: int) -> None:
        """Restore standard MPR if the active screw is being removed."""
        if not self._active or self._selected_index is None:
            return
        if row == self._selected_index:
            self.exit()
        elif row < self._selected_index:
            self._selected_index -= 1
            self._set_review_filter(self._selected_index)

    def on_screw_updated(self, row: int) -> None:
        """Refresh inspector and active MPR axes after screw geometry changes."""
        screw = self._screw_at(row)
        if screw is None:
            return
        edit_controller = getattr(self._window, "_screw_edit_ctrl", None)
        edit_in_progress = bool(
            edit_controller is not None and edit_controller.is_active
        )
        edit_from_3d = bool(
            edit_controller is not None
            and getattr(edit_controller, "active_source", None) == "3D"
        )
        if (
            self._active
            and self._selected_index == row
            and (not edit_in_progress or edit_from_3d)
        ):
            try:
                self._apply_screw(screw)
            except ValueError:
                self.exit()
                return
        self.refresh_selected_screw()
        self.refresh_controls()

    def refresh_controls(self) -> None:
        """Update enabled state from volume, selection, and mode."""
        row = self._window.screw_list_widget.currentRow()
        has_selection = self._screw_at(row) is not None
        can_enter = self._has_loaded_volume() and has_selection and not self._active
        self._window.screw_axis_mpr_btn.setEnabled(can_enter)
        self._window.standard_mpr_btn.setEnabled(self._active)
        self._window.screw_axis_position_slider.setEnabled(self._active)
        previous_button = getattr(self._window, "screw_previous_btn", None)
        next_button = getattr(self._window, "screw_next_btn", None)
        if previous_button is not None:
            previous_button.setEnabled(has_selection and row > 0)
        if next_button is not None:
            next_button.setEnabled(
                has_selection and row < len(self._screws()) - 1
            )

    def _apply_screw(self, screw) -> None:
        axes = build_screw_mpr_axes(
            screw.entry_point,
            screw.target_point,
            self._position_percent / 100.0,
        )
        self._window.axial_viewer.set_custom_reslice_axes(
            axes.oblique_axial,
            f"Oblique Axial · {self._selected_identity(screw)}",
        )
        self._window.sagittal_viewer.set_custom_reslice_axes(
            axes.oblique_sagittal,
            f"Oblique Sagittal · {self._selected_identity(screw)}",
        )
        self._window.coronal_viewer.set_custom_reslice_axes(
            axes.cross_section,
            f"Cross-section · {self._position_percent}%",
        )
        self._window.screw_axis_position_label.setText(
            f"Position: {self._position_percent}% "
            f"({axes.distance_from_entry_mm:.1f} mm from entry)"
        )

    def _selected_identity(self, screw) -> str:
        level = str(getattr(screw, "vertebra_level", "") or "").strip()
        side = str(getattr(screw, "side", "") or "").strip().capitalize()
        if level or side:
            return " ".join(part for part in (level, side) if part)
        index = 0 if self._selected_index is None else self._selected_index
        return f"Screw #{index + 1}"

    def _set_slider_value(self, value: int) -> None:
        slider = self._window.screw_axis_position_slider
        previous = slider.blockSignals(True)
        slider.setValue(value)
        slider.blockSignals(previous)

    def _set_review_filter(self, screw_id: Optional[int]) -> None:
        """Publish the selected screw ID to every MPR overlay renderer."""
        for viewer in self._window._get_mpr_viewers():
            setter = getattr(viewer, "set_review_screw", None)
            if callable(setter):
                setter(screw_id)
