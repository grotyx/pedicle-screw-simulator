"""Controller for explicit screw-aligned multiplanar reconstruction."""

from __future__ import annotations

import logging
import math
from typing import Optional, Tuple

from src.core.mpr_geometry import build_screw_mpr_axes
from src.core.pedicle_analyzer import VERTEBRA_LABELS

logger = logging.getLogger(__name__)

# vertebra_level name ("L4") -> TotalSegmentator label id (28): the fallback
# used when the screw's grader has not (yet) resolved a label from the mask.
_LABEL_BY_LEVEL_NAME = {name: label for label, name in VERTEBRA_LABELS.items()}

ROTATION_STEP_DEG = 5.0
PLANE_STEP_MM = 1.0
MIN_ROTATION_DEG = -180.0
MAX_ROTATION_DEG = 180.0

# Viewer plane name -> logical Screw MPR plane.
_PLANE_BY_VIEWER = {
    "axial": "oblique_axial",
    "sagittal": "oblique_sagittal",
    "coronal": "cross_section",
}

SCREW_MPR_CONTROLS_HELP = (
    "Screw MPR controls\n"
    "\n"
    "Wheel on the cross-section — move the cut 1 mm along the screw axis\n"
    "Wheel on an oblique view — shift that cut 1 mm along its own normal\n"
    "Shift + wheel on any view — rotate 5° about the screw axis\n"
    "Ctrl + left-drag on the cross-section — rotate continuously\n"
    "Middle-button drag — pan any view (the Pan toggle still works too)\n"
    "Ctrl + wheel — zoom; Fit — fit the image to the view\n"
    "\n"
    "Positive rotation follows the right-hand rule about the screw axis, "
    "which points from the entry point to the target point."
)


class ScrewMPRController:
    """Apply screw-aligned axes to the three existing MPR viewers."""

    def __init__(self, volume_manager, main_window):
        self._vm = volume_manager
        self._window = main_window
        self._active = False
        self._selected_index: Optional[int] = None
        self._position_percent = 50.0
        self._rotation_deg = 0.0
        self._long_offsets = [0.0, 0.0]
        self._cross_offset = [0.0, 0.0]
        # (row, entry, target) -> resolved vertebra label, so wheel steps
        # (Position/rotation/offset only, entry/target unchanged) do not
        # re-sample the grader or re-walk VERTEBRA_LABELS on every notch.
        # Keyed against the grader identity below, since a re-run
        # segmentation can replace the grader without the key changing.
        self._label_cache: dict = {}
        # The grader object _label_cache was resolved against. A re-run
        # segmentation (set_grader) or reset_state (set_grader(None)) can
        # replace the grader without leaving Screw MPR, so identity -- not
        # exit() -- decides whether the cache is still valid.
        self._label_cache_grader = None

    @property
    def is_active(self) -> bool:
        return self._active

    @property
    def position_percent(self) -> int:
        return int(round(self._position_percent))

    @property
    def rotation_deg(self) -> float:
        return self._rotation_deg

    @property
    def long_axis_offsets_mm(self) -> Tuple[float, float]:
        return (self._long_offsets[0], self._long_offsets[1])

    @property
    def cross_section_offset_mm(self) -> Tuple[float, float]:
        return (self._cross_offset[0], self._cross_offset[1])

    def _reset_view_state(self) -> None:
        """Clear position, rotation, and plane offsets back to the entry view."""
        self._position_percent = 50.0
        self._rotation_deg = 0.0
        self._long_offsets = [0.0, 0.0]
        self._cross_offset = [0.0, 0.0]

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
        self._reset_view_state()
        self._sync_rotation_control()
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
        self._reset_view_state()
        self._sync_rotation_control()
        # The grader-identity check in _vertebra_label_for covers a grader
        # swap mid-session (a re-run segmentation without leaving Screw MPR);
        # this clear only frees the entries built during this session.
        self._label_cache.clear()
        self._label_cache_grader = None
        if was_active:
            for viewer in self._window._get_mpr_viewers():
                viewer.clear_custom_reslice_axes()
            self._set_review_filter(None)
            clear = getattr(self._viewer_3d(), "clear_screw_mpr", None)
            if callable(clear):
                clear()
        self._set_slider_value(50)
        self._window.screw_axis_position_label.setText("Position: 50%")
        self.refresh_selected_screw()
        self.refresh_controls()
        if was_active:
            self._window.statusbar.showMessage("Std MPR restored")

    def set_position(self, percent: int) -> None:
        """Move the perpendicular cross-section from entry to target."""
        self._position_percent = max(0.0, min(100.0, float(percent)))
        if not self._active or self._selected_index is None:
            self._window.screw_axis_position_label.setText(
                f"Position: {self.position_percent}%"
            )
            return
        self._reapply()

    def set_rotation(self, degrees: float) -> None:
        """Spin both long-axis planes about the screw axis (right-hand rule).

        The stored angle wraps rather than clamps, so a continuous drag or
        wheel gesture never sticks at the range boundary: the convention is
        -180 <= rotation_deg < 180, e.g. 185 degrees is stored as -175.
        """
        span = MAX_ROTATION_DEG - MIN_ROTATION_DEG
        self._rotation_deg = (
            (float(degrees) - MIN_ROTATION_DEG) % span
        ) + MIN_ROTATION_DEG
        self._sync_rotation_control()
        self._reapply()

    def rotate(self, delta_deg: float) -> None:
        """Add ``delta_deg`` to the current screw-axis rotation."""
        self.set_rotation(self._rotation_deg + float(delta_deg))

    def nudge_plane(self, plane: str, steps: int) -> None:
        """Move one Screw MPR plane by ``steps`` millimetres along its normal."""
        if not self._active or self._selected_index is None:
            return
        step_count = int(steps)
        if step_count == 0:
            return
        if plane == "cross_section":
            screw = self._screw_at(self._selected_index)
            if screw is None:
                self.exit()
                return
            length = self._screw_length_mm(screw)
            if length <= 1e-9:
                self.exit()
                return
            percent = self._position_percent + (
                step_count * PLANE_STEP_MM * 100.0 / length
            )
            self._position_percent = max(0.0, min(100.0, percent))
            self._set_slider_value(self.position_percent)
        elif plane == "oblique_axial":
            self._long_offsets[0] += step_count * PLANE_STEP_MM
        elif plane == "oblique_sagittal":
            self._long_offsets[1] += step_count * PLANE_STEP_MM
        else:
            return
        self._reapply()

    def reset_view(self) -> None:
        """Restore the position, rotation, and offsets Screw MPR started with."""
        self._reset_view_state()
        self._sync_rotation_control()
        self._set_slider_value(50)
        if not self._active or self._selected_index is None:
            self._window.screw_axis_position_label.setText("Position: 50%")
            return
        self._reapply()
        self._window.statusbar.showMessage("Screw MPR view reset")

    def handle_scroll(
        self,
        plane: str,
        steps: int,
        modifiers: frozenset = frozenset(),
    ) -> None:
        """Route one wheel notch from a viewer into the Screw MPR state."""
        if not self._active:
            return
        if "shift" in modifiers:
            self.rotate(int(steps) * ROTATION_STEP_DEG)
            return
        logical_plane = _PLANE_BY_VIEWER.get(str(plane))
        if logical_plane is None:
            return
        self.nudge_plane(logical_plane, int(steps))

    def handle_rotate_drag(self, plane: str, delta_deg: float) -> None:
        """Apply a Ctrl+left-drag rotation reported by the cross-section view."""
        if not self._active:
            return
        if _PLANE_BY_VIEWER.get(str(plane)) != "cross_section":
            return
        self.rotate(delta_deg)

    def readout_text(self, plane: str, axes) -> str:
        """Build the per-plane readout shown under each Screw MPR viewport."""
        parts = ["Screw-aligned", f"rot {axes.rotation_deg:.0f}°"]
        if plane == "oblique_axial":
            parts.append(f"{axes.long_axis_offset_mm[0]:+.1f} mm")
        elif plane == "oblique_sagittal":
            parts.append(f"{axes.long_axis_offset_mm[1]:+.1f} mm")
        else:
            parts.append(f"{axes.distance_from_entry_mm:.1f} mm from entry")
            offset_u, offset_v = axes.cross_section_offset_mm
            if abs(offset_u) > 1e-9 or abs(offset_v) > 1e-9:
                parts.append(f"shift {offset_u:+.1f}/{offset_v:+.1f} mm")
        return " · ".join(parts)

    def _reapply(self) -> None:
        """Re-derive the three planes for the active screw, or leave the mode."""
        if not self._active or self._selected_index is None:
            return
        screw = self._screw_at(self._selected_index)
        if screw is None:
            self.exit()
            return
        try:
            self._apply_screw(screw)
        except ValueError:
            self.exit()

    @staticmethod
    def _screw_length_mm(screw) -> float:
        """Straight-line length of a screw in millimetres."""
        entry = screw.entry_point
        target = screw.target_point
        return math.sqrt(
            sum(
                (float(target[index]) - float(entry[index])) ** 2
                for index in range(3)
            )
        )

    def _sync_rotation_control(self) -> None:
        """Mirror the rotation state into the panel spin box without recursion."""
        spin = getattr(self._window, "screw_axis_rotation_spin", None)
        if spin is None:
            return
        previous = spin.blockSignals(True)
        spin.setValue(int(round(self._rotation_deg)))
        spin.blockSignals(previous)

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

    def on_screws_removed(self, rows) -> None:
        """Batch removal: exit when the active screw is gone, else reindex."""
        if not self._active or self._selected_index is None:
            return
        removed = sorted({int(row) for row in rows})
        if self._selected_index in removed:
            self.exit()
            return
        shift = sum(1 for row in removed if row < self._selected_index)
        if shift:
            self._selected_index -= shift
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
        rotation_spin = getattr(self._window, "screw_axis_rotation_spin", None)
        if rotation_spin is not None:
            rotation_spin.setEnabled(self._active)
        reset_button = getattr(self._window, "screw_mpr_reset_btn", None)
        if reset_button is not None:
            reset_button.setEnabled(self._active)
        refresh = getattr(self._window, "refresh_mode_indicators", None)
        if callable(refresh):
            refresh()

    def _apply_screw(self, screw) -> None:
        axes = build_screw_mpr_axes(
            screw.entry_point,
            screw.target_point,
            self._position_percent / 100.0,
            rotation_deg=self._rotation_deg,
            long_axis_offset_mm=(
                self._long_offsets[0],
                self._long_offsets[1],
            ),
            cross_section_offset_mm=(
                self._cross_offset[0],
                self._cross_offset[1],
            ),
        )
        identity = self._selected_identity(screw)
        self._set_plane(
            self._window.axial_viewer,
            axes.oblique_axial,
            f"Oblique Axial · {identity}",
            self.readout_text("oblique_axial", axes),
        )
        self._set_plane(
            self._window.sagittal_viewer,
            axes.oblique_sagittal,
            f"Oblique Sagittal · {identity}",
            self.readout_text("oblique_sagittal", axes),
        )
        self._set_plane(
            self._window.coronal_viewer,
            axes.cross_section,
            f"Cross-section · {self.position_percent}%",
            self.readout_text("cross_section", axes),
        )
        self._window.screw_axis_position_label.setText(
            f"Position: {self.position_percent}% "
            f"({axes.distance_from_entry_mm:.1f} mm from entry)"
        )
        # The same three matrices, into 3D: the screw-aligned planes replace the
        # standard indicators, and a textured slice opens at the cross-section
        # -- with only the screw's own vertebra cut, everything else stays whole.
        show = getattr(self._viewer_3d(), "show_screw_mpr", None)
        if callable(show):
            show(
                axes.oblique_axial,
                axes.oblique_sagittal,
                axes.cross_section,
                vertebra_label=self._vertebra_label_for(screw),
                window_level=self._window_level(),
            )

    def _vertebra_label_for(self, screw) -> Optional[int]:
        """Resolve the TotalSegmentator label the 3D cut should apply to.

        (1) The grader's ``detect_label`` -- the most frequent vertebral
        label along the entry-target axis -- is the same rule the screw's
        grade is computed against, so the vertebra cut in 3D matches the
        vertebra the grade is about.
        (2) Otherwise, the screw's own ``vertebra_level`` name ('L4') mapped
        through the same label table pedicle_analyzer uses.
        (3) Otherwise ``None`` (slice only, no cut).
        """
        entry = tuple(float(value) for value in screw.entry_point)
        target = tuple(float(value) for value in screw.target_point)
        grader = getattr(
            getattr(getattr(self._window, "_tool_ctrl", None), "screw_tool", None),
            "grader",
            None,
        )
        if grader is not self._label_cache_grader:
            self._label_cache.clear()
            self._label_cache_grader = grader

        key = (self._selected_index, entry, target)
        if key in self._label_cache:
            return self._label_cache[key]

        label: Optional[int] = None
        if grader is not None:
            try:
                label = grader.detect_label(entry, target)
            except Exception:
                logger.warning(
                    "_vertebra_label_for: grader.detect_label failed", exc_info=True
                )
                label = None
        if label is None:
            level_name = str(getattr(screw, "vertebra_level", "") or "").strip()
            label = _LABEL_BY_LEVEL_NAME.get(level_name)

        self._label_cache[key] = label
        return label

    def _window_level(self) -> Optional[Tuple[float, float]]:
        """Read the panel's window/level sliders for the 3D cross-section slice."""
        window_slider = getattr(self._window, "window_slider", None)
        level_slider = getattr(self._window, "level_slider", None)
        if window_slider is None or level_slider is None:
            return None
        return (float(window_slider.value()), float(level_slider.value()))

    def on_window_level_changed(self, *_args) -> None:
        """Re-push the 3D cross-section slice's colours when the panel changes them.

        The panel's Window/Level sliders live outside this controller (Study
        tab); the ui task wires their ``valueChanged`` signals here so the 3D
        slice keeps matching what the surgeon set, without this controller
        depending on the sliders existing at all when Screw MPR is inactive.
        """
        if not self._active or self._selected_index is None:
            return
        self._reapply()

    def _viewer_3d(self):
        """The window's 3D viewer, or ``None`` for a stand-in window without one."""
        return getattr(self._window, "viewer_3d", None)

    @staticmethod
    def _set_plane(viewer, axes, title, readout) -> None:
        """Push one plane's matrix, header title, and readout to a viewer."""
        viewer.set_custom_reslice_axes(axes, title)
        setter = getattr(viewer, "set_custom_readout", None)
        if callable(setter):
            setter(readout)

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
