"""Tool controller for screw placement and measurement.

Extracted from MainWindow to reduce God Class complexity.
Manages ScrewTool and MeasurementTool instances, dispatches viewer
clicks to the active tool, and maintains screw/measurement UI lists.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, List, Optional

from src.tools.screw_tool import ScrewTool
from src.tools.measurement_tool import MeasurementTool
from src.models.measurement import Measurement
from src.utils.constants import COLOR_SCREW

if TYPE_CHECKING:
    from src.ui.mpr_viewer import MPRViewer

logger = logging.getLogger(__name__)


class ToolController:
    """Manages screw and measurement tool state, click dispatch, and UI lists."""

    MEASUREMENT_TOOLS = {"distance", "angle", "path"}

    def __init__(self, volume_manager, main_window):
        self._vm = volume_manager
        self._window = main_window
        self._current_tool = "navigate"
        self.screw_tool = ScrewTool(volume_manager)
        self.measurement_tool = MeasurementTool()
        self._screw_actors: list = []
        self._active_measure_plane: Optional[str] = None
        self._measurement_entries: list = []
        self._next_measurement_id: int = 1
        self._editing_measurement_row: Optional[int] = None

        self.screw_tool.set_callbacks(
            on_screw_placed=self._on_screw_placed,
            on_state_changed=self._on_screw_state_changed,
        )
        self.measurement_tool.set_callbacks(
            on_measurement_complete=self._on_measurement_complete,
        )

    @property
    def active_tool(self) -> str:
        return self._current_tool

    def set_tool(self, tool: str):
        """Set the active tool."""
        if tool == "measure":
            tool = str(self._window.measure_mode_combo.currentData() or "distance")
        valid_tools = {"navigate", "screw", *self.MEASUREMENT_TOOLS}
        if tool not in valid_tools:
            raise ValueError(f"Unknown tool: {tool}")
        if self._editing_measurement_row is not None:
            self.cancel_measurement_edit(show_message=False)
        is_measurement = tool in self.MEASUREMENT_TOOLS
        edit_controller = getattr(self._window, "_screw_edit_ctrl", None)
        if (
            tool != "navigate"
            and edit_controller is not None
            and edit_controller.is_active
        ):
            edit_controller.cancel(show_message=False)
        self._current_tool = tool

        if tool != "screw":
            self.screw_tool.cancel()
        if not is_measurement:
            self.measurement_tool.cancel()
            self._active_measure_plane = None

        # QActionGroup handles mutual exclusivity; sync checked state
        action_map = {
            "navigate": self._window._select_tool_action,
            "screw": self._window._add_screw_tool_action,
            "distance": self._window._distance_tool_action,
            "angle": self._window._angle_tool_action,
        }
        action = action_map.get(tool)
        if action and not action.isChecked():
            action.setChecked(True)

        if tool == "screw":
            screw_parameters = getattr(self._window, "screw_parameters_group", None)
            if screw_parameters is not None:
                screw_parameters.expand()
            self._window.statusbar.showMessage(
                "Add Screw — Step 1/2: click the entry point in an MPR view"
            )
        elif is_measurement:
            self.measurement_tool.set_mode(tool)
            combo = self._window.measure_mode_combo
            combo_index = combo.findData(tool)
            if combo_index >= 0:
                previous = combo.blockSignals(True)
                combo.setCurrentIndex(combo_index)
                combo.blockSignals(previous)
            self._window.measure_finish_btn.setEnabled(tool == "path")
            required_points = 3 if tool == "angle" else 2
            self._window.statusbar.showMessage(
                f"{tool.capitalize()} — click {required_points} points in one MPR view"
            )
        else:
            self._window.statusbar.showMessage("Select — navigate and inspect")

    def on_viewer_click(self, plane: str, x: float, y: float, z: float):
        """Handle click events from MPR viewers for active tools."""
        if self._vm.get_vtk_image() is None:
            return

        # Update coordinate and HU display in status bar
        self._window._coord_label.setText(
            f"X: {x:.1f}  Y: {y:.1f}  Z: {z:.1f}"
        )
        hu = self._vm.get_voxel_value(x, y, z)
        if hu is not None:
            self._window._hu_label.setText(f"HU: {hu:.0f}")

        edit_controller = getattr(self._window, "_screw_edit_ctrl", None)
        if edit_controller is not None and edit_controller.handle_click(
            plane, x, y, z
        ):
            return

        if self._current_tool == "navigate":
            return

        if self._current_tool == "screw":
            self.screw_tool.set_screw_parameters(
                length=float(self._window.length_spin.value()),
                diameter=float(self._window.diameter_spin.value()),
            )
            screw = self.screw_tool.on_click(x, y, z, plane=plane)
            if screw is not None:
                screw_id = len(self.screw_tool.get_screws()) - 1
                actor = self._window.viewer_3d.add_screw(
                    screw.entry_point,
                    screw.target_point,
                    radius=screw.diameter / 2.0,
                    screw_id=screw_id,
                )
                self._screw_actors.append(actor)
                self._add_screw_to_list(screw)
                # Add screw projection to all MPR viewers
                self._add_screw_to_mpr(screw_id, screw)
                self._window.screw_list_widget.setCurrentRow(screw_id)
                grade_text = self.screw_tool.get_grade_description(screw.grade)
                self._window.statusbar.showMessage(
                    f"Screw placed: {screw.length:.1f} mm, "
                    f"Grade {screw.grade} ({grade_text})"
                )
            return

        if self._current_tool in self.MEASUREMENT_TOOLS:
            pending_points = len(self.measurement_tool.get_pending_points())
            if pending_points == 0:
                if (
                    self._editing_measurement_row is not None
                    and self._active_measure_plane is not None
                    and plane != self._active_measure_plane
                ):
                    self._window.statusbar.showMessage(
                        "Edit this measurement on the "
                        f"{self._active_measure_plane.capitalize()} view"
                    )
                    return
                if self._editing_measurement_row is None:
                    self._active_measure_plane = plane
            elif (
                self._active_measure_plane is not None
                and plane != self._active_measure_plane
            ):
                self._window.statusbar.showMessage(
                    f"Continue measurement on "
                    f"{self._active_measure_plane.capitalize()} view"
                )
                return

            measurement = self.measurement_tool.on_click(x, y, z)
            if measurement is None:
                points = len(self.measurement_tool.get_pending_points())
                self._window.statusbar.showMessage(
                    f"{self.measurement_tool.get_mode().capitalize()} "
                    f"point {points} selected ({plane})"
                )
            else:
                self._active_measure_plane = None

    def _on_screw_placed(self, screw):
        """Handle screw placement callback."""
        # Status is finalized in on_viewer_click after 3D actor creation.
        pass

    def _on_screw_state_changed(self, state: str):
        """Handle screw tool state transitions."""
        if self._current_tool != "screw":
            return
        if state == "waiting_entry":
            self._window.statusbar.showMessage(
                "Add Screw — Step 1/2: click the entry point in an MPR view"
            )
        elif state == "waiting_target":
            self._window.statusbar.showMessage(
                "Add Screw — Step 2/2: click the tip point in the same MPR view"
            )

    def _on_measurement_complete(self, measurement):
        """Handle measurement completion callback."""
        if self._editing_measurement_row is not None:
            row = self._editing_measurement_row
            if not 0 <= row < len(self._measurement_entries):
                self.cancel_measurement_edit(show_message=False)
                return
            measurements = self.measurement_tool.get_measurements()
            appended_index = len(measurements) - 1
            if appended_index >= 0 and appended_index != row:
                self.measurement_tool.remove_measurement(appended_index)
            self.measurement_tool.replace_measurement(row, measurement)

            entry = self._measurement_entries[row]
            measurement_id = entry["id"]
            entry["mode"] = measurement.mode
            entry["label"] = (
                measurement.label
                or self._format_measurement_result(measurement)
            )
            viewer = self._get_viewer_by_plane(entry["plane"])
            if viewer is not None:
                viewer.add_measurement(
                    measurement_id=measurement_id,
                    points=measurement.points,
                    label=entry["label"],
                )
            self._window.viewer_3d.add_measurement(
                measurement_id=measurement_id,
                points=measurement.points,
                label=entry["label"],
            )
            self._editing_measurement_row = None
            self._active_measure_plane = None
            self._refresh_measurement_list()
            self._window.measurement_list_widget.setCurrentRow(row)
            self._window.statusbar.showMessage(
                f"Measurement #{row + 1} updated: {entry['label']}"
            )
            return

        measurement_id = self._next_measurement_id
        self._next_measurement_id += 1

        entry = {
            "id": measurement_id,
            "plane": self._active_measure_plane,
            "mode": measurement.mode,
            "label": (
                measurement.label
                or self._format_measurement_result(measurement)
            ),
        }
        self._measurement_entries.append(entry)

        viewer = self._get_viewer_by_plane(self._active_measure_plane)
        if viewer is not None:
            viewer.add_measurement(
                measurement_id=measurement_id,
                points=measurement.points,
                label=entry["label"],
            )
        self._window.viewer_3d.add_measurement(
            measurement_id=measurement_id,
            points=measurement.points,
            label=entry["label"],
        )
        self._refresh_measurement_list()
        self._window.measurement_list_widget.setCurrentRow(
            len(self._measurement_entries) - 1
        )
        self._window.statusbar.showMessage(
            self._format_measurement_result(measurement)
        )

    def on_measure_mode_changed(self, _index=None):
        """Handle measurement mode changes from UI."""
        mode = self._window.measure_mode_combo.currentData()
        self.measurement_tool.set_mode(mode)
        self._active_measure_plane = None
        self._window.measure_finish_btn.setEnabled(mode == "path")
        if self._current_tool in self.MEASUREMENT_TOOLS:
            if mode == "path":
                self._window.statusbar.showMessage(
                    "Measure tool (Path): click points, then 'Finish Path'"
                )
            elif mode == "angle":
                self._window.statusbar.showMessage(
                    "Measure tool (Angle): click 3 points"
                )
            else:
                self._window.statusbar.showMessage(
                    "Measure tool (Distance): click 2 points"
                )

    def finish_pending_measurement(self):
        """Finish current path measurement."""
        measurement = self.measurement_tool.finish_pending()
        if measurement is None:
            self._window.statusbar.showMessage(
                "Path measurement needs at least 2 points"
            )
            return
        self._active_measure_plane = None

    def clear_measurements(self):
        """Clear all measurements."""
        self.measurement_tool.clear_measurements()
        self._active_measure_plane = None
        self._editing_measurement_row = None
        self._measurement_entries = []
        self._next_measurement_id = 1
        self._window.measurement_list_widget.clear()
        self._clear_measurement_overlays()
        if self._current_tool in self.MEASUREMENT_TOOLS:
            self._window.statusbar.showMessage("Measurements cleared")

    def remove_selected_measurement(self):
        """Remove selected measurement from data model and overlays."""
        self.cancel_measurement_edit(show_message=False)
        row = self._window.measurement_list_widget.currentRow()
        if row < 0:
            self._window.statusbar.showMessage(
                "Select a measurement to remove"
            )
            return
        if row >= len(self._measurement_entries):
            self._window.statusbar.showMessage(
                "Invalid measurement selection"
            )
            return

        entry = self._measurement_entries.pop(row)
        measurement_id = entry["id"]
        self.measurement_tool.remove_measurement(row)

        viewer = self._get_viewer_by_plane(entry["plane"])
        if viewer is not None:
            viewer.remove_measurement(measurement_id)
        if self._window.viewer_3d is not None:
            self._window.viewer_3d.remove_measurement(measurement_id)

        self._refresh_measurement_list()
        self._window.statusbar.showMessage("Selected measurement removed")

    def jump_to_selected_measurement(self):
        """Restore the standard MPR cut containing the selected measurement."""
        row = self._window.measurement_list_widget.currentRow()
        measurements = self.measurement_tool.get_measurements()
        if not 0 <= row < len(self._measurement_entries) or row >= len(measurements):
            self._window.statusbar.showMessage(
                "Select a measurement to show"
            )
            return
        entry = self._measurement_entries[row]
        measurement = measurements[row]
        plane = entry.get("plane")
        viewer = self._get_viewer_by_plane(plane)
        if viewer is None or not measurement.points:
            self._window.statusbar.showMessage(
                "This measurement has no restorable MPR cut"
            )
            return
        point = measurement.points[0]
        position_by_plane = {
            "sagittal": point[0],
            "coronal": point[1],
            "axial": point[2],
        }
        position = position_by_plane.get(plane)
        if position is None:
            return
        screw_mpr_controller = getattr(self._window, "_screw_mpr_ctrl", None)
        if screw_mpr_controller is not None:
            screw_mpr_controller.exit()
        viewer.set_slice_position(float(position))
        self._window.statusbar.showMessage(
            f"Showing measurement #{row + 1} on {plane.capitalize()} cut"
        )

    def begin_edit_selected_measurement(self):
        """Re-measure the selected item while preserving its identifier."""
        row = self._window.measurement_list_widget.currentRow()
        measurements = self.measurement_tool.get_measurements()
        if not 0 <= row < len(self._measurement_entries) or row >= len(measurements):
            self._window.statusbar.showMessage(
                "Select a measurement to edit"
            )
            return
        entry = self._measurement_entries[row]
        plane = entry.get("plane")
        if self._get_viewer_by_plane(plane) is None:
            self._window.statusbar.showMessage(
                "This measurement cannot be edited on an MPR cut"
            )
            return
        measurement = measurements[row]
        self.measurement_tool.cancel()
        self.set_tool(measurement.mode)
        self._editing_measurement_row = row
        self._active_measure_plane = plane
        required_points = 3 if measurement.mode == "angle" else 2
        self._window.statusbar.showMessage(
            f"Edit measurement #{row + 1}: click {required_points} new points "
            f"on the {plane.capitalize()} view (Esc cancels)"
        )

    def cancel_measurement_edit(self, show_message: bool = True) -> None:
        """Cancel a pending re-measure operation without changing saved data."""
        if self._editing_measurement_row is None:
            return
        self.measurement_tool.cancel()
        self._editing_measurement_row = None
        self._active_measure_plane = None
        if show_message:
            self._window.statusbar.showMessage("Measurement edit canceled")

    def select_measurement_from_view(self, measurement_id: int) -> None:
        """Synchronize an MPR-picked measurement with its list row."""
        row = next(
            (
                index
                for index, entry in enumerate(self._measurement_entries)
                if entry.get("id") == int(measurement_id)
            ),
            -1,
        )
        if row < 0:
            return
        if self._current_tool != "navigate":
            self.set_tool("navigate")
        self._window.measurement_list_widget.setCurrentRow(row)
        self.on_measurement_selection_changed(row)

    def on_measurement_selection_changed(self, row: int) -> None:
        """Expose PACS-style handles for the selected MPR measurement."""
        entry = (
            self._measurement_entries[row]
            if 0 <= row < len(self._measurement_entries)
            else None
        )
        if entry is not None:
            self._window._active_selection_kind = "measurement"
        for viewer in self._window._get_mpr_viewers():
            measurement_id = None
            if entry is not None and viewer.plane == entry.get("plane"):
                measurement_id = entry.get("id")
            viewer.set_selected_measurement(measurement_id)

    def update_measurement_point(
        self,
        measurement_id: int,
        point_index: int,
        world_point,
    ) -> None:
        """Move one measurement point and update every linked representation."""
        row = next(
            (
                index
                for index, entry in enumerate(self._measurement_entries)
                if entry.get("id") == int(measurement_id)
            ),
            -1,
        )
        measurements = self.measurement_tool.get_measurements()
        if not 0 <= row < len(measurements):
            return
        original = measurements[row]
        if not 0 <= int(point_index) < len(original.points):
            return
        points = [tuple(point) for point in original.points]
        points[int(point_index)] = tuple(float(value) for value in world_point)
        distance = MeasurementTool._calculate_total_distance(points)
        angle = None
        label = MeasurementTool.format_distance(distance)
        if original.mode == "angle":
            angle = MeasurementTool._calculate_angle_from_points(points[:3])
            if angle is None:
                return
            label = MeasurementTool.format_angle(angle)
        updated = Measurement(
            points=points,
            distance=distance,
            mode=original.mode,
            angle=angle,
            label=label,
        )
        self.measurement_tool.replace_measurement(row, updated)
        entry = self._measurement_entries[row]
        entry["label"] = label
        viewer = self._get_viewer_by_plane(entry.get("plane"))
        if viewer is not None:
            viewer.add_measurement(
                measurement_id=int(measurement_id),
                points=points,
                label=label,
            )
            viewer.set_selected_measurement(int(measurement_id))
        self._window.viewer_3d.add_measurement(
            measurement_id=int(measurement_id),
            points=points,
            label=label,
        )
        self._refresh_measurement_list()
        self._window.measurement_list_widget.setCurrentRow(row)
        self._window.statusbar.showMessage(
            f"Measurement #{row + 1}: {label}"
        )

    def _refresh_measurement_list(self):
        """Refresh measurement list entries."""
        self._window.measurement_list_widget.clear()
        measurements = self.measurement_tool.get_measurements()
        for index, entry in enumerate(self._measurement_entries, start=1):
            plane = (
                entry["plane"].capitalize() if entry["plane"] else "Unknown"
            )
            mode = entry["mode"].capitalize()
            cut_text = ""
            if index - 1 < len(measurements) and measurements[index - 1].points:
                point = measurements[index - 1].points[0]
                cut_position = {
                    "axial": point[2],
                    "sagittal": point[0],
                    "coronal": point[1],
                }.get(entry.get("plane"))
                if cut_position is not None:
                    cut_text = f" @ {cut_position:.1f} mm"
            self._window.measurement_list_widget.addItem(
                f"#{index} | {mode} | {plane}{cut_text} | {entry['label']}"
            )

    def _clear_measurement_overlays(self):
        """Clear 2D/3D measurement overlays."""
        for viewer in [
            self._window.axial_viewer,
            self._window.sagittal_viewer,
            self._window.coronal_viewer,
        ]:
            if viewer is not None:
                viewer.clear_measurements()
        if self._window.viewer_3d is not None:
            self._window.viewer_3d.clear_measurements()

    def _get_viewer_by_plane(
        self, plane: Optional[str]
    ) -> Optional[MPRViewer]:
        """Return MPR viewer for the given plane name."""
        viewer_map = {
            "axial": self._window.axial_viewer,
            "sagittal": self._window.sagittal_viewer,
            "coronal": self._window.coronal_viewer,
        }
        return viewer_map.get(plane)

    def _add_screw_to_list(self, screw):
        """Append one screw entry to list widget."""
        index = self._window.screw_list_widget.count()
        self._window.screw_list_widget.addItem(
            self._format_screw_list_text(index, screw)
        )

    def add_existing_screw(self, screw, select: bool = False) -> int:
        """Register an existing screw in the model and every linked view."""
        self.screw_tool.add_screw(screw)
        screw_id = len(self.screw_tool.get_screws()) - 1
        actor = self._window.viewer_3d.add_screw(
            screw.entry_point,
            screw.target_point,
            radius=screw.diameter / 2.0,
            screw_id=screw_id,
        )
        self._screw_actors.append(actor)
        self._add_screw_to_list(screw)
        self._add_screw_to_mpr(screw_id, screw)
        if select:
            self._window.screw_list_widget.setCurrentRow(screw_id)
        return screw_id

    @staticmethod
    def _format_screw_list_text(index: int, screw) -> str:
        """Return the canonical list label for one screw."""
        level = screw.vertebra_level or "Manual"
        side = screw.side.capitalize() if screw.side else "--"
        return (
            f"#{index + 1}  {level} · {side}\n"
            f"Ø {screw.diameter:.1f} mm  ·  Len {screw.length:.1f} mm  ·  "
            f"Grade {screw.grade}"
        )

    def _add_screw_to_mpr(self, screw_id: int, screw) -> None:
        """Add screw projection overlay to all MPR viewers."""
        for viewer in self._window._get_mpr_viewers():
            viewer.add_screw_overlay(
                screw_id,
                screw.entry_point,
                screw.target_point,
                color=tuple(value / 255.0 for value in COLOR_SCREW),
                diameter=screw.diameter,
            )

    def refresh_screw(self, index: int, screw) -> None:
        """Refresh model-linked 3D, MPR, list, and inspector representations."""
        if index < 0 or index >= len(self.screw_tool.get_screws()):
            raise IndexError(f"Screw index out of range: {index}")

        if index < len(self._screw_actors):
            old_actor = self._screw_actors[index]
            self._window.viewer_3d.remove_screw(old_actor)
        new_actor = self._window.viewer_3d.add_screw(
            screw.entry_point,
            screw.target_point,
            radius=screw.diameter / 2.0,
            screw_id=index,
        )
        if index < len(self._screw_actors):
            self._screw_actors[index] = new_actor
        else:
            self._screw_actors.append(new_actor)

        self._add_screw_to_mpr(index, screw)
        item = self._window.screw_list_widget.item(index)
        if item is not None:
            item.setText(self._format_screw_list_text(index, screw))
        self._window._screw_mpr_ctrl.on_screw_updated(index)

    def regrade_all(self) -> None:
        """Re-grade every screw and refresh its linked representations."""
        for index, screw in enumerate(self.screw_tool.regrade_all()):
            self.refresh_screw(index, screw)

    def set_selected_screw_diameter(self, diameter: float) -> None:
        """Resize the selected screw and refresh every linked representation."""
        row = self._window.screw_list_widget.currentRow()
        if row < 0 or row >= len(self.screw_tool.get_screws()):
            return
        updated = self.screw_tool.update_screw_diameter(row, diameter)
        self.refresh_screw(row, updated)
        self._window.statusbar.showMessage(
            f"Screw #{row + 1} diameter set to {updated.diameter:.1f} mm"
        )

    def remove_selected_screw(self):
        """Remove selected screw from list, tool state, 3D, and 2D views."""
        row = self._window.screw_list_widget.currentRow()
        if row < 0:
            self._window.statusbar.showMessage("Select a screw to remove")
            return

        edit_controller = getattr(self._window, "_screw_edit_ctrl", None)
        if edit_controller is not None:
            edit_controller.on_screw_removed(row)
        self._window._screw_mpr_ctrl.on_screw_removed(row)

        self.screw_tool.remove_screw(row)

        # Rebuild every index-linked view so later viewport picks remain correct.
        screws = self.screw_tool.get_screws()
        self._window.viewer_3d.clear_screws()
        self._screw_actors.clear()
        self._window.screw_list_widget.clear()
        self._rebuild_mpr_screw_overlays(screws)
        for index, screw in enumerate(screws):
            actor = self._window.viewer_3d.add_screw(
                screw.entry_point,
                screw.target_point,
                radius=screw.diameter / 2.0,
                screw_id=index,
            )
            self._screw_actors.append(actor)
            self._add_screw_to_list(screw)
        if screws:
            self._window.screw_list_widget.setCurrentRow(
                min(row, len(screws) - 1)
            )

        if edit_controller is not None:
            edit_controller.refresh_controls()

        self._window.statusbar.showMessage("Selected screw removed")

    def _format_measurement_result(self, measurement) -> str:
        """Format measurement result text for status bar."""
        if measurement.mode == "angle" and measurement.angle is not None:
            return (
                f"Angle: {MeasurementTool.format_angle(measurement.angle)} "
                f"(Span: {MeasurementTool.format_distance(measurement.distance)})"
            )
        return (
            f"Distance: {MeasurementTool.format_distance(measurement.distance)}"
        )

    def clear_screws(self):
        """Clear all placed screws."""
        edit_controller = getattr(self._window, "_screw_edit_ctrl", None)
        if edit_controller is not None:
            edit_controller.reset()
        self._window._screw_mpr_ctrl.exit()
        self.screw_tool.clear_screws()
        self._window.viewer_3d.clear_screws()
        self._screw_actors.clear()
        self._window.screw_list_widget.clear()
        # Clear 2D screw overlays
        for viewer in self._window._get_mpr_viewers():
            viewer.clear_screw_overlays()
        self._window.statusbar.showMessage("All screws cleared")

    def _rebuild_mpr_screw_overlays(self, screws) -> None:
        """Rebuild all MPR screw overlays after index changes."""
        for viewer in self._window._get_mpr_viewers():
            viewer.clear_screw_overlays()
        for idx, screw in enumerate(screws):
            self._add_screw_to_mpr(idx, screw)
