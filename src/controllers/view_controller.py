"""View controller for window/level, presets, opacity, and camera.

Extracted from MainWindow to reduce God Class complexity.
Manages 2D window/level adjustments, transfer function preset
selection, volume opacity, and camera resets.
"""

import logging

logger = logging.getLogger(__name__)


class ViewController:
    """Manages view-related controls: window/level, presets, opacity, camera."""

    def __init__(self, volume_manager, main_window):
        self._vm = volume_manager
        self._window = main_window

    def on_window_level_changed(self):
        """Handle window/level slider changes."""
        window = self._window.window_slider.value()
        level = self._window.level_slider.value()

        for viewer in [
            self._window.axial_viewer,
            self._window.sagittal_viewer,
            self._window.coronal_viewer,
        ]:
            viewer.set_window_level(window, level)

    def set_preset(self, level: int, window: int):
        """Set window/level preset."""
        self._window.window_slider.setValue(window)
        self._window.level_slider.setValue(level)

    def on_opacity_changed(self):
        """Handle volume opacity slider change."""
        opacity = self._window.opacity_slider.value() / 100.0
        self._window.viewer_3d.set_volume_opacity(opacity)

    def on_preset_changed(self, preset_name: str):
        """Handle transfer function preset change."""
        self._vm.set_transfer_function_preset(preset_name)

    def reset_camera(self):
        """Reset all camera views."""
        self._window.fit_mpr_views()
        self._window.viewer_3d.fit_to_view()
