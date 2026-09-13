"""Direct tests for ViewController window/level, presets, and camera.

Regression scope: src/controllers/view_controller.py — slider changes
reach all three MPR viewers, presets move both sliders, opacity reaches
the 3D viewer, transfer-function presets reach the volume manager, and
camera reset touches MPR plus 3D. Uses lightweight fakes (no Qt, no
MainWindow) so these stay independent of UI construction.
"""

from types import SimpleNamespace

from src.controllers.view_controller import ViewController


class _Slider:
    def __init__(self, value=0):
        self._value = value

    def value(self):
        return self._value

    def setValue(self, value):
        self._value = value


class _Viewer:
    def __init__(self):
        self.window_level = None

    def set_window_level(self, window, level):
        self.window_level = (window, level)


class _Viewer3D:
    def __init__(self):
        self.opacity = None
        self.fit_calls = 0

    def set_volume_opacity(self, opacity):
        self.opacity = opacity

    def fit_to_view(self):
        self.fit_calls += 1


class _VolumeManager:
    def __init__(self):
        self.presets = []

    def set_transfer_function_preset(self, name):
        self.presets.append(name)


def _controller():
    window = SimpleNamespace(
        window_slider=_Slider(1500),
        level_slider=_Slider(400),
        opacity_slider=_Slider(42),
        axial_viewer=_Viewer(),
        sagittal_viewer=_Viewer(),
        coronal_viewer=_Viewer(),
        viewer_3d=_Viewer3D(),
        fit_mpr_views=None,
    )
    calls = []
    window.fit_mpr_views = lambda: calls.append("mpr")
    volume_manager = _VolumeManager()
    return ViewController(volume_manager, window), window, volume_manager, calls


def test_window_level_change_reaches_all_mpr_viewers():
    controller, window, _, _ = _controller()

    controller.on_window_level_changed()

    expected = (1500, 400)
    assert window.axial_viewer.window_level == expected
    assert window.sagittal_viewer.window_level == expected
    assert window.coronal_viewer.window_level == expected


def test_set_preset_moves_both_sliders():
    controller, window, _, _ = _controller()

    controller.set_preset(level=50, window=400)

    assert window.window_slider.value() == 400
    assert window.level_slider.value() == 50


def test_opacity_change_reaches_3d_viewer():
    controller, window, _, _ = _controller()

    controller.on_opacity_changed()

    assert window.viewer_3d.opacity == 0.42


def test_preset_change_reaches_volume_manager():
    controller, _, volume_manager, _ = _controller()

    controller.on_preset_changed("Bone")

    assert volume_manager.presets == ["Bone"]


def test_reset_camera_touches_mpr_and_3d():
    controller, window, _, calls = _controller()

    controller.reset_camera()

    assert calls == ["mpr"]
    assert window.viewer_3d.fit_calls == 1
