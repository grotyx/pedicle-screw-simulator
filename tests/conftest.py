"""Shared Qt fixtures for UI workflow tests.

Canonical home of ``isolated_qsettings`` and ``ui_main_window``. The five
UI test modules used to copy these bodies; they now keep thin shims that
delegate here so behaviour stays identical in one place.
"""

import os

import pytest
from PyQt6.QtWidgets import QWidget

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    from PyQt6.QtCore import pyqtSignal
except Exception:  # pragma: no cover - Qt always present in test env
    pyqtSignal = None


class DummyMPRViewer(QWidget):
    """Lightweight MPR test double for UI workflow tests."""

    if pyqtSignal is not None:
        slice_changed = pyqtSignal(str, float)
        crosshair_moved = pyqtSignal(str, float, float, float)
        header_double_clicked = pyqtSignal(str)

    def __init__(self, plane, volume_manager, parent=None):
        super().__init__(parent)
        self.plane = plane
        self.volume_manager = volume_manager
        self.visible = True
        self.seg_label_value = 0
        self.seg_mask = None
        self.measurements = {}
        self.custom_axes = None
        self.custom_title = None
        self.custom_readout = None
        self.custom_scroll_handler = None
        self.custom_rotate_handler = None
        self.review_screw_id = None
        self.screw_overlays = {}
        self.fit_count = 0
        self.screw_interaction_callbacks = {}
        self.measurement_interaction_callbacks = {}
        self.selected_measurement_id = None
        self.screw_interaction_cancelled = 0
        self.last_slice_position = None
        self.orientation_refresh_count = 0

    def set_window_level(self, _window, _level):
        return

    def set_segmentation_mask(self, mask_image, label_value=0, **_kwargs):
        self.seg_mask = mask_image
        self.seg_label_value = int(label_value)

    def set_segmentation_label(self, label_value):
        self.seg_label_value = int(label_value)

    def set_segmentation_visible(self, visible):
        self.visible = bool(visible)

    def clear_segmentation_mask(self):
        self.seg_mask = None
        self.seg_label_value = 0

    def add_measurement(self, measurement_id, points, label):
        self.measurements[measurement_id] = {"points": points, "label": label}

    def remove_measurement(self, measurement_id):
        self.measurements.pop(measurement_id, None)

    def clear_measurements(self):
        self.measurements = {}

    def set_slice_position(self, position):
        self.last_slice_position = float(position)
        self.volume_manager.set_slice_position(self.plane, float(position))

    def cleanup(self):
        return

    # Stubs for reslice input swap (vertebral isolation)
    def set_reslice_input(self, vtk_image):
        return

    def restore_original_input(self):
        return

    # Stubs for screw projection overlays
    def add_screw_overlay(self, screw_id, entry, target, color=(0.2, 0.8, 0.2), diameter=6.0):
        self.screw_overlays[screw_id] = {
            "entry": tuple(entry),
            "target": tuple(target),
            "diameter": float(diameter),
        }

    def remove_screw_overlay(self, screw_id):
        self.screw_overlays.pop(screw_id, None)

    def clear_screw_overlays(self):
        self.screw_overlays = {}

    def set_custom_reslice_axes(self, axes, title):
        self.custom_axes = axes
        self.custom_title = title

    def set_custom_readout(self, text):
        self.custom_readout = text

    def set_custom_scroll_handler(self, handler):
        self.custom_scroll_handler = handler

    def set_custom_rotate_handler(self, handler):
        self.custom_rotate_handler = handler

    def clear_custom_reslice_axes(self):
        self.custom_axes = None
        self.custom_title = None
        self.custom_readout = None

    def set_review_screw(self, screw_id):
        self.review_screw_id = screw_id

    def set_screw_interaction_callbacks(self, **callbacks):
        self.screw_interaction_callbacks = callbacks

    def set_measurement_interaction_callbacks(self, **callbacks):
        self.measurement_interaction_callbacks = callbacks

    def set_selected_measurement(self, measurement_id):
        self.selected_measurement_id = measurement_id

    def cancel_screw_interaction(self):
        self.screw_interaction_cancelled += 1

    def fit_to_view(self):
        self.fit_count += 1

    def refresh_orientation_markers(self, render=False):
        self.orientation_refresh_count += 1

    # Stubs for _coordinated_initial_render
    _render_guard_active = False
    _settled = False
    _extra_render_count = 0

    def _deferred_initial_render(self):
        return


class DummyViewer3D(QWidget):
    """Lightweight 3D viewer test double for UI workflow tests."""

    if pyqtSignal is not None:
        header_double_clicked = pyqtSignal(str)
        isolation_requested = pyqtSignal(bool)

    def __init__(self, volume_manager, parent=None):
        super().__init__(parent)
        self.visible = True
        self.seg_label_value = 0
        self.seg_mask = None
        self.screws = []
        self.measurements = {}
        self.zoom_factors = []
        self.fit_count = 0
        self.vertebral_mesh_labels = None
        self.volume_visible = True
        self.screw_interaction_cancelled = 0
        # Screw MPR in 3D: the last planes shown, or None once cleared.
        self.screw_mpr_planes = None
        self.screw_mpr_clear_count = 0
        self.screw_mpr_label = None
        self.screw_mpr_window_level = None
        # (isolated, available) as last reported by set_isolation_state.
        self.isolation_state = (False, False)

    def set_isolation_state(self, isolated: bool, available: bool) -> None:
        self.isolation_state = (bool(isolated), bool(available))

    def show_screw_mpr(
        self,
        oblique_axial,
        oblique_sagittal,
        cross_section,
        *,
        vertebra_label=None,
        window_level=None,
    ):
        self.screw_mpr_planes = (oblique_axial, oblique_sagittal, cross_section)
        self.screw_mpr_label = vertebra_label
        self.screw_mpr_window_level = window_level

    def clear_screw_mpr(self):
        self.screw_mpr_planes = None
        self.screw_mpr_clear_count += 1

    def add_screw(
        self, entry_point, target_point, radius=3.0, color=None, screw_id=None
    ):
        class _Property:
            def SetOpacity(self, v): pass
            def GetOpacity(self): return 1.0
        class _Actor:
            def __init__(self, e, t, r, c, sid):
                self.entry_point = e
                self.target_point = t
                self.radius = r
                self.color = c
                self.screw_id = sid
                self._prop = _Property()
            def GetProperty(self):
                return self._prop
        actor = _Actor(
            tuple(entry_point), tuple(target_point), float(radius), color, screw_id
        )
        self.screws.append(actor)
        return actor

    def set_screw_interaction_callbacks(self, **callbacks):
        self.screw_interaction_callbacks = callbacks

    def set_selected_screw(self, screw_id):
        self.selected_screw_id = screw_id

    def cancel_screw_interaction(self):
        self.screw_interaction_cancelled += 1

    def remove_screw(self, actor):
        if actor in self.screws:
            self.screws.remove(actor)

    def clear_screws(self):
        self.screws = []

    def add_measurement(self, measurement_id, points, label):
        self.measurements[measurement_id] = {"points": points, "label": label}

    def remove_measurement(self, measurement_id):
        self.measurements.pop(measurement_id, None)

    def clear_measurements(self):
        self.measurements = {}

    def set_segmentation_mask(self, mask_image, label_value=0, **_kwargs):
        self.seg_mask = mask_image
        self.seg_label_value = int(label_value)

    def set_segmentation_label(self, label_value):
        self.seg_label_value = int(label_value)

    def set_segmentation_visible(self, visible):
        self.visible = bool(visible)

    # Stubs for _coordinated_initial_render
    _render_guard_active = False
    _extra_render_count = 0

    def _deferred_render_phase1(self):
        return

    def clear_segmentation_mask(self):
        self.seg_mask = None
        self.seg_label_value = 0

    def set_vertebral_mesh(self, _mask_image, labels=None):
        self.vertebral_mesh_labels = list(labels) if labels is not None else None

    def clear_vertebral_mesh(self):
        return

    def set_volume_visible(self, visible):
        self.volume_visible = bool(visible)

    def set_bone_opacity(self, _opacity):
        return

    def _update_plane_positions(self):
        return

    def zoom_camera(self, factor):
        self.zoom_factors.append(float(factor))

    def fit_to_view(self):
        self.fit_count += 1

    def cleanup(self):
        return


def make_isolated_qsettings(tmp_path, monkeypatch):
    """Redirect MainWindow QSettings into temp INI files."""
    from PyQt6.QtCore import QSettings

    import src.ui.main_window as main_window_module

    directory = tmp_path / "qsettings"
    directory.mkdir(parents=True, exist_ok=True)

    def factory(organization="Default", application="App", *_args, **_kwargs):
        path = directory / f"{organization}-{application}.ini"
        return QSettings(str(path), QSettings.Format.IniFormat)

    monkeypatch.setattr(main_window_module, "QSettings", factory)
    # app_settings() only checks the legacy scope once per process; reset
    # that guard so each test's migration behaviour is independent.
    monkeypatch.setattr(main_window_module, "_migrated", False)
    return factory


def make_ui_main_window(
    monkeypatch, qtbot, isolated_qsettings, theme_name="soft_light"
):
    """Build MainWindow with lightweight viewer stubs."""
    from PyQt6.QtWidgets import QApplication

    import src.ui.main_window as main_window_module

    monkeypatch.setattr(main_window_module, "MPRViewer", DummyMPRViewer)
    monkeypatch.setattr(main_window_module, "Viewer3D", DummyViewer3D)
    QApplication.instance().setProperty("themeName", theme_name)

    window = main_window_module.MainWindow()
    qtbot.addWidget(window)
    return window


@pytest.fixture
def isolated_qsettings(tmp_path, monkeypatch):
    """Redirect MainWindow QSettings into temp INI files."""
    return make_isolated_qsettings(tmp_path, monkeypatch)


@pytest.fixture
def ui_main_window(monkeypatch, qtbot, isolated_qsettings):
    """Build MainWindow with lightweight viewer stubs."""
    return make_ui_main_window(monkeypatch, qtbot, isolated_qsettings)
