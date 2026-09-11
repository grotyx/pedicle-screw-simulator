"""
Tests for Viewer3D (Phase 2).

Verifies:
- SurfaceGenerationThread is removed
- GPU volume rendering pipeline is present
- Transfer function preset application
- Screw/measurement methods still exist
"""

import os
import sys
from types import SimpleNamespace

import pytest

from src.ui.click_detector import DoubleClickDetector

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


@pytest.fixture
def viewer_3d_with_volume():
    """A Viewer3D with a loaded-volume rendering pipeline (no Qt/VTK widget)."""
    import vtk

    from src.ui.viewer_3d import Viewer3D
    from src.utils.constants import TRANSFER_FUNCTION_PRESETS

    viewer = Viewer3D.__new__(Viewer3D)
    viewer._color_tf = vtk.vtkColorTransferFunction()
    viewer._opacity_tf = vtk.vtkPiecewiseFunction()
    viewer._gradient_opacity_tf = vtk.vtkPiecewiseFunction()
    viewer._volume_property = vtk.vtkVolumeProperty()
    viewer._volume_mapper = vtk.vtkFixedPointVolumeRayCastMapper()
    viewer._volume_added = False
    viewer._model_opacity = 1.0
    viewer.volume_manager = SimpleNamespace(
        get_transfer_function_config=lambda: TRANSFER_FUNCTION_PRESETS["Bone"]
    )
    return viewer


class TestNoSurfaceThread:
    """Verify SurfaceGenerationThread has been completely removed."""

    def test_no_surface_generation_thread_class(self):
        from src.ui import viewer_3d
        assert not hasattr(viewer_3d, "SurfaceGenerationThread")

    def test_no_surface_thread_import(self):
        """Module should not reference SurfaceGenerationThread anywhere."""
        import inspect

        from src.ui import viewer_3d
        source = inspect.getsource(viewer_3d)
        assert "SurfaceGenerationThread" not in source

    def test_no_qthread_import(self):
        """Viewer3D module should not import QThread."""
        import inspect

        from src.ui import viewer_3d
        source = inspect.getsource(viewer_3d)
        assert "QThread" not in source


class TestViewer3DInterface:
    """Verify the public API of Viewer3D is intact."""

    def test_has_update_volume(self):
        from src.ui.viewer_3d import Viewer3D
        assert hasattr(Viewer3D, "update_volume")

    def test_has_apply_transfer_function_preset(self):
        from src.ui.viewer_3d import Viewer3D
        assert hasattr(Viewer3D, "apply_transfer_function_preset")

    def test_has_set_volume_opacity(self):
        from src.ui.viewer_3d import Viewer3D
        assert hasattr(Viewer3D, "set_volume_opacity")

    def test_has_add_screw(self):
        from src.ui.viewer_3d import Viewer3D
        assert hasattr(Viewer3D, "add_screw")

    def test_has_remove_screw(self):
        from src.ui.viewer_3d import Viewer3D
        assert hasattr(Viewer3D, "remove_screw")

    def test_has_clear_screws(self):
        from src.ui.viewer_3d import Viewer3D
        assert hasattr(Viewer3D, "clear_screws")

    def test_has_add_measurement(self):
        from src.ui.viewer_3d import Viewer3D
        assert hasattr(Viewer3D, "add_measurement")

    def test_has_remove_measurement(self):
        from src.ui.viewer_3d import Viewer3D
        assert hasattr(Viewer3D, "remove_measurement")

    def test_has_clear_measurements(self):
        from src.ui.viewer_3d import Viewer3D
        assert hasattr(Viewer3D, "clear_measurements")

    def test_has_set_segmentation_mask(self):
        from src.ui.viewer_3d import Viewer3D
        assert hasattr(Viewer3D, "set_segmentation_mask")

    def test_has_cleanup(self):
        from src.ui.viewer_3d import Viewer3D
        assert hasattr(Viewer3D, "cleanup")

    def test_has_explicit_zoom_and_fit_controls(self):
        from src.ui.viewer_3d import Viewer3D
        assert hasattr(Viewer3D, "zoom_camera")
        assert hasattr(Viewer3D, "fit_to_view")

    def test_has_initial_view_and_model_opacity_controls(self):
        from src.ui.viewer_3d import Viewer3D

        assert hasattr(Viewer3D, "reset_to_initial_view")
        assert hasattr(Viewer3D, "set_model_opacity")
        assert hasattr(Viewer3D, "set_model_transparency")
        assert hasattr(Viewer3D, "set_vertebral_transparency")

    def test_has_set_bone_opacity_backward_compat(self):
        """set_bone_opacity should still exist for backward compatibility."""
        from src.ui.viewer_3d import Viewer3D
        assert hasattr(Viewer3D, "set_bone_opacity")

    def test_has_direct_screw_interaction_api(self):
        from src.ui.viewer_3d import Viewer3D

        assert hasattr(Viewer3D, "set_screw_interaction_callbacks")
        assert hasattr(Viewer3D, "set_selected_screw")

    def test_plane_indicators_use_soft_translucent_palette(self):
        from src.ui.viewer_3d import Viewer3D

        viewer = Viewer3D.__new__(Viewer3D)
        viewer._plane_actors = {}
        viewer._planes_visible = True

        viewer._create_plane_indicators()

        expected_colors = {
            "axial": (255 / 255, 242 / 255, 178 / 255),
            "sagittal": (168 / 255, 225 / 255, 235 / 255),
            "coronal": (226 / 255, 183 / 255, 218 / 255),
        }
        for plane, expected_color in expected_colors.items():
            plane_data = viewer._plane_actors[plane]
            assert plane_data["actor"].GetProperty().GetColor() == pytest.approx(
                expected_color
            )
            assert plane_data["actor"].GetProperty().GetOpacity() == pytest.approx(
                0.14
            )
            assert plane_data[
                "outline_actor"
            ].GetProperty().GetOpacity() == pytest.approx(0.48)

    def test_plane_visibility_toggle_is_inside_3d_viewport(
        self, qtbot, monkeypatch
    ):
        from PyQt6.QtWidgets import QWidget

        from src.ui import viewer_3d

        monkeypatch.setattr(
            viewer_3d,
            "create_vtk_widget",
            lambda parent: QWidget(parent),
        )
        monkeypatch.setattr(
            viewer_3d.Viewer3D,
            "_setup_vtk_pipeline",
            lambda self: None,
        )
        volume_manager = SimpleNamespace(add_observer=lambda *_args: None)

        viewer = viewer_3d.Viewer3D(volume_manager)
        qtbot.addWidget(viewer)

        assert viewer.plane_visibility_toggle.parent() is viewer.viewport_container
        assert viewer.plane_visibility_toggle.isCheckable()
        assert viewer.plane_visibility_toggle.isChecked()
        assert viewer.plane_visibility_toggle.text() == "Planes On"

        assert viewer.pan_mode_toggle.parent() is viewer.viewport_container
        assert viewer.pan_mode_toggle.isCheckable()
        assert viewer.pan_mode_toggle.isChecked() is False
        assert "Shift" in viewer.pan_mode_toggle.toolTip()

        assert viewer.reset_view_button.parent() is viewer.viewport_container
        assert viewer.reset_view_button.text() == "Reset View"
        assert viewer.model_opacity_controls.parent() is viewer.viewport_container
        assert viewer.model_opacity_slider.parent() is viewer.model_opacity_controls
        assert viewer.model_opacity_slider.minimum() == 0
        assert viewer.model_opacity_slider.maximum() == 100
        assert viewer.model_opacity_slider.value() == 50
        assert viewer.model_opacity_label.text() == "Vertebra Transparency 50%"

        viewer.model_opacity_slider.setValue(40)
        assert viewer._vertebral_transparency == pytest.approx(0.40)
        assert viewer._vertebral_surface_opacity == pytest.approx(0.60)

    def test_plane_visibility_toggle_controls_every_3d_plane(self):
        import vtk

        from src.ui.viewer_3d import Viewer3D

        viewer = Viewer3D.__new__(Viewer3D)
        viewer._plane_actors = {
            plane: {
                "actor": vtk.vtkActor(),
                "outline_actor": vtk.vtkActor(),
            }
            for plane in ("axial", "sagittal", "coronal")
        }
        viewer._planes_visible = True
        viewer._renderer = None
        viewer.plane_visibility_toggle = None

        viewer.set_plane_indicators_visible(False)

        assert viewer._planes_visible is False
        for plane_data in viewer._plane_actors.values():
            assert plane_data["actor"].GetVisibility() == 0
            assert plane_data["outline_actor"].GetVisibility() == 0


class TestScrewVisualGeometry:
    def test_visual_has_large_entry_head_shaft_and_pointed_tip(self):
        from src.ui.viewer_3d import create_screw_visual

        visual = create_screw_visual(
            screw_id=4,
            entry_point=(0.0, 0.0, 0.0),
            target_point=(0.0, 0.0, 40.0),
            radius=3.0,
        )

        assert visual.screw_id == 4
        assert visual.entry_actor.GetObjectName() == "screw-entry-head"
        assert visual.shaft_actor.GetObjectName() == "screw-shaft"
        assert visual.tip_actor.GetObjectName() == "screw-pointed-tip"
        assert visual.entry_actor.GetBounds()[1] - visual.entry_actor.GetBounds()[0] > 6.0
        assert visual.tip_actor.GetBounds()[5] == pytest.approx(40.0, abs=0.2)

    def test_zero_length_visual_is_rejected(self):
        from src.ui.viewer_3d import create_screw_visual

        with pytest.raises(ValueError, match="length"):
            create_screw_visual(0, (1, 2, 3), (1, 2, 3), 3.0)

    def test_screw_only_picker_ignores_occluding_anatomy_props(self):
        from src.ui.viewer_3d import Viewer3D, create_screw_visual

        viewer = Viewer3D.__new__(Viewer3D)
        viewer._screw_actors = [
            create_screw_visual(0, (0, 0, 0), (0, 0, 40), 3.0)
        ]

        picker = viewer._build_screw_only_picker()

        assert picker.GetPickFromList() == 1
        assert picker.GetPickList().GetNumberOfItems() == 3

    def test_distal_quarter_of_3d_shaft_is_selected_as_tip(self):
        from src.ui.viewer_3d import Viewer3D, create_screw_visual

        viewer = Viewer3D.__new__(Viewer3D)
        visual = create_screw_visual(0, (0, 0, 0), (0, 0, 40), 3.0)

        assert hasattr(Viewer3D, "_resolve_screw_pick_part")
        assert viewer._resolve_screw_pick_part(
            visual, "shaft", (0.0, 0.0, 31.0)
        ) == "tip"

    def test_focus_on_world_point_recenters_and_zooms_camera(self):
        import vtk

        from src.ui.viewer_3d import Viewer3D

        viewer = Viewer3D.__new__(Viewer3D)
        viewer._renderer = vtk.vtkRenderer()
        viewer._request_render = lambda: None
        camera = viewer._renderer.GetActiveCamera()
        camera.SetFocalPoint(0.0, 0.0, 0.0)
        camera.SetPosition(0.0, 0.0, 100.0)

        assert hasattr(Viewer3D, "focus_on_world_point")
        viewer.focus_on_world_point((10.0, 20.0, 30.0), zoom_factor=2.0)

        assert camera.GetFocalPoint() == pytest.approx((10.0, 20.0, 30.0))
        assert camera.GetPosition() == pytest.approx((10.0, 20.0, 80.0))

    def test_reset_to_initial_view_restores_sagittal_orientation_and_fits(self):
        import vtk

        from src.ui.viewer_3d import Viewer3D

        viewer = Viewer3D.__new__(Viewer3D)
        viewer._renderer = vtk.vtkRenderer()
        cube = vtk.vtkCubeSource()
        cube.SetBounds(-20.0, 20.0, -10.0, 10.0, -30.0, 30.0)
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(cube.GetOutputPort())
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        viewer._renderer.AddActor(actor)
        render_calls = []
        viewer._request_render = lambda: render_calls.append(True)
        camera = viewer._renderer.GetActiveCamera()
        camera.SetPosition(100.0, 20.0, 5.0)
        camera.SetFocalPoint(5.0, 6.0, 7.0)
        camera.SetViewUp(0.2, 0.3, 0.9)

        viewer.reset_to_initial_view()

        assert camera.GetDirectionOfProjection() == pytest.approx(
            (-1.0, 0.0, 0.0), abs=1e-6
        )
        assert camera.GetViewUp() == pytest.approx((0.0, 0.0, 1.0), abs=1e-6)
        assert camera.GetFocalPoint() == pytest.approx((0.0, 0.0, 0.0))
        assert render_calls == [True]

    def test_model_opacity_scales_anatomy_without_changing_screw(self):
        import vtk

        from src.ui.viewer_3d import Viewer3D, create_screw_visual

        viewer = Viewer3D.__new__(Viewer3D)
        viewer._renderer = vtk.vtkRenderer()
        viewer._opacity_tf = vtk.vtkPiecewiseFunction()
        viewer._volume_added = False
        viewer._vertebral_mesh_actor = vtk.vtkActor()
        viewer._segmentation_actor = vtk.vtkActor()
        viewer._vertebral_mesh_default_opacity = 0.86
        viewer._segmentation_default_opacity = 0.35
        viewer._selected_screw_id = None
        viewer._screw_actors = [
            create_screw_visual(0, (0, 0, 0), (0, 0, 40), 3.0)
        ]
        viewer.volume_manager = SimpleNamespace(
            get_transfer_function_config=lambda: {
                "opacity_points": [(-1000.0, 0.0), (300.0, 0.8)]
            }
        )
        render_calls = []
        viewer._request_render = lambda: render_calls.append(True)
        screw_opacity = viewer._screw_actors[0].shaft_actor.GetProperty().GetOpacity()

        viewer.set_model_opacity(0.5)

        assert viewer._model_opacity == pytest.approx(0.5)
        assert viewer._opacity_tf.GetValue(300.0) == pytest.approx(0.4)
        assert viewer._vertebral_mesh_actor.GetProperty().GetOpacity() == pytest.approx(0.43)
        assert viewer._segmentation_actor.GetProperty().GetOpacity() == pytest.approx(0.175)
        assert viewer._screw_actors[0].shaft_actor.GetProperty().GetOpacity() == pytest.approx(
            screw_opacity
        )
        assert render_calls == [True]

    def test_vertebral_transparency_reveals_interior_without_changing_volume(self):
        import vtk

        from src.ui.viewer_3d import Viewer3D, create_screw_visual

        viewer = Viewer3D.__new__(Viewer3D)
        viewer._renderer = vtk.vtkRenderer()
        viewer._opacity_tf = vtk.vtkPiecewiseFunction()
        viewer._opacity_tf.AddPoint(-1000.0, 0.0)
        viewer._opacity_tf.AddPoint(300.0, 0.8)
        viewer._volume_added = False
        viewer._vertebral_mesh_actor = vtk.vtkActor()
        viewer._segmentation_actor = vtk.vtkActor()
        viewer._vertebral_mesh_default_opacity = 0.86
        viewer._segmentation_default_opacity = 0.35
        viewer._selected_screw_id = None
        viewer._screw_actors = [
            create_screw_visual(0, (0, 0, 0), (0, 0, 40), 3.0)
        ]
        viewer.volume_manager = SimpleNamespace(
            get_transfer_function_config=lambda: {
                "opacity_points": [(-1000.0, 0.0), (300.0, 0.8)]
            }
        )
        viewer._request_render = lambda: None
        screw_opacity = viewer._screw_actors[0].shaft_actor.GetProperty().GetOpacity()

        viewer.set_vertebral_transparency(1.0)

        assert viewer._vertebral_transparency == pytest.approx(1.0)
        assert viewer._vertebral_surface_opacity == pytest.approx(0.20)
        assert viewer._vertebral_mesh_actor.GetProperty().GetOpacity() == pytest.approx(
            0.20
        )
        assert viewer._segmentation_actor.GetProperty().GetOpacity() == pytest.approx(
            0.20
        )
        assert viewer._opacity_tf.GetValue(300.0) == pytest.approx(0.8)
        assert viewer._screw_actors[0].shaft_actor.GetProperty().GetOpacity() == pytest.approx(
            screw_opacity
        )

        viewer.set_vertebral_transparency(0.0)

        assert viewer._vertebral_surface_opacity == pytest.approx(1.0)
        assert viewer._vertebral_mesh_actor.GetProperty().GetOpacity() == pytest.approx(1.0)

    def test_selected_screw_focus_preserves_user_vertebral_transparency(self):
        import vtk

        from src.ui.viewer_3d import Viewer3D

        viewer = Viewer3D.__new__(Viewer3D)
        viewer._vertebral_mesh_actor = vtk.vtkActor()
        viewer._segmentation_actor = vtk.vtkActor()
        viewer._vertebral_mesh_default_opacity = 0.86
        viewer._segmentation_default_opacity = 0.35
        viewer._vertebral_surface_opacity = 0.50

        viewer._apply_screw_focus()

        assert viewer._vertebral_mesh_actor.GetProperty().GetOpacity() == pytest.approx(0.50)
        assert viewer._segmentation_actor.GetProperty().GetOpacity() == pytest.approx(0.50)

        viewer._apply_screw_focus()

        assert viewer._vertebral_mesh_actor.GetProperty().GetOpacity() == pytest.approx(0.50)
        assert viewer._segmentation_actor.GetProperty().GetOpacity() == pytest.approx(0.50)

    def test_ray_plane_intersection_preserves_camera_facing_drag_depth(self):
        from src.ui.viewer_3d import intersect_ray_plane

        point = intersect_ray_plane(
            near_point=(5.0, 7.0, -10.0),
            far_point=(5.0, 7.0, 10.0),
            plane_origin=(0.0, 0.0, 3.0),
            plane_normal=(0.0, 0.0, 1.0),
        )

        assert point == pytest.approx((5.0, 7.0, 3.0))

    def test_double_click_toggles_3d_pointer_move_lock(self):
        from src.ui.viewer_3d import Viewer3D

        viewer = Viewer3D.__new__(Viewer3D)
        viewer._double_click_detector = DoubleClickDetector()
        viewer._screw_drag_active = False
        viewer._active_screw_part = None
        viewer._screw_drag_plane_origin = None
        viewer._screw_drag_plane_normal = None
        viewer._screw_drag_begin_callback = None
        viewer._screw_drag_callback = None
        viewer._screw_drag_end_callback = None
        viewer._screw_select_callback = None
        viewer._screw_actors = []
        events = []
        viewer.set_screw_interaction_callbacks(
            on_begin=lambda screw_id, part, world: events.append(
                ("begin", screw_id, part, world)
            ) or True,
            on_drag=lambda world, source: events.append(("drag", world, source)),
            on_end=lambda: events.append(("end",)),
            on_select=lambda screw_id: events.append(("select", screw_id)),
        )

        assert viewer._process_screw_press(
            20,
            20,
            (5, "shaft"),
            (1.0, 2.0, 3.0),
            (0.0, 0.0, 1.0),
            timestamp=1.0,
        ) == "selected"
        assert viewer._process_screw_press(
            22,
            21,
            (5, "shaft"),
            (1.0, 2.0, 3.0),
            (0.0, 0.0, 1.0),
            timestamp=1.2,
        ) == "started"
        viewer._emit_screw_pointer_move((4.0, 5.0, 6.0))
        viewer._on_screw_left_release(None, None)

        assert viewer._screw_drag_active is True
        assert viewer._active_screw_part == (5, "shaft")
        assert viewer._screw_drag_plane_origin == (1.0, 2.0, 3.0)
        assert events[-1] == ("drag", (4.0, 5.0, 6.0), "3D")

        assert viewer._process_screw_press(
            100, 100, None, None, None, timestamp=2.0
        ) == "locked"
        assert viewer._process_screw_press(
            101, 101, None, None, None, timestamp=2.2
        ) == "ended"
        assert viewer._screw_drag_active is False
        assert events[-1] == ("end",)


class TestSegmentationSurfaceSmoothing:
    def test_raw_segmentation_pipeline_smooths_image_mesh_and_normals(self):
        import inspect

        from src.ui.viewer_3d import Viewer3D

        source = inspect.getsource(Viewer3D._render_segmentation_actor)
        assert "vtkImageGaussianSmooth" in source
        assert "vtkWindowedSincPolyDataFilter" in source
        assert "vtkPolyDataNormals" in source
        assert "ComputePointNormalsOn" in source

    def test_no_start_surface_generation(self):
        from src.ui.viewer_3d import Viewer3D
        assert not hasattr(Viewer3D, "_start_surface_generation")

    def test_no_on_surface_ready(self):
        from src.ui.viewer_3d import Viewer3D
        assert not hasattr(Viewer3D, "_on_surface_ready")

    def test_3d_viewport_uses_flat_neutral_charcoal_background(self):
        import inspect

        from src.ui import viewer_3d

        assert viewer_3d.VIEWPORT_BACKGROUND == pytest.approx(
            (0.035, 0.035, 0.035)
        )
        source = inspect.getsource(viewer_3d.Viewer3D._setup_vtk_pipeline)
        assert "GradientBackgroundOff" in source
        assert "SetBackground2" not in source


class TestVolumeMapperPerformanceConfig:
    """Verify vtkFixedPointVolumeRayCastMapper (CPU) config to avoid macOS glFinish hang."""

    def test_source_keeps_the_cpu_mapper_and_never_hard_codes_the_gpu_mapper(self):
        """macOS keeps vtkFixedPointVolumeRayCastMapper; nobody instantiates
        vtkGPUVolumeRayCastMapper directly — vtkSmartVolumeMapper owns that
        choice and falls back to CPU when no context is available."""
        import inspect

        from src.ui import viewer_3d
        source = inspect.getsource(viewer_3d)
        assert "vtkFixedPointVolumeRayCastMapper" in source
        assert "vtkSmartVolumeMapper" in source
        assert "vtkGPUVolumeRayCastMapper" not in source

    def test_source_has_auto_adjust_sample_distances(self):
        """Mapper must enable AutoAdjustSampleDistances."""
        import inspect

        from src.ui import viewer_3d
        source = inspect.getsource(viewer_3d)
        assert "SetAutoAdjustSampleDistances" in source

    def test_source_has_interactive_sample_distance_for_the_cpu_mapper(self):
        """The fixed-point CPU mapper must keep its InteractiveSampleDistance."""
        import inspect

        from src.ui import viewer_3d
        source = inspect.getsource(viewer_3d.create_volume_mapper)
        assert "SetInteractiveSampleDistance(3.0)" in source
        assert "SetInteractiveUpdateRate(1.0 / 15.0)" in source

    def test_source_has_sample_distance(self):
        """update_volume must set SampleDistance per volume tier."""
        import inspect

        from src.ui import viewer_3d
        source = inspect.getsource(viewer_3d)
        assert "SetSampleDistance" in source
        assert "assess_volume_scale" in source

    def test_source_sets_sample_distance_for_all_tiers(self):
        """All volume tiers must get explicit sample distance with absolute floor."""
        import inspect

        from src.ui import viewer_3d
        source = inspect.getsource(viewer_3d)
        assert "else:" in source
        assert "max(" in source
        assert source.count("sample_dist") >= 5

    def test_source_has_desired_update_rate(self):
        """Interactor must set DesiredUpdateRate for LOD during interaction."""
        import inspect

        from src.ui import viewer_3d
        source = inspect.getsource(viewer_3d)
        assert "SetDesiredUpdateRate" in source

    def test_source_has_reasonable_still_update_rate(self):
        """StillUpdateRate must NOT be 0.001 (=1000s/frame hang). Need >= 0.1."""
        import inspect
        import re

        from src.ui import viewer_3d
        source = inspect.getsource(viewer_3d)
        assert "SetStillUpdateRate" in source
        match = re.search(r'SetStillUpdateRate\(([0-9.]+)\)', source)
        assert match, "SetStillUpdateRate call not found"
        rate = float(match.group(1))
        assert rate >= 0.1, f"StillUpdateRate {rate} too low — causes hang"

    def test_source_has_two_phase_rendering(self):
        """update_volume must use deferred two-phase rendering for fast first frame."""
        import inspect

        from src.ui import viewer_3d
        source = inspect.getsource(viewer_3d)
        assert "_target_sample_dist" in source
        assert "_deferred_render_phase1" in source
        assert "_execute_phase2" in source
        assert "QTimer" in source
        assert "_request_render" in source

    def test_source_uses_downsampled_volume(self):
        """Must use downsampled volume (vtkImageShrink3D) for faster CPU ray casting."""
        import inspect

        from src.ui import viewer_3d
        source = inspect.getsource(viewer_3d)
        assert "downsample_vtk_image" in source
        assert "_downsampled_image" in source

    def test_source_never_gives_full_res_to_mapper(self):
        """Phase 2 (in render guard callback) must NOT swap to full-resolution volume."""
        import inspect

        from src.ui import viewer_3d
        callback_src = inspect.getsource(viewer_3d.Viewer3D._render_guard_callback)
        assert "SetInputData" not in callback_src

    def test_source_disables_multisampling(self):
        """Render window must disable MSAA for performance."""
        import inspect

        from src.ui import viewer_3d
        source = inspect.getsource(viewer_3d)
        assert "SetMultiSamples" in source

    def test_source_disables_lock_to_input_spacing(self):
        """Lock must be OFF — input-spacing-derived distance is too fine (~0.67mm)."""
        import inspect

        from src.ui import viewer_3d
        source = inspect.getsource(viewer_3d)
        assert "LockSampleDistanceToInputSpacingOff" in source

    def test_render_guard_aborts_during_setup(self):
        """Render guard aborts renders when state is GUARD."""
        import inspect

        from src.ui import viewer_3d
        callback_src = inspect.getsource(viewer_3d.Viewer3D._render_guard_callback)
        assert "SetAbortRender(1)" in callback_src
        assert "_RS_GUARD" in callback_src

    def test_request_render_calls_safe_render(self):
        """_request_render must call safe_render or Render()."""
        import inspect

        from src.ui import viewer_3d
        method_src = inspect.getsource(viewer_3d.Viewer3D._request_render)
        assert "safe_render" in method_src

    def test_phase2_uses_qtimer(self):
        """Phase 2 must be scheduled via QTimer (no threading.Timer needed)."""
        import inspect

        from src.ui import viewer_3d
        source = inspect.getsource(viewer_3d)
        assert "QTimer.singleShot" in source
        assert "_execute_phase2" in source
        phase1_src = inspect.getsource(viewer_3d.Viewer3D._deferred_render_phase1)
        assert "QTimer.singleShot" in phase1_src

    def test_interaction_coarsens_sampling(self):
        """_on_interaction_start must use coarse sampling for responsiveness."""
        import inspect

        from src.ui import viewer_3d
        start_src = inspect.getsource(viewer_3d.Viewer3D._on_interaction_start)
        assert "_target_sample_dist" in start_src
        assert "SetSampleDistance" in start_src

    def test_phase1_enables_widget_updates(self):
        """Phase 1 must enable widget updates before rendering."""
        import inspect

        from src.ui import viewer_3d
        phase1_src = inspect.getsource(viewer_3d.Viewer3D._deferred_render_phase1)
        assert "setUpdatesEnabled(True)" in phase1_src
        assert "safe_render" in phase1_src

    def test_no_process_events_in_update_volume(self):
        """update_volume must NOT call processEvents — it triggers implicit renders that hang."""
        import inspect

        from src.ui import viewer_3d
        method_source = inspect.getsource(viewer_3d.Viewer3D.update_volume)
        assert "processEvents" not in method_source

    def test_no_render_in_update_volume(self):
        """update_volume must NOT call Render() — deferred to QTimer Phase 1."""
        import inspect

        from src.ui import viewer_3d
        method_source = inspect.getsource(viewer_3d.Viewer3D.update_volume)
        assert "GetRenderWindow().Render()" not in method_source

    def test_has_render_guard(self):
        """Must have VTK-level render guard to block macOS drawRect callbacks."""
        import inspect

        from src.ui import viewer_3d
        source = inspect.getsource(viewer_3d)
        assert "_render_guard_active" in source
        assert "SetAbortRender" in source
        assert "StartEvent" in source

    def test_widget_updates_disabled_during_setup(self):
        """update_volume disables widget updates; Phase 1 re-enables before render."""
        import inspect

        from src.ui import viewer_3d
        update_src = inspect.getsource(viewer_3d.Viewer3D.update_volume)
        phase1_src = inspect.getsource(viewer_3d.Viewer3D._deferred_render_phase1)
        # update_volume DISABLES but does NOT re-enable
        assert "setUpdatesEnabled(False)" in update_src
        assert "setUpdatesEnabled(True)" not in update_src
        # Phase 1 RE-ENABLES before rendering
        assert "setUpdatesEnabled(True)" in phase1_src


class TestVolumeMapperSelection:
    """Platform decides the mapper; macOS keeps the fixed-point CPU caster."""

    def test_macos_keeps_the_fixed_point_cpu_mapper(self):
        from src.ui.viewer_3d import select_volume_mapper_kind
        from src.utils.vtk_helpers import MAPPER_KIND_CPU

        assert select_volume_mapper_kind("darwin") == MAPPER_KIND_CPU

    def test_windows_and_linux_use_the_smart_mapper(self):
        from src.ui.viewer_3d import select_volume_mapper_kind
        from src.utils.vtk_helpers import MAPPER_KIND_SMART

        assert select_volume_mapper_kind("win32") == MAPPER_KIND_SMART
        assert select_volume_mapper_kind("linux") == MAPPER_KIND_SMART

    def test_cpu_mapper_configuration_is_byte_for_byte_todays(self):
        import vtk

        from src.ui.viewer_3d import create_volume_mapper
        from src.utils.vtk_helpers import MAPPER_KIND_CPU

        mapper = create_volume_mapper(MAPPER_KIND_CPU)
        assert isinstance(mapper, vtk.vtkFixedPointVolumeRayCastMapper)
        assert mapper.GetBlendMode() == vtk.vtkVolumeMapper.COMPOSITE_BLEND
        assert mapper.GetAutoAdjustSampleDistances() == 0
        assert mapper.GetLockSampleDistanceToInputSpacing() == 0
        assert mapper.GetInteractiveSampleDistance() == pytest.approx(3.0)

    def test_smart_mapper_requests_the_default_render_mode(self):
        import vtk

        from src.ui.viewer_3d import create_volume_mapper
        from src.utils.vtk_helpers import MAPPER_KIND_SMART

        mapper = create_volume_mapper(MAPPER_KIND_SMART)
        assert isinstance(mapper, vtk.vtkSmartVolumeMapper)
        assert (
            mapper.GetRequestedRenderMode()
            == vtk.vtkSmartVolumeMapper.DefaultRenderMode
        )
        assert mapper.GetBlendMode() == vtk.vtkVolumeMapper.COMPOSITE_BLEND
        assert mapper.GetAutoAdjustSampleDistances() == 0
        assert mapper.GetInteractiveUpdateRate() == pytest.approx(1.0 / 15.0)

    def test_smart_mapper_has_no_interactive_sample_distance_knob(self):
        """VTK 9.7 documents why the LOD drives SetSampleDistance directly."""
        from src.ui.viewer_3d import create_volume_mapper
        from src.utils.vtk_helpers import MAPPER_KIND_SMART

        mapper = create_volume_mapper(MAPPER_KIND_SMART)
        assert not hasattr(mapper, "SetInteractiveSampleDistance")
        assert not hasattr(mapper, "LockSampleDistanceToInputSpacingOff")
        assert hasattr(mapper, "SetSampleDistance")

    def test_unknown_mapper_kind_raises_value_error(self):
        from src.ui.viewer_3d import create_volume_mapper

        with pytest.raises(ValueError, match="mapper kind"):
            create_volume_mapper("quantum")

    def test_pipeline_builds_the_cpu_mapper_on_macos(self, monkeypatch):
        import vtk

        from src.ui import viewer_3d
        from src.utils.vtk_helpers import MAPPER_KIND_CPU

        monkeypatch.setattr(viewer_3d.sys, "platform", "darwin")
        viewer = viewer_3d.Viewer3D.__new__(viewer_3d.Viewer3D)
        mapper = viewer._build_volume_mapper()

        assert isinstance(mapper, vtk.vtkFixedPointVolumeRayCastMapper)
        assert viewer._mapper_kind == MAPPER_KIND_CPU

    def test_pipeline_builds_the_smart_mapper_on_windows(self, monkeypatch):
        import vtk

        from src.ui import viewer_3d
        from src.utils.vtk_helpers import MAPPER_KIND_SMART

        monkeypatch.setattr(viewer_3d.sys, "platform", "win32")
        viewer = viewer_3d.Viewer3D.__new__(viewer_3d.Viewer3D)
        mapper = viewer._build_volume_mapper()

        assert isinstance(mapper, vtk.vtkSmartVolumeMapper)
        assert viewer._mapper_kind == MAPPER_KIND_SMART

    def test_setup_pipeline_delegates_to_build_volume_mapper(self):
        import inspect

        from src.ui.viewer_3d import Viewer3D

        source = inspect.getsource(Viewer3D._setup_vtk_pipeline)
        assert "self._build_volume_mapper()" in source
        assert "vtkFixedPointVolumeRayCastMapper()" not in source


class TestRenderModeNaming:
    """The log must name the mode the mapper actually used."""

    def test_gpu_mode_is_named(self):
        import vtk

        from src.ui.viewer_3d import describe_render_mode

        fake = SimpleNamespace(
            GetLastUsedRenderMode=lambda: vtk.vtkSmartVolumeMapper.GPURenderMode
        )
        assert describe_render_mode(fake) == "gpu"

    def test_cpu_raycast_mode_is_named(self):
        import vtk

        from src.ui.viewer_3d import describe_render_mode

        fake = SimpleNamespace(
            GetLastUsedRenderMode=lambda: vtk.vtkSmartVolumeMapper.RayCastRenderMode
        )
        assert describe_render_mode(fake) == "cpu-raycast"

    def test_smart_mapper_before_first_render_is_undefined(self):
        from src.ui.viewer_3d import create_volume_mapper, describe_render_mode
        from src.utils.vtk_helpers import MAPPER_KIND_SMART

        mapper = create_volume_mapper(MAPPER_KIND_SMART)
        assert describe_render_mode(mapper) == "undefined"

    def test_fixed_point_mapper_has_no_last_used_mode(self):
        from src.ui.viewer_3d import create_volume_mapper, describe_render_mode
        from src.utils.vtk_helpers import MAPPER_KIND_CPU

        mapper = create_volume_mapper(MAPPER_KIND_CPU)
        assert describe_render_mode(mapper) == "cpu-fixed-point"

    def test_unrecognised_mode_value_is_reported_not_raised(self):
        from src.ui.viewer_3d import describe_render_mode

        assert describe_render_mode(SimpleNamespace(GetLastUsedRenderMode=lambda: 99)) == "unknown"


class TestTransferFunctionPresets:
    """Verify transfer function presets are usable from constants."""

    def test_presets_have_required_keys(self):
        from src.utils.constants import TRANSFER_FUNCTION_PRESETS
        required_keys = [
            "color_points", "opacity_points", "gradient_opacity_points",
            "shade", "ambient", "diffuse", "specular", "specular_power",
            "blend_mode",
        ]
        for name, config in TRANSFER_FUNCTION_PRESETS.items():
            for key in required_keys:
                assert key in config, f"Preset '{name}' missing key '{key}'"

    def test_bone_preset_exists(self):
        from src.utils.constants import TRANSFER_FUNCTION_PRESETS
        assert "Bone" in TRANSFER_FUNCTION_PRESETS

    def test_mip_blend_mode(self):
        from src.utils.constants import TRANSFER_FUNCTION_PRESETS
        assert TRANSFER_FUNCTION_PRESETS["MIP (Maximum Intensity)"]["blend_mode"] == "maximum_intensity"

    def test_four_presets_defined(self):
        from src.utils.constants import TRANSFER_FUNCTION_PRESETS
        assert len(TRANSFER_FUNCTION_PRESETS) == 4

    def test_preset_change_preserves_model_opacity_scale(self):
        import vtk

        from src.ui.viewer_3d import Viewer3D
        from src.utils.constants import TRANSFER_FUNCTION_PRESETS

        viewer = Viewer3D.__new__(Viewer3D)
        viewer._color_tf = vtk.vtkColorTransferFunction()
        viewer._opacity_tf = vtk.vtkPiecewiseFunction()
        viewer._gradient_opacity_tf = vtk.vtkPiecewiseFunction()
        viewer._volume_property = vtk.vtkVolumeProperty()
        viewer._volume_mapper = vtk.vtkFixedPointVolumeRayCastMapper()
        viewer._volume_added = False
        viewer._model_opacity = 0.5

        viewer.apply_transfer_function_preset("Bone")

        hu, base_opacity = TRANSFER_FUNCTION_PRESETS["Bone"][
            "opacity_points"
        ][-1]
        assert viewer._opacity_tf.GetValue(hu) == pytest.approx(
            base_opacity * 0.5
        )

    def test_opacity_scale_persists_across_preset_change(self, viewer_3d_with_volume):
        viewer = viewer_3d_with_volume
        viewer.set_volume_opacity(0.3)
        viewer.apply_transfer_function_preset("Soft Tissue")
        from src.utils.constants import TRANSFER_FUNCTION_PRESETS
        hu, base = TRANSFER_FUNCTION_PRESETS["Soft Tissue"]["opacity_points"][-1]
        assert viewer._opacity_tf.GetValue(hu) == pytest.approx(base * 0.3, abs=1e-6)


class TestVolumeRenderSettings:
    """Sample distance is unchanged; only the shrink factors gain a mapper axis."""

    def test_small_volume_is_full_resolution_on_the_smart_mapper(self):
        from src.ui.viewer_3d import volume_render_settings
        from src.utils.vtk_helpers import MAPPER_KIND_SMART

        sample_dist, shrink = volume_render_settings(
            "small", (0.39, 0.39, 1.0), MAPPER_KIND_SMART
        )
        assert shrink == (1, 1, 1)
        assert sample_dist == pytest.approx(1.0)

    def test_small_volume_keeps_todays_shrink_on_the_cpu_mapper(self):
        from src.ui.viewer_3d import volume_render_settings
        from src.utils.vtk_helpers import MAPPER_KIND_CPU

        sample_dist, shrink = volume_render_settings(
            "small", (0.39, 0.39, 1.0), MAPPER_KIND_CPU
        )
        assert shrink == (2, 2, 1)
        assert sample_dist == pytest.approx(1.0)

    def test_sample_distance_is_identical_for_both_mappers(self):
        from src.ui.viewer_3d import volume_render_settings
        from src.utils.vtk_helpers import (
            MAPPER_KIND_CPU,
            MAPPER_KIND_SMART,
            VOLUME_TIERS,
        )

        for tier in VOLUME_TIERS:
            for spacing in ((0.39, 0.39, 1.0), (0.4, 0.4, 0.4), (1.0, 1.0, 1.0)):
                cpu_dist, _ = volume_render_settings(tier, spacing, MAPPER_KIND_CPU)
                smart_dist, _ = volume_render_settings(tier, spacing, MAPPER_KIND_SMART)
                assert cpu_dist == pytest.approx(smart_dist)

    def test_tier_sample_distances_match_the_pre_change_formulas(self):
        from src.ui.viewer_3d import volume_render_settings
        from src.utils.vtk_helpers import MAPPER_KIND_CPU

        assert volume_render_settings("xl", (1.0, 1.0, 1.0), MAPPER_KIND_CPU)[0] == pytest.approx(4.0)
        assert volume_render_settings("xl", (0.3, 0.3, 0.3), MAPPER_KIND_CPU)[0] == pytest.approx(2.0)
        assert volume_render_settings("large", (0.5, 0.5, 0.5), MAPPER_KIND_CPU)[0] == pytest.approx(1.5)
        assert volume_render_settings("medium", (0.4, 0.4, 0.4), MAPPER_KIND_CPU)[0] == pytest.approx(1.2)
        assert volume_render_settings("medium", (1.0, 1.0, 1.0), MAPPER_KIND_CPU)[0] == pytest.approx(2.5)
        assert volume_render_settings("small", (1.0, 1.0, 1.0), MAPPER_KIND_CPU)[0] == pytest.approx(2.0)

    def test_update_volume_uses_the_shared_settings_helper(self):
        import inspect

        from src.ui.viewer_3d import Viewer3D, volume_render_settings

        assert "shrink_factors(" in inspect.getsource(volume_render_settings)
        source = inspect.getsource(Viewer3D.update_volume)
        assert "volume_render_settings(" in source
        assert "self._mapper_kind" in source
        assert "shrink = (3, 3, 3)" not in source


class TestTwoPhaseLodSurvivesTheSmartMapper:
    """SetSampleDistance is the only LOD knob both mappers share."""

    def _viewer_with(self, kind):
        from src.ui.viewer_3d import Viewer3D, create_volume_mapper

        viewer = Viewer3D.__new__(Viewer3D)
        viewer._volume_mapper = create_volume_mapper(kind)
        viewer._target_sample_dist = 1.0
        viewer._render_generation = 7
        viewer._request_render = lambda: None
        return viewer

    def test_smart_mapper_coarsens_during_interaction_and_restores_after(self):
        from src.utils.vtk_helpers import MAPPER_KIND_SMART

        viewer = self._viewer_with(MAPPER_KIND_SMART)
        viewer._on_interaction_start(None, None)
        assert viewer._volume_mapper.GetSampleDistance() == pytest.approx(3.0)
        viewer._on_interaction_end(None, None)
        assert viewer._volume_mapper.GetSampleDistance() == pytest.approx(1.0)

    def test_cpu_mapper_coarsens_during_interaction_and_restores_after(self):
        from src.utils.vtk_helpers import MAPPER_KIND_CPU

        viewer = self._viewer_with(MAPPER_KIND_CPU)
        viewer._on_interaction_start(None, None)
        assert viewer._volume_mapper.GetSampleDistance() == pytest.approx(3.0)
        viewer._on_interaction_end(None, None)
        assert viewer._volume_mapper.GetSampleDistance() == pytest.approx(1.0)

    def test_phase2_restores_fine_sampling_for_the_current_generation(self):
        from src.utils.vtk_helpers import MAPPER_KIND_SMART

        viewer = self._viewer_with(MAPPER_KIND_SMART)
        viewer._volume_mapper.SetSampleDistance(2.0)
        viewer._execute_phase2(7)
        assert viewer._volume_mapper.GetSampleDistance() == pytest.approx(1.0)

    def test_phase2_ignores_a_stale_generation(self):
        from src.utils.vtk_helpers import MAPPER_KIND_SMART

        viewer = self._viewer_with(MAPPER_KIND_SMART)
        viewer._volume_mapper.SetSampleDistance(2.0)
        viewer._execute_phase2(6)
        assert viewer._volume_mapper.GetSampleDistance() == pytest.approx(2.0)

    def test_phase1_logs_the_mapper_and_the_render_mode(self):
        import inspect

        from src.ui.viewer_3d import Viewer3D

        source = inspect.getsource(Viewer3D._deferred_render_phase1)
        assert "self._mapper_kind" in source
        # The mode itself is read one event-loop turn later, after the paint.
        assert "self._log_render_mode" in source
        mode_src = inspect.getsource(Viewer3D._log_render_mode)
        assert "describe_render_mode(mapper)" in mode_src


class _LateModeMapper:
    """A mapper whose render mode is only defined once Render() has run."""

    UNDEFINED = 5
    GPU = 2

    def __init__(self, defines_on_render=True):
        self._defines_on_render = bool(defines_on_render)
        self._mode = self.UNDEFINED
        self.sample_distances = []

    def SetSampleDistance(self, distance):
        self.sample_distances.append(float(distance))

    def GetSampleDistance(self):
        return self.sample_distances[-1] if self.sample_distances else 0.0

    def Render(self):
        if self._defines_on_render:
            self._mode = self.GPU

    def GetLastUsedRenderMode(self):
        return self._mode


class _PaintingWidget:
    """A vtk_widget stub whose safe_render() paints straight away."""

    def __init__(self, mapper):
        self._mapper = mapper
        self.render_count = 0

    def setUpdatesEnabled(self, _enabled):
        return None

    def safe_render(self):
        self.render_count += 1
        self._mapper.Render()


class TestRenderModeIsLoggedAfterTheFirstRealRender:
    """The dirty-flag paint gate means the mode is unknown at request time."""

    @staticmethod
    def _viewer(mapper):
        from src.ui.viewer_3d import Viewer3D

        viewer = Viewer3D.__new__(Viewer3D)
        viewer._volume_mapper = mapper
        viewer._volume_added = True
        viewer._target_sample_dist = 1.0
        viewer._render_generation = 3
        viewer._render_state = Viewer3D._RS_NORMAL
        viewer._mapper_kind = "smart"
        viewer.vtk_widget = _PaintingWidget(mapper)
        viewer._request_render = viewer.vtk_widget.safe_render
        return viewer

    def test_phase1_log_names_the_mode_the_paint_settled_on(self, qtbot, caplog):
        import logging

        mapper = _LateModeMapper()
        viewer = self._viewer(mapper)

        with caplog.at_level(logging.INFO, logger="src.ui.viewer_3d"):
            viewer._deferred_render_phase1()
            assert "render mode" not in caplog.text
            qtbot.wait(20)

        assert "Phase 1 render mode=gpu (mapper=smart)" in caplog.text
        assert "undefined" not in caplog.text

    def test_an_undefined_mode_waits_and_phase2_reports_it_anyway(
        self, qtbot, caplog
    ):
        import logging

        mapper = _LateModeMapper(defines_on_render=False)
        viewer = self._viewer(mapper)

        with caplog.at_level(logging.INFO, logger="src.ui.viewer_3d"):
            viewer._deferred_render_phase1()
            qtbot.wait(20)
            assert "render mode" not in caplog.text

            mapper._defines_on_render = True
            viewer._execute_phase2(3)
            qtbot.wait(20)

        assert "Phase 2 render mode=gpu (mapper=smart)" in caplog.text

    def test_the_mode_is_logged_once_per_volume(self, qtbot, caplog):
        import logging

        mapper = _LateModeMapper()
        viewer = self._viewer(mapper)

        with caplog.at_level(logging.INFO, logger="src.ui.viewer_3d"):
            viewer._deferred_render_phase1()
            qtbot.wait(20)
            viewer._execute_phase2(3)
            qtbot.wait(20)

        assert caplog.text.count("render mode=") == 1


class TestSingleLabelSurfaceExtractor:
    """The generic overlay path must match the vertebral mesh pipeline."""

    @staticmethod
    def _sphere_mask(dims=(24, 24, 24), radius=8.0, label=27):
        import numpy as np
        import vtk
        from vtk.util.numpy_support import numpy_to_vtk

        z_index, y_index, x_index = np.indices((dims[2], dims[1], dims[0]))
        center = (np.asarray(dims) - 1) / 2.0
        values = np.where(
            np.sqrt(
                (x_index - center[0]) ** 2
                + (y_index - center[1]) ** 2
                + (z_index - center[2]) ** 2
            )
            <= radius,
            label,
            0,
        ).astype(np.uint16)

        image = vtk.vtkImageData()
        image.SetDimensions(*dims)
        image.SetSpacing(1.0, 1.0, 1.0)
        image.GetPointData().SetScalars(numpy_to_vtk(values.ravel(), deep=True))
        return image

    def _render(self, label_value):
        import vtk

        from src.ui.viewer_3d import Viewer3D

        viewer = Viewer3D.__new__(Viewer3D)
        viewer._renderer = vtk.vtkRenderer()
        viewer._segmentation_actor = None
        viewer._segmentation_mask_image = self._sphere_mask()
        viewer._segmentation_label_value = label_value
        viewer._segmentation_color = (0.9, 0.8, 0.7)
        viewer._request_render = lambda: None
        viewer._render_segmentation_actor()
        return viewer

    def test_source_uses_flying_edges_not_marching_cubes(self):
        import inspect

        from src.ui.viewer_3d import Viewer3D

        source = inspect.getsource(Viewer3D._render_segmentation_actor)
        assert "vtkFlyingEdges3D()" in source
        assert "vtkMarchingCubes()" not in source

    def test_source_shares_the_vertebral_mesh_fairing_constants(self):
        import inspect

        from src.ui.viewer_3d import Viewer3D

        source = inspect.getsource(Viewer3D._render_segmentation_actor)
        assert "SetNumberOfIterations(MESH_SMOOTHING_ITERATIONS)" in source
        assert "SetPassBand(MESH_SMOOTHING_PASSBAND)" in source

    def test_flying_edges_path_produces_a_non_empty_actor(self):
        viewer = self._render(27)

        assert viewer._segmentation_actor is not None
        mapper = viewer._segmentation_actor.GetMapper()
        mapper.Update()
        assert mapper.GetInput().GetNumberOfCells() > 0

    def test_all_labels_mode_still_produces_a_surface(self):
        viewer = self._render(0)

        assert viewer._segmentation_actor is not None
        mapper = viewer._segmentation_actor.GetMapper()
        mapper.Update()
        assert mapper.GetInput().GetNumberOfCells() > 0

    def test_absent_label_produces_no_actor(self):
        viewer = self._render(31)

        assert viewer._segmentation_actor is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
class TestThe3DHeaderFollowsTheTheme:
    """The 3D title used to be hard-coded white, unreadable on light themes."""

    def test_the_inline_header_sheet_sets_no_colour_of_its_own(self):
        import inspect

        from src.ui.viewer_3d import Viewer3D

        source = inspect.getsource(Viewer3D._setup_ui)
        assert "font-weight: bold; padding: 2px;" in source
        assert "color: white" not in source

    def test_every_palette_colours_the_shared_viewer_header(self):
        from src.ui.styles import THEMES, load_stylesheet

        for name, palette in THEMES.items():
            rule = load_stylesheet(name).split("QLabel#viewerHeader")[1]
            assert palette["viewer_foreground"] in rule.split("}")[0]


class TestCpuRaycastFallback:
    """A smart mapper that ends up on the CPU must stop being fed full-res."""

    @staticmethod
    def _viewer(dims=(64, 64, 48), spacing=(0.39, 0.39, 1.0)):
        import vtk

        from src.ui.viewer_3d import Viewer3D
        from src.utils.vtk_helpers import MAPPER_KIND_SMART

        image = vtk.vtkImageData()
        image.SetDimensions(*dims)
        image.SetSpacing(*spacing)
        image.AllocateScalars(vtk.VTK_SHORT, 1)

        calls = {"input_dims": [], "sample_distance": [], "auto_adjust": [],
                 "renders": 0}

        class _Mapper:
            def SetAutoAdjustSampleDistances(self, value):
                calls["auto_adjust"].append(bool(value))

            def SetInputData(self, data):
                calls["input_dims"].append(data.GetDimensions())

            def SetSampleDistance(self, value):
                calls["sample_distance"].append(float(value))

            def GetLastUsedRenderMode(self):
                return vtk.vtkSmartVolumeMapper.RayCastRenderMode

        viewer = Viewer3D.__new__(Viewer3D)
        viewer._mapper_kind = MAPPER_KIND_SMART
        viewer._shrink_kind = MAPPER_KIND_SMART
        viewer._volume_mapper = _Mapper()
        viewer._volume_added = True
        viewer._render_mode_logged = False
        viewer.volume_manager = SimpleNamespace(
            get_vtk_image=lambda: image,
            dimensions=dims,
            spacing=spacing,
        )
        viewer.vtk_widget = SimpleNamespace(
            safe_render=lambda: calls.__setitem__("renders", calls["renders"] + 1)
        )
        return viewer, calls

    def test_the_fallback_re_downsamples_by_the_cpu_table(self):
        from src.utils.vtk_helpers import MAPPER_KIND_CPU

        viewer, calls = self._viewer()

        viewer._apply_cpu_raycast_fallback()

        assert viewer._shrink_kind == MAPPER_KIND_CPU
        # "small" on the smart table is (1, 1, 1) -- no downsampling at all --
        # which is exactly what a CPU ray caster cannot afford.
        assert calls["input_dims"] == [(32, 32, 48)]
        assert calls["auto_adjust"] == [True]
        assert calls["renders"] == 1

    def test_reading_a_cpu_raycast_mode_triggers_the_fallback(self):
        from src.utils.vtk_helpers import MAPPER_KIND_CPU

        viewer, calls = self._viewer()

        viewer._log_render_mode("Phase 1")

        assert viewer._shrink_kind == MAPPER_KIND_CPU
        assert calls["input_dims"] == [(32, 32, 48)]

    def test_a_gpu_mode_leaves_the_smart_table_alone(self):
        import vtk

        from src.utils.vtk_helpers import MAPPER_KIND_SMART

        viewer, calls = self._viewer()
        viewer._volume_mapper.GetLastUsedRenderMode = (
            lambda: vtk.vtkSmartVolumeMapper.GPURenderMode
        )

        viewer._log_render_mode("Phase 1")

        assert viewer._shrink_kind == MAPPER_KIND_SMART
        assert calls["input_dims"] == []

    def test_the_fallback_runs_only_once(self):
        viewer, calls = self._viewer()
        viewer._apply_cpu_raycast_fallback()
        viewer._render_mode_logged = False

        viewer._log_render_mode("Phase 2")

        assert len(calls["input_dims"]) == 1
