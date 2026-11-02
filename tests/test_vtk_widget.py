"""The dirty-flag VTK widget must repaint after being shown again."""

from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor


def test_show_marks_the_widget_dirty_so_a_restored_pane_repaints(monkeypatch):
    from src.ui.vtk_widget import _VTKWidget

    # A real widget needs a GL context (native crash offscreen): call the
    # override on a bare instance with the Qt base handler stubbed out.
    monkeypatch.setattr(
        QVTKRenderWindowInteractor, "showEvent", lambda self, ev: None
    )
    widget = _VTKWidget.__new__(_VTKWidget)
    widget._render_needed = False
    widget.showEvent(None)
    assert widget._render_needed is True


def test_an_expose_redraws_outside_macos(monkeypatch):
    # A palette dragged over the native surface left copies of itself
    # because paints without the dirty flag were skipped on Windows too.
    import src.ui.vtk_widget as vtk_widget

    rendered = []
    monkeypatch.setattr(QVTKRenderWindowInteractor, "paintEvent", lambda self, ev: rendered.append(1))
    widget = vtk_widget._VTKWidget.__new__(vtk_widget._VTKWidget)
    widget._iren_initialized = True
    widget._paint_count = widget._render_count = 0

    for platform, expected in (("win32", [1]), ("darwin", [])):
        rendered.clear()
        monkeypatch.setattr(vtk_widget.sys, "platform", platform)
        widget._render_needed = False
        widget.paintEvent(None)
        assert rendered == expected, platform


def test_startup_counts_only_visible_render_end_events(monkeypatch):
    import src.ui.vtk_widget as vtk_widget
    from src.core.startup_timing import StartupTiming

    timing = StartupTiming(1, clock=lambda: 5)
    timing.mark_shown()
    monkeypatch.setattr(vtk_widget, 'startup_timing', lambda: timing)
    widget = vtk_widget._VTKWidget.__new__(vtk_widget._VTKWidget)
    visible = [False]
    monkeypatch.setattr(widget, 'isVisible', lambda: visible[0])
    widget._on_render_completed(None, 'EndEvent')
    assert timing.first_render is None
    visible[0] = True
    widget._on_render_completed(None, 'EndEvent')
    assert timing.first_render == 4


def test_startup_counts_visible_render_during_interactor_initialization(monkeypatch):
    import src.ui.vtk_widget as vtk_widget
    from src.core.startup_timing import StartupTiming

    timing = StartupTiming(1, clock=lambda: 5)
    timing.mark_shown()
    monkeypatch.setattr(vtk_widget, 'startup_timing', lambda: timing)
    widget = vtk_widget._VTKWidget.__new__(vtk_widget._VTKWidget)
    monkeypatch.setattr(widget, 'isVisible', lambda: True)
    widget._startup_paint = False  # Native Initialize can render before Qt's paint.
    widget._on_render_completed(None, 'EndEvent')
    assert timing.first_render == 4
