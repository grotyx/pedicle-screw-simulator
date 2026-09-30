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
