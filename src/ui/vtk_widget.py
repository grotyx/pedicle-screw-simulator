"""
Platform-safe VTK widget factory.

Provides create_vtk_widget() which returns a QVTKRenderWindowInteractor
with a safe_render() convenience method.

On macOS, the default vtkCocoaRenderWindow is used with QWidget base.
CPU volume rendering (vtkFixedPointVolumeRayCastMapper) only blits a 2D
result image to screen, avoiding the OpenGL→Metal glFinish() hang that
occurs with GPU volume mappers during 3D texture upload.

Infinite repaint loop fix:
  On macOS, super().paintEvent() → VTK Render() → Cocoa drawRect →
  schedules another Qt update() → paintEvent() again → infinite loop.
  Solution: dirty-flag gating. paintEvent only calls super() when
  _render_needed is True. safe_render() and resizeEvent set the flag.
"""

import time
import logging
from PyQt6.QtCore import QSize
from PyQt6.QtWidgets import QSizePolicy
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

logger = logging.getLogger(__name__)


def create_vtk_widget(parent):
    """Create a VTK render widget.

    Returns a QVTKRenderWindowInteractor with safe_render() method.
    Uses the platform default render window (vtkCocoaRenderWindow on macOS).
    """
    return _VTKWidget(parent)


class _VTKWidget(QVTKRenderWindowInteractor):
    """QVTKRenderWindowInteractor with dirty-flag rendering.

    Breaks the macOS infinite repaint loop:
      paintEvent → Render() → Cocoa drawRect → update() → paintEvent → ...

    Only calls super().paintEvent() (which triggers VTK Render) when
    _render_needed is True. After rendering, the flag is cleared so the
    next Cocoa-triggered paintEvent is a no-op, breaking the cycle.
    """

    def __init__(self, parent=None, **kw):
        super().__init__(parent, **kw)
        self._iren_initialized = False
        self._render_needed = True  # First paint should render
        self._paint_count = 0
        self._render_count = 0
        self.setSizePolicy(
            QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored
        )

    def _ensure_iren_init(self):
        """Initialize the interactor once (sets Enabled=1)."""
        if self._iren_initialized:
            return
        self._iren_initialized = True
        self._Iren.Initialize()
        logger.info(
            "_Iren.Initialize() called — Enabled=%d",
            self._Iren.GetEnabled(),
        )

    def paintEvent(self, ev):
        """Render only when dirty; skip otherwise to break repaint loop."""
        self._ensure_iren_init()
        self._paint_count += 1

        if not self._render_needed:
            # No-op: breaks the Cocoa drawRect → paintEvent loop.
            # The last rendered frame remains visible in the Cocoa view.
            return

        self._render_needed = False
        self._render_count += 1

        t0 = time.perf_counter()
        super().paintEvent(ev)
        dt = time.perf_counter() - t0

        if dt > 0.05:
            logger.info(
                "render[%d] (paint#%d): %.3fs",
                self._render_count,
                self._paint_count,
                dt,
            )

    def resizeEvent(self, ev):
        """Mark dirty on resize so the scene is re-rendered to fit."""
        self._render_needed = True
        super().resizeEvent(ev)

    def Render(self):
        """Override VTK's Render() to go through the dirty-flag system.

        VTK's interactor calls this during mouse interaction (rotate, scroll).
        The base implementation just calls self.update(). We add the dirty
        flag so paintEvent knows to actually render.
        """
        self._render_needed = True
        self.update()

    def safe_render(self):
        """Request a VTK render on the next paint cycle.

        Sets the dirty flag and schedules a Qt repaint via update().
        The actual VTK Render() happens in paintEvent when the flag is set.
        """
        self._render_needed = True
        self.update()
