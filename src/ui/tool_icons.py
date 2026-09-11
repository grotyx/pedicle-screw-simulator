"""Small vector-style icons for the unified planning tool palette."""

from typing import Optional

from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtGui import QColor, QIcon, QPainter, QPainterPath, QPen, QPixmap

from .styles import DEFAULT_THEME, get_theme

#: Every glyph :func:`create_tool_icon` knows how to draw. Icons are drawn on a
#: nominal 22x22 canvas and scaled to the requested size, so a single set of
#: coordinates serves the toolbar, the menus and the in-panel buttons.
TOOL_ICON_KINDS = (
    "select",
    "screw",
    "distance",
    "angle",
    "open",
    "save",
    "fit",
    "zoom_in",
    "zoom_out",
    "pan",
    "run",
    "screw_mpr",
    "layout",
    "reset",
)


def create_tool_icon(
    kind: str,
    color: Optional[str] = None,
    size: int = 22,
) -> QIcon:
    """Create a crisp single-colour icon without external image assets.

    ``color`` is any Qt-parsable colour string; the theme's ``accent`` token is
    the intended source so :meth:`MainWindow.apply_theme` can repaint the whole
    palette when the user switches themes.
    """
    if kind not in TOOL_ICON_KINDS:
        raise ValueError(f"Unknown tool icon: {kind}")

    ink = QColor(color) if color else QColor()
    if not ink.isValid():
        ink = QColor(get_theme(DEFAULT_THEME)["accent"])

    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    try:
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.scale(size / 22.0, size / 22.0)
        pen = QPen(ink, 1.8)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        painter.setPen(pen)
        _draw_glyph(painter, kind, ink)
    finally:
        painter.end()
    return QIcon(pixmap)


def _draw_glyph(painter: QPainter, kind: str, ink: QColor) -> None:
    """Draw one glyph on a 22x22 canvas with the pen already configured."""
    if kind == "select":
        path = QPainterPath(QPointF(5.0, 3.0))
        path.lineTo(QPointF(17.0, 12.0))
        path.lineTo(QPointF(11.5, 13.0))
        path.lineTo(QPointF(14.5, 19.0))
        path.lineTo(QPointF(11.5, 20.0))
        path.lineTo(QPointF(8.7, 14.0))
        path.lineTo(QPointF(5.0, 18.0))
        path.closeSubpath()
        painter.setBrush(ink)
        painter.drawPath(path)
    elif kind == "screw":
        painter.drawLine(QPointF(5.0, 17.0), QPointF(17.0, 5.0))
        painter.setBrush(ink)
        painter.drawEllipse(QPointF(5.0, 17.0), 3.0, 3.0)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawLine(QPointF(9.0, 15.0), QPointF(12.0, 18.0))
        painter.drawLine(QPointF(11.5, 12.5), QPointF(14.5, 15.5))
        painter.drawLine(QPointF(14.0, 10.0), QPointF(17.0, 13.0))
    elif kind == "distance":
        painter.drawLine(QPointF(4.0, 6.0), QPointF(4.0, 17.0))
        painter.drawLine(QPointF(18.0, 6.0), QPointF(18.0, 17.0))
        painter.drawLine(QPointF(5.0, 11.5), QPointF(17.0, 11.5))
        painter.drawLine(QPointF(5.0, 11.5), QPointF(8.0, 9.0))
        painter.drawLine(QPointF(5.0, 11.5), QPointF(8.0, 14.0))
        painter.drawLine(QPointF(17.0, 11.5), QPointF(14.0, 9.0))
        painter.drawLine(QPointF(17.0, 11.5), QPointF(14.0, 14.0))
    elif kind == "angle":
        vertex = QPointF(6.0, 17.0)
        painter.drawLine(vertex, QPointF(6.0, 4.0))
        painter.drawLine(vertex, QPointF(19.0, 17.0))
        painter.drawArc(QRectF(3.0, 8.0, 12.0, 12.0), 0, 90 * 16)
    elif kind == "open":
        path = QPainterPath(QPointF(3.0, 17.5))
        path.lineTo(QPointF(3.0, 5.5))
        path.lineTo(QPointF(9.0, 5.5))
        path.lineTo(QPointF(11.0, 8.0))
        path.lineTo(QPointF(19.0, 8.0))
        path.lineTo(QPointF(19.0, 17.5))
        path.closeSubpath()
        painter.drawPath(path)
        painter.drawLine(QPointF(3.0, 10.5), QPointF(19.0, 10.5))
    elif kind == "save":
        painter.drawRect(QRectF(4.0, 4.0, 14.0, 14.0))
        painter.drawRect(QRectF(7.5, 4.0, 7.0, 5.0))
        painter.drawRect(QRectF(6.5, 12.0, 9.0, 6.0))
    elif kind == "fit":
        painter.drawLine(QPointF(4.0, 8.0), QPointF(4.0, 4.0))
        painter.drawLine(QPointF(4.0, 4.0), QPointF(8.0, 4.0))
        painter.drawLine(QPointF(14.0, 4.0), QPointF(18.0, 4.0))
        painter.drawLine(QPointF(18.0, 4.0), QPointF(18.0, 8.0))
        painter.drawLine(QPointF(18.0, 14.0), QPointF(18.0, 18.0))
        painter.drawLine(QPointF(18.0, 18.0), QPointF(14.0, 18.0))
        painter.drawLine(QPointF(8.0, 18.0), QPointF(4.0, 18.0))
        painter.drawLine(QPointF(4.0, 18.0), QPointF(4.0, 14.0))
    elif kind in {"zoom_in", "zoom_out"}:
        painter.drawEllipse(QPointF(9.5, 9.5), 5.5, 5.5)
        painter.drawLine(QPointF(13.6, 13.6), QPointF(18.5, 18.5))
        painter.drawLine(QPointF(7.0, 9.5), QPointF(12.0, 9.5))
        if kind == "zoom_in":
            painter.drawLine(QPointF(9.5, 7.0), QPointF(9.5, 12.0))
    elif kind == "pan":
        painter.drawLine(QPointF(11.0, 3.5), QPointF(11.0, 18.5))
        painter.drawLine(QPointF(3.5, 11.0), QPointF(18.5, 11.0))
        painter.drawLine(QPointF(11.0, 3.5), QPointF(8.5, 6.0))
        painter.drawLine(QPointF(11.0, 3.5), QPointF(13.5, 6.0))
        painter.drawLine(QPointF(11.0, 18.5), QPointF(8.5, 16.0))
        painter.drawLine(QPointF(11.0, 18.5), QPointF(13.5, 16.0))
        painter.drawLine(QPointF(3.5, 11.0), QPointF(6.0, 8.5))
        painter.drawLine(QPointF(3.5, 11.0), QPointF(6.0, 13.5))
        painter.drawLine(QPointF(18.5, 11.0), QPointF(16.0, 8.5))
        painter.drawLine(QPointF(18.5, 11.0), QPointF(16.0, 13.5))
    elif kind == "run":
        path = QPainterPath(QPointF(6.0, 4.0))
        path.lineTo(QPointF(18.0, 11.0))
        path.lineTo(QPointF(6.0, 18.0))
        path.closeSubpath()
        painter.setBrush(ink)
        painter.drawPath(path)
        painter.setBrush(Qt.BrushStyle.NoBrush)
    elif kind == "screw_mpr":
        painter.drawRect(QRectF(3.0, 3.0, 16.0, 16.0))
        painter.drawLine(QPointF(5.5, 16.5), QPointF(16.5, 5.5))
        painter.setBrush(ink)
        painter.drawEllipse(QPointF(5.5, 16.5), 2.0, 2.0)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawLine(QPointF(9.0, 15.0), QPointF(12.0, 18.0))
    elif kind == "layout":
        painter.drawRect(QRectF(3.0, 4.0, 16.0, 14.0))
        painter.drawLine(QPointF(11.0, 4.0), QPointF(11.0, 18.0))
        painter.drawLine(QPointF(11.0, 11.0), QPointF(19.0, 11.0))
    elif kind == "reset":
        painter.drawArc(QRectF(4.0, 4.0, 14.0, 14.0), 40 * 16, 280 * 16)
        painter.drawLine(QPointF(16.0, 4.5), QPointF(16.5, 9.0))
        painter.drawLine(QPointF(16.5, 9.0), QPointF(12.0, 8.5))
