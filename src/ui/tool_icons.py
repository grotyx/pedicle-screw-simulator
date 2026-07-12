"""Small vector-style icons for the unified planning tool palette."""

from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtGui import QColor, QIcon, QPainter, QPainterPath, QPen, QPixmap


def create_tool_icon(kind: str, size: int = 22) -> QIcon:
    """Create a crisp monochrome icon without external image assets."""
    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    color = QColor("#71D2DE")
    pen = QPen(color, 1.8)
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
    painter.setPen(pen)

    if kind == "select":
        path = QPainterPath(QPointF(5.0, 3.0))
        path.lineTo(QPointF(17.0, 12.0))
        path.lineTo(QPointF(11.5, 13.0))
        path.lineTo(QPointF(14.5, 19.0))
        path.lineTo(QPointF(11.5, 20.0))
        path.lineTo(QPointF(8.7, 14.0))
        path.lineTo(QPointF(5.0, 18.0))
        path.closeSubpath()
        painter.setBrush(color)
        painter.drawPath(path)
    elif kind == "screw":
        painter.drawLine(QPointF(5.0, 17.0), QPointF(17.0, 5.0))
        painter.setBrush(color)
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
    else:
        painter.end()
        raise ValueError(f"Unknown tool icon: {kind}")

    painter.end()
    return QIcon(pixmap)
