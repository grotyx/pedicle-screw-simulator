"""Non-modal Virtual X-ray window: AP and lateral projections with screws."""

from typing import Dict

import numpy as np
from PyQt6.QtCore import QLineF, QPointF, Qt, QTimer
from PyQt6.QtGui import QColor, QFont, QImage, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QRadioButton,
    QVBoxLayout,
)

from src.controllers.tool_controller import screw_display_color
from src.core.projection import VIEW_LETTERS, VIEWS, project_point, project_volume

# shortcut: fixed display size, the window does not rescale on resize; add a
# resize-aware widget if surgeons need larger images.
DISPLAY_LONG_SIDE_PX = 480
HIGHLIGHT_RGB = (255, 220, 0)
VIEW_TITLES = {"ap": "AP", "lateral": "Lateral"}


class VirtualXrayDialog(QDialog):
    """Side-by-side AP and lateral projections of the loaded CT.

    Projections are cached per volume and mode; screw edits only repaint the
    overlay.  The window listens to the screw plan table, which every screw
    add / edit / remove / selection already goes through.
    """

    def __init__(self, volume_manager, screw_tool, screw_table, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Virtual X-ray")
        self.setModal(False)
        self._vm = volume_manager
        self._screw_tool = screw_tool
        self._table = screw_table
        self._cache_image = None
        self._cache: Dict[tuple, object] = {}
        self._overlay_counts = {view: 0 for view in VIEWS}
        # One table edit emits a signal per cell; coalesce them into one repaint.
        self._pending = QTimer(self)
        self._pending.setSingleShot(True)
        self._pending.setInterval(0)
        self._pending.timeout.connect(self.refresh)

        self.raysum_button = QRadioButton("Ray-sum")
        self.raysum_button.setToolTip("Mean attenuation along the beam, like a radiograph")
        self.raysum_button.setChecked(True)
        self.mip_button = QRadioButton("MIP")
        self.mip_button.setToolTip("Maximum intensity along the beam")
        self._mode_group = QButtonGroup(self)
        self._mode_group.addButton(self.raysum_button)
        self._mode_group.addButton(self.mip_button)
        self.screws_checkbox = QCheckBox("Screws")
        self.screws_checkbox.setChecked(True)
        self.refresh_button = QPushButton("Refresh")

        controls = QHBoxLayout()
        for widget in (self.raysum_button, self.mip_button, self.screws_checkbox):
            controls.addWidget(widget)
        controls.addStretch(1)
        controls.addWidget(self.refresh_button)

        self.ap_label = QLabel()
        self.lateral_label = QLabel()
        self._labels = {"ap": self.ap_label, "lateral": self.lateral_label}
        images = QGridLayout()
        for column, view in enumerate(VIEWS):
            images.addWidget(QLabel(VIEW_TITLES[view]), 0, column)
            self._labels[view].setAlignment(Qt.AlignmentFlag.AlignCenter)
            images.addWidget(self._labels[view], 1, column)

        self.status_label = QLabel()
        self.status_label.setWordWrap(True)
        layout = QVBoxLayout(self)
        layout.addLayout(controls)
        layout.addLayout(images)
        layout.addWidget(self.status_label)

        self.mip_button.toggled.connect(self.refresh)
        self.screws_checkbox.toggled.connect(self.refresh)
        self.refresh_button.clicked.connect(lambda: self.refresh(force=True))
        model = screw_table.model()
        for signal in (
            model.rowsInserted,
            model.rowsRemoved,
            model.dataChanged,
            model.modelReset,
        ):
            signal.connect(self._on_screws_changed)
        screw_table.currentRowChanged.connect(self._on_screws_changed)
        add_observer = getattr(volume_manager, "add_observer", None)
        if add_observer is not None:
            add_observer("virtual_xray", self._on_volume_event)

    # -- Queries (also used by tests) --------------------------------------

    def mode(self) -> str:
        return "mip" if self.mip_button.isChecked() else "raysum"

    def overlay_counts(self) -> Dict[str, int]:
        return dict(self._overlay_counts)

    def highlighted_screw(self) -> int:
        return self._table.currentRow()

    # -- Rendering ---------------------------------------------------------

    def showEvent(self, event):
        super().showEvent(event)
        self.refresh()

    def _on_screws_changed(self, *_args):
        if self.isVisible():
            self._pending.start()

    def _on_volume_event(self, event: str, **_kwargs):
        # Slice and crosshair moves notify too; only a new volume matters here.
        if event == "volume_loaded":
            if not self.isVisible():
                # Do not hold the previous study's CT while nobody looks.
                self._cache_image = None
                self._cache.clear()
            self._on_screws_changed()

    def _projection(self, image, view: str, force: bool):
        if image is not self._cache_image or force:
            self._cache_image = image
            self._cache.clear()
        key = (view, self.mode())
        if key not in self._cache:
            import SimpleITK as sitk

            self._cache[key] = project_volume(
                sitk.GetArrayViewFromImage(image),
                image.GetSpacing(),
                image.GetOrigin(),
                image.GetDirection(),
                view,
                self.mode(),
            )
        return self._cache[key]

    def refresh(self, *_args, force: bool = False) -> None:
        """Re-render both views; projections come from cache when possible."""
        image = self._vm.get_sitk_image()
        self._overlay_counts = {view: 0 for view in VIEWS}
        if image is None:
            for label in self._labels.values():
                label.clear()
            self.status_label.setText("Load a CT to see the virtual X-ray.")
            return

        screws = self._screw_tool.get_screws() if self.screws_checkbox.isChecked() else []
        selected = self._table.currentRow()
        notes = []
        for view in VIEWS:
            projection = self._projection(image, view, force)
            force = False
            self._labels[view].setPixmap(self._render(projection, screws, selected))
            self._overlay_counts[view] = len(screws)
            if projection.oblique:
                notes.append(
                    f"{VIEW_TITLES[view]}: volume axes are {projection.oblique_deg:.1f}° "
                    "off the patient axes; orientation labels are approximate."
                )
        notes.append(
            "Parallel projection of the CT, not a calibrated fluoroscopy image. "
            "Research use only."
        )
        self.status_label.setText(" ".join(notes))

    def _render(self, projection, screws, selected: int) -> QPixmap:
        scale = DISPLAY_LONG_SIDE_PX / max(projection.width_mm, projection.height_mm)
        gray = np.ascontiguousarray((projection.image * 255.0).astype(np.uint8))
        rows, cols = gray.shape
        qimage = QImage(gray.data, cols, rows, cols, QImage.Format.Format_Grayscale8)
        width = max(1, round(projection.width_mm * scale))
        height = max(1, round(projection.height_mm * scale))
        # Scaling to the mm extent honours anisotropic spacing.
        pixmap = QPixmap.fromImage(qimage.copy()).scaled(
            width,
            height,
            Qt.AspectRatioMode.IgnoreAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )

        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        def to_display(point) -> QPointF:
            col, row = project_point(projection, point)
            return QPointF(
                (col + 0.5) * projection.col_spacing * scale,
                (row + 0.5) * projection.row_spacing * scale,
            )

        # Selected screw last so its highlight sits on top.
        order = [i for i in range(len(screws)) if i != selected]
        if 0 <= selected < len(screws):
            order.append(selected)
        for index in order:
            screw = screws[index]
            line = QLineF(to_display(screw.entry_point), to_display(screw.target_point))
            width_px = max(1.0, screw.diameter * scale)
            if index == selected:
                painter.setPen(self._pen(HIGHLIGHT_RGB, width_px + 4.0))
                painter.drawLine(line)
            rgb = tuple(round(v * 255) for v in screw_display_color(screw))
            painter.setPen(self._pen(rgb, width_px))
            painter.drawLine(line)

        self._draw_letters(painter, VIEW_LETTERS[projection.view], width, height)
        painter.end()
        return pixmap

    @staticmethod
    def _pen(rgb, width: float) -> QPen:
        pen = QPen(QColor(*rgb))
        pen.setWidthF(width)
        pen.setCapStyle(Qt.PenCapStyle.FlatCap)
        return pen

    @staticmethod
    def _draw_letters(painter: QPainter, letters, width: int, height: int) -> None:
        font = QFont()
        font.setPixelSize(14)
        font.setBold(True)
        painter.setFont(font)
        painter.setPen(QColor(255, 255, 255))
        flags = {
            "top": Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignHCenter,
            "bottom": Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignHCenter,
            "left": Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            "right": Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
        }
        for side, text in letters.items():
            painter.drawText(6, 6, width - 12, height - 12, int(flags[side]), text)
