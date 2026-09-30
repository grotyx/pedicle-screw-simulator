"""Numeric input widgets with direct replacement typing."""

import math

from PyQt6.QtCore import QEvent
from PyQt6.QtWidgets import QDoubleSpinBox


class ReplaceOnTypeDoubleSpinBox(QDoubleSpinBox):
    """Replace the current value when the user starts typing a number."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._replace_on_next_number = True
        self.lineEdit().installEventFilter(self)

    def eventFilter(self, watched, event):
        if watched is self.lineEdit():
            if event.type() in {
                QEvent.Type.FocusIn,
                QEvent.Type.MouseButtonPress,
            }:
                self._replace_on_next_number = True
            elif event.type() == QEvent.Type.KeyPress:
                text = event.text()
                if self._replace_on_next_number and (
                    text.isdigit() or text in {".", ",", "-"}
                ):
                    self.lineEdit().selectAll()
                    self._replace_on_next_number = False
        return super().eventFilter(watched, event)

    def keyPressEvent(self, event):
        text = event.text()
        if self._replace_on_next_number and (
            text.isdigit() or text in {".", ",", "-"}
        ):
            self.lineEdit().selectAll()
            self._replace_on_next_number = False
        super().keyPressEvent(event)


_DECIMAL_POINTS = {".", ","}


class DiameterSpinBox(ReplaceOnTypeDoubleSpinBox):
    """Take typed sizes as written (``7.5``, ``5,5``), and digit-only ``65`` as 6.5 mm."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._diameter_digits = ""

    def eventFilter(self, watched, event):
        if watched is self.lineEdit():
            if event.type() in {
                QEvent.Type.FocusIn,
                QEvent.Type.MouseButtonPress,
            }:
                self._diameter_digits = ""
            elif event.type() == QEvent.Type.KeyPress:
                text = event.text()
                if text.isdigit() or text in _DECIMAL_POINTS:
                    self._apply_diameter_key(text)
                    return True
                self._diameter_digits = ""
        return super().eventFilter(watched, event)

    def keyPressEvent(self, event):
        text = event.text()
        if text.isdigit() or text in _DECIMAL_POINTS:
            self._apply_diameter_key(text)
            return
        self._diameter_digits = ""
        super().keyPressEvent(event)

    def _apply_diameter_key(self, text: str) -> None:
        if self._replace_on_next_number:
            self._diameter_digits = ""
            self._replace_on_next_number = False
        if text in _DECIMAL_POINTS:
            # The value lands with the digit after the point, so the key
            # itself only marks where the decimals start.
            if "." not in self._diameter_digits:
                self._diameter_digits = (self._diameter_digits or "0") + "."
            return
        whole, point, decimals = self._diameter_digits.partition(".")
        if (point and decimals) or (not point and len(whole) >= 2):
            self._diameter_digits = ""
            point = ""
        self._diameter_digits += text
        if point:
            value = float(self._diameter_digits)
        else:
            raw_value = int(self._diameter_digits)
            value = (
                float(raw_value)
                if len(self._diameter_digits) == 1
                else raw_value / 10.0
            )
        normalized = math.floor(value * 2.0 + 0.5) / 2.0
        self.setValue(min(self.maximum(), max(self.minimum(), normalized)))
        self.lineEdit().selectAll()
