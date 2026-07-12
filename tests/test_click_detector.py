"""Tests for platform-independent double-click recognition."""

from src.ui.click_detector import DoubleClickDetector


def test_second_nearby_click_inside_interval_is_double_click():
    detector = DoubleClickDetector(max_interval=0.35, max_distance=6.0)

    assert detector.register(100, 100, timestamp=1.0) is False
    assert detector.register(103, 104, timestamp=1.3) is True


def test_click_outside_time_interval_starts_new_pair():
    detector = DoubleClickDetector(max_interval=0.35, max_distance=6.0)

    assert detector.register(100, 100, timestamp=1.0) is False
    assert detector.register(100, 100, timestamp=1.5) is False


def test_click_outside_pixel_tolerance_starts_new_pair():
    detector = DoubleClickDetector(max_interval=0.35, max_distance=6.0)

    assert detector.register(100, 100, timestamp=1.0) is False
    assert detector.register(120, 100, timestamp=1.2) is False


def test_recognized_double_click_resets_detector():
    detector = DoubleClickDetector(max_interval=0.35, max_distance=6.0)

    detector.register(100, 100, timestamp=1.0)
    assert detector.register(100, 100, timestamp=1.2) is True
    assert detector.register(100, 100, timestamp=1.3) is False
