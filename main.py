#!/usr/bin/env python3
"""
Pedicle Screw Fixation Simulator
================================

A VTK-based medical imaging application for pedicle screw placement planning.

Usage:
    python main.py

Requirements:
    - Python 3.12+
    - VTK 9.5+
    - PyQt6
    - SimpleITK
    - pydicom
    - numpy, scipy
"""

import sys
import os
import logging

# Get the directory containing this script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Add the project root to path for imports
sys.path.insert(0, script_dir)

from src.ui.main_window import main


def setup_logging():
    """Configure logging to file (append) + console."""
    log_dir = os.path.join(script_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "app.log")

    file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    stream_handler = logging.StreamHandler(sys.stderr)

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[file_handler, stream_handler],
        force=True,
    )
    logging.getLogger().info("=" * 60)
    logging.getLogger().info("Session started — Log file: %s", log_path)


if __name__ == "__main__":
    setup_logging()
    main()
