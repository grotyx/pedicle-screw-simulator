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

import importlib
import logging
import multiprocessing
import os
import sys
import traceback
from logging.handlers import RotatingFileHandler
from pathlib import Path

# Get the directory containing this script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Add the project root to path for imports
sys.path.insert(0, script_dir)

def resolve_log_dir(
    *,
    frozen=None,
    platform=None,
    environ=None,
    home=None,
) -> Path:
    """Return a writable log directory for source and packaged builds."""
    is_frozen = getattr(sys, "frozen", False) if frozen is None else frozen
    current_platform = sys.platform if platform is None else platform
    current_environ = os.environ if environ is None else environ
    home_dir = Path.home() if home is None else Path(home)

    if not is_frozen:
        return Path(script_dir) / "logs"
    if current_platform == "darwin":
        return home_dir / "Library" / "Logs" / "PedicleScrewSimulator"
    if current_platform.startswith("win"):
        local_app_data = current_environ.get("LOCALAPPDATA")
        base_dir = Path(local_app_data) if local_app_data else home_dir
        return base_dir / "PedicleScrewSimulator" / "logs"
    state_home = current_environ.get("XDG_STATE_HOME")
    base_dir = Path(state_home) if state_home else home_dir / ".local" / "state"
    return base_dir / "PedicleScrewSimulator" / "logs"


def setup_logging():
    """Configure rotating file logging and an optional console handler."""
    log_dir = resolve_log_dir()
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "app.log"

    file_handler = RotatingFileHandler(
        log_path,
        maxBytes=5 * 1024 * 1024,
        backupCount=3,
        encoding="utf-8",
    )
    handlers = [file_handler]
    if sys.stderr is not None:
        handlers.append(logging.StreamHandler(sys.stderr))
    level_name = os.environ.get("PEDICLE_SCREW_LOG_LEVEL", "INFO").upper()
    log_level = getattr(logging, level_name, logging.INFO)

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
        force=True,
    )
    logging.getLogger().info("=" * 60)
    logging.getLogger().info("Session started — Log file: %s", log_path)
    return log_path


def run_dependency_self_check() -> None:
    """Raise immediately when a required packaged AI module is missing."""
    totalseg_config = importlib.import_module("totalsegmentator.config")
    totalseg_config.setup_nnunet()
    for module_name in (
        "torch",
        "nnunetv2",
        "totalsegmentator.python_api",
        "totalsegmentator.nnunet",
    ):
        importlib.import_module(module_name)


def run_self_check_cli() -> int:
    """
    Run the dependency self-check for the ``--self-check`` CLI entry point.

    A frozen Windows build runs this via ``Start-Process -Wait`` against the
    windowed (console=False) exe, so an unhandled exception here previously
    surfaced only as an undismissable modal traceback dialog, hanging the CI
    job to its timeout. Instead, catch any failure, write the traceback to a
    file under the log directory (falling back to stderr-only if the log
    directory itself can't be created or written), print it to stderr, and
    return a non-zero exit code so the caller can detect and report failure
    without a GUI.
    """
    try:
        run_dependency_self_check()
    except Exception:
        tb_text = traceback.format_exc()
        try:
            log_dir = resolve_log_dir()
            log_dir.mkdir(parents=True, exist_ok=True)
            (log_dir / "self_check_error.log").write_text(tb_text, encoding="utf-8")
        except OSError:
            pass
        print(tb_text, file=sys.stderr)
        return 1
    return 0


def run_application() -> None:
    """Import and launch the GUI after frozen-process setup is complete."""
    from src.core.totalseg_integration import SegmentationWorkspace

    removed = SegmentationWorkspace.purge_stale(older_than_seconds=3600)
    logging.getLogger(__name__).info(
        "Purged %d stale segmentation directories", removed
    )

    from src.ui.main_window import main as launch_main_window

    launch_main_window()


if __name__ == "__main__":
    multiprocessing.freeze_support()
    if "--self-check" in sys.argv:
        raise SystemExit(run_self_check_cli())
    setup_logging()
    run_application()
