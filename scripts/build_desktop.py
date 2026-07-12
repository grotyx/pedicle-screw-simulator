#!/usr/bin/env python3
"""Build and archive a native PyInstaller desktop package."""

from __future__ import annotations

import argparse
import platform
from pathlib import Path
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "packaging" / "PedicleScrewSimulator.spec"
DIST = ROOT / "dist"
RELEASE = ROOT / "release"
VERSION = (ROOT / "VERSION").read_text(encoding="utf-8").strip()
PRODUCT = "PedicleScrewSimulator"


def _copy_metadata(destination: Path) -> None:
    for relative_path in (
        "LICENSE",
        "AUTHORS.md",
        "CITATION.cff",
        "README.md",
        "README.ko.md",
        "docs/USER_GUIDE.md",
        "docs/USER_GUIDE.ko.md",
        "docs/BUILDING_DESKTOP.md",
        "docs/BUILDING_DESKTOP.ko.md",
    ):
        source = ROOT / relative_path
        if not source.exists():
            continue
        target = destination / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def _build() -> None:
    subprocess.run(
        [
            sys.executable,
            "-m",
            "PyInstaller",
            "--noconfirm",
            "--clean",
            str(SPEC),
        ],
        cwd=ROOT,
        check=True,
    )


def _archive_macos() -> Path:
    architecture = platform.machine().lower() or "unknown"
    label = "arm64" if architecture in {"arm64", "aarch64"} else "x64"
    app_source = DIST / f"{PRODUCT}.app"
    if not app_source.exists():
        raise FileNotFoundError(f"macOS app bundle not found: {app_source}")

    package_name = f"{PRODUCT}-{VERSION}-macOS-{label}"
    staging = RELEASE / package_name
    subprocess.run(
        ["ditto", str(app_source), str(staging / f"{PRODUCT}.app")],
        check=True,
    )
    _copy_metadata(staging)
    archive = RELEASE / f"{package_name}.zip"
    subprocess.run(
        [
            "ditto",
            "-c",
            "-k",
            "--sequesterRsrc",
            "--keepParent",
            str(staging),
            str(archive),
        ],
        check=True,
    )
    shutil.rmtree(staging)
    return archive


def _archive_windows() -> Path:
    source = DIST / PRODUCT
    if not source.exists():
        raise FileNotFoundError(f"Windows application folder not found: {source}")

    package_name = f"{PRODUCT}-{VERSION}-Windows-x64"
    staging = RELEASE / package_name
    shutil.copytree(source, staging)
    _copy_metadata(staging)
    archive_base = RELEASE / package_name
    archive = Path(
        shutil.make_archive(str(archive_base), "zip", RELEASE, package_name)
    )
    shutil.rmtree(staging)
    return archive


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--skip-build",
        action="store_true",
        help="Archive an existing dist output without rerunning PyInstaller.",
    )
    args = parser.parse_args()

    if RELEASE.exists():
        shutil.rmtree(RELEASE)
    RELEASE.mkdir(parents=True)

    if not args.skip_build:
        _build()
    if sys.platform == "darwin":
        archive = _archive_macos()
    elif sys.platform.startswith("win"):
        archive = _archive_windows()
    else:
        raise RuntimeError("Desktop packaging currently supports macOS and Windows.")

    print(f"Desktop package created: {archive}")


if __name__ == "__main__":
    main()
