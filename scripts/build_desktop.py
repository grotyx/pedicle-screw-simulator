#!/usr/bin/env python3
"""Build and archive a native PyInstaller desktop package."""

from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "packaging" / "PedicleScrewSimulator.spec"
WINDOWS_VERSION_TEMPLATE = ROOT / "packaging" / "windows_version_info.template.txt"
BUILD_DIR = ROOT / "build"
DIST = ROOT / "dist"
RELEASE = ROOT / "release"
VERSION = (ROOT / "VERSION").read_text(encoding="utf-8").strip()
PRODUCT = "PedicleScrewSimulator"


def _render_windows_version_file() -> Path:
    """Render the Windows version-resource template for the current VERSION.

    PyInstaller's ``version=`` argument needs a file with literal integer
    version components, so the template's ``{major}``/``{minor}``/``{patch}``
    placeholders are filled in from VERSION and the result is written under
    ``build/`` (not tracked in git).
    """
    version_parts = VERSION.split(".")
    major = version_parts[0] if len(version_parts) > 0 else "0"
    minor = version_parts[1] if len(version_parts) > 1 else "0"
    patch = version_parts[2] if len(version_parts) > 2 else "0"

    rendered = WINDOWS_VERSION_TEMPLATE.read_text(encoding="utf-8").format(
        major=major, minor=minor, patch=patch, version=VERSION
    )
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    output = BUILD_DIR / "windows_version_info.txt"
    output.write_text(rendered, encoding="utf-8")
    return output


def _copy_metadata(destination: Path) -> None:
    for relative_path in (
        "LICENSE",
        "AUTHORS.md",
        "CITATION.cff",
        "THIRD_PARTY_NOTICES.md",
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
    env = os.environ.copy()
    if sys.platform.startswith("win"):
        env["PSS_VERSION_FILE"] = str(_render_windows_version_file())
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
        env=env,
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
    _copy_metadata(source)
    archive_base = RELEASE / package_name
    archive = Path(
        shutil.make_archive(str(archive_base), "zip", DIST, PRODUCT)
    )
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
