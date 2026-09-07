# -*- mode: python ; coding: utf-8 -*-

import os
from pathlib import Path
import sys

from PyInstaller.utils.hooks import collect_all, copy_metadata


project_root = Path(SPECPATH).resolve().parent
version = (project_root / "VERSION").read_text(encoding="utf-8").strip()
windows_version_file = os.environ.get("PSS_VERSION_FILE")

datas = [
    (str(project_root / "LICENSE"), "."),
    (str(project_root / "AUTHORS.md"), "."),
    (str(project_root / "CITATION.cff"), "."),
    (str(project_root / "THIRD_PARTY_NOTICES.md"), "."),
    (str(project_root / "VERSION"), "."),
    (str(project_root / "README.md"), "."),
]
binaries = []
hiddenimports = [
    "vtkmodules.qt.QVTKRenderWindowInteractor",
    "vtkmodules.util.numpy_support",
    "scipy.ndimage",
    "torch",
    "totalsegmentator.python_api",
    "totalsegmentator.nnunet",
]

for package_name in (
    "vtkmodules",
    "SimpleITK",
    "totalsegmentator",
    "nnunetv2",
    "dynamic_network_architectures",
    "batchgenerators",
    "batchgeneratorsv2",
    "acvl_utils",
):
    package_datas, package_binaries, package_hiddenimports = collect_all(
        package_name
    )
    datas += package_datas
    binaries += package_binaries
    hiddenimports += package_hiddenimports

for distribution_name in ("TotalSegmentator", "nnunetv2", "torch"):
    datas += copy_metadata(distribution_name)

a = Analysis(
    [str(project_root / "main.py")],
    pathex=[str(project_root)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="PedicleScrewSimulator",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    version=(
        windows_version_file
        if sys.platform.startswith("win") and windows_version_file
        else None
    ),
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="PedicleScrewSimulator",
)

if sys.platform == "darwin":
    app = BUNDLE(
        coll,
        name="PedicleScrewSimulator.app",
        icon=None,
        bundle_identifier="me.sangmin.pedicle-screw-simulator",
        version=version,
        info_plist={
            "CFBundleName": "Pedicle Screw Simulator",
            "CFBundleDisplayName": "Pedicle Screw Simulator",
            "CFBundleShortVersionString": version,
            "CFBundleVersion": version,
            "NSHighResolutionCapable": True,
            "NSRequiresAquaSystemAppearance": False,
        },
    )
