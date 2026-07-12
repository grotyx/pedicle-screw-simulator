# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path
import sys

from PyInstaller.utils.hooks import collect_all


project_root = Path(SPECPATH).resolve().parent
windows_version_file = project_root / "packaging" / "windows_version_info.txt"

datas = [
    (str(project_root / "LICENSE"), "."),
    (str(project_root / "AUTHORS.md"), "."),
    (str(project_root / "CITATION.cff"), "."),
    (str(project_root / "VERSION"), "."),
    (str(project_root / "README.md"), "."),
]
binaries = []
hiddenimports = [
    "vtkmodules.qt.QVTKRenderWindowInteractor",
    "vtkmodules.util.numpy_support",
    "scipy.ndimage",
]

for package_name in ("vtkmodules", "SimpleITK"):
    package_datas, package_binaries, package_hiddenimports = collect_all(
        package_name
    )
    datas += package_datas
    binaries += package_binaries
    hiddenimports += package_hiddenimports

a = Analysis(
    [str(project_root / "main.py")],
    pathex=[str(project_root)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["totalsegmentator", "torch"],
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
        str(windows_version_file)
        if sys.platform.startswith("win")
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
        version="0.1.0",
        info_plist={
            "CFBundleName": "Pedicle Screw Simulator",
            "CFBundleDisplayName": "Pedicle Screw Simulator",
            "CFBundleShortVersionString": "0.1.0",
            "CFBundleVersion": "0.1.0",
            "NSHighResolutionCapable": True,
            "NSRequiresAquaSystemAppearance": False,
        },
    )
