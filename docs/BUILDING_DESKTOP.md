# Building Standalone Desktop Packages

[English](BUILDING_DESKTOP.md) | [한국어](BUILDING_DESKTOP.ko.md)

## Supported outputs

| Platform | Output | Build environment |
|---|---|---|
| macOS Apple Silicon | `.app` inside a ZIP archive | macOS arm64 |
| macOS Intel | `.app` inside a ZIP archive | macOS Intel |
| Windows x64 | `.exe` with its support folder inside a ZIP archive | Windows x64 |

PyInstaller is not a cross-compiler. Each package must be built on its target operating system.

## Local macOS build

```bash
./scripts/run_app.sh --setup-only
source venv/bin/activate
python -m pip install -r requirements-dev.txt
python scripts/build_desktop.py
```

The archive is written to `release/`. The app is unsigned and not notarized, so macOS Gatekeeper may require a right-click **Open** confirmation. Signing and notarization require an Apple Developer ID and are not performed automatically.

## Local Windows build

Run in PowerShell with Python 3.12 installed:

```powershell
py -3.12 -m venv venv
.\venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\venv\Scripts\python.exe scripts\build_desktop.py
```

The archive contains `PedicleScrewSimulator.exe` and its required `_internal` folder. Keep the complete folder together.

## GitHub Actions

The **Build desktop packages** workflow builds:

- Windows x64 on `windows-2025`
- macOS Apple Silicon on `macos-15`
- macOS Intel on `macos-15-intel`

Run it manually from the Actions tab or push a version tag such as `v0.1.0`. Download the resulting artifacts from the completed workflow run.

## Standalone feature scope

The standalone packages include DICOM loading, standard and screw-aligned MPR, 3D rendering, manual tools, planning, measurements, and threshold fallback segmentation.

TotalSegmentator and PyTorch are intentionally not bundled because they add a very large environment, external model weights, and separate task-specific licensing requirements. Use the source installation with `./scripts/run_app.sh --with-totalseg` when AI segmentation is required.

## Runtime files

Packaged logs are written to a user-writable location:

- macOS: `~/Library/Logs/PedicleScrewSimulator/app.log`
- Windows: `%LOCALAPPDATA%\PedicleScrewSimulator\logs\app.log`

The distribution includes the MIT license, author information, citation metadata, and README.
