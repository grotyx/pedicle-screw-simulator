# Building Standalone Desktop Packages

[English](BUILDING_DESKTOP.md) | [한국어](BUILDING_DESKTOP.ko.md)

## Supported outputs

| Platform | Output | Build environment |
|---|---|---|
| macOS Apple Silicon | `.app` inside a ZIP archive | macOS arm64 |
| Windows x64 | `.exe` with its support folder inside a ZIP archive | Windows x64 |

PyInstaller is not a cross-compiler. Each package must be built on its target operating system.

## Local macOS build

```bash
./scripts/run_app.sh --setup-only
source venv/bin/activate
python -m pip install -r requirements-desktop.txt
python scripts/build_desktop.py
```

The archive is written to `release/`. The app is unsigned and not notarized, so macOS Gatekeeper may require a right-click **Open** confirmation. Signing and notarization require an Apple Developer ID and are not performed automatically.

## Local Windows build

Run in PowerShell with Python 3.12 installed:

```powershell
py -3.12 -m venv venv
.\venv\Scripts\python.exe -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128
.\venv\Scripts\python.exe -m pip install -r requirements-desktop.txt
.\venv\Scripts\python.exe scripts\build_desktop.py
```

The archive contains `PedicleScrewSimulator.exe` and its required `_internal` folder. Keep the complete folder together.

## GitHub Actions

The **Build desktop packages** workflow builds:

- Windows x64 on `windows-2025`
- macOS Apple Silicon on `macos-15`

Run it manually from the Actions tab or push a version tag such as `v0.1.0`. Download the resulting artifacts from the completed workflow run.

## Standalone feature scope

The standalone packages include DICOM loading, standard and screw-aligned MPR, 3D rendering, manual tools, planning, measurements, TotalSegmentator 2.12.0, PyTorch, nnU-Net, and threshold fallback segmentation.

Model weights are not embedded in the ZIP. On the first automatic segmentation, TotalSegmentator downloads the openly available `total` task weights to `~/.totalsegmentator/nnunet/results` and reuses them on later runs. The first run therefore requires internet access, additional disk space, and time for the download. The Windows build bundles the PyTorch CUDA 12.8 runtime and automatically falls back to CPU when no supported NVIDIA GPU is available.

The bundled executable can be checked without opening the GUI:

```bash
dist/PedicleScrewSimulator.app/Contents/MacOS/PedicleScrewSimulator --self-check
```

```powershell
.\dist\PedicleScrewSimulator\PedicleScrewSimulator.exe --self-check
```

## Runtime files

Packaged logs are written to a user-writable location:

- macOS: `~/Library/Logs/PedicleScrewSimulator/app.log`
- Windows: `%LOCALAPPDATA%\PedicleScrewSimulator\logs\app.log`

The distribution includes the MIT license, author information, citation metadata, README, and `THIRD_PARTY_NOTICES.md`. TotalSegmentator's default `total` task is openly available under Apache 2.0. License-restricted TotalSegmentator tasks are not exposed by the default application workflow.
