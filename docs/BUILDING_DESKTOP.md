# Building Standalone Desktop Packages

[English](BUILDING_DESKTOP.md) | [한국어](BUILDING_DESKTOP.ko.md)

## Supported outputs

| Platform | Output | Build environment |
|---|---|---|
| macOS Apple Silicon | `.app` inside a ZIP archive | macOS arm64 |
| Windows x64 | Installer (`…-Setup.exe` + `.bin` slices) and a portable ZIP | Windows x64 |

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

Run in PowerShell with Python 3.12 and [Inno Setup 6](https://jrsoftware.org/isinfo.php) installed
(`winget install JRSoftware.InnoSetup --scope user`):

```powershell
py -3.12 -m venv venv
.\venv\Scripts\python.exe -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128
.\venv\Scripts\python.exe -m pip install -r requirements-desktop.txt
.\venv\Scripts\python.exe scripts\build_desktop.py
```

`release\` then holds two outputs:

- **Installer**: `RadWright-Screw-<version>-Windows-x64-Setup.exe` with one or
  more `…-Setup-N.bin` files beside it (the CUDA runtime makes the payload several GB,
  so Inno Setup splits it). Keep them in one folder and run the `.exe`. It installs
  per user into `%LOCALAPPDATA%\Programs\RadWright Screw` without an
  administrator prompt, adds Start menu entries (desktop shortcut optional) and an
  entry under Settings > Apps for uninstalling. Running a newer installer upgrades
  in place. Silent install: `…-Setup.exe /VERYSILENT`.
- **Portable ZIP**: `PedicleScrewSimulator.exe` and its required `_internal` folder.
  Keep the complete folder together.

Pass `--no-installer` to build only the ZIP. The installer is not code-signed, so
Windows SmartScreen may warn on first run (**More info** > **Run anyway**).

## RadWright Screw artwork

Brand assets live in `packaging/icons/`. The hand-written
`radwright-screw.svg` is the 1024-unit master. The two self-contained SVG
wordmarks use system sans fonts: `radwright-screw-wordmark.svg` for dark
backgrounds and `radwright-screw-wordmark-light.svg` for white backgrounds.
When editing the anatomy, keep the embedded icon geometry in both wordmarks
in sync with the master.

Rebuild the nine PNG sizes (16, 24, 32, 48, 64, 128, 256, 512, 1024),
seven-frame Windows ICO, macOS ICNS, and contact sheet:

```bash
python -m pip install Pillow  # artwork tooling only
python scripts/render_icons.py
```

The renderer uses the application's existing PyQt6 SVG support and works
offscreen on Linux. No rasterizer dependency is added to the application.
Run from any working directory; use `--output-dir <directory>` to export
elsewhere. Review `packaging/icons/preview.png` after changes: it shows
1024, 256, 64, 32, and 16 px icons on dark and white backgrounds, enlarged
small-size samples, and both wordmarks. PNG corners remain transparent.
System font choice can change wordmark appearance between machines.

## GitHub Actions

The **Build desktop packages** workflow builds:

- Windows x64 on `windows-2025`
- macOS Apple Silicon on `macos-15`

Run it manually from the Actions tab or push a version tag such as `v0.2.7`. Download the resulting artifacts from the completed workflow run.

## Standalone feature scope

The standalone packages include DICOM loading, standard and screw-aligned MPR, 3D rendering, manual tools, planning, measurements, TotalSegmentator 2.12.0, PyTorch, nnU-Net, and threshold fallback segmentation.

The Windows build bundles the two TotalSegmentator models that Spine Only uses (vertebrae part `Dataset292` and the 6 mm crop model `Dataset298`, about 400 MB) under `_internal/totalseg_weights`; `scripts/build_desktop.py` downloads them at build time, and Spine Only + Accurate runs read them from there and download nothing. The Total task, Fast quality and the macOS build download the openly available weights to `~/.totalsegmentator/nnunet/results` on first use and reuse them later. The Windows build bundles the PyTorch CUDA 12.8 runtime and automatically falls back to CPU when no supported NVIDIA GPU is available.

The bundled executable can be checked without opening the GUI:

```bash
dist/PedicleScrewSimulator.app/Contents/MacOS/PedicleScrewSimulator --self-check
```

```powershell
.\dist\PedicleScrewSimulator\PedicleScrewSimulator.exe --self-check
```

## Optional pedicle subregion model

The application can optionally refine pedicle detection with a locally installed nnU-Net model that segments a vertebra into pedicle/corpus/lamina/spinous/transverse/articular subregions ([MICN-Lab/Spine_Subregions](https://github.com/MICN-Lab/Spine_Subregions); Da Mutten et al., *J Imaging Inform Med* 2026). This is a source-install feature only: unlike TotalSegmentator, this stage always shells out to `nnUNetv2_predict`, and a frozen build has no Python interpreter to fall back to, so the standalone macOS and Windows packages refuse it outright.

To use it from a source install:

1. Download the trained weights from the [Spine_Subregions Releases page](https://github.com/MICN-Lab/Spine_Subregions/releases).
2. Arrange them in the standard nnU-Net v2 results layout, e.g.:
   ```
   <root>/Dataset501_SpineSubregions/nnUNetTrainer__nnUNetPlans__3d_fullres/
       dataset.json
       fold_0/checkpoint_final.pth
   ```
3. Install `nnunetv2` in the same environment running the app (`nnUNetv2_predict` must resolve next to the interpreter's `Scripts`/`bin` folder).
4. Point the app at the `.../nnUNetTrainer__nnUNetPlans__3d_fullres` folder with the **Model directory** field under **Advanced options** on the Segment page, or the `PSS_SUBREGION_MODEL_DIR` environment variable, then enable **Use pedicle subregion model**.

**Current limitation:** the Spine_Subregions release asset currently published upstream uses nnU-Net **v1**-style folder naming (`nnUNetTrainerV2__nnUNetPlansv2.1`), not the nnU-Net **v2** results layout shown above that this app's stage-2 runner expects. Until the weights are re-exported or converted for nnU-Net v2, pointing the **Model directory** field at that release's folder will not be recognized as a valid model, and the segmentation status line will read "pedicle model unavailable" rather than "pedicle model used".

Labels are read from the model's own `dataset.json` at runtime rather than hard-coded, so any nnU-Net checkpoint using recognizable subregion names resolves correctly. Memory needs are similar to TotalSegmentator's `3d_fullres` configuration; a GPU is recommended. A stage-2 failure never blocks the underlying TotalSegmentator result. License: see [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md).

## Publishing an update

A Windows build also writes `release\update\`: one ZIP per layer and
`components.json` (the same manifest is shipped inside the app folder).

| Layer | Contents | Changes when |
|:--|:--|:--|
| `cuda` | `_internal\torch\lib` (CUDA, cuDNN DLLs) | torch is upgraded |
| `runtime` | every other package under `_internal` | a dependency changes |
| `app` | the executable (our code) and loose files | every release |

Each layer is named by a hash of its files, so publishing skips layers the server
already has and the app downloads only layers whose hash changed. Upload with:

```powershell
.\venv\Scripts\python.exe -m pip install boto3
.\venv\Scripts\python.exe scripts\publish_update.py release\update --downloads (Get-ChildItem release\*.exe, release\*.bin, release\*.zip) --dry-run
.\venv\Scripts\python.exe scripts\publish_update.py release\update --downloads (Get-ChildItem release\*.exe, release\*.bin, release\*.zip)
```

It writes to the `spinemetrics-dl` R2 bucket under `radwright-screw/`, served at
`https://dl.spinemetrics.sangmin.me/radwright-screw/`. `--downloads` puts the installer
`.exe` + `.bin` and the ZIPs under `downloads/<version>/` (everything is hosted on R2;
GitHub limits files to 2 GB). Credentials come from
`R2_ACCESS_KEY_ID`, `R2_SECRET_ACCESS_KEY`, `R2_ACCOUNT_ID` or
`%USERPROFILE%\.spinemetrics\r2.env` (never commit it). `latest/components.json` is
uploaded last, so clients never see a release with missing layers. To roll back, copy
`versions/<old>/components.json` over `latest/components.json`. Set `PSS_UPDATE_URL` to
test against another server, or `PSS_NO_UPDATE=1` to turn the check off.

## Package size

Most of the package is the GPU segmentation runtime, not the application.
Measured on the Linux development environment (torch 2.14 + CUDA 13,
2026-10-30):

| Part | Size |
|:--|--:|
| CUDA libraries (`nvidia/*`: cu13, cuDNN, NCCL, cuSPARSELt, NVSHMEM) | 3.2 GB |
| `torch` | 1.2 GB |
| `triton` (Linux only, **excluded from builds**) | 0.9 GB |
| VTK + SimpleITK + PyQt6 + SciPy + NumPy | about 1.25 GB |
| Application code | a few MB |
| TotalSegmentator Spine Only weights (bundled on Windows) | 0.4 GB |
| Other TotalSegmentator weights (downloaded on first use) | 0.16–1.2 GB |

- The package targets GPU inference only. A CPU-only build would drop the
  CUDA libraries but make segmentation take minutes, so it is not offered.
- NCCL, cuSPARSELt and NVSHMEM look like multi-GPU extras but
  `libtorch_cuda` links them directly; removing them stops torch loading.
- Triton only backs `torch.compile`; TotalSegmentator GPU inference was
  checked to run without it. Windows and macOS torch wheels never include it.
- Other weights are fetched per task on first run (`~/.totalsegmentator`), so
  only the tasks a user runs take space. The full `total` task needs all five
  parts (about 1.2 GB); the fast mode needs one 159 MB model. TotalSegmentator
  2.12 would fetch all five parts even for Spine Only, so the app skips the
  four it does not compute when the bundled models are present.
- VTK could shrink to roughly 100–200 MB if the source imported individual
  `vtkmodules.*` instead of `import vtk` (7 files). Not done yet.
- Replacing PyTorch with ONNX Runtime would save the most, but means
  re-implementing the nnU-Net pre/post-processing; deliberately not pursued.

## Runtime files

Packaged logs are written to a user-writable location:

- macOS: `~/Library/Logs/PedicleScrewSimulator/app.log`
- Windows: `%LOCALAPPDATA%\PedicleScrewSimulator\logs\app.log`

The distribution includes the MIT license, author information, citation metadata, README, and `THIRD_PARTY_NOTICES.md`. TotalSegmentator's default `total` task is openly available under Apache 2.0. License-restricted TotalSegmentator tasks are not exposed by the default application workflow.
