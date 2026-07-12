# Pedicle Screw Simulator

[English](README.md) | [한국어](README.ko.md)

Research desktop software for DICOM CT visualization, vertebral segmentation, and interactive pedicle screw planning with synchronized MPR and 3D review.

**Version:** 0.1.0

**Primary tested environment:** macOS, Python 3.12

**Creator:** [Sang-Min Park, MD, Ph.D.](https://sangmin.me)

**Affiliation:** Spine Center and Department of Orthopaedic Surgery, Seoul National University Bundang Hospital; Seoul National University College of Medicine

**Contact:** [psmini@snu.ac.kr](mailto:psmini@snu.ac.kr)

> **Research and education use only.** This software is not a certified medical device and must not be used as the sole basis for diagnosis, surgery, navigation, or patient care. Every segmentation, screw proposal, dimension, breach grade, and warning requires independent review by a qualified clinician.

## Highlights

- Multi-series DICOM CT loading
- Synchronized axial, sagittal, and coronal MPR
- CPU-based VTK volume rendering and selectable vertebral meshes
- Optional GPU-first TotalSegmentator integration with CPU retry
- Multi-level vertebra selection and automatic screw proposals
- Standard and screw-aligned oblique MPR review
- Direct entry, tip, and whole-screw editing in MPR and 3D
- Manual screw placement, distance measurement, and angle measurement
- JSON plan save/load and supported CSV/STL export
- Three UI themes, MPR pan/zoom, and 3D navigation controls

## Quick Start

### macOS / Linux

```bash
git clone https://github.com/grotyx/pedicle-screw-simulator.git
cd pedicle-screw-simulator
./scripts/run_app.sh
```

The launcher creates a local `venv`, installs required packages, and starts the application.

### Optional TotalSegmentator

```bash
./scripts/run_app.sh --with-totalseg
```

The first TotalSegmentator run may download model weights and can require substantial RAM or GPU memory.

### Direct Launch

```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install -r requirements.txt
python main.py
```

### Validation

```bash
./scripts/run_app.sh --check
./scripts/run_app.sh --test
```

The current automated suite contains **465 tests**.

For exact reproduction of the validated macOS/Python 3.12 environment:

```bash
python -m pip install -r requirements-lock.txt
```

`requirements.txt` provides compatible minimum versions. `requirements-lock.txt` records the exact runtime and test environment used for v0.1.0 verification. Optional TotalSegmentator packages and model weights are intentionally excluded.

## Standalone Desktop Packages

The GitHub **Build desktop packages** workflow creates separate archives for:

- macOS Apple Silicon (`.app`)
- macOS Intel (`.app`)
- Windows x64 (`.exe` with its support folder)

See [Desktop Build Guide](docs/BUILDING_DESKTOP.md). The macOS package is not Apple-notarized yet, so Gatekeeper may require right-clicking the app and selecting **Open**.

Standalone packages intentionally exclude TotalSegmentator and PyTorch. Use the source installation with `--with-totalseg` when AI segmentation is required.

## Typical Workflow

1. Open a DICOM CT folder.
2. Verify the study in standard MPR and 3D.
3. Run automatic segmentation.
4. Select the vertebral levels to display and plan.
5. Select **Plan Screws** to generate editable proposals.
6. Select a screw and enter **Screw MPR** for trajectory-aligned review.
7. Edit the entry, tip, shaft, diameter, or length as needed.
8. Add measurements and save the planning file.

## Main Interactions

### MPR

| Action | Control |
|---|---|
| Change standard slice | Mouse wheel |
| Zoom | Ctrl/Cmd + wheel or `− / +` |
| Move image and overlays | Enable `Pan`, then left-drag |
| Restore framing | `Fit` or `Fit MPR` |
| Window/level | Right-drag |

### 3D

| Action | Control |
|---|---|
| Rotate | Left-drag |
| Zoom | Mouse wheel |
| Pan | Enable `Pan`, then drag; or Shift-drag |
| Focus | Double-click anatomy |
| Restore orientation | `Reset View` |
| Show internal screws | Increase `Vertebra Transparency` |

### Screw Editing

| Selection | Result |
|---|---|
| Double-click screw head | Move entry only |
| Double-click tip or distal shaft | Move tip only |
| Double-click middle shaft | Move the complete screw |
| Double-click again | Confirm the current position |
| `Esc` | Leave active movement mode |

## Documentation

- [English User Guide](docs/USER_GUIDE.md)
- [한국어 사용설명서](docs/USER_GUIDE.ko.md)
- [Contributing](CONTRIBUTING.md)
- [Security and medical-data privacy](SECURITY.md)
- [Local DICOM data policy](data/README.md)
- [Authors and credits](AUTHORS.md)
- [Citation metadata](CITATION.cff)
- [macOS and Windows build guide](docs/BUILDING_DESKTOP.md)

## Data Privacy

Clinical DICOM files may expose identity through standard and private attributes, UIDs, filenames, overlays, metadata, or burned-in pixel annotations. This repository intentionally excludes local DICOM, NIfTI, logs, virtual environments, and generated exports.

Never upload clinical images, identifying screenshots, planning exports, or unredacted logs to GitHub issues, pull requests, releases, or CI artifacts.

## Project Structure

```text
main.py          application entry point
src/             application source code
tests/           automated tests
scripts/         launcher and optional TotalSegmentator installer
docs/            English and Korean user guides
data/README.md   local-data privacy instructions; no clinical data
```

## Known Limitations

- Automatic planning is geometric research software, not clinically validated navigation.
- Results depend on DICOM geometry and segmentation quality.
- Deformity, fracture, implants, artifacts, transitional anatomy, and poor segmentation may invalidate proposals.
- The native workflow is primarily tested on macOS; packaged Windows/Linux releases are not yet provided.
- TotalSegmentator is optional and may require model downloads and substantial memory.
- Clinical accuracy, inter-observer agreement, and prospective outcomes have not been established.

## Citation

If this software supports academic work, use GitHub's **Cite this repository** function or the metadata in [`CITATION.cff`](CITATION.cff):

> Park S-M. Pedicle Screw Simulator (Version 0.1.0) [Computer software]. 2026. https://github.com/grotyx/pedicle-screw-simulator

A peer-reviewed software-paper DOI can be added as the preferred citation after publication without replacing the versioned software citation.

## License

Project source code is released under the [MIT License](LICENSE). Use, modification, and redistribution are permitted provided that the copyright and permission notice are retained.

Third-party components retain their own licenses. Distributors must separately comply with the PyQt6 GPL/commercial licensing terms and the licenses of VTK, SimpleITK, TotalSegmentator, and other dependencies.
