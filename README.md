# Pedicle Screw Simulator

[English](README.md) | [한국어](README.ko.md)

Research desktop software for DICOM CT visualization, vertebral segmentation, and interactive pedicle screw planning with synchronized MPR and 3D review.

**Version:** 0.2.2

**Primary tested environment:** macOS, Python 3.12

**Creator:** [Sang-Min Park, MD, Ph.D.](https://sangmin.me)

**Affiliation:** Spine Center and Department of Orthopaedic Surgery, Seoul National University Bundang Hospital; Seoul National University College of Medicine

**Contact:** [psmini@snu.ac.kr](mailto:psmini@snu.ac.kr)

> **Research and education use only.** This software is not a certified medical device and must not be used as the sole basis for diagnosis, surgery, navigation, or patient care. Every segmentation, screw proposal, dimension, breach grade, and warning requires independent review by a qualified clinician.

## Highlights

- Multi-series DICOM CT loading
- Synchronized axial, sagittal, and coronal MPR, anterior-up with A/P/L/R orientation markers
- GPU-capable VTK volume rendering (CPU ray casting on macOS) and selectable vertebral meshes
- Optional GPU-first TotalSegmentator integration with CPU retry, and CT-guided mask refinement that removes upsampling stair-steps
- Optional locally installed pedicle subregion nnU-Net for label-based isthmus refinement (source installs only)
- Multi-level vertebra selection and automatic screw proposals
- Multi-objective trajectory optimizer (default) with a legacy planner fallback, plus a cortical bone trajectory (CBT) planning mode
- Narrow-pedicle policy: the smallest implant, the medial wall protected, and the level marked, instead of skipping the side
- Trajectories aimed parallel to the upper endplate, and construct-level rod-line and convergence harmonisation
- Gertzbein-Robbins grading with medial/lateral/craniocaudal breach split, bone-quality HU metrics, Heary direction, and facet violation grade
- Standard and screw-aligned oblique MPR review, with rotatable and offsettable screw MPR planes
- Direct entry, tip, and whole-screw editing in MPR and 3D
- Manual screw placement, distance measurement, and angle measurement
- JSON plan save/load and supported CSV/STL export
- Editable planning parameters, a per-screw review table and inspector, three UI themes (one light, two dark), MPR pan/zoom, pane maximise, and 3D navigation controls

## Quick Start

### macOS / Linux

```bash
git clone https://github.com/grotyx/pedicle-screw-simulator.git
cd pedicle-screw-simulator
./scripts/run_app.sh
```

The launcher creates a local `venv`, installs required packages, and starts the application.

### Windows

```powershell
git clone https://github.com/grotyx/pedicle-screw-simulator.git
cd pedicle-screw-simulator
powershell -ExecutionPolicy Bypass -File scripts\run_app.ps1
```

The PowerShell launcher creates the same local `venv` and installs the same required packages before starting the application.

### Optional TotalSegmentator

```bash
./scripts/run_app.sh --with-totalseg
```

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_app.ps1 --with-totalseg
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

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_app.ps1 --check
powershell -ExecutionPolicy Bypass -File scripts\run_app.ps1 --test
```

Run `./scripts/run_app.sh --test` (macOS/Linux) or `powershell -ExecutionPolicy Bypass -File scripts\run_app.ps1 --test` (Windows) to run the current automated test suite and see the up-to-date test count.

For exact reproduction of the validated macOS/Python 3.12 environment:

```bash
python -m pip install -r requirements-lock.txt
```

`requirements.txt` provides compatible minimum versions. `requirements-lock.txt` records the exact core runtime and test environment used for v0.2.2 verification. `requirements-desktop.txt` pins the additional TotalSegmentator and PyTorch runtime used in standalone builds.

## Standalone Desktop Packages

The GitHub **Build desktop packages** workflow creates separate archives for:

- macOS Apple Silicon (`.app`)
- Windows x64 (`.exe` with its support folder)

See [Desktop Build Guide](docs/BUILDING_DESKTOP.md). The macOS package is not Apple-notarized yet, so Gatekeeper may require right-clicking the app and selecting **Open**.

Standalone packages include TotalSegmentator 2.12.0, PyTorch, and nnU-Net. No separate Python installation is required. The first automatic segmentation downloads the open `total` task model weights to the user's TotalSegmentator cache, so an internet connection and additional disk space are required once. Later runs reuse the cached models.

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
- [Changelog](CHANGELOG.md) · [변경 이력](CHANGELOG.ko.md)
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
- Pedicle isthmus detection is geometric (coronal cross-section minimum on the TotalSegmentator mask) and not validated on deformity.
- Standalone builds are provided for Apple Silicon macOS and Windows x64; Intel macOS and Linux packages are not provided.
- TotalSegmentator is bundled in standalone builds, but its model weights are downloaded on first use and require substantial memory and disk space.
- The optional pedicle subregion model (nnU-Net pedicle/corpus/lamina label refinement) requires a source install with `nnunetv2`; standalone packages refuse it.
- The currently published Spine_Subregions release uses an nnU-Net v1-style folder layout, not the nnU-Net v2 layout this app expects, so the model directory is not recognized until the weights are re-exported for nnU-Net v2.
- Clinical accuracy, inter-observer agreement, and prospective outcomes have not been established.

## Citation

If this software supports academic work, use GitHub's **Cite this repository** function or the metadata in [`CITATION.cff`](CITATION.cff):

> Park S-M. Pedicle Screw Simulator (Version 0.2.2) [Computer software]. 2026. https://github.com/grotyx/pedicle-screw-simulator

A peer-reviewed software-paper DOI can be added as the preferred citation after publication without replacing the versioned software citation.

## References

The bone-quality thresholds, breach/facet classifications, and plan-validation measures documented in the [User Guide](docs/USER_GUIDE.md) draw on the following literature, all of which backs features documented in this version. The application's constants and formulas are engineering approximations of these sources, not a substitute for them; see `src/utils/constants.py`, `src/core/bone_quality.py`, `src/core/breach_classification.py`, and `src/core/plan_metrics.py` for the exact implementation.

- Götschi et al. (2026), *Journal of Spine Surgery* — pedicle screw sizing margins (pedicle fill ratio, cortical wall clearance, anterior safety margin).
- Yamamoto et al. (2025), *Asian Spine Journal* — trajectory HU cutoff associated with screw loosening risk.
- Dhar et al. (2026), *Asian Spine Journal* — trajectory HU range associated with screw loosening.
- Chen et al. (2024), *Orthopaedic Surgery* — pedicle-region HU sampling methodology.
- Yang et al. (2026), *Global Spine Journal* — trajectory-to-vertebral-body HU ratio and loosening odds.
- Sankar et al. (2026), *Neurosurgery* — L1-L5 trabecular HU thresholds for osteoporosis and low bone density.
- Heary et al. (2004) — classification of pedicle screw cortical breach direction.
- Babu et al. (2012) — grading of facet-joint violation by pedicle screw instrumentation.
- Scherer et al. (2022), *The Spine Journal* — inter-rater agreement in pedicle screw trajectory planning.
- Wang et al. (2024), *Bioengineering* — pedicle screw sizing rule (diameter as a fraction of isthmus width, with cortical wall clearance) and screw-volume Dice overlap / plan-comparison methodology.
- Da Mutten et al. (2026), *Journal of Imaging Informatics in Medicine* — automated nnU-Net segmentation of thoracolumbar spine subregions (pedicle/corpus/lamina/spinous/transverse/articular), used by the optional pedicle subregion model.
- Massalimova et al. (2025), *Scientific Reports* — pedicle-centre offset metric for automated screw trajectory evaluation.
- Herkner et al. (2026), *Journal of Clinical Medicine* — Bland-Altman comparison of automated versus surgeon-selected implant dimensions.
- Zhang et al. (2024), *Asian Spine Journal* — Delphi consensus on cortical bone trajectory (CBT) screw indications — basis for the cortical bone trajectory (CBT) planning mode's contraindication note.
- Zeng et al. (2024), *Orthopaedic Surgery* — CT-based cortical bone trajectory (CBT) screw trajectory parameters — basis for the cortical bone trajectory (CBT) planning mode's default entry angles.

## License

Project source code is released under the [MIT License](LICENSE). Use, modification, and redistribution are permitted provided that the copyright and permission notice are retained.

Third-party components retain their own licenses. See [Third-Party Notices](THIRD_PARTY_NOTICES.md). Distributors must separately comply with the PyQt6 GPL/commercial licensing terms and the licenses of VTK, SimpleITK, TotalSegmentator, PyTorch, nnU-Net, and other dependencies.
