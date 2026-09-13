# Third-Party Notices

Pedicle Screw Simulator v0.2.3 includes third-party software in its standalone desktop packages. Each component remains subject to its own license.

## AI segmentation runtime

- **TotalSegmentator 2.12.0** — Apache License 2.0 — https://github.com/wasserth/TotalSegmentator
- **PyTorch 2.10.0** — BSD 3-Clause License — https://pytorch.org
- **nnU-Net v2** — Apache License 2.0 — https://github.com/MIC-DKFZ/nnUNet
- **dynamic-network-architectures** — Apache License 2.0 — https://github.com/MIC-DKFZ/dynamic-network-architectures
- **batchgenerators / batchgeneratorsv2** — Apache License 2.0 — https://github.com/MIC-DKFZ/batchgenerators
- **acvl-utils** — Apache License 2.0 — https://github.com/MIC-DKFZ/acvl_utils
- **tqdm** — MIT License / Mozilla Public License 2.0 (dual-licensed) — https://github.com/tqdm/tqdm

The application uses TotalSegmentator's openly available `total` CT task. Model weights are downloaded by TotalSegmentator on first use and stored in the user's standard TotalSegmentator cache. Tasks identified upstream as license-restricted are not exposed by the application's default workflow.

### NVIDIA CUDA runtime (Windows build only)

The Windows desktop package installs PyTorch's CUDA-enabled wheels (`cu128` index) to accelerate the bundled AI segmentation runtime. These wheels redistribute NVIDIA CUDA runtime libraries (for example `cudart`, `cublas`, and `cudnn`). Those components are proprietary and are redistributed under NVIDIA's CUDA End User License Agreement (EULA) redistribution terms, not an open-source license. See https://docs.nvidia.com/cuda/eula/index.html for the applicable terms. The macOS build does not include CUDA and is unaffected.

### Optional pedicle subregion model (source installs only)

- **Spine_Subregions nnU-Net model** (Da Mutten et al., *J Imaging Inform Med* 2026) — https://github.com/MICN-Lab/Spine_Subregions — licence: see upstream repository (the repository has no `LICENSE` file and its README states no licence terms as of this writing).

This model is optional, not bundled with the application or its standalone packages, and is never downloaded automatically. A user who wants pedicle-subregion-refined isthmus detection downloads the trained weights separately from the project's GitHub Releases page and points the application at them; see [Desktop Build Guide](docs/BUILDING_DESKTOP.md) and [User Guide](docs/USER_GUIDE.md). It runs on the `nnunetv2` runtime already listed above and requires a non-frozen Python environment — the standalone packages refuse it.

## Desktop and imaging runtime

- **PyQt6 / Qt 6** — GPL v3 or a commercial Riverbank/Qt license
- **VTK** — BSD 3-Clause License
- **SimpleITK / ITK** — Apache License 2.0
- **pydicom** — MIT License
- **NumPy** — BSD 3-Clause License
- **SciPy** — BSD 3-Clause License
- **PyInstaller** — GPL v2 with a special exception for distributing bundled applications

The packaged application contains the license metadata distributed with its Python dependencies. Binary distributors remain responsible for complying with all applicable terms. In particular, distribution containing the GPL edition of PyQt6 must satisfy GPL v3 unless appropriate commercial licenses are held.

The Pedicle Screw Simulator source code is distributed under the MIT License. Standalone binary releases built from this source bundle the GPL v3 edition of PyQt6; the combined binary distribution is therefore made available under GPL v3, which is compatible with redistributing MIT-licensed code alongside it. The MIT-licensed source itself is unaffected and may still be reused under the MIT License.
