# Third-Party Notices

Pedicle Screw Simulator v0.1.0 includes third-party software in its standalone desktop packages. Each component remains subject to its own license.

## AI segmentation runtime

- **TotalSegmentator 2.12.0** — Apache License 2.0 — https://github.com/wasserth/TotalSegmentator
- **PyTorch 2.10.0** — BSD 3-Clause License — https://pytorch.org
- **nnU-Net v2** — Apache License 2.0 — https://github.com/MIC-DKFZ/nnUNet

The application uses TotalSegmentator's openly available `total` CT task. Model weights are downloaded by TotalSegmentator on first use and stored in the user's standard TotalSegmentator cache. Tasks identified upstream as license-restricted are not exposed by the application's default workflow.

## Desktop and imaging runtime

- **PyQt6 / Qt 6** — GPL v3 or a commercial Riverbank/Qt license
- **VTK** — BSD 3-Clause License
- **SimpleITK / ITK** — Apache License 2.0
- **pydicom** — MIT License
- **NumPy** — BSD 3-Clause License
- **SciPy** — BSD 3-Clause License
- **PyInstaller** — GPL v2 with a special exception for distributing bundled applications

The packaged application contains the license metadata distributed with its Python dependencies. Binary distributors remain responsible for complying with all applicable terms. In particular, distribution containing the GPL edition of PyQt6 must satisfy GPL v3 unless appropriate commercial licenses are held.
