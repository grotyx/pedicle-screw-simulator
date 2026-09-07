# Security and Medical Data Privacy

## Supported versions

The project is currently pre-release research software. Security fixes are applied to the current development version only.

## Reporting a vulnerability

Use GitHub private vulnerability reporting after it is enabled for the public repository. Do not open a public issue containing exploit details, credentials, patient information, or identifying screenshots. If private reporting is unavailable, report the vulnerability by e-mail to [psmini@snu.ac.kr](mailto:psmini@snu.ac.kr), the maintainer contact address published in [README.md](README.md).

Include:

- affected version or commit;
- platform and dependency versions;
- minimal reproduction using synthetic data;
- potential impact;
- suggested mitigation, if known.

## Medical data

This repository must not contain protected or identifiable health information. Clinical DICOM can expose identity through standard attributes, private attributes, UIDs, file metadata, overlays, structured content, filenames, and burned-in pixel annotations.

- Do not upload clinical DICOM or derived artifacts to issues, discussions, pull requests, releases, CI artifacts, or tutorial media.
- Use an approved DICOM confidentiality profile and independent review before sharing any medical dataset.
- Prefer synthetic or explicitly redistributable public datasets.
- Treat logs, JSON plans, STL files, segmentation masks, screenshots, and videos as potentially identifiable.

See [Local DICOM Data](data/README.md) for repository privacy rules.

## Clinical safety

The application is not a certified medical device. Segmentation, screw placement, dimensions, trajectories, breach grades, and warnings are software estimates and must not be relied on as the sole basis for patient care.
