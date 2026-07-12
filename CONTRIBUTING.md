# Contributing

Thank you for considering a contribution to Pedicle Screw Fixation Simulator.

## Before contributing

- Read the [English User Guide](docs/USER_GUIDE.md) or [한국어 사용설명서](docs/USER_GUIDE.ko.md).
- This project is research software, not a certified medical device.
- Do not submit clinical DICOM, segmentation files, screenshots, logs, plans, or meshes containing identifiable information.
- Do not describe automatic results as clinically validated unless the repository contains evidence supporting that exact claim.

Contributions are accepted under the project's MIT License. By submitting a contribution, you agree that it may be distributed under that license. Third-party code must retain compatible license and attribution information.

## Development workflow

1. Create a focused branch from the current default branch.
2. Add or update regression tests before changing behavior.
3. Keep changes limited to one problem or feature.
4. Update public documentation when controls, workflows, requirements, or outputs change.
5. Run:

```bash
./scripts/run_app.sh --check
./scripts/run_app.sh --test
venv/bin/python -m compileall -q src tests
git diff --check
```

## Pull-request checklist

- [ ] The change has a clear research or usability purpose.
- [ ] New behavior has automated tests.
- [ ] The full suite passes.
- [ ] No patient data, credentials, logs, local paths, or large generated files are included.
- [ ] Coordinate transforms preserve DICOM LPS geometry.
- [ ] Automatic planning failures do not fabricate a screw.
- [ ] Documentation and screenshots match the implementation.
- [ ] Clinical claims remain appropriately limited.

## Bug reports

Include operating system, Python version, steps to reproduce, expected result, actual result, and a redacted log excerpt. Use synthetic geometry whenever possible. Never attach clinical data to a public issue.
