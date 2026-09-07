"""Project credit, license, citation, and About-dialog metadata tests."""

from pathlib import Path

import src
from src.ui import main_window as main_window_module

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_public_creator_metadata_is_complete():
    assert src.__version__ == (PROJECT_ROOT / "VERSION").read_text(
        encoding="utf-8"
    ).strip()
    assert src.__author__ == "Sang-Min Park, MD, Ph.D."
    assert src.__organization__ == "Seoul National University Bundang Hospital"
    assert src.__email__ == "psmini@snu.ac.kr"
    assert src.__website__ == "https://sangmin.me"
    assert src.__license__ == "MIT"


def test_mit_license_preserves_creator_credit():
    license_text = (PROJECT_ROOT / "LICENSE").read_text(encoding="utf-8")
    assert "MIT License" in license_text
    assert "Copyright (c) 2026 Sang-Min Park, MD, Ph.D." in license_text
    assert "copyright notice and this permission notice" in license_text


def test_citation_file_contains_academic_metadata():
    version = (PROJECT_ROOT / "VERSION").read_text(encoding="utf-8").strip()
    citation = (PROJECT_ROOT / "CITATION.cff").read_text(encoding="utf-8")
    for expected in (
        f"version: {version}",
        'family-names: "Park"',
        'given-names: "Sang-Min"',
        'email: "psmini@snu.ac.kr"',
        'website: "https://sangmin.me"',
        'license: "MIT"',
        "Seoul National University Bundang Hospital",
    ):
        assert expected in citation


def test_about_html_exposes_version_creator_and_contact():
    version = (PROJECT_ROOT / "VERSION").read_text(encoding="utf-8").strip()
    about_html = main_window_module.build_about_html()
    for expected in (
        version,
        "Sang-Min Park, MD, Ph.D.",
        "Seoul National University Bundang Hospital",
        "psmini@snu.ac.kr",
        "https://sangmin.me",
        "MIT License",
        "Research and education use only",
    ):
        assert expected in about_html


def test_reproducibility_lock_pins_validated_environment():
    lock_text = (PROJECT_ROOT / "requirements-lock.txt").read_text(
        encoding="utf-8"
    )
    for expected in (
        "vtk==9.6.2",
        "simpleitk==2.5.5",
        "pydicom==3.0.2",
        "PyQt6==6.11.0",
        "numpy==2.4.6",
        "scipy==1.18.0",
        "pytest==9.1.1",
        "pyinstaller==6.21.0",
    ):
        assert expected in lock_text
    assert ">=" not in lock_text


# --- Task 4: CI workflow metadata -------------------------------------------


def _load_test_workflow():
    """Load .github/workflows/test.yml, preferring PyYAML but falling back to
    a small regex-based parser when PyYAML (a transitive dependency of
    totalsegmentator) is not installed in the current interpreter."""
    workflow_path = PROJECT_ROOT / ".github" / "workflows" / "test.yml"
    text = workflow_path.read_text(encoding="utf-8")
    try:
        import yaml

        return yaml.safe_load(text)
    except ImportError:
        import re

        # Extract the top-level `on:` block (everything indented under it,
        # up to the next top-level key).
        on_match = re.search(r"^on:\n((?:[ \t]+.*\n?)+)", text, re.MULTILINE)
        on_block = on_match.group(1) if on_match else ""
        on_keys = set(re.findall(r"^[ \t]+([A-Za-z_]+):", on_block, re.MULTILINE))

        run_lines = re.findall(r"run:\s*(.+)", text)

        return {
            "on": on_keys,
            "jobs": {
                "__fallback__": {
                    "steps": [{"run": line} for line in run_lines],
                }
            },
        }


def test_ci_runs_tests_on_pull_requests():
    wf = _load_test_workflow()
    on = wf.get("on") or wf.get(True)
    assert "pull_request" in on and "push" in on
    assert any(
        "pytest" in step.get("run", "")
        for job in wf["jobs"].values()
        for step in job["steps"]
    )
