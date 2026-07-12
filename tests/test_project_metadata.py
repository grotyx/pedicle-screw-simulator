"""Project credit, license, citation, and About-dialog metadata tests."""

from pathlib import Path

import src
from src.ui import main_window as main_window_module


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_public_creator_metadata_is_complete():
    assert src.__version__ == "0.1.0"
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
    citation = (PROJECT_ROOT / "CITATION.cff").read_text(encoding="utf-8")
    for expected in (
        'version: "0.1.0"',
        'family-names: "Park"',
        'given-names: "Sang-Min"',
        'email: "psmini@snu.ac.kr"',
        'website: "https://sangmin.me"',
        'license: "MIT"',
        "Seoul National University Bundang Hospital",
    ):
        assert expected in citation


def test_about_html_exposes_version_creator_and_contact():
    about_html = main_window_module.build_about_html()
    for expected in (
        "0.1.0",
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
    ):
        assert expected in lock_text
    assert ">=" not in lock_text
