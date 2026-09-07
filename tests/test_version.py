"""Public release version consistency checks.

The repository-root ``VERSION`` file is the single source of truth for the
application version. These tests confirm every other location that surfaces
the version (the ``src`` package, CITATION.cff, README.md, other public
documents, and the desktop build workflow) stays derived from it rather than
hard-coding a literal, and that a missing ``VERSION`` file degrades gracefully
instead of breaking ``import src``.
"""

import re
from pathlib import Path

import pytest

import src

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_package_version_matches_VERSION_file():
    assert src.__version__ == (PROJECT_ROOT / "VERSION").read_text(
        encoding="utf-8"
    ).strip()


def test_citation_and_notices_match_VERSION():
    version = (PROJECT_ROOT / "VERSION").read_text(encoding="utf-8").strip()
    citation = (PROJECT_ROOT / "CITATION.cff").read_text(encoding="utf-8")
    assert re.search(rf'version:\s*"?{re.escape(version)}"?', citation)
    assert version in (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    assert "0.1.0" not in (
        PROJECT_ROOT / ".github" / "workflows" / "build-desktop.yml"
    ).read_text(encoding="utf-8")


@pytest.mark.parametrize(
    "relative_path",
    [
        "README.ko.md",
        "docs/USER_GUIDE.md",
        "docs/USER_GUIDE.ko.md",
        "THIRD_PARTY_NOTICES.md",
    ],
)
def test_public_documents_contain_VERSION(relative_path):
    version = (PROJECT_ROOT / "VERSION").read_text(encoding="utf-8").strip()
    document = (PROJECT_ROOT / relative_path).read_text(encoding="utf-8")
    assert version in document, f"Missing {version} in {relative_path}"


def test_missing_version_file_falls_back_to_placeholder(monkeypatch, tmp_path):
    """A missing/unreadable VERSION file must not raise at import time."""
    monkeypatch.setattr(src.sys, "frozen", True, raising=False)
    monkeypatch.setattr(src.sys, "_MEIPASS", str(tmp_path), raising=False)
    assert src._read_version() == "0.0.0+unknown"
