"""Public release version consistency checks.

The repository-root ``VERSION`` file is the single source of truth for the
application version. These tests confirm every other location that surfaces
the version (the ``src`` package, CITATION.cff, README.md, and the desktop
build workflow) stays derived from it rather than hard-coding a literal.
"""

from pathlib import Path

import src


def test_package_version_matches_VERSION_file():
    assert src.__version__ == Path("VERSION").read_text(encoding="utf-8").strip()


def test_citation_and_notices_match_VERSION():
    version = Path("VERSION").read_text(encoding="utf-8").strip()
    assert f"version: {version}" in Path("CITATION.cff").read_text(encoding="utf-8")
    assert version in Path("README.md").read_text(encoding="utf-8")
    assert "0.1.0" not in Path(".github/workflows/build-desktop.yml").read_text(
        encoding="utf-8"
    )
