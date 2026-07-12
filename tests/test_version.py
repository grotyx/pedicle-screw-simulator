"""Public release version consistency checks."""

from pathlib import Path

import src


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_public_version_is_0_1_0():
    assert (PROJECT_ROOT / "VERSION").read_text(encoding="utf-8").strip() == "0.1.0"
    assert src.__version__ == "0.1.0"


def test_primary_documents_use_public_version():
    for relative_path in (
        "README.md",
        "README.ko.md",
        "docs/USER_GUIDE.md",
        "docs/USER_GUIDE.ko.md",
    ):
        document = (PROJECT_ROOT / relative_path).read_text(encoding="utf-8")
        assert "0.1.0" in document, f"Missing 0.1.0 in {relative_path}"
