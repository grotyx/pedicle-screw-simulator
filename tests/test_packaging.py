"""Standalone packaging and runtime-path tests."""

from pathlib import Path

import main


def test_source_build_logs_inside_project():
    assert main.resolve_log_dir(
        frozen=False,
        platform="darwin",
        environ={},
        home=Path("/Users/tester"),
    ) == Path(main.script_dir) / "logs"


def test_macos_bundle_logs_in_user_library():
    assert main.resolve_log_dir(
        frozen=True,
        platform="darwin",
        environ={},
        home=Path("/Users/tester"),
    ) == Path("/Users/tester/Library/Logs/PedicleScrewSimulator")


def test_windows_bundle_logs_in_local_app_data():
    assert main.resolve_log_dir(
        frozen=True,
        platform="win32",
        environ={"LOCALAPPDATA": r"C:\Users\tester\AppData\Local"},
        home=Path(r"C:\Users\tester"),
    ) == Path(
        r"C:\Users\tester\AppData\Local/PedicleScrewSimulator/logs"
    )


def test_pyinstaller_spec_contains_platform_metadata():
    spec = (Path(__file__).resolve().parents[1] / "packaging" / "PedicleScrewSimulator.spec").read_text(
        encoding="utf-8"
    )
    assert 'name="PedicleScrewSimulator"' in spec
    assert 'bundle_identifier="me.sangmin.pedicle-screw-simulator"' in spec
    assert 'version="0.1.0"' in spec
    assert "windows_version_info.txt" in spec


def test_desktop_build_workflow_covers_mac_and_windows():
    workflow = (
        Path(__file__).resolve().parents[1]
        / ".github"
        / "workflows"
        / "build-desktop.yml"
    ).read_text(encoding="utf-8")
    for expected in (
        "windows-2025",
        "macos-15",
        "macos-15-intel",
        "actions/upload-artifact",
    ):
        assert expected in workflow

    build_script = (
        Path(__file__).resolve().parents[1] / "scripts" / "build_desktop.py"
    ).read_text(encoding="utf-8")
    assert "README.ko.md" in build_script
    assert "docs/USER_GUIDE.ko.md" in build_script
