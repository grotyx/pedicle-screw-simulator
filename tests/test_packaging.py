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
    assert '"VERSION"' in spec
    assert "PSS_VERSION_FILE" in spec
    assert "0.1.0" not in spec


def test_pyinstaller_spec_bundles_totalsegmentator_runtime():
    spec = (
        Path(__file__).resolve().parents[1]
        / "packaging"
        / "PedicleScrewSimulator.spec"
    ).read_text(encoding="utf-8")
    for package_name in (
        "totalsegmentator",
        "torch",
        "nnunetv2",
        "dynamic_network_architectures",
    ):
        assert package_name in spec
    assert 'excludes=["totalsegmentator", "torch"]' not in spec


def test_desktop_requirements_pin_totalsegmentator():
    requirements = (
        Path(__file__).resolve().parents[1] / "requirements-desktop.txt"
    ).read_text(encoding="utf-8")
    assert "-r requirements-dev.txt" in requirements
    for dependency in (
        "TotalSegmentator==2.12.0",
        "torch==2.10.0",
        "nnunetv2==2.6.4",
        "dynamic-network-architectures==0.4.3",
    ):
        assert dependency in requirements


def test_desktop_build_workflow_covers_apple_silicon_and_windows():
    workflow = (
        Path(__file__).resolve().parents[1]
        / ".github"
        / "workflows"
        / "build-desktop.yml"
    ).read_text(encoding="utf-8")
    for expected in (
        "windows-2025",
        "macos-15",
        "requirements-desktop.txt",
        "--self-check",
        "actions/upload-artifact",
    ):
        assert expected in workflow
    assert "macos-15-intel" not in workflow

    build_script = (
        Path(__file__).resolve().parents[1] / "scripts" / "build_desktop.py"
    ).read_text(encoding="utf-8")
    assert "README.ko.md" in build_script
    assert "docs/USER_GUIDE.ko.md" in build_script


def test_dependency_self_check_imports_ai_runtime(monkeypatch):
    imported = []
    setup_calls = []

    def fake_import_module(module_name):
        imported.append(module_name)
        if module_name == "totalsegmentator.config":
            return type(
                "FakeConfig",
                (),
                {"setup_nnunet": lambda self: setup_calls.append(True)},
            )()
        return object()

    monkeypatch.setattr(main.importlib, "import_module", fake_import_module)

    main.run_dependency_self_check()

    assert imported == [
        "totalsegmentator.config",
        "torch",
        "nnunetv2",
        "totalsegmentator.python_api",
        "totalsegmentator.nnunet",
    ]
    assert setup_calls == [True]


def test_windows_launcher_supports_same_flags_as_bash():
    from pathlib import Path

    text = Path("scripts/run_app.ps1").read_text(encoding="utf-8")
    for flag in ("--check", "--test", "--with-totalseg"):
        assert flag in text
    assert "python -m venv" in text or "py -3.12 -m venv" in text


def test_windows_launcher_is_cwd_independent_and_checks_pip_exit_codes():
    from pathlib import Path

    text = Path("scripts/run_app.ps1").read_text(encoding="utf-8")
    assert "Set-Location $root" in text
    assert text.count("$LASTEXITCODE -ne 0") >= 3
