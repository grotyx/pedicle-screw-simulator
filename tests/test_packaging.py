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


def test_self_check_cli_failure_writes_traceback_log_and_stderr(monkeypatch, tmp_path, capsys):
    def boom():
        raise RuntimeError("nnunetv2 missing")

    monkeypatch.setattr(main, "run_dependency_self_check", boom)
    monkeypatch.setattr(main, "resolve_log_dir", lambda: tmp_path)

    exit_code = main.run_self_check_cli()

    assert exit_code == 1
    captured = capsys.readouterr()
    assert "RuntimeError: nnunetv2 missing" in captured.err

    log_file = tmp_path / "self_check_error.log"
    assert log_file.exists()
    assert "RuntimeError: nnunetv2 missing" in log_file.read_text(encoding="utf-8")


def test_self_check_cli_success_returns_zero_and_writes_no_log(monkeypatch, tmp_path):
    monkeypatch.setattr(main, "run_dependency_self_check", lambda: None)
    monkeypatch.setattr(main, "resolve_log_dir", lambda: tmp_path)

    assert main.run_self_check_cli() == 0
    assert not (tmp_path / "self_check_error.log").exists()


def test_self_check_cli_tolerates_unwritable_log_dir(monkeypatch, capsys):
    def boom():
        raise RuntimeError("nnunetv2 missing")

    def unwritable_log_dir():
        raise OSError("cannot resolve log dir")

    monkeypatch.setattr(main, "run_dependency_self_check", boom)
    monkeypatch.setattr(main, "resolve_log_dir", unwritable_log_dir)

    exit_code = main.run_self_check_cli()

    assert exit_code == 1
    captured = capsys.readouterr()
    assert "RuntimeError: nnunetv2 missing" in captured.err


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


# --- Task 6: packaging hardening --------------------------------------------


def test_pyinstaller_spec_disables_upx():
    spec = (
        Path(__file__).resolve().parents[1]
        / "packaging"
        / "PedicleScrewSimulator.spec"
    ).read_text(encoding="utf-8")
    assert "upx=False" in spec
    assert "upx=True" not in spec


def test_build_workflow_has_job_and_self_check_timeouts():
    workflow = (
        Path(__file__).resolve().parents[1]
        / ".github"
        / "workflows"
        / "build-desktop.yml"
    ).read_text(encoding="utf-8")
    # Job-level timeout guards against an unhandled self-check exception
    # hanging a windowed exe's modal dialog to the 6h default limit.
    assert "timeout-minutes: 90" in workflow
    # Both platform self-check steps get their own short timeout.
    assert workflow.count("timeout-minutes: 5") == 2


def test_build_workflow_prints_self_check_log_on_failure():
    workflow = (
        Path(__file__).resolve().parents[1]
        / ".github"
        / "workflows"
        / "build-desktop.yml"
    ).read_text(encoding="utf-8")
    assert "self_check_error.log" in workflow
    assert "steps.self_check_macos.outcome == 'failure'" in workflow
    assert "steps.self_check_windows.outcome == 'failure'" in workflow


def test_build_workflow_keeps_pip_cache_and_uses_start_process_self_check():
    workflow = (
        Path(__file__).resolve().parents[1]
        / ".github"
        / "workflows"
        / "build-desktop.yml"
    ).read_text(encoding="utf-8")
    assert "pip cache purge" not in workflow
    assert "Start-Process" in workflow


def test_install_totalsegmentator_script_pins_version():
    install_script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "install_totalsegmentator.sh"
    ).read_text(encoding="utf-8")
    assert "TotalSegmentator==2.12.0" in install_script
