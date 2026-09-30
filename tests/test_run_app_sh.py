"""scripts/run_app.sh refuses a python3 older than 3.12 before creating the venv."""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_app.sh"

pytestmark = pytest.mark.skipif(
    os.name != "posix" or shutil.which("bash") is None, reason="POSIX shell script"
)


def test_old_python3_is_rejected_with_an_actionable_message(tmp_path):
    # Copy the script so ROOT_DIR (and its venv) is an empty temp tree.
    (tmp_path / "scripts").mkdir()
    script = shutil.copy(SCRIPT, tmp_path / "scripts" / "run_app.sh")
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake = fake_bin / "python3"
    fake.write_text('#!/bin/sh\n[ "$1" = "-V" ] && echo "Python 3.10.0" && exit 0\nexit 1\n')
    fake.chmod(0o755)
    env = {**os.environ, "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}"}

    result = subprocess.run(
        ["bash", script, "--setup-only"], env=env, capture_output=True, text=True
    )

    assert result.returncode == 1
    assert "3.12" in result.stderr and "Python 3.10.0" in result.stderr
    assert not (tmp_path / "venv").exists()
