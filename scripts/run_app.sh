#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${ROOT_DIR}/venv"
PYTHON_BIN="${VENV_DIR}/bin/python"
REQ_FILE="${ROOT_DIR}/requirements.txt"
STAMP_FILE="${VENV_DIR}/.deps_installed"
DEV_REQ_FILE="${ROOT_DIR}/requirements-dev.txt"
DEV_STAMP_FILE="${VENV_DIR}/.dev_deps_installed"

RUN_MODE="run"
INSTALL_TOTALSEG="false"

usage() {
  cat <<'EOF'
Usage: ./scripts/run_app.sh [options]

Options:
  --setup-only       Create/update venv and install requirements only
  --check            Run import smoke-check and exit
  --test             Run the automated test suite and exit
  --with-totalseg    Install TotalSegmentator in project venv
  -h, --help         Show this help
EOF
}

for arg in "$@"; do
  case "$arg" in
    --setup-only)
      RUN_MODE="setup"
      ;;
    --check)
      RUN_MODE="check"
      ;;
    --test)
      RUN_MODE="test"
      ;;
    --with-totalseg)
      INSTALL_TOTALSEG="true"
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $arg"
      usage
      exit 1
      ;;
  esac
done

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "[setup] Creating virtual environment at ${VENV_DIR}"
  python3 -m venv "${VENV_DIR}"
fi

repair_count="$("${PYTHON_BIN}" - "${VENV_DIR}" <<'PY'
from pathlib import Path
import sys

venv_dir = Path(sys.argv[1]).resolve()
bin_dir = venv_dir / "bin"
old_paths = set()

for path in bin_dir.iterdir():
    if path.is_symlink() or not path.is_file():
        continue
    try:
        first_line = path.read_bytes().splitlines()[0]
    except (OSError, IndexError):
        continue
    if first_line.startswith(b"#!") and b"/venv/bin/python" in first_line:
        old_path = first_line[2:].split(b"/bin/python", 1)[0]
        if old_path != str(venv_dir).encode():
            old_paths.add(old_path)

changed = 0
for path in bin_dir.iterdir():
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 10_000_000:
        continue
    try:
        data = path.read_bytes()
    except OSError:
        continue
    if b"\0" in data:
        continue
    updated = data
    for old_path in old_paths:
        updated = updated.replace(old_path, str(venv_dir).encode())
    if updated != data:
        path.write_bytes(updated)
        changed += 1

print(changed)
PY
)"
if [[ "${repair_count}" != "0" ]]; then
  echo "[setup] Repaired ${repair_count} relocated virtual environment scripts"
fi

if [[ ! -f "${STAMP_FILE}" || "${REQ_FILE}" -nt "${STAMP_FILE}" ]]; then
  echo "[setup] Installing project dependencies"
  "${PYTHON_BIN}" -m pip install --upgrade pip
  "${PYTHON_BIN}" -m pip install -r "${REQ_FILE}"
  date -u +"%Y-%m-%dT%H:%M:%SZ" > "${STAMP_FILE}"
else
  echo "[setup] Requirements are up to date"
fi

if [[ "${INSTALL_TOTALSEG}" == "true" ]]; then
  "${ROOT_DIR}/scripts/install_totalsegmentator.sh"
fi

if [[ "${RUN_MODE}" == "test" && (
  ! -f "${DEV_STAMP_FILE}" || "${DEV_REQ_FILE}" -nt "${DEV_STAMP_FILE}"
) ]]; then
  echo "[setup] Installing development test dependencies"
  "${PYTHON_BIN}" -m pip install -r "${DEV_REQ_FILE}"
  date -u +"%Y-%m-%dT%H:%M:%SZ" > "${DEV_STAMP_FILE}"
fi

if [[ "${RUN_MODE}" == "setup" ]]; then
  echo "[done] Setup complete"
  exit 0
fi

if [[ "${RUN_MODE}" == "check" ]]; then
  echo "[check] Running import smoke-check"
  cd "${ROOT_DIR}"
  "${PYTHON_BIN}" - <<'PY'
import importlib

modules = [
    "src.core.dicom_loader",
    "src.core.volume_manager",
    "src.ui.main_window",
]
for module_name in modules:
    importlib.import_module(module_name)

print("Smoke-check passed: core and UI modules imported.")
PY
  echo "[done] Check complete"
  exit 0
fi

if [[ "${RUN_MODE}" == "test" ]]; then
  echo "[test] Running automated test suite"
  cd "${ROOT_DIR}"
  "${PYTHON_BIN}" -m pytest -q tests/
  echo "[done] Tests complete"
  exit 0
fi

echo "[run] Launching app"
cd "${ROOT_DIR}"
exec "${PYTHON_BIN}" main.py
