#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${ROOT_DIR}/venv/bin/python"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python virtual environment was not found at: ${PYTHON_BIN}"
  echo "Create the venv first (example: python3 -m venv venv)."
  exit 1
fi

echo "[1/3] Upgrading pip in project venv..."
"${PYTHON_BIN}" -m pip install --upgrade pip

echo "[2/3] Installing TotalSegmentator..."
"${PYTHON_BIN}" -m pip install "TotalSegmentator==2.12.0"

TORCH_SHM_MANAGER="$("${PYTHON_BIN}" - <<'PY'
from pathlib import Path

try:
    import torch
except Exception:
    print("")
else:
    shm = Path(torch.__file__).resolve().parent / "bin" / "torch_shm_manager"
    print(str(shm) if shm.exists() else "")
PY
)"
if [[ -n "${TORCH_SHM_MANAGER}" && -f "${TORCH_SHM_MANAGER}" ]]; then
  chmod +x "${TORCH_SHM_MANAGER}" || true
fi

echo "[3/3] Verifying installation..."
"${PYTHON_BIN}" - <<'PY'
import importlib.util

ok = importlib.util.find_spec("totalsegmentator") is not None
if not ok:
    raise SystemExit("TotalSegmentator module was not detected after install.")

print("TotalSegmentator import check: OK")
PY

echo "Done. You can now run Auto Segmentation from the app UI."
echo "Note: first inference may download model weights and take extra time."
