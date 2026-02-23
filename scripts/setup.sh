#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"
INSTALL_TT="${INSTALL_TT:-1}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Error: '$PYTHON_BIN' not found."
  exit 1
fi

if ! "$PYTHON_BIN" -m pip --version >/dev/null 2>&1; then
  echo "Error: pip is unavailable for '$PYTHON_BIN'."
  echo "Create and activate a virtualenv first, then rerun this script."
  exit 1
fi

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  echo "Warning: no virtualenv is active. Dependencies will be installed into the current Python environment."
fi

echo "[setup] upgrading pip/setuptools/wheel"
"$PYTHON_BIN" -m pip install --upgrade pip setuptools wheel

echo "[setup] installing core requirements"
"$PYTHON_BIN" -m pip install -r requirements-core.txt

echo "[setup] installing environment requirements"
"$PYTHON_BIN" -m pip install -r requirements-env.txt

if [[ "$INSTALL_TT" == "1" ]]; then
  echo "[setup] installing trajectory-transformer requirements"
  "$PYTHON_BIN" -m pip install -r requirements-tt.txt
else
  echo "[setup] skipping trajectory-transformer requirements (INSTALL_TT=$INSTALL_TT)"
fi

echo "[setup] done"
