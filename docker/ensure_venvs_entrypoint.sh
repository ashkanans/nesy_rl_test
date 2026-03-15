#!/usr/bin/env bash
set -euo pipefail

VENV_TORCH="${VENV_TORCH:-/opt/venv-torch}"
VENV_JAX="${VENV_JAX:-/opt/venv-jax}"
PYTHON_BIN="${PYTHON_BIN:-}"

if [[ -z "${PYTHON_BIN}" ]]; then
  if command -v python3.10 >/dev/null 2>&1; then
    PYTHON_BIN="python3.10"
  else
    PYTHON_BIN="python3"
  fi
fi

ensure_venv() {
  local venv_path="$1"
  if [[ ! -x "${venv_path}/bin/python" ]]; then
    echo "[entrypoint] creating missing venv: ${venv_path}"
    mkdir -p "$(dirname "${venv_path}")"
    "${PYTHON_BIN}" -m venv "${venv_path}"
  fi
  "${venv_path}/bin/python" -m pip install -q -U pip setuptools wheel
}

ensure_venv "${VENV_TORCH}"
ensure_venv "${VENV_JAX}"

if [[ $# -eq 0 ]]; then
  exec /bin/bash
fi
exec "$@"
