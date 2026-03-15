#!/usr/bin/env bash
set -euo pipefail

VENV_TORCH="${VENV_TORCH:-/opt/venv-torch}"
VENV_JAX="${VENV_JAX:-/opt/venv-jax}"
PYTHON_BIN="${PYTHON_BIN:-}"
WORKSPACE_DIR="${WORKSPACE_DIR:-/workspace/nesy_rl}"
AUTO_START_TELEGRAM_CB_HOURLY="${AUTO_START_TELEGRAM_CB_HOURLY:-1}"

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

start_telegram_cb_hourly() {
  if [[ "${AUTO_START_TELEGRAM_CB_HOURLY}" != "1" ]]; then
    return 0
  fi

  local script_path="${WORKSPACE_DIR}/scripts/telegram_cb_hourly_status.py"
  if [[ ! -f "${script_path}" ]]; then
    echo "[entrypoint] telegram auto-start skipped: script not found at ${script_path}"
    return 0
  fi

  if [[ -z "${TELEGRAM_API_ID:-}" || -z "${TELEGRAM_API_HASH:-}" || -z "${TELEGRAM_PHONE:-}" || -z "${TELEGRAM_GROUP_ID:-}" ]]; then
    echo "[entrypoint] telegram auto-start skipped: TELEGRAM_API_ID / TELEGRAM_API_HASH / TELEGRAM_PHONE / TELEGRAM_GROUP_ID not fully set"
    return 0
  fi

  local monitor_dir="${WORKSPACE_DIR}/runs/_monitor"
  local pid_file="${TELEGRAM_CB_HOURLY_PID_FILE:-${monitor_dir}/telegram_cb_hourly.pid}"
  local log_file="${TELEGRAM_CB_HOURLY_LOG_FILE:-${monitor_dir}/telegram_cb_hourly.log}"
  local session_file="${TELEGRAM_CB_HOURLY_SESSION:-${monitor_dir}/user_session}"
  local interval_sec="${TELEGRAM_CB_HOURLY_INTERVAL_SEC:-3600}"

  mkdir -p "${monitor_dir}"

  if [[ -f "${pid_file}" ]]; then
    local old_pid
    old_pid="$(cat "${pid_file}" 2>/dev/null || true)"
    if [[ -n "${old_pid}" ]] && kill -0 "${old_pid}" 2>/dev/null; then
      echo "[entrypoint] telegram hourly already running pid=${old_pid}"
      return 0
    fi
  fi

  nohup "${VENV_TORCH}/bin/python" "${script_path}" \
    --interval-sec "${interval_sec}" \
    --telethon-session "${session_file}" \
    > "${log_file}" 2>&1 < /dev/null &

  local new_pid=$!
  echo "${new_pid}" > "${pid_file}"
  echo "[entrypoint] started telegram hourly pid=${new_pid} log=${log_file}"
}

ensure_venv "${VENV_TORCH}"
ensure_venv "${VENV_JAX}"

start_telegram_cb_hourly || true

if [[ $# -eq 0 ]]; then
  exec /bin/bash
fi
exec "$@"
