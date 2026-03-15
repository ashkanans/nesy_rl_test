#!/usr/bin/env bash
set -euo pipefail

# Restart nesy_rl container with GPU visibility, optionally run a small CB PRE smoke test.
# No CPU/RAM limits are applied.

CONTAINER_NAME="nesy_rl"
IMAGE_NAME="nesy-rl:latest"
HOST_REPO="$(pwd)"
RUN_PRE_TEST=0
ALLOW_NO_GPU=0

usage() {
  cat <<USAGE
Usage: $0 [options]

Options:
  --container-name NAME   Container name (default: nesy_rl)
  --image NAME            Image name (default: nesy-rl:latest)
  --host-repo PATH        Host repo path to mount at /workspace/nesy_rl (default: current dir)
  --test-cb-pre           Run a quick CB PRE smoke test after startup
  --allow-no-gpu          For local testing on machines without NVIDIA GPU support
  -h, --help              Show this help
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --container-name)
      CONTAINER_NAME="$2"; shift 2 ;;
    --image)
      IMAGE_NAME="$2"; shift 2 ;;
    --host-repo)
      HOST_REPO="$2"; shift 2 ;;
    --test-cb-pre)
      RUN_PRE_TEST=1; shift ;;
    --allow-no-gpu)
      ALLOW_NO_GPU=1; shift ;;
    -h|--help)
      usage; exit 0 ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 2 ;;
  esac
done

if [[ ! -d "$HOST_REPO" ]]; then
  echo "Host repo path not found: $HOST_REPO" >&2
  exit 1
fi

GPU_ARGS=(--gpus all -e NVIDIA_VISIBLE_DEVICES=all -e NVIDIA_DRIVER_CAPABILITIES=compute,utility)
if ! command -v nvidia-smi >/dev/null 2>&1; then
  if [[ "$ALLOW_NO_GPU" -eq 1 ]]; then
    echo "[warn] nvidia-smi not found on host; continuing without GPU runtime (local test mode)."
    GPU_ARGS=()
  else
    echo "[error] nvidia-smi not found. Use --allow-no-gpu only for local CPU testing." >&2
    exit 1
  fi
fi

if command -v nvidia-smi >/dev/null 2>&1; then
  if ! nvidia-smi >/dev/null 2>&1 && [[ "$ALLOW_NO_GPU" -eq 0 ]]; then
    echo "[error] nvidia-smi exists but is not usable. Fix host GPU stack or use --allow-no-gpu for local test." >&2
    exit 1
  fi
fi

echo "[info] Restarting container: $CONTAINER_NAME"
docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true

docker run -d \
  --name "$CONTAINER_NAME" \
  "${GPU_ARGS[@]}" \
  -e PYTHONUNBUFFERED=1 \
  -v "$HOST_REPO":/workspace/nesy_rl:rw \
  -w /workspace/nesy_rl \
  "$IMAGE_NAME" \
  sleep infinity >/dev/null

echo "[info] Container started: $CONTAINER_NAME"

docker exec "$CONTAINER_NAME" bash -lc 'echo "python=$(python -V 2>&1)"; ls -d /opt/venv-torch /opt/venv-jax >/dev/null && echo "venvs=ok"'

if [[ "$RUN_PRE_TEST" -eq 1 ]]; then
  echo "[info] Running CB PRE smoke test (CPU/GPU depending on runtime)..."
  docker exec "$CONTAINER_NAME" bash -lc '
    set -euo pipefail
    cd /workspace/nesy_rl
    source /opt/venv-torch/bin/activate
    export HOME=/workspace/nesy_rl
    export MPLCONFIGDIR=/workspace/nesy_rl/.config/matplotlib
    OUT=/workspace/nesy_rl/runs/_smoke/cb_pre_smoke
    mkdir -p "$OUT" "$MPLCONFIGDIR"
    python scripts/run_baselines.py \
      --env cb \
      --spec reach_goal_while_avoid_bombs \
      --cb_state_semantics pre \
      --seed 0 \
      --num_episodes 64 \
      --max_steps 30 \
      --epochs 1 \
      --batch_size 8 \
      --block_size 32 \
      --n_layer 2 \
      --n_head 2 \
      --n_embd 64 \
      --evaluate \
      --eval_num_episodes 8 \
      --eval_max_steps 30 \
      --baselines vanilla logic \
      --alphas 0.0 0.1 \
      --skip_dataset_analysis \
      --base_run_dir "$OUT" >/tmp/cb_pre_smoke.log 2>&1
    python - <<"PY"
import csv, pathlib
p = pathlib.Path("/workspace/nesy_rl/runs/_smoke/cb_pre_smoke/baseline_metrics.csv")
print("baseline_metrics_csv_exists=", p.exists())
if p.exists():
    rows = list(csv.DictReader(p.open()))
    print("rows=", len(rows))
    print("baselines=", [r.get("baseline") for r in rows])
PY
  '
  echo "[info] CB PRE smoke test finished."
fi

echo "[done] $CONTAINER_NAME is ready."
