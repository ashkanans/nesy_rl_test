# Neuro-Symbolic Offline RL

This repository contains logic-regularized offline sequence RL code (Trajectory Transformer + DFA/LTLf tooling) for:

- `cb` (ColourBomb)
- `nrm_nav` (NRM navigation)
- `frozenlake`
- `antmaze` (dataset/protocol tooling; optional dependency stack)

## Repository Layout

- `scripts/`: canonical train/eval/baseline entrypoints
- `envs/`: benchmark environment wrappers
- `datasets/`: offline dataset generators/loaders
- `logic/`: DFA adapter and canonical token schema
- `planning/`: rollout evaluation and decoding runtime
- `configs/`: lightweight YAML presets

Legacy root entrypoints are still available as deprecated compatibility shims:
- `train_cb.py`
- `evaluate.py`
- `run_baselines.py`

## Setup

### Option A: venv

```bash
git clone https://github.com/ashkanans/nesy_rl_test.git
cd nesy_rl_test

python3 -m venv .venv
source .venv/bin/activate
./scripts/setup.sh
```

### Option B: conda

```bash
git clone https://github.com/ashkanans/nesy_rl_test.git
cd nesy_rl_test

conda create -n nesy-rl python=3.10 -y
conda activate nesy-rl
./scripts/setup.sh
```

### Dependency split

- `requirements-core.txt`: core ML/logic/test tooling
- `requirements-env.txt`: env/render/plot tooling
- `requirements-tt.txt`: editable `trajectory-transformer`

## Submodules

```bash
git submodule sync --recursive
git submodule update --init --recursive
```

## Validation Commands

```bash
pytest -q
python -m py_compile $(git ls-files '*.py')
```

## Canonical Entry Points

- Train: `python scripts/train.py ...`
- Evaluate: `python scripts/evaluate.py ...`
- Baselines: `python scripts/run_baselines.py ...`

Optional root aliases:
- `python train.py ...` (canonical alias for `scripts/train.py`)
- `python train_cb.py ...` (deprecated)
- `python evaluate.py ...` (deprecated)
- `python run_baselines.py ...` (deprecated)

## Smoke Runs

### ColourBomb

Train:

```bash
python scripts/train.py \
  --env cb \
  --smoke \
  --spec avoid_single_bomb_22 \
  --use_safe_dfa \
  --run_dir runs/cb/smoke_train
```

Eval:

```bash
python scripts/evaluate.py \
  --env cb \
  --smoke \
  --spec avoid_single_bomb_22 \
  --use_safe_dfa \
  --checkpoint runs/cb/smoke_train/cb_state_0.pt \
  --run_dir runs/cb/smoke_eval
```

### FrozenLake

Train:

```bash
python scripts/train.py \
  --env frozenlake \
  --smoke \
  --spec reach_goal_while_avoid_holes \
  --use_safe_dfa \
  --frozenlake_map_size 4x4 \
  --run_dir runs/frozenlake/smoke_train
```

Eval:

```bash
python scripts/evaluate.py \
  --env frozenlake \
  --smoke \
  --spec reach_goal_while_avoid_holes \
  --use_safe_dfa \
  --frozenlake_map_size 4x4 \
  --checkpoint runs/frozenlake/smoke_train/cb_state_0.pt \
  --run_dir runs/frozenlake/smoke_eval
```

### NRM Nav

```bash
python scripts/train.py \
  --env nrm_nav \
  --smoke \
  --spec avoid_state_11 \
  --use_safe_dfa \
  --run_dir runs/nrm_nav/smoke_train
```

## Config Presets

`scripts/train.py` accepts `--config` (`.yaml`, `.yml`, `.json`):

```bash
python scripts/train.py --config configs/cb_smoke.yaml --run_dir runs/cb/smoke_from_cfg
python scripts/train.py --config configs/frozenlake_smoke.yaml --run_dir runs/frozenlake/smoke_from_cfg
```

CLI flags override config values.

## Migration Notes

- Old: `python train_cb.py ...`
- New: `python scripts/train.py ...` (or `python train.py ...`)

- Old: `python evaluate.py ...`
- New: `python scripts/evaluate.py ...`

- Old: `python run_baselines.py ...`
- New: `python scripts/run_baselines.py ...`

## Artifacts

By default, runs are created under `runs/<env>/<UTC timestamp>/` unless `--run_dir` is provided.

Per-run outputs:

- `metrics.json`
- `metrics.csv`
- `dfa_summary.json`
- `automaton_rollout_stats.json`
- checkpoints (for train runs): `cb_state_<epoch>.pt`
- optional plots under `plots/`

## Telegram Hourly Status Monitor

You can run a lightweight outbound-only reporter from inside the running container.
It sends one concise status message every hour with:

- currently running experiment jobs (`scripts/train.py`, `scripts/evaluate.py`, `scripts/run_baselines.py`, etc.)
- CPU load, RAM usage, and disk usage
- per-GPU utilization/memory/power/temperature (via `nvidia-smi`)

### 1) Configure Telegram

Default transport matches `YoutubeBot/bot_sender.py` style (`Telethon` user session).

Set:

- `API_ID` (or `TELEGRAM_API_ID`)
- `API_HASH` (or `TELEGRAM_API_HASH`)
- `PHONE` (or `TELEGRAM_PHONE`)
- `GROUP_ID` (or `TELEGRAM_GROUP_ID`) if already known, otherwise set `TELEGRAM_GROUP_TITLE`

Notes:

- If `GROUP_ID` is missing and `TELEGRAM_GROUP_TITLE` is set, the tool finds that group.
- If not found, it creates it once by default (`--create-group-if-missing`).
- Session file defaults to `user_session` (same style as your `bot_sender.py`).

### 2) Run one snapshot (test)

```bash
python scripts/telegram_hourly_status.py --once --telegram-mode telethon
```

### 3) Run hourly loop

```bash
python scripts/telegram_hourly_status.py --interval-sec 3600 --telegram-mode telethon
```

### 4) Initialize and persist group/chat detection

```bash
python scripts/telegram_hourly_status.py --ensure-group --telegram-mode telethon
```

Example with env vars:

```bash
pip install telethon
export API_ID=...
export API_HASH=...
export PHONE=+39...
export TELEGRAM_GROUP_TITLE="NeSy RL Monitor"
python scripts/telegram_hourly_status.py --ensure-group --telegram-mode telethon
```

Useful flags:

- `--dry-run` prints the report without sending
- `--max-jobs N` limits number of listed jobs
- `--disk-path /workspace/nesy_rl` selects the disk path to monitor
- `--chat-id ...` overrides group/chat id
- `--telethon-session ...` changes session file path
- Optional fallback: `--telegram-mode bot` uses Bot API (`TELEGRAM_BOT_TOKEN`)

## AntMaze

AntMaze support is provided through `antmaze_dataset.py` and `antmaze_eval.py` with D4RL/MuJoCo dependencies.
If your environment cannot install the AntMaze stack (e.g., Python/toolchain mismatch), run `cb`, `nrm_nav`, and `frozenlake` smoke paths first and keep AntMaze as a deferred benchmark step.
