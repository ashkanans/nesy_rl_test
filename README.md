# Neuro-Symbolic Offline RL

This repository contains experiments for logic-regularized offline RL and sequence modeling:

- Trajectory Transformer (TT) + logic loss (`train_cb.py`, `evaluate.py`, `run_baselines.py`)
- Safety/grid benchmarks in this branch: ColourBomb (`cb`) and NRM navigation (`nrm_nav`)
- External benchmark stacks included in-tree:
  - `trajectory-transformer/` (AntMaze via D4RL)
  - `implicit_q_learning/` (AntMaze IQL baseline)
  - `suffix-prediction/` and `nesy-suffix-prediction-dfa/` (DFA/LTL tooling)

## Benchmark support in `monolith`

- ColourBomb: supported (`train_cb.py`, `evaluate.py`)
- NRM nav: supported (`train_cb.py`, `evaluate.py`)
- AntMaze: supported via `trajectory-transformer/` scripts (requires D4RL/MuJoCo stack)
- FrozenLake: not yet wired to a training/evaluation pipeline in this branch

## 1) Setup

### Option A: `venv` (recommended for this repo)

```bash
git clone https://github.com/ashkanans/nesy_rl_test.git
cd nesy_rl_test

python3 -m venv .venv
source .venv/bin/activate

# installs requirements in order: core -> env -> tt
./scripts/setup.sh
```

### Option B: `conda`

```bash
git clone https://github.com/ashkanans/nesy_rl_test.git
cd nesy_rl_test

conda create -n nesy-rl python=3.10 -y
conda activate nesy-rl
./scripts/setup.sh
```

### Dependency split

- `requirements-core.txt`: core ML + logic + test dependencies
- `requirements-env.txt`: environment/rendering dependencies
- `requirements-tt.txt`: trajectory-transformer editable install
- `requirements.txt`: compatibility wrapper that includes all three files

### Setup script behavior

- Script: `scripts/setup.sh`
- Default order: `core -> env -> tt`
- Idempotent: safe to rerun; it only revalidates/upgrades installed packages
- Optional: `INSTALL_TT=0 ./scripts/setup.sh` to skip TT install

## 2) Submodule initialization

Run once after clone:

```bash
git submodule sync --recursive
git submodule update --init --recursive
```

Note: in this branch, the relevant code is already present as directories (`trajectory-transformer/`, `suffix-prediction/`, `nesy-suffix-prediction-dfa/`), so the commands above may be a no-op.

## 3) Smoke checks

From repo root:

```bash
# quick repository sanity checks (env + dataset + DFA adapter checks)
python sanity_tests.py

# pytest discovery smoke
pytest -q
```

## 3.1) Formatting and lint

```bash
pre-commit run --all-files
```

## 4) Train + eval commands

All commands below are run from repository root unless specified.

### 4.1 ColourBomb (`cb`)

Smoke train:

```bash
python train_cb.py \
  --env cb \
  --seed 0 \
  --num_episodes 64 \
  --max_steps 30 \
  --block_size 32 \
  --batch_size 8 \
  --epochs 1 \
  --ltl_formula "G(!(s0_bin40))" \
  --use_safe_dfa \
  --save_path runs/cb_smoke
```

Smoke eval (includes train/val/test split and writes metrics):

```bash
python evaluate.py \
  --env cb \
  --seed 0 \
  --num_episodes 64 \
  --max_steps 30 \
  --block_size 32 \
  --batch_size 8 \
  --epochs 1 \
  --eval_every 1 \
  --val_ratio 0.2 \
  --test_ratio 0.2 \
  --ltl_formula "G(!(s0_bin40))" \
  --use_safe_dfa \
  --out_dir runs/cb_eval_smoke
```

### 4.2 NRM nav (`nrm_nav`)

Smoke train:

```bash
python train_cb.py \
  --env nrm_nav \
  --seed 0 \
  --num_episodes 64 \
  --max_steps 30 \
  --block_size 32 \
  --batch_size 8 \
  --epochs 1 \
  --ltl_formulas "G(!(s0_bin11))" "G(!(s0_bin18))" \
  --dfa_mode product \
  --use_safe_dfa \
  --save_path runs/nrm_nav_smoke
```

Smoke eval:

```bash
python evaluate.py \
  --env nrm_nav \
  --seed 0 \
  --num_episodes 64 \
  --max_steps 30 \
  --block_size 32 \
  --batch_size 8 \
  --epochs 1 \
  --eval_every 1 \
  --val_ratio 0.2 \
  --test_ratio 0.2 \
  --ltl_formulas "G(!(s0_bin11))" "G(!(s0_bin18))" \
  --dfa_mode product \
  --use_safe_dfa \
  --out_dir runs/nrm_nav_eval_smoke
```

### 4.3 AntMaze (via `trajectory-transformer`)

Prerequisite: D4RL + MuJoCo stack available for your environment.

Train:

```bash
cd trajectory-transformer
python scripts/train.py --dataset antmaze-umaze-v0
```

Eval:

```bash
cd trajectory-transformer
python scripts/eval.py --dataset antmaze-umaze-v0 --episodes 5 --seed 0
```

### 4.4 FrozenLake

FrozenLake is currently not implemented as a train/eval benchmark in this branch (no `frozenlake` environment option in the root training/evaluation scripts yet).

## 5) Expected output artifacts

### Root TT + logic runs (`train_cb.py`, `evaluate.py`)

- Training checkpoints:
  - `runs/<name>/cb_state_<epoch>.pt`
- Evaluation outputs:
  - `runs/<name>/metrics.json`
  - `runs/<name>/metrics.csv`
  - `runs/<name>/satisfaction.png`
- Optional dataset analysis:
  - `<save_path>/dataset_analysis/summary.json`
  - `<save_path>/dataset_analysis/*.png`

### Baseline sweeps (`run_baselines.py`)

- Per-baseline metrics:
  - `<base_save_path>/<baseline_tag>/metrics.json`
- Aggregated summary:
  - `<base_save_path>/baseline_metrics.json`

### AntMaze TT outputs (`trajectory-transformer`)

- Training logs/checkpoints:
  - `trajectory-transformer/logs/<dataset>/<exp_name>/state_*.pt`
  - `trajectory-transformer/logs/<dataset>/<exp_name>/args.json`
- Eval summary:
  - `trajectory-transformer/logs/<dataset>/<exp_name>/<suffix>/eval.json`

## 6) Minimal acceptance run (from clean env)

```bash
./scripts/setup.sh
pytest -q

python evaluate.py \
  --env cb \
  --seed 0 \
  --num_episodes 64 \
  --max_steps 30 \
  --block_size 32 \
  --batch_size 8 \
  --epochs 1 \
  --eval_every 1 \
  --val_ratio 0.2 \
  --test_ratio 0.2 \
  --ltl_formula "G(!(s0_bin40))" \
  --use_safe_dfa \
  --out_dir runs/cb_eval_smoke
```
