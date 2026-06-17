# NeSy RL Study Workflow

This document is a study map for this repository only:

`/home/laptop-1029/Edu/Sapienza/nesy_rl_test`

The project is about offline reinforcement learning with temporal-logic constraints. In plain words: we train sequence models to act in small benchmark environments, then use LTLf/DFA knowledge to measure or improve whether their trajectories obey safety/reachability rules.

## 1. What This Repo Contains

Core idea:

- Generate or load offline trajectories from an environment.
- Convert each trajectory into token sequences.
- Define a temporal rule as an LTLf formula, for example "always avoid bomb states" or "avoid holes and eventually reach the goal".
- Compile that formula into a DFA.
- Train/evaluate sequence models with optional logic guidance.
- Report reward, violation rate, satisfaction rate, and automaton diagnostics.

Main model families:

- Trajectory Transformer (TT): token-level autoregressive model over full transition tokens.
- Decision Transformer (DT): predicts actions from state, previous action, return-to-go, and timestep.
- Baselines: scripts for simple non-logic comparisons.

Main environments:

- `cb`: ColourBomb grid.
- `frozenlake`: FrozenLake safety/reachability tasks.
- `nrm_nav`: navigation with unsafe states.
- `dsrl`: dataset-backed safety RL support.
- `antmaze`: protocol/dataset tooling, heavier optional dependencies.

## 2. Local Machine Capabilities

From the current Ubuntu setup, these are available:

- Ubuntu 24.04.
- `python3` is available.
- `git` is available.
- `docker` is available.

Important note:

- In my non-interactive shell, `python` was not on PATH, but `python3` was. Your terminal prompt shows `(.venv)`, so inside your activated venv `python` may work. Use `python3` if `python` fails.
- `python3 -m pip` was not available system-wide in my shell. The repo expects a venv or container setup for Python packages.

Useful validation commands:

```bash
python3 --version
git --version
docker --version
```

## 3. Directory Map

Read these first:

- [README.md](/home/laptop-1029/Edu/Sapienza/nesy_rl_test/README.md): top-level commands and smoke runs.
- [docs/workflow/README.md](/home/laptop-1029/Edu/Sapienza/nesy_rl_test/docs/workflow/README.md): rendered data-flow diagrams and editing/export workflow.
- [docs/workflow/MODEL_ARCHITECTURES.md](/home/laptop-1029/Edu/Sapienza/nesy_rl_test/docs/workflow/MODEL_ARCHITECTURES.md): exact TT and DT inputs, architecture, losses, and outputs.
- [paper/notes/method_logic_stack.md](/home/laptop-1029/Edu/Sapienza/nesy_rl_test/paper/notes/method_logic_stack.md): DFA/LTLf termination and hard/soft satisfaction.
- [paper/notes/data_representation.md](/home/laptop-1029/Edu/Sapienza/nesy_rl_test/paper/notes/data_representation.md): transition token schema.
- [paper/notes/eval_protocol.md](/home/laptop-1029/Edu/Sapienza/nesy_rl_test/paper/notes/eval_protocol.md): metric definitions and evaluation contract.

Important code areas:

- `scripts/`: canonical entrypoints for training, evaluation, sweeps, suites, and matrices.
- `datasets/`: offline episode generation and sequence dataset wrappers.
- `envs/`: environment implementations.
- `specs/`: named LTLf specification presets.
- `logic/`: canonical token schema.
- [dfa_adapter.py](/home/laptop-1029/Edu/Sapienza/nesy_rl_test/dfa_adapter.py): maps model tokens to DFA symbols and checks satisfaction.
- [logic_loss_tt.py](/home/laptop-1029/Edu/Sapienza/nesy_rl_test/logic_loss_tt.py): TT differentiable logic loss.
- `planning/`: rollout evaluation, decoding, DT runtime, constrained selection.
- `models/`: TT and DT model definitions.
- `tests/`: behavior checks for DFA, datasets, DT, TT logic, evaluation, schemas, determinism.
- `paper/`: manuscript and experiment notes.
- `runs/`: generated outputs.

## 4. Setup Workflow

Preferred local setup:

```bash
cd /home/laptop-1029/Edu/Sapienza/nesy_rl_test
python3 -m venv .venv
source .venv/bin/activate
./scripts/setup.sh
```

If the venv is already active, start from:

```bash
cd /home/laptop-1029/Edu/Sapienza/nesy_rl_test
source .venv/bin/activate
```

Submodules:

```bash
git submodule sync --recursive
git submodule update --init --recursive
```

Validation:

```bash
pytest -q
python -m py_compile $(git ls-files '*.py')
```

If `python` fails, use:

```bash
python3 -m py_compile $(git ls-files '*.py')
```

## 5. Data Workflow

The data path is:

1. Choose an environment with `--env`.
2. Build an offline dataset with `build_dataset(...)` for TT or `build_dt_offline_source(...)` for DT.
3. Episodes are serialized as fixed-width transition rows.
4. Current active schemas use width `4`:
   - index `0`: state
   - index `1`: action
   - index `2`: reward or reward proxy
   - index `3`: auxiliary/cost field
5. Full episodes append exactly one explicit `END` row.
6. Training uses sliding windows, so a window may or may not contain the `END` row.
7. Evaluation uses full executed rollouts and appends canonical `END` before DFA checking.

Files to study:

- [logic/token_schema.py](/home/laptop-1029/Edu/Sapienza/nesy_rl_test/logic/token_schema.py)
- [datasets/cb_dataset.py](/home/laptop-1029/Edu/Sapienza/nesy_rl_test/datasets/cb_dataset.py)
- [datasets/frozenlake_dataset.py](/home/laptop-1029/Edu/Sapienza/nesy_rl_test/datasets/frozenlake_dataset.py)
- [datasets/dt_dataset.py](/home/laptop-1029/Edu/Sapienza/nesy_rl_test/datasets/dt_dataset.py)

## 6. LTLf and DFA Workflow

The logic path is:

1. A formula comes from `--spec`, `--ltl_formula`, or `--ltl_formulas`.
2. `resolve_formulas(...)` turns the spec name into one or more formula strings.
3. `TTDFAAdapter` creates a symbolic vocabulary like `s0_bin22`, `a0_bin1`, `end`.
4. `create_dfa_from_ltl(...)` builds a DFA.
5. If there are multiple formulas, `dfa_mode` controls the combination:
   - `single`: use first formula.
   - `product`: combine formulas into one product DFA.
   - `multi`: keep several DeepDFAs and average logic losses.
6. DFA can be converted to DeepDFA for differentiable training/evaluation.

Canonical finite-trace rule:

- Consume all symbols before `END`.
- Consume exactly one explicit `END`.
- Check acceptance.
- Missing `END` means incomplete trace and is unsatisfied in canonical mode.

Files to study:

- [train_cb.py](/home/laptop-1029/Edu/Sapienza/nesy_rl_test/train_cb.py): `resolve_formulas`, `build_adapter_and_dfa`.
- [dfa_adapter.py](/home/laptop-1029/Edu/Sapienza/nesy_rl_test/dfa_adapter.py): token/symbol mapping and satisfaction.
- [specs/cb_specs.py](/home/laptop-1029/Edu/Sapienza/nesy_rl_test/specs/cb_specs.py)
- [specs/frozenlake_specs.py](/home/laptop-1029/Edu/Sapienza/nesy_rl_test/specs/frozenlake_specs.py)
- [specs/nrm_nav_specs.py](/home/laptop-1029/Edu/Sapienza/nesy_rl_test/specs/nrm_nav_specs.py)

## 7. TT Training Workflow

Canonical command:

```bash
python scripts/train.py --config configs/cb_smoke.yaml --run_dir runs/cb/study_tt_smoke
```

What happens:

1. `scripts/train.py` loads optional YAML/JSON config.
2. It calls `train_cb.train(...)`.
3. Dataset is built for the selected environment.
4. Adapter + DFA + DeepDFA are built from the selected LTLf spec.
5. TT model is built.
6. `LogicLossModule` combines supervised token loss with logic loss:

```text
total_loss = (1 - alpha) * supervised_loss + alpha * logic_loss
```

7. The logic loss samples soft trajectories with Gumbel-Softmax, maps token probabilities to DFA symbol probabilities, feeds DeepDFA, and penalizes low acceptance probability.
8. Checkpoints and metrics are written to the run directory.

Key TT logic files:

- [scripts/train.py](/home/laptop-1029/Edu/Sapienza/nesy_rl_test/scripts/train.py)
- [train_cb.py](/home/laptop-1029/Edu/Sapienza/nesy_rl_test/train_cb.py)
- [models/tt_model.py](/home/laptop-1029/Edu/Sapienza/nesy_rl_test/models/tt_model.py)
- [logic_loss_tt.py](/home/laptop-1029/Edu/Sapienza/nesy_rl_test/logic_loss_tt.py)

Useful smoke commands:

```bash
python scripts/train.py --config configs/cb_smoke.yaml --run_dir runs/cb/study_tt_cb
python scripts/train.py --config configs/frozenlake_smoke.yaml --run_dir runs/frozenlake/study_tt_fl
python scripts/train.py --config configs/nrm_nav_smoke.yaml --run_dir runs/nrm_nav/study_tt_nrm
```

## 8. DT Training Workflow

Canonical command:

```bash
python scripts/train_dt.py \
  --env frozenlake \
  --smoke \
  --run_dir runs/frozenlake/study_dt_smoke
```

What happens:

1. `build_dt_offline_source(...)` creates or loads environment episodes.
2. `build_dt_dataset(...)` converts episodes into DT windows.
3. Each DT training item contains:
   - current states
   - previous actions
   - return-to-go
   - timesteps
   - target current actions
   - mask for real vs padded positions
4. The DT model input at each timestep is:

```text
state_t + previous_action_t + rtg_t + timestep_t
```

5. The target is `action_t`.
6. Supervised loss is action cross-entropy.
7. If `--logic_alpha > 0`, an extra logic rollout penalty is added:

```text
loss = action_CE + logic_alpha * logic_loss
```

8. Current DT logic loss does not use LTLf/DeepDFA directly. It uses a transition model and hazard mask to estimate short-horizon unsafe-state probability.

Important fairness note:

- With `logic_alpha=0`, DT uses no transition model.
- With `logic_alpha>0`, `build_tabular_dynamics(...)` may use environment transition tables/rules when available, otherwise dataset-estimated counts.
- For strict offline-RL comparisons, treat env-derived dynamics as privileged information and report separately from dataset-estimated dynamics.

Files to study:

- [scripts/train_dt.py](/home/laptop-1029/Edu/Sapienza/nesy_rl_test/scripts/train_dt.py)
- [datasets/dt_dataset.py](/home/laptop-1029/Edu/Sapienza/nesy_rl_test/datasets/dt_dataset.py)
- [models/dt_model.py](/home/laptop-1029/Edu/Sapienza/nesy_rl_test/models/dt_model.py)
- [planning/dt_runtime.py](/home/laptop-1029/Edu/Sapienza/nesy_rl_test/planning/dt_runtime.py)

## 9. Evaluation Workflow

TT checkpoint evaluation:

```bash
python scripts/evaluate.py \
  --env cb \
  --smoke \
  --spec avoid_single_bomb_22 \
  --use_safe_dfa \
  --checkpoint runs/cb/study_tt_cb/cb_state_0.pt \
  --run_dir runs/cb/study_tt_cb_eval
```

What evaluation does:

1. Rebuilds dataset, adapter, and DFA for the given env/spec.
2. Loads model checkpoint.
3. Runs environment rollouts.
4. Tracks DFA state during decoding if using constrained modes.
5. Computes final hard satisfaction on executed traces.
6. Optionally computes soft DeepDFA satisfaction.
7. Writes metrics and rollout diagnostics.

Decoding modes:

- `greedy`: choose best action/token directly.
- `beam`: search several likely sequences.
- `constrained_beam`: track automaton state, prune reject sinks, rerank by satisfaction.

Main output files:

- `metrics.json`: return, violation, satisfaction, runtime.
- `metrics.csv`: tabular metrics.
- `dfa_summary.json`: DFA structure and formula metadata.
- `automaton_rollout_stats.json`: accept counts, reject-sink entries, fallback counts, per-episode traces.

Files to study:

- [scripts/evaluate.py](/home/laptop-1029/Edu/Sapienza/nesy_rl_test/scripts/evaluate.py)
- [planning/eval_runtime.py](/home/laptop-1029/Edu/Sapienza/nesy_rl_test/planning/eval_runtime.py)
- [scripts/eval_dt.py](/home/laptop-1029/Edu/Sapienza/nesy_rl_test/scripts/eval_dt.py)

## 10. Experiment Workflow

Single smoke run:

```bash
python scripts/train.py --config configs/cb_smoke.yaml --run_dir runs/cb/study_single
```

Hyperparameter sweep:

```bash
python scripts/sweep.py \
  --env cb \
  --smoke \
  --spec avoid_single_bomb_22 \
  --use_safe_dfa \
  --alphas 0.0 0.4 0.8 \
  --temperatures 0.5 1.0 \
  --max_combinations 4 \
  --sweep_run_dir runs/cb/study_sweep
```

Specification suite:

```bash
python scripts/eval_suite.py \
  --env cb \
  --smoke \
  --suite v1 \
  --checkpoint runs/cb/study_single/cb_state_0.pt \
  --suite_run_dir runs/cb/study_suite
```

Matrix helpers:

- [scripts/cb_tt_matrix.py](/home/laptop-1029/Edu/Sapienza/nesy_rl_test/scripts/cb_tt_matrix.py)
- [scripts/cb_dt_matrix.py](/home/laptop-1029/Edu/Sapienza/nesy_rl_test/scripts/cb_dt_matrix.py)
- [scripts/dsrl_tt_matrix.py](/home/laptop-1029/Edu/Sapienza/nesy_rl_test/scripts/dsrl_tt_matrix.py)

Figure recreation:

- [recreate_cb_reach_while_safe_figure.py](/home/laptop-1029/Edu/Sapienza/nesy_rl_test/recreate_cb_reach_while_safe_figure.py)
- [recreate_cb_spec_comparison_figure.py](/home/laptop-1029/Edu/Sapienza/nesy_rl_test/recreate_cb_spec_comparison_figure.py)
- [recreate_cb_alpha_sweep_avoid_bombs_figure.py](/home/laptop-1029/Edu/Sapienza/nesy_rl_test/recreate_cb_alpha_sweep_avoid_bombs_figure.py)

## 11. Test Workflow

Run all tests:

```bash
pytest -q
```

Focused tests:

```bash
pytest -q tests/test_dfa.py
pytest -q tests/test_logic_loss.py
pytest -q tests/test_dt.py
pytest -q tests/test_eval_runtime.py
pytest -q tests/test_token_schema.py
```

What the tests protect:

- DFA parsing and exact symbol matching.
- Hard vs soft satisfaction agreement.
- Canonical END semantics.
- Dataset token schema stability.
- DT action/state windowing.
- Eval artifact schemas.
- Determinism and entrypoint behavior.

## 12. Study Order

Best learning path:

1. Read [README.md](/home/laptop-1029/Edu/Sapienza/nesy_rl_test/README.md).
2. Run one smoke TT experiment with `configs/cb_smoke.yaml`.
3. Inspect the run directory:
   - `metrics.json`
   - `dfa_summary.json`
   - `automaton_rollout_stats.json`
   - `dataset_artifacts/*.meta.json`
4. Read [logic/token_schema.py](/home/laptop-1029/Edu/Sapienza/nesy_rl_test/logic/token_schema.py).
5. Read [dfa_adapter.py](/home/laptop-1029/Edu/Sapienza/nesy_rl_test/dfa_adapter.py).
6. Read [logic_loss_tt.py](/home/laptop-1029/Edu/Sapienza/nesy_rl_test/logic_loss_tt.py).
7. Read [planning/eval_runtime.py](/home/laptop-1029/Edu/Sapienza/nesy_rl_test/planning/eval_runtime.py).
8. Run one DT smoke experiment and compare its loss to TT.
9. Read [planning/dt_runtime.py](/home/laptop-1029/Edu/Sapienza/nesy_rl_test/planning/dt_runtime.py), especially `compute_dt_logic_rollout_penalty`.
10. Run focused tests for DFA, DT, logic loss, and eval runtime.
11. Read `paper/notes/` to connect implementation details to paper language.
12. Recreate one figure and trace which run artifacts it consumes.

## 13. Mental Model

Think of the system as four layers:

1. Environment layer:
   Generates executed trajectories and reward/safety signals.

2. Token layer:
   Converts trajectories into fixed-width token rows that models can train on.

3. Logic layer:
   Converts human-readable LTLf specs into automata, then checks if trajectories satisfy them.

4. Learning/planning layer:
   Trains TT/DT models and optionally uses logic during training or decoding.

The most important conceptual distinction:

- TT logic loss uses DeepDFA satisfaction over sampled token sequences.
- DT logic loss currently uses short-horizon hazard probability from dynamics, not generic LTLf/DeepDFA.

## 14. Quick Command Set

```bash
cd /home/laptop-1029/Edu/Sapienza/nesy_rl_test
source .venv/bin/activate

pytest -q tests/test_dfa.py

python scripts/train.py --config configs/cb_smoke.yaml --run_dir runs/cb/study_tt_cb

python scripts/evaluate.py \
  --env cb \
  --smoke \
  --spec avoid_single_bomb_22 \
  --use_safe_dfa \
  --checkpoint runs/cb/study_tt_cb/cb_state_0.pt \
  --run_dir runs/cb/study_tt_cb_eval

python scripts/train_dt.py \
  --env frozenlake \
  --smoke \
  --run_dir runs/frozenlake/study_dt_fl
```
