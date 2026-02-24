# Reproducibility Checkpoint Map (Milestone 0)

This note maps engineering checkpoints to concrete paper updates.

## Scope

- Milestone: `0` (Issues `0.1` to `0.5`)
- Purpose: record implementation details and reproducibility protocol.
- Not a method contribution: these updates strengthen transparency and repeatability.

## Global placement in paper

- Main text:
  - `Implementation Details` subsection (training/evaluation interface, quality controls).
  - `Experimental Protocol` subsection (how runs are launched/validated).
- Appendix:
  - `Reproducibility` section (environment, commands, artifacts, CI policy).

## Issue 0.1 checkpoint (README / runnable commands)

### What to capture in paper notes

- Environment summary:
  - Python: `3.12` validated in local setup.
  - Hardware mode: CPU-compatible baseline; CUDA optional.
  - Dependency install path: `./scripts/setup.sh`.
- Launch protocol:
  - Smoke validation: `pytest -q`.
  - Smoke train/eval command family for `cb` and `nrm_nav`.
  - AntMaze run path via `trajectory-transformer/`.
- Artifact contract:
  - `metrics.json`, `metrics.csv`, `satisfaction.png`.
  - Checkpoints (`cb_state_<epoch>.pt`).
  - Baseline aggregate file (`baseline_metrics.json`).

### Paper text impact

- Appendix, Reproducibility:
  - Add exact setup and execution commands.
  - Add output directory schema and required artifacts.
- Implementation Details:
  - Add one paragraph: all experiments are launched via unified CLI flags and validated with a smoke run before full experiments.

### Suggested paragraph (Implementation Details)

All experiments use a unified command-line interface with fixed seed control and standardized output artifacts. Before full training or evaluation, we execute a smoke validation run to verify dependency consistency, data-loading paths, and metric logging. Each run emits machine-readable files (`metrics.json` and task-specific summaries) to support deterministic post-processing and independent replication.

## Issue 0.2 checkpoint (split requirements + setup script)

### What to capture

- Dependency partitioning:
  - `requirements-core.txt`
  - `requirements-env.txt`
  - `requirements-tt.txt`
- Dependency groups for paper reporting:
  - Core logic stack:
    - `torch`, `numpy`, `ltlf2dfa`, `pythomata`, `sympy`, `typed-argument-parser`
  - TT-specific stack:
    - editable install of `trajectory-transformer` via `requirements-tt.txt`
  - Environment / benchmark-specific stack:
    - `gym`, `gymnasium`, `scikit-video`, `matplotlib`, `graphviz`
- Installation order and idempotency:
  - `./scripts/setup.sh` executes `core -> env -> tt`.
  - Script is safe to rerun.
- Version-sensitive notes to include explicitly:
  - `gym` and `gymnasium` are both installed; wrappers should report which API variant is active.
  - AntMaze reproduction depends on D4RL/MuJoCo availability in addition to Python dependencies.
  - CPU-only fallback should be stated for environments without CUDA.

### Paper text impact

- Appendix, Reproducibility:
  - Add dependency partition table and install-order rationale.
  - Add benchmark-to-dependency mapping so reviewers can run only relevant stacks.
  - Add version-sensitive caveats (`gym`/`gymnasium`, MuJoCo stack for AntMaze).
  - Add one line on idempotent setup for environment drift reduction.
- Implementation Details:
  - Add one sentence: "Dependencies are separated into core logic/modeling, transformer-specific, and benchmark-specific groups to reduce environment conflicts."

## Issue 0.3 checkpoint (formatting/lint baseline + pre-commit)

### What to capture

- Static quality gate:
  - Black + isort formatting.
  - Ruff lint baseline with explicit documented ignores.
  - Pre-commit integration (`pre-commit run --all-files`).

### Paper text impact

- Experimental Protocol:
  - Add code-quality gate statement prior to experiment release.
- Appendix, Reproducibility:
  - Add exact pre-commit command and policy that reported results correspond to lint-clean snapshots.

## Issue 0.4 checkpoint (pytest suite migration)

### What to capture

- Test structure:
  - `tests/test_envs.py`
  - `tests/test_datasets.py`
  - `tests/test_dfa.py`
- Test objective:
  - environment reset sanity,
  - dataset shape/shift invariants,
  - DFA adapter + product DFA behavior.

### Paper text impact

- Implementation Details:
  - Add unit-level checks used to verify environment/data/logic assumptions before running experiments.
- Appendix, Reproducibility:
  - Add `pytest -q` as required validation step.

## Issue 0.5 checkpoint (GitHub Actions CI)

### What to capture

- CI workflow file:
  - `.github/workflows/tests.yml`
- CI trigger policy:
  - run on push and pull request.
- CI actions:
  - setup, dependency install, `pytest -q`.

### Paper text impact

- Appendix, Reproducibility:
  - Add continuous-integration statement: test suite is automatically executed for all changes included in reported experiments.

## Suggested appendix skeleton to paste into manuscript

1. Environment and dependencies
2. Setup script and install order
3. Experiment launch commands (smoke and full)
4. Artifact schema and file-level outputs
5. Unit tests and CI enforcement
6. Seed and determinism controls

## Notes for next milestones

- Extend this map with benchmark-specific protocol cards (dataset version, seed grids, timeout policy, metric definitions).
- Keep one checkpoint block per merged issue to maintain traceability from code to manuscript claims.

## Server Runtime Evidence (Feb 24, 2026)

Downloaded run artifacts include a concrete container execution snapshot under `runs/repro/`.

- Timestamp: `runs/repro/run_utc.txt` shows `Tue Feb 24 11:49:15 UTC 2026`.
- Python runtime: `runs/repro/python_version.txt` shows `Python 3.10.12`.
- Torch package version: `runs/repro/pip_freeze_torch.txt` includes `torch==2.3.1+cu121`.
- CUDA visibility in runtime check: `runs/repro/torch_env.txt` reports:
  - `cuda_available True`
  - `cuda_version 12.1`
  - `cudnn_version 8902`
  - `device_count 2`
- Driver snapshot: `runs/repro/nvidia_smi.txt` records GPU visibility from inside container.

Notes:

- The recorded `torch_env.txt` line for torch version prints the `torch.version` module path instead of a version string due to the command used at run time.
- For future snapshots, prefer `print(torch.__version__)` to capture a human-readable torch version directly in `torch_env.txt`.
