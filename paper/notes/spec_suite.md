# Benchmark and Spec Cards (Milestone 3)

This note tracks benchmark-facing integration for Milestone 3 and maps each benchmark to propositions, specs, and current runtime status.

## FrozenLake benchmark card (Issue 3.1)

- Environment role:
  - controlled finite-state safety benchmark for correctness debugging.
  - intentionally small/discrete; not treated as the main large-scale benchmark.
- Action/state space:
  - actions: 4 (`up/right/down/left`)
  - map sizes supported: `4x4`, `8x8`
  - default config used in smoke: `4x4`, `is_slippery=False`.
- Hazard semantics:
  - hole states are unsafe.
  - goal states are accepting task outcomes.
- Proposition convention:
  - state bins use `s0_bin<state_id>`.
  - in `4x4`, default hole IDs are `{5, 7, 11, 12}`, goal ID is `{15}`.
- Runtime presets:
  - `avoid_holes`: `G(!(holes))`
  - `reach_goal`: `F(goal)`
  - `reach_goal_while_avoid_holes`: `G(!(holes)) & F(goal)`
  - optional bounded proxy: `bounded_reach_goal_while_avoid_holes`.
- Server smoke artifacts:
  - train: `runs/frozenlake/server_m3_smoke_train/metrics.json`
  - eval: `runs/frozenlake/server_m3_smoke_eval/metrics.json`
  - eval plots:
    - `runs/frozenlake/server_m3_smoke_eval/plots/metrics_bar.png`
    - `runs/frozenlake/server_m3_smoke_eval/plots/satisfaction_trend.png`
    - `runs/frozenlake/server_m3_smoke_eval/plots/return_vs_satisfaction.png`

## ColourBomb benchmark card (Issue 3.2)

- Environment role:
  - custom controlled safety environment with interpretable trajectories.
  - used to debug logic behavior and automaton-conditioned evaluation.
- Hazard semantics:
  - bomb cells are unsafe terminal hazards.
  - goal-color cells are successful terminal outcomes.
- Runtime presets:
  - `avoid_bombs` (invariant safety)
  - `reach_goal_while_safe` (reach-while-safe)
  - `memory_sequence_yellow` (weaker memory-style sequencing over existing state propositions).
- Standardized reported metrics:
  - `goal_rate`
  - `bomb_hit_rate`
  - `satisfaction_rate`
  - plus core metrics (`return_mean`, `violation_rate`, etc.).
- Server baseline artifacts (smoke):
  - summary:
    - `runs/cb/server_m3_baselines/baseline_metrics.json`
    - `runs/cb/server_m3_baselines/baseline_metrics.csv`
  - plots:
    - `runs/cb/server_m3_baselines/plots/metrics_bar.png`
    - `runs/cb/server_m3_baselines/plots/satisfaction_trend.png`
    - `runs/cb/server_m3_baselines/plots/return_vs_satisfaction.png`

## AntMaze benchmark card (Issue 3.3)

- Current status:
  - protocol runner and loader are implemented in-repo (`antmaze_eval.py`, `antmaze_dataset.py`).
  - full D4RL-backed runtime validation is pending Python environment upgrade on the server.
- Blocking constraint:
  - current server Python (`3.12`) is incompatible with `d4rl` package requirements.
- TODO for paper/reporting:
  - after Python downgrade/compatibility env setup, run `antmaze-umaze` protocol smoke with real D4RL dataset.
  - then add finalized AntMaze paragraph with concrete variant list and safety proposition definitions.

## Manuscript mapping for Milestone 3

- FrozenLake paragraph:
  - emphasize controlled safety-correctness role and proposition-level specs.
- ColourBomb paragraph:
  - emphasize interpretability and logic-debugging value.
- AntMaze paragraph:
  - include explicit TODO note in current draft; replace with finalized text once D4RL runtime is validated.
