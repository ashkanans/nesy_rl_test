# Evaluation Protocol Notes (Milestone 2: Issues 2.1-2.4)

This note records the training/evaluation reliability protocol for TT runs.

## 1) Metric definitions and trace semantics (Issue 2.1)

All evaluation outputs are written by `evaluate.py` and the shared runtime in `eval_runtime.py`.

Core metric schema keys (always present):

- `return_mean`
- `violation_rate`
- `satisfaction_rate`
- `runtime_sec`
- `env`
- `spec`
- `seed`

Additional standardized keys currently emitted:

- `return_std`
- `num_episodes`
- `satisfaction_soft_mean`
- `violation_rate_episode`
- `violation_rate_step`
- `decoding_mode`
- `beam_width`
- `model_type`
- `checkpoint_path`
- `run_id`
- `timestamp_utc`

Metric semantics:

- `return_mean`: average episodic return across executed rollout episodes.
- `violation_rate_episode`: fraction of episodes that are unsatisfied by DFA acceptance on finalized traces.
- `violation_rate_step`: fraction of unsafe steps according to environment unsafe-state predicates.
- `violation_rate`: alias to episode-level violation rate for stable compatibility.
- `satisfaction_rate`: hard DFA acceptance rate on finalized traces.
- `satisfaction_soft_mean`: mean soft acceptance probability from DeepDFA (when available).

Executed vs decoded traces:

- Action selection is based on decoded token prefixes from the policy model.
- Reported metrics are computed on executed environment rollouts (observed transitions after env stepping).
- For final hard/soft satisfaction evaluation, exactly one explicit end marker is appended to each executed trace before automaton evaluation.

Checkpoint policy:

- Default mode is strict checkpoint evaluation (`--checkpoint` required).
- Fallback mode (`--allow_train_fallback`) is explicit and emits warning because it is not a pure evaluation run.

## 2) Constrained decoding protocol (Issue 2.3)

Decoding options:

- `greedy`
- `beam`
- `constrained_beam`

Per-beam automaton tracking:

- each beam stores its current DFA state,
- DFA state is updated after each decoded transition token group,
- transitions use adapter token-to-symbol mapping.

Beam scoring:

- base score: cumulative log-probability of decoded tokens,
- constrained mode adds soft reranking term favoring beams in accepting states,
- controlled by `--sat_rerank_weight`.

Hard pruning:

- in constrained mode, `--hard_prune_reject_sink` removes beams entering rejecting sink states,
- if all beams are pruned, decoder falls back to greedy for that decision step and records fallback count in rollout diagnostics.

### Qualitative decoding example (saved artifacts)

Representative comparison with identical checkpoint and seed:

- Greedy run: `runs/nrm_nav/reduction_greedy/metrics.json`
  - `violation_rate = 1.0`, `satisfaction_rate = 0.0`
- Constrained beam run: `runs/nrm_nav/reduction_constrained/metrics.json`
  - `violation_rate = 0.0`, `satisfaction_rate = 1.0`

This example is used as a qualitative sanity check that automaton-aware decoding can alter rollout reliability under the same model weights.

## 3) Artifact schema and logging contract (Issue 2.4)

Default run layout:

- `runs/<env>/<UTC timestamp>/`

Override:

- `--run_dir` uses the exact path (no additional nested timestamp).

Standard per-run artifacts:

- `metrics.json`: run-level metric schema (core keys + optional keys).
- `dfa_summary.json`: static DFA/spec metadata and structure summary.
- `automaton_rollout_stats.json`: run-dependent rollout diagnostics (acceptance counts, sink events, decode fallbacks, episode returns/lengths).
- `cb_state_<epoch>.pt`: checkpoint snapshots for train runs.

Baseline sweep outputs:

- `baseline_metrics.json`
- `baseline_metrics.csv`

These are designed for direct parsing in analysis scripts and for consistent aggregation across environments and decoding settings.
