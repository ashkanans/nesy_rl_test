# Validation Notes (Issues 0.4 / 0.5)

This note captures what is validated by unit tests and CI after the testing/automation checkpoints.

## Test classes covered

- Environment correctness:
  - reset/start-state invariants for ColourBomb and NRM navigation environments.
- Dataset format correctness:
  - shape consistency for `(x, y, mask)`,
  - next-step shift invariant (`y[t]` matches `x[t+1]` at transition granularity).
- DFA mapping and automata checks:
  - adapter-to-symbol mapping via a manually constructed DFA acceptance case,
  - product DFA construction sanity.
- Satisfaction-related checks:
  - DFA acceptance on token traces through `batch_check_dfa_sat` in DFA tests.

## CI coverage scope

- Workflow file: `.github/workflows/tests.yml`
- Triggers:
  - `push`
  - `pull_request`
- CI steps:
  - repository checkout (recursive submodules),
  - dependency setup via `./scripts/setup.sh`,
  - test execution via `pytest -q`.

## Paper text impact

Use the following sentence (or equivalent) in Implementation Details:

We use unit tests and CI to validate environment wrappers, trajectory formatting, and automata-based satisfaction evaluation.

This should be referenced in the reproducibility appendix as a trust signal that reported results are generated from continuously tested code snapshots.
