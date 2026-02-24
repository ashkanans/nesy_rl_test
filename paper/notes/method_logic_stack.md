# Method Logic Stack Notes (Milestone 1: Issues 1.1-1.4)

This note records the implementation-level logic semantics that should be reflected in the manuscript.

## 1) Canonical finite-trace termination convention

Final convention used across the codebase:

1. Consume all trace symbols before termination.
2. Consume exactly one explicit `END` symbol.
3. Check acceptance on the resulting DFA state.

Canonical strictness:

- Missing `END` in canonical evaluation means the trace is incomplete.
- Incomplete traces are treated as unsatisfied (`False` on hard path, `0.0` on soft path).
- Multiple `END` markers are normalized by truncating at first `END` and appending one canonical termination step.

## 2) Prior failure mode fixed in Milestone 1

Before unification, end handling was duplicated across dataset creation, adapter logic, and DFA/loss utilities.

Observed risk:

- traces evaluated without explicit termination in some paths,
- extra termination appended in others,
- disagreement between hard DFA checks and differentiable satisfaction scores,
- false unsatisfaction on traces that are trivially satisfiable under intended finite-trace semantics.

Milestone 1 fix:

- adapter-owned token schema and `end_token_id` are now authoritative,
- datasets derive `END` from adapter logic instead of recomputing locally,
- evaluation/loss paths use one canonical termination behavior.

## 3) Adapter interfaces (hard vs probabilistic)

Adapter APIs introduced and aligned:

- `token_ids_to_symbol_ids(tokens) -> [B, T]` integer symbol IDs (hard path)
- `token_probs_to_symbol_probs(probs) -> [B, T, S]` symbol probabilities (soft path)
- `check_sat_token_ids(tokens, dfa) -> [B]` bool satisfaction
- `check_sat_symbol_probs(symbol_probs, deep_dfa) -> [B]` acceptance probability

Design requirement:

- one-hot token distributions in the soft path should match hard-path results (tested with tolerance).

## 4) Dataset and window semantics

- Full episodes contain exactly one explicit `END` row.
- Sliding training windows may or may not include `END`.
- Segment-level satisfaction can be diagnostic, but full-episode evaluation is the default for meaningful finite-trace satisfaction rates.

## 5) DFA inspection support

Inspection utilities now export:

- summary JSON (state count, accepting states, sink states, symbol count),
- `.dot` graph,
- optional `.png` render when Graphviz binary is available.

If PNG export fails, DOT/summary generation still succeeds with warning.

Representative Milestone 1 smoke artifacts were generated under:

- `runs/dfa_inspect_smoke/dfa_summary.json`
- `runs/dfa_inspect_smoke/dfa.dot`

## 6) Manuscript mapping

The paper should now explicitly claim:

- a single finite-trace termination contract,
- unified tokenization-to-automaton semantics,
- dual hard/soft satisfaction interfaces for exact checks and differentiable training,
- reproducible DFA inspection outputs for debugging/audit.
