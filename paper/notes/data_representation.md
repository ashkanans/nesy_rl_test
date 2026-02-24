# Data Representation Notes (Milestone 4: Issues 4.1-4.2)

This note records the canonical transition-token schema after the Milestone 4 refactor.

## 1) Canonical schema ownership

- Authoritative schema module: `logic/token_schema.py`.
- Schema is now a logic-stack contract shared by:
  - dataset generation (`datasets/*`),
  - token-to-symbol mapping (`dfa_adapter.py`),
  - DFA/DeepDFA evaluation and logic loss paths.
- End token ID is derived from the schema-driven bin counts, not recomputed ad hoc in datasets.

## 2) Field ordering and schema variants

Current per-environment schema IDs:

- `cb_v1`
- `nrm_nav_v1`
- `frozenlake_v1`
- `antmaze_v1` (defined for compatibility/protocol alignment)

Canonical field order is fixed by index:

- index `0`: `state`
- index `1`: `action`
- index `2`: `reward` (currently placeholder by default)
- index `3`: env-specific auxiliary field
  - `cb_v1`: `aux` (placeholder by default)
  - `nrm_nav_v1`: `safety_cost` (binary)
  - `frozenlake_v1`: `safety_cost` (binary)
  - `antmaze_v1`: `safety_cost` (binary)

Transition width is schema-specific (currently width `4` for the active schemas), with explicit schema IDs used instead of implicit assumptions.

## 3) END and padding behavior

Full-episode canonical serialization:

1. episode transitions are written as transition rows,
2. exactly one explicit END row is appended,
3. END row uses `state=end_token_id` with neutral fillers in other fields.

Segment/window behavior:

- Training windows are sampled from full episodes and may or may not include END.
- Canonical satisfaction evaluation remains full-trace oriented: missing END implies incomplete trace unless compatibility mode is explicitly enabled.

Padding behavior:

- Current dataset windows are fixed-size slices with masks; no additional PAD token is introduced in the canonical schema for Milestone 4.

## 4) TT/DT usage vs proposition extraction

- TT/DT consumes the full transition vector in canonical field order.
- Proposition extraction for DFA formulas is state-centric in current presets (e.g., `s0_binK` predicates from state bins), while action/reward/cost fields remain available for extended predicates.
- This separation keeps sequence-model input rich while preserving transparent proposition semantics for logic constraints.

## 5) Validation status

Milestone 4 tests cover:

- schema field-index stability,
- per-env dataset compliance with schema width/dtype/end-row contract,
- backward-compatible entrypoint behavior (`train_cb.py`) during migration to `scripts/train.py`.
