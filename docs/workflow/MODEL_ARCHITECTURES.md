# TT and DT Architecture Notes

This note explains exactly what each model receives, what happens inside, and what comes out.

## Trajectory Transformer

Entry files:

- [models/tt_model.py](/home/laptop-1029/Edu/Sapienza/nesy_rl_test/models/tt_model.py)
- [trajectory-transformer/trajectory/models/transformers.py](/home/laptop-1029/Edu/Sapienza/nesy_rl_test/trajectory-transformer/trajectory/models/transformers.py)
- [logic_loss_tt.py](/home/laptop-1029/Edu/Sapienza/nesy_rl_test/logic_loss_tt.py)

Input batch:

```text
X       : [B, T] integer token IDs
Y       : [B, T] next-token targets
mask    : [B, T] valid-token mask
```

The token stream is flattened. With the active schema, one transition is usually:

```text
state, action, reward, aux_or_safety_cost
```

So TT sees a sequence like:

```text
s0, a0, r0, c0, s1, a1, r1, c1, ...
```

Architecture:

1. `offset_tokens` makes field-specific token IDs.
   Example: token `5` in state position and token `5` in action position become different embedding IDs.
2. `tok_emb` embeds tokens.
3. `pos_emb` adds learned position information.
4. GPT blocks apply causal self-attention and MLP layers.
5. Final layer norm normalizes hidden states.
6. `pad_to_full_observation` aligns hidden states to complete transition groups.
7. `EinLinear` head predicts logits separately for each field position.

Output:

```text
logits : [B, T, vocab_size + 1]
loss   : supervised next-token CE, or mixed CE + logic loss during TT logic training
```

Supervised loss:

```text
cross_entropy(logits, Y), masked by mask
```

Optional field weighting:

```text
state positions  : weight 1
action positions : action_weight
reward positions : reward_weight
value/aux slot   : value_weight
```

Logic loss:

```text
total_loss = (1 - alpha) * supervised_loss + alpha * logic_loss
```

The TT logic loss samples differentiable token sequences from logits, maps token probabilities to DFA symbol probabilities, appends one END symbol, runs DeepDFA, and penalizes low satisfaction probability.

What TT generates:

- TT can generate the whole token stream: states, actions, reward/cost slots.
- During environment rollout, the selected action is executed in the real env, and final metrics are computed on executed transitions.

## Decision Transformer

Entry files:

- [models/dt_model.py](/home/laptop-1029/Edu/Sapienza/nesy_rl_test/models/dt_model.py)
- [datasets/dt_dataset.py](/home/laptop-1029/Edu/Sapienza/nesy_rl_test/datasets/dt_dataset.py)
- [scripts/train_dt.py](/home/laptop-1029/Edu/Sapienza/nesy_rl_test/scripts/train_dt.py)
- [planning/dt_runtime.py](/home/laptop-1029/Edu/Sapienza/nesy_rl_test/planning/dt_runtime.py)

Input batch:

```text
states       : [B, C]
prev_actions : [B, C]
rtg          : [B, C]
timesteps    : [B, C]
targets      : [B, C] current action labels
mask         : [B, C]
```

Here `C` is `context_len`.

Important shift:

```text
input action  = previous action
target action = current action
```

At the first timestep, previous action is a pad action ID equal to `num_actions`.

Architecture:

1. `state_emb(states)` gives state embeddings.
2. `prev_action_emb(prev_actions)` gives previous-action embeddings.
3. `rtg_emb(rtg.unsqueeze(-1))` projects scalar return-to-go.
4. `time_emb(timesteps)` gives timestep embeddings.
5. These four vectors are summed at each timestep.
6. Input layer norm is applied.
7. Padding positions are zeroed with `attention_mask`.
8. A causal TransformerEncoder processes the context.
9. A linear head predicts action logits.

Output:

```text
logits : [B, C, num_actions]
loss   : action cross entropy, optionally plus rollout logic penalty
```

Supervised loss:

```text
action_CE = cross_entropy(logits, targets, ignore_index=-100)
```

Optional DT logic loss:

```text
loss = action_CE + logic_alpha * logic_loss
```

The DT logic loss is not generic LTLf/DeepDFA. It does:

1. Convert action logits to action probabilities.
2. Start from known dataset states.
3. Roll state distributions forward with `transition_probs[a, s, s_next]`.
4. Use `hazard_mask[s]` to estimate unsafe-state probability.
5. Average hazard probability over context and rollout horizon.

DT dynamics backend options:

- `tabular_env`: oracle/model-based dynamics. This may use true environment transition tables or grid rules and is not pure offline.
- `tabular_dataset`: pure offline count-based dynamics estimated only from observed dataset transitions.
- `neural_dataset`: pure offline neural dynamics `p_theta(s_next | s, a)` trained only from consecutive dataset rows and token-schema state/action fields.

Important scope boundary:

- The neural dynamics model is not an extra input to DT.
- DT still receives only `state + previous action + RTG + timestep`.
- The learned dynamics model is used only to build `transition_probs` for the auxiliary DT logic loss.
- During training, the state input still comes from the offline dataset.

What DT generates:

- DT generates only actions.
- It does not generate next states.
- During rollout, the environment returns the next state, and that state becomes the next DT input.

## Main Difference

TT:

```text
input: flattened transition tokens
predicts: next token at every token position
can model: states, actions, reward/cost slots
logic training: DeepDFA satisfaction over sampled token sequences
```

DT:

```text
input: state + previous action + RTG + timestep
predicts: current action
can model: policy only, not environment state generation
logic training: optional short-horizon hazard rollout from known states
```
