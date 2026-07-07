# Study Roadmap: Mastering This NeSy Offline RL Project From Zero

This roadmap is for the repository:

```text
/home/laptop-1029/Edu/Sapienza/nesy_rl_test
```

The goal is to help you go from the fundamentals to the exact concepts used in the paper and code:

- reinforcement learning and offline RL
- safe RL and temporal constraints
- sequence modeling with Transformers
- Trajectory Transformer (TT)
- Decision Transformer (DT)
- LTLf, DFA, and finite-trace satisfaction
- DeepDFA and differentiable logic loss
- ColourBomb datasets and experiments
- neural dynamics models for pure offline DT logic loss
- evaluation, reviewer-facing experiments, and reproducibility

Read this roadmap like a curriculum. Each stage has:

- what to learn
- why it matters for this project
- what to read
- which repo files to inspect
- small exercises to check understanding

## 0. Big Picture In Simple Words

This project asks:

```text
Can we train offline RL sequence models so that their generated behavior follows temporal-logic rules?
```

The paper studies transformer-based offline RL policies under LTLf constraints.

The main idea is:

1. We have an offline dataset of trajectories.
2. A trajectory is a sequence of states, actions, rewards, and an END marker.
3. A temporal rule is written in LTLf, for example `G(not bomb)` or `G(not bomb) and F(goal)`.
4. The LTLf formula is compiled into a DFA.
5. The model is trained with normal supervised sequence loss plus a logic loss.
6. The logic loss encourages generated traces to be accepted by the DFA.
7. Evaluation checks return, goal rate, bomb rate, and DFA satisfaction rate.

In one sentence:

```text
We turn high-level temporal rules into automata, then use those automata to guide or measure offline Transformer policies.
```

## 1. Repo Map For Studying

Start with these documents:

- [README.md](/home/laptop-1029/Edu/Sapienza/nesy_rl_test/README.md): top-level repo commands and supported environments.
- [STUDY_WORKFLOW.md](/home/laptop-1029/Edu/Sapienza/nesy_rl_test/STUDY_WORKFLOW.md): existing workflow explanation for this repo.
- [docs/workflow/README.md](/home/laptop-1029/Edu/Sapienza/nesy_rl_test/docs/workflow/README.md): diagrams and rendering instructions.
- [docs/workflow/MODEL_ARCHITECTURES.md](/home/laptop-1029/Edu/Sapienza/nesy_rl_test/docs/workflow/MODEL_ARCHITECTURES.md): TT vs DT architecture notes.
- [docs/empirical_evaluation/README.md](/home/laptop-1029/Edu/Sapienza/nesy_rl_test/docs/empirical_evaluation/README.md): benchmark and experiment expansion plan.

Important paper files:

- [paper/main.tex](/home/laptop-1029/Edu/Sapienza/nesy_rl_test/paper/main.tex): main paper entry.
- [paper/1_introduction.tex](/home/laptop-1029/Edu/Sapienza/nesy_rl_test/paper/1_introduction.tex): motivation and contributions.
- [paper/2_background.tex](/home/laptop-1029/Edu/Sapienza/nesy_rl_test/paper/2_background.tex): LTLf, DFA, offline RL, sequence modeling.
- [paper/3_related_works.tex](/home/laptop-1029/Edu/Sapienza/nesy_rl_test/paper/3_related_works.tex): related work.
- [paper/4_method.tex](/home/laptop-1029/Edu/Sapienza/nesy_rl_test/paper/4_method.tex): logic loss and DeepDFA method.
- [paper/5_exp_setup.tex](/home/laptop-1029/Edu/Sapienza/nesy_rl_test/paper/5_exp_setup.tex): ColourBomb environment and experiment design.
- [paper/6_prelim_exp.tex](/home/laptop-1029/Edu/Sapienza/nesy_rl_test/paper/6_prelim_exp.tex): preliminary TT/DT results.

Important code areas:

- `envs/`: environments, especially ColourBomb.
- `datasets/`: offline dataset builders.
- `logic/`: token schema and symbolic mapping.
- `dfa_adapter.py`: token-to-symbol mapping and DFA satisfaction.
- `logic_loss_tt.py`: TT DeepDFA logic loss.
- `models/`: TT, DT, and neural dynamics models.
- `planning/`: rollout, DT runtime, and dynamics backends.
- `scripts/`: train/eval/matrix commands.
- `tests/`: sanity checks that explain expected behavior.

## 2. The Learning Ladder

Recommended order:

```text
Python/PyTorch basics
-> RL and MDPs
-> Offline RL
-> Transformers and autoregressive modeling
-> TT and DT
-> LTLf and DFA
-> DeepDFA and differentiable logic
-> Repo token schema and data pipeline
-> ColourBomb experiments
-> Neural dynamics for pure offline DT logic
-> Paper writing and reviewer defense
```

Do not start with DeepDFA. It will feel mysterious until RL, finite traces, and autoregressive models are clear.

## 3. Stage 1: Math And PyTorch Foundations

### What To Learn

- vectors, matrices, tensors
- probability distributions
- cross-entropy loss
- gradients and backpropagation
- embeddings
- train/validation split
- classification accuracy

### Why It Matters Here

This project repeatedly uses:

```text
logits -> softmax -> cross entropy -> gradient update
```

For DT:

```text
action logits -> action cross entropy against dataset action
```

For TT:

```text
token logits -> next-token cross entropy against dataset token
```

For neural dynamics:

```text
(state, action) -> next-state logits -> cross entropy against dataset next_state
```

### Textbooks / Resources

Recommended foundations:

- Ian Goodfellow, Yoshua Bengio, Aaron Courville, `Deep Learning`, especially chapters on feedforward networks, regularization, and sequence modeling.
- PyTorch official tutorials for tensors, autograd, datasets, dataloaders, and training loops.

### Repo Files To Read

- [models/dt_model.py](/home/laptop-1029/Edu/Sapienza/nesy_rl_test/models/dt_model.py)
- [models/dynamics_model.py](/home/laptop-1029/Edu/Sapienza/nesy_rl_test/models/dynamics_model.py)
- [planning/dynamics_runtime.py](/home/laptop-1029/Edu/Sapienza/nesy_rl_test/planning/dynamics_runtime.py)

### Exercises

1. Explain what a logit is.
2. Explain why cross entropy decreases when the model gives higher probability to the correct class.
3. Find where DT computes action cross entropy in [scripts/train_dt.py](/home/laptop-1029/Edu/Sapienza/nesy_rl_test/scripts/train_dt.py).
4. Find where neural dynamics computes next-state cross entropy in [planning/dynamics_runtime.py](/home/laptop-1029/Edu/Sapienza/nesy_rl_test/planning/dynamics_runtime.py).

## 4. Stage 2: Reinforcement Learning Basics

### What To Learn

- agent, environment, state, action, reward
- episode and trajectory
- Markov Decision Process, MDP
- transition function
- policy
- value and return
- exploration vs exploitation

### Why It Matters Here

The environment produces the chain:

```text
state_t -> agent chooses action_t -> environment returns state_{t+1}, reward_t
```

In ColourBomb:

```text
state = grid cell ID
action = up/right/down/left
reward = step penalty, goal reward, or bomb penalty
trajectory = sequence of visited cells and actions
```

### Textbook

From the bibliography:

- Sutton and Barto, `Reinforcement Learning: An Introduction`.

Suggested reading order:

1. Chapter 1: RL problem.
2. Chapter 3: finite MDPs.
3. Chapter 4: dynamic programming, only enough to understand transition models.
4. Chapter 5: Monte Carlo, only enough to understand trajectories and returns.

### Repo Files To Read

- [envs/colour_bomb.py](/home/laptop-1029/Edu/Sapienza/nesy_rl_test/envs/colour_bomb.py)
- [frozenlake_env.py](/home/laptop-1029/Edu/Sapienza/nesy_rl_test/frozenlake_env.py)
- [nrm_nav_env.py](/home/laptop-1029/Edu/Sapienza/nesy_rl_test/nrm_nav_env.py)

### Exercises

1. In ColourBomb, list all actions and their integer IDs.
2. Identify which cells terminate the episode.
3. Explain why a wall changes the transition: the action is chosen, but the state may stay the same.
4. Explain the difference between reward and LTLf satisfaction.

## 5. Stage 3: Offline RL

### What To Learn

- online RL vs offline RL
- behavior policy
- fixed dataset of trajectories
- distribution shift / out-of-distribution actions
- why offline RL cannot safely query the environment during training
- model-free vs model-based offline RL

### Why It Matters Here

In pure offline RL, training should use only the fixed dataset.

This became important in this project because DT logic loss needs to imagine what happens after an action. There are multiple options:

- `tabular_env`: uses true environment rules or transition table. This is useful for debugging but can be considered privileged information.
- `tabular_dataset`: estimates transitions by counting only dataset transitions.
- `neural_dataset`: trains a neural model only from dataset transitions.

The pure offline option is:

```text
neural_dataset or tabular_dataset
```

not:

```text
tabular_env
```

### Papers / Resources

From the bibliography:

- Prudencio et al., `A Survey on Offline Reinforcement Learning: Taxonomy, Review, and Open Problems`.
- Kumar et al., `Conservative Q-Learning for Offline Reinforcement Learning`.

Optional for context:

- D4RL paper and benchmark documentation.

### Repo Files To Read

- [datasets/cb_dataset.py](/home/laptop-1029/Edu/Sapienza/nesy_rl_test/datasets/cb_dataset.py)
- [datasets/dt_dataset.py](/home/laptop-1029/Edu/Sapienza/nesy_rl_test/datasets/dt_dataset.py)
- [planning/dynamics_runtime.py](/home/laptop-1029/Edu/Sapienza/nesy_rl_test/planning/dynamics_runtime.py)
- [scripts/analyze_neural_dynamics.py](/home/laptop-1029/Edu/Sapienza/nesy_rl_test/scripts/analyze_neural_dynamics.py)

### Exercises

1. Explain why using a true grid transition table may be considered not pure offline RL.
2. Explain why training a neural transition model from the offline dataset is more defensible.
3. Explain why the neural dynamics model can only be accurate on state-action pairs covered by the dataset.
4. Open a dynamics report and compare `coverage_ratio` with all-pair accuracy.

## 6. Stage 4: Transformers And Sequence Modeling

### What To Learn

- tokenization
- embeddings
- positional embeddings
- causal attention
- autoregressive prediction
- teacher forcing
- next-token prediction
- context length

### Why It Matters Here

The project treats RL trajectories as sequences.

For TT:

```text
state, action, reward, aux, state, action, reward, aux, ...
```

For DT:

```text
(state_t, previous_action_t, return_to_go_t, timestep_t) -> action_t
```

The causal mask means the model cannot look into the future positions while predicting the current target.

### Papers

From the bibliography:

- Janner et al., `Offline Reinforcement Learning as One Big Sequence Modeling Problem`.
- Chen et al., `Decision Transformer: Reinforcement Learning via Sequence Modeling`.

Helpful prerequisite:

- Vaswani et al., `Attention Is All You Need`.

### Repo Files To Read

- [models/tt_model.py](/home/laptop-1029/Edu/Sapienza/nesy_rl_test/models/tt_model.py)
- [models/dt_model.py](/home/laptop-1029/Edu/Sapienza/nesy_rl_test/models/dt_model.py)
- [trajectory-transformer/README.md](/home/laptop-1029/Edu/Sapienza/nesy_rl_test/trajectory-transformer/README.md)
- [decision-transformer/README.md](/home/laptop-1029/Edu/Sapienza/nesy_rl_test/decision-transformer/README.md)
- [docs/workflow/MODEL_ARCHITECTURES.md](/home/laptop-1029/Edu/Sapienza/nesy_rl_test/docs/workflow/MODEL_ARCHITECTURES.md)

### Exercises

1. Explain the difference between a token and a state.
2. Explain why TT predicts many token types, but DT predicts only actions.
3. Explain what teacher forcing means in this repo.
4. Draw the DT input at three timesteps by hand.

## 7. Stage 5: Trajectory Transformer In This Repo

### What To Learn

TT models the whole trajectory as one flattened token sequence.

Example with active schema width 4:

```text
s0, a0, r0, aux0, s1, a1, r1, aux1, ..., END
```

TT predicts the next token at each token position.

### Project-Specific Details

Input batch:

```text
X    : [B, T] token IDs
Y    : [B, T] next-token targets
mask : [B, T]
```

Supervised loss:

```text
cross_entropy(logits, Y)
```

Logic-regularized loss:

```text
total_loss = supervised_loss + alpha_logic * logic_loss
```

### Repo Files To Read

- [models/tt_model.py](/home/laptop-1029/Edu/Sapienza/nesy_rl_test/models/tt_model.py)
- [logic_loss_tt.py](/home/laptop-1029/Edu/Sapienza/nesy_rl_test/logic_loss_tt.py)
- [train_cb.py](/home/laptop-1029/Edu/Sapienza/nesy_rl_test/train_cb.py)
- [scripts/train.py](/home/laptop-1029/Edu/Sapienza/nesy_rl_test/scripts/train.py)

### Commands

Smoke train:

```bash
python scripts/train.py --config configs/cb_smoke.yaml --run_dir runs/cb/study_tt_smoke
```

Run tests related to TT logic:

```bash
python -m pytest tests/test_logic_loss.py tests/test_token_schema.py -q
```

### Exercises

1. Explain what `X` and `Y` are in TT.
2. Explain why `Y` is usually `X` shifted by one token.
3. Explain why TT can generate states but DT does not.
4. Trace one token through `token_ids_to_symbol_ids`.

## 8. Stage 6: Decision Transformer In This Repo

### What To Learn

DT is not a full trajectory generator in the same way TT is.

DT predicts actions from:

```text
state_t + previous_action_t + return_to_go_t + timestep_t
```

Target:

```text
action_t
```

The state is not generated by DT. During training, state comes from the offline dataset. During evaluation, state comes from the environment.

### Project-Specific Details

Input batch:

```text
states       : [B, C]
prev_actions : [B, C]
rtg          : [B, C]
timesteps    : [B, C]
targets      : [B, C]
mask         : [B, C]
```

Important shift:

```text
input action  = previous action
target action = current action
```

At the first timestep:

```text
previous action = pad action ID
```

Supervised loss:

```text
action_CE = cross_entropy(action_logits, target_actions)
```

Optional logic loss:

```text
loss = action_CE + alpha_logic * logic_loss
```

In this repo, DT logic loss is not the same as TT DeepDFA logic loss. DT logic uses action probabilities plus a dynamics backend to estimate hazard probability over possible next states.

### Repo Files To Read

- [models/dt_model.py](/home/laptop-1029/Edu/Sapienza/nesy_rl_test/models/dt_model.py)
- [datasets/dt_dataset.py](/home/laptop-1029/Edu/Sapienza/nesy_rl_test/datasets/dt_dataset.py)
- [scripts/train_dt.py](/home/laptop-1029/Edu/Sapienza/nesy_rl_test/scripts/train_dt.py)
- [planning/dt_runtime.py](/home/laptop-1029/Edu/Sapienza/nesy_rl_test/planning/dt_runtime.py)

### Commands

Smoke train DT:

```bash
python scripts/train_dt.py --env cb --smoke --run_dir runs/cb/study_dt_smoke
```

Run DT tests:

```bash
python -m pytest tests/test_dt.py tests/test_dt_logic.py tests/test_dt_dynamics.py -q
```

### Exercises

1. Given actions `[RIGHT, DOWN, DOWN]`, write the `prev_actions` sequence.
2. Explain why target action is not calculated by a formula. It is copied from the dataset.
3. Explain where state comes from during DT training.
4. Explain where state comes from during DT evaluation/runtime.

## 9. Stage 7: Temporal Logic, LTLf, And DFA

### What To Learn

- propositional logic
- atomic propositions
- finite trace
- LTLf operators:
  - `G`: globally / always
  - `F`: eventually
  - `X`: next
  - `U`: until
- DFA components:
  - states
  - initial state
  - transition function
  - accepting states
- why LTLf formulas can be compiled into DFAs

### Why It Matters Here

The paper uses LTLf formulas such as:

```text
G(not bomb)
G(not bomb) and F(goal)
```

The model does not directly understand those formulas. The formula is compiled into a DFA, and then trajectories are checked by running the DFA over symbols extracted from states/actions/tokens.

### Papers / Resources

From the bibliography:

- Pnueli, `The Temporal Logic of Programs`.
- De Giacomo and Vardi, `Linear Temporal Logic and Linear Dynamic Logic on Finite Traces`.
- Fuggitti, `LTLf2DFA`.

Useful background:

- Michael Sipser, `Introduction to the Theory of Computation`, chapters on regular languages and finite automata.

### Paper Sections To Read

- [paper/2_background.tex](/home/laptop-1029/Edu/Sapienza/nesy_rl_test/paper/2_background.tex), subsection `Linear Temporal Logic over finite traces`.
- [paper/4_method.tex](/home/laptop-1029/Edu/Sapienza/nesy_rl_test/paper/4_method.tex), automata-based satisfaction.

### Repo Files To Read

- [dfa_adapter.py](/home/laptop-1029/Edu/Sapienza/nesy_rl_test/dfa_adapter.py)
- [specs/cb_specs.py](/home/laptop-1029/Edu/Sapienza/nesy_rl_test/specs/cb_specs.py)
- [specs/frozenlake_specs.py](/home/laptop-1029/Edu/Sapienza/nesy_rl_test/specs/frozenlake_specs.py)
- [specs/nrm_nav_specs.py](/home/laptop-1029/Edu/Sapienza/nesy_rl_test/specs/nrm_nav_specs.py)

### Exercises

1. Explain why `G(not bomb)` is safety.
2. Explain why `F(goal)` is liveness.
3. Explain why `G(not bomb) and F(goal)` can fail in two ways.
4. Explain why satisfaction is checked on the whole trace, not one state.

## 10. Stage 8: Finite-Trace END/EOT Semantics

### What To Learn

The paper and code use an explicit end-of-trace marker:

```text
END or EOT
```

The canonical convention in the code is:

1. Consume all trace symbols before termination.
2. Consume exactly one END symbol.
3. Check DFA acceptance after consuming END.

### Why It Matters Here

A finite trace has valid semantic positions only inside the trace. If the trace has length `T`, positions are usually:

```text
0, 1, ..., T - 1
```

Index `T` is one position past the end. That is why entailment/satisfaction is not normally defined at `i = T`. The END marker is a technical marker consumed by the automaton, not a normal environment state unless explicitly defined as one.

### Repo Files To Read

- [paper/notes/method_logic_stack.md](/home/laptop-1029/Edu/Sapienza/nesy_rl_test/paper/notes/method_logic_stack.md)
- [logic/token_schema.py](/home/laptop-1029/Edu/Sapienza/nesy_rl_test/logic/token_schema.py)
- [dfa_adapter.py](/home/laptop-1029/Edu/Sapienza/nesy_rl_test/dfa_adapter.py)
- [tests/test_token_schema.py](/home/laptop-1029/Edu/Sapienza/nesy_rl_test/tests/test_token_schema.py)
- [tests/test_dfa.py](/home/laptop-1029/Edu/Sapienza/nesy_rl_test/tests/test_dfa.py)

### Exercises

1. Explain the difference between a terminal environment state and the END token.
2. Explain why missing END means incomplete trace in canonical evaluation.
3. Explain why multiple END markers should be normalized.
4. Answer the reviewer question: why is entailment not defined for `i = T`?

## 11. Stage 9: DeepDFA And Differentiable Logic Loss

### What To Learn

- one-hot vectors
- soft probability vectors
- Gumbel-Softmax
- Monte Carlo sampling
- acceptance probability
- negative log likelihood
- differentiable automata relaxation

### Why It Matters Here

A normal DFA consumes discrete symbols, but neural models output probabilities.

DeepDFA lets us feed soft/probabilistic symbols and get a differentiable satisfaction probability.

TT logic loss pipeline:

```text
logits
-> Gumbel-Softmax samples
-> soft token distributions
-> soft DFA symbol distributions
-> DeepDFA acceptance probability
-> logic_loss = -log(probability of satisfaction)
```

### Papers

From the bibliography:

- Umili et al., `DeepDFA: Automata Learning through Neural Probabilistic Relaxations`.
- Umili et al., `Enhancing Deep Sequence Generation with Logical Temporal Knowledge`.
- Mezini et al., `Neuro-Symbolic Predictive Process Monitoring`.

### Repo Files To Read

- [logic_loss_tt.py](/home/laptop-1029/Edu/Sapienza/nesy_rl_test/logic_loss_tt.py)
- [dfa_adapter.py](/home/laptop-1029/Edu/Sapienza/nesy_rl_test/dfa_adapter.py)
- [suffix-prediction/DeepAutoma.py](/home/laptop-1029/Edu/Sapienza/nesy_rl_test/suffix-prediction/DeepAutoma.py)

### Exercises

1. Explain why hard argmax blocks gradients.
2. Explain why Gumbel-Softmax helps training.
3. Explain what acceptance probability means.
4. Explain why `-log(p_accept)` is high when satisfaction probability is low.

## 12. Stage 10: ColourBomb Environment And Dataset

### What To Learn

ColourBomb is the main environment in the current paper.

It is a `9x9` grid with:

- `S`: start
- `B`: bomb/hazard terminal cells
- `P`, `Y`, `BLU`/`U`: goal terminal cells
- `#` / `W`: walls
- `G`: green cell, currently rendered but not used as a terminal goal in the code
- `.`: empty cells

Actions:

```text
0 = up
1 = right
2 = down
3 = left
```

Current LTLf specs:

```text
avoid_bombs              -> G(not bomb)
reach_goal_while_safe    -> G(not bomb) and F(goal)
```

### Important Code Detail

The code treats goal positions as:

```python
P + Y + BLU
```

The `G` cell appears in the grid/rendering, but it is not currently included in `goal_positions` unless the code is changed.

### Repo Files To Read

- [envs/colour_bomb.py](/home/laptop-1029/Edu/Sapienza/nesy_rl_test/envs/colour_bomb.py)
- [datasets/cb_dataset.py](/home/laptop-1029/Edu/Sapienza/nesy_rl_test/datasets/cb_dataset.py)
- [specs/cb_specs.py](/home/laptop-1029/Edu/Sapienza/nesy_rl_test/specs/cb_specs.py)
- [paper/5_exp_setup.tex](/home/laptop-1029/Edu/Sapienza/nesy_rl_test/paper/5_exp_setup.tex)

### Exercises

1. Draw the grid and mark bombs/goals/walls.
2. Explain the difference between wall, bomb, and goal.
3. Explain what `state_semantics=pre` means.
4. Explain what `state_semantics=post` means.
5. Explain why pre/post matters for dynamics training.

## 13. Stage 11: Token Schema And Data Flow

### What To Learn

The active transition schema uses width 4:

```text
[state, action, reward, aux_or_safety_cost]
```

For ColourBomb:

```text
state  = discrete grid state ID
action = action ID
reward = placeholder token, currently 0
aux    = placeholder token, currently 0
```

Each full episode ends with one END row.

### Repo Files To Read

- [logic/token_schema.py](/home/laptop-1029/Edu/Sapienza/nesy_rl_test/logic/token_schema.py)
- [datasets/cb_dataset.py](/home/laptop-1029/Edu/Sapienza/nesy_rl_test/datasets/cb_dataset.py)
- [datasets/dt_dataset.py](/home/laptop-1029/Edu/Sapienza/nesy_rl_test/datasets/dt_dataset.py)
- [paper/notes/data_representation.md](/home/laptop-1029/Edu/Sapienza/nesy_rl_test/paper/notes/data_representation.md)

### Exercises

1. Build a fake transition row by hand.
2. Explain why END row stores END in the state slot.
3. Explain how TT flattens rows into tokens.
4. Explain how DT strips END and builds windows.

## 14. Stage 12: Neural Dynamics Model For Pure Offline DT Logic

### What To Learn

DT outputs only actions. To calculate a logic loss that depends on future states, we need an estimate of next states.

Options:

```text
tabular_env      = true environment/rule table, privileged
tabular_dataset  = counts from offline dataset only
neural_dataset   = neural model trained from offline dataset only
```

The `neural_dataset` backend learns:

```text
(state_t, action_t) -> state_{t+1}
```

It is not an extra input to DT. It is used only inside the auxiliary logic-loss computation.

### Why It Matters

This is the cleanest answer to the offline-RL concern:

```text
We do not use the environment transition table for the logic loss. Instead, we learn an approximate dynamics model from the same offline dataset.
```

But it has one limitation:

```text
The model is reliable mainly on state-action pairs covered by the dataset.
```

### Repo Files To Read

- [models/dynamics_model.py](/home/laptop-1029/Edu/Sapienza/nesy_rl_test/models/dynamics_model.py)
- [planning/dynamics_runtime.py](/home/laptop-1029/Edu/Sapienza/nesy_rl_test/planning/dynamics_runtime.py)
- [scripts/train_dt.py](/home/laptop-1029/Edu/Sapienza/nesy_rl_test/scripts/train_dt.py)
- [scripts/analyze_neural_dynamics.py](/home/laptop-1029/Edu/Sapienza/nesy_rl_test/scripts/analyze_neural_dynamics.py)
- [tests/test_dt_dynamics.py](/home/laptop-1029/Edu/Sapienza/nesy_rl_test/tests/test_dt_dynamics.py)

### Exercises

1. Explain why learning dynamics from data is not the same as using an oracle transition table.
2. Explain why deterministic ColourBomb can still be learned perfectly on observed pairs.
3. Explain why all-pair accuracy is limited when coverage is low.
4. Inspect `dynamics_training_log.csv` from a run and explain train/validation accuracy.

## 15. Stage 13: Experiment Workflow

### What To Learn

Main metrics:

- return
- satisfaction rate
- goal rate
- bomb hit rate
- runtime/overhead
- dynamics accuracy for neural dynamics runs

Main experiment axes:

- model family: TT or DT
- spec: `avoid_bombs` or `reach_goal_while_safe`
- alpha / logic weight
- seed
- decoding mode
- dynamics backend
- dataset policy mix

### Repo Files To Read

- [scripts/train.py](/home/laptop-1029/Edu/Sapienza/nesy_rl_test/scripts/train.py)
- [scripts/train_dt.py](/home/laptop-1029/Edu/Sapienza/nesy_rl_test/scripts/train_dt.py)
- [scripts/cb_dt_matrix.py](/home/laptop-1029/Edu/Sapienza/nesy_rl_test/scripts/cb_dt_matrix.py)
- [scripts/eval_suite.py](/home/laptop-1029/Edu/Sapienza/nesy_rl_test/scripts/eval_suite.py)
- [tools/metrics_compare.py](/home/laptop-1029/Edu/Sapienza/nesy_rl_test/tools/metrics_compare.py)
- [paper/notes/eval_protocol.md](/home/laptop-1029/Edu/Sapienza/nesy_rl_test/paper/notes/eval_protocol.md)

### Useful Commands

Run full tests:

```bash
python -m pytest -q
```

Compile Python files:

```bash
python -m py_compile $(git ls-files '*.py')
```

Run DT dynamics tests:

```bash
python -m pytest tests/test_dt_dynamics.py -q
```

Analyze neural dynamics in an existing run:

```bash
python scripts/analyze_neural_dynamics.py runs/cb/YOUR_RUN_DIR
```

### Exercises

1. Explain each metric in the paper table.
2. Explain what alpha controls.
3. Explain why higher alpha can improve satisfaction but hurt goal-reaching.
4. Explain why multiple seeds are necessary.

## 16. Stage 14: Reading The Paper Like A Researcher

Read the paper in this order:

1. [paper/5_exp_setup.tex](/home/laptop-1029/Edu/Sapienza/nesy_rl_test/paper/5_exp_setup.tex)
   - Understand the environment first.
2. [paper/2_background.tex](/home/laptop-1029/Edu/Sapienza/nesy_rl_test/paper/2_background.tex)
   - Learn LTLf, DFA, offline RL, sequence modeling.
3. [paper/4_method.tex](/home/laptop-1029/Edu/Sapienza/nesy_rl_test/paper/4_method.tex)
   - Understand logic loss and DeepDFA.
4. [paper/6_prelim_exp.tex](/home/laptop-1029/Edu/Sapienza/nesy_rl_test/paper/6_prelim_exp.tex)
   - Understand what the results claim.
5. [paper/3_related_works.tex](/home/laptop-1029/Edu/Sapienza/nesy_rl_test/paper/3_related_works.tex)
   - Place the work in the literature.
6. [paper/1_introduction.tex](/home/laptop-1029/Edu/Sapienza/nesy_rl_test/paper/1_introduction.tex)
   - Rewrite the story in your own words.

For each section, answer:

1. What problem is this section solving?
2. What assumptions does it make?
3. Which code file implements it?
4. What could a reviewer criticize?
5. How would I defend it?

## 17. Core Reading List

### Reinforcement Learning

- Sutton and Barto, `Reinforcement Learning: An Introduction`.
- Prudencio et al., `A Survey on Offline Reinforcement Learning: Taxonomy, Review, and Open Problems`.
- Kumar et al., `Conservative Q-Learning for Offline Reinforcement Learning`.

### Safe RL And Constraints

- Kushwaha et al., `A Survey of Safe Reinforcement Learning and Constrained MDPs`.
- Le Court et al., `Probabilistic Shielding for Safe Reinforcement Learning`.

### Sequence Modeling For Offline RL

- Janner et al., `Offline Reinforcement Learning as One Big Sequence Modeling Problem`.
- Chen et al., `Decision Transformer: Reinforcement Learning via Sequence Modeling`.
- Vaswani et al., `Attention Is All You Need`.

### Temporal Logic And Automata

- Pnueli, `The Temporal Logic of Programs`.
- De Giacomo and Vardi, `Linear Temporal Logic and Linear Dynamic Logic on Finite Traces`.
- Sipser, `Introduction to the Theory of Computation`, finite automata chapters.
- Fuggitti, `LTLf2DFA`.

### Logic In RL

- Icarte et al., `Reward Machines: Exploiting Reward Function Structure in Reinforcement Learning`.
- Camacho et al., `LTL and Beyond: Formal Languages for Reward Function Specification in Reinforcement Learning`.
- De Giacomo et al., `Foundations for Restraining Bolts: Reinforcement Learning with LTLf/LDLf Restraining Specifications`.
- Toro Icarte et al., `Teaching Multiple Tasks to an RL Agent Using LTL`.
- Vaezipoor et al., `LTL2Action: Generalizing LTL Instructions for Multi-Task RL`.

### DeepDFA And Neuro-Symbolic Logic Loss

- Umili et al., `DeepDFA: Automata Learning through Neural Probabilistic Relaxations`.
- Umili et al., `Enhancing Deep Sequence Generation with Logical Temporal Knowledge`.
- Mezini et al., `Neuro-Symbolic Predictive Process Monitoring`.

## 18. Practical 8-Week Study Plan

### Week 1: RL Basics

Goal:

```text
Understand state, action, reward, policy, trajectory, MDP.
```

Read:

- Sutton and Barto chapters 1 and 3.
- [envs/colour_bomb.py](/home/laptop-1029/Edu/Sapienza/nesy_rl_test/envs/colour_bomb.py).

Do:

```bash
python -m pytest tests/test_envs.py -q
```

Deliverable:

- One-page explanation of ColourBomb as an MDP.

### Week 2: Offline RL

Goal:

```text
Understand why fixed datasets change the problem.
```

Read:

- Offline RL survey overview.
- [datasets/cb_dataset.py](/home/laptop-1029/Edu/Sapienza/nesy_rl_test/datasets/cb_dataset.py).

Do:

```bash
python -m pytest tests/test_datasets.py tests/test_dataset_artifact.py -q
```

Deliverable:

- Explain why `neural_dataset` is more defensible than `tabular_env` for pure offline RL.

### Week 3: Transformers

Goal:

```text
Understand tokens, embeddings, causal attention, next-token prediction.
```

Read:

- `Attention Is All You Need`, high-level first.
- TT and DT papers, introduction/method only.
- [docs/workflow/MODEL_ARCHITECTURES.md](/home/laptop-1029/Edu/Sapienza/nesy_rl_test/docs/workflow/MODEL_ARCHITECTURES.md).

Do:

```bash
python -m pytest tests/test_dt.py tests/test_dt_dataset_artifact.py -q
```

Deliverable:

- Draw TT vs DT input/output diagrams from memory.

### Week 4: LTLf And DFA

Goal:

```text
Understand G, F, U, finite traces, and automata acceptance.
```

Read:

- Paper background LTLf section.
- De Giacomo and Vardi LTLf paper intro/definitions.
- [dfa_adapter.py](/home/laptop-1029/Edu/Sapienza/nesy_rl_test/dfa_adapter.py).

Do:

```bash
python -m pytest tests/test_dfa.py tests/test_specs.py tests/test_token_schema.py -q
```

Deliverable:

- Explain `G(not bomb)` and `G(not bomb) and F(goal)` using DFA states.

### Week 5: TT Logic Loss

Goal:

```text
Understand differentiable satisfaction for generated token sequences.
```

Read:

- DeepDFA paper.
- Umili PMAI paper.
- [logic_loss_tt.py](/home/laptop-1029/Edu/Sapienza/nesy_rl_test/logic_loss_tt.py).

Do:

```bash
python -m pytest tests/test_logic_loss.py -q
```

Deliverable:

- Explain `logic_loss = -log(acceptance_probability)` in simple words.

### Week 6: DT Logic And Neural Dynamics

Goal:

```text
Understand why DT needs a dynamics backend to estimate future-state logic risk.
```

Read:

- DT paper method.
- [datasets/dt_dataset.py](/home/laptop-1029/Edu/Sapienza/nesy_rl_test/datasets/dt_dataset.py).
- [planning/dynamics_runtime.py](/home/laptop-1029/Edu/Sapienza/nesy_rl_test/planning/dynamics_runtime.py).

Do:

```bash
python -m pytest tests/test_dt_dynamics.py tests/test_dt_logic.py -q
```

Deliverable:

- Explain how `(state, action) -> next_state` is learned from offline data.

### Week 7: Experiments And Metrics

Goal:

```text
Understand what each experiment measures and how to reproduce it.
```

Read:

- [paper/5_exp_setup.tex](/home/laptop-1029/Edu/Sapienza/nesy_rl_test/paper/5_exp_setup.tex).
- [paper/6_prelim_exp.tex](/home/laptop-1029/Edu/Sapienza/nesy_rl_test/paper/6_prelim_exp.tex).
- [paper/notes/eval_protocol.md](/home/laptop-1029/Edu/Sapienza/nesy_rl_test/paper/notes/eval_protocol.md).

Do:

```bash
python scripts/train.py --config configs/cb_smoke.yaml --run_dir runs/cb/study_week7_tt
python scripts/train_dt.py --env cb --smoke --run_dir runs/cb/study_week7_dt
```

Deliverable:

- Explain return, satisfaction, goal rate, and bomb hit rate.

### Week 8: Paper Defense And Reviewer Questions

Goal:

```text
Be able to defend the method clearly.
```

Read:

- [paper/3_related_works.tex](/home/laptop-1029/Edu/Sapienza/nesy_rl_test/paper/3_related_works.tex).
- Reviewer comments if available.
- [docs/empirical_evaluation/README.md](/home/laptop-1029/Edu/Sapienza/nesy_rl_test/docs/empirical_evaluation/README.md).

Do:

- Prepare short answers for:
  - Why LTLf instead of reward shaping?
  - Why DFA?
  - Why DeepDFA?
  - Why offline RL is difficult here?
  - Why neural dynamics is not cheating if trained only from the dataset?
  - Why entailment is not defined at `i = T`?
  - Why DT outputs only actions but still receives states during training?

Deliverable:

- A two-minute oral explanation of the whole project.

## 19. Minimum Mastery Checklist

You are ready to discuss the project deeply when you can answer these without notes:

- What is offline RL?
- What is a trajectory?
- What is the difference between TT and DT?
- What does DT receive as input?
- Why does DT output only actions?
- Where do DT states come from during training?
- Where do DT states come from during evaluation?
- What is LTLf?
- What is a DFA?
- What does `G(not bomb)` mean?
- What does `F(goal)` mean?
- Why does satisfaction depend on the full trace?
- What is the END/EOT marker?
- Why is `i = T` outside the finite trace?
- What is DeepDFA?
- What is Gumbel-Softmax?
- Why is logic loss differentiable?
- What is the difference between `tabular_env`, `tabular_dataset`, and `neural_dataset`?
- Why is neural dynamics only as good as dataset coverage?
- What metrics are reported in the paper?

## 20. Suggested Order For Code Reading

Read in this exact order:

1. [envs/colour_bomb.py](/home/laptop-1029/Edu/Sapienza/nesy_rl_test/envs/colour_bomb.py)
2. [logic/token_schema.py](/home/laptop-1029/Edu/Sapienza/nesy_rl_test/logic/token_schema.py)
3. [datasets/cb_dataset.py](/home/laptop-1029/Edu/Sapienza/nesy_rl_test/datasets/cb_dataset.py)
4. [datasets/dt_dataset.py](/home/laptop-1029/Edu/Sapienza/nesy_rl_test/datasets/dt_dataset.py)
5. [models/dt_model.py](/home/laptop-1029/Edu/Sapienza/nesy_rl_test/models/dt_model.py)
6. [models/tt_model.py](/home/laptop-1029/Edu/Sapienza/nesy_rl_test/models/tt_model.py)
7. [dfa_adapter.py](/home/laptop-1029/Edu/Sapienza/nesy_rl_test/dfa_adapter.py)
8. [logic_loss_tt.py](/home/laptop-1029/Edu/Sapienza/nesy_rl_test/logic_loss_tt.py)
9. [planning/dynamics_runtime.py](/home/laptop-1029/Edu/Sapienza/nesy_rl_test/planning/dynamics_runtime.py)
10. [scripts/train.py](/home/laptop-1029/Edu/Sapienza/nesy_rl_test/scripts/train.py)
11. [scripts/train_dt.py](/home/laptop-1029/Edu/Sapienza/nesy_rl_test/scripts/train_dt.py)
12. [scripts/cb_dt_matrix.py](/home/laptop-1029/Edu/Sapienza/nesy_rl_test/scripts/cb_dt_matrix.py)

## 21. One-Sentence Summary For Each Core Concept

- Offline RL: learn from a fixed dataset without new training-time environment interaction.
- MDP: formal model of states, actions, transitions, and rewards.
- Trajectory: one episode sequence of states/actions/rewards.
- TT: predicts the next token in a flattened trajectory.
- DT: predicts the current action from state, previous action, return-to-go, and timestep.
- LTLf: temporal logic for finite traces.
- DFA: machine that accepts or rejects a trace according to a formula.
- DeepDFA: differentiable/probabilistic DFA relaxation.
- Logic loss: penalty when generated traces have low satisfaction probability.
- END/EOT: explicit marker that tells the DFA the finite trace is complete.
- Neural dynamics: learned model of next state from state/action using offline dataset transitions.

## 22. How To Use This Roadmap

Recommended daily routine:

1. Read one short concept section.
2. Open the linked repo file.
3. Write a tiny example by hand.
4. Run one related test or smoke command.
5. Explain the concept out loud in simple words.

If you cannot explain it simply, do not move on yet. That is not a failure; that is the exact signal that the concept needs one more pass.
