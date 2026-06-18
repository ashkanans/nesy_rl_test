# Empirical Evaluation Plan

This README extracts the environments, benchmark families, experiment setups, LTLf/STL specifications, metrics, and recommended evaluation stages from the empirical-evaluation planning document for the Neuro-Symbolic LTLf-injection project.

The goal is to make it easy to decide what to run next after the current ColourBomb experiments, and to understand which environments are low-effort additions versus high-value reviewer-facing benchmarks.

## Current Project Baseline

### ColourBomb

ColourBomb is the current internal gridworld environment used in this repository.

Current specifications:

- `avoid_bombs`
- `reach_goal_while_safe`

Current model families:

- Trajectory Transformer (TT)
- Decision Transformer (DT)

Current comparison style:

- No-logic baseline, usually `alpha = 0`
- Logic-injected model, usually `alpha > 0`
- Satisfaction rate and return/cost-style metrics depending on the script

## Environment Inventory

### Low-Effort / High-Value Discrete Environments

These environments are the easiest next additions because they are discrete, proposition-grounded, and compatible with DFA/LTLf-style constraints.

#### LetterWorld / LetterEnv

LetterWorld is a discrete `7x7` gridworld used by LTL2Action, DeepLTL, and GenZ-LTL.

Key properties:

- Grid size: `7x7`
- Propositions are letters
- 12 unique propositions/letters are mentioned in the document
- Each letter appears twice in the grid in the LTL2Action setup
- Cardinal actions
- Wraparound dynamics in the LTL2Action setup
- Timeout around 75 steps in the referenced setup

Useful for:

- Sequencing
- Response constraints
- Avoidance
- Ordering
- Testing whether logic loss helps with temporally extended constraints

Why it matters:

- It is a standard LTL-RL benchmark
- It provides a clean bridge from the current ColourBomb setting to accepted LTL-RL comparisons

#### OfficeWorld

OfficeWorld is a Reward Machine domain with sequential symbolic tasks.

Typical tasks:

- Coffee delivery
- Mail delivery
- Patrolling
- Ordered symbolic objectives

Useful for:

- DFA/reward-machine-compatible constraints
- Multi-stage ordering
- Showing that the method handles non-trivial symbolic task structure

#### CraftWorld

CraftWorld is a Minecraft-style Reward Machine domain.

Typical tasks:

- Collecting objects
- Crafting items
- Visiting locations in order
- Multi-stage symbolic plans

The document mentions 11 maps.

Useful for:

- Longer sequential tasks
- Ordering constraints
- Demonstrating that reward shaping alone is not enough without hand-engineered stages

#### WaterWorld

WaterWorld is another Reward Machine domain using colored objects/balls.

Useful for:

- Symbolic interactions
- Reward-machine style sequential objectives
- DFA-compatible task structure

#### Neural Reward Machines Minecraft-Like Gridworld

This is a Sapienza/Umili/Patrizi-related non-Markovian gridworld.

Typical propositions/tasks:

- Visit pickaxe cell
- Visit lava cell
- Visit door cell
- Complete non-Markovian navigation requirements

Why it matters:

- It comes from the same research ecosystem as DeepDFA and related DFA infrastructure
- It should be a natural fit for the existing LTLf/DFA pipeline

### Medium-Effort Continuous / Hybrid Environments

#### FlatWorld

FlatWorld is a continuous 2D environment with colored regions acting as propositions.

Key properties:

- Low-dimensional continuous control
- Colored regions define propositions
- Multiple propositions may be true simultaneously in some versions
- Used by DeepLTL and related work

Useful for:

- Bridging from discrete gridworlds to continuous dynamics
- Testing proposition grounding in continuous space
- Recurrence, persistence, and reach-avoid specifications

Example specification mentioned:

```text
G(F r and F y and F g) and G(not b)
```

Informally: always keep eventually visiting red, yellow, and green, while always avoiding blue.

#### ZoneEnv

ZoneEnv is a continuous Safety-Gym-style environment with a Point robot and colored zones as propositions.

Used by:

- LTL2Action
- DeepLTL
- GenZ-LTL

Useful for:

- Reach-avoid tasks
- Recurrence tasks
- Persistence tasks
- Continuous proposition grounding

Higher effort because it may require MuJoCo/Safety-Gym-style dependencies.

### High-Effort / High-Value Offline Safe RL Environments

#### DSRL

DSRL is a large offline safe RL benchmark suite.

Key properties:

- More than 75,000 trajectories
- 38 tasks
- Offline datasets exposed through `env.get_dataset()`
- Normalized reward and normalized cost metrics
- Used by offline safe RL methods such as CDT, SDT, and FISOR

Suites inside DSRL:

- SafetyGymnasium
- Bullet-Safety-Gym
- MetaDrive

Why it matters:

- Reviewers from offline safe RL will expect DSRL-style comparisons
- It enables comparison against CDT, SDT, FISOR, and OSRL baselines

#### DSRL SafetyGymnasium

SafetyGymnasium is MuJoCo-based.

Task families:

- Button
- Circle
- Goal
- Push
- Velocity tasks

Robot/task combinations mentioned:

- PointButton1 / PointButton2
- CarButton1 / CarButton2
- PointCircle1 / PointCircle2
- CarCircle1 / CarCircle2
- PointGoal1 / PointGoal2
- CarGoal1 / CarGoal2
- PointPush1 / PointPush2
- CarPush1 / CarPush2
- Ant velocity tasks
- HalfCheetah velocity tasks
- Swimmer velocity tasks

Potential LTLf overlays:

- Press buttons in order
- Reach goal while safe
- Push object while avoiding hazards
- Avoid gremlins/hazards
- Velocity constraints

#### DSRL Bullet-Safety-Gym

Bullet-Safety-Gym is PyBullet-based.

Task types:

- Run
- Circle

Robots:

- Ball
- Car
- Drone
- Ant

Total tasks mentioned:

- 8 tasks

Why it matters:

- SDT and CDT use this benchmark family
- It is the clearest high-value comparison target for offline safe RL reviewers

#### DSRL MetaDrive

MetaDrive provides self-driving environments with safety costs.

Useful for:

- Lane discipline
- Stop-then-go behavior
- Safe driving constraints
- Rich temporal specifications

Higher effort because propositions must be grounded from driving state.

### Other Related Settings

#### Grounding LTLf in Image Sequences / MNIST + DECLARE

This setting uses image sequences and DECLARE constraints.

Useful for:

- Showing connection to Sapienza/whitemech work
- Demonstrating DECLARE-style temporal logic in perception-heavy settings

This is more related background than an immediate offline RL benchmark.

## Competing Paper Experiment Setups

### SDT: Temporal Logic Specification-Conditioned Decision Transformer

Closest competitor to the DT part of this project.

Important distinction:

- SDT uses Signal Temporal Logic (STL), not LTLf
- It conditions the Decision Transformer on STL robustness values
- It does not use the same differentiable DFA/LTLf logic loss as this project

Environment family:

- DSRL Bullet-Safety-Gym

Evaluated tasks:

- Ant-Run
- Ball-Run
- Drone-Run
- Ball-Circle
- Car-Circle
- Drone-Circle

The document notes that SDT does not evaluate Car-Run or Ant-Circle.

Datasets:

- DSRL offline datasets

Tools:

- STLCG for robustness computation
- OSRL-based codebase

Metrics:

- Normalized cumulative reward
- Cumulative relabeled cost
- STL satisfaction rate

Evaluation setup:

- 3 seeds
- 20 trajectories

Baselines:

- CDT
- RvS-Rrho
- RvS-RC
- BC-Safe
- BCQ-Lag
- BEAR-Lag
- CPQ
- COptiDICE

#### SDT Run Specification

Informal meaning:

- Always stay between boundaries
- If velocity exceeds the threshold, slow down within the next 5 steps

This is an STL-style boundary and velocity constraint.

#### SDT Circle Specification

Informal meaning:

- If the agent enters an unsafe region, it must leave within the next 5 steps

This is also STL-style and continuous-state grounded.

### CDT: Constrained Decision Transformer

CDT is an offline safe RL baseline using Decision Transformer ideas with cost constraints.

Environment family:

- DSRL Bullet-Safety-Gym
- Broader DSRL suite

Metrics:

- Normalized reward
- Normalized cost

Focus:

- Offline safe RL
- Zero-shot adaptation to different cost thresholds

Baselines mentioned around CDT:

- BC-Safe
- DT-Cost
- CPQ
- COptiDICE
- BCQ-Lag
- BEAR-Lag
- PID-Lagrangian CPPO

### FISOR: Feasibility-Guided Safe Offline RL

FISOR is a diffusion-based safe offline RL method.

Environment family:

- Full DSRL benchmark

Includes:

- SafetyGymnasium Point/Car Button/Circle/Goal/Push difficulty 1/2
- Bullet-Safety-Gym
- MetaDrive

Focus:

- Hard safety constraints
- Feasibility-guided policy learning
- Normalized cost below threshold
- High return under safety constraints

### Tian et al.: RL Under Temporal Logic Constraints as Sequence Modeling

Directly relevant to the TT part of this project.

Method:

- Casts LTLf-constrained RL as a Trajectory Transformer sequence-modeling problem
- Designs dense reward functions for LTLf
- Uses sparse transformer attention for credit assignment

Why it matters:

- It is a baseline-of-record for TT + LTLf style work
- It should be cited even if environments differ

### LTL2Action

Environment family:

- LetterWorld
- ZoneEnv

Method components:

- LTL progression
- GNN specification encoder
- Spot for LTL handling

Task scale:

- Co-safe LTL tasks up to around `10^39` unique tasks are mentioned

Why it matters:

- Establishes LetterWorld and ZoneEnv as standard LTL-RL benchmarks

### DeepLTL

Environment family:

- LetterWorld
- ZoneEnv
- FlatWorld

Logic machinery:

- Buchi automata
- Handles finite- and infinite-horizon specifications

Metrics:

- Success rate
- Number of steps

Baselines:

- LTL2Action
- GCRL-LTL
- RAD-embeddings

Example task patterns:

- Reach-avoid
- Sequential reachability
- Recurrence
- Persistence

### GenZ-LTL

Environment family:

- LetterWorld / `LetterSafetyEnv-v0`
- ZoneEnv / `PointLltSafety2-v0`

Logic machinery:

- Rabinizer 4
- Decomposition into reach-avoid subgoals

Metrics:

- Satisfaction rate
- Violation rate
- Number of steps

Baselines:

- LTL2Action
- GCRL-LTL
- DeepLTL
- RAD-embeddings

## DSRL Benchmark Details

### Dataset Scale

The document describes DSRL as providing:

- More than 75,000 trajectories
- 38 tasks

Breakdown:

- SafetyGymnasium: 16 Goal/Button/Push/Circle tasks plus 5 velocity tasks
- Bullet-Safety-Gym: 8 Run/Circle tasks
- MetaDrive: 9 driving tasks

### Dataset API

DSRL exposes offline data with fields like:

- observations
- next_observations
- actions
- rewards
- costs
- terminals
- timeouts

### Normalization

Reward normalization:

```text
R = (R_pi - R_min) / (R_max - R_min)
```

Cost normalization:

```text
C = C_pi / kappa
```

Cost thresholds mentioned:

- MetaDrive and Bullet-Safety-Gym: `[10, 20, 40]`
- SafetyGymnasium: `[20, 40, 80]`

## LTLf / Temporal Specification Battery

These are the recommended specifications to add beyond the current two ColourBomb specs.

### 1. Avoidance / Safety

Formula:

```text
G(not bomb)
```

Meaning:

- The agent should never visit a bomb/unsafe state

Current project connection:

- Similar to `avoid_bombs`

Approximate DFA complexity:

- Very small, around 2 states

### 2. Reach While Safe

Formula:

```text
G(not bomb) and F(goal)
```

Meaning:

- Avoid bombs for the whole episode
- Eventually reach the goal

Current project connection:

- Similar to `reach_goal_while_safe`

Approximate DFA complexity:

- Small product automaton

### 3. Strict Sequencing

Formula:

```text
F(a and X F(b and X F c))
```

Meaning:

- Eventually do `a`
- After that, eventually do `b`
- After that, eventually do `c`

Why it matters:

- This is a headline experiment because plain Markovian reward shaping cannot express ordering cleanly without hand-engineered stage rewards

Approximate DFA complexity:

- About `k + 1` states for `k` ordered goals
- For three goals, around 4 states

### 4. Response

Formula:

```text
G(a -> F b)
```

Meaning:

- Whenever `a` happens, `b` must eventually happen later

DECLARE connection:

- Response

Approximate DFA complexity:

- Small, around 2 states for simple cases

### 5. Chain Response

Formula:

```text
G(a -> X b)
```

Meaning:

- Whenever `a` happens, `b` must happen immediately at the next step

DECLARE connection:

- ChainResponse

Approximate DFA complexity:

- Around 3 states including sink/failure state

### 6. Persistence / Stability

Formula:

```text
F G(safe)
```

Meaning:

- Eventually, from some point onward, safety should always hold

Important caveat:

- In finite-trace LTLf, `FG(p)` behaves differently from infinite-trace persistence
- Over finite traces it can collapse toward a last-step property depending on semantics
- This must be explained carefully in the paper

### 7. Strict-Until Reach-Avoid

Formula:

```text
not p1 U (p2 and (not p3 U p4))
```

Meaning:

- Avoid `p1` until reaching `p2`
- Then avoid `p3` until reaching `p4`

Why it matters:

- Direct comparison point with DeepLTL-style task patterns

### 8. Multi-Constraint DECLARE Conjunction

Example components:

```text
Precedence(a, b) = not b W a
NotCoexistence(a, b)
Response(a, b) = G(a -> F b)
```

Meaning:

- Multiple temporal process constraints hold at the same time

Why it matters:

- Shows the method can handle conjunctions of symbolic constraints, not only one simple safety formula

### 9. Optional LDLf Example

Example idea:

```text
p holds at every even step
```

Why it matters:

- Shows the DeepDFA-style representation can generalize beyond plain LTLf

## DECLARE Templates Mentioned

Common process-mining templates from the document:

- Response: `G(a -> F b)`
- Precedence: `not b W a`
- ChainResponse: `G(a -> X b)`
- ChainPrecedence: `G(X b -> a) and not b`
- AlternateResponse: `G(a -> X(not a U b))`
- Coexistence: `F a <-> F b`
- NotCoexistence: `not (F a and F b)`

## Recommended Evaluation Metrics

The document recommends adopting `rliable`-style evaluation.

### Main Metrics

Report:

- Normalized return / normalized reward
- Normalized cost, when safety costs exist
- Satisfaction rate
- Violation rate, where relevant
- Number of steps, where relevant

### rliable Metrics

Recommended aggregate statistics:

- IQM: interquartile mean
- Stratified bootstrap 95% confidence intervals
- Probability of improvement
- Performance profiles
- Optimality gap

### Seeds

Recommendation:

- Minimum: 5 seeds
- Better: 10 seeds if compute allows

The document notes that 3 seeds is common but weak for reviewer confidence.

## Recommended Experimental Stages

### Stage 1: Minimum Competitive Evaluation

Add:

- LetterWorld
- One Reward Machine domain: OfficeWorld or CraftWorld

Expand specifications to include:

- Avoidance
- Reach-while-safe
- Strict sequencing
- Response
- Chain-response
- Persistence
- Strict-until reach-avoid
- Multi-constraint DECLARE conjunction

Methodology:

- At least 5 seeds
- rliable metrics
- Satisfaction rate alongside return/cost
- At least one external baseline

Suggested baselines:

- Reward-shaping baseline
- PPO-Lagrangian-style baseline if online interaction is allowed
- CDT or another sequence-modeling baseline where practical
- Tian et al.-style TT + LTLf baseline if feasible

### Stage 2: Stronger KR / AAAI / IJCAI Evaluation

Add:

- Neural Reward Machines Minecraft-like gridworld
- and/or FlatWorld
- One LDLf specification

Also report:

- DFA size for every specification
- LTLf2DFA/MONA-generated automata where possible

Why this stage matters:

- It connects strongly to the Sapienza/DeepDFA ecosystem
- It demonstrates richer logic expressiveness

### Stage 3: Offline Safe RL Reviewer Evaluation

Add:

- DSRL Bullet-Safety-Gym Run/Circle

Compare against:

- SDT
- CDT
- Other OSRL baselines if feasible

Use metrics:

- DSRL normalized reward
- DSRL normalized cost
- Satisfaction rate

Use SDT-style propositions:

- Boundary safety
- Velocity threshold recovery
- Unsafe-region recovery within a fixed time window

Why this stage matters:

- This is the benchmark family expected by offline safe RL reviewers
- It positions the method against the closest Decision Transformer competitors

## Feasibility Ranking

### Tier A: Lowest Effort / High Value

1. LetterWorld
2. OfficeWorld or CraftWorld
3. Neural Reward Machines Minecraft-like gridworld

Why Tier A first:

- Discrete or symbolic
- Compatible with current DFA/LTLf logic pipeline
- Low proposition-grounding complexity
- Good reviewer value

### Tier B: Medium Effort

4. FlatWorld

Why Tier B:

- Continuous but low-dimensional
- Useful bridge from gridworld to continuous control
- Proposition grounding is harder but manageable

### Tier C: High Effort / High Value

5. DSRL Bullet-Safety-Gym Run/Circle
6. ZoneEnv
7. SafetyGymnasium / MetaDrive overlays

Why Tier C:

- High reviewer value
- Direct comparison to SDT/CDT/FISOR
- Requires careful continuous proposition grounding
- More dependency and engineering effort

## Practical Warnings

### SDT Is Not Exactly LTLf

SDT uses STL, not LTLf. It is still a close competitor because it injects temporal logic into Decision Transformers, but the semantics differ.

Positioning suggestion:

- Treat SDT as complementary and competitor-adjacent
- Emphasize that this project uses finite-trace automata/DFA-backed satisfaction

### Continuous Proposition Grounding Is Hard

For DSRL, ZoneEnv, and MetaDrive, the hard part is not only training the model. The hard part is defining propositions from continuous states.

Examples:

- `boundary_safe`
- `velocity_ok`
- `inside_unsafe_region`
- `button_pressed`
- `goal_reached`
- `lane_centered`
- `stopped_before_crossing`

Bad proposition grounding can make satisfaction results meaningless.

### Persistence Needs Careful Explanation

`FG(p)` in finite-trace LTLf is not the same as infinite-horizon persistence.

Before publication:

- Confirm the exact DFA using LTLf2DFA/MONA
- Explain finite-trace semantics explicitly

### License Checks Are Needed

Before redistributing derived datasets or code, check licenses for:

- LTL2Action
- Reward Machines
- DSRL
- OSRL
- SafetyGymnasium
- FlatWorld / whitemech repositories

The document says DSRL data is CC BY 4.0 and code is Apache 2.0, but other repositories still need confirmation.

## Suggested Next Action

The cleanest next empirical path is:

1. Finish the current ColourBomb DT/TT runs.
2. Add LetterWorld with the 8-spec battery.
3. Add OfficeWorld or CraftWorld.
4. Move evaluation aggregation to rliable.
5. Only then scale to FlatWorld or DSRL Bullet-Safety-Gym.

This path gives the strongest improvement in reviewer-facing evidence for the least engineering risk.
