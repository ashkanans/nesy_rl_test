"""
Symbol-based DFA logic loss for LTLGrid7Env (7x7 grid, the object layout changes from game to game), with the
DFA run by DeepDFA (suffix-prediction/DeepAutoma.py), the differentiable automaton of the earlier DFA losses.

The DT's action distribution is rolled forward on a joint belief over (agent cell, DFA state), the symbols come
from a frozen grounder, and the loss penalises a low DFA output after the rollout. Because of the layouts:

  * the state token is layout * 49 + cell, so the layout of a training window is read from the token and the
    grounder table of that layout is used (the grounder maps (layout, cell, action) to symbol probabilities);
  * the board dynamics do not depend on the layout, so the belief is only over the 49 cells and the move
    tensor P[a, cell, cell'] is shared by all layouts.

Every DFA transition and the final output go through DeepDFA.step_pi: its state is the probability of each DFA
state and its input the symbol probabilities, for each (cell, action) of the belief. The DeepDFA is built from the
task's DFA (initFromDfa) with two outputs: column 0 the DFA state's progress potential in [-1, 1], column 1 its
acceptance. The loss target picks the column: "potential" (1 - E[potential]) / 2, "accept" -log P(accept).

Grounders and their training are those of planning/ltlgrid_logic.py (train_grounder works on the 7x7 dataset
as is: its state input is the layout-carrying token).
"""

import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "suffix-prediction"))

from DeepAutoma import DeepDFA  # noqa: E402


def cell_transition_tensor(env) -> torch.Tensor:
    """Exact deterministic moves P[a, cell, cell'] on the (layout independent) board."""
    n, A = env.n_cells, env.action_space.n
    P = torch.zeros(A, n, n)
    for cell in range(n):
        pos = divmod(cell, env.size)
        for a in range(A):
            x, y = env.move(pos, a)
            P[a, cell, x * env.size + y] = 1.0
    return P


def build_deep_dfa(machine, potential_scale: float = 0.01, device=None) -> DeepDFA:
    """DeepDFA of a reward machine: transitions as in the DFA, outputs [potential * scale, accepting]."""
    q_count, n_symbols = machine.num_states, machine.num_symbols
    dfa = DeepDFA(n_symbols, q_count, 2)
    dfa.initFromDfa(machine.transitions, [q for q in range(q_count) if machine.acceptance[q]])
    with torch.no_grad():
        dfa.fin_matrix[:, 0] = torch.tensor(machine.potentials, dtype=torch.float32) * float(potential_scale)
    dfa.trans_prob = dfa.trans_prob.to(device)
    dfa.fin_matrix = dfa.fin_matrix.to(device)
    return dfa


class LTLGrid7SymbolLogic:
    def __init__(self, env, grounder, reward_scale: float, device=None, target: str = "potential"):
        if env.machine is None:
            raise ValueError("the symbol logic loss needs the task's reward machine (set task_id)")
        if target not in {"potential", "accept"}:
            raise ValueError("target must be 'potential' or 'accept'")
        self.env = env
        self.grounder = grounder
        self.target = target
        self.device = device
        self.num_cells = env.n_cells
        self.num_layouts = env.n_layouts
        self.num_dfa_states = env.machine.num_states
        self.P = cell_transition_tensor(env).to(device)  # [A, cell, cell']
        self.dfa = build_deep_dfa(env.machine, 0.01, device)

    def loss(self, logits, states, dfa_state_ids, attention_mask, horizon, temperature=1.0, eps=1e-6):
        if horizon <= 0:
            return logits.new_zeros(())
        B, T_, A = logits.shape
        n, Q = self.num_cells, self.num_dfa_states
        valid = attention_mask.float().reshape(-1)

        pi = torch.softmax(logits / max(float(temperature), 1e-6), dim=-1).reshape(-1, A)  # [N, A]
        N = pi.shape[0]
        flat = states.reshape(-1)
        layout, cell = flat // n, flat % n
        # symbol probabilities of the layout of every item, for every (cell, action): [N * cell * A, Y]
        g = self.grounder.table().reshape(self.num_layouts, n, A, -1)[layout.clamp(0, self.num_layouts - 1)]
        symbols = g.reshape(N * n * A, -1)

        s0 = torch.nn.functional.one_hot(cell, n).float()
        q0 = torch.nn.functional.one_hot(dfa_state_ids.clamp(0, Q - 1).reshape(-1), Q).float()
        J = s0.unsqueeze(-1) * q0.unsqueeze(1)  # joint belief [N, cell, Q]: starts at the known cell and DFA state

        for _ in range(int(horizon)):
            # DeepDFA step for every (cell, action): its DFA-state belief is the mass J[n, cell, :], its input the
            # symbol probabilities of that move
            state = J.unsqueeze(2).expand(N, n, A, Q).reshape(N * n * A, Q)
            next_state, _ = self.dfa.step_pi(state, symbols)
            u = next_state.reshape(N, n, A, Q) * pi[:, None, :, None]  # weight by the probability of the action
            J = torch.einsum("ast,nsar->ntr", self.P, u)  # move to the landing cell

        out = J.sum(dim=1) @ self.dfa.fin_matrix  # [N, 2]: expected potential, probability of acceptance
        if self.target == "potential":
            per_item = (1.0 - out[:, 0]) / 2.0
        else:
            per_item = -torch.log(out[:, 1].clamp(min=eps))
        return (per_item * valid).sum() / valid.sum().clamp(min=1.0)
