"""
Symbol grounding and symbol-based DFA logic loss for the ltlgrid env.

The grid symbol emitted by a step (P, L, D, G, E) is a function of the pair (state, action):
it is the symbol of the cell the step ends in. A grounder maps that pair to a distribution over
symbols. Three grounders are available:

  oracle      exact table computed from the grid dynamics (the symbolic baseline)
  supervised  NN trained with cross-entropy on the symbol labels of the dataset
  weak        NN trained without symbol labels: the soft symbols are pushed through the task's DFA
              and the predicted shaped rewards are regressed onto the dataset rewards (Neural Reward
              Machine style grounding)

The logic loss rolls the DT's action distribution forward on the joint (cell, DFA state) belief,
using the grounder for the symbols, and penalizes a low expected DFA potential after the rollout.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from envs.ltlgrid_env import SYMBOLS

NUM_SYMBOLS = len(SYMBOLS)


def machine_tensors(machine, potential_scale: float, device=None):
    """Transition tensor T[symbol, q, q'] and potentials[q] * potential_scale."""
    T = torch.zeros(machine.num_symbols, machine.num_states, machine.num_states)
    for q, row in machine.transitions.items():
        for sym, nxt in row.items():
            T[sym, q, nxt] = 1.0
    pot = torch.tensor(machine.potentials, dtype=torch.float32) * float(potential_scale)
    return T.to(device), pot.to(device)


def grid_transition_tensor(env) -> torch.Tensor:
    """Exact deterministic dynamics P[a, s, s']."""
    S, A = env.observation_space.n, env.action_space.n
    P = torch.zeros(A, S, S)
    for s in range(S):
        pos = env.state_to_pos(s)
        for a in range(A):
            P[a, s, env.pos_to_state(env.move(pos, a))] = 1.0
    return P


def oracle_symbol_table(env) -> torch.Tensor:
    """Exact symbol table [S, A, Y] (one-hot)."""
    S, A = env.observation_space.n, env.action_space.n
    table = torch.zeros(S, A, NUM_SYMBOLS)
    if hasattr(env, "layouts"):  # LTLGrid7Env: the state token carries the layout
        for s in range(S):
            layout, pos = env.state_to_layout_pos(s)
            for a in range(A):
                table[s, a, env.symbol_of_pos(layout, env.move(pos, a))] = 1.0
        return table
    for s in range(S):
        pos = env.state_to_pos(s)
        for a in range(A):
            table[s, a, env.symbol_of_pos(env.move(pos, a))] = 1.0
    return table


class OracleGrounder(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.register_buffer("_table", oracle_symbol_table(env))
        self.use_rtg = False

    def forward(self, states, actions, rtg=None):
        return torch.log(self._table[states, actions].clamp(min=1e-8))

    def table(self) -> torch.Tensor:
        return self._table


class SymbolGrounder(nn.Module):
    """g(state, action[, rtg]) -> logits over symbols."""

    def __init__(self, num_states, num_actions, num_symbols=NUM_SYMBOLS, hidden=64, use_rtg=False):
        super().__init__()
        self.num_states, self.num_actions = int(num_states), int(num_actions)
        self.use_rtg = bool(use_rtg)
        self.state_emb = nn.Embedding(self.num_states, hidden)
        self.action_emb = nn.Embedding(self.num_actions, hidden)
        self.rtg_proj = nn.Linear(1, hidden) if self.use_rtg else None
        self.mlp = nn.Sequential(
            nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, num_symbols)
        )

    def forward(self, states, actions, rtg=None):
        h = self.state_emb(states) + self.action_emb(actions)
        if self.use_rtg:
            if rtg is None:
                raise ValueError("this grounder was built with use_rtg=True and needs rtg")
            h = h + self.rtg_proj(rtg.unsqueeze(-1))
        return self.mlp(h)

    def table(self) -> torch.Tensor:
        """Symbol probabilities for every (state, action) pair, [S, A, Y]."""
        if self.use_rtg:
            raise ValueError("an rtg-conditioned grounder has no (state, action) table")
        s = torch.arange(self.num_states, device=self.state_emb.weight.device)
        a = torch.arange(self.num_actions, device=self.state_emb.weight.device)
        ss, aa = torch.meshgrid(s, a, indexing="ij")
        return torch.softmax(self(ss, aa), dim=-1)


def grounder_arrays(dataset):
    """Flat (state, action, symbol, rtg) arrays over all steps of all episodes."""
    states, actions, symbols, rtgs = [], [], [], []
    for tr, sym, rew in zip(dataset.episode_transitions, dataset.episode_symbols, dataset.episode_rewards):
        states.append(tr[:, 0])
        actions.append(tr[:, 1])
        symbols.append(sym)
        rtgs.append(np.cumsum(rew[::-1])[::-1])
    return (
        np.concatenate(states).astype(np.int64),
        np.concatenate(actions).astype(np.int64),
        np.concatenate(symbols).astype(np.int64),
        np.concatenate(rtgs).astype(np.float32),
    )


@torch.no_grad()
def evaluate_grounder(grounder, dataset, device=None) -> dict:
    """Accuracy on the dataset steps, and on every (state, action) pair of the grid."""
    grounder.eval()
    s, a, y, rtg = (torch.from_numpy(x).to(device) for x in grounder_arrays(dataset))
    logits = grounder(s, a, rtg) if grounder.use_rtg else grounder(s, a)
    pred = logits.argmax(-1)
    stats = {"dataset_acc": float((pred == y).float().mean())}
    per_symbol = {}
    for i, name in enumerate(SYMBOLS):
        m = y == i
        if m.any():
            per_symbol[name] = float((pred[m] == i).float().mean())
    stats["per_symbol_acc"] = per_symbol
    stats["non_empty_acc"] = float((pred[y != SYMBOLS.index("E")] == y[y != SYMBOLS.index("E")]).float().mean())
    machine = dataset.env.machine
    if machine is not None:
        # Replay the predicted symbols through the DFA: the label-free success criterion. Weak
        # grounding can only identify symbols up to the formula's symmetries (e.g. P <-> L in
        # "visit both"), so reward-consistency, not label accuracy, is what it must achieve.
        pred_np = pred.cpu().numpy()
        exact, steps_ok, steps = 0, 0, 0
        offset = 0
        for rew in dataset.episode_rewards:
            q, ok = machine.initial_state, True
            for t, r_true in enumerate(rew):
                nxt = machine.next_state(q, int(pred_np[offset + t]))
                good = abs(machine.reward(q, nxt) * dataset.reward_scale - float(r_true)) < 1e-4
                ok &= good
                steps_ok += int(good)
                q = nxt
            exact += int(ok)
            steps += len(rew)
            offset += len(rew)
        stats["reward_replay_episode_acc"] = exact / max(1, len(dataset.episode_rewards))
        stats["reward_replay_step_acc"] = steps_ok / max(1, steps)
    if not grounder.use_rtg:
        oracle = oracle_symbol_table(dataset.env).to(device).argmax(-1)
        grid_pred = grounder.table().argmax(-1)
        stats["grid_acc"] = float((grid_pred == oracle).float().mean())
        seen = torch.zeros_like(oracle, dtype=torch.bool)
        seen[s, a] = True
        stats["num_seen_pairs"] = int(seen.sum())
        stats["unseen_pair_acc"] = (
            float((grid_pred[~seen] == oracle[~seen]).float().mean()) if (~seen).any() else None
        )
    return stats


def _padded_episodes(dataset, device):
    L = max(len(tr) for tr in dataset.episode_transitions)
    E = len(dataset.episode_transitions)
    S = torch.zeros(E, L, dtype=torch.long)
    A = torch.zeros(E, L, dtype=torch.long)
    R = torch.zeros(E, L)
    M = torch.zeros(E, L)
    for i, (tr, rew) in enumerate(zip(dataset.episode_transitions, dataset.episode_rewards)):
        n = len(tr)
        S[i, :n] = torch.from_numpy(tr[:, 0])
        A[i, :n] = torch.from_numpy(tr[:, 1])
        R[i, :n] = torch.from_numpy(rew)
        M[i, :n] = 1.0
    return S.to(device), A.to(device), R.to(device), M.to(device)


def train_grounder(
    dataset,
    mode: str = "supervised",
    use_rtg: bool = False,
    epochs: int = 300,
    lr: float = 1e-2,
    hidden: int = 64,
    seed: int = 0,
    device=None,
):
    """Return (grounder, stats). mode: oracle | supervised | weak."""
    env = dataset.env
    if mode == "oracle":
        grounder = OracleGrounder(env).to(device)
        return grounder, {"mode": "oracle", **evaluate_grounder(grounder, dataset, device)}
    if mode not in {"supervised", "weak"}:
        raise ValueError(f"unknown grounder mode '{mode}'")

    # Train inside a forked RNG scope: the grounder uses its own seed and leaves the global RNG
    # stream untouched, so a DT trained afterwards initialises exactly as it would without a
    # grounder (this keeps logic-on and logic-off runs paired).
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        grounder = SymbolGrounder(
            env.observation_space.n, env.action_space.n, hidden=hidden, use_rtg=use_rtg
        ).to(device)
        opt = torch.optim.Adam(grounder.parameters(), lr=lr)
        final_loss = float("nan")

        if mode == "supervised":
            s, a, y, rtg = (torch.from_numpy(x).to(device) for x in grounder_arrays(dataset))
            for _ in range(int(epochs)):
                logits = grounder(s, a, rtg) if use_rtg else grounder(s, a)
                loss = F.cross_entropy(logits, y)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
            final_loss = float(loss.detach())
        else:
            if use_rtg:
                raise ValueError("weak grounding needs a rtg-free grounder (rtg is derived from the same rewards)")
            if env.machine is None:
                raise ValueError("weak grounding needs the task's reward machine (set task_id)")
            T, pot = machine_tensors(env.machine, dataset.reward_scale, device)
            S, A, R, M = _padded_episodes(dataset, device)
            for _ in range(int(epochs)):
                probs = torch.softmax(grounder(S, A), dim=-1)  # [E, L, Y]
                belief = F.one_hot(
                    torch.full((S.shape[0],), env.machine.initial_state, device=device), env.machine.num_states
                ).float()
                loss = probs.new_zeros(())
                for t in range(S.shape[1]):
                    nxt = torch.einsum("ey,eq,yqr->er", probs[:, t], belief, T)
                    pred_r = (nxt - belief) @ pot
                    loss = loss + (((pred_r - R[:, t]) ** 2) * M[:, t]).sum()
                    belief = nxt
                loss = loss / M.sum()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
            final_loss = float(loss.detach())

    stats = {"mode": mode, "use_rtg": bool(use_rtg), "epochs": int(epochs), "final_loss": final_loss}
    stats.update(evaluate_grounder(grounder, dataset, device))
    return grounder, stats


class LTLGridSymbolLogic:
    """DT logic loss through grounded symbols and the task's DFA (see module docstring)."""

    def __init__(self, env, grounder, reward_scale: float, device=None, target: str = "potential"):
        if env.machine is None:
            raise ValueError("the symbol logic loss needs the task's reward machine (set task_id)")
        if target not in {"potential", "accept"}:
            raise ValueError("target must be 'potential' or 'accept'")
        self.env = env
        self.grounder = grounder
        self.target = target
        self.device = device
        self.num_states = env.observation_space.n
        self.num_dfa_states = env.machine.num_states
        self.P = grid_transition_tensor(env).to(device)  # [A, S, S']
        # potentials in [-1, 1] regardless of the reward scale, so the loss scale is task independent
        self.T, self.pot = machine_tensors(env.machine, 0.01, device)
        self.success = torch.tensor(
            [1.0 if env.machine.is_success(q) else 0.0 for q in range(self.num_dfa_states)], device=device
        )

    def loss(self, logits, states, dfa_state_ids, attention_mask, horizon, temperature=1.0, eps=1e-6):
        if horizon <= 0:
            return logits.new_zeros(())
        B, T_, A = logits.shape
        S, Q = self.num_states, self.num_dfa_states
        valid = attention_mask.float().reshape(-1)

        pi = torch.softmax(logits / max(float(temperature), 1e-6), dim=-1).reshape(-1, A)
        table = self.grounder.table()  # [S, A, Y]
        # K[a, s, y, s'] = P[a, s, s'] * g(y | s, a)
        K = torch.einsum("ast,say->asyt", self.P, table)

        s0 = F.one_hot(states.clamp(0, S - 1).reshape(-1), S).float()
        q0 = F.one_hot(dfa_state_ids.clamp(0, Q - 1).reshape(-1), Q).float()
        J = s0.unsqueeze(-1) * q0.unsqueeze(1)  # joint belief [N, S, Q]

        for _ in range(int(horizon)):
            after_dfa = torch.einsum("nsq,yqr->nsyr", J, self.T)
            J = torch.einsum("na,asyt,nsyr->ntr", pi, K, after_dfa)

        q_marg = J.sum(dim=1)  # [N, Q]
        if self.target == "potential":
            per_item = (1.0 - q_marg @ self.pot) / 2.0
        else:
            per_item = -torch.log((q_marg @ self.success).clamp(min=eps))
        return (per_item * valid).sum() / valid.sum().clamp(min=1.0)
