from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from datasets.cb_dataset import CBSequenceDataset
from datasets.dt_dataset import DTSequenceDataset
from datasets.frozenlake_dataset import FrozenLakeSequenceDataset
from planning.eval_runtime import write_json


@dataclass
class DTRolloutConfig:
    eval_num_episodes: int = 100
    eval_max_steps: int | None = None
    rtg_target: float = 1.0


@dataclass
class DTConstrainedConfig:
    dt_mode: str = "greedy"
    num_action_candidates: int = 4
    lookahead_horizon: int = 2
    lookahead_backend: str = "env"
    hard_prune_reject_sink: bool = True
    sat_rerank_weight: float = 2.0
    candidate_sampling: str = "topk"
    knn_k: int = 16
    knn_return_weight: float = 1.0
    knn_satisfaction_weight: float = 2.0


def apply_smoke_mode_dt(args):
    if not getattr(args, "smoke", False):
        return args
    if hasattr(args, "num_episodes"):
        args.num_episodes = min(int(args.num_episodes), 200)
    if hasattr(args, "max_steps"):
        args.max_steps = min(int(args.max_steps), 30)
    if hasattr(args, "epochs"):
        args.epochs = min(int(args.epochs), 1)
    if hasattr(args, "batch_size"):
        args.batch_size = min(int(args.batch_size), 32)
    if hasattr(args, "context_len"):
        args.context_len = min(int(args.context_len), 20)
    if hasattr(args, "n_layer"):
        args.n_layer = min(int(args.n_layer), 2)
    if hasattr(args, "n_head"):
        args.n_head = min(int(args.n_head), 2)
    if hasattr(args, "n_embd"):
        args.n_embd = min(int(args.n_embd), 64)
    if hasattr(args, "eval_num_episodes"):
        args.eval_num_episodes = min(int(args.eval_num_episodes), 16)
    if hasattr(args, "num_action_candidates"):
        args.num_action_candidates = min(int(args.num_action_candidates), 4)
    if hasattr(args, "lookahead_horizon"):
        args.lookahead_horizon = min(int(args.lookahead_horizon), 2)
    if hasattr(args, "knn_k"):
        args.knn_k = min(int(args.knn_k), 16)
    if getattr(args, "env", None) == "frozenlake" and hasattr(args, "policy_mix"):
        # Ensure smoke dataset has some successful trajectories for a non-degenerate DT signal.
        args.policy_mix = max(float(args.policy_mix), 1.0)
    return args


def build_dt_offline_source(args):
    if args.env == "cb":
        dataset = CBSequenceDataset(
            num_episodes=args.num_episodes,
            max_steps=args.max_steps,
            sequence_length=max(8, args.context_len * 4),
            stochastic=getattr(args, "stochastic", False),
            seed=args.seed,
        )
        return dataset, None
    if args.env == "frozenlake":
        dataset = FrozenLakeSequenceDataset(
            num_episodes=args.num_episodes,
            max_steps=args.max_steps,
            sequence_length=max(8, args.context_len * 4),
            seed=args.seed,
            map_size=args.frozenlake_map_size,
            is_slippery=args.frozenlake_is_slippery,
            policy_mix=args.policy_mix,
        )
        return dataset, None
    if args.env == "antmaze":
        reason = (
            "DT v1 in this milestone supports only discrete-action envs (cb/frozenlake). "
            "AntMaze is skipped."
        )
        return None, reason
    raise ValueError(f"Unsupported DT env '{args.env}'")


def build_dt_dataset(base_dataset, context_len: int):
    episode_rewards = getattr(base_dataset, "episode_rewards", None)
    if episode_rewards is None:
        episode_rewards = [np.zeros(ep.shape[0] - 1, dtype=np.float32) for ep in base_dataset.episodes_tokens]
    return DTSequenceDataset(
        episodes_tokens=base_dataset.episodes_tokens,
        episode_rewards=episode_rewards,
        context_len=context_len,
        num_actions=base_dataset.env.action_space.n,
        state_index=0,
        action_index=1,
    )


def build_tabular_dynamics(base_dataset):
    env = base_dataset.env
    num_states = int(env.observation_space.n)
    num_actions = int(env.action_space.n)

    # Prefer authoritative transition models when exposed by the environment.
    if hasattr(env, "_gym_env") and hasattr(env._gym_env.unwrapped, "P"):
        probs = np.zeros((num_actions, num_states, num_states), dtype=np.float32)
        for s in range(num_states):
            for a in range(num_actions):
                for prob, ns, _, _ in env._gym_env.unwrapped.P[s][a]:
                    probs[a, s, int(ns)] += float(prob)
        denom = probs.sum(axis=-1, keepdims=True)
        probs = probs / np.maximum(denom, 1e-12)
        return probs

    if hasattr(env, "grid") and hasattr(env, "ACTIONS") and hasattr(env, "_state_to_pos"):
        probs = np.zeros((num_actions, num_states, num_states), dtype=np.float32)
        for s in range(num_states):
            r, c = env._state_to_pos(int(s))
            for a in range(num_actions):
                dr, dc = env.ACTIONS.get(int(a), (0, 0))
                nr, nc = r + dr, c + dc
                if 0 <= nr < env.n_rows and 0 <= nc < env.n_cols and env.grid[nr][nc] != "#":
                    ns = int(env._pos_to_state((nr, nc)))
                else:
                    ns = int(s)
                probs[a, s, ns] = 1.0
        return probs

    counts = np.zeros((num_actions, num_states, num_states), dtype=np.float32)

    for ep in getattr(base_dataset, "episodes_tokens", []):
        if ep.shape[0] < 2:
            continue
        rows = ep[:-1]
        next_rows = ep[1:]
        for cur, nxt in zip(rows, next_rows):
            s = int(cur[0])
            a = int(cur[1])
            ns = int(nxt[0])
            if 0 <= s < num_states and 0 <= ns < num_states and 0 <= a < num_actions:
                counts[a, s, ns] += 1.0

    # Add tiny smoothing and normalize to probabilities.
    counts += 1e-6
    denom = counts.sum(axis=-1, keepdims=True)
    probs = counts / np.maximum(denom, 1e-12)
    return probs


def hazard_mask_for_env(env_name: str, env):
    n_states = int(env.observation_space.n)
    mask = np.zeros(n_states, dtype=np.float32)
    if env_name == "frozenlake":
        for s in range(n_states):
            mask[s] = 1.0 if env.is_hole_state(s) else 0.0
    elif env_name == "cb":
        for s in range(n_states):
            mask[s] = 1.0 if env.is_bomb_state(s) else 0.0
    elif env_name == "nrm_nav" and hasattr(env, "_state_to_pos"):
        for s in range(n_states):
            pos = env._state_to_pos(s)
            mask[s] = 1.0 if pos in getattr(env, "unsafe_positions", set()) else 0.0
    return mask


def compute_dt_logic_rollout_penalty(
    logits: torch.Tensor,
    states: torch.Tensor,
    attention_mask: torch.Tensor,
    transition_probs: torch.Tensor,
    hazard_mask: torch.Tensor,
    rollout_horizon: int,
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    Differentiable logic regularizer via learned dynamics rollouts.

    For each training token, compute policy-conditioned transition dynamics and
    penalize expected hazard occupancy over a short rollout horizon.
    """
    if rollout_horizon <= 0:
        return logits.new_zeros(())

    B, T, A = logits.shape
    S = int(transition_probs.shape[-1])
    valid = attention_mask.float().clamp(min=0.0, max=1.0)
    valid_count = valid.sum().clamp(min=1.0)

    temp = max(float(temperature), 1e-6)
    action_probs = torch.softmax(logits / temp, dim=-1).clamp(min=1e-8, max=1.0)
    action_probs = action_probs / action_probs.sum(dim=-1, keepdim=True)

    state_ids = states.clamp(min=0, max=S - 1)
    state_dist = F.one_hot(state_ids, num_classes=S).float()
    hazard = hazard_mask.view(1, 1, S)

    penalty = logits.new_zeros(())
    probs_flat = action_probs.reshape(-1, A)
    trans = transition_probs

    for _ in range(int(rollout_horizon)):
        # P_pi = sum_a pi(a) * P(a)
        p_pi_flat = torch.einsum("na,asr->nsr", probs_flat, trans)
        sd_flat = state_dist.reshape(-1, S).unsqueeze(1)
        next_sd = torch.bmm(sd_flat, p_pi_flat).squeeze(1).reshape(B, T, S)
        state_dist = next_sd
        hazard_prob = (state_dist * hazard).sum(dim=-1)
        penalty = penalty + (hazard_prob * valid).sum() / valid_count

    return penalty / float(rollout_horizon)


def build_knn_suffix_memory(base_dataset, env_name: str):
    """
    Build state-indexed suffix candidates for kNN continuation scoring.
    """
    env = base_dataset.env
    state_to_entries: dict[int, list[dict[str, float | int]]] = {}
    episodes = getattr(base_dataset, "episodes_tokens", [])
    rewards_all = getattr(base_dataset, "episode_rewards", None)

    for ep_idx, ep in enumerate(episodes):
        if ep.shape[0] < 2:
            continue
        rows = ep[:-1]
        T = rows.shape[0]
        rewards = None
        if rewards_all is not None and ep_idx < len(rewards_all):
            rewards = np.asarray(rewards_all[ep_idx], dtype=np.float32).reshape(-1)
            if len(rewards) != T:
                rewards = None
        if rewards is None:
            rewards = np.zeros(T, dtype=np.float32)

        ret_suffix = np.cumsum(rewards[::-1])[::-1]
        if env_name in {"frozenlake", "nrm_nav"} and rows.shape[1] > 3:
            hazard_now = rows[:, 3] > 0
        elif env_name == "cb":
            hazard_now = rewards <= float(getattr(env.cfg, "step_reward", -0.01) - 0.5)
        else:
            hazard_now = np.zeros(T, dtype=bool)
        hazard_suffix = np.zeros(T, dtype=bool)
        running_hazard = False
        for t in range(T - 1, -1, -1):
            running_hazard = bool(running_hazard or bool(hazard_now[t]))
            hazard_suffix[t] = running_hazard

        for t in range(T):
            s = int(rows[t, 0])
            a = int(rows[t, 1])
            entry = {
                "action": a,
                "return_proxy": float(ret_suffix[t]),
                "satisfaction_proxy": float(0.0 if hazard_suffix[t] else 1.0),
            }
            state_to_entries.setdefault(s, []).append(entry)

    known_states = sorted(state_to_entries.keys())
    return {
        "state_to_entries": state_to_entries,
        "known_states": known_states,
    }


def compute_default_rtg_target(base_dataset, quantile: float = 0.75) -> float:
    rewards = getattr(base_dataset, "episode_rewards", None)
    if not rewards:
        return 1.0
    returns = np.asarray([float(np.sum(r)) for r in rewards], dtype=np.float32)
    if len(returns) == 0:
        return 1.0
    return float(np.quantile(returns, quantile))


def _step_env(env, action: int):
    out = env.step(int(action))
    if len(out) == 5:
        obs, reward, terminated, truncated, info = out
        done = bool(terminated or truncated)
    else:
        obs, reward, done, info = out
    return int(obs), float(reward), bool(done), dict(info)


def _append_context(history, value, max_len: int):
    history.append(value)
    if len(history) > max_len:
        history.pop(0)


def _predict_action_log_probs(model, device, states, prev_actions, rtgs, timesteps):
    states_t = torch.tensor([states], dtype=torch.long, device=device)
    prev_actions_t = torch.tensor([prev_actions], dtype=torch.long, device=device)
    rtg_t = torch.tensor([rtgs], dtype=torch.float32, device=device)
    ts_t = torch.tensor([timesteps], dtype=torch.long, device=device)
    mask_t = torch.ones_like(states_t, dtype=torch.float32, device=device)
    with torch.no_grad():
        logits = model(states_t, prev_actions_t, rtg_t, ts_t, attention_mask=mask_t)
    return torch.log_softmax(logits[0, -1, :], dim=-1)


def _predict_action(model, device, states, prev_actions, rtgs, timesteps):
    log_probs = _predict_action_log_probs(model, device, states, prev_actions, rtgs, timesteps)
    return int(torch.argmax(log_probs).item())


def _transition_single_dfa(dfa, state: int, symbol_id: int) -> int:
    return int(dfa.transitions[state].get(symbol_id, state))


def _initial_dfa_state(raw_dfa):
    if isinstance(raw_dfa, (list, tuple)):
        return tuple(0 for _ in raw_dfa)
    return 0


def _advance_dfa_state(raw_dfa, state, symbol_id: int):
    if isinstance(raw_dfa, (list, tuple)):
        return tuple(_transition_single_dfa(d, state[i], symbol_id) for i, d in enumerate(raw_dfa))
    return _transition_single_dfa(raw_dfa, state, symbol_id)


def _advance_state_with_tokens(
    adapter, raw_dfa, state, token_ids_1d: torch.Tensor, token_offset: int = 0
):
    token_ids_1d = token_ids_1d.long().view(-1)
    positions = (
        torch.arange(token_ids_1d.shape[0], device=token_ids_1d.device) + int(token_offset)
    ) % int(adapter.transition_dim)
    mapper = adapter.pos_bin_to_sym_idx.to(token_ids_1d.device)[positions]
    symbol_ids = torch.gather(mapper, 1, token_ids_1d.unsqueeze(-1)).squeeze(-1)
    next_state = state
    for sym in symbol_ids.tolist():
        next_state = _advance_dfa_state(raw_dfa, next_state, int(sym))
    return next_state


def _is_accepting_state(raw_dfa, state) -> bool:
    if isinstance(raw_dfa, (list, tuple)):
        return all(bool(d.acceptance[state[i]]) for i, d in enumerate(raw_dfa))
    return bool(raw_dfa.acceptance[state])


def _reject_sink_states_single(dfa):
    num_symbols = len(dfa.dictionary_symbols)
    out = set()
    for s, transitions in dfa.transitions.items():
        self_loop_all = True
        for sym in range(num_symbols):
            nxt = transitions.get(sym, s)
            if nxt != s:
                self_loop_all = False
                break
        if self_loop_all and not bool(dfa.acceptance[s]):
            out.add(int(s))
    return out


def _reject_sink_checker(raw_dfa):
    if isinstance(raw_dfa, (list, tuple)):
        reject_sets = [_reject_sink_states_single(d) for d in raw_dfa]

        def checker(state):
            return any(state[i] in reject_sets[i] for i in range(len(reject_sets)))

        return checker, reject_sets

    reject_set = _reject_sink_states_single(raw_dfa)

    def checker(state):
        return state in reject_set

    return checker, reject_set


def _cost_token_from_terminal(env_name: str, terminal_type: str | None) -> int:
    if env_name == "frozenlake":
        return 1 if terminal_type == "H" else 0
    if env_name == "nrm_nav":
        return 1 if terminal_type == "X" else 0
    return 0


def _is_hazard_terminal(env_name: str, terminal_type: str | None) -> bool:
    if env_name == "frozenlake":
        return terminal_type == "H"
    if env_name == "cb":
        return terminal_type == "B"
    if env_name == "nrm_nav":
        return terminal_type == "X"
    return False


def _simulate_step_for_lookahead(env, env_name: str, obs: int, action: int, step_idx: int):
    action = int(action)
    obs = int(obs)
    next_obs = obs
    reward = 0.0
    done = False
    info: dict[str, Any] = {}

    max_steps = int(getattr(env.cfg, "max_steps", 100))
    if env_name == "frozenlake":
        # Use Gym transition table when slippery, otherwise deterministic grid move.
        if bool(getattr(env.cfg, "is_slippery", False)) and hasattr(env, "_gym_env"):
            transitions = env._gym_env.unwrapped.P[obs][action]
            prob, next_obs, reward, done = max(transitions, key=lambda t: (float(t[0]), int(t[1])))
            _ = prob
        else:
            drdc = {
                env.ACTION_LEFT: (0, -1),
                env.ACTION_DOWN: (1, 0),
                env.ACTION_RIGHT: (0, 1),
                env.ACTION_UP: (-1, 0),
            }
            dr, dc = drdc.get(action, (0, 0))
            r, c = env._state_to_pos(obs)
            nr = min(max(r + dr, 0), env.n_rows - 1)
            nc = min(max(c + dc, 0), env.n_cols - 1)
            next_obs = int(env._pos_to_state((nr, nc)))
            cell = env._cell_at_state(next_obs)
            done = cell in {"G", "H"}
            reward = 1.0 if cell == "G" else 0.0

        cell = env._cell_at_state(int(next_obs))
        if done:
            if cell == "G":
                info["terminal_type"] = "G"
            elif cell == "H":
                info["terminal_type"] = "H"
    elif env_name == "cb":
        r, c = env._state_to_pos(obs)
        dr, dc = env.ACTIONS.get(action, (0, 0))
        nr, nc = r + dr, c + dc
        if 0 <= nr < env.n_rows and 0 <= nc < env.n_cols and env.grid[nr][nc] != "#":
            r, c = nr, nc
        next_obs = int(env._pos_to_state((r, c)))
        cell = env._cell_type((r, c))
        reward = float(env.cfg.step_reward)
        if cell in {"P", "Y", "BLU"}:
            reward += float(env.cfg.goal_reward)
            done = True
            info["terminal_type"] = cell
        elif cell == "B":
            reward += float(env.cfg.bomb_reward)
            done = True
            info["terminal_type"] = "B"
    else:
        return None

    if (step_idx + 1) >= max_steps and not done:
        done = True
        info["truncated"] = True

    return int(next_obs), float(reward), bool(done), info


def _candidate_actions(log_probs: torch.Tensor, constrained_cfg: DTConstrainedConfig):
    n_actions = int(log_probs.shape[0])
    k = min(max(1, int(constrained_cfg.num_action_candidates)), n_actions)
    if constrained_cfg.candidate_sampling == "sample":
        probs = torch.softmax(log_probs, dim=-1)
        if int(torch.count_nonzero(probs > 0).item()) <= k:
            idx = torch.topk(log_probs, k=k).indices
        else:
            idx = torch.multinomial(probs, num_samples=k, replacement=False)
    else:
        idx = torch.topk(log_probs, k=k).indices
    return [int(i) for i in idx.tolist()]


def _select_dt_action(
    model,
    device: torch.device,
    env,
    env_name: str,
    obs: int,
    prev_action: int,
    step_idx: int,
    cum_reward: float,
    cfg: DTRolloutConfig,
    context_len: int,
    states_hist: list[int],
    prev_actions_hist: list[int],
    rtg_hist: list[float],
    t_hist: list[int],
    token_length: int,
    constrained_cfg: DTConstrainedConfig,
    adapter,
    raw_dfa,
    dfa_state,
    reject_checker,
    knn_memory=None,
):
    log_probs = _predict_action_log_probs(
        model=model,
        device=device,
        states=states_hist,
        prev_actions=prev_actions_hist,
        rtgs=rtg_hist,
        timesteps=t_hist,
    )
    greedy_action = int(torch.argmax(log_probs).item())
    mode = constrained_cfg.dt_mode
    if mode != "constrained":
        if mode != "knn":
            return greedy_action, {"mode": "greedy", "fallback": False, "pruned": 0}
        if knn_memory is None:
            return greedy_action, {
                "mode": "knn",
                "fallback": True,
                "fallback_reason": "missing_knn_memory",
                "pruned": 0,
            }
        state_to_entries = knn_memory["state_to_entries"]
        known_states = knn_memory["known_states"]
        candidates = list(state_to_entries.get(int(obs), []))
        if not candidates:
            for st in sorted(known_states, key=lambda s: abs(int(s) - int(obs))):
                candidates.extend(state_to_entries.get(int(st), []))
                if len(candidates) >= int(constrained_cfg.knn_k):
                    break
        if not candidates:
            return greedy_action, {
                "mode": "knn",
                "fallback": True,
                "fallback_reason": "empty_knn_candidates",
                "pruned": 0,
            }
        k = min(max(1, int(constrained_cfg.knn_k)), len(candidates))
        scored = []
        for entry in candidates:
            score = (
                float(constrained_cfg.knn_return_weight) * float(entry["return_proxy"])
                + float(constrained_cfg.knn_satisfaction_weight) * float(entry["satisfaction_proxy"])
            )
            scored.append((score, int(entry["action"])))
        scored.sort(key=lambda x: x[0], reverse=True)
        topk = scored[:k]
        by_action: dict[int, float] = {}
        for score, action in topk:
            by_action[action] = max(score, by_action.get(action, -1e18))
        action = max(by_action.items(), key=lambda kv: kv[1])[0]
        return int(action), {
            "mode": "knn",
            "fallback": False,
            "pruned": 0,
            "candidates": int(len(candidates)),
        }
    if constrained_cfg.lookahead_backend != "env":
        return greedy_action, {
            "mode": "greedy",
            "fallback": True,
            "fallback_reason": "lookahead_backend_not_supported",
            "pruned": 0,
        }
    if env_name not in {"cb", "frozenlake"}:
        return greedy_action, {
            "mode": "greedy",
            "fallback": True,
            "fallback_reason": "env_lookahead_not_supported",
            "pruned": 0,
        }
    if adapter is None or raw_dfa is None or dfa_state is None or reject_checker is None:
        return greedy_action, {
            "mode": "greedy",
            "fallback": True,
            "fallback_reason": "missing_dfa_components",
            "pruned": 0,
        }

    candidates = _candidate_actions(log_probs, constrained_cfg)
    horizon = max(1, int(constrained_cfg.lookahead_horizon))
    def _search(prune_reject: bool):
        best_action_local = None
        best_score_local = None
        pruned_local = 0

        for first_action in candidates:
            sim_obs = int(obs)
            sim_prev_action = int(prev_action)
            sim_step_idx = int(step_idx)
            sim_cum_reward = float(cum_reward)
            sim_states = list(states_hist)
            sim_prev_actions = list(prev_actions_hist)
            sim_rtg = list(rtg_hist)
            sim_ts = list(t_hist)
            sim_dfa_state = dfa_state
            sim_token_length = int(token_length)
            sim_done = False
            prune_candidate = False
            score = float(log_probs[first_action].item())

            for depth in range(horizon):
                if sim_done:
                    break
                if depth == 0:
                    action = int(first_action)
                else:
                    _append_context(sim_states, int(sim_obs), context_len)
                    _append_context(sim_prev_actions, int(sim_prev_action), context_len)
                    _append_context(sim_rtg, float(cfg.rtg_target - sim_cum_reward), context_len)
                    _append_context(sim_ts, int(sim_step_idx), context_len)
                    action = _predict_action(
                        model=model,
                        device=device,
                        states=sim_states,
                        prev_actions=sim_prev_actions,
                        rtgs=sim_rtg,
                        timesteps=sim_ts,
                    )

                sim_out = _simulate_step_for_lookahead(env, env_name, sim_obs, action, sim_step_idx)
                if sim_out is None:
                    return None, None, -1
                next_obs, reward, sim_done, info = sim_out
                sim_step_idx += 1
                sim_cum_reward += float(reward)
                score += float(reward)

                cost_token = _cost_token_from_terminal(env_name, info.get("terminal_type"))
                is_hazard = _is_hazard_terminal(env_name, info.get("terminal_type"))
                trans_tokens = torch.tensor(
                    [int(action), 0, int(cost_token), int(next_obs)],
                    dtype=torch.long,
                    device=device,
                )
                sim_dfa_state = _advance_state_with_tokens(
                    adapter,
                    raw_dfa,
                    sim_dfa_state,
                    trans_tokens,
                    token_offset=sim_token_length,
                )
                sim_token_length += int(trans_tokens.shape[0])
                if reject_checker(sim_dfa_state):
                    if prune_reject and constrained_cfg.hard_prune_reject_sink:
                        prune_candidate = True
                        break
                    score -= float(constrained_cfg.sat_rerank_weight)
                if is_hazard and constrained_cfg.hard_prune_reject_sink:
                    prune_candidate = True
                    break

                sim_prev_action = int(action)
                sim_obs = int(next_obs)

            if prune_candidate:
                pruned_local += 1
                continue

            sat_bonus = 1.0 if _is_accepting_state(raw_dfa, sim_dfa_state) else 0.0
            score += float(constrained_cfg.sat_rerank_weight) * sat_bonus
            if best_score_local is None or score > best_score_local:
                best_score_local = score
                best_action_local = int(first_action)

        return best_action_local, best_score_local, pruned_local

    best_action, _, pruned = _search(prune_reject=True)
    if pruned < 0:
        return greedy_action, {
            "mode": "greedy",
            "fallback": True,
            "fallback_reason": "lookahead_simulation_unavailable",
            "pruned": 0,
        }
    if best_action is None and constrained_cfg.hard_prune_reject_sink:
        best_action, _, pruned_relaxed = _search(prune_reject=False)
        if pruned_relaxed >= 0:
            pruned = pruned_relaxed

    if best_action is None:
        return greedy_action, {
            "mode": "constrained",
            "fallback": True,
            "fallback_reason": "all_candidates_pruned",
            "pruned": int(max(0, pruned)),
            "candidates": int(len(candidates)),
        }

    return int(best_action), {
        "mode": "constrained",
        "fallback": False,
        "pruned": int(pruned),
        "candidates": int(len(candidates)),
    }


def evaluate_dt_policy(
    model,
    env,
    env_name: str,
    seed: int,
    cfg: DTRolloutConfig,
    context_len: int,
    device: torch.device,
    adapter=None,
    raw_dfa=None,
    checkpoint_path: str | None = None,
    spec_name: str | None = None,
    return_rollout_stats: bool = False,
    constrained_cfg: DTConstrainedConfig | None = None,
    knn_memory=None,
) -> dict[str, Any] | tuple[dict[str, Any], dict[str, Any]]:
    model.eval()
    returns = []
    lengths = []
    goal_hits = 0
    hazard_hits = 0
    step_violations = 0
    step_count = 0
    violation_episodes = 0
    episode_sats = []
    soft_sats = []
    reject_sink_entries = 0
    fallback_decodes = 0
    fallback_reason_counts: dict[str, int] = {}
    episode_goal_hits = []
    episode_hazard_hits = []

    if constrained_cfg is None:
        constrained_cfg = DTConstrainedConfig()

    deep_dfa = None
    if adapter is not None and raw_dfa is not None and not isinstance(raw_dfa, (list, tuple)):
        try:
            deep_dfa = raw_dfa.return_deep_dfa().to(device)
        except Exception:
            deep_dfa = None
    reject_checker = None
    if adapter is not None and raw_dfa is not None:
        reject_checker, _ = _reject_sink_checker(raw_dfa)

    max_steps = cfg.eval_max_steps if cfg.eval_max_steps is not None else getattr(env.cfg, "max_steps", 100)
    pad_action = env.action_space.n

    for ep in range(int(cfg.eval_num_episodes)):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        total_r = 0.0
        step_idx = 0
        prev_action = pad_action
        cum_reward = 0.0
        hit_goal = False
        hit_hazard = False
        tokens = [int(obs)]

        states_hist: list[int] = []
        prev_actions_hist: list[int] = []
        rtg_hist: list[float] = []
        t_hist: list[int] = []
        dfa_state = None
        was_in_reject = False
        if adapter is not None and raw_dfa is not None:
            dfa_state = _initial_dfa_state(raw_dfa)
            init_token = torch.tensor([int(obs)], dtype=torch.long, device=device)
            dfa_state = _advance_state_with_tokens(
                adapter, raw_dfa, dfa_state, init_token, token_offset=0
            )
            was_in_reject = bool(reject_checker and reject_checker(dfa_state))

        while not done and step_idx < int(max_steps):
            _append_context(states_hist, int(obs), context_len)
            _append_context(prev_actions_hist, int(prev_action), context_len)
            rtg_now = float(cfg.rtg_target - cum_reward)
            _append_context(rtg_hist, rtg_now, context_len)
            _append_context(t_hist, int(step_idx), context_len)

            action, decode_info = _select_dt_action(
                model=model,
                device=device,
                env=env,
                env_name=env_name,
                obs=int(obs),
                prev_action=int(prev_action),
                step_idx=int(step_idx),
                cum_reward=float(cum_reward),
                cfg=cfg,
                context_len=context_len,
                states_hist=states_hist,
                prev_actions_hist=prev_actions_hist,
                rtg_hist=rtg_hist,
                t_hist=t_hist,
                token_length=len(tokens),
                constrained_cfg=constrained_cfg,
                adapter=adapter,
                raw_dfa=raw_dfa,
                dfa_state=dfa_state,
                reject_checker=reject_checker,
                knn_memory=knn_memory,
            )
            if decode_info.get("fallback"):
                fallback_decodes += 1
                reason = str(decode_info.get("fallback_reason", "unspecified"))
                fallback_reason_counts[reason] = fallback_reason_counts.get(reason, 0) + 1
            next_obs, reward, done, info = _step_env(env, action)
            total_r += reward
            cum_reward += reward
            prev_action = action
            obs = next_obs
            step_idx += 1
            step_count += 1

            terminal_type = info.get("terminal_type")
            if env_name == "frozenlake":
                if terminal_type == "G":
                    hit_goal = True
                if terminal_type == "H":
                    hit_hazard = True
                cost_token = 1 if terminal_type == "H" else 0
            elif env_name == "cb":
                if terminal_type in {"P", "Y", "BLU"}:
                    hit_goal = True
                if terminal_type == "B":
                    hit_hazard = True
                cost_token = 0
            elif env_name == "nrm_nav":
                if terminal_type == "G":
                    hit_goal = True
                if terminal_type == "X":
                    hit_hazard = True
                cost_token = 1 if terminal_type == "X" else 0
            else:
                cost_token = 0

            step_violations += int(cost_token > 0)
            trans_tokens = [int(action), 0, int(cost_token), int(next_obs)]
            token_offset = len(tokens)
            tokens.extend(trans_tokens)
            if dfa_state is not None:
                trans_tensor = torch.tensor(trans_tokens, dtype=torch.long, device=device)
                dfa_state = _advance_state_with_tokens(
                    adapter,
                    raw_dfa,
                    dfa_state,
                    trans_tensor,
                    token_offset=token_offset,
                )
                in_reject = bool(reject_checker and reject_checker(dfa_state))
                if in_reject and not was_in_reject:
                    reject_sink_entries += 1
                was_in_reject = in_reject

        returns.append(total_r)
        lengths.append(int(step_idx))
        goal_hits += int(hit_goal)
        hazard_hits += int(hit_hazard)
        violation_episodes += int(hit_hazard)
        episode_goal_hits.append(1.0 if hit_goal else 0.0)
        episode_hazard_hits.append(1.0 if hit_hazard else 0.0)

        if adapter is not None and raw_dfa is not None:
            tokens_with_end = tokens + [int(adapter.end_token_id), 0, 0, 0]
            token_tensor = torch.tensor(tokens_with_end, dtype=torch.long, device=device).view(1, -1)
            if isinstance(raw_dfa, (list, tuple)):
                sats = [adapter.check_sat_token_ids(token_tensor, d, mask_to_state_only=True) for d in raw_dfa]
                sat_val = bool(torch.stack(sats, dim=0).all())
            else:
                sat_val = bool(
                    adapter.check_sat_token_ids(token_tensor, raw_dfa, mask_to_state_only=True)[
                        0
                    ].item()
                )
            episode_sats.append(1.0 if sat_val else 0.0)

            if deep_dfa is not None:
                token_for_soft = adapter._apply_mask_to_state_only(token_tensor)  # noqa: SLF001
                token_probs = torch.nn.functional.one_hot(
                    token_for_soft, num_classes=adapter.num_token_ids
                ).float()
                sym_probs = adapter.token_probs_to_symbol_probs(token_probs)
                soft_sat = float(adapter.check_sat_symbol_probs(sym_probs, deep_dfa)[0].item())
                soft_sats.append(soft_sat)

    satisfaction_source = "dfa"
    if spec_name is not None and str(spec_name).startswith("avoid_") and episode_hazard_hits:
        if not episode_sats or float(np.mean(episode_sats)) == 0.0:
            episode_sats = [1.0 - float(x) for x in episode_hazard_hits]
            if not soft_sats:
                soft_sats = list(episode_sats)
            satisfaction_source = "hazard_proxy"

    n = max(1, int(cfg.eval_num_episodes))
    metrics = {
        "return_mean": float(np.mean(returns)) if returns else None,
        "return_std": float(np.std(returns)) if returns else None,
        "goal_rate": float(goal_hits / n),
        "hazard_hit_rate": float(hazard_hits / n),
        "violation_rate_episode": float(violation_episodes / n),
        "violation_rate": float(violation_episodes / n),
        "violation_rate_step": float(step_violations / step_count) if step_count > 0 else None,
        "num_episodes": int(cfg.eval_num_episodes),
        "satisfaction_rate": float(np.mean(episode_sats)) if episode_sats else None,
        "satisfaction_soft_mean": float(np.mean(soft_sats)) if soft_sats else None,
        "spec": spec_name,
        "decoding_mode": constrained_cfg.dt_mode,
        "beam_width": int(
            constrained_cfg.knn_k
            if constrained_cfg.dt_mode == "knn"
            else (
                constrained_cfg.num_action_candidates
                if constrained_cfg.dt_mode == "constrained"
                else 1
            )
        ),
        "model_type": "dt",
        "checkpoint_path": checkpoint_path,
        "satisfaction_source": satisfaction_source,
    }

    if not return_rollout_stats:
        return metrics

    rollout_stats = {
        "num_episodes": int(cfg.eval_num_episodes),
        "num_steps": int(step_count),
        "accept_count": int(sum(1 for x in episode_sats if x >= 0.5)),
        "violation_count": int(sum(1 for x in episode_sats if x < 0.5)),
        "reject_sink_entries": int(reject_sink_entries),
        "fallback_decodes": int(fallback_decodes),
        "fallback_reason_counts": fallback_reason_counts,
        "decoding_mode": constrained_cfg.dt_mode,
        "beam_width": int(
            constrained_cfg.knn_k
            if constrained_cfg.dt_mode == "knn"
            else (
                constrained_cfg.num_action_candidates
                if constrained_cfg.dt_mode == "constrained"
                else 1
            )
        ),
        "lookahead_horizon": int(constrained_cfg.lookahead_horizon),
        "lookahead_backend": constrained_cfg.lookahead_backend,
        "hard_prune_reject_sink": bool(constrained_cfg.hard_prune_reject_sink),
        "sat_rerank_weight": float(constrained_cfg.sat_rerank_weight),
        "knn_k": int(constrained_cfg.knn_k),
        "knn_return_weight": float(constrained_cfg.knn_return_weight),
        "knn_satisfaction_weight": float(constrained_cfg.knn_satisfaction_weight),
        "episode_returns": [float(x) for x in returns],
        "episode_lengths": lengths,
        "episode_satisfaction": episode_sats,
        "episode_goal_hits": episode_goal_hits,
        "episode_hazard_hits": episode_hazard_hits,
        "satisfaction_source": satisfaction_source,
    }
    return metrics, rollout_stats


def evaluate_random_policy(env, env_name: str, seed: int, eval_num_episodes: int, eval_max_steps: int | None):
    returns = []
    goal_hits = 0
    hazard_hits = 0
    max_steps = eval_max_steps if eval_max_steps is not None else getattr(env.cfg, "max_steps", 100)
    rng = np.random.RandomState(seed)

    for ep in range(int(eval_num_episodes)):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        total_r = 0.0
        steps = 0
        hit_goal = False
        hit_hazard = False
        while not done and steps < int(max_steps):
            action = int(rng.randint(env.action_space.n))
            obs, reward, done, info = _step_env(env, action)
            total_r += reward
            steps += 1
            terminal_type = info.get("terminal_type")
            if env_name == "frozenlake":
                if terminal_type == "G":
                    hit_goal = True
                if terminal_type == "H":
                    hit_hazard = True
            elif env_name == "cb":
                if terminal_type in {"P", "Y", "BLU"}:
                    hit_goal = True
                if terminal_type == "B":
                    hit_hazard = True
        returns.append(total_r)
        goal_hits += int(hit_goal)
        hazard_hits += int(hit_hazard)

    n = max(1, int(eval_num_episodes))
    return {
        "return_mean": float(np.mean(returns)) if returns else None,
        "return_std": float(np.std(returns)) if returns else None,
        "goal_rate": float(goal_hits / n),
        "hazard_hit_rate": float(hazard_hits / n),
        "violation_rate": float(hazard_hits / n),
    }


def dt_metrics_template(
    env: str,
    seed: int,
    rtg_target: float,
    runtime_sec: float,
    run_id: str,
    ts: str,
    checkpoint_path: str | None,
) -> dict[str, Any]:
    return {
        "return_mean": None,
        "return_std": None,
        "violation_rate": None,
        "satisfaction_rate": None,
        "runtime_sec": float(runtime_sec),
        "env": env,
        "spec": None,
        "seed": int(seed),
        "num_episodes": None,
        "satisfaction_soft_mean": None,
        "violation_rate_episode": None,
        "violation_rate_step": None,
        "goal_rate": None,
        "bomb_hit_rate": None,
        "hazard_hit_rate": None,
        "decoding_mode": None,
        "beam_width": None,
        "model_type": "dt",
        "checkpoint_path": checkpoint_path,
        "satisfaction_source": None,
        "run_id": run_id,
        "timestamp_utc": ts,
        "rtg_target": float(rtg_target),
        "action_loss": None,
        "logic_loss": None,
        "success_rate": None,
        "dataset_size": None,
        "context_len": None,
    }


def write_metrics_files(run_dir: str, metrics: dict[str, Any]) -> tuple[str, str]:
    os.makedirs(run_dir, exist_ok=True)
    json_path = os.path.join(run_dir, "metrics.json")
    csv_path = os.path.join(run_dir, "metrics.csv")
    write_json(json_path, metrics)
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics.keys()))
        writer.writeheader()
        writer.writerow(metrics)
    return json_path, csv_path


def write_skip_metrics(
    run_dir: str,
    env: str,
    seed: int,
    rtg_target: float,
    run_id: str,
    ts: str,
    reason: str,
):
    metrics = dt_metrics_template(
        env=env,
        seed=seed,
        rtg_target=rtg_target,
        runtime_sec=0.0,
        run_id=run_id,
        ts=ts,
        checkpoint_path=None,
    )
    metrics["skipped"] = True
    metrics["skip_reason"] = reason
    write_metrics_files(run_dir, metrics)
    write_json(
        os.path.join(run_dir, "dt_summary.json"),
        {
            "status": "skipped",
            "reason": reason,
            "env": env,
            "seed": int(seed),
            "run_id": run_id,
            "timestamp_utc": ts,
        },
    )
    return metrics
