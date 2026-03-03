import json
import os
import random
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import numpy as np
import torch

from dfa_utils import summarize_dfa


@dataclass
class DecodingConfig:
    mode: str = "greedy"
    beam_width: int = 4
    plan_horizon: int = 2
    sat_rerank_weight: float = 1.0
    hard_prune_reject_sink: bool = True
    target_shift: str = "token"


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def ensure_run_dir(env: str, run_dir: str | None = None, base_dir: str = "runs") -> tuple[str, str, str]:
    ts = utc_timestamp()
    if run_dir is None:
        run_dir = os.path.join(base_dir, env, ts)
    os.makedirs(run_dir, exist_ok=True)
    run_id = os.path.basename(os.path.normpath(run_dir))
    return run_dir, run_id, ts


def write_json(path: str, payload: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Best-effort deterministic configuration across training/evaluation entrypoints.
    try:
        torch.use_deterministic_algorithms(True)
    except Exception as exc:
        warnings.warn(
            f"Could not enable strict deterministic algorithms ({exc}); falling back to warn-only mode.",
            RuntimeWarning,
        )
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception as inner_exc:
            warnings.warn(
                f"Could not enable deterministic algorithms in warn-only mode ({inner_exc}).",
                RuntimeWarning,
            )
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def apply_smoke_mode(args):
    if not getattr(args, "smoke", False):
        return args

    # Caps tuned for <=2 minutes end-to-end on CPU in this repository.
    if hasattr(args, "num_episodes"):
        cap = 200 if getattr(args, "env", None) in {"frozenlake", "dsrl"} else 64
        args.num_episodes = min(int(args.num_episodes), cap)
    if hasattr(args, "max_steps"):
        args.max_steps = min(int(args.max_steps), 30)
    if hasattr(args, "epochs"):
        args.epochs = min(int(args.epochs), 1)
    if hasattr(args, "block_size"):
        args.block_size = min(int(args.block_size), 32)
    if hasattr(args, "batch_size"):
        args.batch_size = min(int(args.batch_size), 8)
    if hasattr(args, "n_layer"):
        args.n_layer = min(int(args.n_layer), 2)
    if hasattr(args, "n_head"):
        args.n_head = min(int(args.n_head), 2)
    if hasattr(args, "n_embd"):
        args.n_embd = min(int(args.n_embd), 64)
    if hasattr(args, "eval_num_episodes"):
        args.eval_num_episodes = min(int(args.eval_num_episodes), 16)
    if hasattr(args, "beam_width"):
        args.beam_width = min(int(args.beam_width), 4)
    if hasattr(args, "plan_horizon"):
        args.plan_horizon = min(int(args.plan_horizon), 2)
    if hasattr(args, "policy_mix") and getattr(args, "env", None) == "frozenlake":
        args.policy_mix = 0.0

    # Convenience only in smoke mode: inject a default spec if user did not provide one.
    has_manual_formula = getattr(args, "ltl_formula", None) is not None or getattr(
        args, "ltl_formulas", None
    ) is not None
    if getattr(args, "spec", None) is None and not has_manual_formula:
        if getattr(args, "env", None) == "cb":
            args.spec = "avoid_single_bomb_22"
        elif getattr(args, "env", None) == "nrm_nav":
            args.spec = "avoid_state_11"
        elif getattr(args, "env", None) == "frozenlake":
            args.spec = "reach_goal_while_avoid_holes"
        elif getattr(args, "env", None) == "dsrl":
            args.spec = "avoid_unsafe"

    return args


def spec_label_from_args(args) -> str | None:
    if getattr(args, "spec", None) is not None:
        return args.spec
    if getattr(args, "ltl_formulas", None) is not None:
        return " && ".join(args.ltl_formulas)
    if getattr(args, "ltl_formula", None) is not None:
        return args.ltl_formula
    return None


def summarize_dfa_bundle(raw_dfa, spec_name=None, formulas=None, dfa_mode=None):
    if isinstance(raw_dfa, (list, tuple)):
        components = [summarize_dfa(d) for d in raw_dfa]
        return {
            "spec": spec_name,
            "formulas": formulas,
            "dfa_mode": dfa_mode or "multi",
            "num_components": len(components),
            "components": components,
        }

    summary = summarize_dfa(raw_dfa)
    summary["spec"] = spec_name
    summary["formulas"] = formulas
    summary["dfa_mode"] = dfa_mode or "single"
    return summary


def _crop_history(
    history: torch.Tensor, block_size: int, transition_dim: int | None = None
) -> torch.Tensor:
    if history.shape[1] > block_size:
        if transition_dim is None or transition_dim <= 1:
            return history[:, -block_size:]

        # Keep token-phase alignment by dropping a multiple of transition_dim.
        # History lengths in this code path are not guaranteed to be exactly
        # transition-aligned after truncation if we simply take the last block.
        excess = int(history.shape[1] - block_size)
        drop = int(np.ceil(excess / transition_dim) * transition_dim)
        if drop >= history.shape[1]:
            return history[:, -block_size:]
        return history[:, drop:]
    return history


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


def _jsonable_reject_states(reject_states):
    if isinstance(reject_states, set):
        return sorted(int(s) for s in reject_states)
    if isinstance(reject_states, (list, tuple)):
        out = []
        for item in reject_states:
            if isinstance(item, set):
                out.append(sorted(int(s) for s in item))
            else:
                out.append(item)
        return out
    return reject_states


def _advance_state_with_tokens(adapter, raw_dfa, state, token_ids_1d: torch.Tensor):
    symbol_ids = adapter.token_ids_to_symbol_ids(token_ids_1d.unsqueeze(0))[0]
    next_state = state
    for sym in symbol_ids.tolist():
        next_state = _advance_dfa_state(raw_dfa, next_state, int(sym))
    return next_state


def _extract_action_log_probs(
    logits: torch.Tensor, n_actions: int, transition_dim: int, target_shift: str
) -> torch.Tensor:
    """
    Extract next-action log-probabilities from sequence logits.

    For token-shift training, the next action distribution is read at the last
    history position. For legacy transition-shift training, action_{t+1} is
    predicted at index T-transition_dim.
    """
    if logits.dim() != 3:
        raise ValueError("Expected logits shape [B, T, V].")
    seq_len = int(logits.shape[1])
    if seq_len <= 0:
        raise ValueError("Empty logits sequence.")

    if target_shift == "token":
        action_idx = seq_len - 1
    else:
        if seq_len <= transition_dim:
            action_idx = seq_len - 1
        else:
            action_idx = seq_len - int(transition_dim)

    action_logits = logits[:, action_idx, :n_actions]
    return torch.log_softmax(action_logits, dim=-1).squeeze(0)


def _decode_action(
    model,
    adapter,
    raw_dfa,
    history: torch.Tensor,
    dfa_state,
    n_actions: int,
    decoding_cfg: DecodingConfig,
):
    model_device = next(model.parameters()).device
    history = history.to(model_device)
    idx = _crop_history(history, model.block_size, transition_dim=adapter.transition_dim)
    logits, _ = model(idx)
    action_log_probs = _extract_action_log_probs(
        logits,
        n_actions=n_actions,
        transition_dim=adapter.transition_dim,
        target_shift=decoding_cfg.target_shift,
    )

    if decoding_cfg.mode == "greedy":
        action = int(torch.argmax(action_log_probs).item())
        return action, {"beam_size": 1, "fallback": False}

    reject_checker, _ = _reject_sink_checker(raw_dfa)
    init_beam = {
        "history": history,
        "dfa_state": dfa_state,
        "actions": [],
        "logprob": 0.0,
        "score": 0.0,
        "fallback": False,
    }
    beams = [init_beam]

    k = max(1, int(decoding_cfg.beam_width))
    horizon = max(1, int(decoding_cfg.plan_horizon))

    for _ in range(horizon):
        expanded = []
        for beam in beams:
            local_idx = _crop_history(
                beam["history"], model.block_size, transition_dim=adapter.transition_dim
            )
            local_logits, _ = model(local_idx)
            local_action_log_probs = _extract_action_log_probs(
                local_logits,
                n_actions=n_actions,
                transition_dim=adapter.transition_dim,
                target_shift=decoding_cfg.target_shift,
            )
            topk = torch.topk(local_action_log_probs, k=min(k, n_actions))

            for logp, action_tensor in zip(topk.values.tolist(), topk.indices.tolist()):
                action = int(action_tensor)
                next_history = beam["history"]
                step_tokens = [action]
                next_history = torch.cat(
                    [next_history, torch.tensor([[action]], dtype=torch.long, device=model_device)],
                    dim=1,
                )
                logprob = beam["logprob"] + float(logp)

                # Complete the transition tokens after action via greedy token decoding.
                for _ in range(adapter.transition_dim - 1):
                    token_idx = _crop_history(
                        next_history, model.block_size, transition_dim=adapter.transition_dim
                    )
                    token_logits, _ = model(token_idx)
                    logits_last = token_logits[:, -1, :]
                    token_log_probs = torch.log_softmax(logits_last, dim=-1)
                    token_id = int(torch.argmax(logits_last, dim=-1).item())
                    logprob += float(token_log_probs[0, token_id].item())
                    step_tokens.append(token_id)
                    next_history = torch.cat(
                        [
                            next_history,
                            torch.tensor([[token_id]], dtype=torch.long, device=model_device),
                        ],
                        dim=1,
                    )

                next_state = _advance_state_with_tokens(
                    adapter,
                    raw_dfa,
                    beam["dfa_state"],
                    torch.tensor(step_tokens, dtype=torch.long, device=model_device),
                )
                enters_reject = reject_checker(next_state)

                if (
                    decoding_cfg.mode == "constrained_beam"
                    and decoding_cfg.hard_prune_reject_sink
                    and enters_reject
                ):
                    continue

                score = logprob
                if decoding_cfg.mode == "constrained_beam":
                    accept_bonus = 1.0 if _is_accepting_state(raw_dfa, next_state) else 0.0
                    score += decoding_cfg.sat_rerank_weight * accept_bonus
                    if enters_reject:
                        score -= decoding_cfg.sat_rerank_weight

                expanded.append(
                    {
                        "history": next_history,
                        "dfa_state": next_state,
                        "actions": beam["actions"] + [action],
                        "logprob": logprob,
                        "score": score,
                        "fallback": False,
                    }
                )

        if not expanded:
            action = int(torch.argmax(action_log_probs).item())
            return action, {"beam_size": 0, "fallback": True}

        expanded.sort(key=lambda b: b["score"], reverse=True)
        beams = expanded[:k]

    best = beams[0]
    if not best["actions"]:
        action = int(torch.argmax(action_log_probs).item())
        return action, {"beam_size": len(beams), "fallback": True}

    return int(best["actions"][0]), {"beam_size": len(beams), "fallback": False}


def _is_unsafe_state(env_name: str, env, state_id: int) -> bool:
    if env_name == "nrm_nav":
        pos = env._state_to_pos(int(state_id))
        return pos in env.unsafe_positions
    if env_name == "cb":
        return bool(env.is_bomb_state(int(state_id)))
    if env_name == "frozenlake":
        return bool(env.is_hole_state(int(state_id)))
    if env_name == "dsrl" and hasattr(env, "is_unsafe_state"):
        return bool(env.is_unsafe_state(int(state_id)))
    return False


def _episode_violation_from_signals(env_name: str, sat_val: bool, ep_hazard: bool) -> float:
    """
    Compute per-episode violation indicator used by aggregated violation_rate.

    For DSRL we currently treat observed hazard/cost hits as the canonical safety
    signal for comparisons, because DFA-satisfaction can be optimistic depending on
    proposition calibration.
    """
    if env_name == "dsrl":
        return 1.0 if ep_hazard else 0.0
    return 0.0 if sat_val else 1.0


def _episode_satisfaction_from_signals(env_name: str, sat_val: bool, ep_hazard: bool) -> float:
    """
    Compute per-episode satisfaction indicator used by aggregated satisfaction_rate.

    For DSRL, keep satisfaction aligned with the primary hazard-based safety signal.
    """
    if env_name == "dsrl":
        return 0.0 if ep_hazard else 1.0
    return 1.0 if sat_val else 0.0


def evaluate_policy_rollouts(
    model,
    adapter,
    raw_dfa,
    dataset,
    env_name: str,
    spec_name: str | None,
    seed: int,
    checkpoint_path: str | None,
    num_episodes: int = 100,
    max_steps: int | None = None,
    decoding_cfg: DecodingConfig | None = None,
):
    if decoding_cfg is None:
        decoding_cfg = DecodingConfig()

    model_device = next(model.parameters()).device
    model.eval()

    env_cfg = dataset.env.cfg
    if hasattr(dataset.env, "clone"):
        env = dataset.env.clone()
    else:
        env = type(dataset.env)(env_cfg)
    if max_steps is None:
        max_steps = int(env_cfg.max_steps)

    episode_returns = []
    episode_lengths = []
    episode_sats = []
    episode_violations = []
    episode_goal_hits = []
    episode_hazard_hits = []
    step_violations = 0
    step_count = 0
    reject_sink_entries = 0
    fallback_decodes = 0

    deep_dfa = None
    if not isinstance(raw_dfa, (list, tuple)):
        try:
            deep_dfa = raw_dfa.return_deep_dfa().to(model_device)
        except Exception:
            deep_dfa = None

    soft_sats = []
    reject_checker, reject_states = _reject_sink_checker(raw_dfa)

    with torch.no_grad():
        for ep in range(int(num_episodes)):
            obs, _ = env.reset(seed=seed + ep)
            tokens = [int(obs)]
            history = torch.tensor([[int(obs)]], dtype=torch.long, device=model_device)

            dfa_state = _initial_dfa_state(raw_dfa)
            dfa_state = _advance_state_with_tokens(
                adapter,
                raw_dfa,
                dfa_state,
                torch.tensor([int(obs)], dtype=torch.long, device=model_device),
            )

            ep_return = 0.0
            ep_len = 0
            was_in_reject = reject_checker(dfa_state)
            done = False
            ep_goal = False
            ep_hazard = False

            while not done and ep_len < int(max_steps):
                action, decode_info = _decode_action(
                    model,
                    adapter,
                    raw_dfa,
                    history,
                    dfa_state,
                    env.action_space.n,
                    decoding_cfg,
                )
                if decode_info.get("fallback", False):
                    fallback_decodes += 1

                next_obs, reward, done, info = env.step(action)
                ep_return += float(reward)
                ep_len += 1
                step_count += 1

                unsafe_hit = _is_unsafe_state(env_name, env, int(next_obs))
                if unsafe_hit:
                    step_violations += 1

                terminal_type = info.get("terminal_type")
                if env_name == "nrm_nav":
                    cost_token = 1 if terminal_type == "X" else 0
                elif env_name == "frozenlake":
                    cost_token = 1 if terminal_type == "H" else 0
                elif env_name == "dsrl":
                    cost_token = 1 if float(info.get("cost", 0.0)) > 0.0 else 0
                else:
                    cost_token = 0
                reward_token = 0
                if env_name == "dsrl":
                    goal_thresh = float(getattr(dataset, "reward_goal_threshold", 0.0))
                    reward_token = 1 if (float(reward) > goal_thresh or bool(info.get("goal", False))) else 0
                transition_tokens = [int(action), int(reward_token), int(cost_token), int(next_obs)]
                tokens.extend(transition_tokens)

                transition_tensor = torch.tensor(
                    transition_tokens, dtype=torch.long, device=model_device
                )
                history = torch.cat([history, transition_tensor.view(1, -1)], dim=1)

                dfa_state = _advance_state_with_tokens(adapter, raw_dfa, dfa_state, transition_tensor)
                in_reject = reject_checker(dfa_state)
                if in_reject and not was_in_reject:
                    reject_sink_entries += 1
                was_in_reject = in_reject

                if env_name == "cb":
                    if terminal_type in {"P", "Y", "BLU"}:
                        ep_goal = True
                    if terminal_type == "B":
                        ep_hazard = True
                elif env_name == "nrm_nav":
                    if terminal_type == "G":
                        ep_goal = True
                    if terminal_type == "X":
                        ep_hazard = True
                elif env_name == "frozenlake":
                    if terminal_type == "G":
                        ep_goal = True
                    if terminal_type == "H":
                        ep_hazard = True
                elif env_name == "dsrl":
                    if bool(info.get("goal", False)) or terminal_type == "G":
                        ep_goal = True
                    if float(info.get("cost", 0.0)) > 0.0:
                        ep_hazard = True

            tokens_with_end = tokens + [int(adapter.end_token_id), 0, 0, 0]
            token_tensor = torch.tensor(tokens_with_end, dtype=torch.long, device=model_device).view(
                1, -1
            )
            if isinstance(raw_dfa, (list, tuple)):
                sats = [adapter.check_sat_token_ids(token_tensor, d) for d in raw_dfa]
                sat_val = bool(torch.stack(sats, dim=0).all())
            else:
                sat_val = bool(adapter.check_sat_token_ids(token_tensor, raw_dfa)[0].item())

            if deep_dfa is not None:
                token_probs = torch.nn.functional.one_hot(
                    token_tensor, num_classes=adapter.num_token_ids
                ).float()
                sym_probs = adapter.token_probs_to_symbol_probs(token_probs)
                soft_sat = float(adapter.check_sat_symbol_probs(sym_probs, deep_dfa)[0].item())
                soft_sats.append(soft_sat)

            episode_violation = _episode_violation_from_signals(env_name, sat_val, ep_hazard)
            episode_sat = _episode_satisfaction_from_signals(env_name, sat_val, ep_hazard)

            episode_returns.append(float(ep_return))
            episode_lengths.append(int(ep_len))
            episode_sats.append(float(episode_sat))
            episode_violations.append(float(episode_violation))
            episode_goal_hits.append(1.0 if ep_goal else 0.0)
            episode_hazard_hits.append(1.0 if ep_hazard else 0.0)

    return_mean = float(np.mean(episode_returns)) if episode_returns else None
    return_std = float(np.std(episode_returns)) if episode_returns else None
    satisfaction_rate = float(np.mean(episode_sats)) if episode_sats else None
    violation_rate_episode = float(np.mean(episode_violations)) if episode_violations else None
    violation_rate_step = float(step_violations / step_count) if step_count > 0 else None
    goal_rate = float(np.mean(episode_goal_hits)) if episode_goal_hits else None
    hazard_hit_rate = float(np.mean(episode_hazard_hits)) if episode_hazard_hits else None

    metrics = {
        "return_mean": return_mean,
        "return_std": return_std,
        "violation_rate": violation_rate_episode,
        "satisfaction_rate": satisfaction_rate,
        "runtime_sec": None,
        "env": env_name,
        "spec": spec_name,
        "seed": int(seed),
        "num_episodes": int(num_episodes),
        "satisfaction_soft_mean": float(np.mean(soft_sats)) if soft_sats else None,
        "violation_rate_episode": violation_rate_episode,
        "violation_rate_step": violation_rate_step,
        "goal_rate": goal_rate,
        "bomb_hit_rate": hazard_hit_rate if env_name == "cb" else None,
        "hazard_hit_rate": hazard_hit_rate,
        "decoding_mode": decoding_cfg.mode,
        "beam_width": int(decoding_cfg.beam_width),
        "target_shift": decoding_cfg.target_shift,
        "model_type": "tt",
        "checkpoint_path": checkpoint_path,
        "run_id": None,
        "timestamp_utc": None,
    }

    rollout_stats = {
        "num_episodes": int(num_episodes),
        "num_steps": int(step_count),
        "accept_count": int(sum(1 for x in episode_sats if x >= 0.5)),
        "violation_count": int(sum(1 for x in episode_violations if x >= 0.5)),
        "reject_sink_entries": int(reject_sink_entries),
        "reject_sink_states": _jsonable_reject_states(reject_states),
        "fallback_decodes": int(fallback_decodes),
        "decoding_mode": decoding_cfg.mode,
        "beam_width": int(decoding_cfg.beam_width),
        "plan_horizon": int(decoding_cfg.plan_horizon),
        "hard_prune_reject_sink": bool(decoding_cfg.hard_prune_reject_sink),
        "sat_rerank_weight": float(decoding_cfg.sat_rerank_weight),
        "target_shift": decoding_cfg.target_shift,
        "episode_returns": episode_returns,
        "episode_lengths": episode_lengths,
        "episode_satisfaction": episode_sats,
        "episode_goal_hits": episode_goal_hits,
        "episode_hazard_hits": episode_hazard_hits,
    }

    return metrics, rollout_stats


def _save_metrics_plots(run_dir: str, metrics: dict, rollout_stats: dict):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    plots_dir = os.path.join(run_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    bar_keys = ["goal_rate", "bomb_hit_rate", "satisfaction_rate", "return_mean"]
    labels = []
    values = []
    for key in bar_keys:
        val = metrics.get(key)
        if val is None:
            continue
        labels.append(key)
        values.append(float(val))
    if values:
        plt.figure(figsize=(6, 4))
        plt.bar(labels, values)
        plt.xticks(rotation=25, ha="right")
        plt.ylim(-0.1, max(1.0, max(values) + 0.1))
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "metrics_bar.png"))
        plt.close()

    sats = rollout_stats.get("episode_satisfaction") or []
    if sats:
        xs = np.arange(1, len(sats) + 1)
        running = np.cumsum(np.asarray(sats, dtype=np.float32)) / np.maximum(1, xs)
        plt.figure(figsize=(6, 4))
        plt.plot(xs, running)
        plt.xlabel("Episode")
        plt.ylabel("Running satisfaction")
        plt.ylim(-0.05, 1.05)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "satisfaction_trend.png"))
        plt.close()

    if metrics.get("return_mean") is not None and metrics.get("satisfaction_rate") is not None:
        plt.figure(figsize=(4, 4))
        plt.scatter([float(metrics["return_mean"])], [float(metrics["satisfaction_rate"])])
        plt.xlabel("return_mean")
        plt.ylabel("satisfaction_rate")
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "return_vs_satisfaction.png"))
        plt.close()


def save_evaluation_artifacts(
    run_dir: str,
    metrics: dict,
    dfa_summary: dict,
    rollout_stats: dict,
    save_plots: bool = False,
):
    write_json(os.path.join(run_dir, "metrics.json"), metrics)
    write_json(os.path.join(run_dir, "dfa_summary.json"), dfa_summary)
    write_json(os.path.join(run_dir, "automaton_rollout_stats.json"), rollout_stats)
    if save_plots:
        _save_metrics_plots(run_dir, metrics, rollout_stats)


def warn_if_train_fallback(enabled: bool):
    if enabled:
        warnings.warn(
            "Evaluation is running with training fallback. This is not a pure checkpoint-only evaluation run.",
            RuntimeWarning,
        )
