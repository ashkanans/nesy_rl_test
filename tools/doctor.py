from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from dfa_adapter import TTDFAAdapter
from logic.token_schema import build_end_row, get_schema_for_env
from models.dt_model import DecisionTransformerDiscrete
from planning.dt_runtime import compute_default_rtg_target
from planning.eval_runtime import (
    DecodingConfig,
    ensure_run_dir,
    evaluate_policy_rollouts,
    set_global_seed,
    write_json,
)
from scripts.evaluate import load_model_from_checkpoint as load_tt_model_from_checkpoint
from train_cb import build_adapter_and_dfa, build_dataset, get_arg_parser, resolve_formulas


def _parse_args():
    parent = get_arg_parser(add_help=False)
    parser = argparse.ArgumentParser(parents=[parent], description="Unified doctor tool for satisfaction debugging.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Optional TT/DT checkpoint for model-trace diagnostics.")
    parser.add_argument(
        "--model_type",
        type=str,
        choices=["auto", "tt", "dt"],
        default="auto",
        help="Checkpoint model type. Use auto to infer from checkpoint config.",
    )
    parser.add_argument("--output_json", type=str, default=None, help="Write diagnostics JSON to this path.")
    parser.add_argument("--dataset_sample_size", type=int, default=64)
    parser.add_argument("--rollout_sample_size", type=int, default=16)
    parser.add_argument("--rollout_max_steps", type=int, default=None)
    parser.add_argument("--small_sample", action="store_true", help="Alias for --smoke.")
    args = parser.parse_args()

    if args.small_sample:
        args.smoke = True
    if args.env not in {"cb", "frozenlake", "nrm_nav"}:
        raise ValueError("tools.doctor currently supports env in {cb, frozenlake, nrm_nav}.")
    if args.spec is not None and (args.ltl_formula is not None or args.ltl_formulas is not None):
        raise ValueError("Provide either --spec or --ltl_formula(s), not both.")
    if args.spec is None and args.ltl_formula is None and args.ltl_formulas is None:
        raise ValueError("Provide one of --spec, --ltl_formula, --ltl_formulas.")

    if args.smoke:
        args.num_episodes = min(int(args.num_episodes), 64 if args.env != "frozenlake" else 200)
        args.max_steps = min(int(args.max_steps), 30)
        args.block_size = min(int(args.block_size), 32)
        args.dataset_sample_size = min(int(args.dataset_sample_size), 32)
        args.rollout_sample_size = min(int(args.rollout_sample_size), 8)

    args.inspect_dfa_only = False
    args.analyze_dataset_only = False
    args.replay_dataset_episode = False
    return args


def _to_flat_tensor(tokens_2d: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(tokens_2d.reshape(-1).astype(np.int64)).unsqueeze(0)


def _end_token_alignment(dataset, schema) -> dict:
    episodes = getattr(dataset, "episodes_tokens", [])
    end_id = int(getattr(dataset, "end_token_id"))
    expected_end = build_end_row(schema, end_id)
    invalid = []
    ok = 0
    for i, ep in enumerate(episodes):
        if ep.ndim != 2:
            invalid.append({"episode_index": i, "reason": "not_2d"})
            continue
        state_col = ep[:, schema.field_index("state")]
        end_positions = np.where(state_col == end_id)[0]
        if len(end_positions) != 1:
            invalid.append(
                {"episode_index": i, "reason": "end_count", "count": int(len(end_positions))}
            )
            continue
        if int(end_positions[0]) != int(ep.shape[0] - 1):
            invalid.append({"episode_index": i, "reason": "end_not_last"})
            continue
        if not np.array_equal(ep[-1], expected_end):
            invalid.append(
                {
                    "episode_index": i,
                    "reason": "end_row_mismatch",
                    "expected": expected_end.tolist(),
                    "actual": ep[-1].tolist(),
                }
            )
            continue
        ok += 1

    return {
        "num_episodes": int(len(episodes)),
        "end_token_id": end_id,
        "valid_episodes": int(ok),
        "invalid_episodes": int(len(invalid)),
        "invalid_samples": invalid[:10],
    }


def _symbol_coverage_from_token_flats(adapter: TTDFAAdapter, token_flats: list[np.ndarray]) -> dict:
    token_counter: Counter[int] = Counter()
    symbol_counter: Counter[int] = Counter()
    for flat in token_flats:
        tok = torch.from_numpy(flat.astype(np.int64)).unsqueeze(0)
        symbol_ids = adapter.token_ids_to_symbol_ids(tok)[0].tolist()
        token_counter.update(int(t) for t in flat.tolist())
        symbol_counter.update(int(s) for s in symbol_ids)

    observed_symbols = set(symbol_counter.keys())
    total_symbols = set(range(adapter.num_symbols))
    missing = sorted(total_symbols - observed_symbols)

    top_symbols = []
    for sym_id, count in symbol_counter.most_common(20):
        top_symbols.append(
            {"symbol_id": int(sym_id), "symbol": adapter.symbolic_vocab[int(sym_id)], "count": int(count)}
        )

    return {
        "num_symbols_total": int(adapter.num_symbols),
        "num_symbols_observed": int(len(observed_symbols)),
        "coverage_ratio": float(len(observed_symbols) / max(1, adapter.num_symbols)),
        "num_tokens_observed": int(sum(token_counter.values())),
        "top_symbols": top_symbols,
        "missing_symbols_sample": [adapter.symbolic_vocab[i] for i in missing[:20]],
    }


def _dataset_satisfaction(dataset, adapter, raw_dfa, sample_size: int) -> dict:
    episodes = getattr(dataset, "episodes_tokens", [])
    n = min(int(sample_size), len(episodes))
    if n == 0:
        return {"num_episodes_checked": 0, "joint_satisfaction_rate": None, "per_formula_rates": []}

    dfa_list = raw_dfa if isinstance(raw_dfa, list) else [raw_dfa]
    per_formula = [[] for _ in dfa_list]
    joint = []
    for i in range(n):
        tok = _to_flat_tensor(episodes[i])
        sat_vals = []
        for j, dfa in enumerate(dfa_list):
            sat = bool(adapter.check_sat_token_ids(tok, dfa)[0].item())
            per_formula[j].append(1.0 if sat else 0.0)
            sat_vals.append(sat)
        joint.append(1.0 if all(sat_vals) else 0.0)

    return {
        "num_episodes_checked": int(n),
        "joint_satisfaction_rate": float(np.mean(joint)),
        "per_formula_rates": [float(np.mean(v)) if v else None for v in per_formula],
    }


def _infer_model_type(checkpoint_path: str) -> str:
    payload = torch.load(checkpoint_path, map_location="cpu")
    cfg = payload.get("config", {})
    if {"num_states", "num_actions", "context_len"}.issubset(cfg.keys()):
        return "dt"
    return "tt"


def _load_dt_model(checkpoint_path: str, device: torch.device):
    payload = torch.load(checkpoint_path, map_location=device)
    cfg = payload.get("config", {})
    model = DecisionTransformerDiscrete(
        num_states=int(cfg["num_states"]),
        num_actions=int(cfg["num_actions"]),
        context_len=int(cfg["context_len"]),
        n_embd=int(cfg["n_embd"]),
        n_layer=int(cfg["n_layer"]),
        n_head=int(cfg["n_head"]),
        dropout=float(cfg["dropout"]),
        max_timestep=int(cfg.get("max_timestep", 4096)),
    ).to(device)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    return model, cfg


def _dt_action(model, states, prev_actions, rtgs, timesteps, device):
    with torch.no_grad():
        logits = model(
            torch.tensor([states], dtype=torch.long, device=device),
            torch.tensor([prev_actions], dtype=torch.long, device=device),
            torch.tensor([rtgs], dtype=torch.float32, device=device),
            torch.tensor([timesteps], dtype=torch.long, device=device),
            attention_mask=torch.ones((1, len(states)), dtype=torch.float32, device=device),
        )
    return int(torch.argmax(logits[0, -1, :]).item())


def _doctor_dt_rollout_satisfaction(
    args,
    model,
    cfg,
    adapter,
    raw_dfa,
    dataset,
    rollout_sample_size: int,
    rollout_max_steps: int | None,
    rtg_target: float,
) -> dict:
    env = dataset.env
    context_len = int(cfg.get("context_len", 20))
    device = next(model.parameters()).device
    end_row = build_end_row(get_schema_for_env(args.env), int(dataset.end_token_id))
    dfa_list = raw_dfa if isinstance(raw_dfa, list) else [raw_dfa]

    sats = []
    token_flats = []
    for ep in range(int(rollout_sample_size)):
        obs, _ = env.reset(seed=args.seed + ep)
        done = False
        prev_action = int(env.action_space.n)
        cum_reward = 0.0
        step = 0
        max_steps = int(rollout_max_steps if rollout_max_steps is not None else args.max_steps)

        states_hist: list[int] = []
        prev_hist: list[int] = []
        rtg_hist: list[float] = []
        time_hist: list[int] = []
        rows: list[np.ndarray] = []

        while not done and step < max_steps:
            states_hist.append(int(obs))
            prev_hist.append(int(prev_action))
            rtg_hist.append(float(rtg_target - cum_reward))
            time_hist.append(int(step))
            states_hist = states_hist[-context_len:]
            prev_hist = prev_hist[-context_len:]
            rtg_hist = rtg_hist[-context_len:]
            time_hist = time_hist[-context_len:]

            action = _dt_action(model, states_hist, prev_hist, rtg_hist, time_hist, device=device)
            next_obs, reward, done, info = env.step(action)
            terminal_type = info.get("terminal_type")
            cost = 0
            if args.env == "frozenlake":
                cost = 1 if terminal_type == "H" else 0
            elif args.env == "nrm_nav":
                cost = 1 if terminal_type == "X" else 0

            rows.append(
                np.asarray([int(obs), int(action), 0, int(cost)], dtype=np.int64)
            )
            cum_reward += float(reward)
            prev_action = int(action)
            obs = int(next_obs)
            step += 1

        if not rows:
            continue
        tokens = np.vstack(rows + [end_row])
        flat = tokens.reshape(-1)
        token_flats.append(flat)
        tok = torch.from_numpy(flat.astype(np.int64)).unsqueeze(0)
        sat_vals = [bool(adapter.check_sat_token_ids(tok, d)[0].item()) for d in dfa_list]
        sats.append(1.0 if all(sat_vals) else 0.0)

    return {
        "checked": True,
        "model_type": "dt",
        "num_rollouts_checked": int(len(sats)),
        "satisfaction_rate": float(np.mean(sats)) if sats else None,
        "symbol_coverage": _symbol_coverage_from_token_flats(adapter, token_flats) if token_flats else None,
    }


def _doctor_tt_rollout_satisfaction(args, adapter, raw_dfa, dataset, checkpoint_path: str) -> dict:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(checkpoint_path, map_location=device)
    model = load_tt_model_from_checkpoint(args, dataset, ckpt, device)
    decoding_cfg = DecodingConfig(
        mode=args.decoding_mode,
        beam_width=args.beam_width,
        plan_horizon=args.plan_horizon,
        sat_rerank_weight=args.sat_rerank_weight,
        hard_prune_reject_sink=args.hard_prune_reject_sink,
    )
    metrics, rollout_stats = evaluate_policy_rollouts(
        model=model,
        adapter=adapter,
        raw_dfa=raw_dfa,
        dataset=dataset,
        env_name=args.env,
        spec_name=args.spec,
        seed=args.seed,
        checkpoint_path=checkpoint_path,
        num_episodes=int(args.rollout_sample_size),
        max_steps=args.rollout_max_steps,
        decoding_cfg=decoding_cfg,
    )
    return {
        "checked": True,
        "model_type": "tt",
        "num_rollouts_checked": int(metrics.get("num_episodes", 0)),
        "satisfaction_rate": metrics.get("satisfaction_rate"),
        "satisfaction_soft_mean": metrics.get("satisfaction_soft_mean"),
        "violation_rate": metrics.get("violation_rate"),
        "goal_rate": metrics.get("goal_rate"),
        "hazard_hit_rate": metrics.get("hazard_hit_rate"),
        "rollout_stats": {
            "accept_count": rollout_stats.get("accept_count"),
            "violation_count": rollout_stats.get("violation_count"),
            "reject_sink_entries": rollout_stats.get("reject_sink_entries"),
        },
    }


def main():
    args = _parse_args()
    set_global_seed(args.seed)
    t0 = time.time()

    run_dir, run_id, ts = ensure_run_dir(args.env, run_dir=args.run_dir, base_dir=args.base_runs_dir)
    out_path = args.output_json or os.path.join(run_dir, "doctor_diagnostics.json")
    os.makedirs(Path(out_path).parent, exist_ok=True)

    dataset = build_dataset(args)
    adapter, _, raw_dfa = build_adapter_and_dfa(args, dataset)
    formulas = resolve_formulas(args, dataset=dataset)

    schema = get_schema_for_env(args.env)
    sampled_eps = getattr(dataset, "episodes_tokens", [])[: int(args.dataset_sample_size)]
    sampled_flats = [ep.reshape(-1).astype(np.int64) for ep in sampled_eps]

    report = {
        "doctor_version": "v1",
        "status": "ok",
        "env": args.env,
        "seed": int(args.seed),
        "spec": args.spec,
        "formulas": formulas,
        "run_id": run_id,
        "timestamp_utc": ts,
        "run_dir": run_dir,
        "dataset": {
            "num_segments": int(len(dataset)),
            "num_episodes": int(len(getattr(dataset, "episodes_tokens", []))),
            "dataset_sample_size": int(min(len(getattr(dataset, "episodes_tokens", [])), args.dataset_sample_size)),
        },
        "checks": {
            "end_token_alignment": _end_token_alignment(dataset, schema),
            "symbol_coverage_dataset": _symbol_coverage_from_token_flats(adapter, sampled_flats) if sampled_flats else None,
            "satisfaction_dataset": _dataset_satisfaction(dataset, adapter, raw_dfa, sample_size=args.dataset_sample_size),
        },
    }

    model_trace_checks = {"checked": False, "reason": "no_checkpoint"}
    if args.checkpoint is not None:
        if not os.path.exists(args.checkpoint):
            raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
        model_type = args.model_type if args.model_type != "auto" else _infer_model_type(args.checkpoint)
        if model_type == "tt":
            model_trace_checks = _doctor_tt_rollout_satisfaction(args, adapter, raw_dfa, dataset, args.checkpoint)
        elif model_type == "dt":
            if args.env not in {"cb", "frozenlake"}:
                model_trace_checks = {
                    "checked": False,
                    "model_type": "dt",
                    "reason": "DT doctor rollout checks currently support env in {cb, frozenlake}.",
                }
            else:
                device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                model, cfg = _load_dt_model(args.checkpoint, device=device)
                rtg_target = float(cfg.get("rtg_target", compute_default_rtg_target(dataset)))
                if args.env == "frozenlake":
                    rtg_target = max(1.0, rtg_target)
                model_trace_checks = _doctor_dt_rollout_satisfaction(
                    args=args,
                    model=model,
                    cfg=cfg,
                    adapter=adapter,
                    raw_dfa=raw_dfa,
                    dataset=dataset,
                    rollout_sample_size=args.rollout_sample_size,
                    rollout_max_steps=args.rollout_max_steps,
                    rtg_target=rtg_target,
                )
        else:
            model_trace_checks = {
                "checked": False,
                "reason": f"Unsupported model_type '{model_type}'.",
            }

    report["checks"]["model_trace_satisfaction"] = model_trace_checks
    report["runtime_sec"] = float(time.time() - t0)

    write_json(out_path, report)
    print(f"Doctor diagnostics saved to {out_path}")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
