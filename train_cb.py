import argparse
import csv
import json
import os
import sys
import time
import warnings
from collections import deque
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).parent
sys.path.insert(0, str(REPO_ROOT / "trajectory-transformer"))
sys.path.insert(0, str(REPO_ROOT / "suffix-prediction"))

import FiniteStateMachine as FSM
from FiniteStateMachine import DFA

from datasets.cb_dataset import CBSequenceDataset
from dfa_adapter import TTDFAAdapter, get_num_bins_per_dim_for_env
from dfa_utils import export_dfa_artifacts
from datasets.dsrl_dataset import DSRLSequenceDataset
from datasets.frozenlake_dataset import FrozenLakeSequenceDataset
from datasets.nrm_nav_dataset import NRMSafetySequenceDataset
from envs.nrm_nav_env import NRMSafetyNavEnv
from models.tt_model import build_tt_model
from planning.eval_runtime import (
    DecodingConfig,
    apply_smoke_mode,
    ensure_run_dir,
    evaluate_policy_rollouts,
    save_evaluation_artifacts,
    set_global_seed,
    spec_label_from_args,
    summarize_dfa_bundle,
)
from logic_loss_tt import LogicLossModule
from specs import get_spec
from specs.frozenlake_specs import build_frozenlake_formulas

if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"


def _to_jsonable(value):
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    return str(value)


def _save_args_snapshot(args, out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    payload = {k: _to_jsonable(v) for k, v in vars(args).items()}
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def build_product_dfa(dfas):
    """
    Build a product DFA that accepts only if all component DFAs accept.
    Assumes all DFAs share the same dictionary_symbols and num_of_symbols.
    """
    if not dfas:
        raise ValueError("No DFAs provided for product construction")

    dict_syms = dfas[0].dictionary_symbols
    num_syms = len(dict_syms)
    for d in dfas[1:]:
        if d.dictionary_symbols != dict_syms:
            raise ValueError("All DFAs must share the same dictionary_symbols for product")

    init_state = tuple(0 for _ in dfas)
    state_to_idx = {init_state: 0}
    transitions = {}
    acceptance = []
    queue = deque([init_state])

    def is_accept(state_tuple):
        return all(dfas[i].acceptance[s] for i, s in enumerate(state_tuple))

    while queue:
        state_tuple = queue.popleft()
        s_idx = state_to_idx[state_tuple]
        transitions[s_idx] = {}

        # acceptance list is indexed by s_idx
        if len(acceptance) <= s_idx:
            acceptance.append(is_accept(state_tuple))

        for sym in range(num_syms):
            next_tuple = []
            for i, d in enumerate(dfas):
                next_state = d.transitions[state_tuple[i]].get(sym, state_tuple[i])
                next_tuple.append(next_state)
            next_tuple = tuple(next_tuple)
            if next_tuple not in state_to_idx:
                state_to_idx[next_tuple] = len(state_to_idx)
                queue.append(next_tuple)
            transitions[s_idx][sym] = state_to_idx[next_tuple]

    product_dfa = DFA(transitions, acceptance, None, dictionary_symbols=dict_syms)
    return product_dfa


def resolve_formulas(args, dataset=None):
    if args.spec is not None and (args.ltl_formula is not None or args.ltl_formulas is not None):
        raise ValueError("Provide either --spec or --ltl_formula(s), not both.")

    if args.spec is not None:
        if args.env in {"cb", "nrm_nav", "dsrl"}:
            spec = get_spec(args.env, args.spec)
            return list(spec["formulas"])
        if args.env == "frozenlake":
            if dataset is not None and hasattr(dataset.env, "hole_state_ids"):
                return build_frozenlake_formulas(
                    args.spec,
                    dataset.env.hole_state_ids,
                    dataset.env.goal_state_ids,
                    include_position_props=getattr(args, "frozenlake_use_position_props", False),
                )
            spec = get_spec(args.env, args.spec)
            return list(spec["formulas"])
        raise ValueError(
            "--spec runtime support is currently only available for cb/nrm_nav/frozenlake/dsrl, "
            f"got env={args.env}"
        )

    if args.ltl_formulas is not None:
        return list(args.ltl_formulas)
    if args.ltl_formula is not None:
        return [args.ltl_formula]
    raise ValueError("You must provide --spec or --ltl_formula/--ltl_formulas.")


def _inspect_dfa_artifacts(args, raw_dfa):
    inspect_dir = (
        args.inspect_output_dir
        if args.inspect_output_dir is not None
        else os.path.join((getattr(args, "run_dir", None) or args.save_path or "artifacts"), "dfa_inspect")
    )
    os.makedirs(inspect_dir, exist_ok=True)

    if isinstance(raw_dfa, list):
        for i, dfa in enumerate(raw_dfa):
            info = export_dfa_artifacts(dfa, inspect_dir, stem=f"dfa_{i}")
            print(
                f"DFA {i} artifacts: summary={info['summary_path']} dot={info['dot_path']} "
                f"png={info['png_path']}"
            )
    else:
        info = export_dfa_artifacts(raw_dfa, inspect_dir, stem="dfa")
        print(
            f"DFA artifacts: summary={info['summary_path']} dot={info['dot_path']} "
            f"png={info['png_path']}"
        )


def build_adapter_and_dfa(args, dataset):
    """
    Build TTDFAAdapter + DeepDFA for Colour Bomb.

    Build adapter + DFA stack under canonical explicit-END semantics.
    """
    # Canonical path: disable legacy DFA END hack globally.
    FSM.USE_END_HACK = False

    env = dataset.env
    if hasattr(dataset, "num_bins_per_dim"):
        num_bins_per_dim = list(dataset.num_bins_per_dim)
    else:
        obs_bins = env.observation_space.n
        act_bins = env.action_space.n
        num_bins_per_dim = get_num_bins_per_dim_for_env(args.env, obs_bins, act_bins)

    adapter = TTDFAAdapter(
        observation_dim=dataset.observation_dim,
        action_dim=dataset.action_dim,
        num_bins=num_bins_per_dim,
        include_reward=True,
        include_value=True,
        constraint_dims=args.constraint_dims,
        abstraction_fn=None,
        use_stop_token=True,
    )

    formulas = resolve_formulas(args, dataset=dataset)

    dfas = [
        adapter.create_dfa_from_ltl(
            f,
            f"cb_constraint_{i}",
            use_safe_dfa=args.use_safe_dfa,
            dfa_backend=getattr(args, "dfa_backend", "auto"),
        )
        for i, f in enumerate(formulas)
    ]

    if getattr(args, "inspect_dfa_only", False):
        print("=== DFA inspection (per-formula) ===")
        print(f"Number of formulas: {len(dfas)}")
        for i, d in enumerate(dfas):
            num_states = getattr(d, "num_of_states", None)
            num_symbols = len(getattr(d, "dictionary_symbols", []))
            num_accept = sum(getattr(d, "acceptance", []))
            print(
                f"  DFA {i}: states={num_states}, accepting_states={num_accept}, symbols={num_symbols}"
            )

    if len(dfas) == 1 or args.dfa_mode == "single":
        dfa = dfas[0]
        deep_dfa = dfa.return_deep_dfa()
        raw_dfa = dfa
    elif args.dfa_mode == "product":
        raw_dfa = build_product_dfa(dfas)
        deep_dfa = raw_dfa.return_deep_dfa()
    elif args.dfa_mode == "multi":
        deep_dfa = [d.return_deep_dfa() for d in dfas]
        raw_dfa = dfas
    else:
        raise ValueError(f"Unknown dfa_mode {args.dfa_mode}")

    if getattr(args, "inspect_dfa_only", False):
        # Report combined DFA size as well
        if isinstance(raw_dfa, list):
            print("=== DFA inspection (combined: multi) ===")
            for i, d in enumerate(raw_dfa):
                num_states = getattr(d, "num_of_states", None)
                num_symbols = len(getattr(d, "dictionary_symbols", []))
                num_accept = sum(getattr(d, "acceptance", []))
                print(
                    f"  DFA {i}: states={num_states}, accepting_states={num_accept}, symbols={num_symbols}"
                )
        else:
            print("=== DFA inspection (combined) ===")
            num_states = getattr(raw_dfa, "num_of_states", None)
            num_symbols = len(getattr(raw_dfa, "dictionary_symbols", []))
            num_accept = sum(getattr(raw_dfa, "acceptance", []))
            print(
                f"  DFA: states={num_states}, accepting_states={num_accept}, symbols={num_symbols}"
            )
        _inspect_dfa_artifacts(args, raw_dfa)

    return adapter, deep_dfa, raw_dfa


def build_model(args, dataset, vocab_size):
    return build_tt_model(args=args, dataset=dataset, vocab_size=vocab_size, device=device)


def build_dataset(args):
    if args.env == "frozenlake":
        num_episodes = int(args.num_episodes)
        max_steps = int(args.max_steps)
        if num_episodes == 2000:
            num_episodes = 5000
        if max_steps == 200:
            max_steps = 100
        return FrozenLakeSequenceDataset(
            num_episodes=num_episodes,
            max_steps=max_steps,
            sequence_length=args.block_size,
            discount=args.discount,
            seed=args.seed,
            map_size=args.frozenlake_map_size,
            is_slippery=args.frozenlake_is_slippery,
            policy_mix=args.policy_mix,
            target_shift=args.target_shift,
        )
    if args.env == "cb":
        return CBSequenceDataset(
            num_episodes=args.num_episodes,
            max_steps=args.max_steps,
            sequence_length=args.block_size,
            discount=args.discount,
            stochastic=args.stochastic,
            seed=args.seed,
            target_shift=args.target_shift,
            policy_mix_spec=getattr(args, "cb_policy_mix_spec", "random:1.0"),
            policy_mix_sampling=getattr(args, "cb_policy_mix_sampling", "fixed"),
            policy_mix_normal_spec=getattr(args, "cb_policy_mix_normal_spec", None),
            state_semantics=getattr(args, "cb_state_semantics", "post"),
            longest_path_max_expansions=getattr(args, "cb_longest_path_max_expansions", 500000),
        )
    elif args.env == "nrm_nav":
        return NRMSafetySequenceDataset(
            num_episodes=args.num_episodes,
            max_steps=args.max_steps,
            sequence_length=args.block_size,
            discount=args.discount,
            stochastic=args.stochastic,
            seed=args.seed,
            grid=None,
            target_shift=args.target_shift,
        )
    elif args.env == "dsrl":
        return DSRLSequenceDataset(
            dataset_path=args.dsrl_dataset_path,
            dataset_key=args.dsrl_dataset_key,
            sequence_length=args.block_size,
            seed=args.seed,
            max_steps=args.max_steps,
            num_episodes=args.num_episodes,
            state_bins=args.dsrl_state_bins,
            action_bins=args.dsrl_action_bins,
            reward_goal_threshold=args.dsrl_reward_goal_threshold,
            cost_unsafe_threshold=args.dsrl_cost_unsafe_threshold,
            cost_unsafe_quantile=args.dsrl_cost_unsafe_quantile,
            target_shift=args.target_shift,
            download=args.dsrl_download,
        )
    else:
        raise ValueError(f"Unknown env {args.env}")


def train(args, return_state=False):
    args = apply_smoke_mode(args)
    set_global_seed(args.seed)
    run_dir, run_id, ts = ensure_run_dir(
        args.env,
        run_dir=(getattr(args, "run_dir", None) or getattr(args, "save_path", None)),
        base_dir=getattr(args, "base_runs_dir", "runs"),
    )
    args.run_dir = run_dir
    args.save_path = run_dir  # backward-compatible alias
    train_t0 = time.time()
    os.makedirs(args.run_dir, exist_ok=True)
    _save_args_snapshot(args, os.path.join(args.run_dir, "run_args.json"))

    if getattr(args, "no_end_state_hack", False):
        warnings.warn(
            "--no_end_state_hack is deprecated; canonical explicit END semantics are always used.",
            DeprecationWarning,
        )
    if getattr(args, "append_end_token_to_dfa", False):
        warnings.warn(
            "--append_end_token_to_dfa is deprecated; canonical explicit END semantics are always used.",
            DeprecationWarning,
        )
    FSM.USE_END_HACK = False

    dataset = build_dataset(args)

    # Optional: replay a single dataset episode and exit.
    if getattr(args, "replay_dataset_episode", False):
        replay_dataset_episode(args, dataset)
        sys.exit(0)

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    adapter, deep_dfa, raw_dfa = build_adapter_and_dfa(args, dataset)

    # If requested, only inspect DFA sizes and exit before training.
    if getattr(args, "inspect_dfa_only", False):
        print("inspect_dfa_only flag set; skipping training and evaluation.")
        sys.exit(0)

    # Optional: analyze dataset in depth (including constraint satisfaction) and exit.
    if getattr(args, "analyze_dataset_only", False):
        analyze_dataset(args, dataset, adapter, raw_dfa)
        sys.exit(0)
    # GPT expects vocab_size without the extra stop token it appends internally
    model = build_model(args, dataset, vocab_size=adapter.num_token_ids - 1)

    logic = LogicLossModule(
        deep_dfa=deep_dfa,
        adapter=adapter,
        mode="global",
        num_samples=args.num_samples,
        temperature=args.temperature,
        alpha=args.alpha,
        eps=getattr(args, "logic_eps", 1e-10),
        clamp_acceptance=not getattr(args, "no_logic_clamp", False),
        acceptance_floor_mode=getattr(args, "logic_acceptance_floor_mode", None),
        sample_weighting=getattr(args, "logic_sample_weighting", "importance"),
        logic_state_only=getattr(args, "logic_state_only", False),
    )

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    latest_ckpt_path = None

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        total_sup = 0.0
        total_log = 0.0
        total_prob_acc = 0.0
        total_prob_acc_min = 0.0
        total_prob_acc_le_eps = 0.0
        n_logic_stats = 0
        n_batches = 0

        for batch in loader:
            batch = [b.to(device) for b in batch]
            loss, sup_loss, logic_loss = logic.compute_loss(model, batch, return_components=True)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()

            total_loss += loss.item()
            total_sup += sup_loss.item()
            total_log += logic_loss.item()
            stats = getattr(logic, "last_logic_stats", None)
            if isinstance(stats, dict) and stats:
                total_prob_acc += float(stats.get("prob_acceptance_mean", 0.0))
                total_prob_acc_min += float(stats.get("prob_acceptance_min", 0.0))
                frac = stats.get("frac_prob_acceptance_le_eps")
                if frac is not None:
                    total_prob_acc_le_eps += float(frac)
                n_logic_stats += 1
            n_batches += 1

        msg = (
            "epoch %d | loss %.4f | sup %.4f | logic %.4f"
            % (
                epoch,
                total_loss / max(1, n_batches),
                total_sup / max(1, n_batches),
                total_log / max(1, n_batches),
            )
        )
        if getattr(args, "logic_report_stats", False) and n_logic_stats > 0:
            msg += (
                " | p_acc_mean %.6f | p_acc_min %.6f | frac_p_acc<=eps %.3f"
                % (
                    total_prob_acc / n_logic_stats,
                    total_prob_acc_min / n_logic_stats,
                    total_prob_acc_le_eps / n_logic_stats,
                )
            )
        print(msg)

        if args.run_dir is not None:
            os.makedirs(args.run_dir, exist_ok=True)
            ckpt_path = os.path.join(args.run_dir, f"cb_state_{epoch}.pt")
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": {
                        "vocab_size": adapter.num_token_ids - 1,
                        "block_size": args.block_size,
                        "n_layer": args.n_layer,
                        "n_head": args.n_head,
                        "n_embd": args.n_embd,
                        "observation_dim": dataset.observation_dim,
                        "action_dim": dataset.action_dim,
                        "transition_dim": dataset.joined_dim,
                        "action_weight": args.action_weight,
                        "reward_weight": args.reward_weight,
                        "value_weight": args.value_weight,
                        "embd_pdrop": args.embd_pdrop,
                        "resid_pdrop": args.resid_pdrop,
                        "attn_pdrop": args.attn_pdrop,
                    },
                },
                ckpt_path,
            )
            latest_ckpt_path = ckpt_path

    if return_state:
        return model, adapter, deep_dfa, dataset, raw_dfa

    spec_name = spec_label_from_args(args)
    formulas = resolve_formulas(args, dataset=dataset)
    dfa_summary = summarize_dfa_bundle(
        raw_dfa, spec_name=spec_name, formulas=formulas, dfa_mode=args.dfa_mode
    )

    if getattr(args, "no_eval_after_train", False):
        metrics = {
            "return_mean": None,
            "return_std": None,
            "violation_rate": None,
            "satisfaction_rate": None,
            "runtime_sec": float(time.time() - train_t0),
            "env": args.env,
            "spec": spec_name,
            "seed": int(args.seed),
            "num_episodes": None,
            "satisfaction_soft_mean": None,
            "violation_rate_episode": None,
            "violation_rate_step": None,
            "goal_rate": None,
            "bomb_hit_rate": None,
            "hazard_hit_rate": None,
            "decoding_mode": None,
            "beam_width": None,
            "model_type": "tt",
            "checkpoint_path": latest_ckpt_path,
            "run_id": run_id,
            "timestamp_utc": ts,
        }
        rollout_stats = {"skipped": True, "reason": "no_eval_after_train"}
    else:
        decoding_cfg = DecodingConfig(
            mode=args.decoding_mode,
            beam_width=args.beam_width,
            plan_horizon=args.plan_horizon,
            sat_rerank_weight=args.sat_rerank_weight,
            hard_prune_reject_sink=args.hard_prune_reject_sink,
            target_shift=args.target_shift,
        )
        metrics, rollout_stats = evaluate_policy_rollouts(
            model=model,
            adapter=adapter,
            raw_dfa=raw_dfa,
            dataset=dataset,
            env_name=args.env,
            spec_name=spec_name,
            seed=args.seed,
            checkpoint_path=latest_ckpt_path,
            num_episodes=args.eval_num_episodes,
            max_steps=args.eval_max_steps,
            decoding_cfg=decoding_cfg,
        )
        metrics["runtime_sec"] = float(time.time() - train_t0)
        metrics["run_id"] = run_id
        metrics["timestamp_utc"] = ts

    save_evaluation_artifacts(
        args.run_dir,
        metrics,
        dfa_summary,
        rollout_stats,
        save_plots=getattr(args, "save_plots", False),
    )
    print(f"Saved run artifacts to {args.run_dir}")


def _append_end_token(tensor_batch, adapter, append_flag):
    if append_flag:
        warnings.warn(
            "_append_end_token compatibility path is deprecated; canonical END handling is adapter-driven.",
            DeprecationWarning,
        )
    return tensor_batch


def evaluate_model(model, adapter, dfa, dataset, batch_size=64, append_end_token=False):
    warnings.warn(
        "train_cb.evaluate_model is deprecated; use evaluate.py or eval_runtime.evaluate_policy_rollouts.",
        DeprecationWarning,
    )
    _ = batch_size, append_end_token
    spec_name = None
    decoding_cfg = DecodingConfig(mode="greedy", beam_width=1, plan_horizon=1, target_shift="token")
    metrics, _ = evaluate_policy_rollouts(
        model=model,
        adapter=adapter,
        raw_dfa=dfa,
        dataset=dataset,
        env_name=(
            getattr(dataset, "env_name")
            if getattr(dataset, "env_name", None) is not None
            else (
                "nrm_nav"
                if isinstance(dataset.env, NRMSafetyNavEnv)
                else (
                    "frozenlake"
                    if "frozenlake" in dataset.env.__class__.__name__.lower()
                    else "cb"
                )
            )
        ),
        spec_name=spec_name,
        seed=0,
        checkpoint_path=None,
        num_episodes=min(50, max(1, len(getattr(dataset, "episodes_tokens", [])))),
        max_steps=getattr(dataset.env.cfg, "max_steps", None),
        decoding_cfg=decoding_cfg,
    )
    return metrics


def rollout_nrm_nav_policy(
    model,
    adapter,
    dfa,
    env_cfg,
    num_episodes=100,
    max_steps=None,
    greedy=True,
    append_end_token=False,
):
    warnings.warn(
        "train_cb.rollout_nrm_nav_policy is deprecated; use evaluate.py or "
        "eval_runtime.evaluate_policy_rollouts.",
        DeprecationWarning,
    )
    env = NRMSafetyNavEnv(env_cfg)

    class _DatasetProxy:
        def __init__(self, env_obj):
            self.env = env_obj
            self.episodes_tokens = []

    proxy = _DatasetProxy(env)
    mode = "greedy" if greedy else "beam"
    decoding_cfg = DecodingConfig(mode=mode, beam_width=4, plan_horizon=2, target_shift="token")
    _ = append_end_token
    metrics, _ = evaluate_policy_rollouts(
        model=model,
        adapter=adapter,
        raw_dfa=dfa,
        dataset=proxy,
        env_name="nrm_nav",
        spec_name=None,
        seed=0,
        checkpoint_path=None,
        num_episodes=num_episodes,
        max_steps=max_steps,
        decoding_cfg=decoding_cfg,
    )
    return {
        "num_episodes": metrics.get("num_episodes"),
        "avg_return": metrics.get("return_mean"),
        "std_return": metrics.get("return_std"),
        "unsafe_episode_rate": metrics.get("violation_rate_step"),
        "satisfaction_rate_rollout": metrics.get("satisfaction_rate"),
    }


def analyze_dataset(args, dataset, adapter, raw_dfa):
    """
    Analyze the offline dataset in depth and save statistics and plots.

    Works for both cb and nrm_nav. If LTL constraints are provided, also
    reports satisfaction rates under the corresponding DFA(s).
    """

    save_root = args.save_path or f"{args.env}_dataset_analysis"
    os.makedirs(save_root, exist_ok=True)
    out_dir = os.path.join(save_root, "dataset_analysis")
    os.makedirs(out_dir, exist_ok=True)

    summary = {}
    summary["env"] = args.env
    summary["num_segments"] = len(dataset)

    # episodes_tokens is a list of per-episode token arrays (including end row)
    episodes = getattr(dataset, "episodes_tokens", [])
    summary["num_episodes"] = len(episodes)

    if episodes:
        ep_lengths = [int(ep.shape[0] - 1) for ep in episodes]  # minus end row
        summary["episode_length_min"] = int(np.min(ep_lengths))
        summary["episode_length_max"] = int(np.max(ep_lengths))
        summary["episode_length_mean"] = float(np.mean(ep_lengths))

        # Episode length histogram
        plt.figure(figsize=(6, 4))
        plt.hist(ep_lengths, bins=20)
        plt.xlabel("Episode length (transitions)")
        plt.ylabel("Count")
        plt.title(f"{args.env} episode length distribution")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "episode_length_hist.png"))
        plt.close()

        # State ID histogram (from episodes, ignoring end row)
        state_ids = []
        for ep in episodes:
            state_ids.extend(ep[:-1, 0].tolist())
        plt.figure(figsize=(6, 4))
        plt.hist(state_ids, bins=dataset.env.observation_space.n)
        plt.xlabel("State ID")
        plt.ylabel("Count")
        plt.title(f"{args.env} state visitation histogram")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "state_hist.png"))
        plt.close()

        ep_returns = getattr(dataset, "episode_rewards", None)
        if isinstance(ep_returns, list) and ep_returns:
            returns = [float(np.sum(r)) for r in ep_returns]
            summary["episode_return_min"] = float(np.min(returns))
            summary["episode_return_max"] = float(np.max(returns))
            summary["episode_return_mean"] = float(np.mean(returns))

            plt.figure(figsize=(6, 4))
            plt.hist(returns, bins=20)
            plt.xlabel("Episode return")
            plt.ylabel("Count")
            plt.title(f"{args.env} episode return distribution")
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, "episode_return_hist.png"))
            plt.close()

        if args.env == "nrm_nav":
            # Cost histogram for nrm_nav (last column)
            costs = []
            for ep in episodes:
                costs.extend(ep[:-1, 3].tolist())
            plt.figure(figsize=(4, 4))
            plt.hist(costs, bins=[-0.5, 0.5, 1.5])
            plt.xticks([0, 1])
            plt.xlabel("Cost")
            plt.ylabel("Count")
            plt.title("Cost distribution")
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, "cost_hist.png"))
            plt.close()
        elif args.env == "cb":
            policy_labels = getattr(dataset, "episode_policy_labels", None)
            if isinstance(policy_labels, list) and policy_labels:
                uniq = sorted(set(policy_labels))
                counts = {name: int(sum(1 for x in policy_labels if x == name)) for name in uniq}
                summary["episode_policy_counts"] = counts
                csv_path = os.path.join(out_dir, "policy_mix_counts.csv")
                with open(csv_path, "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(["policy_name", "count"])
                    for name in uniq:
                        writer.writerow([name, counts[name]])

                plt.figure(figsize=(6, 4))
                plt.bar(list(counts.keys()), list(counts.values()))
                plt.xticks(rotation=20, ha="right")
                plt.xlabel("Policy source")
                plt.ylabel("Episode count")
                plt.title("CB dataset composition by policy source")
                plt.tight_layout()
                plt.savefig(os.path.join(out_dir, "policy_mix_counts.png"))
                plt.close()

    # Position-wise token stats on a subset of segments
    sample_limit = min(len(dataset), 500)
    pos_stats = {}
    for i in range(sample_limit):
        x, _, _ = dataset[i]
        x_np = x.numpy()
        for pos in range(dataset.joined_dim):
            vals = x_np[pos :: dataset.joined_dim]
            st = pos_stats.setdefault(
                pos, {"min": float("inf"), "max": float("-inf"), "values": []}
            )
            st["min"] = min(st["min"], float(vals.min()))
            st["max"] = max(st["max"], float(vals.max()))
            st["values"].extend(vals.tolist())

    for pos, st in pos_stats.items():
        vals = np.array(st["values"])
        st["mean"] = float(vals.mean()) if vals.size else None
        st["std"] = float(vals.std()) if vals.size else None
        st["unique_count"] = int(len(np.unique(vals))) if vals.size else 0
        # drop raw values to keep JSON small
        st.pop("values", None)
    summary["pos_token_stats"] = pos_stats

    # Constraint satisfaction using provided LTL formula(s)
    dfa_list = raw_dfa if isinstance(raw_dfa, list) else [raw_dfa]
    seg_sat_results = []
    ep_sat_results = []
    ep_sat_vectors = []

    for idx, dfa in enumerate(dfa_list):
        # satisfaction on segments (dataset items)
        n_seg = min(len(dataset), 1000)
        seg_sats = []
        for i in range(n_seg):
            x, _, _ = dataset[i]
            x_eval = x.unsqueeze(0)
            sat = adapter.batch_check_dfa_sat(x_eval, dfa)
            seg_sats.append(float(sat[0].item()))
        seg_sat_results.append(
            {
                "dfa_index": idx,
                "mean_sat_segments": float(np.mean(seg_sats)) if seg_sats else None,
                "num_segments_checked": n_seg,
            }
        )

        # satisfaction on full episodes
        if episodes:
            n_eps = min(len(episodes), 1000)
            ep_sats = []
            for ei in range(n_eps):
                ep = episodes[ei]
                flat = torch.from_numpy(ep.astype(np.int64).reshape(-1))
                flat = flat.unsqueeze(0)
                sat = adapter.batch_check_dfa_sat(flat, dfa)
                ep_sats.append(float(sat[0].item()))
            ep_arr = np.asarray(ep_sats, dtype=np.float32)
            ep_sat_vectors.append(ep_arr)
            ep_sat_results.append(
                {
                    "dfa_index": idx,
                    "mean_sat_episodes": float(np.mean(ep_sats)) if ep_sats else None,
                    "sat_episode_count": int(np.sum(ep_arr >= 0.5)) if ep_sats else 0,
                    "num_episodes_checked": n_eps,
                }
            )

    summary["constraint_satisfaction_segments"] = seg_sat_results
    summary["constraint_satisfaction_episodes"] = ep_sat_results

    if ep_sat_vectors:
        n = min(len(v) for v in ep_sat_vectors)
        if n > 0:
            sat_mat = np.stack([v[:n] >= 0.5 for v in ep_sat_vectors], axis=1)
            sat_all = np.all(sat_mat, axis=1)
            sat_any = np.any(sat_mat, axis=1)
            summary["constraint_satisfaction_all_formulas"] = {
                "num_episodes_checked": int(n),
                "sat_all_count": int(np.sum(sat_all)),
                "sat_all_rate": float(np.mean(sat_all)),
                "sat_any_count": int(np.sum(sat_any)),
                "sat_any_rate": float(np.mean(sat_any)),
            }

    # Explicit episode outcome analysis (most relevant for CB).
    if args.env == "cb":
        ep_rewards = getattr(dataset, "episode_rewards", [])
        if ep_rewards:
            env_cfg = getattr(getattr(dataset, "env", None), "cfg", None)
            max_steps_cfg = int(getattr(env_cfg, "max_steps", 200))
            step_r = float(getattr(env_cfg, "step_reward", -0.01))
            goal_r = float(getattr(env_cfg, "goal_reward", 1.0))
            bomb_r = float(getattr(env_cfg, "bomb_reward", -1.0))
            # Terminal reward levels in this env:
            # goal terminal ~= step_r + goal_r, bomb terminal ~= step_r + bomb_r
            goal_thresh = 0.5 * (step_r + goal_r)
            bomb_thresh = 0.5 * (step_r + bomb_r)

            outcome_counts = {"goal": 0, "bomb_hit": 0, "timeout": 0, "other": 0}
            for rew in ep_rewards:
                if len(rew) == 0:
                    outcome_counts["other"] += 1
                    continue
                T = int(len(rew))
                last = float(rew[-1])
                if T >= max_steps_cfg:
                    outcome_counts["timeout"] += 1
                elif last >= goal_thresh:
                    outcome_counts["goal"] += 1
                elif last <= bomb_thresh:
                    outcome_counts["bomb_hit"] += 1
                else:
                    outcome_counts["other"] += 1

            total = float(sum(outcome_counts.values()))
            safe_count = int(outcome_counts["goal"] + outcome_counts["timeout"] + outcome_counts["other"])
            outcome_rates = {f"{k}_rate": (v / total if total > 0 else 0.0) for k, v in outcome_counts.items()}
            outcome_rates["safe_rate"] = safe_count / total if total > 0 else 0.0
            outcome_rates["bomb_hit_rate"] = outcome_counts["bomb_hit"] / total if total > 0 else 0.0

            summary["episode_outcomes"] = {
                "counts": outcome_counts,
                "rates": outcome_rates,
                "safe_count": safe_count,
                "total_episodes": int(total),
            }

            sat_all_rate = (
                summary.get("constraint_satisfaction_all_formulas", {}).get("sat_all_rate", None)
            )
            if sat_all_rate is not None:
                diff = float(sat_all_rate) - float(outcome_rates["safe_rate"])
                summary["constraint_outcome_consistency"] = {
                    "constraint_sat_all_rate": float(sat_all_rate),
                    "empirical_safe_rate": float(outcome_rates["safe_rate"]),
                    "rate_gap_sat_minus_safe": float(diff),
                    "potential_semantic_mismatch": bool(abs(diff) > 0.05),
                    "note": (
                        "Large gap often means DFA propositions are defined on pre-action states, "
                        "while bomb hits are post-action outcomes."
                    ),
                }

            # Save explicit outcome table
            outcomes_csv = os.path.join(out_dir, "episode_outcomes.csv")
            with open(outcomes_csv, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["outcome", "count", "rate"])
                for k in ["goal", "bomb_hit", "timeout", "other"]:
                    writer.writerow([k, outcome_counts[k], outcome_rates[f"{k}_rate"]])
                writer.writerow(["safe", safe_count, outcome_rates["safe_rate"]])

            # Plot outcome distribution
            labels = ["goal", "bomb_hit", "timeout", "other", "safe"]
            values = [
                outcome_counts["goal"],
                outcome_counts["bomb_hit"],
                outcome_counts["timeout"],
                outcome_counts["other"],
                safe_count,
            ]
            plt.figure(figsize=(7, 4))
            plt.bar(labels, values)
            plt.ylabel("Episode count")
            plt.title("CB dataset episode outcomes")
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, "episode_outcomes_bar.png"))
            plt.close()

    # Save summary JSON
    summary_path = os.path.join(out_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Dataset analysis saved to: {summary_path}")


def replay_dataset_episode(args, dataset):
    """
    Visually replay a single offline dataset episode in the environment.

    Uses dataset.episodes_tokens[replay_episode_index] and the associated env.
    """

    episodes = getattr(dataset, "episodes_tokens", [])
    if not episodes:
        print("Dataset has no episodes_tokens; nothing to replay.")
        return

    ep_idx = getattr(args, "replay_episode_index", 0)
    if ep_idx < 0 or ep_idx >= len(episodes):
        print(f"Invalid replay_episode_index {ep_idx}; dataset has {len(episodes)} episodes.")
        return

    env = dataset.env
    ep_tokens = episodes[ep_idx]

    print(f"Replaying dataset episode {ep_idx} in env '{args.env}'")
    obs, _ = env.reset(seed=getattr(args, "seed", None))
    print("Initial observation (state index):", obs)
    if args.env == "cb":
        print(env.render(mode="ansi"))
    else:
        print(env.render())

    total_reward = 0.0
    for t, row in enumerate(ep_tokens[:-1]):  # skip end row
        state_token = int(row[0])
        action = int(row[1])

        obs, r, done, info = env.step(action)
        total_reward += r

        print(f"\nStep {t}:")
        print(
            f"  dataset_state_token={state_token}, action={action}, reward={r:.3f}, done={done}, info={info}"
        )
        if args.env == "cb":
            print(env.render(mode="ansi"))
        else:
            print(env.render())

        if done:
            break

    print(f"\nEpisode replay finished. Total reward (replayed): {total_reward:.3f}")


def get_arg_parser(add_help=True):
    p = argparse.ArgumentParser(add_help=add_help)

    p.add_argument("--num_episodes", type=int, default=2000)
    p.add_argument("--max_steps", type=int, default=200)
    p.add_argument("--block_size", type=int, default=128)
    p.add_argument("--discount", type=float, default=0.99)
    p.add_argument("--stochastic", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--env", type=str, choices=["cb", "nrm_nav", "frozenlake", "dsrl"], default="cb")
    p.add_argument(
        "--smoke",
        action="store_true",
        help="Fast CPU validation mode that caps episodes/epochs/model size.",
    )

    p.add_argument("--n_layer", type=int, default=4)
    p.add_argument("--n_head", type=int, default=4)
    p.add_argument("--n_embd", type=int, default=64)
    p.add_argument("--embd_pdrop", type=float, default=0.1)
    p.add_argument("--resid_pdrop", type=float, default=0.1)
    p.add_argument("--attn_pdrop", type=float, default=0.1)

    p.add_argument("--action_weight", type=float, default=1.0)
    p.add_argument("--reward_weight", type=float, default=0.0)
    p.add_argument("--value_weight", type=float, default=0.0)

    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--grad_clip", type=float, default=1.0)

    p.add_argument(
        "--frozenlake_map_size",
        type=str,
        choices=["4x4", "8x8"],
        default="4x4",
        help="FrozenLake map size (used when --env frozenlake).",
    )
    p.add_argument(
        "--frozenlake_is_slippery",
        action="store_true",
        help="Enable slippery stochastic transitions in FrozenLake.",
    )
    p.add_argument(
        "--policy_mix",
        type=float,
        default=0.0,
        help=(
            "For FrozenLake dataset generation, fraction of scripted-policy episodes. "
            "0.0=random only, 1.0=scripted only."
        ),
    )
    p.add_argument(
        "--cb_policy_mix_spec",
        type=str,
        default="random:1.0",
        help=(
            "ColourBomb dataset policy mix specification. "
            "Format: name:weight[,name:weight...]. "
            "Supported names: random, shortest_safe, longest_safe, shortest_any, longest_any."
        ),
    )
    p.add_argument(
        "--cb_policy_mix_sampling",
        type=str,
        choices=["fixed", "normal"],
        default="fixed",
        help=(
            "How to sample episode-level policy source in ColourBomb dataset generation: "
            "'fixed' uses constant mix probabilities, 'normal' samples per-policy weights "
            "from Normal(mean,std) then normalizes."
        ),
    )
    p.add_argument(
        "--cb_policy_mix_normal_spec",
        type=str,
        default=None,
        help=(
            "Optional per-policy normal parameters for --cb_policy_mix_sampling normal. "
            "Format: name:mean:std[,name:mean:std...]. "
            "Example: random:0.7:0.1,shortest_safe:0.2:0.08,longest_safe:0.1:0.05"
        ),
    )
    p.add_argument(
        "--cb_state_semantics",
        type=str,
        choices=["pre", "post"],
        default="post",
        help=(
            "Which state to place in the CB transition token state slot: "
            "'pre' uses state before action (legacy), "
            "'post' uses next state after action (recommended; aligns with hazards/outcomes)."
        ),
    )
    p.add_argument(
        "--cb_longest_path_max_expansions",
        type=int,
        default=500000,
        help="DFS expansion budget when computing ColourBomb longest-path scripted trajectories.",
    )
    p.add_argument(
        "--frozenlake_use_position_props",
        action="store_true",
        help="Enable FrozenLake position-bin proposition expansion when building spec formulas.",
    )
    p.add_argument(
        "--dsrl_dataset_path",
        type=str,
        default=None,
        help="Path to DSRL HDF5 dataset file. If omitted, resolves from local DSRL catalog.",
    )
    p.add_argument(
        "--dsrl_dataset_key",
        type=str,
        default="PointGoal1",
        help="DSRL dataset key (e.g., PointGoal1, PointCircle1, AntVelocity).",
    )
    p.add_argument(
        "--dsrl_state_bins",
        type=int,
        default=128,
        help="Number of discrete bins used for DSRL state-token discretization.",
    )
    p.add_argument(
        "--dsrl_action_bins",
        type=int,
        default=16,
        help="Number of discrete bins used for DSRL action-token discretization.",
    )
    p.add_argument(
        "--dsrl_reward_goal_threshold",
        type=float,
        default=0.0,
        help="Reward threshold for goal proposition tokenization in DSRL.",
    )
    p.add_argument(
        "--dsrl_cost_unsafe_threshold",
        type=float,
        default=0.0,
        help="Cost threshold above which DSRL transition is tagged unsafe.",
    )
    p.add_argument(
        "--dsrl_cost_unsafe_quantile",
        type=float,
        default=None,
        help=(
            "If set in [0,1], compute unsafe threshold from DSRL cost quantile "
            "(over positive costs when available). Overrides --dsrl_cost_unsafe_threshold."
        ),
    )
    p.add_argument(
        "--dsrl_download",
        action="store_true",
        help="Allow downloading DSRL dataset from URL resolved by catalog when local file is missing.",
    )

    p.add_argument("--ltl_formula", type=str, default=None)
    p.add_argument("--ltl_formulas", type=str, nargs="+", default=None, help="List of LTL formulas")
    p.add_argument(
        "--spec",
        type=str,
        default=None,
        help=(
            "Named spec preset for the selected env. Runtime support currently: "
            "cb, nrm_nav, frozenlake, dsrl."
        ),
    )
    p.add_argument(
        "--dfa_mode",
        type=str,
        choices=["single", "product", "multi"],
        default="product",
        help="How to combine multiple formulas: single (first only), product DFA, or multi (separate DFAs with averaged loss)",
    )
    p.add_argument(
        "--use_safe_dfa",
        action="store_true",
        help="Build simple safety DFA for G(!unsafe) formulas",
    )
    p.add_argument(
        "--dfa_backend",
        type=str,
        choices=["auto", "ltlf", "template"],
        default="auto",
        help=(
            "DFA construction backend: auto (template when supported, else ltlf), "
            "ltlf (generic compiler), template (only supported structured formulas)."
        ),
    )
    p.add_argument("--constraint_dims", type=int, nargs="+", default=[0])

    p.add_argument("--num_samples", type=int, default=10)
    p.add_argument("--temperature", type=float, default=0.5)
    p.add_argument("--alpha", type=float, default=0.4)
    p.add_argument(
        "--logic_sample_weighting",
        type=str,
        choices=["importance", "uniform"],
        default="importance",
        help="How to aggregate sampled-trace acceptance probabilities.",
    )
    p.add_argument(
        "--logic_acceptance_floor_mode",
        type=str,
        choices=["clamp", "add", "none"],
        default=None,
        help=(
            "Stabilization before log in logic loss. "
            "If unset, defaults to 'clamp' unless --no_logic_clamp is used."
        ),
    )
    p.add_argument(
        "--logic_state_only",
        action="store_true",
        help="Evaluate logic loss only on state-token positions.",
    )
    p.add_argument(
        "--logic_report_stats",
        action="store_true",
        help="Print epoch-level acceptance diagnostics for logic loss.",
    )

    p.add_argument(
        "--logic_eps",
        type=float,
        default=1e-10,
        help="Epsilon used to clamp acceptance probabilities before log in logic loss; <=0 disables clamping.",
    )
    p.add_argument(
        "--no_logic_clamp",
        action="store_true",
        help="Disable epsilon clamp in logic loss (allow log(0) with -inf).",
    )

    p.add_argument(
        "--inspect_dfa_only",
        action="store_true",
        help=(
            "Build adapter and DFA(s), print their sizes, and exit "
            "without training or evaluation."
        ),
    )
    p.add_argument(
        "--no_end_state_hack",
        action="store_true",
        help="Deprecated no-op. Canonical explicit END semantics are always used.",
    )
    p.add_argument(
        "--append_end_token_to_dfa",
        action="store_true",
        help="Deprecated no-op. Canonical explicit END semantics are always used.",
    )

    p.add_argument(
        "--analyze_dataset_only",
        action="store_true",
        help=(
            "Build dataset, adapter and DFA(s), run an in-depth dataset "
            "analysis (including constraint satisfaction), save plots, and "
            "exit without training."
        ),
    )
    p.add_argument(
        "--replay_dataset_episode",
        action="store_true",
        help=(
            "Replay a single offline dataset episode step by step in the "
            "environment and exit without training."
        ),
    )
    p.add_argument(
        "--replay_episode_index",
        type=int,
        default=0,
        help="Index of the dataset episode to replay when --replay_dataset_episode is set.",
    )
    p.add_argument(
        "--run_dir",
        type=str,
        default=None,
        help="Explicit run directory. If unset, uses runs/<env>/<UTC timestamp>/.",
    )
    p.add_argument(
        "--base_runs_dir",
        type=str,
        default="runs",
        help="Base directory for auto-created run directories.",
    )
    p.add_argument(
        "--save_path",
        type=str,
        default=None,
        help="Deprecated alias for --run_dir.",
    )
    p.add_argument(
        "--no_eval_after_train",
        action="store_true",
        help="Skip post-training rollout evaluation (metrics fields remain null).",
    )
    p.add_argument(
        "--eval_num_episodes",
        type=int,
        default=100,
        help="Number of episodes for post-training rollout evaluation.",
    )
    p.add_argument(
        "--eval_max_steps",
        type=int,
        default=None,
        help="Max steps per evaluation episode (default: env max_steps).",
    )
    p.add_argument(
        "--decoding_mode",
        type=str,
        choices=["greedy", "beam", "constrained_beam"],
        default="greedy",
        help="Decoding mode for rollout evaluation.",
    )
    p.add_argument(
        "--target_shift",
        type=str,
        choices=["token", "transition"],
        default="token",
        help=(
            "Training target alignment: token=next-token prediction (recommended), "
            "transition=legacy next-transition prediction."
        ),
    )
    p.add_argument("--beam_width", type=int, default=4)
    p.add_argument("--plan_horizon", type=int, default=2)
    p.add_argument("--sat_rerank_weight", type=float, default=1.0)
    p.add_argument(
        "--hard_prune_reject_sink",
        action="store_true",
        help="In constrained beam mode, prune beams that enter reject sink states.",
    )
    p.add_argument(
        "--inspect_output_dir",
        type=str,
        default=None,
        help="Output directory for DFA inspection artifacts (summary, DOT, optional PNG).",
    )
    p.add_argument(
        "--save_plots",
        action="store_true",
        help="Save evaluation plots under <run_dir>/plots/.",
    )

    return p


def parse_args():
    return get_arg_parser().parse_args()


if __name__ == "__main__":
    warnings.warn(
        "train_cb.py is deprecated. Use scripts/train.py instead.",
        DeprecationWarning,
    )
    args = parse_args()
    train(args)
