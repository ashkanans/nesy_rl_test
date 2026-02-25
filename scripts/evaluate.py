import argparse
import copy
import os
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from planning.eval_runtime import (
    DecodingConfig,
    apply_smoke_mode,
    ensure_run_dir,
    evaluate_policy_rollouts,
    save_evaluation_artifacts,
    set_global_seed,
    spec_label_from_args,
    summarize_dfa_bundle,
    warn_if_train_fallback,
)
from train_cb import (
    build_adapter_and_dfa,
    build_dataset,
    build_model,
    get_arg_parser,
    resolve_formulas,
    train,
)


def _infer_vocab_size_from_state_dict(state_dict, transition_dim):
    if "tok_emb.weight" not in state_dict:
        return None
    rows = int(state_dict["tok_emb.weight"].shape[0])
    return int((rows - 1) // transition_dim)


def load_model_from_checkpoint(args, dataset, checkpoint, device):
    if "model_state_dict" not in checkpoint:
        raise ValueError("Checkpoint does not include model_state_dict")

    state_dict = checkpoint["model_state_dict"]
    cfg = checkpoint.get("config", {})

    model_args = copy.deepcopy(args)
    for key in [
        "block_size",
        "n_layer",
        "n_head",
        "n_embd",
        "action_weight",
        "reward_weight",
        "value_weight",
        "embd_pdrop",
        "resid_pdrop",
        "attn_pdrop",
    ]:
        if key in cfg:
            setattr(model_args, key, cfg[key])

    transition_dim = int(cfg.get("transition_dim", dataset.joined_dim))
    inferred_vocab = _infer_vocab_size_from_state_dict(state_dict, transition_dim)
    if inferred_vocab is None:
        inferred_vocab = int(cfg.get("vocab_size", dataset.env.observation_space.n))

    model = build_model(model_args, dataset, vocab_size=inferred_vocab)
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    return model


def parse_eval_args():
    parent = get_arg_parser(add_help=False)
    parser = argparse.ArgumentParser(parents=[parent], description="Evaluate checkpoint rollouts")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to a training checkpoint (*.pt). Required unless --allow_train_fallback is set.",
    )
    parser.add_argument(
        "--allow_train_fallback",
        action="store_true",
        help="If checkpoint is missing, run training first and evaluate the resulting model.",
    )
    return parser


def main():
    parser = parse_eval_args()
    args = parser.parse_args()
    args = apply_smoke_mode(args)

    if args.checkpoint is None and not args.allow_train_fallback:
        parser.error("--checkpoint is required unless --allow_train_fallback is set.")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    set_global_seed(args.seed)

    dataset = build_dataset(args)
    adapter, _, raw_dfa = build_adapter_and_dfa(args, dataset)

    run_dir, run_id, ts = ensure_run_dir(args.env, run_dir=args.run_dir, base_dir=args.base_runs_dir)

    spec_name = spec_label_from_args(args)
    formulas = resolve_formulas(args, dataset=dataset)
    dfa_summary = summarize_dfa_bundle(raw_dfa, spec_name=spec_name, formulas=formulas, dfa_mode=args.dfa_mode)

    checkpoint_path = args.checkpoint
    if checkpoint_path is not None and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model = load_model_from_checkpoint(args, dataset, checkpoint, device)
    else:
        if not args.allow_train_fallback:
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        warn_if_train_fallback(True)
        model, _, _, dataset, raw_dfa = train(args, return_state=True)
        checkpoint_path = None

    decoding_cfg = DecodingConfig(
        mode=args.decoding_mode,
        beam_width=args.beam_width,
        plan_horizon=args.plan_horizon,
        sat_rerank_weight=args.sat_rerank_weight,
        hard_prune_reject_sink=args.hard_prune_reject_sink,
        target_shift=args.target_shift,
    )

    t0 = time.time()
    metrics, rollout_stats = evaluate_policy_rollouts(
        model=model,
        adapter=adapter,
        raw_dfa=raw_dfa,
        dataset=dataset,
        env_name=args.env,
        spec_name=spec_name,
        seed=args.seed,
        checkpoint_path=checkpoint_path,
        num_episodes=args.eval_num_episodes,
        max_steps=args.eval_max_steps,
        decoding_cfg=decoding_cfg,
    )
    metrics["runtime_sec"] = float(time.time() - t0)
    metrics["run_id"] = run_id
    metrics["timestamp_utc"] = ts

    save_evaluation_artifacts(
        run_dir,
        metrics,
        dfa_summary,
        rollout_stats,
        save_plots=getattr(args, "save_plots", False),
    )
    print(f"Saved evaluation artifacts to {run_dir}")


if __name__ == "__main__":
    main()
