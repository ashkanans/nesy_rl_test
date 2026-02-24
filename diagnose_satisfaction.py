import argparse
import json

import numpy as np

from train_cb import build_adapter_and_dfa, build_dataset


def describe_dataset(ds):
    xs = []
    for i in range(min(len(ds), 100)):
        x, _, _ = ds[i]
        xs.append(x.numpy())
    flat = np.concatenate(xs) if xs else np.array([], dtype=np.int64)
    return {
        "len": len(ds),
        "seq_len": xs[0].shape[0] if xs else 0,
        "min_token": int(flat.min()) if flat.size else None,
        "max_token": int(flat.max()) if flat.size else None,
        "unique_tokens": len(np.unique(flat)) if flat.size else 0,
    }


def satisfaction_on_segments(ds, adapter, dfa, sample_limit=200):
    n = min(len(ds), sample_limit)
    sats = []
    for i in range(n):
        x, _, _ = ds[i]
        sat = adapter.batch_check_dfa_sat(x.unsqueeze(0), dfa)
        sats.append(float(sat[0].item()))
    sats = np.array(sats)
    return {"mean_sat": float(sats.mean()) if len(sats) else None, "num_samples": n}


def satisfaction_on_episodes(ds, adapter, dfa, sample_limit=200):
    episodes = getattr(ds, "episodes_tokens", [])
    n = min(len(episodes), sample_limit)
    sats = []
    for i in range(n):
        flat = episodes[i].reshape(-1)
        sat = adapter.batch_check_dfa_sat(
            np_to_torch(flat).unsqueeze(0),
            dfa,
        )
        sats.append(float(sat[0].item()))
    sats = np.array(sats)
    return {"mean_sat": float(sats.mean()) if len(sats) else None, "num_samples": n}


def np_to_torch(arr):
    import torch

    return torch.from_numpy(arr.astype(np.int64))


def unsafe_fraction(ds, unsafe_ids, sample_limit=200):
    n = min(len(ds), sample_limit)
    hits = 0
    total = 0
    for i in range(n):
        x, _, _ = ds[i]
        total += x.numel()
        hits += sum(int(t.item()) in unsafe_ids for t in x)
    return {"unsafe_rate": hits / total if total else None, "tokens_checked": total}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, choices=["cb", "nrm_nav", "frozenlake"], default="nrm_nav")
    parser.add_argument("--spec", type=str, default=None)
    parser.add_argument("--ltl_formulas", type=str, nargs="+", default=None)
    parser.add_argument(
        "--dfa_mode", type=str, choices=["single", "product", "multi"], default="product"
    )
    parser.add_argument("--num_episodes", type=int, default=200)
    parser.add_argument("--max_steps", type=int, default=50)
    parser.add_argument("--sequence_length", type=int, default=64)
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--eval_mode",
        type=str,
        choices=["episodes", "segments"],
        default="episodes",
        help="Default is full-episode evaluation for meaningful finite-trace satisfaction.",
    )
    args = parser.parse_args()

    if args.spec is None and args.ltl_formulas is None:
        parser.error("Provide --spec or --ltl_formulas.")
    if args.spec is not None and args.ltl_formulas is not None:
        parser.error("Provide either --spec or --ltl_formulas, not both.")

    return args


def main():
    args = parse_args()

    class DummyArgs:
        pass

    dummy = DummyArgs()
    dummy.env = args.env
    dummy.num_episodes = args.num_episodes
    dummy.max_steps = args.max_steps
    dummy.block_size = args.sequence_length
    dummy.discount = 0.99
    dummy.stochastic = args.stochastic
    dummy.seed = args.seed
    dummy.spec = args.spec
    dummy.ltl_formulas = args.ltl_formulas
    dummy.ltl_formula = None
    dummy.dfa_mode = args.dfa_mode
    dummy.constraint_dims = [0]
    dummy.use_safe_dfa = True
    dummy.inspect_dfa_only = False
    dummy.save_path = "cb_runs"
    dummy.inspect_output_dir = None
    dummy.frozenlake_map_size = "4x4"
    dummy.frozenlake_is_slippery = False
    dummy.policy_mix = 0.0
    dummy.frozenlake_use_position_props = False

    ds = build_dataset(dummy)
    adapter, _, raw_dfa = build_adapter_and_dfa(dummy, ds)

    report = {}
    report["dataset"] = describe_dataset(ds)
    report["adapter"] = {
        "num_token_ids": adapter.num_token_ids,
        "end_token_id": adapter.end_token_id,
        "max_num_bins": adapter.max_num_bins,
        "num_symbols": adapter.num_symbols,
    }
    report["dfa_mode"] = args.dfa_mode
    report["eval_mode"] = args.eval_mode
    report["spec"] = args.spec
    report["ltl_formulas"] = args.ltl_formulas

    target_dfas = raw_dfa if isinstance(raw_dfa, list) else [raw_dfa]
    sat_reports = []
    for dfa in target_dfas:
        if args.eval_mode == "episodes":
            sat_reports.append(satisfaction_on_episodes(ds, adapter, dfa))
        else:
            sat_reports.append(satisfaction_on_segments(ds, adapter, dfa))
    report["satisfaction"] = sat_reports

    if args.env == "nrm_nav":
        unsafe_ids = {11, 18}
        report["unsafe"] = unsafe_fraction(ds, unsafe_ids)

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
