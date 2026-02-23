import argparse

import numpy as np
import torch

from train_cb import build_adapter_and_dfa, build_dataset


def describe_dataset(ds):
    xs = []
    for i in range(min(len(ds), 100)):
        x, _, _ = ds[i]
        xs.append(x.numpy())
    flat = np.concatenate(xs)
    return {
        "len": len(ds),
        "seq_len": xs[0].shape[0] if xs else 0,
        "min_token": int(flat.min()) if flat.size else None,
        "max_token": int(flat.max()) if flat.size else None,
        "unique_tokens": len(np.unique(flat)) if flat.size else 0,
    }


def satisfaction_on_dataset(ds, adapter, dfa, sample_limit=200):
    n = min(len(ds), sample_limit)
    sats = []
    for i in range(n):
        x, _, _ = ds[i]
        sat = adapter.batch_check_dfa_sat(x.unsqueeze(0), dfa)
        sats.append(float(sat[0].item()))
    sats = np.array(sats)
    return {"mean_sat": float(sats.mean()) if len(sats) else None, "num_samples": n}


def unsafe_fraction(ds, unsafe_ids, sample_limit=200):
    n = min(len(ds), sample_limit)
    hits = 0
    total = 0
    for i in range(n):
        x, _, _ = ds[i]
        total += x.numel()
        hits += sum(int(t.item()) in unsafe_ids for t in x)
    return {"unsafe_rate": hits / total if total else None, "tokens_checked": total}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, choices=["cb", "nrm_nav"], default="nrm_nav")
    parser.add_argument("--ltl_formulas", type=str, nargs="+", required=True)
    parser.add_argument(
        "--dfa_mode", type=str, choices=["single", "product", "multi"], default="product"
    )
    parser.add_argument("--num_episodes", type=int, default=200)
    parser.add_argument("--max_steps", type=int, default=50)
    parser.add_argument("--sequence_length", type=int, default=64)
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    # build dataset
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
    dummy.ltl_formulas = args.ltl_formulas
    dummy.ltl_formula = None
    dummy.dfa_mode = args.dfa_mode
    dummy.constraint_dims = [0]

    ds = build_dataset(dummy)
    adapter, deep_dfa, raw_dfa = build_adapter_and_dfa(dummy, ds)

    report = {}
    report["dataset"] = describe_dataset(ds)
    report["adapter"] = {
        "num_token_ids": adapter.num_token_ids,
        "max_num_bins": adapter.max_num_bins,
        "num_symbols": adapter.num_symbols,
    }
    report["dfa_mode"] = args.dfa_mode
    if isinstance(raw_dfa, list):
        report["dfas"] = [len(d.dictionary_symbols) for d in raw_dfa]
    else:
        report["dfa_symbols"] = len(raw_dfa.dictionary_symbols)

    # satisfaction on ground-truth tokens
    target_dfas = raw_dfa if isinstance(raw_dfa, list) else [raw_dfa]
    sat_reports = []
    for dfa in target_dfas:
        sat_reports.append(satisfaction_on_dataset(ds, adapter, dfa))
    report["satisfaction"] = sat_reports

    # unsafe hit rate (only for nrm_nav default unsafe ids)
    if args.env == "nrm_nav":
        unsafe_ids = {11, 18}
        report["unsafe"] = unsafe_fraction(ds, unsafe_ids)

    import json

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
