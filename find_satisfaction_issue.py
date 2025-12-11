"""
Diagnose why satisfaction stays zero.
Checks:
 - unsafe state IDs
 - token ranges per position
 - DFA symbols and transitions coverage for dataset tokens
 - satisfaction on ground-truth tokens (with/without masking)
"""
import argparse
import json
import numpy as np
import torch

from train_cb import build_dataset, build_adapter_and_dfa
from dfa_adapter import TTDFAAdapter


def pos_token_stats(ds, transition_dim, sample_limit=100):
    # assumes flat tokens
    n = min(len(ds), sample_limit)
    stats = {}
    for i in range(n):
        x, _, _ = ds[i]
        for pos in range(transition_dim):
            vals = x[pos::transition_dim].numpy()
            st = stats.setdefault(pos, {"min": 1e9, "max": -1e9, "unique": set()})
            st["min"] = min(st["min"], vals.min())
            st["max"] = max(st["max"], vals.max())
            st["unique"].update(vals.tolist())
    for pos, st in stats.items():
        st["unique_count"] = int(len(st["unique"]))
        st["unique_sample"] = [int(v) for v in sorted(list(st["unique"]))[:10]]
        st.pop("unique", None)
    return stats


def dfa_symbol_coverage(adapter, dfa):
    # which symbols are in DFA dictionary
    symbols = set(dfa.dictionary_symbols)
    return {"num_symbols": int(len(symbols)), "has_end": "end" in symbols}


def satisfaction_on_dataset(ds, adapter, dfa, mask_to_state_only=False, sample_limit=200):
    n = min(len(ds), sample_limit)
    sats = []
    for i in range(n):
        x, _, _ = ds[i]
        sat = adapter.batch_check_dfa_sat(x.unsqueeze(0), dfa, mask_to_state_only=mask_to_state_only)
        sats.append(float(sat[0].item()))
    sats = np.array(sats)
    return {"mean_sat": float(sats.mean()) if len(sats) else None, "num_samples": n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", type=str, default="nrm_nav")
    ap.add_argument("--ltl_formulas", nargs="+", required=True)
    ap.add_argument("--dfa_mode", type=str, choices=["single", "product", "multi"], default="product")
    ap.add_argument("--num_episodes", type=int, default=200)
    ap.add_argument("--max_steps", type=int, default=50)
    ap.add_argument("--block_size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    # build dataset/adapter/dfa
    class Dummy:
        pass
    dummy = Dummy()
    dummy.env = args.env
    dummy.num_episodes = args.num_episodes
    dummy.max_steps = args.max_steps
    dummy.block_size = args.block_size
    dummy.discount = 0.99
    dummy.stochastic = False
    dummy.seed = args.seed
    dummy.ltl_formulas = args.ltl_formulas
    dummy.ltl_formula = None
    dummy.dfa_mode = args.dfa_mode
    dummy.constraint_dims = [0]
    dummy.use_safe_dfa = True

    ds = build_dataset(dummy)
    adapter, deep_dfa, raw_dfa = build_adapter_and_dfa(dummy, ds)

    target_dfas = raw_dfa if isinstance(raw_dfa, list) else [raw_dfa]

    report = {}
    report["dataset_len"] = int(len(ds))
    report["transition_dim"] = int(adapter.transition_dim)
    report["pos_token_stats"] = pos_token_stats(ds, adapter.transition_dim)
    report["dfa_symbol_coverage"] = [dfa_symbol_coverage(adapter, d) for d in target_dfas]
    report["satisfaction_unmasked"] = [satisfaction_on_dataset(ds, adapter, d, mask_to_state_only=False) for d in target_dfas]
    report["satisfaction_masked_state_only"] = [satisfaction_on_dataset(ds, adapter, d, mask_to_state_only=True) for d in target_dfas]
    # unsafe coverage if env is nrm_nav
    if args.env == "nrm_nav":
        unsafe_ids = {11, 18}
        hits = 0
        total = 0
        for i in range(min(len(ds), 200)):
            x, _, _ = ds[i]
            vals = x[0::adapter.transition_dim]  # state positions
            hits += sum(int(v.item()) in unsafe_ids for v in vals)
            total += len(vals)
        report["unsafe_rate_states"] = hits / total if total else None

    def _default(o):
        if isinstance(o, (np.integer, )):
            return int(o)
        if isinstance(o, (np.floating, )):
            return float(o)
        return str(o)

    print(json.dumps(report, indent=2, default=_default))


if __name__ == "__main__":
    main()
