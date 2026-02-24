from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from planning.eval_runtime import ensure_run_dir, write_json
from specs import get_spec
from specs.suites import get_suite
from train_cb import get_arg_parser


def _parse_args():
    parent = get_arg_parser(add_help=False)
    parser = argparse.ArgumentParser(
        parents=[parent],
        description="Evaluate a suite of specification presets for one environment.",
    )
    parser.add_argument("--suite", type=str, default="v1", help="Suite name from specs/suites.py")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument(
        "--allow_train_fallback",
        action="store_true",
        help="Allow evaluation to train inline when checkpoint is missing.",
    )
    parser.add_argument("--suite_run_dir", type=str, default=None)
    parser.add_argument(
        "--max_suite_items",
        type=int,
        default=None,
        help="Optional cap on number of suite entries (useful for quick smoke).",
    )
    args = parser.parse_args()

    if args.spec is not None or args.ltl_formula is not None or args.ltl_formulas is not None:
        raise ValueError("--suite cannot be combined with --spec/--ltl_formula(s).")
    if args.checkpoint is None and not args.allow_train_fallback:
        raise ValueError("--checkpoint is required unless --allow_train_fallback is set.")
    if args.env not in {"cb", "frozenlake", "nrm_nav"}:
        raise ValueError("eval_suite currently supports env in {cb, frozenlake, nrm_nav}.")
    return args


def _run_eval_subprocess(args, preset: str, run_dir: str) -> dict:
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "evaluate.py"),
        "--env",
        args.env,
        "--seed",
        str(args.seed),
        "--spec",
        preset,
        "--run_dir",
        run_dir,
        "--dfa_mode",
        args.dfa_mode,
    ]
    if args.use_safe_dfa:
        cmd.append("--use_safe_dfa")
    if args.smoke:
        cmd.append("--smoke")
    if args.checkpoint is not None:
        cmd.extend(["--checkpoint", args.checkpoint])
    if args.allow_train_fallback:
        cmd.append("--allow_train_fallback")

    if args.eval_num_episodes is not None:
        cmd.extend(["--eval_num_episodes", str(args.eval_num_episodes)])
    if args.eval_max_steps is not None:
        cmd.extend(["--eval_max_steps", str(args.eval_max_steps)])
    cmd.extend(["--beam_width", str(args.beam_width)])
    cmd.extend(["--plan_horizon", str(args.plan_horizon)])
    cmd.extend(["--sat_rerank_weight", str(args.sat_rerank_weight)])
    cmd.extend(["--decoding_mode", str(args.decoding_mode)])
    if args.hard_prune_reject_sink:
        cmd.append("--hard_prune_reject_sink")

    # Keep fallback-training behavior consistent with training defaults when used.
    cmd.extend(["--num_episodes", str(args.num_episodes)])
    cmd.extend(["--max_steps", str(args.max_steps)])
    cmd.extend(["--block_size", str(args.block_size)])
    cmd.extend(["--batch_size", str(args.batch_size)])
    cmd.extend(["--epochs", str(args.epochs)])
    cmd.extend(["--n_layer", str(args.n_layer)])
    cmd.extend(["--n_head", str(args.n_head)])
    cmd.extend(["--n_embd", str(args.n_embd)])
    cmd.extend(["--temperature", str(args.temperature)])
    cmd.extend(["--num_samples", str(args.num_samples)])
    cmd.extend(["--alpha", str(args.alpha)])
    cmd.extend(["--discount", str(args.discount)])

    if args.env == "frozenlake":
        cmd.extend(["--frozenlake_map_size", args.frozenlake_map_size])
        if args.frozenlake_is_slippery:
            cmd.append("--frozenlake_is_slippery")
        cmd.extend(["--policy_mix", str(args.policy_mix)])

    subprocess.run(cmd, check=True, cwd=REPO_ROOT)
    with open(os.path.join(run_dir, "metrics.json"), "r") as f:
        return json.load(f)


def _write_csv(path: str, rows: list[dict]):
    if not rows:
        with open(path, "w", newline="") as f:
            f.write("")
        return
    keys = sorted({k for row in rows for k in row.keys()})
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _print_table(rows: list[dict]):
    headers = ["preset", "formula_index", "return_mean", "violation_rate", "satisfaction_rate"]
    widths = {h: len(h) for h in headers}
    for row in rows:
        for h in headers:
            widths[h] = max(widths[h], len(str(row.get(h, ""))))
    sep = " | "
    line = sep.join(h.ljust(widths[h]) for h in headers)
    print(line)
    print("-" * len(line))
    for row in rows:
        print(sep.join(str(row.get(h, "")).ljust(widths[h]) for h in headers))


def main():
    args = _parse_args()
    presets = get_suite(args.env, args.suite)
    if args.smoke:
        presets = presets[:2]
    if args.max_suite_items is not None:
        presets = presets[: int(args.max_suite_items)]

    if args.suite_run_dir is None:
        root, _, _ = ensure_run_dir(args.env, run_dir=None, base_dir=args.base_runs_dir)
        suite_dir = os.path.join(root, f"suite_{args.suite}")
    else:
        suite_dir = args.suite_run_dir
    os.makedirs(suite_dir, exist_ok=True)

    rows = []
    row_idx = 0
    for preset in presets:
        spec_cfg = get_spec(args.env, preset)
        formulas = list(spec_cfg.get("formulas", []))
        for formula_idx, _ in enumerate(formulas):
            run_dir = os.path.join(suite_dir, "runs", f"{row_idx:03d}_{preset}_f{formula_idx}")
            metrics = _run_eval_subprocess(args, preset=preset, run_dir=run_dir)
            rows.append(
                {
                    "suite": args.suite,
                    "env": args.env,
                    "preset": preset,
                    "formula_index": int(formula_idx),
                    "formula": formulas[formula_idx],
                    "description": spec_cfg.get("description"),
                    "run_dir": run_dir,
                    "return_mean": metrics.get("return_mean"),
                    "violation_rate": metrics.get("violation_rate"),
                    "satisfaction_rate": metrics.get("satisfaction_rate"),
                    "runtime_sec": metrics.get("runtime_sec"),
                    "seed": metrics.get("seed"),
                    "checkpoint_path": metrics.get("checkpoint_path"),
                }
            )
            row_idx += 1

    payload = {
        "suite": args.suite,
        "env": args.env,
        "num_rows": len(rows),
        "rows": rows,
    }
    json_path = os.path.join(suite_dir, "suite_metrics.json")
    csv_path = os.path.join(suite_dir, "suite_metrics.csv")
    write_json(json_path, payload)
    _write_csv(csv_path, rows)

    _print_table(rows)
    print(f"Suite artifacts saved to {suite_dir}")
    print(f"JSON: {json_path}")
    print(f"CSV: {csv_path}")


if __name__ == "__main__":
    main()
