from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datasets.frozenlake_dataset import FrozenLakeSequenceDataset
from specs import get_spec


def parse_args():
    p = argparse.ArgumentParser(
        description="FL-01 FrozenLake 4x4 non-slippery dataset sanity package."
    )
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--episodes_per_seed", type=int, default=200)
    p.add_argument("--max_episode_steps", type=int, default=30)
    p.add_argument("--policy_mix", type=float, default=0.5)
    p.add_argument(
        "--specs",
        type=str,
        nargs="+",
        default=["avoid_holes", "reach_goal", "reach_goal_while_avoid_holes"],
    )
    p.add_argument(
        "--output_root",
        type=str,
        default="runs/frozenlake/fl01_dataset_sanity",
    )
    p.add_argument("--skip_doctor", action="store_true")
    return p.parse_args()


def _write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def _save_dataset_npz(
    dataset: FrozenLakeSequenceDataset,
    out_stem: Path,
    metadata: dict,
):
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    episodes = np.asarray([ep for ep in dataset.episodes_tokens], dtype=object)
    rewards = np.asarray([r for r in dataset.episode_rewards], dtype=object)
    lengths = np.asarray([int(ep.shape[0] - 1) for ep in dataset.episodes_tokens], dtype=np.int32)
    returns = np.asarray(
        [float(np.sum(r)) for r in dataset.episode_rewards],
        dtype=np.float32,
    )
    np.savez_compressed(
        str(out_stem) + ".npz",
        trajectories=episodes,
        rewards=rewards,
        returns=returns,
        lengths=lengths,
        metadata_json=np.asarray(json.dumps(metadata)),
    )
    _write_json(Path(str(out_stem) + ".json"), metadata)


def _run_cmd(cmd: list[str], quiet: bool = False):
    if not quiet:
        subprocess.run(cmd, cwd=REPO_ROOT, check=True)
        return
    proc = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        check=False,
        text=True,
        capture_output=True,
    )
    if proc.returncode != 0:
        if proc.stdout:
            print(proc.stdout)
        if proc.stderr:
            print(proc.stderr, file=sys.stderr)
        raise subprocess.CalledProcessError(proc.returncode, cmd)


def _run_doctor_for_seed(
    seed: int,
    spec: str,
    policy_type: str,
    episodes_per_seed: int,
    max_episode_steps: int,
    policy_mix: float,
    doctor_dir: Path,
) -> Path:
    out_path = doctor_dir / f"{policy_type}_seed{seed}_{spec}.json"
    run_dir = doctor_dir / "runs" / f"{policy_type}_seed{seed}_{spec}"
    run_dir.mkdir(parents=True, exist_ok=True)
    mix_val = 0.0 if policy_type == "random" else policy_mix
    cmd = [
        sys.executable,
        "-m",
        "tools.doctor",
        "--env",
        "frozenlake",
        "--spec",
        spec,
        "--seed",
        str(seed),
        "--num_episodes",
        str(episodes_per_seed),
        "--max_steps",
        str(max_episode_steps),
        "--dataset_sample_size",
        str(episodes_per_seed),
        "--frozenlake_map_size",
        "4x4",
        "--policy_mix",
        str(mix_val),
        "--run_dir",
        str(run_dir),
        "--output_json",
        str(out_path),
    ]
    _run_cmd(cmd, quiet=True)
    return out_path


def _inspect_dfa_for_spec(spec: str, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "train.py"),
        "--env",
        "frozenlake",
        "--spec",
        spec,
        "--seed",
        "0",
        "--num_episodes",
        "32",
        "--max_steps",
        "20",
        "--block_size",
        "8",
        "--frozenlake_map_size",
        "4x4",
        "--policy_mix",
        "0.0",
        "--inspect_dfa_only",
        "--inspect_output_dir",
        str(out_dir),
        "--run_dir",
        str(out_dir / "run"),
    ]
    _run_cmd(cmd, quiet=True)


def _save_csv(path: Path, rows: list[dict], fieldnames: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _render_charts(
    charts_dir: Path,
    composition_rows: list[dict],
    hist_rows: list[dict],
    spec_rows: list[dict],
):
    import matplotlib.pyplot as plt

    charts_dir.mkdir(parents=True, exist_ok=True)

    # dataset_composition.png
    labels = [r["dataset_type"] for r in composition_rows]
    vals = [int(r["total_episodes"]) for r in composition_rows]
    plt.figure(figsize=(6, 4))
    plt.bar(labels, vals)
    plt.ylabel("episodes")
    plt.title("FrozenLake FL-01 dataset composition")
    plt.tight_layout()
    plt.savefig(charts_dir / "dataset_composition.png")
    plt.close()

    # trace_length_hist.png
    bins = [int(r["bin_left"]) for r in hist_rows]
    random_counts = [int(r["random_count"]) for r in hist_rows]
    mixed_counts = [int(r["mixed_count"]) for r in hist_rows]
    width = 0.45
    x = np.arange(len(bins), dtype=np.float32)
    plt.figure(figsize=(9, 4))
    plt.bar(x - width / 2, random_counts, width=width, label="random")
    plt.bar(x + width / 2, mixed_counts, width=width, label="mixed")
    tick_labels = [str(b) for b in bins]
    if len(tick_labels) > 40:
        step = max(1, int(math.ceil(len(tick_labels) / 40)))
        show = np.arange(0, len(tick_labels), step)
        plt.xticks(show, [tick_labels[i] for i in show], rotation=90)
    else:
        plt.xticks(x, tick_labels, rotation=90)
    plt.xlabel("episode length")
    plt.ylabel("count")
    plt.title("FrozenLake FL-01 trace length histogram")
    plt.legend()
    plt.tight_layout()
    plt.savefig(charts_dir / "trace_length_hist.png")
    plt.close()

    # spec_acceptance_bar.png
    specs = sorted({r["spec"] for r in spec_rows})
    random_means = []
    mixed_means = []
    for spec in specs:
        rrow = next(r for r in spec_rows if r["dataset_type"] == "random" and r["spec"] == spec)
        mrow = next(r for r in spec_rows if r["dataset_type"] == "mixed" and r["spec"] == spec)
        random_means.append(float(rrow["mean_acceptance"]))
        mixed_means.append(float(mrow["mean_acceptance"]))
    x = np.arange(len(specs), dtype=np.float32)
    width = 0.35
    plt.figure(figsize=(8, 4))
    plt.bar(x - width / 2, random_means, width=width, label="random")
    plt.bar(x + width / 2, mixed_means, width=width, label="mixed")
    plt.xticks(x, specs, rotation=20, ha="right")
    plt.ylim(-0.05, 1.05)
    plt.ylabel("acceptance rate")
    plt.title("FrozenLake FL-01 spec acceptance on dataset traces")
    plt.legend()
    plt.tight_layout()
    plt.savefig(charts_dir / "spec_acceptance_bar.png")
    plt.close()


def main():
    args = parse_args()

    output_root = Path(args.output_root)
    datasets_dir = output_root / "datasets"
    diagnostics_dir = output_root / "diagnostics"
    charts_dir = output_root / "charts"
    tables_dir = output_root / "tables"
    dfa_dir = diagnostics_dir / "dfa"
    doctor_dir = diagnostics_dir / "doctor"

    for d in [datasets_dir, diagnostics_dir, charts_dir, tables_dir, dfa_dir, doctor_dir]:
        d.mkdir(parents=True, exist_ok=True)

    policy_cfg = {
        "random": 0.0,
        "mixed": float(args.policy_mix),
    }

    lengths_by_type: dict[str, list[int]] = {"random": [], "mixed": []}
    dataset_files = []
    metadata_files = []
    doctor_rows = []
    doctor_files = []

    # 1) Dataset generation and save artifacts
    for policy_type, mix_val in policy_cfg.items():
        for seed in args.seeds:
            ds = FrozenLakeSequenceDataset(
                num_episodes=int(args.episodes_per_seed),
                max_steps=int(args.max_episode_steps),
                sequence_length=max(8, int(args.max_episode_steps) * 4),
                seed=int(seed),
                map_size="4x4",
                is_slippery=False,
                policy_mix=float(mix_val),
            )
            lengths = [int(ep.shape[0] - 1) for ep in ds.episodes_tokens]
            lengths_by_type[policy_type].extend(lengths)
            stem = datasets_dir / f"frozenlake4x4_nonslippery_{policy_type}_seed{seed}"
            metadata = {
                "seed": int(seed),
                "env": "frozenlake",
                "map_size": "4x4",
                "is_slippery": False,
                "policy_type": policy_type,
                "policy_mix": float(mix_val),
                "episode_count_requested": int(args.episodes_per_seed),
                "episode_count_generated": int(len(ds.episodes_tokens)),
                "max_episode_steps": int(args.max_episode_steps),
                "schema_id": str(ds.schema_id),
            }
            _save_dataset_npz(ds, stem, metadata)
            dataset_files.append(str(stem) + ".npz")
            metadata_files.append(str(stem) + ".json")

    # 2) Spec compile + DFA inspect artifacts
    dfa_summary_rows = []
    for spec in args.specs:
        spec_out = dfa_dir / spec
        _inspect_dfa_for_spec(spec, spec_out)
        spec_cfg = get_spec("frozenlake", spec)
        summary_candidates = sorted(spec_out.glob("*_summary.json"))
        dfa_summary_rows.append(
            {
                "spec": spec,
                "formulas": spec_cfg.get("formulas", []),
                "dfa_summary_files": [str(p) for p in summary_candidates],
            }
        )

    # 3) Doctor diagnostics by seed/spec/policy
    if not args.skip_doctor:
        for policy_type in policy_cfg:
            for seed in args.seeds:
                for spec in args.specs:
                    out_path = _run_doctor_for_seed(
                        seed=int(seed),
                        spec=str(spec),
                        policy_type=policy_type,
                        episodes_per_seed=int(args.episodes_per_seed),
                        max_episode_steps=int(args.max_episode_steps),
                        policy_mix=float(args.policy_mix),
                        doctor_dir=doctor_dir,
                    )
                    doctor_files.append(str(out_path))
                    payload = json.loads(out_path.read_text())
                    sat = (
                        payload.get("checks", {})
                        .get("satisfaction_dataset", {})
                        .get("joint_satisfaction_rate")
                    )
                    doctor_rows.append(
                        {
                            "dataset_type": policy_type,
                            "seed": int(seed),
                            "spec": str(spec),
                            "joint_acceptance_rate": None if sat is None else float(sat),
                            "doctor_json": str(out_path),
                        }
                    )

    # 4) Build aggregated tables
    composition_rows = []
    for policy_type in ["random", "mixed"]:
        total_eps = int(len(args.seeds) * args.episodes_per_seed)
        composition_rows.append(
            {
                "dataset_type": policy_type,
                "seed_count": int(len(args.seeds)),
                "episodes_per_seed": int(args.episodes_per_seed),
                "total_episodes": total_eps,
                "map_size": "4x4",
                "is_slippery": False,
                "policy_mix": 0.0 if policy_type == "random" else float(args.policy_mix),
            }
        )

    all_lengths = lengths_by_type["random"] + lengths_by_type["mixed"]
    if all_lengths:
        max_len = int(max(all_lengths))
    else:
        max_len = int(args.max_episode_steps)
    bins = np.arange(0, max_len + 2, dtype=np.int32)
    rnd_hist, _ = np.histogram(np.asarray(lengths_by_type["random"], dtype=np.int32), bins=bins)
    mix_hist, _ = np.histogram(np.asarray(lengths_by_type["mixed"], dtype=np.int32), bins=bins)
    hist_rows = []
    for i in range(len(bins) - 1):
        hist_rows.append(
            {
                "bin_left": int(bins[i]),
                "bin_right": int(bins[i + 1] - 1),
                "random_count": int(rnd_hist[i]),
                "mixed_count": int(mix_hist[i]),
            }
        )

    spec_summary_rows = []
    for policy_type in ["random", "mixed"]:
        for spec in args.specs:
            vals = [
                float(r["joint_acceptance_rate"])
                for r in doctor_rows
                if r["dataset_type"] == policy_type
                and r["spec"] == spec
                and r["joint_acceptance_rate"] is not None
            ]
            spec_summary_rows.append(
                {
                    "dataset_type": policy_type,
                    "spec": spec,
                    "mean_acceptance": float(np.mean(vals)) if vals else None,
                    "std_acceptance": float(np.std(vals)) if vals else None,
                    "num_seeds": int(len(vals)),
                }
            )

    _save_csv(
        tables_dir / "dataset_composition.csv",
        composition_rows,
        [
            "dataset_type",
            "seed_count",
            "episodes_per_seed",
            "total_episodes",
            "map_size",
            "is_slippery",
            "policy_mix",
        ],
    )
    _save_csv(
        tables_dir / "trace_length_hist.csv",
        hist_rows,
        ["bin_left", "bin_right", "random_count", "mixed_count"],
    )
    _save_csv(
        tables_dir / "spec_acceptance_summary.csv",
        spec_summary_rows,
        ["dataset_type", "spec", "mean_acceptance", "std_acceptance", "num_seeds"],
    )
    _save_csv(
        tables_dir / "doctor_per_seed.csv",
        doctor_rows,
        ["dataset_type", "seed", "spec", "joint_acceptance_rate", "doctor_json"],
    )

    _render_charts(charts_dir, composition_rows, hist_rows, spec_summary_rows)

    # 5) Global summary
    file_inventory = {
        "dataset_npz": sorted(dataset_files),
        "dataset_metadata_json": sorted(metadata_files),
        "doctor_json": sorted(doctor_files),
        "dfa_summary": sorted(str(p) for p in dfa_dir.glob("**/*_summary.json")),
        "dfa_dot": sorted(str(p) for p in dfa_dir.glob("**/*.dot")),
        "dfa_png": sorted(str(p) for p in dfa_dir.glob("**/*.png")),
        "tables": sorted(str(p) for p in tables_dir.glob("*.csv")),
        "charts": sorted(str(p) for p in charts_dir.glob("*.png")),
    }

    summary = {
        "job": "FL-01",
        "env": "frozenlake",
        "map_size": "4x4",
        "is_slippery": False,
        "seeds": [int(s) for s in args.seeds],
        "episodes_per_seed": int(args.episodes_per_seed),
        "policy_mix": float(args.policy_mix),
        "dataset_types": ["random", "mixed"],
        "specs": list(args.specs),
        "directories": {
            "output_root": str(output_root),
            "datasets": str(datasets_dir),
            "diagnostics": str(diagnostics_dir),
            "charts": str(charts_dir),
            "tables": str(tables_dir),
        },
        "file_inventory": file_inventory,
        "spec_acceptance_aggregates": spec_summary_rows,
        "dfa_compile_summary": dfa_summary_rows,
    }
    _write_json(output_root / "fl01_summary.json", summary)
    print(f"FL-01 package saved to: {output_root}")


if __name__ == "__main__":
    main()
