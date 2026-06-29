import argparse
import contextlib
import csv
import json
import multiprocessing as mp
import os
import re
import shlex
import subprocess
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from queue import Empty
from types import SimpleNamespace

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
PYTHON = sys.executable


@dataclass(frozen=True)
class Job:
    spec: str
    policy_mix_spec: str
    seed: int
    seed_root: str
    train_root: str


@dataclass
class WorkerConfig:
    worker_id: int
    gpu_id: int | None
    require_gpu: bool
    stop_on_gpu_unavailable: bool
    continue_on_error: bool
    skip_completed: bool
    dry_run: bool
    train_cmd_common: list[str]
    eval_cmd_common: list[str]
    decoding_modes: list[str]
    baseline_keys: list[str]
    logic_alphas: list[float]
    extra_args: list[str]
    save_generated_dataset: bool
    dataset_artifact_dir: str | None
    dataset_artifact_name: str
    skip_dataset_analysis: bool
    num_episodes: int
    max_steps: int
    context_len: int
    stochastic: bool
    cb_longest_path_max_expansions: int
    cb_policy_mix_sampling: str
    cb_policy_mix_normal_spec: str | None
    cb_policy_mix_normal_mean_mode: str
    cb_state_semantics: str
    dfa_mode: str
    use_safe_dfa: bool
    num_action_candidates: int
    knn_k: int
    epochs: int


SUMMARY_FIELDS = [
    "baseline",
    "return_mean",
    "return_std",
    "violation_rate",
    "satisfaction_rate",
    "runtime_sec",
    "env",
    "spec",
    "seed",
    "num_episodes",
    "satisfaction_soft_mean",
    "violation_rate_episode",
    "violation_rate_step",
    "goal_rate",
    "bomb_hit_rate",
    "hazard_hit_rate",
    "decoding_mode",
    "beam_width",
    "model_type",
    "checkpoint_path",
    "satisfaction_source",
    "run_id",
    "timestamp_utc",
    "random_return_mean",
    "random_goal_rate",
    "random_violation_rate",
    "better_than_random",
    "rtg_target",
    "action_loss",
    "logic_loss",
    "success_rate",
    "dataset_size",
    "context_len",
]


def _slug(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", str(text)).strip("_")


def _cmd_arg(cmd: list[str], flag: str, default: str | None = None) -> str | None:
    value = default
    i = 0
    while i < len(cmd):
        if cmd[i] == flag and i + 1 < len(cmd):
            value = cmd[i + 1]
            i += 2
            continue
        i += 1
    return value


def _build_seed_dataset_artifact_name(job: Job, cfg: WorkerConfig) -> str:
    # Keep dataset cache stable per seed+mix+semantics so repeated runs reuse same artifact.
    state_semantics = _cmd_arg(cfg.train_cmd_common, "--cb_state_semantics", "post") or "post"
    base_name = cfg.dataset_artifact_name or "dataset_snapshot"
    mix_slug = _slug(job.policy_mix_spec) or "mix"
    return _slug(f"{base_name}_{state_semantics}_{mix_slug}_seed{int(job.seed)}")


def _shared_dynamics_checkpoint_path(job: Job, cfg: WorkerConfig) -> str | None:
    backend = _cmd_arg(cfg.train_cmd_common, "--dt_logic_dynamics_backend", "tabular_env") or "tabular_env"
    if backend != "neural_dataset":
        return None
    explicit = _cmd_arg(cfg.train_cmd_common, "--dynamics_checkpoint_path", None)
    if explicit:
        return explicit
    save_flag = "--save_dynamics_checkpoint" in cfg.train_cmd_common
    no_save_flag = "--no-save_dynamics_checkpoint" in cfg.train_cmd_common
    if no_save_flag or not save_flag:
        return None
    return os.path.join(job.train_root, "shared_dynamics_model.pt")


def _write_json(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def _gpu_available(gpu_id: int | None) -> bool:
    if gpu_id is None:
        return False
    cmd = ["nvidia-smi", "-i", str(gpu_id), "--query-gpu=index", "--format=csv,noheader"]
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True, timeout=10)
    except Exception:
        return False
    return str(gpu_id) in out


def _visible_gpu_ids() -> list[int]:
    cmd = ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"]
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True, timeout=10)
    except Exception:
        return []
    ids: list[int] = []
    for line in out.splitlines():
        raw = line.strip()
        if not raw:
            continue
        token = raw.split(",")[0].strip()
        try:
            ids.append(int(token))
        except Exception:
            continue
    return ids


def _parse_gpu_worker_map(raw: str | None, workers: int) -> list[int | None]:
    if not raw:
        return [None for _ in range(workers)]
    vals = [x.strip() for x in str(raw).split(",") if x.strip() != ""]
    parsed = [int(v) for v in vals]
    if len(parsed) != workers:
        raise ValueError(
            f"--gpu_worker_map length ({len(parsed)}) must equal --parallel_workers ({workers})."
        )
    return parsed


def _extract_dataset_artifact_options(extra_args: list[str]) -> tuple[bool, str | None, str]:
    save_generated_dataset = True
    dataset_artifact_dir: str | None = None
    dataset_artifact_name = "dataset_snapshot"

    i = 0
    while i < len(extra_args):
        tok = extra_args[i]
        if tok == "--save_generated_dataset":
            save_generated_dataset = True
            i += 1
            continue
        if tok == "--no-save_generated_dataset":
            save_generated_dataset = False
            i += 1
            continue
        if tok == "--dataset_artifact_dir":
            if i + 1 < len(extra_args):
                dataset_artifact_dir = extra_args[i + 1]
                i += 2
                continue
            i += 1
            continue
        if tok == "--dataset_artifact_name":
            if i + 1 < len(extra_args):
                dataset_artifact_name = str(extra_args[i + 1]).strip() or "dataset_snapshot"
                i += 2
                continue
            i += 1
            continue
        i += 1

    return save_generated_dataset, dataset_artifact_dir, dataset_artifact_name


def _strip_dataset_artifact_flags(extra_args: list[str]) -> list[str]:
    out: list[str] = []
    i = 0
    while i < len(extra_args):
        tok = extra_args[i]
        if tok in {"--save_generated_dataset", "--no-save_generated_dataset"}:
            i += 1
            continue
        if tok in {"--dataset_artifact_dir", "--dataset_artifact_name"}:
            if i + 1 < len(extra_args):
                i += 2
                continue
            i += 1
            continue
        out.append(tok)
        i += 1
    return out


def _alpha_tag(alpha: float) -> str:
    return str(alpha)


def _baseline_keys(alphas: list[float]) -> list[str]:
    return ["vanilla"] + [f"logic_alpha{_alpha_tag(a)}" for a in alphas]


def _normalize_logic_alphas(raw_alphas: list[float]) -> list[float]:
    seen = set()
    out: list[float] = []
    for a in raw_alphas:
        fa = float(a)
        if abs(fa) < 1e-12:
            continue
        key = f"{fa:.12g}"
        if key in seen:
            continue
        seen.add(key)
        out.append(fa)
    return out


def _baseline_alpha(baseline_key: str) -> float:
    if baseline_key == "vanilla":
        return 0.0
    if baseline_key.startswith("logic_alpha"):
        return float(baseline_key[len("logic_alpha") :])
    raise ValueError(f"Unknown baseline key '{baseline_key}'")


def _checkpoint_path(train_root: str, baseline_key: str, epochs: int) -> str:
    return os.path.join(train_root, baseline_key, f"dt_state_{epochs - 1}.pt")


def _all_checkpoints_exist(train_root: str, baselines: list[str], epochs: int) -> bool:
    for baseline in baselines:
        if not os.path.exists(_checkpoint_path(train_root, baseline, epochs)):
            return False
    return True


def _mode_seed_dir(seed_root: str, mode: str) -> str:
    seed_name = os.path.basename(seed_root)
    mix_root = os.path.dirname(seed_root)
    return os.path.join(mix_root, f"mode_{mode}", seed_name)


def _has_mode_complete(mode_dir: str, baselines: list[str]) -> bool:
    csv_path = os.path.join(mode_dir, "baseline_metrics.csv")
    if not os.path.exists(csv_path):
        return False
    for baseline in baselines:
        if not os.path.exists(os.path.join(mode_dir, baseline, "metrics.json")):
            return False
    return True


def _load_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def _decode_width(mode: str, cfg: WorkerConfig) -> int:
    if mode == "constrained":
        return int(cfg.num_action_candidates)
    if mode == "knn":
        return int(cfg.knn_k)
    return 1


def _null_metrics(
    *,
    env: str,
    spec: str,
    seed: int,
    checkpoint_path: str | None,
    decoding_mode: str,
    decode_width: int,
) -> dict:
    return {
        "return_mean": None,
        "return_std": None,
        "violation_rate": None,
        "satisfaction_rate": None,
        "runtime_sec": None,
        "env": env,
        "spec": spec,
        "seed": int(seed),
        "num_episodes": None,
        "satisfaction_soft_mean": None,
        "violation_rate_episode": None,
        "violation_rate_step": None,
        "goal_rate": None,
        "bomb_hit_rate": None,
        "hazard_hit_rate": None,
        "decoding_mode": decoding_mode,
        "beam_width": int(decode_width),
        "model_type": "dt",
        "checkpoint_path": checkpoint_path,
        "satisfaction_source": None,
        "run_id": None,
        "timestamp_utc": None,
        "random_return_mean": None,
        "random_goal_rate": None,
        "random_violation_rate": None,
        "better_than_random": None,
        "rtg_target": None,
        "action_loss": None,
        "logic_loss": None,
        "success_rate": None,
        "dataset_size": None,
        "context_len": None,
    }


def _write_summary_artifacts(base_dir: str, results: dict[str, dict]) -> tuple[str, str]:
    os.makedirs(base_dir, exist_ok=True)
    json_path = os.path.join(base_dir, "baseline_metrics.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    csv_path = os.path.join(base_dir, "baseline_metrics.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for baseline_name, metrics in results.items():
            row = {"baseline": baseline_name}
            row.update({k: metrics.get(k) for k in SUMMARY_FIELDS if k != "baseline"})
            writer.writerow(row)
    return json_path, csv_path


def _load_rollout_stats(base_dir: str, labels: list[str]) -> dict[str, dict]:
    """Load per-episode rollout stats for each alpha from automaton_rollout_stats.json."""
    rollout = {}
    for label in labels:
        path = os.path.join(base_dir, label, "automaton_rollout_stats.json")
        if os.path.exists(path):
            with open(path) as f:
                rollout[label] = json.load(f)
    return rollout


def _save_summary_plots(base_dir: str, results: dict[str, dict]) -> None:
    try:
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mticker
    except Exception:
        return

    plots_dir = os.path.join(base_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    labels = list(results.keys())
    if not labels:
        return

    def _vals(key):
        out = []
        for label in labels:
            val = results[label].get(key)
            out.append(np.nan if val is None else float(val))
        return np.asarray(out, dtype=np.float32)

    rollout = _load_rollout_stats(base_dir, labels)

    # ── 1. Multi-metric bar ───────────────────────────────────────────────────
    width = 0.18
    x = np.arange(len(labels))
    bar_keys = ["goal_rate", "hazard_hit_rate", "satisfaction_rate", "return_mean"]
    bar_vals = [_vals(k) for k in bar_keys]
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.7), 4))
    offsets = np.linspace(-1.5 * width, 1.5 * width, len(bar_keys))
    for offset, k, v in zip(offsets, bar_keys, bar_vals):
        ax.bar(x + offset, np.nan_to_num(v, nan=0.0), width=width, label=k)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "metrics_bar.png"), dpi=120)
    plt.close(fig)

    # ── 2. Multi-metric trend (goal / hazard / return vs alpha) ──────────────
    goal_r = _vals("goal_rate")
    hazard_r = _vals("hazard_hit_rate")
    sat_r = _vals("satisfaction_rate")
    ret_r = _vals("return_mean")
    xs = range(len(labels))
    fig, ax1 = plt.subplots(figsize=(max(7, len(labels) * 0.6), 4))
    ax2 = ax1.twinx()
    ax1.plot(xs, np.nan_to_num(goal_r), marker="o", color="green", label="goal_rate")
    ax1.plot(xs, np.nan_to_num(hazard_r), marker="x", color="red", label="bomb_hit_rate")
    ax1.plot(xs, np.nan_to_num(sat_r), marker="s", color="blue", linestyle="--", label="satisfaction_rate")
    ax2.plot(xs, np.nan_to_num(ret_r), marker="^", color="orange", label="return_mean")
    ax1.set_xticks(list(xs))
    ax1.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    ax1.set_ylim(-0.05, 1.05)
    ax1.set_ylabel("rate (0–1)")
    ax2.set_ylabel("return")
    lines1, leg1 = ax1.get_legend_handles_labels()
    lines2, leg2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, leg1 + leg2, loc="best", fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "satisfaction_trend.png"), dpi=120)
    plt.close(fig)

    # ── 3. Return vs satisfaction scatter ────────────────────────────────────
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(np.nan_to_num(ret_r), np.nan_to_num(sat_r))
    for i, label in enumerate(labels):
        ax.annotate(label, (np.nan_to_num(ret_r[i]), np.nan_to_num(sat_r[i])), fontsize=7)
    ax.set_xlabel("return_mean")
    ax.set_ylabel("satisfaction_rate")
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "return_vs_satisfaction.png"), dpi=120)
    plt.close(fig)

    # ── 4. Episode outcome stacked bar (goal / bomb / timeout) ───────────────
    if rollout:
        goals_frac, bombs_frac, timeout_frac = [], [], []
        valid_labels = []
        for label in labels:
            rs = rollout.get(label)
            if rs is None:
                continue
            g = np.asarray(rs.get("episode_goal_hits", []), dtype=float)
            h = np.asarray(rs.get("episode_hazard_hits", []), dtype=float)
            n = max(len(g), 1)
            goal_ep = float(np.mean(g > 0)) if len(g) else 0.0
            bomb_ep = float(np.mean(h > 0)) if len(h) else 0.0
            to_ep = max(0.0, 1.0 - goal_ep - bomb_ep)
            goals_frac.append(goal_ep)
            bombs_frac.append(bomb_ep)
            timeout_frac.append(to_ep)
            valid_labels.append(label)
        if valid_labels:
            x2 = np.arange(len(valid_labels))
            fig, ax = plt.subplots(figsize=(max(7, len(valid_labels) * 0.65), 4))
            ax.bar(x2, goals_frac, label="goal reached", color="green")
            ax.bar(x2, bombs_frac, bottom=goals_frac, label="bomb hit", color="red")
            ax.bar(x2, timeout_frac, bottom=np.add(goals_frac, bombs_frac), label="timeout", color="gray")
            ax.set_xticks(x2)
            ax.set_xticklabels(valid_labels, rotation=20, ha="right", fontsize=8)
            ax.set_ylim(0, 1.05)
            ax.set_ylabel("episode fraction")
            ax.legend(loc="upper right", fontsize=8)
            fig.tight_layout()
            fig.savefig(os.path.join(plots_dir, "outcome_breakdown.png"), dpi=120)
            plt.close(fig)

    # ── 5. Return distribution overlay ───────────────────────────────────────
    if rollout:
        plot_labels = [l for l in labels if l in rollout and rollout[l].get("episode_returns")]
        if plot_labels:
            fig, ax = plt.subplots(figsize=(7, 4))
            cmap = plt.cm.get_cmap("tab10", len(plot_labels))
            for idx, label in enumerate(plot_labels):
                ep_rets = np.asarray(rollout[label]["episode_returns"], dtype=float)
                ax.hist(ep_rets, bins=30, alpha=0.45, label=label, color=cmap(idx), density=True)
            ax.set_xlabel("episode return")
            ax.set_ylabel("density")
            ax.legend(loc="best", fontsize=7, ncol=2)
            fig.tight_layout()
            fig.savefig(os.path.join(plots_dir, "return_hist.png"), dpi=120)
            plt.close(fig)

    # ── 6. Episode length distribution ───────────────────────────────────────
    if rollout:
        plot_labels = [l for l in labels if l in rollout and rollout[l].get("episode_lengths")]
        if plot_labels:
            fig, ax = plt.subplots(figsize=(7, 4))
            cmap = plt.cm.get_cmap("tab10", len(plot_labels))
            for idx, label in enumerate(plot_labels):
                ep_lens = np.asarray(rollout[label]["episode_lengths"], dtype=float)
                ax.hist(ep_lens, bins=30, alpha=0.45, label=label, color=cmap(idx), density=True)
            ax.set_xlabel("episode length (steps)")
            ax.set_ylabel("density")
            ax.legend(loc="best", fontsize=7, ncol=2)
            fig.tight_layout()
            fig.savefig(os.path.join(plots_dir, "episode_length_hist.png"), dpi=120)
            plt.close(fig)

    # ── 7. Return box plot per alpha ──────────────────────────────────────────
    if rollout:
        box_labels = [l for l in labels if l in rollout and rollout[l].get("episode_returns")]
        if box_labels:
            data = [np.asarray(rollout[l]["episode_returns"], dtype=float) for l in box_labels]
            fig, ax = plt.subplots(figsize=(max(7, len(box_labels) * 0.65), 4))
            ax.boxplot(data, labels=box_labels, patch_artist=True, notch=False)
            ax.set_xticklabels(box_labels, rotation=20, ha="right", fontsize=8)
            ax.set_ylabel("episode return")
            ax.axhline(0, color="green", linewidth=0.8, linestyle="--", label="break-even")
            ax.legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(os.path.join(plots_dir, "return_boxplot.png"), dpi=120)
            plt.close(fig)


def _run_subprocess(
    cmd: list[str],
    *,
    log_path: str,
    gpu_id: int | None,
    dry_run: bool,
    log_prefix: str,
) -> tuple[int, str]:
    env = os.environ.copy()
    env["HOME"] = env.get("HOME", str(REPO_ROOT))
    env["MPLCONFIGDIR"] = env.get("MPLCONFIGDIR", str(REPO_ROOT / ".config" / "matplotlib"))
    os.makedirs(env["MPLCONFIGDIR"], exist_ok=True)
    if gpu_id is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    cmd_str = " ".join(shlex.quote(x) for x in cmd)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    if dry_run:
        with open(log_path, "a") as lf:
            lf.write(f"[dry-run:{log_prefix}] {cmd_str}\n")
        return 0, cmd_str

    with open(log_path, "a") as lf:
        lf.write(f"[{log_prefix}] ts={time.time()}\n")
        lf.write(f"[cmd] {cmd_str}\n")
        lf.flush()
        proc = subprocess.run(cmd, cwd=REPO_ROOT, env=env, stdout=lf, stderr=lf)
    return int(proc.returncode), cmd_str


def _run_tt_dataset_analysis(job: Job, cfg: WorkerConfig) -> tuple[str, str | None]:
    if cfg.skip_dataset_analysis:
        return "skipped", None

    summary_path = os.path.join(job.train_root, "dataset_analysis", "summary.json")
    if cfg.skip_completed and os.path.exists(summary_path):
        return "skipped", None

    log_path = os.path.join(job.train_root, "dataset_analysis", "console.log")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    if cfg.dry_run:
        with open(log_path, "a") as lf:
            lf.write(
                "[dry-run:dataset-analysis] run_tt_style_dataset_analysis "
                f"spec={job.spec} mix={job.policy_mix_spec} seed={job.seed}\n"
            )
        return "ok", None

    try:
        from planning.dt_runtime import build_dt_offline_source, run_tt_style_dataset_analysis

        analysis_args = SimpleNamespace(
            env="cb",
            seed=int(job.seed),
            num_episodes=int(cfg.num_episodes),
            max_steps=int(cfg.max_steps),
            context_len=int(cfg.context_len),
            stochastic=bool(cfg.stochastic),
            cb_policy_mix_spec=str(job.policy_mix_spec),
            cb_policy_mix_sampling=str(cfg.cb_policy_mix_sampling),
            cb_policy_mix_normal_spec=cfg.cb_policy_mix_normal_spec,
            cb_policy_mix_normal_mean_mode=str(cfg.cb_policy_mix_normal_mean_mode),
            cb_state_semantics=str(cfg.cb_state_semantics),
            cb_longest_path_max_expansions=int(cfg.cb_longest_path_max_expansions),
            spec=str(job.spec),
            ltl_formula=None,
            ltl_formulas=None,
            dfa_mode=str(cfg.dfa_mode),
            use_safe_dfa=bool(cfg.use_safe_dfa),
            constraint_dims=[0],
            frozenlake_use_position_props=False,
            dfa_backend="auto",
            save_path=job.train_root,
            run_dir=job.train_root,
        )

        with open(log_path, "a") as lf, contextlib.redirect_stdout(lf), contextlib.redirect_stderr(lf):
            print(f"[dataset-analysis] ts={time.time()}")
            dataset, skip_reason = build_dt_offline_source(analysis_args)
            if skip_reason is not None:
                raise RuntimeError(f"Dataset analysis skipped unexpectedly: {skip_reason}")
            run_tt_style_dataset_analysis(analysis_args, dataset)
        if not os.path.exists(summary_path):
            return "failed", "dataset_analysis_missing_summary"
        return "ok", None
    except Exception:
        with open(log_path, "a") as lf:
            lf.write("[dataset-analysis:error]\n")
            traceback.print_exc(file=lf)
        return "failed", "dataset_analysis_failed"


def _train_baselines(job: Job, cfg: WorkerConfig) -> tuple[str, str | None]:
    if cfg.skip_completed and _all_checkpoints_exist(job.train_root, cfg.baseline_keys, cfg.epochs):
        return "skipped", None

    for baseline in cfg.baseline_keys:
        alpha = _baseline_alpha(baseline)
        baseline_dir = os.path.join(job.train_root, baseline)
        train_log = os.path.join(baseline_dir, "console.log")
        ckpt = _checkpoint_path(job.train_root, baseline, cfg.epochs)
        if cfg.skip_completed and os.path.exists(ckpt):
            continue

        cmd = list(cfg.train_cmd_common)
        cmd.extend(
            [
                "--seed",
                str(job.seed),
                "--run_dir",
                baseline_dir,
                "--cb_policy_mix_spec",
                job.policy_mix_spec,
                "--spec",
                job.spec,
                "--logic_alpha",
                str(alpha),
            ]
        )
        cmd.extend(cfg.extra_args)
        # Save one stable dataset artifact per seed (and policy-mix/semantics).
        # Skip redundant saves once the shared artifact already exists.
        artifact_name = _build_seed_dataset_artifact_name(job, cfg)
        artifact_dir = cfg.dataset_artifact_dir or os.path.join(job.seed_root, "dataset_artifacts")
        artifact_npz = os.path.join(artifact_dir, f"{artifact_name}.npz")
        artifact_meta = os.path.join(artifact_dir, f"{artifact_name}.meta.json")
        if os.path.exists(artifact_npz) and os.path.exists(artifact_meta):
            cmd.extend(["--dataset_artifact_path", artifact_npz])
            cmd.append("--no-save_generated_dataset")
        elif cfg.save_generated_dataset:
            cmd.extend(["--save_generated_dataset"])
            cmd.extend(["--dataset_artifact_dir", artifact_dir])
            cmd.extend(["--dataset_artifact_name", artifact_name])
        else:
            cmd.append("--no-save_generated_dataset")
        dynamics_ckpt = _shared_dynamics_checkpoint_path(job, cfg)
        if dynamics_ckpt is not None:
            cmd.extend(["--dynamics_checkpoint_path", dynamics_ckpt])
        rc, cmd_str = _run_subprocess(
            cmd, log_path=train_log, gpu_id=cfg.gpu_id, dry_run=cfg.dry_run, log_prefix=f"train:{baseline}"
        )
        if rc != 0:
            return "failed", cmd_str

    return "ok", None


def _evaluate_mode(job: Job, cfg: WorkerConfig, mode: str) -> tuple[str, dict[str, dict]]:
    mode_dir = _mode_seed_dir(job.seed_root, mode)
    os.makedirs(mode_dir, exist_ok=True)

    if cfg.skip_completed and _has_mode_complete(mode_dir, cfg.baseline_keys):
        results = {}
        for baseline in cfg.baseline_keys:
            mpath = os.path.join(mode_dir, baseline, "metrics.json")
            results[baseline] = _load_json(mpath)
        return "skipped", results

    results: dict[str, dict] = {}
    for baseline in cfg.baseline_keys:
        ckpt = _checkpoint_path(job.train_root, baseline, cfg.epochs)
        baseline_dir = os.path.join(mode_dir, baseline)
        os.makedirs(baseline_dir, exist_ok=True)
        mpath = os.path.join(baseline_dir, "metrics.json")
        if cfg.skip_completed and os.path.exists(mpath):
            results[baseline] = _load_json(mpath)
            continue

        if not os.path.exists(ckpt):
            results[baseline] = _null_metrics(
                env="cb",
                spec=job.spec,
                seed=job.seed,
                checkpoint_path=ckpt,
                decoding_mode=mode,
                decode_width=_decode_width(mode, cfg),
            )
            continue

        eval_cmd = list(cfg.eval_cmd_common)
        eval_cmd.extend(
            [
                "--checkpoint",
                ckpt,
                "--spec",
                job.spec,
                "--seed",
                str(job.seed),
                "--cb_policy_mix_spec",
                job.policy_mix_spec,
                "--dt_mode",
                mode,
                "--run_dir",
                baseline_dir,
            ]
        )
        eval_cmd.extend(cfg.extra_args)
        # Eval can run many times per seed (baseline x decoding mode); never save datasets here.
        artifact_name = _build_seed_dataset_artifact_name(job, cfg)
        artifact_dir = cfg.dataset_artifact_dir or os.path.join(job.seed_root, "dataset_artifacts")
        artifact_npz = os.path.join(artifact_dir, f"{artifact_name}.npz")
        artifact_meta = os.path.join(artifact_dir, f"{artifact_name}.meta.json")
        if os.path.exists(artifact_npz) and os.path.exists(artifact_meta):
            eval_cmd.extend(["--dataset_artifact_path", artifact_npz])
        eval_cmd.append("--no-save_generated_dataset")

        eval_log = os.path.join(baseline_dir, "console.log")
        rc, _ = _run_subprocess(
            eval_cmd,
            log_path=eval_log,
            gpu_id=cfg.gpu_id,
            dry_run=cfg.dry_run,
            log_prefix=f"eval:{mode}:{baseline}",
        )
        if rc == 0 and os.path.exists(mpath):
            results[baseline] = _load_json(mpath)
        else:
            results[baseline] = _null_metrics(
                env="cb",
                spec=job.spec,
                seed=job.seed,
                checkpoint_path=ckpt,
                decoding_mode=mode,
                decode_width=_decode_width(mode, cfg),
            )

    _write_summary_artifacts(mode_dir, results)
    _save_summary_plots(mode_dir, results)
    return "ok", results


def _run_single_job(job: Job, cfg: WorkerConfig) -> dict:
    started = time.time()
    os.makedirs(job.seed_root, exist_ok=True)

    if cfg.require_gpu and not _gpu_available(cfg.gpu_id):
        return {
            "status": "gpu_unavailable",
            "job": asdict(job),
            "worker_id": cfg.worker_id,
            "gpu_id": cfg.gpu_id,
            "elapsed_sec": 0.0,
            "reason": "GPU check failed before job start",
        }

    _write_json(
        os.path.join(job.seed_root, "job_args.json"),
        {
            "spec": job.spec,
            "policy_mix_spec": job.policy_mix_spec,
            "seed": int(job.seed),
            "worker_id": cfg.worker_id,
            "gpu_id": cfg.gpu_id,
            "require_gpu": cfg.require_gpu,
            "decoding_modes": cfg.decoding_modes,
            "baselines": cfg.baseline_keys,
            "logic_alphas": cfg.logic_alphas,
            "skip_dataset_analysis": bool(cfg.skip_dataset_analysis),
            "extra_args": cfg.extra_args,
        },
    )

    analysis_status, analysis_error = _run_tt_dataset_analysis(job, cfg)
    if analysis_status == "failed":
        return {
            "status": "failed",
            "job": asdict(job),
            "worker_id": cfg.worker_id,
            "gpu_id": cfg.gpu_id,
            "elapsed_sec": float(time.time() - started),
            "train_command": analysis_error,
        }

    train_status, train_cmd = _train_baselines(job, cfg)
    if train_status == "failed":
        status = "failed"
        if cfg.require_gpu and not _gpu_available(cfg.gpu_id):
            status = "gpu_unavailable"
        return {
            "status": status,
            "job": asdict(job),
            "worker_id": cfg.worker_id,
            "gpu_id": cfg.gpu_id,
            "elapsed_sec": float(time.time() - started),
            "train_command": train_cmd,
        }

    mode_status = {}
    for mode in cfg.decoding_modes:
        if cfg.require_gpu and not _gpu_available(cfg.gpu_id):
            return {
                "status": "gpu_unavailable",
                "job": asdict(job),
                "worker_id": cfg.worker_id,
                "gpu_id": cfg.gpu_id,
                "elapsed_sec": float(time.time() - started),
                "reason": f"GPU became unavailable before mode={mode}",
            }
        mstatus, _ = _evaluate_mode(job, cfg, mode)
        mode_status[mode] = mstatus

    return {
        "status": "ok",
        "job": asdict(job),
        "worker_id": cfg.worker_id,
        "gpu_id": cfg.gpu_id,
        "elapsed_sec": float(time.time() - started),
        "train_status": train_status,
        "mode_status": mode_status,
    }


def _worker_loop(cfg: WorkerConfig, q_in: mp.Queue, q_out: mp.Queue, stop_event: mp.Event):
    while True:
        if stop_event.is_set():
            break
        try:
            item = q_in.get(timeout=2)
        except Empty:
            continue
        if item is None:
            break
        job = Job(**item)
        result = _run_single_job(job, cfg)
        q_out.put(result)

        if result["status"] == "gpu_unavailable" and cfg.stop_on_gpu_unavailable:
            stop_event.set()
            break
        if result["status"] == "failed" and not cfg.continue_on_error:
            stop_event.set()
            break


def _build_jobs(args: argparse.Namespace) -> list[Job]:
    jobs: list[Job] = []
    for spec in args.specs:
        for mix in args.policy_mix_specs:
            for seed in args.seeds:
                spec_dir = os.path.join(args.output_root, f"spec_{_slug(spec)}")
                mix_dir = os.path.join(spec_dir, f"mix_{_slug(mix)}")
                seed_root = os.path.join(mix_dir, f"seed_{int(seed)}")
                train_root = os.path.join(seed_root, "train_shared")
                jobs.append(
                    Job(
                        spec=str(spec),
                        policy_mix_spec=str(mix),
                        seed=int(seed),
                        seed_root=seed_root,
                        train_root=train_root,
                    )
                )
    return jobs


def _build_train_cmd_common(args: argparse.Namespace) -> list[str]:
    cmd = [PYTHON, "scripts/train_dt.py", "--env", "cb"]
    cmd.extend(["--num_episodes", str(args.num_episodes)])
    cmd.extend(["--max_steps", str(args.max_steps)])
    cmd.extend(["--context_len", str(args.context_len)])
    cmd.extend(["--epochs", str(args.epochs)])
    cmd.extend(["--batch_size", str(args.batch_size)])
    cmd.extend(["--lr", str(args.lr)])
    cmd.extend(["--weight_decay", str(args.weight_decay)])
    cmd.extend(["--grad_clip", str(args.grad_clip)])
    cmd.extend(["--dt_logic_loss_type", str(args.dt_logic_loss_type)])
    cmd.extend(["--logic_rollout_horizon", str(args.logic_rollout_horizon)])
    cmd.extend(["--logic_temperature", str(args.logic_temperature)])
    cmd.extend(["--dt_logic_dynamics_backend", str(args.dt_logic_dynamics_backend)])
    cmd.extend(["--dynamics_epochs", str(args.dynamics_epochs)])
    cmd.extend(["--dynamics_batch_size", str(args.dynamics_batch_size)])
    cmd.extend(["--dynamics_lr", str(args.dynamics_lr)])
    cmd.extend(["--dynamics_hidden_dim", str(args.dynamics_hidden_dim)])
    cmd.extend(["--dynamics_layers", str(args.dynamics_layers)])
    cmd.extend(["--dynamics_weight_decay", str(args.dynamics_weight_decay)])
    cmd.extend(["--dynamics_val_fraction", str(args.dynamics_val_fraction)])
    cmd.extend(["--dynamics_max_transition_entries", str(args.dynamics_max_transition_entries)])
    cmd.extend(["--dynamics_temperature", str(args.dynamics_temperature)])
    cmd.extend(["--n_layer", str(args.n_layer)])
    cmd.extend(["--n_head", str(args.n_head)])
    cmd.extend(["--n_embd", str(args.n_embd)])
    cmd.extend(["--dropout", str(args.dropout)])
    cmd.extend(["--cb_longest_path_max_expansions", str(args.cb_longest_path_max_expansions)])
    cmd.extend(["--cb_policy_mix_sampling", str(args.cb_policy_mix_sampling)])
    cmd.extend(["--cb_state_semantics", str(args.cb_state_semantics)])
    cmd.extend(["--cb_policy_mix_normal_mean_mode", str(args.cb_policy_mix_normal_mean_mode)])
    cmd.extend(["--dfa_mode", str(args.dfa_mode)])
    if args.dynamics_freeze_after_fit:
        cmd.append("--dynamics_freeze_after_fit")
    else:
        cmd.append("--no-dynamics_freeze_after_fit")
    if args.save_dynamics_checkpoint:
        cmd.append("--save_dynamics_checkpoint")
    else:
        cmd.append("--no-save_dynamics_checkpoint")
    if args.dynamics_checkpoint_path is not None:
        cmd.extend(["--dynamics_checkpoint_path", str(args.dynamics_checkpoint_path)])
    if args.cb_policy_mix_normal_spec is not None:
        cmd.extend(["--cb_policy_mix_normal_spec", str(args.cb_policy_mix_normal_spec)])
    if args.use_safe_dfa:
        cmd.append("--use_safe_dfa")
    if args.stochastic:
        cmd.append("--stochastic")
    if args.rtg_target is not None:
        cmd.extend(["--rtg_target", str(args.rtg_target)])
    # Matrix flow evaluates by checkpoint in dedicated mode dirs.
    cmd.append("--no_eval_after_train")
    return cmd


def _build_eval_cmd_common(args: argparse.Namespace) -> list[str]:
    cmd = [PYTHON, "scripts/eval_dt.py", "--env", "cb"]
    cmd.extend(["--num_episodes", str(args.num_episodes)])
    cmd.extend(["--max_steps", str(args.max_steps)])
    cmd.extend(["--context_len", str(args.context_len)])
    cmd.extend(["--eval_num_episodes", str(args.eval_num_episodes)])
    cmd.extend(["--eval_max_steps", str(args.eval_max_steps)])
    cmd.extend(["--dfa_mode", str(args.dfa_mode)])
    cmd.extend(["--num_action_candidates", str(args.num_action_candidates)])
    cmd.extend(["--lookahead_horizon", str(args.lookahead_horizon)])
    cmd.extend(["--lookahead_backend", str(args.lookahead_backend)])
    cmd.extend(["--sat_rerank_weight", str(args.sat_rerank_weight)])
    cmd.extend(["--candidate_sampling", str(args.candidate_sampling)])
    cmd.extend(["--knn_k", str(args.knn_k)])
    cmd.extend(["--knn_return_weight", str(args.knn_return_weight)])
    cmd.extend(["--knn_satisfaction_weight", str(args.knn_satisfaction_weight)])
    cmd.extend(["--cb_longest_path_max_expansions", str(args.cb_longest_path_max_expansions)])
    cmd.extend(["--cb_policy_mix_sampling", str(args.cb_policy_mix_sampling)])
    cmd.extend(["--cb_state_semantics", str(args.cb_state_semantics)])
    cmd.extend(["--cb_policy_mix_normal_mean_mode", str(args.cb_policy_mix_normal_mean_mode)])
    if args.cb_policy_mix_normal_spec is not None:
        cmd.extend(["--cb_policy_mix_normal_spec", str(args.cb_policy_mix_normal_spec)])
    if args.use_safe_dfa:
        cmd.append("--use_safe_dfa")
    if args.hard_prune_reject_sink:
        cmd.append("--hard_prune_reject_sink")
    if args.stochastic:
        cmd.append("--stochastic")
    if args.rtg_target is not None:
        cmd.extend(["--rtg_target", str(args.rtg_target)])
    return cmd


def parse_args(argv: list[str] | None = None):
    p = argparse.ArgumentParser(
        description="ColourBomb DT matrix runner (optimized: train once, decode many)"
    )
    p.add_argument("--output_root", type=str, required=True)
    p.add_argument("--specs", nargs="+", required=True)
    p.add_argument("--policy_mix_specs", nargs="+", required=True)
    p.add_argument(
        "--decoding_modes",
        nargs="+",
        default=["greedy", "constrained", "knn"],
        choices=["greedy", "constrained", "knn"],
    )
    p.add_argument("--alphas", nargs="+", type=float, default=[0.0, 0.01, 0.05, 0.1, 0.2, 0.4])
    p.add_argument("--seeds", nargs="+", type=int, default=[0])

    p.add_argument("--parallel_workers", type=int, default=1)
    p.add_argument("--gpu_worker_map", type=str, default=None)
    p.add_argument("--require_gpu", action="store_true")
    p.add_argument("--stop_on_gpu_unavailable", action="store_true")
    p.add_argument("--continue_on_error", action="store_true")
    p.add_argument("--no_skip_completed", action="store_true")
    p.add_argument("--dry_run", action="store_true")

    p.add_argument("--num_episodes", type=int, default=5000)
    p.add_argument("--max_steps", type=int, default=200)
    p.add_argument("--context_len", type=int, default=20)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--dt_logic_loss_type", type=str, choices=["auto", "hazard", "dfa"], default="auto")
    p.add_argument("--logic_rollout_horizon", type=int, default=2)
    p.add_argument("--logic_temperature", type=float, default=1.0)
    p.add_argument(
        "--dt_logic_dynamics_backend",
        type=str,
        choices=["tabular_env", "tabular_dataset", "neural_dataset"],
        default="tabular_env",
    )
    p.add_argument("--dynamics_epochs", type=int, default=20)
    p.add_argument("--dynamics_batch_size", type=int, default=256)
    p.add_argument("--dynamics_lr", type=float, default=1e-3)
    p.add_argument("--dynamics_hidden_dim", type=int, default=128)
    p.add_argument("--dynamics_layers", type=int, default=2)
    p.add_argument("--dynamics_weight_decay", type=float, default=1e-4)
    p.add_argument("--dynamics_val_fraction", type=float, default=0.1)
    p.add_argument("--dynamics_freeze_after_fit", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--dynamics_checkpoint_path", type=str, default=None)
    p.add_argument("--save_dynamics_checkpoint", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--dynamics_max_transition_entries", type=int, default=10000000)
    p.add_argument("--dynamics_temperature", type=float, default=1.0)
    p.add_argument("--n_layer", type=int, default=4)
    p.add_argument("--n_head", type=int, default=4)
    p.add_argument("--n_embd", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--rtg_target", type=float, default=None)

    p.add_argument("--eval_num_episodes", type=int, default=200)
    p.add_argument("--eval_max_steps", type=int, default=200)
    p.add_argument("--dfa_mode", type=str, choices=["single", "product", "multi"], default="product")
    p.add_argument("--use_safe_dfa", action="store_true")
    p.add_argument("--num_action_candidates", type=int, default=4)
    p.add_argument("--lookahead_horizon", type=int, default=2)
    p.add_argument("--lookahead_backend", type=str, choices=["env", "dynamics"], default="env")
    p.add_argument("--hard_prune_reject_sink", action="store_true")
    p.add_argument("--sat_rerank_weight", type=float, default=2.0)
    p.add_argument("--candidate_sampling", type=str, choices=["topk", "sample"], default="topk")
    p.add_argument("--knn_k", type=int, default=16)
    p.add_argument("--knn_return_weight", type=float, default=1.0)
    p.add_argument("--knn_satisfaction_weight", type=float, default=2.0)

    p.add_argument("--stochastic", action="store_true")
    p.add_argument("--cb_longest_path_max_expansions", type=int, default=500000)
    p.add_argument("--cb_policy_mix_sampling", type=str, choices=["fixed", "normal"], default="fixed")
    p.add_argument("--cb_policy_mix_normal_spec", type=str, default=None)
    p.add_argument(
        "--cb_policy_mix_normal_mean_mode",
        type=str,
        choices=["base", "absolute", "delta"],
        default="base",
    )
    p.add_argument("--cb_state_semantics", type=str, choices=["pre", "post"], default="post")
    p.add_argument("--skip_dataset_analysis", action="store_true")

    args, extra = p.parse_known_args(argv)
    return args, extra


def main(argv: list[str] | None = None):
    args, extra = parse_args(argv)
    os.makedirs(args.output_root, exist_ok=True)

    jobs = _build_jobs(args)
    if not jobs:
        print("No jobs to run.")
        return

    worker_count = max(1, int(args.parallel_workers))
    gpu_map = _parse_gpu_worker_map(args.gpu_worker_map, worker_count)
    if args.require_gpu and any(g is None for g in gpu_map):
        raise ValueError("When --require_gpu is enabled, provide --gpu_worker_map (e.g. 0,0,1,1).")
    if args.require_gpu:
        visible = _visible_gpu_ids()
        if not visible:
            raise RuntimeError(
                "--require_gpu is enabled but no visible GPUs were detected via nvidia-smi."
            )
        missing = sorted({int(g) for g in gpu_map if g is not None} - set(visible))
        if missing:
            raise ValueError(
                f"--gpu_worker_map references unavailable GPU ids {missing}; visible GPUs are {visible}."
            )

    save_generated_dataset, dataset_artifact_dir, dataset_artifact_name = _extract_dataset_artifact_options(
        list(extra)
    )
    passthrough_extra = _strip_dataset_artifact_flags(list(extra))

    logic_alphas = _normalize_logic_alphas(list(args.alphas))
    baseline_keys = _baseline_keys(logic_alphas)
    train_cmd_common = _build_train_cmd_common(args)
    eval_cmd_common = _build_eval_cmd_common(args)
    warnings: list[str] = []
    if args.use_safe_dfa and set(args.decoding_modes) == {"greedy"}:
        msg = (
            "--use_safe_dfa is enabled with decoding_modes=['greedy']; "
            "DFA constraints affect evaluation metrics but not greedy action selection."
        )
        print(f"[warn] {msg}")
        warnings.append(msg)

    manifest = {
        "output_root": args.output_root,
        "workers": worker_count,
        "gpu_worker_map": gpu_map,
        "require_gpu": bool(args.require_gpu),
        "stop_on_gpu_unavailable": bool(args.stop_on_gpu_unavailable),
        "continue_on_error": bool(args.continue_on_error),
        "skip_completed": not bool(args.no_skip_completed),
        "dry_run": bool(args.dry_run),
        "raw_extra_args": list(extra),
        "extra_args": passthrough_extra,
        "save_generated_dataset": bool(save_generated_dataset),
        "dataset_artifact_dir": dataset_artifact_dir,
        "dataset_artifact_name": dataset_artifact_name,
        "skip_dataset_analysis": bool(args.skip_dataset_analysis),
        "train_cmd_common": train_cmd_common,
        "eval_cmd_common": eval_cmd_common,
        "decoding_modes": list(args.decoding_modes),
        "baseline_keys": baseline_keys,
        "logic_alphas_effective": logic_alphas,
        "warnings": warnings,
        "jobs_total": len(jobs),
        "jobs": [asdict(j) for j in jobs],
        "created_ts": time.time(),
        "optimization_mode": "train_once_shared_decode",
        "model_type": "dt",
    }
    _write_json(os.path.join(args.output_root, "matrix_manifest.json"), manifest)

    q_in: mp.Queue = mp.Queue()
    q_out: mp.Queue = mp.Queue()
    stop_event: mp.Event = mp.Event()

    for j in jobs:
        q_in.put(asdict(j))
    for _ in range(worker_count):
        q_in.put(None)

    workers: list[mp.Process] = []
    for worker_id in range(worker_count):
        wcfg = WorkerConfig(
            worker_id=worker_id,
            gpu_id=gpu_map[worker_id],
            require_gpu=bool(args.require_gpu),
            stop_on_gpu_unavailable=bool(args.stop_on_gpu_unavailable),
            continue_on_error=bool(args.continue_on_error),
            skip_completed=not bool(args.no_skip_completed),
            dry_run=bool(args.dry_run),
            train_cmd_common=train_cmd_common,
            eval_cmd_common=eval_cmd_common,
            decoding_modes=list(args.decoding_modes),
            baseline_keys=baseline_keys,
            logic_alphas=logic_alphas,
            extra_args=list(passthrough_extra),
            save_generated_dataset=bool(save_generated_dataset),
            dataset_artifact_dir=dataset_artifact_dir,
            dataset_artifact_name=dataset_artifact_name,
            skip_dataset_analysis=bool(args.skip_dataset_analysis),
            num_episodes=int(args.num_episodes),
            max_steps=int(args.max_steps),
            context_len=int(args.context_len),
            stochastic=bool(args.stochastic),
            cb_longest_path_max_expansions=int(args.cb_longest_path_max_expansions),
            cb_policy_mix_sampling=str(args.cb_policy_mix_sampling),
            cb_policy_mix_normal_spec=(
                None if args.cb_policy_mix_normal_spec is None else str(args.cb_policy_mix_normal_spec)
            ),
            cb_policy_mix_normal_mean_mode=str(args.cb_policy_mix_normal_mean_mode),
            cb_state_semantics=str(args.cb_state_semantics),
            dfa_mode=str(args.dfa_mode),
            use_safe_dfa=bool(args.use_safe_dfa),
            num_action_candidates=int(args.num_action_candidates),
            knn_k=int(args.knn_k),
            epochs=int(args.epochs),
        )
        proc = mp.Process(target=_worker_loop, args=(wcfg, q_in, q_out, stop_event), daemon=False)
        proc.start()
        workers.append(proc)

    results = []
    alive = True
    while alive:
        alive = any(p.is_alive() for p in workers)
        try:
            item = q_out.get(timeout=1)
            results.append(item)
            status = item.get("status")
            job = item.get("job", {})
            print(
                f"[result] status={status} spec={job.get('spec')} mix={job.get('policy_mix_spec')} "
                f"seed={job.get('seed')} gpu={item.get('gpu_id')}"
            )
            _write_json(
                os.path.join(args.output_root, "matrix_progress.json"),
                {"results": results, "updated_ts": time.time()},
            )
        except Empty:
            pass

        if stop_event.is_set():
            while True:
                try:
                    q_in.get_nowait()
                except Exception:
                    break

    for p in workers:
        p.join(timeout=2)

    while True:
        try:
            item = q_out.get_nowait()
            results.append(item)
        except Empty:
            break

    status_counts = {}
    for r in results:
        s = r.get("status", "unknown")
        status_counts[s] = status_counts.get(s, 0) + 1

    summary = {
        "output_root": args.output_root,
        "jobs_total": len(jobs),
        "results_count": len(results),
        "status_counts": status_counts,
        "finished_ts": time.time(),
    }
    _write_json(os.path.join(args.output_root, "matrix_summary.json"), summary)
    _write_json(os.path.join(args.output_root, "matrix_results.json"), {"results": results})
    print(f"Done. status_counts={status_counts} results={len(results)}/{len(jobs)}")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
