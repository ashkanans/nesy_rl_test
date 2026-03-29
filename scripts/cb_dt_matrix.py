import argparse
import csv
import json
import multiprocessing as mp
import os
import re
import shlex
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from queue import Empty

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
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
    base_name = _cmd_arg(cfg.extra_args, "--dataset_artifact_name", "dataset_snapshot") or "dataset_snapshot"
    mix_slug = _slug(job.policy_mix_spec) or "mix"
    return _slug(f"{base_name}_{state_semantics}_{mix_slug}_seed{int(job.seed)}")


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


def _save_summary_plots(base_dir: str, results: dict[str, dict]) -> None:
    try:
        import matplotlib.pyplot as plt
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

    width = 0.2
    x = np.arange(len(labels))
    keys = ["goal_rate", "hazard_hit_rate", "satisfaction_rate", "return_mean"]
    vals = [_vals(k) for k in keys]

    plt.figure(figsize=(8, 4))
    for i, (k, v) in enumerate(zip(keys, vals)):
        plt.bar(x + (i - 1.5) * width, np.nan_to_num(v, nan=0.0), width=width, label=k)
    plt.xticks(x, labels, rotation=15, ha="right")
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "metrics_bar.png"))
    plt.close()

    sats = _vals("satisfaction_rate")
    plt.figure(figsize=(6, 4))
    plt.plot(range(len(labels)), np.nan_to_num(sats, nan=0.0), marker="o")
    plt.xticks(range(len(labels)), labels, rotation=15, ha="right")
    plt.ylim(-0.05, 1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "satisfaction_trend.png"))
    plt.close()

    rets = _vals("return_mean")
    plt.figure(figsize=(5, 4))
    plt.scatter(np.nan_to_num(rets, nan=0.0), np.nan_to_num(sats, nan=0.0))
    for i, label in enumerate(labels):
        plt.annotate(label, (np.nan_to_num(rets[i], nan=0.0), np.nan_to_num(sats[i], nan=0.0)))
    plt.xlabel("return_mean")
    plt.ylabel("satisfaction_rate")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "return_vs_satisfaction.png"))
    plt.close()


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
                "--logic_alpha",
                str(alpha),
            ]
        )
        cmd.extend(cfg.extra_args)
        # Save one stable dataset artifact per seed (and policy-mix/semantics).
        # Skip redundant saves once the shared artifact already exists.
        artifact_name = _build_seed_dataset_artifact_name(job, cfg)
        artifact_dir = os.path.join(job.seed_root, "dataset_artifacts")
        artifact_npz = os.path.join(artifact_dir, f"{artifact_name}.npz")
        artifact_meta = os.path.join(artifact_dir, f"{artifact_name}.meta.json")
        if "--no-save_generated_dataset" not in cfg.extra_args:
            if os.path.exists(artifact_npz) and os.path.exists(artifact_meta):
                cmd.append("--no-save_generated_dataset")
            else:
                cmd.extend(["--save_generated_dataset"])
                cmd.extend(["--dataset_artifact_dir", artifact_dir])
                cmd.extend(["--dataset_artifact_name", artifact_name])
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
        # Eval can run many times per seed (baseline x decoding mode); avoid dataset clobber spam.
        if "--save_generated_dataset" not in cfg.extra_args:
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
            "extra_args": cfg.extra_args,
        },
    )

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
    cmd.extend(["--logic_rollout_horizon", str(args.logic_rollout_horizon)])
    cmd.extend(["--logic_temperature", str(args.logic_temperature)])
    cmd.extend(["--n_layer", str(args.n_layer)])
    cmd.extend(["--n_head", str(args.n_head)])
    cmd.extend(["--n_embd", str(args.n_embd)])
    cmd.extend(["--dropout", str(args.dropout)])
    cmd.extend(["--cb_longest_path_max_expansions", str(args.cb_longest_path_max_expansions)])
    cmd.extend(["--cb_policy_mix_sampling", str(args.cb_policy_mix_sampling)])
    cmd.extend(["--cb_state_semantics", str(args.cb_state_semantics)])
    cmd.extend(["--cb_policy_mix_normal_mean_mode", str(args.cb_policy_mix_normal_mean_mode)])
    if args.cb_policy_mix_normal_spec is not None:
        cmd.extend(["--cb_policy_mix_normal_spec", str(args.cb_policy_mix_normal_spec)])
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
    p.add_argument("--logic_rollout_horizon", type=int, default=2)
    p.add_argument("--logic_temperature", type=float, default=1.0)
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

    logic_alphas = _normalize_logic_alphas(list(args.alphas))
    baseline_keys = _baseline_keys(logic_alphas)
    train_cmd_common = _build_train_cmd_common(args)
    eval_cmd_common = _build_eval_cmd_common(args)

    manifest = {
        "output_root": args.output_root,
        "workers": worker_count,
        "gpu_worker_map": gpu_map,
        "require_gpu": bool(args.require_gpu),
        "stop_on_gpu_unavailable": bool(args.stop_on_gpu_unavailable),
        "continue_on_error": bool(args.continue_on_error),
        "skip_completed": not bool(args.no_skip_completed),
        "dry_run": bool(args.dry_run),
        "extra_args": extra,
        "train_cmd_common": train_cmd_common,
        "eval_cmd_common": eval_cmd_common,
        "decoding_modes": list(args.decoding_modes),
        "baseline_keys": baseline_keys,
        "logic_alphas_effective": logic_alphas,
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
            extra_args=list(extra),
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
