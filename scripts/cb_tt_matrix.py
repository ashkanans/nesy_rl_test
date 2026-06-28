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
    extra_args: list[str]
    beam_width: int
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
    "target_bomb22_hit_rate",
    "target_hazard_hit_rate",
    "decoding_mode",
    "beam_width",
    "model_type",
    "checkpoint_path",
    "run_id",
    "timestamp_utc",
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
    # Keep dataset cache stable per seed+mix+semantics so repeated spec runs reuse same artifact.
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


def _checkpoint_path(train_root: str, baseline_key: str, epochs: int) -> str:
    return os.path.join(train_root, baseline_key, f"cb_state_{epochs - 1}.pt")


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


def _null_metrics(
    *,
    env: str,
    spec: str,
    seed: int,
    checkpoint_path: str | None,
    decoding_mode: str,
    beam_width: int,
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
        "target_bomb22_hit_rate": None,
        "target_hazard_hit_rate": None,
        "decoding_mode": decoding_mode,
        "beam_width": int(beam_width),
        "model_type": "tt",
        "checkpoint_path": checkpoint_path,
        "run_id": None,
        "timestamp_utc": None,
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
    keys = ["goal_rate", "bomb_hit_rate", "satisfaction_rate", "return_mean"]
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


def _train_once(job: Job, cfg: WorkerConfig) -> tuple[str, str | None]:
    train_log = os.path.join(job.train_root, "console.log")
    if cfg.skip_completed and _all_checkpoints_exist(job.train_root, cfg.baseline_keys, cfg.epochs):
        return "skipped", None

    cmd = list(cfg.train_cmd_common)
    cmd.extend(
        [
            "--spec",
            job.spec,
            "--seed",
            str(job.seed),
            "--cb_policy_mix_spec",
            job.policy_mix_spec,
            "--base_run_dir",
            job.train_root,
        ]
    )
    cmd.extend(cfg.extra_args)
    # Ensure one stable train dataset artifact per seed (and policy-mix/semantics).
    artifact_name = _build_seed_dataset_artifact_name(job, cfg)
    artifact_dir = cfg.dataset_artifact_dir or os.path.join(job.seed_root, "dataset_artifacts")
    artifact_npz = os.path.join(artifact_dir, f"{artifact_name}.npz")
    artifact_meta = os.path.join(artifact_dir, f"{artifact_name}.meta.json")
    if os.path.exists(artifact_npz) and os.path.exists(artifact_meta):
        cmd.extend(["--dataset_artifact_path", artifact_npz])
        cmd.append("--no-save_generated_dataset")
    elif "--no-save_generated_dataset" not in cfg.extra_args:
        cmd.extend(["--save_generated_dataset"])
        cmd.extend(["--dataset_artifact_dir", artifact_dir])
        cmd.extend(["--dataset_artifact_name", artifact_name])
    else:
        cmd.extend(["--dataset_artifact_name", artifact_name])
    rc, cmd_str = _run_subprocess(
        cmd, log_path=train_log, gpu_id=cfg.gpu_id, dry_run=cfg.dry_run, log_prefix="train"
    )
    if rc != 0:
        return "failed", cmd_str
    return "ok", cmd_str


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
                beam_width=cfg.beam_width,
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
                "--decoding_mode",
                mode,
                "--run_dir",
                baseline_dir,
            ]
        )
        if mode in {"beam", "constrained_beam"}:
            pass
        eval_cmd.extend(cfg.extra_args)
        # Eval can be called many times per seed (baseline x decoding mode); avoid dataset clobber spam.
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
                beam_width=cfg.beam_width,
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
            "extra_args": cfg.extra_args,
        },
    )

    train_status, train_cmd = _train_once(job, cfg)
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


def _build_train_cmd_common(args: argparse.Namespace, logic_alphas: list[float]) -> list[str]:
    cmd = [PYTHON, "scripts/run_baselines.py", "--env", "cb"]
    cmd.extend(["--num_episodes", str(args.num_episodes)])
    cmd.extend(["--max_steps", str(args.max_steps)])
    cmd.extend(["--epochs", str(args.epochs)])
    cmd.extend(["--batch_size", str(args.batch_size)])
    cmd.extend(["--block_size", str(args.block_size)])
    cmd.extend(["--discount", str(args.discount)])
    cmd.extend(["--n_layer", str(args.n_layer)])
    cmd.extend(["--n_head", str(args.n_head)])
    cmd.extend(["--n_embd", str(args.n_embd)])
    cmd.extend(["--action_weight", str(args.action_weight)])
    cmd.extend(["--temperature", str(args.temperature)])
    cmd.extend(["--num_samples", str(args.num_samples)])
    cmd.extend(["--logic_sample_weighting", str(args.logic_sample_weighting)])
    cmd.extend(["--cb_longest_path_max_expansions", str(args.cb_longest_path_max_expansions)])
    cmd.extend(["--cb_policy_mix_sampling", str(args.cb_policy_mix_sampling)])
    cmd.extend(["--cb_state_semantics", str(args.cb_state_semantics)])
    cmd.extend(["--cb_policy_mix_normal_mean_mode", str(args.cb_policy_mix_normal_mean_mode)])
    if args.cb_policy_mix_normal_spec is not None:
        cmd.extend(["--cb_policy_mix_normal_spec", str(args.cb_policy_mix_normal_spec)])
    cmd.extend(["--target_shift", str(args.target_shift)])
    if args.logic_state_only:
        cmd.append("--logic_state_only")
    if args.use_safe_dfa:
        cmd.append("--use_safe_dfa")
    if args.skip_dataset_analysis:
        cmd.append("--skip_dataset_analysis")
    if logic_alphas:
        cmd.extend(["--baselines", "vanilla", "logic"])
        cmd.extend(["--alphas", *[str(a) for a in logic_alphas]])
    else:
        cmd.extend(["--baselines", "vanilla"])
    return cmd


def _build_eval_cmd_common(args: argparse.Namespace) -> list[str]:
    cmd = [PYTHON, "scripts/evaluate.py", "--env", "cb"]
    # Avoid rebuilding huge offline datasets during checkpoint-only evaluation.
    eval_dataset_eps = args.eval_dataset_num_episodes
    if eval_dataset_eps is None:
        eval_dataset_eps = min(int(args.num_episodes), int(args.eval_num_episodes))
    cmd.extend(["--num_episodes", str(int(eval_dataset_eps))])
    cmd.extend(["--max_steps", str(args.eval_max_steps)])
    cmd.extend(["--eval_num_episodes", str(args.eval_num_episodes)])
    cmd.extend(["--eval_max_steps", str(args.eval_max_steps)])
    cmd.extend(["--target_shift", str(args.target_shift)])
    cmd.extend(["--beam_width", str(args.beam_width)])
    cmd.extend(["--plan_horizon", str(args.plan_horizon)])
    cmd.extend(["--sat_rerank_weight", str(args.sat_rerank_weight)])
    cmd.extend(["--cb_policy_mix_sampling", str(args.cb_policy_mix_sampling)])
    cmd.extend(["--cb_state_semantics", str(args.cb_state_semantics)])
    cmd.extend(["--cb_policy_mix_normal_mean_mode", str(args.cb_policy_mix_normal_mean_mode)])
    if args.cb_policy_mix_normal_spec is not None:
        cmd.extend(["--cb_policy_mix_normal_spec", str(args.cb_policy_mix_normal_spec)])
    if args.logic_state_only:
        cmd.append("--logic_state_only")
    if args.use_safe_dfa:
        cmd.append("--use_safe_dfa")
    if args.hard_prune_reject_sink:
        cmd.append("--hard_prune_reject_sink")
    cmd.append("--save_plots")
    return cmd


def parse_args(argv: list[str] | None = None):
    p = argparse.ArgumentParser(
        description="ColourBomb TT matrix runner (optimized: train once, decode many)"
    )
    p.add_argument("--output_root", type=str, required=True)
    p.add_argument("--specs", nargs="+", required=True)
    p.add_argument("--policy_mix_specs", nargs="+", required=True)
    p.add_argument(
        "--decoding_modes",
        nargs="+",
        default=["greedy", "beam", "constrained_beam"],
        choices=["greedy", "beam", "constrained_beam"],
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

    p.add_argument("--num_episodes", type=int, default=4000)
    p.add_argument("--max_steps", type=int, default=200)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--block_size", type=int, default=128)
    p.add_argument("--discount", type=float, default=0.99)
    p.add_argument("--n_layer", type=int, default=4)
    p.add_argument("--n_head", type=int, default=4)
    p.add_argument("--n_embd", type=int, default=128)
    p.add_argument("--action_weight", type=float, default=1.0)
    p.add_argument("--temperature", type=float, default=0.5)
    p.add_argument("--num_samples", type=int, default=32)
    p.add_argument(
        "--logic_sample_weighting", type=str, default="importance", choices=["importance", "uniform"]
    )

    p.add_argument("--target_shift", type=str, default="token", choices=["token", "transition"])
    p.add_argument("--eval_num_episodes", type=int, default=200)
    p.add_argument("--eval_max_steps", type=int, default=200)
    p.add_argument("--eval_dataset_num_episodes", type=int, default=None)
    p.add_argument("--beam_width", type=int, default=4)
    p.add_argument("--plan_horizon", type=int, default=4)
    p.add_argument("--sat_rerank_weight", type=float, default=1.0)
    p.add_argument("--hard_prune_reject_sink", action="store_true")

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
    p.add_argument("--logic_state_only", action="store_true")
    p.add_argument("--use_safe_dfa", action="store_true")
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

    logic_alphas = _normalize_logic_alphas(list(args.alphas))
    baseline_keys = _baseline_keys(logic_alphas)
    train_cmd_common = _build_train_cmd_common(args, logic_alphas)
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
            extra_args=list(extra),
            beam_width=int(args.beam_width),
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
