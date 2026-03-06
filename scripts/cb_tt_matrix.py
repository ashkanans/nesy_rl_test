import argparse
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


REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable


@dataclass(frozen=True)
class Job:
    spec: str
    policy_mix_spec: str
    decoding_mode: str
    seed: int
    run_dir: str


@dataclass
class WorkerConfig:
    worker_id: int
    gpu_id: int | None
    require_gpu: bool
    stop_on_gpu_unavailable: bool
    continue_on_error: bool
    skip_completed: bool
    dry_run: bool
    cmd_common: list[str]
    extra_args: list[str]


def _slug(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", str(text)).strip("_")


def _gpu_available(gpu_id: int | None) -> bool:
    if gpu_id is None:
        return False
    cmd = ["nvidia-smi", "-i", str(gpu_id), "--query-gpu=index", "--format=csv,noheader"]
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True, timeout=10)
    except Exception:
        return False
    return str(gpu_id) in out


def _write_json(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def _run_single_job(job: Job, cfg: WorkerConfig) -> dict:
    started = time.time()
    os.makedirs(job.run_dir, exist_ok=True)
    metrics_csv = os.path.join(job.run_dir, "baseline_metrics.csv")
    if cfg.skip_completed and os.path.exists(metrics_csv):
        return {
            "status": "skipped",
            "job": asdict(job),
            "worker_id": cfg.worker_id,
            "gpu_id": cfg.gpu_id,
            "elapsed_sec": 0.0,
            "reason": "baseline_metrics.csv exists",
        }

    if cfg.require_gpu and not _gpu_available(cfg.gpu_id):
        return {
            "status": "gpu_unavailable",
            "job": asdict(job),
            "worker_id": cfg.worker_id,
            "gpu_id": cfg.gpu_id,
            "elapsed_sec": 0.0,
            "reason": "GPU check failed before job start",
        }

    run_args = {
        "spec": job.spec,
        "policy_mix_spec": job.policy_mix_spec,
        "decoding_mode": job.decoding_mode,
        "seed": int(job.seed),
        "worker_id": cfg.worker_id,
        "gpu_id": cfg.gpu_id,
        "require_gpu": cfg.require_gpu,
        "extra_args": cfg.extra_args,
    }
    _write_json(os.path.join(job.run_dir, "job_args.json"), run_args)

    cmd = list(cfg.cmd_common)
    cmd.extend([
        "--spec",
        job.spec,
        "--seed",
        str(job.seed),
        "--cb_policy_mix_spec",
        job.policy_mix_spec,
        "--decoding_mode",
        job.decoding_mode,
        "--base_run_dir",
        job.run_dir,
    ])
    if job.decoding_mode in {"beam", "constrained_beam"}:
        # beam/constrained settings are expected in cmd_common; no-op here.
        pass
    cmd.extend(cfg.extra_args)

    log_path = os.path.join(job.run_dir, "console.log")
    env = os.environ.copy()
    env["HOME"] = env.get("HOME", str(REPO_ROOT))
    env["MPLCONFIGDIR"] = env.get("MPLCONFIGDIR", str(REPO_ROOT / ".config" / "matplotlib"))
    os.makedirs(env["MPLCONFIGDIR"], exist_ok=True)
    if cfg.gpu_id is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(cfg.gpu_id)

    cmd_str = " ".join(shlex.quote(x) for x in cmd)
    if cfg.dry_run:
        with open(log_path, "a") as lf:
            lf.write(f"[dry-run] {cmd_str}\n")
        return {
            "status": "dry_run",
            "job": asdict(job),
            "worker_id": cfg.worker_id,
            "gpu_id": cfg.gpu_id,
            "elapsed_sec": float(time.time() - started),
            "command": cmd_str,
        }

    with open(log_path, "a") as lf:
        lf.write(f"[start] worker={cfg.worker_id} gpu={cfg.gpu_id} ts={time.time()}\n")
        lf.write(f"[cmd] {cmd_str}\n")
        lf.flush()
        proc = subprocess.run(cmd, cwd=REPO_ROOT, env=env, stdout=lf, stderr=lf)

    status = "ok" if proc.returncode == 0 else "failed"
    result = {
        "status": status,
        "returncode": int(proc.returncode),
        "job": asdict(job),
        "worker_id": cfg.worker_id,
        "gpu_id": cfg.gpu_id,
        "elapsed_sec": float(time.time() - started),
        "command": cmd_str,
    }

    if proc.returncode != 0 and cfg.require_gpu and not _gpu_available(cfg.gpu_id):
        result["status"] = "gpu_unavailable"
        result["reason"] = "GPU became unavailable during/after job"

    return result


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


def _build_jobs(args: argparse.Namespace) -> list[Job]:
    jobs: list[Job] = []
    for spec in args.specs:
        for mix in args.policy_mix_specs:
            for mode in args.decoding_modes:
                for seed in args.seeds:
                    run_dir = os.path.join(
                        args.output_root,
                        f"spec_{_slug(spec)}",
                        f"mix_{_slug(mix)}",
                        f"mode_{mode}",
                        f"seed_{int(seed)}",
                    )
                    jobs.append(
                        Job(
                            spec=str(spec),
                            policy_mix_spec=str(mix),
                            decoding_mode=str(mode),
                            seed=int(seed),
                            run_dir=run_dir,
                        )
                    )
    return jobs


def _build_common_cmd(args: argparse.Namespace) -> list[str]:
    cmd = [PYTHON, "scripts/run_baselines.py", "--env", "cb", "--evaluate", "--save_plots"]

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
    if args.cb_policy_mix_normal_spec is not None:
        cmd.extend(["--cb_policy_mix_normal_spec", str(args.cb_policy_mix_normal_spec)])

    cmd.extend(["--target_shift", str(args.target_shift)])
    cmd.extend(["--eval_num_episodes", str(args.eval_num_episodes)])
    cmd.extend(["--eval_max_steps", str(args.eval_max_steps)])
    cmd.extend(["--beam_width", str(args.beam_width)])
    cmd.extend(["--plan_horizon", str(args.plan_horizon)])
    cmd.extend(["--sat_rerank_weight", str(args.sat_rerank_weight)])

    if args.logic_state_only:
        cmd.append("--logic_state_only")
    if args.use_safe_dfa:
        cmd.append("--use_safe_dfa")
    if args.hard_prune_reject_sink:
        cmd.append("--hard_prune_reject_sink")
    if args.skip_dataset_analysis:
        cmd.append("--skip_dataset_analysis")

    cmd.extend(["--baselines", "vanilla", "logic"])
    cmd.extend(["--alphas", *[str(a) for a in args.alphas]])
    return cmd


def parse_args(argv: list[str] | None = None):
    p = argparse.ArgumentParser(description="ColourBomb TT matrix runner (multiprocess)")
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
    p.add_argument("--logic_sample_weighting", type=str, default="importance", choices=["importance", "uniform"])

    p.add_argument("--target_shift", type=str, default="token", choices=["token", "transition"])
    p.add_argument("--eval_num_episodes", type=int, default=200)
    p.add_argument("--eval_max_steps", type=int, default=200)
    p.add_argument("--beam_width", type=int, default=4)
    p.add_argument("--plan_horizon", type=int, default=4)
    p.add_argument("--sat_rerank_weight", type=float, default=1.0)
    p.add_argument("--hard_prune_reject_sink", action="store_true")

    p.add_argument("--cb_longest_path_max_expansions", type=int, default=500000)
    p.add_argument("--cb_policy_mix_sampling", type=str, choices=["fixed", "normal"], default="fixed")
    p.add_argument("--cb_policy_mix_normal_spec", type=str, default=None)
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

    common_cmd = _build_common_cmd(args)

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
        "common_cmd": common_cmd,
        "jobs_total": len(jobs),
        "jobs": [asdict(j) for j in jobs],
        "created_ts": time.time(),
    }
    _write_json(os.path.join(args.output_root, "matrix_manifest.json"), manifest)

    if args.dry_run:
        print(f"[dry-run] total_jobs={len(jobs)} output_root={args.output_root}")

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
            cmd_common=common_cmd,
            extra_args=list(extra),
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
                f"mode={job.get('decoding_mode')} seed={job.get('seed')} gpu={item.get('gpu_id')}"
            )
            _write_json(os.path.join(args.output_root, "matrix_progress.json"), {"results": results, "updated_ts": time.time()})
        except Empty:
            pass

        if stop_event.is_set():
            # Drain soon; do not start new work across workers.
            while True:
                try:
                    q_in.get_nowait()
                except Exception:
                    break

    for p in workers:
        p.join(timeout=2)

    # Drain any remaining queue items.
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
