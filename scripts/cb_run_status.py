#!/usr/bin/env python3
"""Summarize ColourBomb DT matrix runs for terminal/Semaphore output."""

from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import time
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError as exc:
        return {"_error": f"invalid json: {exc}"}


def _fmt_duration(seconds: float | int | None) -> str:
    if seconds is None:
        return "unknown"
    seconds = max(0, int(seconds))
    days, rem = divmod(seconds, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, secs = divmod(rem, 60)
    if days:
        return f"{days}d {hours}h {minutes}m"
    if hours:
        return f"{hours}h {minutes}m {secs}s"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def _fmt_ts(ts: float | int | None) -> str:
    if not ts:
        return "unknown"
    return time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime(float(ts)))


def _short_path(path: Path) -> str:
    try:
        return str(path.relative_to(Path.cwd()))
    except ValueError:
        return str(path)


def _active_processes() -> list[str]:
    try:
        result = subprocess.run(
            ["ps", "-eo", "pid,etime,pcpu,pmem,args"],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
    except OSError:
        return []
    lines = []
    for line in result.stdout.splitlines():
        if "cb_dt_matrix.py" in line or "train_dt.py" in line or "eval_dt.py" in line:
            if "cb_run_status.py" not in line:
                lines.append(line.strip())
    return lines


def _main_log_for_run(run_dir: Path) -> Path | None:
    suffix = run_dir.name
    candidates = sorted(Path("logs").glob(f"*{suffix.replace('dt_', 'cb_dt_')}*.log"))
    if candidates:
        return candidates[-1]
    # Fallback: pick recent CB logs near the run creation time if exact naming differs.
    logs = sorted(Path("logs").glob("cb_dt*.log"), key=lambda p: p.stat().st_mtime if p.exists() else 0)
    return logs[-1] if logs else None


def _run_summary(run_dir: Path) -> dict[str, Any]:
    manifest = _load_json(run_dir / "matrix_manifest.json")
    progress = _load_json(run_dir / "matrix_progress.json")
    summary = _load_json(run_dir / "matrix_summary.json")
    results_file = _load_json(run_dir / "matrix_results.json")

    results = []
    for source in (progress, results_file):
        if isinstance(source.get("results"), list):
            results = source["results"]
    jobs_total = int(manifest.get("jobs_total") or len(manifest.get("jobs", [])) or summary.get("jobs_total") or 0)
    done = len(results)
    status_counts = Counter(str(r.get("status", "unknown")) for r in results if isinstance(r, dict))
    if summary.get("status_counts") and not status_counts:
        status_counts.update(summary.get("status_counts"))

    created_ts = manifest.get("created_ts")
    finished_ts = summary.get("finished_ts")
    updated_ts = progress.get("updated_ts") or finished_ts or created_ts
    now = time.time()
    elapsed_wall = (finished_ts or now) - created_ts if created_ts else None
    pct = (100.0 * done / jobs_total) if jobs_total else 0.0

    job_elapsed = [float(r.get("elapsed_sec")) for r in results if isinstance(r, dict) and r.get("elapsed_sec")]
    avg_job = mean(job_elapsed) if job_elapsed else None
    workers = int(manifest.get("workers") or 1)
    remaining = max(0, jobs_total - done)
    eta = None
    if remaining and avg_job:
        eta = remaining * avg_job / max(1, workers)

    run_state = "finished" if finished_ts else "running_or_interrupted"
    if jobs_total and done >= jobs_total:
        run_state = "finished"
    elif done == 0 and created_ts and now - float(created_ts) > 300:
        run_state = "starting_or_stalled"

    recent_results = []
    for r in results[-5:]:
        job = r.get("job", {}) if isinstance(r, dict) else {}
        recent_results.append(
            {
                "status": r.get("status", "unknown"),
                "spec": job.get("spec", "?"),
                "mix": job.get("policy_mix_spec", "?"),
                "seed": job.get("seed", "?"),
                "gpu": r.get("gpu_id", "?"),
                "elapsed": _fmt_duration(r.get("elapsed_sec")),
            }
        )

    return {
        "path": run_dir,
        "state": run_state,
        "jobs_total": jobs_total,
        "done": done,
        "remaining": remaining,
        "pct": pct,
        "workers": workers,
        "status_counts": dict(status_counts),
        "created_ts": created_ts,
        "updated_ts": updated_ts,
        "finished_ts": finished_ts,
        "elapsed_wall": elapsed_wall,
        "avg_job": avg_job,
        "eta": eta,
        "main_log": _main_log_for_run(run_dir),
        "recent_results": recent_results,
        "manifest_error": manifest.get("_error"),
        "progress_error": progress.get("_error"),
    }


def _discover_runs(root_glob: str, limit: int) -> list[Path]:
    paths = [Path(p) for p in glob.glob(root_glob)]
    paths = [p for p in paths if (p / "matrix_manifest.json").exists()]
    paths.sort(key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True)
    return paths[: max(1, limit)]


def _print_human(run_summaries: list[dict[str, Any]], show_processes: bool) -> None:
    print("ColourBomb DT Matrix Runs")
    print("=" * 80)
    if show_processes:
        procs = _active_processes()
        print("\nActive container processes:")
        if procs:
            for line in procs:
                print(f"  {line}")
        else:
            print("  none found")

    if not run_summaries:
        print("\nNo runs found.")
        return

    for idx, info in enumerate(run_summaries, start=1):
        print("\n" + "-" * 80)
        print(f"Run {idx}: {_short_path(info['path'])}")
        print(f"State      : {info['state']}")
        print(
            f"Progress   : {info['done']}/{info['jobs_total']} jobs "
            f"({info['pct']:.1f}%), remaining {info['remaining']}"
        )
        print(f"Workers    : {info['workers']}")
        print(f"Statuses   : {info['status_counts'] or '{}'}")
        print(f"Started    : {_fmt_ts(info['created_ts'])}")
        print(f"Updated    : {_fmt_ts(info['updated_ts'])}")
        if info["finished_ts"]:
            print(f"Finished   : {_fmt_ts(info['finished_ts'])}")
        print(f"Wall time  : {_fmt_duration(info['elapsed_wall'])}")
        print(f"Avg/job    : {_fmt_duration(info['avg_job'])}")
        print(f"ETA        : {_fmt_duration(info['eta'])}")
        if info["main_log"]:
            print(f"Main log   : {_short_path(info['main_log'])}")
        if info["manifest_error"] or info["progress_error"]:
            print(f"JSON issues: manifest={info['manifest_error']} progress={info['progress_error']}")
        if info["recent_results"]:
            print("Recent completed jobs:")
            for r in info["recent_results"]:
                print(
                    f"  {r['status']:>8} | gpu={r['gpu']} | seed={r['seed']} | "
                    f"{r['spec']} | {r['mix']} | {r['elapsed']}"
                )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root-glob",
        default="runs/cb/dt_*",
        help="Glob for matrix run directories, relative to the current repo root.",
    )
    parser.add_argument("--limit", type=int, default=5, help="Number of newest runs to show.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    parser.add_argument("--no-processes", action="store_true", help="Do not show active ps output.")
    args = parser.parse_args()

    runs = _discover_runs(args.root_glob, args.limit)
    summaries = [_run_summary(p) for p in runs]
    if args.json:
        serializable = []
        for info in summaries:
            item = dict(info)
            item["path"] = str(item["path"])
            item["main_log"] = str(item["main_log"]) if item["main_log"] else None
            serializable.append(item)
        print(json.dumps({"runs": serializable, "active_processes": _active_processes()}, indent=2))
    else:
        _print_human(summaries, show_processes=not args.no_processes)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
