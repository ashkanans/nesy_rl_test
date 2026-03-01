#!/usr/bin/env python3
"""
Build a catalog of DSRL environments/datasets from the published wheel metadata.

This script is intentionally standalone and does not modify the existing training
or evaluation pipelines. It is safe to run in repositories that do not have DSRL
runtime dependencies installed.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import zipfile
from pathlib import Path


DEFAULT_WHEEL_GLOB = "dsrl_tmp/dsrl-*.whl"
DEFAULT_OUT_DIR = "artifacts/dsrl_catalog"


BULLET_KEYS = {
    "AntCircle",
    "AntRun",
    "BallCircle",
    "BallRun",
    "CarCircle",
    "CarRun",
    "DroneCircle",
    "DroneRun",
}

SAFETY_GYM_TASK_PAT = re.compile(
    r"^(Point|Car)(Circle|Goal|Button|Push)(1|2)$"
)
VELOCITY_PAT = re.compile(
    r"^(Ant|HalfCheetah|Hopper|Swimmer|Walker2d)Velocity$"
)
METADRIVE_PAT = re.compile(r"^(easy|medium|hard)(sparse|mean|dense)$")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="List DSRL datasets/envs and difficulties.")
    p.add_argument(
        "--wheel",
        type=str,
        default=None,
        help=(
            "Path to dsrl wheel (e.g. dsrl_tmp/dsrl-0.1.0-py3-none-any.whl). "
            "If omitted, script auto-discovers from dsrl_tmp/."
        ),
    )
    p.add_argument(
        "--out_dir",
        type=str,
        default=DEFAULT_OUT_DIR,
        help="Output directory for catalog artifacts.",
    )
    p.add_argument(
        "--print_table",
        action="store_true",
        help="Print a compact table to stdout.",
    )
    return p.parse_args()


def _find_wheel_path(user_path: str | None) -> Path:
    if user_path:
        p = Path(user_path)
        if not p.exists():
            raise FileNotFoundError(f"Wheel not found: {p}")
        return p
    matches = sorted(Path(".").glob(DEFAULT_WHEEL_GLOB))
    if not matches:
        raise FileNotFoundError(
            "No DSRL wheel found. Expected something like dsrl_tmp/dsrl-*.whl. "
            "Download with: ./.venv/bin/python -m pip download --no-deps dsrl==0.1.0 -d dsrl_tmp"
        )
    return matches[-1]


def _load_infos_from_wheel(wheel_path: Path) -> dict:
    with zipfile.ZipFile(wheel_path, "r") as zf:
        src = zf.read("dsrl/infos.py").decode("utf-8")
    scope: dict = {}
    exec(src, scope)  # infos.py is constant dict assignments
    required = [
        "DATASET_URLS",
        "DEFAULT_MAX_EPISODE_STEPS",
        "MAX_EPISODE_REWARD",
        "MIN_EPISODE_REWARD",
        "MAX_EPISODE_COST",
        "MIN_EPISODE_COST",
    ]
    missing = [k for k in required if k not in scope]
    if missing:
        raise RuntimeError(f"Missing fields in infos.py: {missing}")
    return {k: scope[k] for k in required}


def _classify_key(key: str) -> dict:
    if key in BULLET_KEYS:
        agent = re.match(r"^(Ant|Ball|Car|Drone)", key).group(1)  # type: ignore[union-attr]
        task = "Circle" if key.endswith("Circle") else "Run"
        return {
            "family": "bullet_safety_gym",
            "difficulty": "unspecified",
            "agent": agent,
            "task": task,
            "variant": "",
            "env_ids": [f"Offline{key}-v0"],
        }

    m = SAFETY_GYM_TASK_PAT.match(key)
    if m:
        agent, task, level = m.groups()
        return {
            "family": "safety_gymnasium",
            "difficulty": f"level_{level}",
            "agent": agent,
            "task": task,
            "variant": level,
            "env_ids": [f"Offline{key}-v0", f"Offline{key}Gymnasium-v0"],
        }

    m = VELOCITY_PAT.match(key)
    if m:
        agent = m.group(1)
        return {
            "family": "safety_gymnasium",
            "difficulty": "velocity",
            "agent": agent,
            "task": "Velocity",
            "variant": "v1",
            "env_ids": [f"Offline{key}-v1", f"Offline{key}Gymnasium-v1"],
        }

    m = METADRIVE_PAT.match(key)
    if m:
        road, density = m.groups()
        return {
            "family": "metadrive",
            "difficulty": road,
            "agent": "car",
            "task": "driving",
            "variant": density,
            "env_ids": [f"OfflineMetadrive-{key}-v0"],
        }

    return {
        "family": "unknown",
        "difficulty": "unknown",
        "agent": "",
        "task": "",
        "variant": "",
        "env_ids": [],
    }


def build_catalog(info: dict) -> list[dict]:
    urls = info["DATASET_URLS"]
    rows: list[dict] = []
    for key in sorted(urls.keys()):
        cls = _classify_key(key)
        rows.append(
            {
                "dataset_key": key,
                "family": cls["family"],
                "difficulty": cls["difficulty"],
                "agent": cls["agent"],
                "task": cls["task"],
                "variant": cls["variant"],
                "env_ids": cls["env_ids"],
                "dataset_url": urls.get(key),
                "max_episode_steps": info["DEFAULT_MAX_EPISODE_STEPS"].get(key),
                "max_episode_reward": info["MAX_EPISODE_REWARD"].get(key),
                "min_episode_reward": info["MIN_EPISODE_REWARD"].get(key),
                "max_episode_cost": info["MAX_EPISODE_COST"].get(key),
                "min_episode_cost": info["MIN_EPISODE_COST"].get(key),
            }
        )
    return rows


def build_summary(rows: list[dict]) -> dict:
    by_family: dict[str, dict] = {}
    for r in rows:
        fam = r["family"]
        item = by_family.setdefault(
            fam,
            {
                "count": 0,
                "difficulties": set(),
                "agents": set(),
                "tasks": set(),
            },
        )
        item["count"] += 1
        item["difficulties"].add(r["difficulty"])
        if r["agent"]:
            item["agents"].add(r["agent"])
        if r["task"]:
            item["tasks"].add(r["task"])
    for v in by_family.values():
        v["difficulties"] = sorted(v["difficulties"])
        v["agents"] = sorted(v["agents"])
        v["tasks"] = sorted(v["tasks"])
    return {
        "total_datasets": len(rows),
        "families": by_family,
    }


def write_csv(rows: list[dict], csv_path: Path) -> None:
    fields = [
        "dataset_key",
        "family",
        "difficulty",
        "agent",
        "task",
        "variant",
        "env_ids",
        "dataset_url",
        "max_episode_steps",
        "max_episode_reward",
        "min_episode_reward",
        "max_episode_cost",
        "min_episode_cost",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            row = dict(r)
            row["env_ids"] = ";".join(r.get("env_ids", []))
            writer.writerow(row)


def print_table(rows: list[dict]) -> None:
    print(
        f"{'dataset_key':28s} {'family':20s} {'difficulty':12s} "
        f"{'task':10s} {'max_steps':9s} {'max_cost':8s}"
    )
    print("-" * 100)
    for r in rows:
        print(
            f"{r['dataset_key'][:28]:28s} {r['family'][:20]:20s} {r['difficulty'][:12]:12s} "
            f"{str(r['task'])[:10]:10s} {str(r['max_episode_steps']):9s} {str(r['max_episode_cost']):8s}"
        )


def main() -> None:
    args = parse_args()
    wheel_path = _find_wheel_path(args.wheel)
    info = _load_infos_from_wheel(wheel_path)
    rows = build_catalog(info)
    summary = build_summary(rows)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "dsrl_catalog.json"
    csv_path = out_dir / "dsrl_catalog.csv"
    summary_path = out_dir / "dsrl_catalog_summary.json"

    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_csv(rows, csv_path)

    print(f"Wheel: {wheel_path}")
    print(f"Wrote: {json_path}")
    print(f"Wrote: {csv_path}")
    print(f"Wrote: {summary_path}")
    print(f"Total datasets: {summary['total_datasets']}")
    for fam, details in sorted(summary["families"].items()):
        print(
            f"- {fam}: count={details['count']} difficulties={details['difficulties']} "
            f"agents={details['agents']}"
        )
    if args.print_table:
        print()
        print_table(rows)


if __name__ == "__main__":
    main()

