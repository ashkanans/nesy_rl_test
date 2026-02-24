import json
import subprocess
import sys
from pathlib import Path


def test_sweep_smoke_writes_summary_and_pareto_artifacts(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    sweep_dir = tmp_path / "sweep_fl"
    cmd = [
        sys.executable,
        str(repo_root / "scripts" / "sweep.py"),
        "--env",
        "frozenlake",
        "--smoke",
        "--spec",
        "reach_goal_while_avoid_holes",
        "--use_safe_dfa",
        "--alphas",
        "0.1",
        "0.2",
        "--discounts",
        "0.99",
        "--temperatures",
        "0.5",
        "--num_samples_grid",
        "2",
        "--sweep_run_dir",
        str(sweep_dir),
    ]
    subprocess.run(cmd, check=True, cwd=repo_root)

    summary_json = sweep_dir / "summary.json"
    summary_csv = sweep_dir / "summary.csv"
    pareto_json = sweep_dir / "pareto_points.json"
    p1 = sweep_dir / "plots" / "pareto_return_vs_violation.png"
    p2 = sweep_dir / "plots" / "pareto_return_vs_satisfaction.png"

    assert summary_json.exists()
    assert summary_csv.exists()
    assert pareto_json.exists()
    assert p1.exists()
    assert p2.exists()

    payload = json.loads(summary_json.read_text())
    assert payload["env"] == "frozenlake"
    assert payload["num_combinations"] >= 1
