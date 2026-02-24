import json
import subprocess
import sys
import time
from pathlib import Path


def _run_doctor(repo_root: Path, env_name: str, spec: str, output_json: Path):
    cmd = [
        sys.executable,
        "-m",
        "tools.doctor",
        "--env",
        env_name,
        "--spec",
        spec,
        "--smoke",
        "--output_json",
        str(output_json),
    ]
    t0 = time.time()
    subprocess.run(cmd, check=True, cwd=repo_root)
    elapsed = time.time() - t0
    payload = json.loads(output_json.read_text())
    return elapsed, payload


def test_doctor_smoke_runs_under_30s_for_frozenlake_and_cb(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    out_fl = tmp_path / "doctor_frozenlake.json"
    out_cb = tmp_path / "doctor_cb.json"

    fl_elapsed, fl_payload = _run_doctor(
        repo_root=repo_root,
        env_name="frozenlake",
        spec="reach_goal_while_avoid_holes",
        output_json=out_fl,
    )
    cb_elapsed, cb_payload = _run_doctor(
        repo_root=repo_root,
        env_name="cb",
        spec="avoid_single_bomb_22",
        output_json=out_cb,
    )

    assert fl_elapsed < 30.0
    assert cb_elapsed < 30.0

    for payload, env_name in [(fl_payload, "frozenlake"), (cb_payload, "cb")]:
        assert payload["status"] == "ok"
        assert payload["env"] == env_name
        assert "checks" in payload
        assert "end_token_alignment" in payload["checks"]
        assert "symbol_coverage_dataset" in payload["checks"]
        assert "satisfaction_dataset" in payload["checks"]
