import json
import subprocess
import sys
from pathlib import Path


def test_antmaze_eval_smoke_writes_metrics(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    run_dir = tmp_path / "antmaze_smoke"
    cmd = [
        sys.executable,
        str(repo_root / "antmaze_eval.py"),
        "--smoke",
        "--allow_train_fallback",
        "--run_dir",
        str(run_dir),
    ]
    subprocess.run(cmd, check=True, cwd=repo_root)

    metrics_path = run_dir / "metrics.json"
    assert metrics_path.exists()
    payload = json.loads(metrics_path.read_text())
    assert isinstance(payload, list)
    assert len(payload) >= 1
    row = payload[0]
    for key in ["return_mean", "runtime_sec", "env", "seed", "dataset_source"]:
        assert key in row
