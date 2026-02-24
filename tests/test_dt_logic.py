import json
import subprocess
import sys
from pathlib import Path

import torch

from planning.dt_runtime import compute_dt_logic_rollout_penalty


def test_dt_logic_penalty_is_finite_and_backpropagates():
    torch.manual_seed(0)
    logits = torch.randn(2, 4, 3, requires_grad=True)
    states = torch.tensor([[0, 1, 2, 1], [2, 1, 0, 0]], dtype=torch.long)
    mask = torch.ones(2, 4, dtype=torch.float32)

    # 3 actions, 3 states.
    trans = torch.tensor(
        [
            [[1.0, 0.0, 0.0], [0.5, 0.5, 0.0], [0.0, 1.0, 0.0]],
            [[0.0, 1.0, 0.0], [0.0, 0.2, 0.8], [0.0, 0.0, 1.0]],
            [[0.8, 0.2, 0.0], [0.1, 0.9, 0.0], [0.0, 0.4, 0.6]],
        ],
        dtype=torch.float32,
    )
    hazard = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32)
    penalty = compute_dt_logic_rollout_penalty(
        logits=logits,
        states=states,
        attention_mask=mask,
        transition_probs=trans,
        hazard_mask=hazard,
        rollout_horizon=2,
        temperature=1.0,
    )
    assert torch.isfinite(penalty)
    penalty.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_train_dt_with_logic_smoke_writes_logic_metrics(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    run_dir = tmp_path / "dt_logic_train"
    cmd = [
        sys.executable,
        str(repo_root / "scripts" / "train_dt.py"),
        "--env",
        "frozenlake",
        "--smoke",
        "--logic_alpha",
        "0.5",
        "--logic_rollout_horizon",
        "2",
        "--run_dir",
        str(run_dir),
    ]
    subprocess.run(cmd, check=True, cwd=repo_root)
    metrics_path = run_dir / "metrics.json"
    assert metrics_path.exists()
    payload = json.loads(metrics_path.read_text())
    assert payload["model_type"] == "dt"
    assert "logic_loss" in payload
    assert payload["logic_loss"] is not None
