import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

from planning.dt_runtime import build_dt_offline_source


def test_build_dt_offline_source_cb_uses_mix_and_state_semantics():
    args = SimpleNamespace(
        env="cb",
        num_episodes=8,
        max_steps=20,
        context_len=6,
        stochastic=False,
        seed=7,
        cb_policy_mix_spec="random:0.6,shortest_safe:0.4",
        cb_policy_mix_sampling="normal",
        cb_policy_mix_normal_spec="random:0.6:0.2,shortest_safe:0.4:0.2",
        cb_policy_mix_normal_mean_mode="base",
        cb_state_semantics="pre",
        cb_longest_path_max_expansions=1000,
    )

    dataset, skip_reason = build_dt_offline_source(args)
    assert skip_reason is None
    assert dataset.policy_mix_spec == "random:0.6,shortest_safe:0.4"
    assert dataset.policy_mix_sampling == "normal"
    assert dataset.policy_mix_normal_spec == "random:0.6:0.2,shortest_safe:0.4:0.2"
    assert dataset.policy_mix_normal_mean_mode == "base"
    assert dataset.state_semantics == "pre"
    assert "shortest_safe" in dataset.policy_mix_names


def test_cb_dt_matrix_dry_run_smoke_writes_manifest_and_summary(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    out_dir = tmp_path / "cb_dt_matrix_dry"
    cmd = [
        sys.executable,
        str(repo_root / "scripts" / "cb_dt_matrix.py"),
        "--output_root",
        str(out_dir),
        "--specs",
        "avoid_single_bomb_22",
        "--policy_mix_specs",
        "random:0.7,shortest_safe:0.3",
        "--decoding_modes",
        "greedy",
        "constrained",
        "--alphas",
        "0.0",
        "0.1",
        "--seeds",
        "0",
        "--epochs",
        "1",
        "--parallel_workers",
        "1",
        "--dry_run",
    ]
    subprocess.run(cmd, check=True, cwd=repo_root)

    manifest_path = out_dir / "matrix_manifest.json"
    summary_path = out_dir / "matrix_summary.json"
    assert manifest_path.exists()
    assert summary_path.exists()

    manifest = json.loads(manifest_path.read_text())
    summary = json.loads(summary_path.read_text())

    assert manifest["optimization_mode"] == "train_once_shared_decode"
    assert manifest["logic_alphas_effective"] == [0.1]
    assert manifest["baseline_keys"] == ["vanilla", "logic_alpha0.1"]
    assert manifest["jobs_total"] == 1
    assert summary["status_counts"].get("ok") == 1

    mode_csv = (
        out_dir
        / "spec_avoid_single_bomb_22"
        / "mix_random_0.7_shortest_safe_0.3"
        / "mode_greedy"
        / "seed_0"
        / "baseline_metrics.csv"
    )
    assert mode_csv.exists()

    train_log = (
        out_dir
        / "spec_avoid_single_bomb_22"
        / "mix_random_0.7_shortest_safe_0.3"
        / "seed_0"
        / "train_shared"
        / "vanilla"
        / "console.log"
    )
    assert train_log.exists()
    train_text = train_log.read_text()
    assert "--save_generated_dataset" in train_text
    assert "--dataset_artifact_dir" in train_text
    assert "--dataset_artifact_name" in train_text
