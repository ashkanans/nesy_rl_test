import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from planning.dt_runtime import build_dt_offline_source
from scripts import cb_dt_matrix


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


def test_dataset_artifact_extra_args_are_sanitized_and_extracted():
    extra = [
        "--save_generated_dataset",
        "--dataset_artifact_dir",
        "/tmp/global-artifacts",
        "--dataset_artifact_name",
        "snapshot_x",
        "--foo",
        "bar",
        "--no-save_generated_dataset",
        "--dataset_artifact_name",
        "snapshot_y",
    ]
    save, art_dir, art_name = cb_dt_matrix._extract_dataset_artifact_options(extra)
    assert save is False
    assert art_dir == "/tmp/global-artifacts"
    assert art_name == "snapshot_y"

    stripped = cb_dt_matrix._strip_dataset_artifact_flags(extra)
    assert stripped == ["--foo", "bar"]


def test_cb_dt_matrix_dry_run_eval_disables_dataset_saving(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    out_dir = tmp_path / "cb_dt_matrix_eval_no_save"
    train_root = (
        out_dir
        / "spec_avoid_single_bomb_22"
        / "mix_random_1.0"
        / "seed_0"
        / "train_shared"
    )
    for baseline in ["vanilla", "logic_alpha0.1"]:
        ckpt = train_root / baseline / "dt_state_0.pt"
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        ckpt.write_bytes(b"stub")

    cmd = [
        sys.executable,
        str(repo_root / "scripts" / "cb_dt_matrix.py"),
        "--output_root",
        str(out_dir),
        "--specs",
        "avoid_single_bomb_22",
        "--policy_mix_specs",
        "random:1.0",
        "--decoding_modes",
        "greedy",
        "--alphas",
        "0.1",
        "--seeds",
        "0",
        "--epochs",
        "1",
        "--parallel_workers",
        "1",
        "--dry_run",
        "--save_generated_dataset",
        "--dataset_artifact_dir",
        str(tmp_path / "global_dataset_dir"),
        "--dataset_artifact_name",
        "dataset_snapshot",
    ]
    subprocess.run(cmd, check=True, cwd=repo_root)

    eval_log = (
        out_dir
        / "spec_avoid_single_bomb_22"
        / "mix_random_1.0"
        / "mode_greedy"
        / "seed_0"
        / "vanilla"
        / "console.log"
    )
    assert eval_log.exists()
    eval_text = eval_log.read_text()
    assert "--no-save_generated_dataset" in eval_text
    assert "--dataset_artifact_dir" not in eval_text
    assert "--dataset_artifact_name" not in eval_text


def test_cb_dt_matrix_gpu_map_validation_fails_fast(monkeypatch, tmp_path):
    monkeypatch.setattr(cb_dt_matrix, "_visible_gpu_ids", lambda: [0, 1])
    with pytest.raises(ValueError, match="unavailable GPU ids"):
        cb_dt_matrix.main(
            [
                "--output_root",
                str(tmp_path / "out"),
                "--specs",
                "avoid_single_bomb_22",
                "--policy_mix_specs",
                "random:1.0",
                "--decoding_modes",
                "greedy",
                "--alphas",
                "0.1",
                "--seeds",
                "0",
                "--parallel_workers",
                "2",
                "--gpu_worker_map",
                "0,3",
                "--require_gpu",
                "--dry_run",
            ]
        )


def test_cb_dt_matrix_dry_run_runs_dataset_analysis_by_default(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    out_dir = tmp_path / "cb_dt_matrix_analysis_dry"
    cmd = [
        sys.executable,
        str(repo_root / "scripts" / "cb_dt_matrix.py"),
        "--output_root",
        str(out_dir),
        "--specs",
        "avoid_single_bomb_22",
        "--policy_mix_specs",
        "random:1.0",
        "--decoding_modes",
        "greedy",
        "--alphas",
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

    analysis_log = (
        out_dir
        / "spec_avoid_single_bomb_22"
        / "mix_random_1.0"
        / "seed_0"
        / "train_shared"
        / "dataset_analysis"
        / "console.log"
    )
    assert analysis_log.exists()
    assert "[dry-run:dataset-analysis]" in analysis_log.read_text()


def test_cb_dt_matrix_skip_dataset_analysis_disables_overview(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    out_dir = tmp_path / "cb_dt_matrix_skip_analysis_dry"
    cmd = [
        sys.executable,
        str(repo_root / "scripts" / "cb_dt_matrix.py"),
        "--output_root",
        str(out_dir),
        "--specs",
        "avoid_single_bomb_22",
        "--policy_mix_specs",
        "random:1.0",
        "--decoding_modes",
        "greedy",
        "--alphas",
        "0.1",
        "--seeds",
        "0",
        "--epochs",
        "1",
        "--parallel_workers",
        "1",
        "--dry_run",
        "--skip_dataset_analysis",
    ]
    subprocess.run(cmd, check=True, cwd=repo_root)

    analysis_log = (
        out_dir
        / "spec_avoid_single_bomb_22"
        / "mix_random_1.0"
        / "seed_0"
        / "train_shared"
        / "dataset_analysis"
        / "console.log"
    )
    assert not analysis_log.exists()
