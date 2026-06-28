import argparse
from pathlib import Path

import numpy as np

from datasets.artifact_io import load_sequence_dataset_artifact, save_sequence_dataset_artifact
from scripts.evaluate import main as evaluate_main
from scripts.train import main as train_main
from train_cb import build_dataset
from planning.dt_runtime import build_dt_offline_source


def _make_dataset(class_name: str, **attrs):
    cls = type(class_name, (), {"__len__": lambda self: len(self.indices)})
    obj = cls()
    for key, value in attrs.items():
        setattr(obj, key, value)
    return obj


def _make_args(tmp_path: Path, env: str, **kwargs):
    base = dict(
        save_generated_dataset=True,
        dataset_artifact_dir=None,
        dataset_artifact_name="dataset_snapshot",
        dataset_artifact_path=None,
        run_dir=str(tmp_path / "run"),
        save_path=str(tmp_path / "run"),
        base_runs_dir="runs",
        env=env,
        seed=7,
        spec="avoid_bombs",
        target_shift="token",
        block_size=8,
        context_len=2,
    )
    base.update(kwargs)
    return argparse.Namespace(**base)


def test_cb_artifact_loader_reindexes_and_ignores_stored_indices(tmp_path):
    ds = _make_dataset(
        "CBSequenceDataset",
        schema_id="cb/v1",
        observation_dim=1,
        action_dim=1,
        joined_dim=4,
        rows_per_seg=2,
        required_rows=2,
        episodes_tokens=[
            np.asarray([[1, 0, 0, 0], [2, 1, 0, 0], [81, 0, 0, 0]], dtype=np.int64),
            np.asarray([[3, 2, 0, 0], [4, 1, 0, 0], [5, 0, 0, 0], [81, 0, 0, 0]], dtype=np.int64),
        ],
        episode_rewards=[
            np.asarray([0.1, 0.2], dtype=np.float32),
            np.asarray([0.0, 0.1, 0.2], dtype=np.float32),
        ],
        episode_policy_labels=["random", "shortest_safe"],
        indices=[(0, 0)],
    )
    args = _make_args(
        tmp_path,
        "cb",
        max_steps=5,
        stochastic=False,
        cb_state_semantics="post",
        spec="avoid_bombs",
    )
    info = save_sequence_dataset_artifact(args, ds)
    assert info is not None

    loaded_8 = load_sequence_dataset_artifact(info["npz_path"], sequence_length=8, target_shift="token")
    loaded_12 = load_sequence_dataset_artifact(info["npz_path"], sequence_length=12, target_shift="token")

    assert loaded_8.env.__class__.__name__ == "ColourBombGridworldV1Env"
    assert len(loaded_8.indices) == 4
    assert len(loaded_12.indices) == 3
    assert len(loaded_8.indices) != len(ds.indices)
    x, y, mask = loaded_8[0]
    assert x.shape == y.shape == mask.shape


def test_frozenlake_artifact_keeps_short_episode_and_reconstructs_env(tmp_path):
    ds = _make_dataset(
        "FrozenLakeSequenceDataset",
        schema_id="frozenlake/v1",
        observation_dim=1,
        action_dim=1,
        joined_dim=4,
        rows_per_seg=2,
        required_rows=3,
        episodes_tokens=[
            np.asarray([[1, 0, 0, 0], [16, 0, 0, 0]], dtype=np.int64),
        ],
        episode_rewards=[np.asarray([1.0], dtype=np.float32)],
        episode_policy_labels=None,
        indices=[(0, 0)],
    )
    args = _make_args(
        tmp_path,
        "frozenlake",
        spec="reach_goal_while_avoid_holes",
        frozenlake_map_size="4x4",
        frozenlake_is_slippery=False,
        max_steps=5,
    )
    info = save_sequence_dataset_artifact(args, ds)
    loaded = load_sequence_dataset_artifact(info["npz_path"], sequence_length=12, target_shift="token")

    assert loaded.env.__class__.__name__ == "FrozenLakeEnv"
    assert len(loaded) == 1
    x, y, mask = loaded[0]
    assert x.shape == y.shape == mask.shape
    assert float(mask.sum().item()) < float(mask.numel())
    assert loaded.env.hole_state_ids == [5, 7, 11, 12]


def test_nrm_nav_artifact_reconstructs_env(tmp_path):
    ds = _make_dataset(
        "NRMSafetySequenceDataset",
        schema_id="nrm_nav/v1",
        observation_dim=1,
        action_dim=1,
        joined_dim=4,
        rows_per_seg=2,
        required_rows=2,
        episodes_tokens=[
            np.asarray([[0, 1, 0, 0], [1, 2, 0, 1], [25, 0, 0, 0]], dtype=np.int64),
        ],
        episode_rewards=[np.asarray([0.0, -1.0], dtype=np.float32)],
        episode_policy_labels=None,
        indices=[(0, 0)],
    )
    args = _make_args(
        tmp_path,
        "nrm_nav",
        max_steps=5,
        stochastic=False,
        spec="avoid_state_11",
    )
    info = save_sequence_dataset_artifact(args, ds)
    loaded = load_sequence_dataset_artifact(info["npz_path"], sequence_length=8, target_shift="token")

    assert loaded.env.__class__.__name__ == "NRMSafetyNavEnv"
    assert len(loaded) == 2
    assert len(loaded.env.unsafe_positions) > 0


def test_dsrl_artifact_reconstructs_replay_env(tmp_path):
    ds = _make_dataset(
        "DSRLSequenceDataset",
        schema_id="dsrl/v1",
        observation_dim=1,
        action_dim=1,
        joined_dim=4,
        rows_per_seg=2,
        required_rows=2,
        episodes_tokens=[
            np.asarray([[0, 0, 1, 0], [1, 1, 0, 1], [2, 0, 0, 0]], dtype=np.int64),
        ],
        episode_rewards=[np.asarray([1.0, 0.0], dtype=np.float32)],
        episode_policy_labels=None,
        indices=[(0, 0)],
    )
    args = _make_args(
        tmp_path,
        "dsrl",
        spec="avoid_unsafe",
        dsrl_dataset_key="PointGoal1",
        dsrl_dataset_path=str(tmp_path / "dummy.h5"),
        max_steps=5,
    )
    info = save_sequence_dataset_artifact(args, ds)
    loaded = load_sequence_dataset_artifact(info["npz_path"], sequence_length=8, target_shift="token")

    assert loaded.env.__class__.__name__ == "DSRLReplayEnv"
    assert loaded.env.is_goal_state(1)
    assert loaded.env.is_unsafe_state(1)
    x, y, mask = loaded[0]
    assert x.shape == y.shape == mask.shape


def test_builders_use_dataset_artifact_path(tmp_path):
    ds = _make_dataset(
        "CBSequenceDataset",
        schema_id="cb/v1",
        observation_dim=1,
        action_dim=1,
        joined_dim=4,
        rows_per_seg=2,
        required_rows=2,
        episodes_tokens=[
            np.asarray([[1, 0, 0, 0], [2, 1, 0, 0], [81, 0, 0, 0]], dtype=np.int64),
        ],
        episode_rewards=[np.asarray([0.1, 0.2], dtype=np.float32)],
        episode_policy_labels=["random"],
        indices=[(0, 0)],
    )
    args = _make_args(
        tmp_path,
        "cb",
        max_steps=5,
        stochastic=False,
        cb_state_semantics="post",
        spec="avoid_bombs",
    )
    info = save_sequence_dataset_artifact(args, ds)

    build_args = _make_args(
        tmp_path,
        "cb",
        dataset_artifact_path=info["npz_path"],
        max_steps=5,
        stochastic=False,
        cb_state_semantics="post",
        spec="avoid_bombs",
        block_size=8,
    )
    loaded = build_dataset(build_args)
    assert len(loaded) == 2

    dt_args = _make_args(
        tmp_path,
        "cb",
        dataset_artifact_path=info["npz_path"],
        max_steps=5,
        stochastic=False,
        cb_state_semantics="post",
        spec="avoid_bombs",
        context_len=2,
    )
    base_dataset, skip_reason = build_dt_offline_source(dt_args)
    assert skip_reason is None
    assert len(base_dataset.episodes_tokens) == 1


def test_train_and_evaluate_with_dataset_artifact_path(tmp_path):
    ds = _make_dataset(
        "CBSequenceDataset",
        schema_id="cb/v1",
        observation_dim=1,
        action_dim=1,
        joined_dim=4,
        rows_per_seg=2,
        required_rows=2,
        episodes_tokens=[
            np.asarray([[1, 0, 0, 0], [2, 1, 0, 0], [81, 0, 0, 0]], dtype=np.int64),
            np.asarray([[3, 2, 0, 0], [4, 1, 0, 0], [81, 0, 0, 0]], dtype=np.int64),
        ],
        episode_rewards=[
            np.asarray([0.1, 0.2], dtype=np.float32),
            np.asarray([0.0, 0.1], dtype=np.float32),
        ],
        episode_policy_labels=["random", "random"],
        indices=[(0, 0), (1, 0)],
    )
    args = _make_args(
        tmp_path,
        "cb",
        max_steps=5,
        stochastic=False,
        cb_state_semantics="post",
        spec=None,
    )
    info = save_sequence_dataset_artifact(args, ds)
    artifact_path = info["npz_path"]

    train_dir = tmp_path / "train_run"
    eval_dir = tmp_path / "eval_run"
    train_main(
        [
            "--env",
            "cb",
            "--smoke",
            "--block_size",
            "8",
            "--dataset_artifact_path",
            artifact_path,
            "--no-save_generated_dataset",
            "--no_eval_after_train",
            "--run_dir",
            str(train_dir),
        ]
    )
    ckpt = train_dir / "cb_state_0.pt"
    assert ckpt.exists()

    evaluate_main(
        [
            "--env",
            "cb",
            "--smoke",
            "--block_size",
            "8",
            "--dataset_artifact_path",
            artifact_path,
            "--no-save_generated_dataset",
            "--checkpoint",
            str(ckpt),
            "--run_dir",
            str(eval_dir),
        ]
    )
    assert (eval_dir / "metrics.json").exists()
