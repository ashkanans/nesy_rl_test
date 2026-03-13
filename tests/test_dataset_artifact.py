import argparse
import json

import numpy as np

from train_cb import save_dataset_artifact


class _DummyDataset:
    def __init__(self):
        self.schema_id = "cb/v1"
        self.observation_dim = 1
        self.action_dim = 1
        self.joined_dim = 4
        self.rows_per_seg = 2
        self.required_rows = 3
        self.episodes_tokens = [
            np.asarray([[1, 0, 0, 0], [2, 1, 0, 0], [99, 0, 0, 0]], dtype=np.int64),
            np.asarray([[4, 2, 0, 1], [99, 0, 0, 0]], dtype=np.int64),
        ]
        self.episode_rewards = [
            np.asarray([0.1, 0.2], dtype=np.float32),
            np.asarray([-1.0], dtype=np.float32),
        ]
        self.episode_policy_labels = ["random", "shortest_safe"]
        self.indices = [(0, 0), (1, 0)]

    def __len__(self):
        return len(self.indices)


def test_save_dataset_artifact_writes_npz_and_metadata(tmp_path):
    args = argparse.Namespace(
        save_generated_dataset=True,
        dataset_artifact_dir=None,
        dataset_artifact_name="dataset_snapshot",
        run_dir=str(tmp_path / "run"),
        save_path=str(tmp_path / "run"),
        base_runs_dir="runs",
        env="cb",
        seed=7,
        spec="avoid_bombs",
    )
    ds = _DummyDataset()

    info = save_dataset_artifact(args, ds, artifact_tag="dataset_train")
    assert info is not None
    assert info["npz_path"].endswith("dataset_snapshot.npz")
    assert info["meta_path"].endswith("dataset_snapshot.meta.json")

    npz = np.load(info["npz_path"], allow_pickle=True)
    assert "episodes_tokens" in npz
    assert len(npz["episodes_tokens"]) == 2
    assert "episode_rewards" in npz
    assert "episode_policy_labels" in npz
    assert "indices" in npz

    meta = json.loads(open(info["meta_path"]).read())
    assert meta["env"] == "cb"
    assert meta["seed"] == 7
    assert meta["num_episodes"] == 2
    assert meta["num_segments"] == 2


def test_save_dataset_artifact_respects_disable_flag(tmp_path):
    args = argparse.Namespace(
        save_generated_dataset=False,
        dataset_artifact_dir=str(tmp_path / "artifacts"),
        dataset_artifact_name="ignored",
        run_dir=str(tmp_path / "run"),
        save_path=str(tmp_path / "run"),
        base_runs_dir="runs",
        env="cb",
        seed=0,
        spec=None,
    )
    ds = _DummyDataset()
    info = save_dataset_artifact(args, ds, artifact_tag="dataset_train")
    assert info is None
