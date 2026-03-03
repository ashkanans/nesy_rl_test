from __future__ import annotations

import h5py
import numpy as np

from datasets.dsrl_dataset import DSRLSequenceDataset
from logic.token_schema import get_schema_for_env, validate_episode_tokens


def _write_tiny_dsrl_h5(path):
    # Two short episodes over 6 transitions total.
    observations = np.asarray(
        [
            [0.0, 0.0],
            [0.2, 0.1],
            [0.4, 0.2],
            [1.0, 1.0],
            [1.1, 1.0],
            [1.2, 1.1],
        ],
        dtype=np.float32,
    )
    actions = np.asarray(
        [
            [0.0],
            [0.5],
            [1.0],
            [0.0],
            [0.2],
            [0.9],
        ],
        dtype=np.float32,
    )
    rewards = np.asarray([0.0, 0.0, 1.0, 0.0, 0.0, 1.0], dtype=np.float32)
    costs = np.asarray([0.0, 1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    terminals = np.asarray([0, 0, 1, 0, 0, 1], dtype=np.float32)
    timeouts = np.zeros_like(terminals)

    with h5py.File(path, "w") as h5f:
        h5f.create_dataset("observations", data=observations)
        h5f.create_dataset("actions", data=actions)
        h5f.create_dataset("rewards", data=rewards)
        h5f.create_dataset("costs", data=costs)
        h5f.create_dataset("terminals", data=terminals)
        h5f.create_dataset("timeouts", data=timeouts)


def test_dsrl_dataset_builds_and_matches_schema(tmp_path):
    h5_path = tmp_path / "tiny_dsrl.hdf5"
    _write_tiny_dsrl_h5(h5_path)

    dataset = DSRLSequenceDataset(
        dataset_path=str(h5_path),
        sequence_length=8,
        seed=0,
        max_steps=10,
        num_episodes=8,
        state_bins=32,
        action_bins=8,
        reward_goal_threshold=0.0,
        target_shift="token",
        download=False,
    )
    assert len(dataset) > 0
    assert len(dataset.episodes_tokens) > 0

    schema = get_schema_for_env("dsrl")
    for ep in dataset.episodes_tokens:
        validate_episode_tokens(
            ep,
            schema=schema,
            end_token_id=dataset.end_token_id,
            observation_space_n=dataset.env.observation_space.n,
            action_space_n=dataset.env.action_space.n,
        )


def test_dsrl_replay_env_step(tmp_path):
    h5_path = tmp_path / "tiny_dsrl.hdf5"
    _write_tiny_dsrl_h5(h5_path)
    dataset = DSRLSequenceDataset(
        dataset_path=str(h5_path),
        sequence_length=8,
        seed=0,
        max_steps=10,
        num_episodes=8,
        state_bins=16,
        action_bins=4,
        target_shift="token",
        download=False,
    )

    obs, _ = dataset.env.reset(seed=0)
    assert 0 <= int(obs) < dataset.env.observation_space.n
    action = 0
    next_obs, reward, done, info = dataset.env.step(action)
    assert 0 <= int(next_obs) < dataset.env.observation_space.n
    assert isinstance(float(reward), float)
    assert isinstance(bool(done), bool)
    assert "cost" in info


def test_dsrl_cost_unsafe_threshold_changes_tokenization(tmp_path):
    h5_path = tmp_path / "tiny_dsrl_threshold.hdf5"
    _write_tiny_dsrl_h5(h5_path)

    base = DSRLSequenceDataset(
        dataset_path=str(h5_path),
        sequence_length=8,
        seed=0,
        max_steps=10,
        num_episodes=8,
        state_bins=16,
        action_bins=4,
        cost_unsafe_threshold=0.0,
        target_shift="token",
        download=False,
    )
    strict = DSRLSequenceDataset(
        dataset_path=str(h5_path),
        sequence_length=8,
        seed=0,
        max_steps=10,
        num_episodes=8,
        state_bins=16,
        action_bins=4,
        cost_unsafe_threshold=1.1,
        target_shift="token",
        download=False,
    )

    def _unsafe_count(ds):
        # schema is [state, action, reward, safety_cost]
        return int(sum(int(ep[:-1, 3].sum()) for ep in ds.episodes_tokens))

    assert _unsafe_count(base) > 0
    assert _unsafe_count(strict) == 0


def test_dsrl_cost_quantile_sets_threshold(tmp_path):
    h5_path = tmp_path / "tiny_dsrl_quantile.hdf5"
    _write_tiny_dsrl_h5(h5_path)

    ds = DSRLSequenceDataset(
        dataset_path=str(h5_path),
        sequence_length=8,
        seed=0,
        max_steps=10,
        num_episodes=8,
        state_bins=16,
        action_bins=4,
        cost_unsafe_quantile=1.0,
        target_shift="token",
        download=False,
    )
    assert ds.cost_unsafe_threshold >= 1.0
