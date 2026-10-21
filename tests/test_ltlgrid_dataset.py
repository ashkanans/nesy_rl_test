import os
from pathlib import Path

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")

from datasets.ltlgrid_dataset import LTLGridSequenceDataset
from envs.ltlgrid_env import SYMBOL_TO_ID, LTLGridEnv
from logic.token_schema import validate_episode_tokens

REAL_H5 = (
    Path(os.environ.get("LTLGRID_DATA_DIR", "/home/laptop-1029/Edu/susana_data_new"))
    / "Dataset/Symbolic/distance/task1/dataset_medium_expert.h5"
)


def _write_episode(f, key, start, actions, rewards, dfa, env, corrupt_obs=False):
    pos = [start]
    for a in actions[:-1]:
        pos.append(env.move(pos[-1], a))
    obs = np.array(pos, dtype=np.float32)
    if corrupt_obs:
        obs[1] = (obs[1] + 1) % 4
    g = f.create_group(key)
    g.attrs["start_pos"] = np.array(start)
    g.attrs["length"] = len(actions)
    g.create_dataset("observations", data=obs)
    g.create_dataset("actions", data=np.array(actions))
    g.create_dataset("rewards", data=np.array(rewards, dtype=np.float64))
    g.create_dataset("returns_to_go", data=np.cumsum(rewards[::-1])[::-1].astype(np.float64))
    onehot = np.zeros((len(actions), 3), dtype=np.float32)
    onehot[np.arange(len(actions)), dfa] = 1.0
    g.create_dataset("dfa_states", data=onehot)
    term = np.zeros(len(actions), dtype=bool)
    term[-1] = True
    g.create_dataset("terminals", data=term)


@pytest.fixture
def h5_path(tmp_path):
    env = LTLGridEnv()
    path = tmp_path / "task1" / "dataset.h5"
    path.parent.mkdir()
    with h5py.File(path, "w") as f:
        # (0,0) -down-> (0,1) -right-> (1,1)=pickaxe -right-> (2,1)
        _write_episode(f, "0", (0, 0), [0, 1, 1], [0.0, 50.0, 50.0], [0, 0, 1], env)
        _write_episode(f, "1", (0, 0), [1], [0.0], [0], env)
    return path


def test_loader_tokens_rewards_and_annotations(h5_path):
    ds = LTLGridSequenceDataset(h5_path, sequence_length=16)

    assert ds.task_id == 0
    assert len(ds.episodes_tokens) == 2
    ep = ds.episodes_tokens[0]
    validate_episode_tokens(ep, ds.token_schema, ds.end_token_id, 16, 4)
    # states (0,0),(0,1),(1,1) -> x*4+y = 0,1,5; actions 0,1,1
    assert ep[:-1, 0].tolist() == [0, 1, 5]
    assert ep[:-1, 1].tolist() == [0, 1, 1]

    # rewards[i] is kept aligned with actions[i], scaled by 0.01
    assert ds.episode_rewards[0].tolist() == pytest.approx([0.0, 0.5, 0.5])
    assert ds.dfa_state_ids[0].tolist() == [0, 0, 1]

    # symbol of step i is the cell reached by (s_i, a_i): (0,1)=E, (1,1)=P, (2,1)=E
    assert ds.episode_symbols[0].tolist() == [SYMBOL_TO_ID["E"], SYMBOL_TO_ID["P"], SYMBOL_TO_ID["E"]]
    # the terminal next state, absent from observations, is recovered from the dynamics
    assert ds.episode_transitions[0].tolist() == [[0, 0, 1, 0], [1, 1, 5, 0], [5, 1, 9, 1]]


def test_loader_batches_have_token_shift_and_short_episodes_are_padded(h5_path):
    ds = LTLGridSequenceDataset(h5_path, sequence_length=16)
    assert len(ds) >= 2
    for i in range(len(ds)):
        x, y, mask = ds[i]
        assert x.shape == y.shape == mask.shape
        assert (x[1:] == y[:-1]).all()
    # the 1-step episode is shorter than a segment, so part of its mask must be zero
    short = [i for i, (e, _) in enumerate(ds.indices) if e == 1][0]
    assert float(ds[short][2].sum()) < float(ds[short][2].numel())


def test_loader_rejects_misaligned_observations(tmp_path):
    env = LTLGridEnv()
    path = tmp_path / "bad.h5"
    with h5py.File(path, "w") as f:
        _write_episode(f, "0", (0, 0), [0, 1, 1], [0.0, 50.0, 50.0], [0, 0, 1], env, corrupt_obs=True)
    with pytest.raises(ValueError, match="misaligned"):
        LTLGridSequenceDataset(path, sequence_length=16)


@pytest.mark.skipif(not REAL_H5.exists(), reason="Susana's H5 datasets not downloaded")
def test_real_dataset_loads_and_is_consistent():
    ds = LTLGridSequenceDataset(REAL_H5, sequence_length=80)
    assert len(ds.episodes_tokens) == 2016
    assert ds.num_final_observation_checked == 2016 and ds.num_truncated_episodes == 0
    # distance reward: RTG of the first step of a successful episode is 100 (1.0 scaled)
    rtg0 = [float(r.sum()) for r in ds.episode_rewards]
    assert max(rtg0) == pytest.approx(1.0, abs=1e-4)
    # every positive reward is earned on a step that emits an object symbol
    for rew, sym in zip(ds.episode_rewards, ds.episode_symbols):
        assert (sym[rew != 0] != SYMBOL_TO_ID["E"]).all()


def test_final_observation_is_checked_against_the_dynamics(tmp_path):
    env = LTLGridEnv()

    def build(final):
        path = tmp_path / f"task1_{final[0]}{final[1]}" / "distance" / "d.h5"
        path.parent.mkdir(parents=True)
        with h5py.File(path, "w") as f:
            _write_episode(f, "0", (0, 0), [0, 1, 1], [0.0, 50.0, 50.0], [0, 0, 1], env)
            f["0"].create_dataset("final_observation", data=np.array(final, dtype=np.float32))
        return path

    # (0,0) -down-> (0,1) -right-> (1,1) -right-> (2,1): the last action reaches (2,1)
    ds = LTLGridSequenceDataset(build((2, 1)), sequence_length=16)
    assert ds.num_final_observation_checked == 1
    with pytest.raises(ValueError, match="final_observation"):
        LTLGridSequenceDataset(build((3, 3)), sequence_length=16)


def test_reward_type_is_inferred_from_the_path(tmp_path):
    env = LTLGridEnv()
    for rt in ("distance", "three_value"):
        path = tmp_path / "Symbolic" / rt / "task1" / "d.h5"
        path.parent.mkdir(parents=True)
        with h5py.File(path, "w") as f:
            _write_episode(f, "0", (0, 0), [0, 1], [0.0, 0.0], [0, 0], env)
        assert LTLGridSequenceDataset(path, validate_dynamics=False).reward_type == rt
