import os
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("flloat")

from envs.ltlgrid_env import SYMBOL_TO_ID, LTLGridConfig, LTLGridEnv
from logic.ltlgrid_dfa import LTLGRID_FORMULAS, LTLGridRewardMachine

# unzipped Dataset_Symbolic.zip: <LTLGRID_DATA_DIR>/Dataset/Symbolic/<reward>/task<N>/dataset_*.h5
DATA_DIR = Path(os.environ.get("LTLGRID_DATA_DIR", "/home/laptop-1029/Edu/susana_data_new"))

# Down=0, right=1, up=2, left=3
PICKAXE_THEN_LAVA = [0, 1, 1, 1, 0, 0]  # (0,0) -> (0,1) -> (1,1)=P -> (2,1) -> (3,1) -> (3,2) -> (3,3)=L


def test_reward_machine_sizes_match_upstream_datasets():
    # DFA widths of the one-hot dfa_states stored in Susana's datasets, tasks 1-10
    expected = [4, 8, 3, 9, 6, 12, 5, 5, 4, 4]
    got = [LTLGridRewardMachine(t).num_states for t in range(len(LTLGRID_FORMULAS))]
    assert got == expected


def test_distance_potentials_task1():
    rm = LTLGridRewardMachine(0, "distance")
    # states are numbered breadth-first from the initial state: 0 -P-> 1, 0 -L-> 2, both -> 3
    assert rm.potentials == [0.0, 50.0, 50.0, 100.0]
    q = rm.initial_state
    q1 = rm.next_state(q, SYMBOL_TO_ID["P"])
    assert rm.reward(q, q1) == 50.0
    assert rm.next_state(q1, SYMBOL_TO_ID["E"]) == q1  # empty cells never change the state
    assert rm.is_success(rm.next_state(q1, SYMBOL_TO_ID["L"]))


def test_avoidance_task_fails_on_door():
    rm = LTLGridRewardMachine(6, "distance")  # task7: visit(P, L) + glob_av(D)
    q = rm.next_state(rm.initial_state, SYMBOL_TO_ID["D"])
    assert rm.is_terminal(q) and not rm.is_success(q)
    assert rm.potentials[q] == -100


def test_env_runs_task_and_scales_reward():
    env = LTLGridEnv(LTLGridConfig(task_id=0, reward_scale=0.01))
    env.reset(start_pos=(0, 0))
    rewards, last = [], None
    for a in PICKAXE_THEN_LAVA:
        _, r, done, info = env.step(a)
        rewards.append(r)
        last = (done, info)
    assert rewards == pytest.approx([0, 0.5, 0, 0, 0, 0.5])  # P at step 2, L at step 6
    assert last[0] and last[1]["terminal_type"] == "G"


def test_env_without_task_has_no_reward_and_only_truncates():
    env = LTLGridEnv(LTLGridConfig(max_steps=3))
    env.reset(start_pos=(0, 0))
    outs = [env.step(1) for _ in range(3)]
    assert all(r == 0.0 for _, r, _, _ in outs)
    assert [d for _, _, d, _ in outs] == [False, False, True]
    assert outs[-1][3]["truncated"]


@pytest.mark.parametrize("reward", ["distance", "three_value"])
@pytest.mark.parametrize("task", range(1, 11))
def test_env_replays_upstream_datasets_exactly(task, reward):
    h5 = DATA_DIR / f"Dataset/Symbolic/{reward}/task{task}/dataset_medium_expert.h5"
    if not h5.exists():
        pytest.skip("Susana's H5 datasets not downloaded")
    pytest.importorskip("h5py")
    from datasets.ltlgrid_dataset import LTLGridSequenceDataset

    ds = LTLGridSequenceDataset(h5, reward_scale=0.01)
    env = ds.env
    for tr, rew, start in zip(ds.episode_transitions, ds.episode_rewards, ds.episode_start_pos):
        env.reset(start_pos=start)
        rs, done = [], False
        for a in tr[:, 1]:
            assert not done, "env ended before the stored episode did"
            _, r, done, _ = env.step(int(a))
            rs.append(r)
        assert done
        assert np.allclose(rs, rew, atol=1e-6)


def test_stored_dfa_tables_match_flloat():
    # logic/ltlgrid_dfa_tables.json is a cache of the flloat output; regenerate it with
    # `python -m logic.ltlgrid_dfa` if this fails after changing the formulas
    from logic.ltlgrid_dfa import SYMBOLS, _build_transitions, _build_transitions_flloat

    for formula, num_c_symbols, name in LTLGRID_FORMULAS:
        assert _build_transitions(formula, num_c_symbols, len(SYMBOLS)) == _build_transitions_flloat(
            formula, num_c_symbols, len(SYMBOLS)
        ), name


def test_three_value_rewards_follow_the_current_upstream_rule():
    # task7 (avoid the door): the accepting state is not absorbing (it can still fall into the failure
    # sink), but reaching it is a win; the failure sink is -100; everything else is 0
    rm = LTLGridRewardMachine(6, "three_value")
    for q in range(rm.num_states):
        if rm.acceptance[q]:
            assert rm.potentials[q] == 100
        elif q in rm.absorbing_states:
            assert rm.potentials[q] == -100
        else:
            assert rm.potentials[q] == 0
    # task1: purely sparse, 100 only in the accepting state
    assert sorted(LTLGridRewardMachine(0, "three_value").potentials) == [0, 0, 0, 100]
