import numpy as np
import pytest
import torch

h5py = pytest.importorskip("h5py")
pytest.importorskip("flloat")

from datasets.ltlgrid_dataset import LTLGridSequenceDataset
from envs.ltlgrid_env import SYMBOL_TO_ID, LTLGridConfig, LTLGridEnv
from planning.ltlgrid_logic import (
    LTLGridSymbolLogic,
    OracleGrounder,
    SymbolGrounder,
    oracle_symbol_table,
    train_grounder,
)

# Down=0, right=1, up=2, left=3; state token = x * 4 + y


def _write_random_h5(path, task_id, n_episodes=300, seed=0):
    """Random-policy episodes from the env itself, in the upstream H5 layout (raw, unscaled rewards)."""
    env = LTLGridEnv(LTLGridConfig(task_id=task_id, reward_scale=1.0, max_steps=30))
    rng = np.random.RandomState(seed)
    with h5py.File(path, "w") as f:
        for i in range(n_episodes):
            start = env.cfg.starts[rng.randint(len(env.cfg.starts))]
            env.reset(start_pos=start)
            obs, acts, rews = [], [], []
            pos = start
            done = False
            while not done:
                a = int(rng.randint(4))
                obs.append(pos)
                acts.append(a)
                _, r, done, _ = env.step(a)
                rews.append(r)
                pos = env._cur_pos
            g = f.create_group(str(i))
            g.attrs["start_pos"] = np.array(start)
            g.attrs["length"] = len(acts)
            g.create_dataset("observations", data=np.array(obs, dtype=np.float32))
            g.create_dataset("actions", data=np.array(acts))
            g.create_dataset("rewards", data=np.array(rews, dtype=np.float64))
            g.create_dataset("returns_to_go", data=np.cumsum(rews[::-1])[::-1].astype(np.float64))
            g.create_dataset("dfa_states", data=np.zeros((len(acts), env.machine.num_states), dtype=np.float32))
            term = np.zeros(len(acts), dtype=bool)
            term[-1] = True
            g.create_dataset("terminals", data=term)


@pytest.fixture(scope="module")
def task1_dataset(tmp_path_factory):
    path = tmp_path_factory.mktemp("data") / "task1" / "dataset.h5"
    path.parent.mkdir()
    _write_random_h5(path, task_id=0)
    return LTLGridSequenceDataset(path, reward_scale=0.01)


def test_oracle_table_is_the_symbol_of_the_reached_cell():
    env = LTLGridEnv()
    table = oracle_symbol_table(env)
    # (0,1) -right-> (1,1) = pickaxe; (3,2) -down-> (3,3) = lava; (0,0) -right-> (1,0) = empty
    assert table[1, 1].argmax().item() == SYMBOL_TO_ID["P"]
    assert table[3 * 4 + 2, 0].argmax().item() == SYMBOL_TO_ID["L"]
    assert table[0, 1].argmax().item() == SYMBOL_TO_ID["E"]
    assert torch.allclose(table.sum(-1), torch.ones(16, 4))


def test_dataset_dfa_state_ids_follow_the_reward_machine(task1_dataset):
    ds = task1_dataset
    m = ds.env.machine
    for syms, ids in zip(ds.episode_symbols[:50], ds.dfa_state_ids[:50]):
        q = m.initial_state
        for sym, got in zip(syms, ids):
            assert got == q
            q = m.next_state(q, sym)


def test_supervised_grounder_reproduces_labels_and_rewards(task1_dataset):
    grounder, stats = train_grounder(task1_dataset, mode="supervised", epochs=300)
    assert stats["dataset_acc"] == 1.0
    assert stats["reward_replay_episode_acc"] == 1.0
    assert stats["grid_acc"] > 0.9


def test_rtg_conditioned_grounder_trains(task1_dataset):
    _, stats = train_grounder(task1_dataset, mode="supervised", use_rtg=True, epochs=300)
    assert stats["dataset_acc"] == 1.0
    assert "grid_acc" not in stats  # no (state, action) table exists with an rtg input


def test_weak_grounder_is_reward_consistent_without_labels(task1_dataset):
    # task1 is symmetric in P and L, so only the replayed rewards (not the labels) are identifiable
    _, stats = train_grounder(task1_dataset, mode="weak", epochs=300)
    assert stats["reward_replay_episode_acc"] > 0.95


def _logic(task_id=0, target="potential"):
    env = LTLGridEnv(LTLGridConfig(task_id=task_id))
    return LTLGridSymbolLogic(env, OracleGrounder(env), reward_scale=0.01, target=target), env


def _loss(logic, action, dfa_state, state, horizon=1):
    logits = torch.full((1, 1, 4), -20.0)
    logits[0, 0, action] = 20.0
    logits.requires_grad_(True)
    loss = logic.loss(
        logits,
        torch.tensor([[state]]),
        torch.tensor([[dfa_state]]),
        torch.ones(1, 1),
        horizon=horizon,
    )
    return loss, logits


def test_symbol_logic_loss_prefers_the_action_that_makes_progress():
    logic, _ = _logic(task_id=0)  # task1: visit pickaxe and lava
    state = 1  # cell (0,1); "right" reaches the pickaxe (1,1), "left" stays in place
    good, _ = _loss(logic, action=1, dfa_state=0, state=state)
    bad, _ = _loss(logic, action=3, dfa_state=0, state=state)
    assert good.item() == pytest.approx(0.25, abs=1e-3)  # potential 0.5 -> (1 - 0.5) / 2
    assert bad.item() == pytest.approx(0.5, abs=1e-3)


def test_symbol_logic_loss_gradient_points_to_the_progress_action():
    logic, _ = _logic(task_id=0)
    logits = torch.zeros(1, 1, 4, requires_grad=True)
    loss = logic.loss(logits, torch.tensor([[1]]), torch.tensor([[0]]), torch.ones(1, 1), horizon=1)
    loss.backward()
    assert logits.grad[0, 0].argmin().item() == 1  # decreasing the loss raises the right-action logit


def test_symbol_logic_loss_absorbing_states():
    logic, _ = _logic(task_id=0)
    accept = logic.env.machine.potentials.index(100.0)
    done, _ = _loss(logic, action=0, dfa_state=accept, state=5, horizon=3)
    assert done.item() == pytest.approx(0.0, abs=1e-6)

    logic7, _ = _logic(task_id=6)  # task7 avoids the door
    fail = logic7.env.machine.potentials.index(-100)
    lost, _ = _loss(logic7, action=0, dfa_state=fail, state=5, horizon=3)
    assert lost.item() == pytest.approx(1.0, abs=1e-6)


def test_symbol_logic_accept_target_is_log_probability():
    logic, _ = _logic(task_id=0, target="accept")
    accept = logic.env.machine.potentials.index(100.0)
    done, _ = _loss(logic, action=0, dfa_state=accept, state=5, horizon=1)
    assert done.item() == pytest.approx(0.0, abs=1e-5)


def test_logic_loss_respects_the_attention_mask():
    logic, _ = _logic(task_id=0)
    logits = torch.zeros(1, 2, 4)
    states = torch.tensor([[1, 1]])
    dfa = torch.tensor([[0, 0]])
    full = logic.loss(logits, states, dfa, torch.ones(1, 2), horizon=1)
    masked = logic.loss(logits, states, dfa, torch.tensor([[1.0, 0.0]]), horizon=1)
    assert full.item() == pytest.approx(masked.item(), abs=1e-6)  # identical positions
    none = logic.loss(logits, states, dfa, torch.zeros(1, 2), horizon=1)
    assert none.item() == 0.0


@pytest.mark.parametrize("mode", ["oracle", "supervised", "weak"])
def test_grounder_training_leaves_the_global_rng_untouched(task1_dataset, mode):
    # a DT built after the grounder must initialise exactly as without one, so that logic-on and
    # logic-off runs of a sweep are paired
    torch.manual_seed(123)
    before = torch.get_rng_state().clone()
    train_grounder(task1_dataset, mode=mode, epochs=5)
    assert torch.equal(before, torch.get_rng_state())
