import glob
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from datasets.ltlgrid7_dataset import LTLGrid7SequenceDataset, layout_state_features
from envs.ltlgrid7_env import EMPTY, LTLGrid7Config, LTLGrid7Env
from models.dt_model import DecisionTransformerDiscrete
from planning.ltlgrid7_logic import LTLGrid7SymbolLogic, build_deep_dfa, cell_transition_tensor
from planning.ltlgrid_logic import OracleGrounder, SymbolGrounder, evaluate_grounder, machine_tensors, train_grounder

P, L, D, G = 0, 1, 2, 3


def make_layouts():
    """Two 7x7 layouts, indexed [x, y]: layout 0 has P at (1, 0) and L at (3, 0); layout 1 has them swapped."""
    a = np.full((7, 7), EMPTY, dtype=np.int64)
    a[1, 0], a[3, 0], a[5, 5], a[6, 6] = P, L, D, G
    b = np.full((7, 7), EMPTY, dtype=np.int64)
    b[1, 0], b[3, 0], b[5, 5], b[6, 6] = L, P, D, G
    return np.stack([a, b])


def make_env(task_id=0, reward_type="distance", scale=1.0, **kw):
    return LTLGrid7Env(LTLGrid7Config(make_layouts(), task_id=task_id, reward_type=reward_type, reward_scale=scale, **kw))


def test_state_token_roundtrip_and_moves():
    env = make_env()
    assert env.observation_space.n == 2 * 49 and env.action_space.n == 4
    s = env.pos_to_state((3, 5), layout=1)
    assert s == 49 + 3 * 7 + 5 and env.state_to_layout_pos(s) == (1, (3, 5))
    assert env.move((0, 0), 3) == (0, 0) and env.move((6, 6), 1) == (6, 6)  # walls
    assert env.move((2, 2), 0) == (2, 3) and env.move((2, 2), 1) == (3, 2)  # 0: y+1, 1: x+1


def test_symbol_is_object_on_landing_cell_of_the_layout():
    env = make_env()
    obs, _ = env.reset(layout=0, start_pos=(0, 0))
    _, _, _, info = env.step(1)  # (1, 0): pickaxe in layout 0
    assert info["symbol"] == P
    obs, _ = env.reset(layout=1, start_pos=(0, 0))
    _, _, _, info = env.step(1)  # same move, layout 1: lava
    assert info["symbol"] == L


def test_task1_visit_both_wins_and_reward_sums_to_100():
    env = make_env(task_id=0)
    env.reset(layout=0, start_pos=(0, 0))
    total, done = 0.0, False
    for a in (1, 1, 1):  # (1,0) P, (2,0) empty, (3,0) L
        _, r, done, info = env.step(a)
        total += r
    assert done and info["terminal_type"] == "G" and total == pytest.approx(100.0)


def test_avoid_task_ends_in_failure():
    env = make_env(task_id=6)  # task 7: visit P and L, never D
    env.reset(layout=0, start_pos=(5, 4))
    _, r, done, info = env.step(0)  # (5, 5) is the door
    assert done and info["terminal_type"] == "X" and r < 0


def test_max_steps_truncates():
    env = make_env(task_id=0, max_steps=3)
    env.reset(layout=0, start_pos=(6, 0))
    for _ in range(3):
        _, _, done, info = env.step(2)
    assert done and info.get("truncated") and "terminal_type" not in info


def test_fixed_test_episodes_follow_seed_modulo():
    env = make_env(test_layouts=[1, 0, 1], test_starts=[(0, 0), (2, 2), (4, 4)])
    for seed in range(6):
        obs, _ = env.reset(seed=seed)
        i = seed % 3
        assert env.state_to_layout_pos(obs) == ([1, 0, 1][i], [(0, 0), (2, 2), (4, 4)][i])
    obs, _ = env.reset(seed=11)  # without test episodes: random but reproducible, on an empty cell
    obs2, _ = make_env().reset(seed=11)
    assert obs2 == make_env().reset(seed=11)[0]


def test_state_features_are_the_symbolic_map_plus_agent():
    feats = layout_state_features(make_layouts())
    assert feats.shape == (2 * 49, 5 * 49)
    s = 49 + 1 * 7 + 0  # layout 1, agent on (1, 0)
    planes = feats[s].reshape(5, 7, 7)
    assert planes[L, 1, 0] == 1 and planes[P, 3, 0] == 1 and planes[4, 1, 0] == 1 and planes[4].sum() == 1
    assert planes[:4].sum() == 4  # one cell per object type in this toy layout


def random_trajectories(n, task_id, reward_type, seed=0):
    env = make_env(task_id=task_id, reward_type=reward_type)
    rng = np.random.RandomState(seed)
    out = []
    for i in range(n):
        layout = i % 2
        env.reset(layout=layout, start_pos=(int(rng.randint(7)), 6))
        pos, acts, rews, terms = [env._pos], [], [], []
        for _ in range(100):
            a = int(rng.randint(4))
            _, r, done, info = env.step(a)
            acts.append(a); rews.append(r); pos.append(env._pos); terms.append("terminal_type" in info)
            if done:
                break
        out.append((layout, pos, acts, rews, terms))
    return out


def write_h5(path, trajs, corrupt=None):
    import h5py

    layouts = make_layouts()
    with h5py.File(path, "w") as f:
        for i, (layout, pos, acts, rews, terms) in enumerate(trajs):
            g = f.create_group(str(i))
            g.attrs.update(layout=layout, length=len(acts), kind="noisy", seed=i)
            g["grid"] = layouts[layout].astype(np.int8)
            g["positions"] = np.asarray(pos, dtype=np.int8)
            g["actions"] = np.asarray(acts, dtype=np.int8)
            r = np.asarray(rews, dtype=np.float64)
            if corrupt == "reward" and i == 0:
                r[0] += 5.0
            g["rewards"] = r
            g["terminals"] = np.asarray(terms, dtype=bool)
            g["returns_to_go"] = np.cumsum(r[::-1])[::-1]


@pytest.mark.parametrize("reward_type,task_id", [("distance", 0), ("three_value", 6), ("distance", 3)])
def test_loader_replays_a_synthetic_file(tmp_path, reward_type, task_id):
    path = tmp_path / "Dataset" / reward_type / f"task{task_id + 1}" / "dataset_medium_fixed50.h5"
    path.parent.mkdir(parents=True)
    write_h5(path, random_trajectories(12, task_id, reward_type))
    ds = LTLGrid7SequenceDataset(path, sequence_length=80)
    assert len(ds.episodes_tokens) == 12 and ds.num_reward_checked == 12
    assert ds.layouts.shape == (2, 7, 7) and ds.state_features.shape == (98, 245)
    tr = ds.episode_transitions[0]
    assert tr[:, 0].max() < 98 and set(ds.episode_layout) == {0, 1}
    x, y, m = ds[0]
    assert x.shape == y.shape == m.shape


def test_loader_rejects_a_wrong_reward(tmp_path):
    path = tmp_path / "Dataset" / "distance" / "task1" / "dataset_medium_fixed50.h5"
    path.parent.mkdir(parents=True)
    write_h5(path, random_trajectories(4, 0, "distance"), corrupt="reward")
    with pytest.raises(ValueError, match="stored reward"):
        LTLGrid7SequenceDataset(path, sequence_length=80)


def test_dt_with_state_features_has_the_same_interface():
    feats = torch.from_numpy(layout_state_features(make_layouts()))
    dt = DecisionTransformerDiscrete(98, 4, 20, n_embd=32, n_layer=1, n_head=2, max_timestep=128, state_features=feats)
    states = torch.randint(0, 98, (3, 5))
    out = dt(states, torch.randint(0, 5, (3, 5)), torch.rand(3, 5), torch.arange(5)[None].expand(3, 5))
    assert out.shape == (3, 5, 4)
    assert "state_features" in dt.state_dict() and not hasattr(dt, "state_emb")
    with pytest.raises(ValueError):
        DecisionTransformerDiscrete(50, 4, 20, state_features=feats)


def toy_dataset(tmp_path, task_id=0, n=40):
    path = tmp_path / "Dataset" / "distance" / f"task{task_id + 1}" / "dataset_medium_fixed50.h5"
    path.parent.mkdir(parents=True, exist_ok=True)
    write_h5(path, random_trajectories(n, task_id, "distance"))
    return LTLGrid7SequenceDataset(path, sequence_length=80)


def test_oracle_and_supervised_grounders_on_the_7x7_dataset(tmp_path):
    ds = toy_dataset(tmp_path)
    g, stats = train_grounder(ds, mode="oracle")
    assert stats["dataset_acc"] == 1.0 and stats["grid_acc"] == 1.0
    g, stats = train_grounder(ds, mode="supervised", epochs=200)
    assert stats["dataset_acc"] > 0.99 and stats["reward_replay_episode_acc"] > 0.99


def logic_for(ds, grounder=None):
    return LTLGrid7SymbolLogic(ds.env, grounder or OracleGrounder(ds.env), reward_scale=0.01)


def test_logic_loss_horizon_one_matches_hand_computation(tmp_path):
    ds = toy_dataset(tmp_path)  # task 1: visit P and L; layout 0 has P at (1, 0)
    logic = logic_for(ds)
    state = ds.env.pos_to_state((0, 0), layout=0)  # moving right (action 1) lands on the pickaxe
    probs = torch.tensor([0.1, 0.6, 0.2, 0.1])
    logits = probs.log().reshape(1, 1, 4)
    loss = logic.loss(logits, torch.tensor([[state]]), torch.tensor([[0]]), torch.ones(1, 1), horizon=1)
    # one of P and L visited = 50 of 100 progress: expected progress 0.6 * 0.5, loss (1 - 0.3) / 2
    assert loss.item() == pytest.approx((1 - 0.6 * 0.5) / 2, abs=1e-5)


def test_logic_loss_uses_the_layout_of_the_state(tmp_path):
    ds = toy_dataset(tmp_path)
    logic = logic_for(ds)
    logits = torch.tensor([0.0, 5.0, 0.0, 0.0]).reshape(1, 1, 4)  # almost surely "right"
    a = logic.loss(logits, torch.tensor([[ds.env.pos_to_state((0, 0), 0)]]), torch.tensor([[0]]), torch.ones(1, 1), 1)
    # in layout 1 the cell (1, 0) is the lava, in layout 0 the pickaxe: same progress for "visit both" -> equal
    b = logic.loss(logits, torch.tensor([[ds.env.pos_to_state((0, 0), 1)]]), torch.tensor([[0]]), torch.ones(1, 1), 1)
    assert a.item() == pytest.approx(b.item(), abs=1e-5)
    # task 3 (P then L): reaching L first earns nothing, so the layouts must give different losses
    ds3 = toy_dataset(tmp_path, task_id=2)
    logic3 = logic_for(ds3)
    a3 = logic3.loss(logits, torch.tensor([[ds3.env.pos_to_state((0, 0), 0)]]), torch.tensor([[0]]), torch.ones(1, 1), 1)
    b3 = logic3.loss(logits, torch.tensor([[ds3.env.pos_to_state((0, 0), 1)]]), torch.tensor([[0]]), torch.ones(1, 1), 1)
    assert a3.item() < b3.item() - 0.1


def test_logic_loss_gradient_raises_the_winning_move(tmp_path):
    ds = toy_dataset(tmp_path, task_id=2)
    logic = logic_for(ds)
    logits = torch.zeros(1, 1, 4, requires_grad=True)
    state = ds.env.pos_to_state((0, 0), layout=0)
    loss = logic.loss(logits, torch.tensor([[state]]), torch.tensor([[0]]), torch.ones(1, 1), horizon=2)
    loss.backward()
    g = logits.grad[0, 0]
    assert torch.isfinite(g).all() and g[1] < 0 and g[1] == g.min()  # right is the way to the pickaxe


def test_masked_items_do_not_contribute(tmp_path):
    ds = toy_dataset(tmp_path)
    logic = logic_for(ds)
    st = torch.tensor([[ds.env.pos_to_state((0, 0), 0), ds.env.pos_to_state((3, 3), 1)]])
    logits = torch.randn(1, 2, 4)
    full = logic.loss(logits, st, torch.zeros(1, 2, dtype=torch.long), torch.tensor([[1.0, 0.0]]), horizon=3)
    first = logic.loss(logits[:, :1], st[:, :1], torch.zeros(1, 1, dtype=torch.long), torch.ones(1, 1), horizon=3)
    assert full.item() == pytest.approx(first.item(), abs=1e-6)


@pytest.mark.parametrize("task_id", [0, 2, 6, 9])
def test_deep_dfa_is_the_task_dfa(task_id):
    machine = make_env(task_id=task_id).machine
    dfa = build_deep_dfa(machine)
    T, _ = machine_tensors(machine, 0.01)
    assert torch.equal(dfa.trans_prob.cpu(), T)  # trans_prob[symbol, q, q'] = 1 for the DFA's transitions
    assert torch.allclose(dfa.fin_matrix[:, 0].cpu(), torch.tensor(machine.potentials) * 0.01)
    assert dfa.fin_matrix[:, 1].tolist() == [1.0 if a else 0.0 for a in machine.acceptance]


def reference_loss(logic, logits, states, q_ids, horizon, target):
    """The loss written with plain einsums over the DFA tensor (no DeepDFA), as an independent check."""
    env, n = logic.env, logic.num_cells
    T, pot = machine_tensors(env.machine, 0.01)
    A = logits.shape[-1]
    pi = torch.softmax(logits, -1).reshape(-1, A)
    flat = states.reshape(-1)
    g = logic.grounder.table().reshape(logic.num_layouts, n, A, -1)[flat // n]
    J = torch.nn.functional.one_hot(flat % n, n).float()[:, :, None] * torch.nn.functional.one_hot(q_ids.reshape(-1), T.shape[1]).float()[:, None, :]
    P = cell_transition_tensor(env)
    for _ in range(horizon):
        after = torch.einsum("nsq,yqr->nsyr", J, T)
        J = torch.einsum("na,ast,nsay,nsyr->ntr", pi, P, g, after)
    q = J.sum(1)
    if target == "potential":
        return ((1.0 - q @ pot) / 2.0).mean()
    acc = torch.tensor([1.0 if env.machine.is_success(i) else 0.0 for i in range(T.shape[1])])
    return (-torch.log((q @ acc).clamp(min=1e-6))).mean()


@pytest.mark.parametrize("target", ["potential", "accept"])
@pytest.mark.parametrize("task_id", [2, 6])
def test_deep_dfa_loss_equals_the_plain_einsum_loss(tmp_path, task_id, target):
    ds = toy_dataset(tmp_path, task_id=task_id)
    logic = LTLGrid7SymbolLogic(ds.env, OracleGrounder(ds.env), reward_scale=0.01, target=target)
    torch.manual_seed(0)
    states = torch.tensor([[ds.env.pos_to_state((1, 1), 0), ds.env.pos_to_state((4, 2), 1), ds.env.pos_to_state((0, 3), 0)]])
    q_ids = torch.tensor([[0, 1 if task_id == 2 else 0, 0]])
    logits = torch.randn(1, 3, 4)
    got = logic.loss(logits, states, q_ids, torch.ones(1, 3), horizon=4)
    want = reference_loss(logic, logits, states, q_ids, 4, target)
    assert got.item() == pytest.approx(want.item(), abs=1e-5)


# ---- real data (set LTLGRID7_DATA_DIR to the folder that contains Dataset/<reward>/task<N>/...) ----
DATA_DIR = os.environ.get("LTLGRID7_DATA_DIR")
real = pytest.mark.skipif(not DATA_DIR, reason="set LTLGRID7_DATA_DIR to the unzipped fixed50 archive")


@real
@pytest.mark.parametrize("reward", ["distance", "three_value"])
@pytest.mark.parametrize("task", [1, 4, 7, 10])
def test_real_files_replay_exactly(reward, task):
    path = Path(DATA_DIR) / "Dataset" / reward / f"task{task}" / "dataset_medium_expert_fixed50.h5"
    ds = LTLGrid7SequenceDataset(path, sequence_length=80)
    assert len(ds.episodes_tokens) == 5000 and ds.layouts.shape == (50, 7, 7)
    assert ds.num_reward_checked == 5000
    assert len(ds.env.cfg.test_layouts) == 500


@real
def test_real_oracle_grounder_matches_the_dataset_symbols():
    path = Path(DATA_DIR) / "Dataset" / "distance" / "task3" / "dataset_medium_fixed50.h5"
    ds = LTLGrid7SequenceDataset(path, sequence_length=80, max_episodes=300, subsample_seed=0)
    stats = evaluate_grounder(OracleGrounder(ds.env), ds)
    assert stats["dataset_acc"] == 1.0 and stats["reward_replay_episode_acc"] == 1.0


# ---- comparison of grounders: runner result files and figures ----
def test_alpha_figures_draw_one_line_per_grounder(tmp_path):
    import csv

    from scripts import ltlgrid_alpha_figures as figs
    from scripts.ltlgrid_train_all import COLUMNS

    rows = []
    for seed in range(3):
        rows.append(["distance", 1, "medium_replay", seed, 0.0, "none", 0.5, 0.0, 0.5, 100, 0.4, 0.0, 1, "", ""])
        for g, gain in (("oracle", 0.3), ("supervised", 0.25), ("label_free", 0.1)):
            for a in (0.5, 0.9):
                rows.append(["distance", 1, "medium_replay", seed, a, g, 0.5 + gain * a, 0.0, 0.5, 100, 0.4, 0.2, 1, 1.0, 1.0])
    with open(tmp_path / "results.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(COLUMNS); w.writerows(rows)
    lines = figs.split_by_grounder(figs.load_rows([tmp_path]))
    assert [g for g, _ in lines] == ["oracle", "supervised", "label_free"]
    for g, rs in lines:  # every line starts from the shared alpha = 0 baseline
        assert {r["alpha"] for r in rs} == {0.0, 0.5, 0.9}
    figs.main(["--results", str(tmp_path), "--out_dir", str(tmp_path / "fig")])
    assert (tmp_path / "fig" / "distance" / "medium_replay" / "task1_alpha_sweep.png").exists()
