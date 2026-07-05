import json
import subprocess
import sys
from pathlib import Path

import torch

from datasets.cb_dataset import CBSequenceDataset
from planning.dt_runtime import _advance_state_with_tokens, compute_dt_logic_rollout_penalty
from planning.product_value import (
    ProductValueTable,
    build_dfa_prefix_state_ids,
    compute_dt_dfa_product_value_loss,
    validate_product_value_table,
)
from train_cb import build_adapter_and_dfa


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


def test_product_value_loss_is_finite_and_backpropagates():
    torch.manual_seed(0)
    logits = torch.randn(2, 3, 2, requires_grad=True)
    states = torch.tensor([[0, 1, 2], [1, 0, 2]], dtype=torch.long)
    q_ids = torch.zeros_like(states)
    mask = torch.ones(2, 3, dtype=torch.float32)
    trans = torch.tensor(
        [
            [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]],
            [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        ],
        dtype=torch.float32,
    )
    phi = torch.tensor([[-2.0, -1.0, 0.0]], dtype=torch.float32)
    next_q = torch.zeros(1, 2, 3, dtype=torch.long)
    loss = compute_dt_dfa_product_value_loss(
        logits=logits,
        states=states,
        dfa_state_ids=q_ids,
        attention_mask=mask,
        transition_probs=trans,
        phi=phi,
        next_q=next_q,
    )
    assert torch.isfinite(loss)
    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_product_value_validation_rejects_bad_tables():
    good = ProductValueTable(
        phi=torch.tensor([[0.0, 0.0], [-5.0, -5.0]]).numpy(),
        costs=torch.tensor([[0.0, 0.0], [5.0, 5.0]]).numpy(),
        next_q=torch.zeros(2, 1, 2, dtype=torch.long).numpy(),
        accepting_q=torch.tensor([True, False]).numpy(),
        reject_q=torch.tensor([False, True]).numpy(),
        dmax=5.0,
        metadata={"num_actions": 1, "num_states": 2, "tol": 1e-6, "spec": "toy"},
    )
    report = validate_product_value_table(good, strict=True)
    assert report["passed"] is True

    bad = ProductValueTable(
        phi=torch.tensor([[0.0, 0.0], [0.0, 0.0]]).numpy(),
        costs=torch.tensor([[0.0, 0.0], [5.0, 5.0]]).numpy(),
        next_q=torch.zeros(2, 1, 2, dtype=torch.long).numpy(),
        accepting_q=torch.tensor([True, False]).numpy(),
        reject_q=torch.tensor([False, True]).numpy(),
        dmax=5.0,
        metadata={"num_actions": 1, "num_states": 2, "tol": 1e-6, "spec": "toy"},
    )
    try:
        validate_product_value_table(bad, strict=True)
    except ValueError as exc:
        assert "reject_states_not_distinctly_negative" in str(exc)
    else:
        raise AssertionError("bad product-value table should fail validation")


def test_cb_dfa_prefix_state_ids_match_full_eval_order_satisfaction():
    dataset = CBSequenceDataset(
        num_episodes=1,
        max_steps=8,
        sequence_length=16,
        seed=0,
        policy_mix_spec="random:1.0",
        state_semantics="pre",
    )
    args = type(
        "Args",
        (),
        {
            "env": "cb",
            "spec": "avoid_upward_action",
            "ltl_formula": None,
            "ltl_formulas": None,
            "constraint_dims": [0],
            "dfa_mode": "single",
            "use_safe_dfa": False,
            "dfa_backend": "auto",
        },
    )()
    adapter, _, raw_dfa = build_adapter_and_dfa(args, dataset)
    prefix_ids = build_dfa_prefix_state_ids(dataset, adapter, raw_dfa)
    rows = dataset.episodes_tokens[0][:-1]
    assert len(prefix_ids[0]) == rows.shape[0]

    q = 0
    q = _advance_state_with_tokens(
        adapter,
        raw_dfa,
        q,
        torch.tensor([int(rows[0, 0])], dtype=torch.long),
        token_offset=0,
    )
    assert int(prefix_ids[0][0]) == int(q)
    if rows.shape[0] > 1:
        q = _advance_state_with_tokens(
            adapter,
            raw_dfa,
            q,
            torch.tensor(
                [int(rows[0, 1]), int(rows[0, 2]), int(rows[0, 3]), int(rows[1, 0])],
                dtype=torch.long,
            ),
            token_offset=1,
        )
        assert int(prefix_ids[0][1]) == int(q)

    eval_tokens = [int(rows[0, 0])]
    for t in range(rows.shape[0] - 1):
        eval_tokens.extend(
            [
                int(rows[t, 1]),
                int(rows[t, 2]),
                int(rows[t, 3]),
                int(rows[t + 1, 0]),
            ]
        )
    eval_tokens.extend([int(adapter.end_token_id), 0, 0, 0])
    token_tensor = torch.tensor(eval_tokens, dtype=torch.long).view(1, -1)
    sat_by_eval_path = bool(adapter.check_sat_token_ids(token_tensor, raw_dfa)[0].item())

    q_final = q
    for t in range(1, rows.shape[0] - 1):
        q_final = _advance_state_with_tokens(
            adapter,
            raw_dfa,
            q_final,
            torch.tensor(
                [int(rows[t, 1]), int(rows[t, 2]), int(rows[t, 3]), int(rows[t + 1, 0])],
                dtype=torch.long,
            ),
            token_offset=1,
        )
    q_final = _advance_state_with_tokens(
        adapter,
        raw_dfa,
        q_final,
        torch.tensor([int(adapter.end_token_id), 0, 0, 0], dtype=torch.long),
        token_offset=1,
    )
    sat_by_prefix_replay = bool(raw_dfa.acceptance[q_final])
    assert sat_by_prefix_replay == sat_by_eval_path


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
