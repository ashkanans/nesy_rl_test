from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from logic.token_schema import TokenField, TokenSchemaDefinition
from models.dynamics_model import NeuralDiscreteDynamics
from planning.dynamics_runtime import (
    build_dataset_tabular_dynamics,
    build_offline_transition_examples,
    fit_neural_dynamics_model,
    neural_dynamics_to_transition_tensor,
)
from scripts.train_dt import get_arg_parser, train


class _FakeDataset:
    def __init__(self, *, episodes_tokens, num_states, num_actions, token_schema=None, schema_id=None):
        self.episodes_tokens = episodes_tokens
        self.env = SimpleNamespace(
            observation_space=SimpleNamespace(n=int(num_states)),
            action_space=SimpleNamespace(n=int(num_actions)),
        )
        self.token_schema = token_schema
        self.schema_id = schema_id


def _schema_with_swapped_positions() -> TokenSchemaDefinition:
    return TokenSchemaDefinition(
        schema_id="test_swapped_v1",
        env_name="test",
        width=4,
        dtype="int64",
        fields=(
            TokenField("reward", 0, "reward"),
            TokenField("state", 1, "state"),
            TokenField("action", 2, "action"),
            TokenField("aux", 3, "aux"),
        ),
    )


def test_schema_aware_extraction():
    schema = _schema_with_swapped_positions()
    dataset = _FakeDataset(
        episodes_tokens=[
            np.asarray(
                [
                    [7, 0, 1, 0],
                    [7, 2, 0, 0],
                    [7, 1, 1, 0],
                    [0, 3, 0, 0],
                ],
                dtype=np.int64,
            )
        ],
        num_states=4,
        num_actions=2,
        token_schema=schema,
        schema_id=schema.schema_id,
    )

    out = build_offline_transition_examples(dataset)
    assert out["state_idx"] == 1
    assert out["action_idx"] == 2
    np.testing.assert_array_equal(out["states"], np.asarray([0, 2], dtype=np.int64))
    np.testing.assert_array_equal(out["actions"], np.asarray([1, 0], dtype=np.int64))
    np.testing.assert_array_equal(out["next_states"], np.asarray([2, 1], dtype=np.int64))


def test_fallback_extraction_warns():
    dataset = _FakeDataset(
        episodes_tokens=[
            np.asarray(
                [
                    [0, 1, 0, 0],
                    [2, 0, 0, 0],
                    [1, 1, 0, 0],
                    [3, 0, 0, 0],
                ],
                dtype=np.int64,
            )
        ],
        num_states=4,
        num_actions=2,
    )

    with pytest.warns(UserWarning, match="token_schema"):
        out = build_offline_transition_examples(dataset)
    assert out["state_idx"] == 0
    assert out["action_idx"] == 1
    np.testing.assert_array_equal(out["states"], np.asarray([0, 2], dtype=np.int64))
    np.testing.assert_array_equal(out["actions"], np.asarray([1, 0], dtype=np.int64))
    np.testing.assert_array_equal(out["next_states"], np.asarray([2, 1], dtype=np.int64))


def test_build_dataset_tabular_dynamics():
    dataset = _FakeDataset(
        episodes_tokens=[
            np.asarray([[0, 0], [1, 0], [2, 0]], dtype=np.int64),
            np.asarray([[0, 0], [1, 0], [2, 0]], dtype=np.int64),
        ],
        num_states=3,
        num_actions=2,
    )

    probs, stats = build_dataset_tabular_dynamics(dataset)
    assert probs.shape == (2, 3, 3)
    assert probs[0, 0, 1] == pytest.approx(1.0)
    assert probs[1, 0, 0] == pytest.approx(1.0)
    assert stats["num_transition_examples"] == 2


def test_neural_discrete_dynamics_shape():
    model = NeuralDiscreteDynamics(num_states=5, num_actions=3, hidden_dim=16, num_layers=2)
    states = torch.tensor([0, 1, 2, 3], dtype=torch.long)
    actions = torch.tensor([1, 2, 0, 1], dtype=torch.long)
    logits = model(states, actions)
    assert logits.shape == (4, 5)


def test_fit_neural_dynamics_model_learns_tiny_dataset():
    dataset = _FakeDataset(
        episodes_tokens=[
            np.asarray([[0, 0], [1, 0], [0, 0], [1, 0], [2, 0]], dtype=np.int64),
            np.asarray([[2, 1], [0, 1], [2, 1], [0, 1], [2, 0]], dtype=np.int64),
            np.asarray([[0, 0], [1, 0], [0, 0], [1, 0], [2, 0]], dtype=np.int64),
            np.asarray([[2, 1], [0, 1], [2, 1], [0, 1], [2, 0]], dtype=np.int64),
        ],
        num_states=3,
        num_actions=2,
    )

    model, stats = fit_neural_dynamics_model(
        base_dataset=dataset,
        hidden_dim=32,
        num_layers=2,
        epochs=60,
        batch_size=8,
        lr=1e-2,
        weight_decay=0.0,
        val_fraction=0.0,
        device=torch.device("cpu"),
        seed=0,
    )

    assert isinstance(model, NeuralDiscreteDynamics)
    assert stats["train_accuracy_last"] > 0.8


def test_neural_dynamics_to_transition_tensor():
    model = NeuralDiscreteDynamics(num_states=4, num_actions=2, hidden_dim=8, num_layers=2)
    tensor = neural_dynamics_to_transition_tensor(
        model=model,
        num_states=4,
        num_actions=2,
        device=torch.device("cpu"),
        temperature=1.0,
        max_entries=1000,
    )
    assert tensor.shape == (2, 4, 4)
    sums = tensor.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)


def test_dt_smoke_with_neural_dynamics(tmp_path):
    parser = get_arg_parser()
    args = parser.parse_args(
        [
            "--env",
            "cb",
            "--smoke",
            "--logic_alpha",
            "0.1",
            "--dt_logic_dynamics_backend",
            "neural_dataset",
            "--dynamics_epochs",
            "3",
            "--dynamics_batch_size",
            "32",
            "--run_dir",
            str(tmp_path / "dt_cb_neural"),
        ]
    )

    model, base_dataset, run_dir = train(args)
    assert model is not None
    assert base_dataset is not None

    summary_path = Path(run_dir) / "dt_summary.json"
    assert summary_path.exists()
    payload = json.loads(summary_path.read_text())
    assert payload["dynamics_stats"]["backend"] == "neural_dataset"
    assert payload["dynamics_stats"]["pure_offline"] is True
