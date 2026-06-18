from __future__ import annotations

import csv
import json
import os
import warnings
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from models.dynamics_model import NeuralDiscreteDynamics


def get_dataset_state_action_indices(base_dataset) -> tuple[int, int]:
    schema = getattr(base_dataset, "token_schema", None)
    if schema is None:
        warnings.warn(
            "Dataset has no token_schema; falling back to state_idx=0 and action_idx=1.",
            UserWarning,
        )
        return 0, 1
    return int(schema.field_index("state")), int(schema.field_index("action"))


def build_dynamics_dataset_identifiers(base_dataset, args=None) -> dict[str, Any]:
    ids = {
        "env": getattr(args, "env", None),
        "seed": int(getattr(args, "seed", 0)) if args is not None and hasattr(args, "seed") else None,
        "cb_policy_mix_spec": getattr(args, "cb_policy_mix_spec", None),
        "cb_policy_mix_sampling": getattr(args, "cb_policy_mix_sampling", None),
        "cb_state_semantics": getattr(args, "cb_state_semantics", None),
        "num_episodes": int(getattr(args, "num_episodes", 0))
        if args is not None and hasattr(args, "num_episodes")
        else None,
        "max_steps": int(getattr(args, "max_steps", 0))
        if args is not None and hasattr(args, "max_steps")
        else None,
        "schema_id": getattr(base_dataset, "schema_id", None),
        "dataset_class": base_dataset.__class__.__name__,
    }
    return ids


def build_offline_transition_examples(base_dataset) -> dict[str, Any]:
    num_states = int(base_dataset.env.observation_space.n)
    num_actions = int(base_dataset.env.action_space.n)
    state_idx, action_idx = get_dataset_state_action_indices(base_dataset)

    states: list[int] = []
    actions: list[int] = []
    next_states: list[int] = []
    skipped_rows = 0
    skipped_episodes = 0

    for ep in getattr(base_dataset, "episodes_tokens", []) or []:
        rows = np.asarray(ep)
        if rows.ndim != 2 or rows.shape[0] < 3:
            skipped_episodes += 1
            continue

        core_rows = rows[:-1]
        if core_rows.shape[0] < 2:
            skipped_episodes += 1
            continue

        state_semantics = str(getattr(base_dataset, "state_semantics", "pre"))
        for t in range(core_rows.shape[0] - 1):
            cur = core_rows[t]
            nxt = core_rows[t + 1]
            if max(state_idx, action_idx) >= cur.shape[0] or state_idx >= nxt.shape[0]:
                skipped_rows += 1
                continue
            try:
                s = int(cur[state_idx])
                # With post-state serialization, row[t].state is the state reached
                # after row[t].action. The action that leaves this state is stored
                # on the next row.
                action_row = nxt if state_semantics == "post" else cur
                a = int(action_row[action_idx])
                s_next = int(nxt[state_idx])
            except Exception:
                skipped_rows += 1
                continue

            if not (0 <= s < num_states and 0 <= s_next < num_states and 0 <= a < num_actions):
                skipped_rows += 1
                continue

            states.append(s)
            actions.append(a)
            next_states.append(s_next)

    states_arr = np.asarray(states, dtype=np.int64)
    actions_arr = np.asarray(actions, dtype=np.int64)
    next_states_arr = np.asarray(next_states, dtype=np.int64)

    unique_pairs = (
        np.unique(np.stack([states_arr, actions_arr], axis=1), axis=0)
        if states_arr.size > 0
        else np.zeros((0, 2), dtype=np.int64)
    )
    possible_pairs = int(num_states * num_actions)
    coverage_ratio = float(len(unique_pairs) / possible_pairs) if possible_pairs > 0 else 0.0

    return {
        "states": states_arr,
        "actions": actions_arr,
        "next_states": next_states_arr,
        "num_states": num_states,
        "num_actions": num_actions,
        "state_idx": int(state_idx),
        "action_idx": int(action_idx),
        "stats": {
            "num_transition_examples": int(states_arr.shape[0]),
            "num_unique_state_action_pairs": int(len(unique_pairs)),
            "possible_state_action_pairs": possible_pairs,
            "coverage_ratio": coverage_ratio,
            "skipped_rows": int(skipped_rows),
            "skipped_episodes": int(skipped_episodes),
            "state_semantics": str(getattr(base_dataset, "state_semantics", "pre")),
        },
    }


def build_dataset_tabular_dynamics(base_dataset) -> tuple[np.ndarray, dict[str, Any]]:
    examples = build_offline_transition_examples(base_dataset)
    num_states = int(examples["num_states"])
    num_actions = int(examples["num_actions"])
    counts = np.zeros((num_actions, num_states, num_states), dtype=np.float32)

    states = examples["states"]
    actions = examples["actions"]
    next_states = examples["next_states"]
    for s, a, s_next in zip(states, actions, next_states):
        counts[int(a), int(s), int(s_next)] += 1.0

    probs = np.zeros_like(counts)
    for a in range(num_actions):
        for s in range(num_states):
            total = float(counts[a, s].sum())
            if total > 0.0:
                probs[a, s] = counts[a, s] / total
            else:
                probs[a, s, s] = 1.0

    stats = dict(examples["stats"])
    stats.update(
        {
            "num_states": num_states,
            "num_actions": num_actions,
            "state_idx": int(examples["state_idx"]),
            "action_idx": int(examples["action_idx"]),
        }
    )
    return probs, stats


def freeze_dynamics_model(model: NeuralDiscreteDynamics) -> NeuralDiscreteDynamics:
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model


def _compute_dataset_accuracy(
    model: NeuralDiscreteDynamics,
    dataset: TensorDataset,
    batch_size: int,
    device: torch.device,
) -> tuple[float, float]:
    if len(dataset) == 0:
        return 0.0, float("nan")

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    total_loss = 0.0
    total_examples = 0
    correct = 0
    model.eval()
    with torch.no_grad():
        for states, actions, next_states in loader:
            states = states.to(device)
            actions = actions.to(device)
            next_states = next_states.to(device)
            logits = model(states, actions)
            loss = F.cross_entropy(logits, next_states, reduction="sum")
            total_loss += float(loss.item())
            total_examples += int(next_states.numel())
            correct += int((logits.argmax(dim=-1) == next_states).sum().item())
    mean_loss = total_loss / max(1, total_examples)
    accuracy = float(correct / max(1, total_examples))
    return mean_loss, accuracy


def fit_neural_dynamics_model(
    base_dataset,
    hidden_dim,
    num_layers,
    epochs,
    batch_size,
    lr,
    weight_decay,
    val_fraction,
    device,
    seed,
    temperature=1.0,
    dropout=0.0,
    freeze_after_fit=True,
    log_path: str | None = None,
) -> tuple[NeuralDiscreteDynamics, dict[str, Any]]:
    examples = build_offline_transition_examples(base_dataset)
    num_examples = int(examples["states"].shape[0])
    if num_examples <= 0:
        raise ValueError("Offline dynamics dataset is empty; cannot fit neural dynamics model.")

    states_t = torch.from_numpy(examples["states"]).long()
    actions_t = torch.from_numpy(examples["actions"]).long()
    next_states_t = torch.from_numpy(examples["next_states"]).long()

    full_dataset = TensorDataset(states_t, actions_t, next_states_t)
    val_size = int(num_examples * max(0.0, float(val_fraction)))
    if val_size <= 0 or val_size >= num_examples:
        train_dataset = full_dataset
        val_dataset = None
    else:
        gen = torch.Generator().manual_seed(int(seed))
        perm = torch.randperm(num_examples, generator=gen)
        val_idx = perm[:val_size]
        train_idx = perm[val_size:]
        if train_idx.numel() == 0:
            train_dataset = full_dataset
            val_dataset = None
        else:
            train_dataset = TensorDataset(
                states_t[train_idx],
                actions_t[train_idx],
                next_states_t[train_idx],
            )
            val_dataset = TensorDataset(
                states_t[val_idx],
                actions_t[val_idx],
                next_states_t[val_idx],
            )

    model = NeuralDiscreteDynamics(
        num_states=int(examples["num_states"]),
        num_actions=int(examples["num_actions"]),
        hidden_dim=int(hidden_dim),
        num_layers=int(num_layers),
        dropout=float(dropout),
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(lr),
        weight_decay=float(weight_decay),
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(batch_size),
        shuffle=True,
        drop_last=False,
        generator=torch.Generator().manual_seed(int(seed)),
    )

    log_rows: list[dict[str, Any]] = []
    train_loss_last = None
    for epoch in range(int(epochs)):
        model.train()
        running_loss = 0.0
        running_examples = 0
        for states, actions, next_states in train_loader:
            states = states.to(device)
            actions = actions.to(device)
            next_states = next_states.to(device)
            logits = model(states, actions)
            loss = F.cross_entropy(logits, next_states)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            batch_items = int(next_states.numel())
            running_loss += float(loss.item()) * batch_items
            running_examples += batch_items
        train_loss_last = running_loss / max(1, running_examples)

        val_loss_epoch = None
        val_accuracy_epoch = None
        if val_dataset is not None and len(val_dataset) > 0:
            val_loss_epoch, val_accuracy_epoch = _compute_dataset_accuracy(
                model=model,
                dataset=val_dataset,
                batch_size=int(batch_size),
                device=device,
            )

        row = {
            "epoch": int(epoch),
            "train_loss": float(train_loss_last),
            "val_loss": None if val_loss_epoch is None else float(val_loss_epoch),
            "val_accuracy": None if val_accuracy_epoch is None else float(val_accuracy_epoch),
            "num_train_examples": int(len(train_dataset)),
            "num_val_examples": int(0 if val_dataset is None else len(val_dataset)),
        }
        log_rows.append(row)
        if log_path:
            directory = os.path.dirname(log_path)
            if directory:
                os.makedirs(directory, exist_ok=True)
            with open(log_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(row.keys()))
                writer.writeheader()
                writer.writerows(log_rows)
            json_path = os.path.splitext(log_path)[0] + ".json"
            with open(json_path, "w") as f:
                json.dump(log_rows, f, indent=2)

    train_loss_eval, train_accuracy = _compute_dataset_accuracy(
        model=model,
        dataset=train_dataset,
        batch_size=int(batch_size),
        device=device,
    )
    val_loss_last = None
    val_accuracy_last = None
    if val_dataset is not None and len(val_dataset) > 0:
        val_loss_last, val_accuracy_last = _compute_dataset_accuracy(
            model=model,
            dataset=val_dataset,
            batch_size=int(batch_size),
            device=device,
        )

    if freeze_after_fit:
        freeze_dynamics_model(model)

    stats = dict(examples["stats"])
    stats.update(
        {
            "train_loss_last": float(train_loss_eval if train_loss_last is None else train_loss_last),
            "train_accuracy_last": float(train_accuracy),
            "val_loss_last": None if val_loss_last is None else float(val_loss_last),
            "val_accuracy_last": None if val_accuracy_last is None else float(val_accuracy_last),
            "num_states": int(examples["num_states"]),
            "num_actions": int(examples["num_actions"]),
            "state_idx": int(examples["state_idx"]),
            "action_idx": int(examples["action_idx"]),
            "hidden_dim": int(hidden_dim),
            "num_layers": int(num_layers),
            "temperature": float(temperature),
            "loaded_from_checkpoint": False,
            "training_log_path": log_path,
            "training_history": log_rows,
        }
    )
    return model, stats


def neural_dynamics_to_transition_tensor(
    model,
    num_states,
    num_actions,
    device,
    temperature=1.0,
    max_entries=10000000,
) -> torch.Tensor:
    num_states = int(num_states)
    num_actions = int(num_actions)
    entries = int(num_actions * num_states * num_states)
    if entries > int(max_entries):
        raise ValueError(
            "Neural dynamics transition tensor would require "
            f"{entries} entries; increase --dynamics_max_transition_entries "
            "or use tabular_dataset/tabular_env."
        )

    states = torch.arange(num_states, device=device).repeat_interleave(num_actions)
    actions = torch.arange(num_actions, device=device).repeat(num_states)
    with torch.no_grad():
        probs = model.predict_next_state_probs(states, actions, temperature=float(temperature))
    return probs.reshape(num_states, num_actions, num_states).permute(1, 0, 2).contiguous()


def save_neural_dynamics_checkpoint(
    path: str,
    model: NeuralDiscreteDynamics,
    stats: dict[str, Any],
    dataset_identifiers: dict[str, Any] | None = None,
) -> str:
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    payload = {
        "model_state_dict": model.state_dict(),
        "num_states": int(model.num_states),
        "num_actions": int(model.num_actions),
        "hidden_dim": int(model.hidden_dim),
        "num_layers": int(model.num_layers),
        "dropout": float(model.dropout),
        "stats": dict(stats),
        "dataset_identifiers": dict(dataset_identifiers or {}),
    }
    torch.save(payload, path)
    return path


def load_neural_dynamics_checkpoint(
    path: str,
    device: torch.device,
    expected_num_states: int,
    expected_num_actions: int,
    expected_hidden_dim: int,
    expected_num_layers: int,
    freeze_after_load: bool = True,
) -> tuple[NeuralDiscreteDynamics, dict[str, Any]]:
    payload = torch.load(path, map_location=device)
    required = ["model_state_dict", "num_states", "num_actions", "hidden_dim", "num_layers"]
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError(f"Dynamics checkpoint missing required keys: {missing}")

    observed = {
        "num_states": int(payload["num_states"]),
        "num_actions": int(payload["num_actions"]),
        "hidden_dim": int(payload["hidden_dim"]),
        "num_layers": int(payload["num_layers"]),
    }
    expected = {
        "num_states": int(expected_num_states),
        "num_actions": int(expected_num_actions),
        "hidden_dim": int(expected_hidden_dim),
        "num_layers": int(expected_num_layers),
    }
    mismatches = [
        f"{name}: checkpoint={observed[name]} current={expected[name]}"
        for name in expected
        if observed[name] != expected[name]
    ]
    if mismatches:
        raise ValueError(
            "Dynamics checkpoint metadata mismatch: " + ", ".join(mismatches)
        )

    model = NeuralDiscreteDynamics(
        num_states=observed["num_states"],
        num_actions=observed["num_actions"],
        hidden_dim=observed["hidden_dim"],
        num_layers=observed["num_layers"],
        dropout=float(payload.get("dropout", 0.0)),
    ).to(device)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    if freeze_after_load:
        freeze_dynamics_model(model)

    stats = dict(payload.get("stats", {}))
    stats["loaded_from_checkpoint"] = True
    stats["checkpoint_path"] = path
    if "dataset_identifiers" in payload:
        stats["dataset_identifiers"] = dict(payload["dataset_identifiers"])
    return model, stats
