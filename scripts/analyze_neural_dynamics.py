from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datasets.cb_dataset import CBSequenceDataset
from models.dynamics_model import NeuralDiscreteDynamics
from planning.dynamics_runtime import build_offline_transition_examples


def _build_cb_dataset(ids: dict[str, Any]) -> CBSequenceDataset:
    return CBSequenceDataset(
        num_episodes=int(ids.get("num_episodes") or 5000),
        max_steps=int(ids.get("max_steps") or 200),
        sequence_length=int(ids.get("max_steps") or 200),
        stochastic=False,
        seed=int(ids.get("seed") or 0),
        policy_mix_spec=str(ids.get("cb_policy_mix_spec") or "random:1.0"),
        policy_mix_sampling=str(ids.get("cb_policy_mix_sampling") or "fixed"),
        state_semantics=str(ids.get("cb_state_semantics") or "post"),
    )


def _build_legacy_post_examples(base_dataset) -> dict[str, np.ndarray]:
    """Reproduce the old post-state alignment: (row[t].state, row[t].action) -> row[t+1].state."""
    state_idx = int(base_dataset.token_schema.field_index("state"))
    action_idx = int(base_dataset.token_schema.field_index("action"))
    num_states = int(base_dataset.env.observation_space.n)
    num_actions = int(base_dataset.env.action_space.n)
    states: list[int] = []
    actions: list[int] = []
    next_states: list[int] = []
    for ep in getattr(base_dataset, "episodes_tokens", []) or []:
        rows = np.asarray(ep)
        if rows.ndim != 2 or rows.shape[0] < 3:
            continue
        core = rows[:-1]
        for t in range(core.shape[0] - 1):
            s = int(core[t, state_idx])
            a = int(core[t, action_idx])
            ns = int(core[t + 1, state_idx])
            if 0 <= s < num_states and 0 <= ns < num_states and 0 <= a < num_actions:
                states.append(s)
                actions.append(a)
                next_states.append(ns)
    return {
        "states": np.asarray(states, dtype=np.int64),
        "actions": np.asarray(actions, dtype=np.int64),
        "next_states": np.asarray(next_states, dtype=np.int64),
    }


def _true_cb_next_state_table(env) -> np.ndarray:
    num_states = int(env.observation_space.n)
    num_actions = int(env.action_space.n)
    table = np.zeros((num_actions, num_states), dtype=np.int64)
    for s in range(num_states):
        r, c = env._state_to_pos(int(s))
        for a in range(num_actions):
            dr, dc = env.ACTIONS.get(int(a), (0, 0))
            nr, nc = r + dr, c + dc
            if 0 <= nr < env.n_rows and 0 <= nc < env.n_cols and env.grid[nr][nc] != "#":
                table[a, s] = int(env._pos_to_state((nr, nc)))
            else:
                table[a, s] = int(s)
    return table


def _load_model(checkpoint_path: Path, device: torch.device) -> tuple[NeuralDiscreteDynamics, dict[str, Any]]:
    payload = torch.load(checkpoint_path, map_location=device)
    model = NeuralDiscreteDynamics(
        num_states=int(payload["num_states"]),
        num_actions=int(payload["num_actions"]),
        hidden_dim=int(payload["hidden_dim"]),
        num_layers=int(payload["num_layers"]),
        dropout=float(payload.get("dropout", 0.0)),
    ).to(device)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval()
    return model, payload


def _score_examples(
    model: NeuralDiscreteDynamics,
    examples: dict[str, np.ndarray],
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    states = torch.from_numpy(examples["states"]).long()
    actions = torch.from_numpy(examples["actions"]).long()
    next_states = torch.from_numpy(examples["next_states"]).long()
    n = int(next_states.numel())
    if n == 0:
        return {"num_examples": 0}
    correct = 0
    loss_sum = 0.0
    prob_sum = 0.0
    for start in range(0, n, int(batch_size)):
        end = min(n, start + int(batch_size))
        s = states[start:end].to(device)
        a = actions[start:end].to(device)
        ns = next_states[start:end].to(device)
        with torch.no_grad():
            logits = model(s, a)
            loss_sum += float(F.cross_entropy(logits, ns, reduction="sum").item())
            probs = torch.softmax(logits, dim=-1)
            pred = logits.argmax(dim=-1)
            correct += int((pred == ns).sum().item())
            prob_sum += float(probs.gather(1, ns.view(-1, 1)).sum().item())
    return {
        "num_examples": n,
        "accuracy": float(correct / max(1, n)),
        "nll": float(loss_sum / max(1, n)),
        "mean_true_next_prob": float(prob_sum / max(1, n)),
    }


def _score_true_table(
    model: NeuralDiscreteDynamics,
    true_next: np.ndarray,
    device: torch.device,
    observed_pairs: set[tuple[int, int]] | None = None,
) -> dict[str, Any]:
    num_actions, num_states = true_next.shape
    states = torch.arange(num_states, device=device).repeat_interleave(num_actions)
    actions = torch.arange(num_actions, device=device).repeat(num_states)
    targets = torch.tensor(
        [int(true_next[a, s]) for s in range(num_states) for a in range(num_actions)],
        dtype=torch.long,
        device=device,
    )
    with torch.no_grad():
        logits = model(states, actions)
        probs = torch.softmax(logits, dim=-1)
        pred = logits.argmax(dim=-1)
        correct = pred == targets
        true_probs = probs.gather(1, targets.view(-1, 1)).squeeze(1)

    out = {
        "num_state_action_pairs": int(num_states * num_actions),
        "top1_accuracy": float(correct.float().mean().item()),
        "mean_true_next_prob": float(true_probs.mean().item()),
        "nll": float((-torch.log(true_probs.clamp_min(1e-12))).mean().item()),
    }
    if observed_pairs:
        mask = torch.tensor(
            [(int(s), int(a)) in observed_pairs for s in range(num_states) for a in range(num_actions)],
            dtype=torch.bool,
            device=device,
        )
        if bool(mask.any().item()):
            out.update(
                {
                    "observed_pair_count": int(mask.sum().item()),
                    "observed_pair_top1_accuracy": float(correct[mask].float().mean().item()),
                    "observed_pair_mean_true_next_prob": float(true_probs[mask].mean().item()),
                    "observed_pair_nll": float((-torch.log(true_probs[mask].clamp_min(1e-12))).mean().item()),
                }
            )
    return out


def _score_examples_against_table(examples: dict[str, np.ndarray], true_next: np.ndarray) -> dict[str, Any]:
    states = examples["states"]
    actions = examples["actions"]
    next_states = examples["next_states"]
    n = int(next_states.shape[0])
    if n == 0:
        return {"num_examples": 0}
    correct = 0
    for s, a, ns in zip(states, actions, next_states):
        correct += int(int(true_next[int(a), int(s)]) == int(ns))
    return {
        "num_examples": n,
        "accuracy": float(correct / max(1, n)),
    }


def _unique_pairs(examples: dict[str, np.ndarray]) -> set[tuple[int, int]]:
    return set(zip(examples["states"].astype(int).tolist(), examples["actions"].astype(int).tolist()))


def _analyze_checkpoint(checkpoint_path: Path, device: torch.device, batch_size: int) -> dict[str, Any]:
    model, payload = _load_model(checkpoint_path, device=device)
    ids = dict(payload.get("dataset_identifiers") or payload.get("stats", {}).get("dataset_identifiers") or {})
    if ids.get("env") not in {None, "cb"}:
        raise ValueError(f"Only ColourBomb checkpoints are supported for now: {checkpoint_path}")

    dataset = _build_cb_dataset(ids)
    correct_examples = build_offline_transition_examples(dataset)
    corrected = {
        "states": correct_examples["states"],
        "actions": correct_examples["actions"],
        "next_states": correct_examples["next_states"],
    }
    legacy = _build_legacy_post_examples(dataset)
    true_next = _true_cb_next_state_table(dataset.env)

    return {
        "checkpoint_path": str(checkpoint_path),
        "dataset_identifiers": ids,
        "checkpoint_stats": payload.get("stats", {}),
        "corrected_dataset_examples": dict(correct_examples["stats"]),
        "score_on_corrected_dataset_examples": _score_examples(
            model, corrected, device=device, batch_size=batch_size
        ),
        "corrected_examples_agreement_with_true_table": _score_examples_against_table(
            corrected, true_next=true_next
        ),
        "score_on_legacy_post_alignment_examples": _score_examples(
            model, legacy, device=device, batch_size=batch_size
        ),
        "legacy_examples_agreement_with_true_table": _score_examples_against_table(
            legacy, true_next=true_next
        ),
        "score_against_true_colourbomb_transition_table": _score_true_table(
            model,
            true_next=true_next,
            device=device,
            observed_pairs=_unique_pairs(corrected),
        ),
    }


def _write_reports(report: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)

    csv_path = output_path.with_suffix(".csv")
    rows = []
    for item in report["checkpoints"]:
        ids = item["dataset_identifiers"]
        corrected = item["score_on_corrected_dataset_examples"]
        legacy = item["score_on_legacy_post_alignment_examples"]
        true_table = item["score_against_true_colourbomb_transition_table"]
        rows.append(
            {
                "checkpoint_path": item["checkpoint_path"],
                "spec_dir": Path(item["checkpoint_path"]).parents[3].name,
                "seed": ids.get("seed"),
                "state_semantics": ids.get("cb_state_semantics"),
                "corrected_examples_true_table_accuracy": item[
                    "corrected_examples_agreement_with_true_table"
                ].get("accuracy"),
                "legacy_examples_true_table_accuracy": item[
                    "legacy_examples_agreement_with_true_table"
                ].get("accuracy"),
                "corrected_dataset_accuracy": corrected.get("accuracy"),
                "corrected_dataset_nll": corrected.get("nll"),
                "legacy_dataset_accuracy": legacy.get("accuracy"),
                "legacy_dataset_nll": legacy.get("nll"),
                "true_table_top1_accuracy": true_table.get("top1_accuracy"),
                "true_table_mean_true_next_prob": true_table.get("mean_true_next_prob"),
                "true_table_nll": true_table.get("nll"),
                "observed_pair_top1_accuracy": true_table.get("observed_pair_top1_accuracy"),
                "observed_pair_mean_true_next_prob": true_table.get("observed_pair_mean_true_next_prob"),
                "observed_pair_nll": true_table.get("observed_pair_nll"),
            }
        )
    if rows:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze saved neural dynamics checkpoints.")
    parser.add_argument("--run_root", type=str, required=True)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--batch_size", type=int, default=4096)
    args = parser.parse_args()

    run_root = Path(args.run_root)
    ckpts = sorted(run_root.glob("spec_*/mix_*/seed_*/train_shared/shared_dynamics_model.pt"))
    if not ckpts:
        raise FileNotFoundError(f"No shared_dynamics_model.pt files found under {run_root}")

    device = torch.device(args.device)
    report = {
        "run_root": str(run_root),
        "num_checkpoints": len(ckpts),
        "checkpoints": [],
    }
    for ckpt in ckpts:
        print(f"[analyze] {ckpt}")
        report["checkpoints"].append(
            _analyze_checkpoint(ckpt, device=device, batch_size=int(args.batch_size))
        )

    output_path = (
        Path(args.output)
        if args.output
        else run_root / "dynamics_analysis" / "neural_dynamics_report.json"
    )
    _write_reports(report, output_path)
    print(f"[done] wrote {output_path}")
    print(f"[done] wrote {output_path.with_suffix('.csv')}")


if __name__ == "__main__":
    main()
