from __future__ import annotations

import json
import os
from dataclasses import dataclass
from itertools import product
from typing import Any

import numpy as np
import torch

from planning.dt_runtime import (
    _advance_state_with_tokens,
    _initial_dfa_state,
    _is_accepting_state,
    _reject_sink_checker,
)


@dataclass(frozen=True)
class ProductValueConfig:
    max_iter: int = 10000
    tol: float = 1e-6
    dmax: float | None = None
    backup: str = "hard"
    gamma: float = 1.0
    zero_support: str = "pessimistic"
    support_penalty: float = 0.0
    soft_tau: float = 1.0


@dataclass
class ProductValueTable:
    phi: np.ndarray
    costs: np.ndarray
    next_q: np.ndarray
    accepting_q: np.ndarray
    reject_q: np.ndarray
    dmax: float
    metadata: dict[str, Any]
    support_counts: np.ndarray | None = None


def default_product_value_cache_path(dataset_artifact_path: str | None, run_dir: str, backend: str, spec: str) -> str:
    if dataset_artifact_path:
        p = os.path.abspath(str(dataset_artifact_path))
        if p.endswith(".meta.json"):
            stem = p[: -len(".meta.json")]
        elif p.endswith(".npz"):
            stem = p[: -len(".npz")]
        else:
            stem = p
        dataset_dir = os.path.dirname(stem)
        return os.path.join(dataset_dir, "analysis", "product_value", str(backend), str(spec), "phi_table.npz")
    return os.path.join(run_dir, "product_value", str(backend), str(spec), "phi_table.npz")


def load_product_value_table(path: str) -> ProductValueTable:
    with np.load(path, allow_pickle=True) as data:
        metadata_raw = data["metadata"].item() if data["metadata"].shape == () else data["metadata"].tolist()
        support_counts = data["support_counts"] if "support_counts" in data.files else None
        return ProductValueTable(
            phi=np.asarray(data["phi"], dtype=np.float32),
            costs=np.asarray(data["costs"], dtype=np.float32),
            next_q=np.asarray(data["next_q"], dtype=np.int64),
            accepting_q=np.asarray(data["accepting_q"], dtype=bool),
            reject_q=np.asarray(data["reject_q"], dtype=bool),
            dmax=float(data["dmax"]),
            metadata=dict(metadata_raw),
            support_counts=None if support_counts is None else np.asarray(support_counts, dtype=np.float32),
        )


def save_product_value_table(path: str, table: ProductValueTable) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "phi": table.phi.astype(np.float32),
        "costs": table.costs.astype(np.float32),
        "next_q": table.next_q.astype(np.int64),
        "accepting_q": table.accepting_q.astype(bool),
        "reject_q": table.reject_q.astype(bool),
        "dmax": np.asarray(float(table.dmax), dtype=np.float32),
        "metadata": np.asarray(dict(table.metadata), dtype=object),
    }
    if table.support_counts is not None:
        payload["support_counts"] = table.support_counts.astype(np.float32)
    np.savez_compressed(path, **payload)
    return path


def _dfa_states(raw_dfa):
    if isinstance(raw_dfa, (list, tuple)):
        component_states = [list(range(len(d.acceptance))) for d in raw_dfa]
        states = list(product(*component_states))
        return states, {tuple(s): i for i, s in enumerate(states)}
    states = list(range(len(raw_dfa.acceptance)))
    return states, {int(s): int(s) for s in states}


def _flatten_dfa_state(state, state_to_id: dict) -> int:
    if isinstance(state, tuple):
        return int(state_to_id[tuple(int(x) for x in state)])
    return int(state_to_id[int(state)])


def build_dfa_prefix_state_ids(dataset, adapter, raw_dfa) -> list[np.ndarray]:
    """
    Return one flattened DFA prefix state per DT decision timestep.

    For CB pre-state artifacts this matches eval-time semantics: the DFA consumes
    the initial state token, then each action/reward/cost/next-state token group.
    Therefore the stored q_t is the DFA state after the current state proposition
    has fired and immediately before the current action token is consumed.
    """
    dfa_states, state_to_id = _dfa_states(raw_dfa)
    _ = dfa_states
    schema = getattr(dataset, "token_schema", None)
    state_idx = int(schema.field_index("state")) if schema is not None else 0
    action_idx = int(schema.field_index("action")) if schema is not None else 1
    reward_idx = int(schema.field_index("reward")) if schema is not None and "reward" in schema.field_names() else 2
    if schema is not None and "safety_cost" in schema.field_names():
        cost_idx = int(schema.field_index("safety_cost"))
    elif schema is not None and "aux" in schema.field_names():
        cost_idx = int(schema.field_index("aux"))
    else:
        cost_idx = 3

    semantics = str(getattr(dataset, "state_semantics", "pre") or "pre")
    out: list[np.ndarray] = []
    for ep in getattr(dataset, "episodes_tokens", []) or []:
        rows = np.asarray(ep, dtype=np.int64)
        if rows.ndim != 2 or rows.shape[0] < 2:
            out.append(np.zeros(0, dtype=np.int64))
            continue
        trans_rows = rows[:-1]
        q = _initial_dfa_state(raw_dfa)

        if semantics == "pre":
            first_state = torch.tensor([int(trans_rows[0, state_idx])], dtype=torch.long)
            q = _advance_state_with_tokens(adapter, raw_dfa, q, first_state, token_offset=0)
            q_ids = np.zeros(trans_rows.shape[0], dtype=np.int64)
            q_ids[0] = _flatten_dfa_state(q, state_to_id)
            for t in range(1, trans_rows.shape[0]):
                prev = trans_rows[t - 1]
                cur = trans_rows[t]
                tokens = torch.tensor(
                    [
                        int(prev[action_idx]),
                        int(prev[reward_idx]) if reward_idx < prev.shape[0] else 0,
                        int(prev[cost_idx]) if cost_idx < prev.shape[0] else 0,
                        int(cur[state_idx]),
                    ],
                    dtype=torch.long,
                )
                q = _advance_state_with_tokens(adapter, raw_dfa, q, tokens, token_offset=1)
                q_ids[t] = _flatten_dfa_state(q, state_to_id)
            out.append(q_ids)
            continue

        # Post-state artifacts do not store the pre-action initial state. Use the
        # same hard-token path row by row so action/state propositions still fire
        # in canonical adapter order, but prefer pre artifacts for paper runs.
        q_ids = np.zeros(trans_rows.shape[0], dtype=np.int64)
        for t, row in enumerate(trans_rows):
            tokens = torch.tensor(
                [
                    int(row[state_idx]),
                    int(row[action_idx]),
                    int(row[reward_idx]) if reward_idx < row.shape[0] else 0,
                    int(row[cost_idx]) if cost_idx < row.shape[0] else 0,
                ],
                dtype=torch.long,
            )
            q = _advance_state_with_tokens(adapter, raw_dfa, q, tokens, token_offset=0)
            q_ids[t] = _flatten_dfa_state(q, state_to_id)
        out.append(q_ids)
    return out


def _build_next_q(adapter, raw_dfa, num_actions: int, num_states: int, dfa_states: list, state_to_id: dict) -> np.ndarray:
    next_q = np.zeros((len(dfa_states), int(num_actions), int(num_states)), dtype=np.int64)
    for q_id, q_state in enumerate(dfa_states):
        for a in range(int(num_actions)):
            for ns in range(int(num_states)):
                tokens = torch.tensor([int(a), 0, 0, int(ns)], dtype=torch.long)
                q_next = _advance_state_with_tokens(adapter, raw_dfa, q_state, tokens, token_offset=1)
                next_q[q_id, a, ns] = _flatten_dfa_state(q_next, state_to_id)
    return next_q


def _softmin(values: np.ndarray, tau: float) -> np.ndarray:
    tau = max(float(tau), 1e-6)
    x = -values / tau
    m = np.max(x, axis=-1, keepdims=True)
    return -tau * (np.log(np.exp(x - m).sum(axis=-1)) + np.squeeze(m, axis=-1))


def build_product_value_table(
    *,
    transition_probs: np.ndarray,
    adapter,
    raw_dfa,
    support_counts: np.ndarray | None = None,
    config: ProductValueConfig | None = None,
    metadata: dict[str, Any] | None = None,
) -> ProductValueTable:
    cfg = config or ProductValueConfig()
    probs = np.asarray(transition_probs, dtype=np.float64)
    if probs.ndim != 3:
        raise ValueError("transition_probs must have shape [num_actions, num_states, num_states].")
    num_actions, num_states, _ = probs.shape
    dfa_states, state_to_id = _dfa_states(raw_dfa)
    num_q = len(dfa_states)
    dmax = float(cfg.dmax if cfg.dmax is not None else (num_states * num_q + 1))
    accepting_q = np.asarray([_is_accepting_state(raw_dfa, q) for q in dfa_states], dtype=bool)
    reject_checker, _ = _reject_sink_checker(raw_dfa)
    reject_q = np.asarray([bool(reject_checker(q)) for q in dfa_states], dtype=bool)
    next_q = _build_next_q(adapter, raw_dfa, num_actions, num_states, dfa_states, state_to_id)

    support = None if support_counts is None else np.asarray(support_counts, dtype=np.float64)
    if support is not None and support.shape != (num_actions, num_states):
        raise ValueError(
            f"support_counts shape {support.shape} does not match {(num_actions, num_states)}."
        )

    costs = np.full((num_q, num_states), dmax, dtype=np.float64)
    costs[accepting_q, :] = 0.0
    fixed = accepting_q | reject_q

    backup = str(cfg.backup)
    if backup not in {"hard", "soft"}:
        raise ValueError("--product_value_backup must be 'hard' or 'soft'.")
    zero_support = str(cfg.zero_support)
    if zero_support not in {"pessimistic", "self_loop"}:
        raise ValueError("--product_value_zero_support must be 'pessimistic' or 'self_loop'.")

    converged = False
    for iteration in range(int(cfg.max_iter)):
        old = costs.copy()
        action_costs = np.full((num_q, num_states, num_actions), dmax, dtype=np.float64)
        for a in range(num_actions):
            p_a = probs[a]
            for q_id in range(num_q):
                q_next = next_q[q_id, a]
                next_cost_by_ns = old[q_next, np.arange(num_states)]
                expected = p_a @ next_cost_by_ns
                action_costs[q_id, :, a] = np.minimum(dmax, 1.0 + expected)

        if support is not None and zero_support == "pessimistic":
            unsupported = support.T <= 0.0  # [S, A]
            action_costs[:, unsupported] = dmax

        if backup == "hard":
            updated = np.min(action_costs, axis=-1)
        else:
            updated = _softmin(action_costs, tau=float(cfg.soft_tau))
            updated = np.clip(updated, 0.0, dmax)

        updated[accepting_q, :] = 0.0
        updated[reject_q, :] = dmax
        costs[~fixed, :] = updated[~fixed, :]
        delta = float(np.max(np.abs(costs - old)))
        if delta <= float(cfg.tol):
            converged = True
            break

    phi = -np.clip(costs, 0.0, dmax).astype(np.float32)
    meta = dict(metadata or {})
    meta.update(
        {
            "num_states": int(num_states),
            "num_actions": int(num_actions),
            "num_dfa_states": int(num_q),
            "dmax": float(dmax),
            "backup": backup,
            "gamma": float(cfg.gamma),
            "zero_support": zero_support,
            "support_penalty": float(cfg.support_penalty),
            "max_iter": int(cfg.max_iter),
            "tol": float(cfg.tol),
            "iterations": int(iteration + 1),
            "converged": bool(converged),
            "unreachable_cells": int(np.count_nonzero(costs >= dmax)),
            "accepting_dfa_states": int(np.count_nonzero(accepting_q)),
            "reject_dfa_states": int(np.count_nonzero(reject_q)),
        }
    )
    return ProductValueTable(
        phi=phi,
        costs=costs.astype(np.float32),
        next_q=next_q,
        accepting_q=accepting_q,
        reject_q=reject_q,
        dmax=dmax,
        metadata=meta,
        support_counts=None if support is None else support.astype(np.float32),
    )


def _expected_advantages(
    *,
    table: ProductValueTable,
    transition_probs: np.ndarray,
    q_id: int,
    state_id: int,
) -> np.ndarray:
    probs = np.asarray(transition_probs, dtype=np.float64)
    num_actions = int(probs.shape[0])
    state_range = np.arange(int(probs.shape[-1]))
    current = float(table.phi[int(q_id), int(state_id)])
    out = np.zeros(num_actions, dtype=np.float64)
    for a in range(num_actions):
        q_next = table.next_q[int(q_id), int(a), :]
        phi_next = table.phi[q_next, state_range]
        out[a] = float(np.sum(probs[int(a), int(state_id), :] * phi_next) - current)
    return out


def validate_product_value_table(
    table: ProductValueTable,
    *,
    transition_probs: np.ndarray | None = None,
    start_state: int | None = None,
    up_action_id: int = 0,
    strict: bool = True,
) -> dict[str, Any]:
    phi = np.asarray(table.phi, dtype=np.float32)
    dmax = float(table.dmax)
    tol = float(table.metadata.get("tol", 1e-6))
    failures: list[str] = []
    checks: dict[str, Any] = {}

    accepting_count = int(np.count_nonzero(table.accepting_q))
    accepting_max_abs = None
    if accepting_count > 0:
        accepting_vals = phi[table.accepting_q, :]
        accepting_max_abs = float(np.max(np.abs(accepting_vals))) if accepting_vals.size else 0.0
        if accepting_max_abs > max(1e-5, 10.0 * tol):
            failures.append("accepting_states_not_zero")
    else:
        failures.append("no_accepting_dfa_state")
    checks["accepting_states_highest"] = {
        "accepting_dfa_states": accepting_count,
        "accepting_max_abs_phi": accepting_max_abs,
        "passed": accepting_count > 0 and (accepting_max_abs is not None and accepting_max_abs <= max(1e-5, 10.0 * tol)),
    }

    reject_count = int(np.count_nonzero(table.reject_q))
    reject_min = None
    reject_max = None
    reject_passed = True
    if reject_count > 0:
        reject_vals = phi[table.reject_q, :]
        reject_min = float(np.min(reject_vals)) if reject_vals.size else None
        reject_max = float(np.max(reject_vals)) if reject_vals.size else None
        reject_passed = bool(reject_max is not None and reject_max <= -dmax + max(1e-5, 10.0 * tol))
        if not reject_passed:
            failures.append("reject_states_not_distinctly_negative")
    checks["reject_distinctly_negative"] = {
        "reject_dfa_states": reject_count,
        "reject_phi_min": reject_min,
        "reject_phi_max": reject_max,
        "target": -dmax,
        "passed": reject_passed,
    }

    if transition_probs is not None:
        probs = np.asarray(transition_probs, dtype=np.float64)
        if probs.shape != (int(table.metadata["num_actions"]), int(table.metadata["num_states"]), int(table.metadata["num_states"])):
            failures.append("transition_probs_shape_mismatch")
        else:
            reachable = (table.costs > max(1e-5, 10.0 * tol)) & (table.costs < dmax - max(1e-5, 10.0 * tol))
            reachable[table.accepting_q, :] = False
            reachable[table.reject_q, :] = False
            min_best_adv = None
            checked = 0
            bad = 0
            for q_id, s_id in zip(*np.where(reachable)):
                adv = _expected_advantages(
                    table=table,
                    transition_probs=probs,
                    q_id=int(q_id),
                    state_id=int(s_id),
                )
                best = float(np.max(adv))
                min_best_adv = best if min_best_adv is None else min(min_best_adv, best)
                checked += 1
                if best < -max(1e-5, 10.0 * tol):
                    bad += 1
            if bad > 0:
                failures.append("reachable_states_without_nonnegative_progress")
            checks["monotonic_reachable_progress"] = {
                "checked_state_count": int(checked),
                "bad_state_count": int(bad),
                "min_best_advantage": min_best_adv,
                "passed": bad == 0,
            }

            if start_state is not None and 0 <= int(start_state) < probs.shape[1]:
                q0 = 0
                s0 = int(start_state)
                start_adv = _expected_advantages(
                    table=table,
                    transition_probs=probs,
                    q_id=q0,
                    state_id=s0,
                )
                start_phi = float(phi[q0, s0])
                start_requires_progress = bool(
                    not table.accepting_q[q0]
                    and not table.reject_q[q0]
                    and table.costs[q0, s0] > max(1e-5, 10.0 * tol)
                    and table.costs[q0, s0] < dmax - max(1e-5, 10.0 * tol)
                )
                start_best = float(np.max(start_adv))
                start_passed = (not start_requires_progress) or start_best > max(1e-5, 10.0 * tol)
                if not start_passed:
                    failures.append("no_improving_action_from_start")
                checks["improving_action_from_start"] = {
                    "start_state": s0,
                    "start_phi": start_phi,
                    "start_requires_progress": start_requires_progress,
                    "advantages": [float(x) for x in start_adv.tolist()],
                    "best_advantage": start_best,
                    "passed": bool(start_passed),
                }

    if str(table.metadata.get("spec")) == "avoid_upward_action":
        if not (0 <= int(up_action_id) < table.next_q.shape[1]):
            failures.append("up_action_id_out_of_range")
            up_passed = False
            bad_count = None
        else:
            non_reject_q = np.where(~table.reject_q)[0]
            q_next = table.next_q[non_reject_q, int(up_action_id), :]
            bad_count = int(np.count_nonzero(~table.reject_q[q_next]))
            up_passed = bad_count == 0
            if not up_passed:
                failures.append("up_action_does_not_enter_reject")
        checks["up_action_rejects"] = {
            "up_action_id": int(up_action_id),
            "bad_transition_count": bad_count,
            "passed": bool(up_passed),
        }

    report = {
        "passed": len(failures) == 0,
        "failures": failures,
        "checks": checks,
        "metadata": dict(table.metadata),
    }
    if strict and failures:
        raise ValueError(
            "Product-value validation failed: "
            + ", ".join(failures)
        )
    return report


def write_product_value_validation(
    table: ProductValueTable,
    out_dir: str,
    *,
    transition_probs: np.ndarray | None = None,
    start_state: int | None = None,
    grid_shape: tuple[int, int] | None = None,
    strict: bool = True,
) -> dict[str, Any]:
    os.makedirs(out_dir, exist_ok=True)
    report = validate_product_value_table(
        table,
        transition_probs=transition_probs,
        start_state=start_state,
        strict=strict,
    )
    with open(os.path.join(out_dir, "monotonicity_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, sort_keys=True)
    with open(os.path.join(out_dir, "coverage_summary.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "has_support_counts": table.support_counts is not None,
                "observed_state_action_pairs": None
                if table.support_counts is None
                else int(np.count_nonzero(table.support_counts > 0)),
                "possible_state_action_pairs": None
                if table.support_counts is None
                else int(table.support_counts.size),
            },
            f,
            indent=2,
            sort_keys=True,
        )
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    phi0 = table.phi[0]
    if grid_shape is not None and int(np.prod(grid_shape)) == int(phi0.shape[0]):
        grid = phi0.reshape(grid_shape)
        plt.figure(figsize=(7, 6))
        plt.imshow(grid, cmap="viridis", interpolation="nearest")
        plt.xlabel("Column")
        plt.ylabel("Row")
        plt.title("Product potential Phi(q0, state)")
        plt.colorbar(label="Phi")
        for r in range(grid_shape[0]):
            for c in range(grid_shape[1]):
                sid = r * grid_shape[1] + c
                plt.text(c, r, str(sid), ha="center", va="center", fontsize=7, color="white")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "phi_heatmap.png"))
        plt.close()
    else:
        plt.figure(figsize=(7, 4))
        plt.plot(np.arange(phi0.shape[0]), phi0)
        plt.xlabel("State ID")
        plt.ylabel("Phi(q0, state)")
        plt.title("Product potential from initial DFA state")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "phi_heatmap.png"))
        plt.close()

    if grid_shape is not None and int(np.prod(grid_shape)) == int(phi0.shape[0]):
        cols = min(4, int(table.phi.shape[0]))
        rows = int(np.ceil(table.phi.shape[0] / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.5 * rows), squeeze=False)
        vmin = float(np.min(table.phi))
        vmax = float(np.max(table.phi))
        for q_id in range(rows * cols):
            ax = axes[q_id // cols][q_id % cols]
            if q_id >= table.phi.shape[0]:
                ax.axis("off")
                continue
            im = ax.imshow(table.phi[q_id].reshape(grid_shape), cmap="viridis", vmin=vmin, vmax=vmax)
            ax.set_title(f"q={q_id}")
            ax.set_xticks([])
            ax.set_yticks([])
        fig.colorbar(im, ax=axes.ravel().tolist(), label="Phi", shrink=0.8)
        fig.savefig(os.path.join(out_dir, "phi_by_dfa_state.png"), bbox_inches="tight")
        plt.close(fig)

    # Store the best immediate progress per state for quick visual inspection.
    if table.next_q.ndim == 3:
        num_actions = table.next_q.shape[1]
        best = np.zeros((phi0.shape[0], num_actions), dtype=np.float32)
        if transition_probs is not None:
            for s in range(phi0.shape[0]):
                best[s] = _expected_advantages(
                    table=table,
                    transition_probs=np.asarray(transition_probs),
                    q_id=0,
                    state_id=s,
                )
        else:
            for a in range(num_actions):
                best[:, a] = table.phi[table.next_q[0, a], np.arange(phi0.shape[0])] - phi0
        if grid_shape is not None and int(np.prod(grid_shape)) == int(phi0.shape[0]):
            best_adv = np.max(best, axis=1).reshape(grid_shape)
            best_action = np.argmax(best, axis=1).reshape(grid_shape)
            plt.figure(figsize=(7, 6))
            plt.imshow(best_adv, cmap="coolwarm", interpolation="nearest")
            plt.xlabel("Column")
            plt.ylabel("Row")
            plt.title("Best immediate product-value advantage from q0")
            plt.colorbar(label="Best advantage")
            for r in range(grid_shape[0]):
                for c in range(grid_shape[1]):
                    plt.text(
                        c,
                        r,
                        f"{r * grid_shape[1] + c}\na{int(best_action[r, c])}",
                        ha="center",
                        va="center",
                        fontsize=7,
                        color="black",
                    )
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, "advantage_map.png"))
            plt.close()
        else:
            plt.figure(figsize=(7, 4))
            plt.imshow(best.T, aspect="auto", interpolation="nearest")
            plt.xlabel("State ID")
            plt.ylabel("Action")
            plt.title("Immediate product-value advantage from q0")
            plt.colorbar(label="Phi(next)-Phi(current)")
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, "advantage_map.png"))
            plt.close()
    return report


def write_offline_vs_oracle_comparison(
    *,
    offline_table: ProductValueTable,
    oracle_table: ProductValueTable,
    out_dir: str,
) -> str:
    os.makedirs(out_dir, exist_ok=True)
    delta = np.asarray(offline_table.phi, dtype=np.float32) - np.asarray(oracle_table.phi, dtype=np.float32)
    payload = {
        "mean_abs_phi_delta": float(np.mean(np.abs(delta))),
        "max_abs_phi_delta": float(np.max(np.abs(delta))) if delta.size else 0.0,
        "mean_phi_delta": float(np.mean(delta)) if delta.size else 0.0,
        "offline_backend": offline_table.metadata.get("backend"),
        "oracle_backend": oracle_table.metadata.get("backend"),
        "num_entries": int(delta.size),
    }
    path = os.path.join(out_dir, "offline_vs_oracle.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    return path


def compute_dt_dfa_product_value_loss(
    *,
    logits: torch.Tensor,
    states: torch.Tensor,
    dfa_state_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    transition_probs: torch.Tensor,
    phi: torch.Tensor,
    next_q: torch.Tensor,
    support_counts: torch.Tensor | None = None,
    support_penalty: float = 0.0,
    gamma: float = 1.0,
    temperature: float = 1.0,
) -> torch.Tensor:
    B, T, A = logits.shape
    S = int(transition_probs.shape[-1])
    valid = attention_mask.float().clamp(min=0.0, max=1.0).reshape(-1)
    valid_count = valid.sum().clamp(min=1.0)

    temp = max(float(temperature), 1e-6)
    action_probs = torch.softmax(logits / temp, dim=-1).clamp(min=1e-8, max=1.0)
    action_probs = action_probs / action_probs.sum(dim=-1, keepdim=True)
    action_probs = action_probs.reshape(-1, A)

    state_ids = states.clamp(min=0, max=S - 1).reshape(-1).long()
    q_ids = dfa_state_ids.clamp(min=0, max=int(phi.shape[0]) - 1).reshape(-1).long()

    current_phi = phi[q_ids, state_ids]
    advantages = logits.new_zeros((state_ids.shape[0], A))
    state_range = torch.arange(S, device=logits.device)
    for a in range(A):
        q_next = next_q[q_ids, a, :]  # [N, S]
        phi_next = phi[q_next, state_range.view(1, -1).expand(q_next.shape[0], -1)]
        probs = transition_probs[a, state_ids, :]
        expected_next = (probs * phi_next).sum(dim=-1)
        advantages[:, a] = float(gamma) * expected_next - current_phi

    if support_counts is not None and float(support_penalty) > 0.0:
        counts = support_counts.t()[state_ids, :].clamp(min=0.0)
        advantages = advantages - float(support_penalty) / torch.sqrt(counts + 1.0)

    per_item = -(action_probs * advantages).sum(dim=-1)
    return (per_item * valid).sum() / valid_count
