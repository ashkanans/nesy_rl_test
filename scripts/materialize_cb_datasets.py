from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datasets.artifact_io import save_sequence_dataset_artifact
from datasets.cb_dataset import CBSequenceDataset
from planning.dynamics_runtime import (
    build_dynamics_dataset_identifiers,
    fit_neural_dynamics_model,
    save_neural_dynamics_checkpoint,
)
from specs.cb_specs import SPECS as CB_SPECS
from train_cb import build_adapter_and_dfa, resolve_formulas

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - fallback when tqdm is unavailable

    class _TqdmFallback:
        def __init__(self, iterable=None, total=None, **kwargs):
            self._iterable = iterable if iterable is not None else range(int(total or 0))

        def __iter__(self):
            return iter(self._iterable)

        def set_postfix_str(self, *args, **kwargs):
            return None

        def update(self, *args, **kwargs):
            return None

        def close(self):
            return None

    def tqdm(iterable=None, *args, **kwargs):
        kw = dict(kwargs)
        total = kw.pop("total", None)
        return _TqdmFallback(iterable=iterable, total=total, **kw)


DEFAULT_POLICY_NAMES = [
    "random",
    "shortest_safe",
    "longest_safe",
    "shortest_any",
    "longest_any",
]

DEFAULT_POLICY_MIX_SPECS = [
    "random:1.0",
    "shortest_safe:1.0",
    "longest_safe:1.0",
    "shortest_any:1.0",
    "longest_any:1.0",
    "random:0.7,shortest_safe:0.3",
    "random:0.6,shortest_safe:0.4",
]


def _slug(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", str(text)).strip("_")


def _write_json(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def _fmt_weight(weight: float) -> str:
    return f"{float(weight):.1f}"


def _compositions(total: int, parts: int):
    if parts <= 1:
        yield (total,)
        return
    for first in range(total, -1, -1):
        for rest in _compositions(total - first, parts - 1):
            yield (first,) + rest


def _generate_full_simplex_mix_specs(policy_names: list[str], grid_step: float) -> list[str]:
    denom = int(round(1.0 / float(grid_step)))
    if denom <= 0:
        raise ValueError("--mix_grid_step must be positive.")
    if not math.isclose(denom * float(grid_step), 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("--mix_grid_step must evenly divide 1.0 (e.g. 0.1 or 0.25).")

    specs: list[str] = []
    for counts in _compositions(denom, len(policy_names)):
        active = sum(1 for c in counts if c > 0)
        parts = [
            f"{name}:{_fmt_weight(count / denom)}"
            for name, count in zip(policy_names, counts)
            if count > 0
        ]
        specs.append((",".join(parts), active, tuple(counts)))

    specs.sort(key=lambda item: (item[1], tuple(-v for v in item[2]), item[0]))
    return [spec for spec, _, _ in specs]


def _artifact_stem(base_name: str, semantics: str, mix_spec: str, seed: int) -> str:
    mix_slug = _slug(mix_spec) or "mix"
    return _slug(f"{base_name}_{semantics}_{mix_slug}_seed{int(seed)}")


def _artifact_dir(base_dir: str, stem: str) -> str:
    return os.path.join(base_dir, stem)


def _artifact_paths(base_dir: str, stem: str) -> tuple[str, str]:
    artifact_dir = _artifact_dir(base_dir, stem)
    npz_path = os.path.join(artifact_dir, f"{stem}.npz")
    meta_path = os.path.join(artifact_dir, f"{stem}.meta.json")
    return npz_path, meta_path


def _analysis_root(base_dir: str, stem: str) -> str:
    return os.path.join(_artifact_dir(base_dir, stem), "analysis")


def _analysis_summary_path(base_dir: str, stem: str) -> str:
    return os.path.join(_analysis_root(base_dir, stem), "dataset_analysis", "summary.json")


def _dynamics_checkpoint_path(base_dir: str, stem: str) -> str:
    return os.path.join(_artifact_dir(base_dir, stem), "dynamics", "neural_dataset.pt")


def parse_args(argv: list[str] | None = None):
    p = argparse.ArgumentParser(
        description="Materialize reusable ColourBomb dataset artifacts and analysis reports."
    )
    p.add_argument(
        "--dataset_artifact_dir",
        type=str,
        default=str(REPO_ROOT / "artifacts" / "datasets" / "colorbomb"),
        help="Directory where dataset artifacts will be written.",
    )
    p.add_argument(
        "--dataset_artifact_name",
        type=str,
        default="dataset_snapshot",
        help="Base artifact name used in generated filenames.",
    )
    p.add_argument(
        "--full_simplex",
        action="store_true",
        help="Generate the full grid simplex instead of the explicit mix list.",
    )
    p.add_argument(
        "--policy_names",
        nargs="+",
        default=DEFAULT_POLICY_NAMES,
        help="Policy sources used when --full_simplex is enabled.",
    )
    p.add_argument(
        "--mix_grid_step",
        type=float,
        default=0.1,
        help="Grid step used for full simplex generation, e.g. 0.1 or 0.25.",
    )
    p.add_argument(
        "--policy_mix_specs",
        nargs="+",
        default=DEFAULT_POLICY_MIX_SPECS,
        help="Explicit CB policy mix specifications to materialize when not using --full_simplex.",
    )
    p.add_argument(
        "--state_semantics",
        nargs="+",
        default=["pre", "post"],
        choices=["pre", "post", "both"],
        help="CB state semantics to materialize.",
    )
    p.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[0],
        help="Dataset seeds to materialize.",
    )
    p.add_argument("--num_episodes", type=int, default=5000)
    p.add_argument("--max_steps", type=int, default=200)
    p.add_argument(
        "--sequence_length",
        type=int,
        default=16,
        help=(
            "Artifact-side sequence length used when saving the snapshot. "
            "The downstream loader reindexes raw episodes at the caller's sequence_length."
        ),
    )
    p.add_argument("--stochastic", action="store_true")
    p.add_argument(
        "--policy_mix_sampling",
        type=str,
        choices=["fixed", "normal"],
        default="fixed",
    )
    p.add_argument("--policy_mix_normal_spec", type=str, default=None)
    p.add_argument(
        "--policy_mix_normal_mean_mode",
        type=str,
        choices=["base", "absolute", "delta"],
        default="base",
    )
    p.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip artifacts that already have both .npz and .meta.json files.",
    )
    p.add_argument(
        "--analyze_all_specs",
        action="store_true",
        help="Write a full CB spec-comparison report for every generated dataset artifact.",
    )
    p.add_argument(
        "--analysis_segment_limit",
        type=int,
        default=1000,
        help=(
            "Maximum number of dataset segments to score per spec in the report. "
            "Use -1 to score all segments exactly."
        ),
    )
    p.add_argument(
        "--analysis_episode_limit",
        type=int,
        default=None,
        help="Maximum number of episodes to score per spec in the report. Default: all.",
    )
    p.add_argument(
        "--fit_neural_dynamics",
        action="store_true",
        help="Train one neural dynamics model per materialized dataset artifact.",
    )
    p.add_argument("--overwrite_dynamics", action="store_true")
    p.add_argument("--dynamics_device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--dynamics_epochs", type=int, default=20)
    p.add_argument("--dynamics_batch_size", type=int, default=256)
    p.add_argument("--dynamics_lr", type=float, default=1e-3)
    p.add_argument("--dynamics_hidden_dim", type=int, default=128)
    p.add_argument("--dynamics_layers", type=int, default=2)
    p.add_argument("--dynamics_weight_decay", type=float, default=1e-4)
    p.add_argument("--dynamics_val_fraction", type=float, default=0.1)
    p.add_argument("--dynamics_temperature", type=float, default=1.0)
    return p.parse_args(argv)


def _dataset_spec_list(args) -> list[str]:
    if bool(args.full_simplex):
        return _generate_full_simplex_mix_specs(list(args.policy_names), float(args.mix_grid_step))
    return list(args.policy_mix_specs)


def _resolve_state_semantics(state_semantics: list[str]) -> list[str]:
    resolved: list[str] = []
    for semantics in state_semantics:
        if semantics == "both":
            resolved.extend(["pre", "post"])
        else:
            resolved.append(str(semantics))
    deduped: list[str] = []
    for semantics in resolved:
        if semantics not in deduped:
            deduped.append(semantics)
    return deduped


def _make_dataset(args, mix_spec: str, semantics: str, seed: int) -> CBSequenceDataset:
    return CBSequenceDataset(
        num_episodes=int(args.num_episodes),
        max_steps=int(args.max_steps),
        sequence_length=int(args.sequence_length),
        stochastic=bool(args.stochastic),
        seed=int(seed),
        policy_mix_spec=str(mix_spec),
        policy_mix_sampling=str(args.policy_mix_sampling),
        policy_mix_normal_spec=args.policy_mix_normal_spec,
        policy_mix_normal_mean_mode=str(args.policy_mix_normal_mean_mode),
        state_semantics=str(semantics),
    )


def _save_dataset_artifact(base_dir: str, stem: str, args, dataset, mix_spec: str, semantics: str, seed: int):
    artifact_dir = _artifact_dir(base_dir, stem)
    ds_args = argparse.Namespace(
        env="cb",
        seed=int(seed),
        spec=None,
        save_generated_dataset=True,
        dataset_artifact_dir=artifact_dir,
        dataset_artifact_name=stem,
        num_episodes=int(args.num_episodes),
        max_steps=int(args.max_steps),
        sequence_length=int(args.sequence_length),
        stochastic=bool(args.stochastic),
        cb_policy_mix_spec=str(mix_spec),
        cb_policy_mix_sampling=str(args.policy_mix_sampling),
        cb_policy_mix_normal_spec=args.policy_mix_normal_spec,
        cb_policy_mix_normal_mean_mode=str(args.policy_mix_normal_mean_mode),
        cb_state_semantics=str(semantics),
    )
    info = save_sequence_dataset_artifact(ds_args, dataset, artifact_tag="dataset_snapshot")
    if info is None:
        raise RuntimeError("Dataset artifact saving was disabled unexpectedly.")
    return info


def _dataset_overview(dataset) -> dict:
    episodes = list(getattr(dataset, "episodes_tokens", []) or [])
    rewards = getattr(dataset, "episode_rewards", None)
    policy_labels = list(getattr(dataset, "episode_policy_labels", []) or [])
    lengths = [int(np.asarray(ep).shape[0] - 1) for ep in episodes]
    returns = [float(np.sum(np.asarray(r, dtype=np.float32))) for r in rewards] if rewards else []
    outcome_counts = {"goal": 0, "bomb_hit": 0, "timeout": 0, "other": 0}
    if rewards:
        env_cfg = getattr(getattr(dataset, "env", None), "cfg", None)
        max_steps_cfg = int(getattr(env_cfg, "max_steps", 200))
        step_r = float(getattr(env_cfg, "step_reward", -0.01))
        goal_r = float(getattr(env_cfg, "goal_reward", 1.0))
        bomb_r = float(getattr(env_cfg, "bomb_reward", -1.0))
        goal_thresh = 0.5 * (step_r + goal_r)
        bomb_thresh = 0.5 * (step_r + bomb_r)
        for rew in rewards:
            rew = np.asarray(rew, dtype=np.float32)
            if rew.size == 0:
                outcome_counts["other"] += 1
                continue
            last = float(rew[-1])
            if int(rew.shape[0]) >= max_steps_cfg:
                outcome_counts["timeout"] += 1
            elif last >= goal_thresh:
                outcome_counts["goal"] += 1
            elif last <= bomb_thresh:
                outcome_counts["bomb_hit"] += 1
            else:
                outcome_counts["other"] += 1

    return {
        "env": str(getattr(dataset, "env_name", "cb")),
        "num_episodes": int(len(episodes)),
        "num_segments": int(len(dataset)),
        "episode_length_min": int(np.min(lengths)) if lengths else 0,
        "episode_length_max": int(np.max(lengths)) if lengths else 0,
        "episode_length_mean": float(np.mean(lengths)) if lengths else 0.0,
        "episode_return_min": float(np.min(returns)) if returns else None,
        "episode_return_max": float(np.max(returns)) if returns else None,
        "episode_return_mean": float(np.mean(returns)) if returns else None,
        "episode_policy_counts": dict(Counter(policy_labels)),
        "episode_outcomes": {
            "counts": outcome_counts,
            "rates": {
                f"{k}_rate": (v / float(sum(outcome_counts.values())) if outcome_counts else 0.0)
                for k, v in outcome_counts.items()
            },
        }
        if rewards
        else None,
    }


def _write_base_plots(dataset, out_dir: str, summary: dict) -> None:
    try:
        import matplotlib.pyplot as plt
        from matplotlib import cm
        from matplotlib.colors import Normalize
    except Exception:
        return

    episodes = list(getattr(dataset, "episodes_tokens", []) or [])
    rewards = getattr(dataset, "episode_rewards", None)
    policy_labels = list(getattr(dataset, "episode_policy_labels", []) or [])

    if episodes:
        lengths = [int(np.asarray(ep).shape[0] - 1) for ep in episodes]
        plt.figure(figsize=(6, 4))
        plt.hist(lengths, bins=20)
        plt.xlabel("Episode length (transitions)")
        plt.ylabel("Count")
        plt.title("CB episode length distribution")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "episode_length_hist.png"))
        plt.close()

        state_ids = []
        for ep in episodes:
            state_ids.extend(np.asarray(ep[:-1, 0]).tolist())
        plt.figure(figsize=(6, 4))
        plt.hist(state_ids, bins=int(dataset.env.observation_space.n))
        plt.xlabel("State ID")
        plt.ylabel("Count")
        plt.title("CB state visitation histogram")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "state_hist.png"))
        plt.close()

        state_counts = np.zeros(int(dataset.env.observation_space.n), dtype=np.int64)
        for sid in state_ids:
            sid_int = int(sid)
            if 0 <= sid_int < state_counts.shape[0]:
                state_counts[sid_int] += 1

        state_csv_path = os.path.join(out_dir, "state_visitation.csv")
        with open(state_csv_path, "w", encoding="utf-8", newline="") as f:
            f.write("state_id,visit_count\n")
            for sid, count in enumerate(state_counts.tolist()):
                f.write(f"{int(sid)},{int(count)}\n")

        env = dataset.env
        n_rows = int(getattr(env, "n_rows", 0) or 0)
        n_cols = int(getattr(env, "n_cols", 0) or 0)
        if n_rows > 0 and n_cols > 0:
            grid_counts = np.zeros((n_rows, n_cols), dtype=np.int64)
            for sid, count in enumerate(state_counts.tolist()):
                r, c = divmod(int(sid), n_cols)
                if 0 <= r < n_rows and 0 <= c < n_cols:
                    grid_counts[r, c] = int(count)

            vmax = int(grid_counts.max()) if grid_counts.size else 0
            norm = Normalize(vmin=0, vmax=max(1, vmax))
            cmap = plt.get_cmap("Reds")

            plt.figure(figsize=(max(6, n_cols * 0.85), max(6, n_rows * 0.85)))
            ax = plt.gca()
            ax.set_xlim(0, n_cols)
            ax.set_ylim(n_rows, 0)
            ax.set_aspect("equal")
            ax.set_xticks(range(n_cols + 1))
            ax.set_yticks(range(n_rows + 1))
            ax.grid(color="#cbd5e1", linewidth=0.6)

            grid = getattr(env, "grid", None)
            for r in range(n_rows):
                for c in range(n_cols):
                    count = int(grid_counts[r, c])
                    color = cmap(norm(count))
                    ax.add_patch(
                        plt.Rectangle((c, r), 1, 1, facecolor=color, edgecolor="#94a3b8", linewidth=0.8)
                    )
                    cell_label = ""
                    if grid is not None:
                        cell_label = str(grid[r][c])
                    state_id = r * n_cols + c
                    text_color = "#111827" if count < max(1, vmax * 0.35) else "#ffffff"
                    label = f"{state_id}\n{count}"
                    ax.text(
                        c + 0.5,
                        r + 0.52,
                        label,
                        ha="center",
                        va="center",
                        fontsize=8,
                        color=text_color,
                        fontweight="bold",
                    )
                    if cell_label and cell_label not in {".", "#"}:
                        ax.text(
                            c + 0.5,
                            r + 0.18,
                            cell_label,
                            ha="center",
                            va="center",
                            fontsize=7,
                            color=text_color,
                        )

            title = "CB state visitation heatmap"
            ax.set_title(title)
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            cbar = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label("Visit count")
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, "state_visitation_heatmap.png"))
            plt.close()

    if rewards:
        returns = [float(np.sum(np.asarray(r, dtype=np.float32))) for r in rewards]
        plt.figure(figsize=(6, 4))
        plt.hist(returns, bins=20)
        plt.xlabel("Episode return")
        plt.ylabel("Count")
        plt.title("CB episode return distribution")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "episode_return_hist.png"))
        plt.close()

    if policy_labels:
        counts = Counter(policy_labels)
        summary["episode_policy_counts"] = dict(counts)
        csv_path = os.path.join(out_dir, "policy_mix_counts.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            f.write("policy_name,count\n")
            for name, count in sorted(counts.items()):
                f.write(f"{name},{int(count)}\n")

        plt.figure(figsize=(8, 4))
        plt.bar(list(counts.keys()), list(counts.values()))
        plt.xticks(rotation=20, ha="right")
        plt.xlabel("Policy source")
        plt.ylabel("Episode count")
        plt.title("CB dataset composition by policy source")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "policy_mix_counts.png"))
        plt.close()

    if rewards:
        env_cfg = getattr(getattr(dataset, "env", None), "cfg", None)
        max_steps_cfg = int(getattr(env_cfg, "max_steps", 200))
        step_r = float(getattr(env_cfg, "step_reward", -0.01))
        goal_r = float(getattr(env_cfg, "goal_reward", 1.0))
        bomb_r = float(getattr(env_cfg, "bomb_reward", -1.0))
        goal_thresh = 0.5 * (step_r + goal_r)
        bomb_thresh = 0.5 * (step_r + bomb_r)
        outcome_counts = {"goal": 0, "bomb_hit": 0, "timeout": 0, "other": 0}
        for rew in rewards:
            rew = np.asarray(rew, dtype=np.float32)
            if rew.size == 0:
                outcome_counts["other"] += 1
                continue
            last = float(rew[-1])
            if int(rew.shape[0]) >= max_steps_cfg:
                outcome_counts["timeout"] += 1
            elif last >= goal_thresh:
                outcome_counts["goal"] += 1
            elif last <= bomb_thresh:
                outcome_counts["bomb_hit"] += 1
            else:
                outcome_counts["other"] += 1

        summary["episode_outcomes"] = {
            "counts": outcome_counts,
            "rates": {
                f"{k}_rate": (v / float(sum(outcome_counts.values())) if outcome_counts else 0.0)
                for k, v in outcome_counts.items()
            },
        }

        labels = ["goal", "bomb_hit", "timeout", "other"]
        values = [outcome_counts[k] for k in labels]
        plt.figure(figsize=(7, 4))
        plt.bar(labels, values)
        plt.ylabel("Episode count")
        plt.title("CB dataset episode outcomes")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "episode_outcomes_bar.png"))
        plt.close()


def _cb_deterministic_next_state(env, state: int, action: int) -> int:
    r, c = env._state_to_pos(int(state))
    dr, dc = env.ACTIONS[int(action)]
    nr, nc = r + dr, c + dc
    if 0 <= nr < int(env.n_rows) and 0 <= nc < int(env.n_cols):
        if env.grid[nr][nc] != "#":
            r, c = nr, nc
    return int(env._pos_to_state((r, c)))


def _spec_scoring_episode_tokens(dataset) -> tuple[list[np.ndarray], str, str | None]:
    """Return episode tokens in the state semantics needed for CB spec scoring."""
    episodes = [np.asarray(ep, dtype=np.int64) for ep in (getattr(dataset, "episodes_tokens", []) or [])]
    state_semantics = str(getattr(dataset, "state_semantics", "post") or "post")
    if state_semantics != "pre":
        return [ep.copy() for ep in episodes], state_semantics, None

    env = getattr(dataset, "env", None)
    if env is None:
        raise ValueError("Cannot reconstruct post-state traces for pre-semantics CB data without env.")
    env_cfg = getattr(env, "cfg", None)
    if bool(getattr(env_cfg, "stochastic", False)):
        raise ValueError(
            "Cannot exactly reconstruct post-state traces from pre-semantics stochastic CB data. "
            "Regenerate the artifact with post semantics or store next states explicitly."
        )

    schema = getattr(dataset, "token_schema", None)
    state_idx = int(schema.field_index("state")) if schema is not None else 0
    action_idx = int(schema.field_index("action")) if schema is not None else 1

    reconstructed: list[np.ndarray] = []
    for ep in episodes:
        rows = ep.copy()
        for row in rows[:-1]:
            row[state_idx] = _cb_deterministic_next_state(
                env=env,
                state=int(row[state_idx]),
                action=int(row[action_idx]),
            )
        reconstructed.append(rows)

    note = (
        "pre artifact scored on reconstructed post-action states, because CB terminal "
        "goal/bomb states are reached after the stored pre-state action."
    )
    return reconstructed, "post_reconstructed_from_pre", note


def _segment_inputs_from_episode_tokens(dataset, episodes: list[np.ndarray], segment_limit: int):
    limit = max(0, min(int(segment_limit), len(getattr(dataset, "indices", []))))
    joined_dim = int(getattr(dataset, "joined_dim", episodes[0].shape[1] if episodes else 4))
    required_rows = int(getattr(dataset, "required_rows", 1))
    target_shift = str(getattr(dataset, "target_shift", "token"))

    inputs = []
    for ep_idx, start_row in list(getattr(dataset, "indices", []))[:limit]:
        rows = episodes[int(ep_idx)]
        seg_rows = rows[int(start_row) : int(start_row) + required_rows]
        flat = seg_rows.reshape(-1)
        if target_shift == "transition":
            x = flat[:-joined_dim]
        else:
            x = flat[:-1]
        inputs.append(torch.from_numpy(x.astype(np.int64)))
    return inputs


def _score_dfa_list(
    dataset,
    adapter,
    dfa_list,
    *,
    segment_inputs,
    episode_inputs,
    segment_limit: int,
    episode_limit: int | None,
):
    formula_reports = []
    segment_masks = []
    episode_masks = []
    segment_scores_per_formula = []
    episode_scores_per_formula = []

    for formula_idx, dfa in enumerate(dfa_list):
        seg_scores = []
        if segment_inputs is not None and len(segment_inputs) > 0:
            seg_batch = (
                torch.stack(segment_inputs, dim=0)
                if segment_limit < 0
                else torch.stack(segment_inputs[:segment_limit], dim=0)
            )
            sat = adapter.batch_check_dfa_sat(seg_batch, dfa)
            seg_scores = [float(x) for x in sat.detach().cpu().tolist()]
        segment_scores_per_formula.append(np.asarray(seg_scores, dtype=np.float32))
        seg_mask = np.asarray(seg_scores, dtype=np.float32) >= 0.5 if seg_scores else np.asarray([])
        segment_masks.append(seg_mask)

        ep_scores = []
        ep_iter = episode_inputs if episode_limit is None else episode_inputs[:episode_limit]
        for ep in ep_iter:
            sat = adapter.batch_check_dfa_sat(ep.unsqueeze(0), dfa)
            ep_scores.append(float(sat[0].item()))
        episode_scores_per_formula.append(np.asarray(ep_scores, dtype=np.float32))
        ep_mask = np.asarray(ep_scores, dtype=np.float32) >= 0.5 if ep_scores else np.asarray([])
        episode_masks.append(ep_mask)

        formula_reports.append(
            {
                "formula_index": int(formula_idx),
                "dfa_num_states": int(getattr(dfa, "num_of_states", 0) or 0),
                "dfa_num_symbols": int(len(getattr(dfa, "dictionary_symbols", []) or [])),
                "segment_sample_size": int(len(seg_scores)),
                "segment_satisfaction_rate": float(np.mean(seg_mask)) if seg_mask.size else None,
                "segment_satisfied_count": int(np.sum(seg_mask)) if seg_mask.size else 0,
                "episode_sample_size": int(len(ep_scores)),
                "episode_satisfaction_rate": float(np.mean(ep_mask)) if ep_mask.size else None,
                "episode_satisfied_count": int(np.sum(ep_mask)) if ep_mask.size else 0,
                "example_satisfied_episode_indices": [
                    int(i) for i in np.flatnonzero(ep_mask)[:20].tolist()
                ],
                "example_violating_episode_indices": [
                    int(i) for i in np.flatnonzero(~ep_mask)[:20].tolist()
                ]
                if ep_mask.size
                else [],
            }
        )

    combined = {}
    if segment_masks:
        n = min(len(m) for m in segment_masks if len(m) > 0) if any(len(m) > 0 for m in segment_masks) else 0
        if n > 0:
            mat = np.stack([m[:n] for m in segment_masks], axis=1)
            all_mask = np.all(mat, axis=1)
            any_mask = np.any(mat, axis=1)
            combined["segment_sample_size"] = int(n)
            combined["segment_satisfaction_all_rate"] = float(np.mean(all_mask))
            combined["segment_satisfaction_any_rate"] = float(np.mean(any_mask))
            combined["segment_satisfied_all_count"] = int(np.sum(all_mask))
            combined["segment_satisfied_any_count"] = int(np.sum(any_mask))

    if episode_masks:
        n = min(len(m) for m in episode_masks if len(m) > 0) if any(len(m) > 0 for m in episode_masks) else 0
        if n > 0:
            mat = np.stack([m[:n] for m in episode_masks], axis=1)
            all_mask = np.all(mat, axis=1)
            any_mask = np.any(mat, axis=1)
            combined["episode_sample_size"] = int(n)
            combined["episode_satisfaction_all_rate"] = float(np.mean(all_mask))
            combined["episode_satisfaction_any_rate"] = float(np.mean(any_mask))
            combined["episode_satisfied_all_count"] = int(np.sum(all_mask))
            combined["episode_satisfied_any_count"] = int(np.sum(any_mask))

    return formula_reports, combined, {
        "segment_scores": segment_scores_per_formula,
        "episode_scores": episode_scores_per_formula,
    }


def _write_spec_report_plots(out_dir: str, spec_names: list[str], spec_reports: dict[str, dict]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    if not spec_names:
        return

    ep_rates = []
    seg_rates = []
    for spec_name in spec_names:
        combined = spec_reports[spec_name]["combined"]
        ep_rates.append(float(combined.get("episode_satisfaction_all_rate") or 0.0))
        seg_rates.append(float(combined.get("segment_satisfaction_all_rate") or 0.0))

    x = np.arange(len(spec_names))
    width = 0.35
    plt.figure(figsize=(max(10, len(spec_names) * 0.8), 4.5))
    plt.bar(x - width / 2, ep_rates, width=width, label="episode_all_rate")
    plt.bar(x + width / 2, seg_rates, width=width, label="segment_all_rate")
    plt.xticks(x, spec_names, rotation=25, ha="right")
    plt.ylim(0.0, 1.0)
    plt.ylabel("Satisfaction rate")
    plt.title("CB spec satisfaction summary")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "spec_satisfaction_bar.png"))
    plt.close()


def _episode_outcome_arrays(dataset, episode_count: int | None = None):
    episodes = list(getattr(dataset, "episodes_tokens", []) or [])
    rewards = list(getattr(dataset, "episode_rewards", []) or [])
    n = len(episodes)
    if episode_count is not None:
        n = min(n, int(episode_count))

    returns = np.zeros(n, dtype=np.float32)
    goal_mask = np.zeros(n, dtype=bool)
    bomb_mask = np.zeros(n, dtype=bool)
    timeout_mask = np.zeros(n, dtype=bool)
    other_mask = np.zeros(n, dtype=bool)

    env_cfg = getattr(getattr(dataset, "env", None), "cfg", None)
    max_steps_cfg = int(getattr(env_cfg, "max_steps", 200))
    step_r = float(getattr(env_cfg, "step_reward", -0.01))
    goal_r = float(getattr(env_cfg, "goal_reward", 1.0))
    bomb_r = float(getattr(env_cfg, "bomb_reward", -1.0))
    goal_thresh = 0.5 * (step_r + goal_r)
    bomb_thresh = 0.5 * (step_r + bomb_r)

    for idx in range(n):
        rew = np.asarray(rewards[idx], dtype=np.float32) if idx < len(rewards) else np.asarray([], dtype=np.float32)
        returns[idx] = float(np.sum(rew)) if rew.size else 0.0
        if rew.size == 0:
            other_mask[idx] = True
            continue

        last = float(rew[-1])
        if int(rew.shape[0]) >= max_steps_cfg:
            timeout_mask[idx] = True
        elif last >= goal_thresh:
            goal_mask[idx] = True
        elif last <= bomb_thresh:
            bomb_mask[idx] = True
        else:
            other_mask[idx] = True

    return {
        "returns": returns,
        "goal_mask": goal_mask,
        "bomb_mask": bomb_mask,
        "timeout_mask": timeout_mask,
        "other_mask": other_mask,
    }


def _safe_mean_std(values: np.ndarray) -> tuple[float | None, float | None]:
    if values.size == 0:
        return None, None
    return float(np.mean(values)), float(np.std(values, ddof=0))


def _write_spec_outcome_plots(out_dir: str, spec_names: list[str], spec_rows: list[dict]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    if not spec_names:
        return

    metric_specs = [
        ("satisfaction_mean", "satisfaction_sd", "Episode satisfaction rate"),
        ("goal_hit_mean_on_satisfied", "goal_hit_sd_on_satisfied", "Goal-hit rate on satisfied episodes"),
        ("bomb_hit_mean_on_satisfied", "bomb_hit_sd_on_satisfied", "Bomb-hit rate on satisfied episodes"),
        ("return_mean_on_satisfied", "return_sd_on_satisfied", "Return on satisfied episodes"),
    ]

    x = np.arange(len(spec_names))
    fig, axes = plt.subplots(2, 2, figsize=(max(12, len(spec_names) * 0.9), 8.5))
    axes = axes.flatten()

    for ax, (mean_key, sd_key, title) in zip(axes, metric_specs):
        means = [
            float(row[mean_key]) if row.get(mean_key) is not None and np.isfinite(row[mean_key]) else np.nan
            for row in spec_rows
        ]
        sds = [
            float(row[sd_key]) if row.get(sd_key) is not None and np.isfinite(row[sd_key]) else np.nan
            for row in spec_rows
        ]
        ax.bar(x, means, yerr=sds, capsize=3, color="#2563eb")
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(spec_names, rotation=25, ha="right")
        ax.grid(axis="y", alpha=0.2)
        if "Return" in title:
            finite_means = np.asarray([m for m in means if np.isfinite(m)], dtype=np.float32)
            finite_sds = np.asarray([s for s in sds if np.isfinite(s)], dtype=np.float32)
            if finite_means.size:
                lo = float(np.min(finite_means - finite_sds)) if finite_sds.size else float(np.min(finite_means))
                hi = float(np.max(finite_means + finite_sds)) if finite_sds.size else float(np.max(finite_means))
                pad = max(0.1, 0.1 * (hi - lo if hi > lo else 1.0))
                ax.set_ylim(lo - pad, hi + pad)
        else:
            ax.set_ylim(0.0, 1.0)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "spec_outcome_stats_bar.png"))
    plt.close(fig)


def _analyze_dataset_artifact(
    *,
    args,
    dataset,
    base_dir: str,
    stem: str,
    artifact_info: dict,
    seed: int,
) -> dict:
    analysis_root = _analysis_root(base_dir, stem)
    out_dir = os.path.join(analysis_root, "dataset_analysis")
    os.makedirs(out_dir, exist_ok=True)

    summary = {
        "env": "cb",
        "artifact_stem": stem,
        "artifact_paths": {
            "npz": artifact_info["npz_path"],
            "meta_json": artifact_info["meta_path"],
        },
        "analysis_root": analysis_root,
        "analysis_sequence_length": int(args.sequence_length),
        "analysis_segment_limit": int(args.analysis_segment_limit),
        "analysis_segment_mode": "all" if int(args.analysis_segment_limit) < 0 else "sampled",
        "policy_mix_spec": str(
            getattr(dataset, "policy_mix_spec", getattr(dataset, "cb_policy_mix_spec", ""))
        ),
        "policy_mix_sampling": str(
            getattr(dataset, "policy_mix_sampling", getattr(dataset, "cb_policy_mix_sampling", ""))
        ),
        "policy_mix_normal_spec": getattr(
            dataset, "policy_mix_normal_spec", getattr(dataset, "cb_policy_mix_normal_spec", None)
        ),
        "policy_mix_normal_mean_mode": str(
            getattr(
                dataset,
                "policy_mix_normal_mean_mode",
                getattr(dataset, "cb_policy_mix_normal_mean_mode", ""),
            )
        ),
        "state_semantics": str(getattr(dataset, "state_semantics", "")),
        "seed": int(seed),
        "analysis_spec_count": int(len(CB_SPECS)),
    }
    summary.update(_dataset_overview(dataset))
    _write_base_plots(dataset, out_dir, summary)

    scoring_episodes, scoring_state_semantics, scoring_note = _spec_scoring_episode_tokens(dataset)
    summary["spec_scoring_state_semantics"] = scoring_state_semantics
    if scoring_note is not None:
        summary["spec_scoring_note"] = scoring_note

    raw_segment_limit = int(args.analysis_segment_limit)
    exact_segments = raw_segment_limit < 0
    segment_limit = len(dataset) if exact_segments else min(raw_segment_limit, len(dataset))
    segment_inputs = _segment_inputs_from_episode_tokens(dataset, scoring_episodes, segment_limit)
    episode_limit = args.analysis_episode_limit
    episodes = scoring_episodes
    if episode_limit is not None:
        episodes = episodes[: int(episode_limit)]
    episode_inputs = [
        torch.from_numpy(np.asarray(ep, dtype=np.int64).reshape(-1)) for ep in episodes
    ]
    outcome_arrays = _episode_outcome_arrays(dataset, episode_count=len(episode_inputs))

    spec_names = list(CB_SPECS.keys())
    spec_reports: dict[str, dict] = {}
    spec_rows: list[dict] = []
    outcome_rows: list[dict] = []

    spec_iter = tqdm(spec_names, desc=f"CB specs | {stem}", leave=False, unit="spec")
    for spec_name in spec_iter:
        spec_def = CB_SPECS[spec_name]
        spec_args = SimpleNamespace(
            env="cb",
            spec=spec_name,
            ltl_formula=None,
            ltl_formulas=None,
            use_safe_dfa=True,
            dfa_backend="auto",
            dfa_mode="single",
            constraint_dims=[0],
        )
        adapter, _, raw_dfa = build_adapter_and_dfa(spec_args, dataset)
        formulas = resolve_formulas(spec_args, dataset=dataset)
        dfa_list = raw_dfa if isinstance(raw_dfa, list) else [raw_dfa]
        formula_reports, combined, score_detail = _score_dfa_list(
            dataset,
            adapter,
            dfa_list,
            segment_inputs=segment_inputs,
            episode_inputs=episode_inputs,
            segment_limit=segment_limit,
            episode_limit=episode_limit,
        )
        episode_scores = score_detail["episode_scores"][0] if score_detail["episode_scores"] else np.asarray([], dtype=np.float32)
        episode_sat_mask = (
            np.asarray(episode_scores, dtype=np.float32) >= 0.5
            if episode_scores.size
            else np.asarray([], dtype=bool)
        )
        sat_count = int(np.sum(episode_sat_mask)) if episode_sat_mask.size else 0
        sat_returns = outcome_arrays["returns"][episode_sat_mask] if episode_sat_mask.size else np.asarray([], dtype=np.float32)
        sat_goal = outcome_arrays["goal_mask"][episode_sat_mask] if episode_sat_mask.size else np.asarray([], dtype=bool)
        sat_bomb = outcome_arrays["bomb_mask"][episode_sat_mask] if episode_sat_mask.size else np.asarray([], dtype=bool)
        sat_satisfaction_mean, sat_satisfaction_sd = _safe_mean_std(episode_sat_mask.astype(np.float32))
        goal_mean, goal_sd = _safe_mean_std(sat_goal.astype(np.float32)) if sat_count else (None, None)
        bomb_mean, bomb_sd = _safe_mean_std(sat_bomb.astype(np.float32)) if sat_count else (None, None)
        ret_mean, ret_sd = _safe_mean_std(np.asarray(sat_returns, dtype=np.float32)) if sat_count else (None, None)

        report = {
            "description": spec_def.get("description"),
            "formulas": formulas,
            "formula_reports": formula_reports,
            "combined": combined,
        }
        spec_reports[spec_name] = report
        spec_rows.append(
            {
                "spec": spec_name,
                "description": spec_def.get("description"),
                "formula": formulas[0] if formulas else None,
                "segment_satisfaction_all_rate": combined.get("segment_satisfaction_all_rate"),
                "episode_satisfaction_all_rate": combined.get("episode_satisfaction_all_rate"),
                "segment_satisfied_all_count": combined.get("segment_satisfied_all_count"),
                "episode_satisfied_all_count": combined.get("episode_satisfied_all_count"),
            }
        )
        outcome_rows.append(
            {
                "spec": spec_name,
                "description": spec_def.get("description"),
                "formula": formulas[0] if formulas else None,
                "episode_sample_size": int(episode_sat_mask.size),
                "satisfaction_mean": sat_satisfaction_mean,
                "satisfaction_sd": sat_satisfaction_sd,
                "satisfied_episode_count": sat_count,
                "goal_hit_mean_on_satisfied": goal_mean,
                "goal_hit_sd_on_satisfied": goal_sd,
                "bomb_hit_mean_on_satisfied": bomb_mean,
                "bomb_hit_sd_on_satisfied": bomb_sd,
                "return_mean_on_satisfied": ret_mean,
                "return_sd_on_satisfied": ret_sd,
            }
        )

    summary["spec_order"] = spec_names
    summary["spec_reports"] = spec_reports
    summary["spec_summary_rows"] = spec_rows
    summary["spec_outcome_rows"] = outcome_rows

    _write_spec_report_plots(out_dir, spec_names, spec_reports)
    _write_spec_outcome_plots(out_dir, spec_names, outcome_rows)

    csv_path = os.path.join(out_dir, "spec_satisfaction.csv")
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        f.write(
            "spec,description,formula,segment_satisfaction_all_rate,episode_satisfaction_all_rate,"
            "segment_satisfied_all_count,episode_satisfied_all_count\n"
        )
        for row in spec_rows:
            f.write(
                f"{row['spec']},{json.dumps(row['description'])},{json.dumps(row['formula'])},"
                f"{row['segment_satisfaction_all_rate']},{row['episode_satisfaction_all_rate']},"
                f"{row['segment_satisfied_all_count']},{row['episode_satisfied_all_count']}\n"
            )

    outcome_csv_path = os.path.join(out_dir, "spec_outcome_stats.csv")
    with open(outcome_csv_path, "w", encoding="utf-8", newline="") as f:
        f.write(
            "spec,description,formula,episode_sample_size,satisfaction_mean,satisfaction_sd,"
            "satisfied_episode_count,goal_hit_mean_on_satisfied,goal_hit_sd_on_satisfied,"
            "bomb_hit_mean_on_satisfied,bomb_hit_sd_on_satisfied,return_mean_on_satisfied,"
            "return_sd_on_satisfied\n"
        )
        for row in outcome_rows:
            f.write(
                f"{row['spec']},{json.dumps(row['description'])},{json.dumps(row['formula'])},"
                f"{row['episode_sample_size']},{row['satisfaction_mean']},{row['satisfaction_sd']},"
                f"{row['satisfied_episode_count']},{row['goal_hit_mean_on_satisfied']},"
                f"{row['goal_hit_sd_on_satisfied']},{row['bomb_hit_mean_on_satisfied']},"
                f"{row['bomb_hit_sd_on_satisfied']},{row['return_mean_on_satisfied']},"
                f"{row['return_sd_on_satisfied']}\n"
            )

    summary["analysis_outputs"] = {
        "summary_json": os.path.join(out_dir, "summary.json"),
        "spec_satisfaction_csv": csv_path,
        "spec_outcome_stats_csv": outcome_csv_path,
        "spec_outcome_stats_png": os.path.join(out_dir, "spec_outcome_stats_bar.png"),
        "state_visitation_csv": os.path.join(out_dir, "state_visitation.csv"),
        "state_visitation_heatmap_png": os.path.join(out_dir, "state_visitation_heatmap.png"),
        "plots_dir": out_dir,
    }

    _write_json(summary["analysis_outputs"]["summary_json"], summary)
    print(f"[analysis] wrote {summary['analysis_outputs']['summary_json']}")
    return summary


def _resolve_dynamics_device(raw: str) -> torch.device:
    if raw == "cpu":
        return torch.device("cpu")
    if raw == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--dynamics_device cuda requested but CUDA is not available.")
        return torch.device("cuda:0")
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _fit_dynamics_for_artifact(*, args, dataset, base_dir: str, stem: str, seed: int, mix_spec: str, semantics: str) -> dict:
    ckpt_path = _dynamics_checkpoint_path(base_dir, stem)
    log_path = os.path.join(os.path.dirname(ckpt_path), "train_log.csv")
    summary_path = os.path.join(os.path.dirname(ckpt_path), "summary.json")
    if os.path.exists(ckpt_path) and not bool(args.overwrite_dynamics):
        return {
            "status": "skipped_existing",
            "checkpoint_path": ckpt_path,
            "summary_path": summary_path if os.path.exists(summary_path) else None,
        }

    device = _resolve_dynamics_device(str(args.dynamics_device))
    dyn_args = SimpleNamespace(
        env="cb",
        seed=int(seed),
        cb_policy_mix_spec=str(mix_spec),
        cb_policy_mix_sampling=str(args.policy_mix_sampling),
        cb_state_semantics=str(semantics),
        num_episodes=int(args.num_episodes),
        max_steps=int(args.max_steps),
    )
    model, stats = fit_neural_dynamics_model(
        base_dataset=dataset,
        hidden_dim=int(args.dynamics_hidden_dim),
        num_layers=int(args.dynamics_layers),
        epochs=int(args.dynamics_epochs),
        batch_size=int(args.dynamics_batch_size),
        lr=float(args.dynamics_lr),
        weight_decay=float(args.dynamics_weight_decay),
        val_fraction=float(args.dynamics_val_fraction),
        device=device,
        seed=int(seed),
        temperature=float(args.dynamics_temperature),
        freeze_after_fit=True,
        log_path=log_path,
    )
    dataset_ids = build_dynamics_dataset_identifiers(dataset, args=dyn_args)
    stats["dataset_identifiers"] = dataset_ids
    stats["checkpoint_path"] = ckpt_path
    save_neural_dynamics_checkpoint(
        path=ckpt_path,
        model=model,
        stats=stats,
        dataset_identifiers=dataset_ids,
    )
    summary = {
        "status": "trained",
        "checkpoint_path": ckpt_path,
        "training_log_path": log_path,
        "device": str(device),
        "stats": stats,
    }
    _write_json(summary_path, summary)
    print(f"[dynamics] wrote {ckpt_path}")
    return summary


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    base_dir = os.path.abspath(args.dataset_artifact_dir)
    os.makedirs(base_dir, exist_ok=True)

    policy_mix_specs = _dataset_spec_list(args)
    state_semantics = _resolve_state_semantics(list(args.state_semantics))
    total_jobs = len(policy_mix_specs) * len(state_semantics) * len(args.seeds)

    created: list[dict] = []
    skipped: list[dict] = []
    analysis_skipped: list[dict] = []
    dynamics_reports: list[dict] = []

    artifact_iter = tqdm(
        [
            (semantics, mix_spec, seed)
            for semantics in state_semantics
            for mix_spec in policy_mix_specs
            for seed in args.seeds
        ],
        total=total_jobs,
        desc="CB artifacts",
        unit="artifact",
    )

    for semantics, mix_spec, seed in artifact_iter:
        stem = _artifact_stem(args.dataset_artifact_name, semantics, mix_spec, seed)
        npz_path, meta_path = _artifact_paths(base_dir, stem)
        analysis_summary_path = _analysis_summary_path(base_dir, stem)
        dynamics_path = _dynamics_checkpoint_path(base_dir, stem)
        artifact_done = os.path.exists(npz_path) and os.path.exists(meta_path)
        analysis_done = os.path.exists(analysis_summary_path)
        dynamics_done = os.path.exists(dynamics_path)

        artifact_iter.set_postfix_str(f"{semantics} | seed={seed} | {_slug(mix_spec)}")

        if (
            args.skip_existing
            and artifact_done
            and (not args.analyze_all_specs or analysis_done)
            and (not args.fit_neural_dynamics or dynamics_done)
        ):
            skipped.append(
                {
                    "seed": int(seed),
                    "state_semantics": semantics,
                    "policy_mix_spec": mix_spec,
                    "npz_path": npz_path,
                    "meta_path": meta_path,
                    "analysis_summary_path": analysis_summary_path if args.analyze_all_specs else None,
                    "dynamics_checkpoint_path": dynamics_path if args.fit_neural_dynamics else None,
                }
            )
            continue

        dataset = _make_dataset(args, mix_spec, semantics, seed)

        if not artifact_done:
            info = _save_dataset_artifact(base_dir, stem, args, dataset, mix_spec, semantics, seed)
            created.append(
                {
                    "seed": int(seed),
                    "state_semantics": semantics,
                    "policy_mix_spec": mix_spec,
                    "npz_path": info["npz_path"],
                    "meta_path": info["meta_path"],
                    "num_segments": int(len(dataset)),
                    "num_episodes": int(len(dataset.episodes_tokens)),
                    "episode_length_mean": float(
                        np.mean([len(ep) - 1 for ep in dataset.episodes_tokens])
                    )
                    if getattr(dataset, "episodes_tokens", None)
                    else 0.0,
                }
            )
            print(f"[created] semantics={semantics} mix={mix_spec} seed={seed} -> {info['npz_path']}")
        else:
            created.append(
                {
                    "seed": int(seed),
                    "state_semantics": semantics,
                    "policy_mix_spec": mix_spec,
                    "npz_path": npz_path,
                    "meta_path": meta_path,
                    "num_segments": int(len(dataset)),
                    "num_episodes": int(len(dataset.episodes_tokens)),
                    "episode_length_mean": float(
                        np.mean([len(ep) - 1 for ep in dataset.episodes_tokens])
                    )
                    if getattr(dataset, "episodes_tokens", None)
                    else 0.0,
                    "reused_existing_artifact": True,
                }
            )

        if args.analyze_all_specs:
            artifact_info = {"npz_path": npz_path, "meta_path": meta_path}
            if not analysis_done or not args.skip_existing:
                _analyze_dataset_artifact(
                    args=args,
                    dataset=dataset,
                    base_dir=base_dir,
                    stem=stem,
                    artifact_info=artifact_info,
                    seed=int(seed),
                )
            else:
                analysis_skipped.append(
                    {
                        "seed": int(seed),
                        "state_semantics": semantics,
                        "policy_mix_spec": mix_spec,
                        "analysis_summary_path": analysis_summary_path,
                    }
                )

        if args.fit_neural_dynamics:
            dyn_report = _fit_dynamics_for_artifact(
                args=args,
                dataset=dataset,
                base_dir=base_dir,
                stem=stem,
                seed=int(seed),
                mix_spec=mix_spec,
                semantics=semantics,
            )
            dyn_report.update(
                {
                    "seed": int(seed),
                    "state_semantics": semantics,
                    "policy_mix_spec": mix_spec,
                    "artifact_stem": stem,
                }
            )
            dynamics_reports.append(dyn_report)

    manifest = {
        "output_root": base_dir,
        "dataset_artifact_name": args.dataset_artifact_name,
        "mode": "full_simplex" if bool(args.full_simplex) else "explicit_mix_list",
        "policy_names": list(args.policy_names),
        "mix_grid_step": float(args.mix_grid_step) if bool(args.full_simplex) else None,
        "policy_mix_spec_count": len(policy_mix_specs),
        "policy_mix_spec_preview": policy_mix_specs[:20],
        "state_semantics": state_semantics,
        "state_semantics_requested": list(args.state_semantics),
        "seeds": [int(s) for s in args.seeds],
        "num_episodes": int(args.num_episodes),
        "max_steps": int(args.max_steps),
        "sequence_length": int(args.sequence_length),
        "stochastic": bool(args.stochastic),
        "policy_mix_sampling": str(args.policy_mix_sampling),
        "policy_mix_normal_spec": args.policy_mix_normal_spec,
        "policy_mix_normal_mean_mode": str(args.policy_mix_normal_mean_mode),
        "skip_existing": bool(args.skip_existing),
        "analyze_all_specs": bool(args.analyze_all_specs),
        "analysis_segment_limit": int(args.analysis_segment_limit),
        "analysis_segment_mode": "all" if int(args.analysis_segment_limit) < 0 else "sampled",
        "analysis_episode_limit": args.analysis_episode_limit,
        "fit_neural_dynamics": bool(args.fit_neural_dynamics),
        "overwrite_dynamics": bool(args.overwrite_dynamics),
        "dynamics_device": str(args.dynamics_device),
        "dynamics_epochs": int(args.dynamics_epochs),
        "jobs_total": int(total_jobs),
        "created_count": int(len(created)),
        "skipped_count": int(len(skipped)),
        "analysis_skipped_count": int(len(analysis_skipped)),
        "dynamics_report_count": int(len(dynamics_reports)),
        "created_preview": created[:20],
        "skipped_preview": skipped[:20],
        "analysis_skipped_preview": analysis_skipped[:20],
        "dynamics_preview": dynamics_reports[:20],
    }
    _write_json(os.path.join(base_dir, "colorbomb_dataset_manifest.json"), manifest)

    print(
        f"[done] created={len(created)} skipped={len(skipped)} analysis_skipped={len(analysis_skipped)} "
        f"output_dir={base_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
