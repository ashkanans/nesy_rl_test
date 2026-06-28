from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datasets.artifact_io import load_sequence_dataset_artifact


CB_CELL_COLORS = {
    ".": "#f8fafc",
    "#": "#334155",
    "B": "#ef4444",
    "P": "#84cc16",
    "Y": "#facc15",
    "BLU": "#38bdf8",
    "S": "#60a5fa",
}


def parse_args(argv: list[str] | None = None):
    p = argparse.ArgumentParser(
        description="Inspect a CB dataset artifact and visually replay its trajectories."
    )
    p.add_argument("artifact_path", type=str, help="Path to a dataset artifact .npz or directory.")
    p.add_argument(
        "--sequence_length",
        type=int,
        default=16,
        help="Sequence length used to reindex segments when loading the artifact.",
    )
    p.add_argument(
        "--episode_index",
        type=int,
        default=0,
        help="Episode to inspect when not browsing interactively.",
    )
    p.add_argument(
        "--browse",
        action="store_true",
        help="Interactive episode browser in the terminal.",
    )
    p.add_argument(
        "--mode",
        type=str,
        choices=["text", "plot", "animate"],
        default="plot",
        help="How to display the selected episode.",
    )
    p.add_argument(
        "--save_path",
        type=str,
        default=None,
        help="Optional path to save the plot/animation. Defaults to the artifact directory.",
    )
    p.add_argument(
        "--fps",
        type=int,
        default=2,
        help="Animation frame rate when --mode animate is used.",
    )
    p.add_argument(
        "--sleep",
        type=float,
        default=0.4,
        help="Pause between steps in text mode.",
    )
    p.add_argument(
        "--max_list",
        type=int,
        default=20,
        help="How many episodes to list in browse mode.",
    )
    return p.parse_args(argv)


def _artifact_npz_path(dataset) -> Path:
    paths = getattr(dataset, "meta", {}).get("paths", {}) or {}
    npz_path = paths.get("npz")
    if npz_path:
        return Path(str(npz_path)).resolve()
    source = Path(str(getattr(dataset, "source_artifact_path", "")))
    if source.suffix == ".npz":
        return source.resolve()
    return source.resolve()


def _default_output_path(dataset, ep_idx: int, suffix: str) -> Path:
    npz_path = _artifact_npz_path(dataset)
    stem = npz_path.stem
    return npz_path.parent / f"{stem}_episode{int(ep_idx)}{suffix}"


def _resolve_output_path(dataset, ep_idx: int, save_path: str | None, suffix: str) -> Path | None:
    if save_path:
        return Path(save_path).expanduser().resolve()
    return _default_output_path(dataset, ep_idx, suffix)


def _episode_return(episode_rewards, idx: int):
    if episode_rewards is None or idx >= len(episode_rewards):
        return None
    return float(np.sum(np.asarray(episode_rewards[idx], dtype=np.float32)))


def _episode_outcome(env, episode_rewards, idx: int):
    if episode_rewards is None or idx >= len(episode_rewards):
        return None
    rew = np.asarray(episode_rewards[idx], dtype=np.float32)
    if rew.size == 0:
        return "other"
    env_cfg = getattr(env, "cfg", None)
    max_steps_cfg = int(getattr(env_cfg, "max_steps", 200))
    step_r = float(getattr(env_cfg, "step_reward", -0.01))
    goal_r = float(getattr(env_cfg, "goal_reward", 1.0))
    bomb_r = float(getattr(env_cfg, "bomb_reward", -1.0))
    goal_thresh = 0.5 * (step_r + goal_r)
    bomb_thresh = 0.5 * (step_r + bomb_r)
    if int(rew.shape[0]) >= max_steps_cfg:
        return "timeout"
    last = float(rew[-1])
    if last >= goal_thresh:
        return "goal"
    if last <= bomb_thresh:
        return "bomb_hit"
    return "other"


def _episode_states(dataset, ep_idx: int):
    ep = np.asarray(dataset.episodes_tokens[ep_idx], dtype=np.int64)
    core = ep[:-1]
    raw_states = [int(row[0]) for row in core.tolist()]
    semantics = str(getattr(dataset, "state_semantics", "post"))
    env = dataset.env
    if semantics == "post":
        start_state = int(env._pos_to_state(env.start_pos)) if hasattr(env, "_pos_to_state") else raw_states[0]
        return [start_state] + raw_states, core
    return raw_states, core


def _state_to_pos(env, state: int):
    if hasattr(env, "_state_to_pos"):
        return env._state_to_pos(int(state))
    n_cols = int(getattr(env, "n_cols", 1))
    return divmod(int(state), n_cols)


def _draw_cb_grid(ax, env):
    from matplotlib.patches import Rectangle

    n_rows = int(getattr(env, "n_rows", len(getattr(env, "grid", []))))
    n_cols = int(getattr(env, "n_cols", len(getattr(env, "grid", [[]])[0]) if getattr(env, "grid", None) else 0))
    ax.set_xlim(0, n_cols)
    ax.set_ylim(n_rows, 0)
    ax.set_aspect("equal")
    ax.set_xticks(range(n_cols + 1))
    ax.set_yticks(range(n_rows + 1))
    ax.grid(color="#cbd5e1", linewidth=0.6)

    grid = getattr(env, "grid", [])
    for r in range(n_rows):
        for c in range(n_cols):
            cell = grid[r][c]
            color = CB_CELL_COLORS.get(cell, "#f8fafc")
            ax.add_patch(Rectangle((c, r), 1, 1, facecolor=color, edgecolor="#94a3b8", linewidth=0.8))
            label = cell if cell != "." else ""
            if label:
                ax.text(c + 0.5, r + 0.56, label, ha="center", va="center", fontsize=8, color="#111827")

    start_r, start_c = getattr(env, "start_pos", (0, 0))
    ax.add_patch(
        Rectangle((start_c, start_r), 1, 1, fill=False, edgecolor="#2563eb", linewidth=2.2)
    )
    ax.text(start_c + 0.5, start_r + 0.2, "START", ha="center", va="center", fontsize=8, color="#1d4ed8")


def _episode_summary(dataset, ep_idx: int):
    episode_rewards = getattr(dataset, "episode_rewards", None)
    episode_policy_labels = getattr(dataset, "episode_policy_labels", None)
    length = int(np.asarray(dataset.episodes_tokens[ep_idx]).shape[0] - 1)
    ret = _episode_return(episode_rewards, ep_idx)
    outcome = _episode_outcome(dataset.env, episode_rewards, ep_idx)
    policy = episode_policy_labels[ep_idx] if episode_policy_labels and ep_idx < len(episode_policy_labels) else None
    return {
        "episode_index": int(ep_idx),
        "length": length,
        "return": ret,
        "outcome": outcome,
        "policy": policy,
    }


def _print_episode_table(dataset, max_list: int):
    count = min(int(max_list), len(dataset.episodes_tokens))
    print("Episodes:")
    for i in range(count):
        s = _episode_summary(dataset, i)
        print(
            f"  [{s['episode_index']:4d}] len={s['length']:4d} "
            f"return={s['return'] if s['return'] is not None else 'n/a':>8} "
            f"outcome={s['outcome']:<8} policy={s['policy']}"
        )
    if len(dataset.episodes_tokens) > count:
        print(f"  ... {len(dataset.episodes_tokens) - count} more episodes")


def _replay_text(dataset, ep_idx: int, sleep: float):
    import time as _time

    env = dataset.env
    ep = np.asarray(dataset.episodes_tokens[ep_idx], dtype=np.int64)
    core = ep[:-1]
    print(f"Replaying episode {ep_idx}")
    print(json.dumps(_episode_summary(dataset, ep_idx), indent=2))
    print(env.render(mode="ansi") if hasattr(env, "render") else "")
    total_reward = 0.0
    for t, row in enumerate(core):
        state_token = int(row[0])
        action = int(row[1])
        reward = float(dataset.episode_rewards[ep_idx][t]) if getattr(dataset, "episode_rewards", None) else 0.0
        total_reward += reward
        print(f"\nstep={t} state={state_token} action={action} reward={reward:.3f}")
        if hasattr(env, "render"):
            print(env.render(mode="ansi"))
        _time.sleep(float(sleep))
    print(f"\nTotal reward: {total_reward:.3f}")


def _build_episode_path(dataset, ep_idx: int):
    env = dataset.env
    states, core = _episode_states(dataset, ep_idx)
    positions = []
    for s in states:
        r, c = _state_to_pos(env, int(s))
        positions.append((float(c) + 0.5, float(r) + 0.5))
    actions = [int(row[1]) for row in core.tolist()]
    rewards = (
        [float(x) for x in np.asarray(dataset.episode_rewards[ep_idx], dtype=np.float32).tolist()]
        if getattr(dataset, "episode_rewards", None) is not None
        and ep_idx < len(dataset.episode_rewards)
        else []
    )
    return positions, actions, rewards


def _render_static_plot(dataset, ep_idx: int, save_path: str | None):
    import matplotlib

    output_path = _resolve_output_path(dataset, ep_idx, save_path, ".png")
    if not os.environ.get("DISPLAY"):
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    env = dataset.env
    positions, actions, rewards = _build_episode_path(dataset, ep_idx)
    fig, ax = plt.subplots(figsize=(8, 8))
    _draw_cb_grid(ax, env)

    xs = [p[0] for p in positions]
    ys = [p[1] for p in positions]
    ax.plot(xs, ys, color="#dc2626", linewidth=2.5, marker="o", markersize=4)
    ax.scatter(xs[0], ys[0], s=120, color="#2563eb", zorder=5, label="start")
    ax.scatter(xs[-1], ys[-1], s=120, color="#16a34a", zorder=5, label="end")

    for i, (x, y) in enumerate(positions):
        if len(positions) <= 40 or i % 5 == 0 or i in {0, len(positions) - 1}:
            ax.text(x, y, str(i), color="#111827", fontsize=8, ha="center", va="center")

    s = _episode_summary(dataset, ep_idx)
    title = (
        f"Episode {ep_idx} | len={s['length']} | return={s['return'] if s['return'] is not None else 'n/a'} "
        f"| outcome={s['outcome']} | policy={s['policy']}"
    )
    ax.set_title(title)
    ax.legend(loc="upper right")
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=160)
        print(f"Saved plot to {output_path}")
    plt.close(fig)


def _render_animation(dataset, ep_idx: int, fps: int, save_path: str | None):
    import matplotlib

    output_path = _resolve_output_path(dataset, ep_idx, save_path, ".gif")
    if not os.environ.get("DISPLAY"):
        matplotlib.use("Agg")
    import matplotlib.animation as animation
    import matplotlib.pyplot as plt

    env = dataset.env
    positions, actions, rewards = _build_episode_path(dataset, ep_idx)
    fig, ax = plt.subplots(figsize=(8, 8))
    _draw_cb_grid(ax, env)

    xs = [p[0] for p in positions]
    ys = [p[1] for p in positions]
    (line,) = ax.plot([], [], color="#dc2626", linewidth=2.5, marker="o", markersize=4)
    current = ax.scatter([], [], s=160, color="#0f172a", zorder=6)
    step_text = ax.text(0.02, 1.02, "", transform=ax.transAxes, fontsize=10)
    title = ax.set_title("")

    def _update(frame):
        upto = frame + 1
        line.set_data(xs[:upto], ys[:upto])
        current.set_offsets(np.array([[xs[frame], ys[frame]]], dtype=np.float32))
        title.set_text(
            f"Episode {ep_idx} | frame={frame}/{len(xs) - 1} | "
            f"state={frame} | action={actions[frame - 1] if frame > 0 and frame - 1 < len(actions) else 'n/a'}"
        )
        if frame > 0 and frame - 1 < len(rewards):
            step_text.set_text(f"reward={rewards[frame - 1]:.3f}")
        else:
            step_text.set_text("")
        return line, current, title, step_text

    anim = animation.FuncAnimation(fig, _update, frames=len(xs), interval=max(50, int(1000 / max(1, fps))), blit=False)
    fig.tight_layout()

    if output_path:
        writer = animation.PillowWriter(fps=max(1, int(fps)))
        anim.save(output_path, writer=writer)
        print(f"Saved animation to {output_path}")
        plt.close(fig)


def _interactive_browse(dataset, args):
    max_list = int(args.max_list)
    while True:
        _print_episode_table(dataset, max_list=max_list)
        raw = input("Episode index, 'q' quit, 't' text, 'p' plot, 'a' animate: ").strip().lower()
        if raw in {"q", "quit", "exit"}:
            return
        if raw in {"t", "p", "a"}:
            idx = int(input("Episode index: ").strip())
            mode = {"t": "text", "p": "plot", "a": "animate"}[raw]
        else:
            try:
                idx = int(raw)
            except Exception:
                print("Invalid input.")
                continue
            mode = args.mode

        if idx < 0 or idx >= len(dataset.episodes_tokens):
            print(f"Invalid episode index {idx}.")
            continue
        if mode == "text":
            _replay_text(dataset, idx, args.sleep)
        elif mode == "plot":
            _render_static_plot(dataset, idx, args.save_path)
        else:
            _render_animation(dataset, idx, args.fps, args.save_path)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    dataset = load_sequence_dataset_artifact(
        args.artifact_path,
        sequence_length=int(args.sequence_length),
    )

    print(f"Artifact: {dataset.source_artifact_path}")
    print(f"Env: {dataset.env_name}")
    print(f"Episodes: {len(dataset.episodes_tokens)}")
    print(f"Segments: {len(dataset)}")
    print(f"State semantics: {getattr(dataset, 'state_semantics', 'n/a')}")
    print(f"Policy mix: {getattr(dataset, 'cb_policy_mix_spec', 'n/a')}")
    print(f"Seed: {getattr(dataset, 'dataset_config', {}).get('seed', 'n/a')}")
    _print_episode_table(dataset, max_list=int(args.max_list))

    if args.browse:
        _interactive_browse(dataset, args)
        return 0

    ep_idx = int(args.episode_index)
    if ep_idx < 0 or ep_idx >= len(dataset.episodes_tokens):
        raise ValueError(f"Invalid episode index {ep_idx}; dataset has {len(dataset.episodes_tokens)} episodes.")

    if args.mode == "text":
        _replay_text(dataset, ep_idx, args.sleep)
    elif args.mode == "plot":
        _render_static_plot(dataset, ep_idx, args.save_path)
    else:
        _render_animation(dataset, ep_idx, args.fps, args.save_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
