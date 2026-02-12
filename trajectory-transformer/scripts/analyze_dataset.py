#!/usr/bin/env python3
import argparse
import csv
import json
import os
from datetime import datetime

import gym
import d4rl  # noqa: F401
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.animation as animation  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze a D4RL dataset and produce stats + plots."
    )
    parser.add_argument("--dataset", type=str, default="antmaze-umaze-v0")
    parser.add_argument(
        "--outdir",
        type=str,
        default=None,
        help="Output directory. Defaults to artifacts/dataset_analysis/<dataset>/<timestamp>",
    )
    parser.add_argument(
        "--max_points_for_plots",
        type=int,
        default=200_000,
        help="Maximum sampled transitions used for scatter/hist plots.",
    )
    parser.add_argument(
        "--num_trajectories_plot",
        type=int,
        default=25,
        help="Number of full trajectories to draw in XY trajectory plot.",
    )
    parser.add_argument(
        "--num_trajectories_gif",
        type=int,
        default=8,
        help="Number of trajectories to animate in the GIF.",
    )
    parser.add_argument(
        "--gif_max_steps",
        type=int,
        default=400,
        help="Max steps shown in the trajectory GIF.",
    )
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def to_numpy(x):
    return np.asarray(x)


def infer_episode_boundaries(terminals, timeouts):
    terminals = terminals.astype(bool)
    if timeouts is None:
        done = terminals.copy()
    else:
        done = np.logical_or(terminals, timeouts.astype(bool))

    end_idxs = np.where(done)[0]
    n = terminals.shape[0]
    if end_idxs.size == 0 or end_idxs[-1] != n - 1:
        end_idxs = np.append(end_idxs, n - 1)
    start_idxs = np.concatenate(([0], end_idxs[:-1] + 1))

    valid = start_idxs <= end_idxs
    start_idxs = start_idxs[valid]
    end_idxs = end_idxs[valid]
    return start_idxs, end_idxs


def basic_stats(x):
    x = to_numpy(x).astype(np.float64)
    return {
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
        "p01": float(np.percentile(x, 1)),
        "p05": float(np.percentile(x, 5)),
        "p50": float(np.percentile(x, 50)),
        "p95": float(np.percentile(x, 95)),
        "p99": float(np.percentile(x, 99)),
    }


def dim_stats(arr):
    arr = to_numpy(arr)
    rows = []
    for d in range(arr.shape[1]):
        v = arr[:, d].astype(np.float64)
        rows.append(
            {
                "dim": d,
                "mean": float(np.mean(v)),
                "std": float(np.std(v)),
                "min": float(np.min(v)),
                "max": float(np.max(v)),
                "p01": float(np.percentile(v, 1)),
                "p50": float(np.percentile(v, 50)),
                "p99": float(np.percentile(v, 99)),
            }
        )
    return rows


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def choose_indices(n, max_points, seed):
    if n <= max_points:
        return np.arange(n)
    rng = np.random.default_rng(seed)
    return rng.choice(n, size=max_points, replace=False)


def plot_rewards(rewards, outdir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(rewards, bins=50)
    axes[0].set_title("Reward Distribution")
    axes[0].set_xlabel("reward")
    axes[0].set_ylabel("count")

    window = max(1, min(5000, rewards.shape[0] // 200))
    kernel = np.ones(window, dtype=np.float64) / float(window)
    smooth = np.convolve(rewards.astype(np.float64), kernel, mode="valid")
    axes[1].plot(smooth, linewidth=1.0)
    axes[1].set_title(f"Smoothed Reward (window={window})")
    axes[1].set_xlabel("transition")
    axes[1].set_ylabel("reward")

    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "rewards.png"), dpi=150)
    plt.close(fig)


def plot_episode_stats(lengths, returns, outdir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(lengths, bins=40)
    axes[0].set_title("Episode Length Distribution")
    axes[0].set_xlabel("episode length")
    axes[0].set_ylabel("count")

    axes[1].hist(returns, bins=40)
    axes[1].set_title("Episode Return Distribution")
    axes[1].set_xlabel("episode return")
    axes[1].set_ylabel("count")

    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "episode_stats.png"), dpi=150)
    plt.close(fig)


def plot_per_dim_hists(values, name, outdir):
    n_dims = values.shape[1]
    n_cols = 4
    n_rows = int(np.ceil(n_dims / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 2.8 * n_rows))
    axes = np.array(axes).reshape(-1)

    for d in range(n_dims):
        ax = axes[d]
        ax.hist(values[:, d], bins=60)
        ax.set_title(f"{name}[{d}]")
    for d in range(n_dims, axes.size):
        axes[d].axis("off")

    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f"{name}_per_dim_hist.png"), dpi=150)
    plt.close(fig)


def plot_xy(observations, start_idxs, end_idxs, outdir, num_trajectories_plot, seed):
    if observations.shape[1] < 2:
        return None

    x = observations[:, 0]
    y = observations[:, 1]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    hb = axes[0].hexbin(x, y, gridsize=70, mincnt=1)
    fig.colorbar(hb, ax=axes[0], label="count")
    axes[0].set_title("XY Visitation Heatmap")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")

    rng = np.random.default_rng(seed)
    total_eps = start_idxs.shape[0]
    count = min(total_eps, num_trajectories_plot)
    chosen = rng.choice(total_eps, size=count, replace=False)
    for ep in chosen:
        s = start_idxs[ep]
        e = end_idxs[ep] + 1
        axes[1].plot(x[s:e], y[s:e], linewidth=1.0, alpha=0.7)
    axes[1].set_title(f"Random Episode Trajectories (n={count})")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")

    fig.tight_layout()
    plot_path = os.path.join(outdir, "xy_plots.png")
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    return plot_path


def make_trajectory_gif(
    observations,
    start_idxs,
    end_idxs,
    outdir,
    num_trajectories_gif,
    gif_max_steps,
    seed,
):
    if observations.shape[1] < 2:
        return None

    x = observations[:, 0]
    y = observations[:, 1]
    total_eps = start_idxs.shape[0]
    if total_eps == 0:
        return None

    rng = np.random.default_rng(seed)
    count = min(total_eps, num_trajectories_gif)
    chosen = rng.choice(total_eps, size=count, replace=False)

    traces = []
    for ep in chosen:
        s = start_idxs[ep]
        e = min(end_idxs[ep] + 1, s + gif_max_steps)
        traces.append((x[s:e], y[s:e]))

    max_len = max(t[0].shape[0] for t in traces)
    if max_len <= 1:
        return None

    all_x = np.concatenate([t[0] for t in traces])
    all_y = np.concatenate([t[1] for t in traces])
    margin_x = max(1e-3, (np.max(all_x) - np.min(all_x)) * 0.05)
    margin_y = max(1e-3, (np.max(all_y) - np.min(all_y)) * 0.05)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(np.min(all_x) - margin_x, np.max(all_x) + margin_x)
    ax.set_ylim(np.min(all_y) - margin_y, np.max(all_y) + margin_y)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Dataset Trajectory Simulation (XY)")
    lines = [ax.plot([], [], linewidth=1.5)[0] for _ in traces]

    def init():
        for line in lines:
            line.set_data([], [])
        return lines

    def update(frame):
        step = frame + 1
        for line, (tx, ty) in zip(lines, traces):
            k = min(step, tx.shape[0])
            line.set_data(tx[:k], ty[:k])
        return lines

    anim = animation.FuncAnimation(
        fig,
        update,
        init_func=init,
        frames=max_len,
        interval=90,
        blit=True,
    )
    gif_path = os.path.join(outdir, "trajectory_simulation.gif")
    try:
        anim.save(gif_path, writer=animation.PillowWriter(fps=12))
    except Exception as exc:
        print(f"[warn] could not save GIF: {exc}")
        gif_path = None
    plt.close(fig)
    return gif_path


def write_report(path, summary, generated_files):
    lines = []
    lines.append("# Dataset Analysis Report")
    lines.append("")
    lines.append(f"- dataset: `{summary['dataset']}`")
    lines.append(f"- num_transitions: `{summary['num_transitions']}`")
    lines.append(f"- observation_dim: `{summary['observation_dim']}`")
    lines.append(f"- action_dim: `{summary['action_dim']}`")
    lines.append(f"- num_episodes: `{summary['num_episodes']}`")
    lines.append(f"- max_episode_steps(env): `{summary['env_max_episode_steps']}`")
    lines.append("")
    lines.append("## Core Metrics")
    lines.append(f"- reward_mean: `{summary['reward_stats']['mean']:.6f}`")
    lines.append(f"- reward_std: `{summary['reward_stats']['std']:.6f}`")
    lines.append(f"- positive_reward_fraction: `{summary['positive_reward_fraction']:.6f}`")
    lines.append(f"- terminal_fraction: `{summary['terminal_fraction']:.6f}`")
    lines.append(f"- timeout_fraction: `{summary['timeout_fraction']:.6f}`")
    lines.append(f"- success_episode_fraction: `{summary['success_episode_fraction']:.6f}`")
    lines.append("")
    lines.append("## Episode Metrics")
    lines.append(f"- episode_length_mean: `{summary['episode_length_stats']['mean']:.2f}`")
    lines.append(f"- episode_length_std: `{summary['episode_length_stats']['std']:.2f}`")
    lines.append(f"- episode_return_mean: `{summary['episode_return_stats']['mean']:.6f}`")
    lines.append(f"- episode_return_std: `{summary['episode_return_stats']['std']:.6f}`")
    lines.append("")
    lines.append("## Generated Files")
    for f in generated_files:
        lines.append(f"- `{f}`")

    with open(path, "w") as f:
        f.write("\n".join(lines))


def main():
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = args.outdir
    if outdir is None:
        outdir = os.path.join(
            "artifacts",
            "dataset_analysis",
            args.dataset.replace("/", "_"),
            timestamp,
        )
    os.makedirs(outdir, exist_ok=True)

    print(f"[info] loading env: {args.dataset}")
    env = gym.make(args.dataset)
    dataset = env.get_dataset()

    observations = to_numpy(dataset["observations"]).astype(np.float32)
    actions = to_numpy(dataset["actions"]).astype(np.float32)
    rewards = to_numpy(dataset["rewards"]).astype(np.float32).reshape(-1)
    terminals = to_numpy(dataset["terminals"]).reshape(-1).astype(np.int8)
    timeouts = dataset.get("timeouts", None)
    if timeouts is not None:
        timeouts = to_numpy(timeouts).reshape(-1).astype(np.int8)

    n = rewards.shape[0]
    start_idxs, end_idxs = infer_episode_boundaries(terminals, timeouts)
    lengths = (end_idxs - start_idxs + 1).astype(np.int32)
    returns = np.array(
        [float(np.sum(rewards[s : e + 1])) for s, e in zip(start_idxs, end_idxs)],
        dtype=np.float64,
    )
    successes = np.array(
        [float(np.any(rewards[s : e + 1] > 0.0)) for s, e in zip(start_idxs, end_idxs)],
        dtype=np.float64,
    )

    obs_dim_rows = dim_stats(observations)
    act_dim_rows = dim_stats(actions)
    write_csv(os.path.join(outdir, "observation_dimension_stats.csv"), obs_dim_rows)
    write_csv(os.path.join(outdir, "action_dimension_stats.csv"), act_dim_rows)

    summary = {
        "dataset": args.dataset,
        "num_transitions": int(n),
        "observation_dim": int(observations.shape[1]),
        "action_dim": int(actions.shape[1]),
        "dataset_keys": sorted(list(dataset.keys())),
        "num_episodes": int(lengths.shape[0]),
        "reward_stats": basic_stats(rewards),
        "episode_length_stats": basic_stats(lengths),
        "episode_return_stats": basic_stats(returns),
        "positive_reward_fraction": float(np.mean(rewards > 0.0)),
        "terminal_fraction": float(np.mean(terminals > 0)),
        "timeout_fraction": float(np.mean(timeouts > 0)) if timeouts is not None else 0.0,
        "success_episode_fraction": float(np.mean(successes)),
        "env_max_episode_steps": int(getattr(env, "_max_episode_steps", -1)),
        "observation_space_shape": tuple(int(x) for x in env.observation_space.shape),
        "action_space_shape": tuple(int(x) for x in env.action_space.shape),
    }

    with open(os.path.join(outdir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    idx = choose_indices(n, args.max_points_for_plots, args.seed)
    obs_sample = observations[idx]
    act_sample = actions[idx]
    rew_sample = rewards[idx]

    plot_rewards(rewards, outdir)
    plot_episode_stats(lengths, returns, outdir)
    plot_per_dim_hists(act_sample, "actions", outdir)
    plot_per_dim_hists(obs_sample, "observations", outdir)
    plot_xy(
        observations,
        start_idxs,
        end_idxs,
        outdir,
        num_trajectories_plot=args.num_trajectories_plot,
        seed=args.seed,
    )
    gif_path = make_trajectory_gif(
        observations,
        start_idxs,
        end_idxs,
        outdir,
        num_trajectories_gif=args.num_trajectories_gif,
        gif_max_steps=args.gif_max_steps,
        seed=args.seed,
    )

    generated_files = [
        "summary.json",
        "observation_dimension_stats.csv",
        "action_dimension_stats.csv",
        "rewards.png",
        "episode_stats.png",
        "actions_per_dim_hist.png",
        "observations_per_dim_hist.png",
        "xy_plots.png",
    ]
    if gif_path is not None:
        generated_files.append("trajectory_simulation.gif")

    write_report(os.path.join(outdir, "README.md"), summary, generated_files)

    print(f"[done] analysis written to: {outdir}")
    print("[done] key numbers:")
    print(
        f"  transitions={summary['num_transitions']}, "
        f"episodes={summary['num_episodes']}, "
        f"obs_dim={summary['observation_dim']}, "
        f"act_dim={summary['action_dim']}"
    )
    print(
        f"  reward_mean={summary['reward_stats']['mean']:.6f}, "
        f"reward_std={summary['reward_stats']['std']:.6f}, "
        f"success_episode_fraction={summary['success_episode_fraction']:.6f}"
    )
    print(f"  sampled_points_for_plots={rew_sample.shape[0]}")


if __name__ == "__main__":
    main()
