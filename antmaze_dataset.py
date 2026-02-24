from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class AntMazeDatasetBundle:
    env_name: str
    source: str
    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    terminals: np.ndarray
    timeouts: np.ndarray
    episode_returns: np.ndarray
    episode_lengths: np.ndarray
    normalized_scores: np.ndarray | None


def _try_make_env(env_name):
    import gym

    try:
        import d4rl  # noqa: F401
    except Exception:
        pass
    return gym.make(env_name)


def _split_episodes(rewards, terminals, timeouts):
    ep_returns = []
    ep_lengths = []
    running_ret = 0.0
    running_len = 0
    for r, term, tout in zip(rewards, terminals, timeouts):
        running_ret += float(r)
        running_len += 1
        if bool(term) or bool(tout):
            ep_returns.append(running_ret)
            ep_lengths.append(running_len)
            running_ret = 0.0
            running_len = 0
    if running_len > 0:
        ep_returns.append(running_ret)
        ep_lengths.append(running_len)
    return np.asarray(ep_returns, dtype=np.float32), np.asarray(ep_lengths, dtype=np.int64)


def _load_real_antmaze(env_name):
    env = _try_make_env(env_name)
    ds = env.get_dataset()
    rewards = np.asarray(ds["rewards"], dtype=np.float32)
    terminals = np.asarray(ds["terminals"], dtype=bool)
    if "timeouts" in ds:
        timeouts = np.asarray(ds["timeouts"], dtype=bool)
    else:
        timeouts = np.zeros_like(terminals, dtype=bool)

    ep_returns, ep_lengths = _split_episodes(rewards, terminals, timeouts)
    normalized = None
    if hasattr(env, "get_normalized_score") and len(ep_returns) > 0:
        normalized = np.asarray([env.get_normalized_score(float(x)) for x in ep_returns], dtype=np.float32)

    return AntMazeDatasetBundle(
        env_name=env_name,
        source="d4rl",
        observations=np.asarray(ds["observations"], dtype=np.float32),
        actions=np.asarray(ds["actions"], dtype=np.float32),
        rewards=rewards,
        terminals=terminals,
        timeouts=timeouts,
        episode_returns=ep_returns,
        episode_lengths=ep_lengths,
        normalized_scores=normalized,
    )


def _load_mock_antmaze(env_name, seed=0, num_episodes=32, max_steps=50):
    rng = np.random.RandomState(int(seed))
    obs_dim = 29
    act_dim = 8

    observations = []
    actions = []
    rewards = []
    terminals = []
    timeouts = []
    ep_returns = []
    ep_lengths = []

    for _ in range(int(num_episodes)):
        ep_len = int(rng.randint(max(2, max_steps // 2), max_steps + 1))
        ep_ret = 0.0
        for t in range(ep_len):
            observations.append(rng.normal(size=obs_dim).astype(np.float32))
            actions.append(rng.uniform(-1.0, 1.0, size=act_dim).astype(np.float32))
            r = float(rng.normal(loc=0.0, scale=0.1))
            ep_ret += r
            rewards.append(r)
            terminals.append(t == ep_len - 1)
            timeouts.append(False)
        ep_returns.append(ep_ret)
        ep_lengths.append(ep_len)

    return AntMazeDatasetBundle(
        env_name=env_name,
        source="mock",
        observations=np.asarray(observations, dtype=np.float32),
        actions=np.asarray(actions, dtype=np.float32),
        rewards=np.asarray(rewards, dtype=np.float32),
        terminals=np.asarray(terminals, dtype=bool),
        timeouts=np.asarray(timeouts, dtype=bool),
        episode_returns=np.asarray(ep_returns, dtype=np.float32),
        episode_lengths=np.asarray(ep_lengths, dtype=np.int64),
        normalized_scores=None,
    )


def load_antmaze_dataset(env_name, seed=0, allow_mock=True):
    try:
        return _load_real_antmaze(env_name)
    except Exception:
        if not allow_mock:
            raise
        return _load_mock_antmaze(env_name=env_name, seed=seed)
