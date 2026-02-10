import argparse
import json
import socketserver
from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np

from common import Model
from value_net import DoubleCritic


def _load_config(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def _load_norm_stats(path: str) -> dict:
    stats = np.load(path)
    return {
        "obs_mean": stats["obs_mean"],
        "obs_std": stats["obs_std"],
        "act_mean": stats["act_mean"],
        "act_std": stats["act_std"],
    }


def _build_critic(checkpoint_dir: str) -> Tuple[Model, dict]:
    config = _load_config(f"{checkpoint_dir}/config.json")
    stats = _load_norm_stats(f"{checkpoint_dir}/normalization_stats.npz")

    obs_dim = stats["obs_mean"].shape[0]
    act_dim = stats["act_mean"].shape[0]
    hidden_dims = tuple(config.get("hidden_dims", (256, 256)))

    critic_def = DoubleCritic(hidden_dims)
    dummy_obs = jnp.zeros((1, obs_dim), dtype=jnp.float32)
    dummy_act = jnp.zeros((1, act_dim), dtype=jnp.float32)
    critic = Model.create(critic_def, inputs=[jax.random.PRNGKey(0), dummy_obs, dummy_act])
    critic = critic.load(f"{checkpoint_dir}/critic.pkl")

    return critic, stats


def _make_q_fn(critic: Model):
    @jax.jit
    def _q_min(obs: jnp.ndarray, acts: jnp.ndarray) -> jnp.ndarray:
        q1, q2 = critic.apply({"params": critic.params}, obs, acts)
        return jnp.minimum(q1, q2)

    return _q_min


class QServerHandler(socketserver.StreamRequestHandler):
    q_fn = None
    normalize = False
    stats = None

    def handle(self):
        for line in self.rfile:
            if not line:
                break
            try:
                req = json.loads(line.decode("utf-8"))
                obs = np.asarray(req["obs"], dtype=np.float32)
                acts = np.asarray(req["acts"], dtype=np.float32)
                if obs.ndim == 1:
                    obs = obs[None, :]
                if acts.ndim == 1:
                    acts = acts[None, :]
                if self.normalize:
                    obs = (obs - self.stats["obs_mean"]) / self.stats["obs_std"]
                    acts = (acts - self.stats["act_mean"]) / self.stats["act_std"]
                q = np.asarray(self.q_fn(obs, acts)).reshape(-1).tolist()
                resp = {"q": q}
            except Exception as e:
                resp = {"error": str(e)}
            self.wfile.write((json.dumps(resp) + "\n").encode("utf-8"))
            self.wfile.flush()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        required=True,
        help="Path to IQL checkpoint dir containing critic.pkl and normalization_stats.npz",
    )
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Apply obs/action normalization using normalization_stats.npz",
    )
    args = parser.parse_args()

    critic, stats = _build_critic(args.checkpoint_dir)
    q_fn = _make_q_fn(critic)

    QServerHandler.q_fn = q_fn
    QServerHandler.normalize = args.normalize
    QServerHandler.stats = stats

    with socketserver.ThreadingTCPServer((args.host, args.port), QServerHandler) as server:
        server.allow_reuse_address = True
        print(f"[q_server] listening on {args.host}:{args.port}")
        server.serve_forever()


if __name__ == "__main__":
    main()
