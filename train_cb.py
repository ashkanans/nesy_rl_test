import argparse
import os
from pathlib import Path
import sys
from collections import deque
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).parent
sys.path.insert(0, str(REPO_ROOT / "trajectory-transformer"))
sys.path.insert(0, str(REPO_ROOT / "suffix-prediction"))

from trajectory.models.transformers import GPT
from dfa_adapter import TTDFAAdapter
from logic_loss_tt import LogicLossModule
from cb_dataset import CBSequenceDataset
from nrm_nav_dataset import NRMSafetySequenceDataset
from nrm_nav_env import NRMSafetyNavEnv
from DeepAutoma import DeepDFA
from FiniteStateMachine import DFA
import FiniteStateMachine as FSM

if torch.cuda.is_available():
    device = 'cuda:0'
else:
    device = 'cpu'


def build_product_dfa(dfas):
    """
    Build a product DFA that accepts only if all component DFAs accept.
    Assumes all DFAs share the same dictionary_symbols and num_of_symbols.
    """
    if not dfas:
        raise ValueError("No DFAs provided for product construction")

    dict_syms = dfas[0].dictionary_symbols
    num_syms = len(dict_syms)
    for d in dfas[1:]:
        if d.dictionary_symbols != dict_syms:
            raise ValueError("All DFAs must share the same dictionary_symbols for product")

    init_state = tuple(0 for _ in dfas)
    state_to_idx = {init_state: 0}
    transitions = {}
    acceptance = []
    queue = deque([init_state])

    def is_accept(state_tuple):
        return all(dfas[i].acceptance[s] for i, s in enumerate(state_tuple))

    while queue:
        state_tuple = queue.popleft()
        s_idx = state_to_idx[state_tuple]
        transitions[s_idx] = {}

        # acceptance list is indexed by s_idx
        if len(acceptance) <= s_idx:
            acceptance.append(is_accept(state_tuple))

        for sym in range(num_syms):
            next_tuple = []
            for i, d in enumerate(dfas):
                next_state = d.transitions[state_tuple[i]].get(sym, state_tuple[i])
                next_tuple.append(next_state)
            next_tuple = tuple(next_tuple)
            if next_tuple not in state_to_idx:
                state_to_idx[next_tuple] = len(state_to_idx)
                queue.append(next_tuple)
            transitions[s_idx][sym] = state_to_idx[next_tuple]

    product_dfa = DFA(transitions, acceptance, None, dictionary_symbols=dict_syms)
    return product_dfa


def build_adapter_and_dfa(args, dataset):
    """
    Build TTDFAAdapter + DeepDFA for Colour Bomb.

    We treat:
        - one observation dim with cardinality = env.n_states
        - one action dim with cardinality = env.action_space.n
        - one reward dim with 1 bin (always 0 for now)
        - one value dim  with 1 bin (always 0 for now)

    This yields an identity mapping for state/action tokens.
    """
    env = dataset.env
    obs_bins = env.observation_space.n
    act_bins = env.action_space.n
    rew_bins = 1
    val_bins = 1

    num_bins_per_dim = [obs_bins, act_bins, rew_bins, val_bins]

    adapter = TTDFAAdapter(
        observation_dim=dataset.observation_dim,
        action_dim=dataset.action_dim,
        num_bins=num_bins_per_dim,
        include_reward=True,
        include_value=True,
        constraint_dims=args.constraint_dims,
        abstraction_fn=None,
        use_stop_token=True,
    )

    formulas = []
    if args.ltl_formulas is not None:
        formulas = args.ltl_formulas
    elif args.ltl_formula is not None:
        formulas = [args.ltl_formula]
    else:
        raise ValueError("You must provide --ltl_formula or --ltl_formulas")

    dfas = [
        adapter.create_dfa_from_ltl(
            f, f"cb_constraint_{i}", use_safe_dfa=args.use_safe_dfa
        )
        for i, f in enumerate(formulas)
    ]

    if getattr(args, "inspect_dfa_only", False):
        print("=== DFA inspection (per-formula) ===")
        print(f"Number of formulas: {len(dfas)}")
        for i, d in enumerate(dfas):
            num_states = getattr(d, "num_of_states", None)
            num_symbols = len(getattr(d, "dictionary_symbols", []))
            num_accept = sum(getattr(d, "acceptance", []))
            print(
                f"  DFA {i}: states={num_states}, accepting_states={num_accept}, symbols={num_symbols}"
            )

    if len(dfas) == 1 or args.dfa_mode == "single":
        dfa = dfas[0]
        deep_dfa = dfa.return_deep_dfa()
        raw_dfa = dfa
    elif args.dfa_mode == "product":
        raw_dfa = build_product_dfa(dfas)
        deep_dfa = raw_dfa.return_deep_dfa()
    elif args.dfa_mode == "multi":
        deep_dfa = [d.return_deep_dfa() for d in dfas]
        raw_dfa = dfas
    else:
        raise ValueError(f"Unknown dfa_mode {args.dfa_mode}")

    if getattr(args, "inspect_dfa_only", False):
        # Report combined DFA size as well
        if isinstance(raw_dfa, list):
            print("=== DFA inspection (combined: multi) ===")
            for i, d in enumerate(raw_dfa):
                num_states = getattr(d, "num_of_states", None)
                num_symbols = len(getattr(d, "dictionary_symbols", []))
                num_accept = sum(getattr(d, "acceptance", []))
                print(
                    f"  DFA {i}: states={num_states}, accepting_states={num_accept}, symbols={num_symbols}"
                )
        else:
            print("=== DFA inspection (combined) ===")
            num_states = getattr(raw_dfa, "num_of_states", None)
            num_symbols = len(getattr(raw_dfa, "dictionary_symbols", []))
            num_accept = sum(getattr(raw_dfa, "acceptance", []))
            print(
                f"  DFA: states={num_states}, accepting_states={num_accept}, symbols={num_symbols}"
            )

    return adapter, deep_dfa, raw_dfa


def build_model(args, dataset, vocab_size):
    """
    Build a GPT model configured for the Colour Bomb dataset and adapter.
    """
    class Cfg:
        pass

    cfg = Cfg()
    cfg.vocab_size = vocab_size
    cfg.block_size = args.block_size
    cfg.n_layer = args.n_layer
    cfg.n_head = args.n_head
    cfg.n_embd = args.n_embd
    cfg.observation_dim = dataset.observation_dim
    cfg.action_dim = dataset.action_dim
    cfg.transition_dim = dataset.joined_dim
    cfg.action_weight = args.action_weight
    cfg.reward_weight = args.reward_weight
    cfg.value_weight = args.value_weight
    cfg.embd_pdrop = args.embd_pdrop
    cfg.resid_pdrop = args.resid_pdrop
    cfg.attn_pdrop = args.attn_pdrop

    model = GPT(cfg).to(device)
    return model


def build_dataset(args):
    if args.env == "cb":
        return CBSequenceDataset(
            num_episodes=args.num_episodes, max_steps=args.max_steps,
            sequence_length=args.block_size, discount=args.discount,
            stochastic=args.stochastic, seed=args.seed
        )
    elif args.env == "nrm_nav":
        return NRMSafetySequenceDataset(
            num_episodes=args.num_episodes, max_steps=args.max_steps,
            sequence_length=args.block_size, discount=args.discount,
            stochastic=args.stochastic, seed=args.seed,
            grid=None,
        )
    else:
        raise ValueError(f"Unknown env {args.env}")


def train(args, return_state=False):
    # control generic DFA end-state hack
    FSM.USE_END_HACK = not getattr(args, "no_end_state_hack", False)

    dataset = build_dataset(args)

    # Optional: replay a single dataset episode and exit.
    if getattr(args, "replay_dataset_episode", False):
        replay_dataset_episode(args, dataset)
        sys.exit(0)

    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, drop_last=True
    )

    adapter, deep_dfa, raw_dfa = build_adapter_and_dfa(args, dataset)

    # If requested, only inspect DFA sizes and exit before training.
    if getattr(args, "inspect_dfa_only", False):
        print("inspect_dfa_only flag set; skipping training and evaluation.")
        sys.exit(0)

    # Optional: analyze dataset in depth (including constraint satisfaction) and exit.
    if getattr(args, "analyze_dataset_only", False):
        analyze_dataset(args, dataset, adapter, raw_dfa)
        sys.exit(0)
    # GPT expects vocab_size without the extra stop token it appends internally
    model = build_model(args, dataset, vocab_size=adapter.num_token_ids - 1)

    logic = LogicLossModule(
        deep_dfa=deep_dfa,
        adapter=adapter,
        mode='global',
        num_samples=args.num_samples,
        temperature=args.temperature,
        alpha=args.alpha,
        append_end_symbol=getattr(args, "append_end_token_to_dfa", False),
        eps=getattr(args, "logic_eps", 1e-10),
        clamp_acceptance=not getattr(args, "no_logic_clamp", False),
    )

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        total_sup = 0.0
        total_log = 0.0
        n_batches = 0

        for batch in loader:
            batch = [b.to(device) for b in batch]
            loss, sup_loss, logic_loss = logic.compute_loss(
                model, batch, return_components=True
            )

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()

            total_loss += loss.item()
            total_sup += sup_loss.item()
            total_log += logic_loss.item()
            n_batches += 1

        print(
            "epoch %d | loss %.4f | sup %.4f | logic %.4f"
            % (
                epoch,
                total_loss / max(1, n_batches),
                total_sup / max(1, n_batches),
                total_log / max(1, n_batches),
            )
        )

        if args.save_path is not None:
            os.makedirs(args.save_path, exist_ok=True)
            ckpt_path = os.path.join(args.save_path, "cb_state_%d.pt" % epoch)
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": {
                        "vocab_size": adapter.num_token_ids,
                        "block_size": args.block_size,
                        "n_layer": args.n_layer,
                        "n_head": args.n_head,
                        "n_embd": args.n_embd,
                        "observation_dim": dataset.observation_dim,
                        "action_dim": dataset.action_dim,
                        "transition_dim": dataset.joined_dim,
                        "action_weight": args.action_weight,
                        "reward_weight": args.reward_weight,
                        "value_weight": args.value_weight,
                        "embd_pdrop": args.embd_pdrop,
                        "resid_pdrop": args.resid_pdrop,
                        "attn_pdrop": args.attn_pdrop
                    }
                },
                ckpt_path,
            )

    if return_state:
        return model, adapter, deep_dfa, dataset, raw_dfa


def _append_end_token(tensor_batch, adapter, append_flag):
    if not append_flag:
        return tensor_batch
    end_token = adapter.num_token_ids - 1
    end_col = torch.full((tensor_batch.shape[0], 1), end_token, device=tensor_batch.device, dtype=tensor_batch.dtype)
    return torch.cat([tensor_batch, end_col], dim=1)


def evaluate_model(model, adapter, dfa, dataset, batch_size=64, append_end_token=False):
    """
    Simple evaluation: supervised loss and DFA satisfaction rate on a dataset.

    For nrm_nav, also reports additional safety and reward metrics:
      - ground-truth vs predicted unsafe-state rates (using fixed unsafe IDs),
      - ground-truth DFA satisfaction rate,
      - dataset episode-level average return and unsafe episode rate.
    """
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    model.eval()
    total_loss = 0.0
    total_batches = 0
    total_sat = 0.0
    total_sat_gt = 0.0
    total_tokens = 0

    # unsafe statistics for nrm_nav (state IDs that are unsafe)
    track_unsafe = hasattr(dataset, "env") and isinstance(dataset.env, NRMSafetyNavEnv)
    if track_unsafe:
        unsafe_ids = {11, 18}
        gt_unsafe_count = 0
        pred_unsafe_count = 0
        state_token_count = 0

    with torch.no_grad():
        for batch in loader:
            x, y, mask = [b.to(device) for b in batch]
            logits, sup_loss = model(x, targets=y, mask=mask)
            preds = logits.argmax(dim=-1)
            if append_end_token:
                preds = _append_end_token(preds, adapter, True)
                x_eval = _append_end_token(x, adapter, True)
            else:
                x_eval = x
            if isinstance(dfa, (list, tuple)):
                sats = [adapter.batch_check_dfa_sat(preds, d) for d in dfa]
                sat = torch.stack(sats, dim=0).min(dim=0).values
                sats_gt = [adapter.batch_check_dfa_sat(x_eval, d) for d in dfa]
                sat_gt = torch.stack(sats_gt, dim=0).min(dim=0).values
            else:
                sat = adapter.batch_check_dfa_sat(preds, dfa)
                sat_gt = adapter.batch_check_dfa_sat(x_eval, dfa)

            total_loss += sup_loss.item()
            total_batches += 1
            total_sat += sat.sum().item()
            total_sat_gt += sat_gt.sum().item()
            total_tokens += sat.numel()

            if track_unsafe:
                # state positions are those where position % transition_dim == 0
                seq_len = x.shape[1]
                positions = torch.arange(seq_len, device=x.device)
                pos_mod = positions % adapter.transition_dim
                state_positions = pos_mod == 0

                x_states = x[:, state_positions]
                preds_states = preds[:, state_positions]

                gt_unsafe = torch.zeros_like(x_states, dtype=torch.bool)
                pred_unsafe = torch.zeros_like(preds_states, dtype=torch.bool)
                for uid in unsafe_ids:
                    gt_unsafe |= (x_states == uid)
                    pred_unsafe |= (preds_states == uid)

                gt_unsafe_count += gt_unsafe.sum().item()
                pred_unsafe_count += pred_unsafe.sum().item()
                state_token_count += x_states.numel()

    avg_loss = total_loss / max(1, total_batches)
    sat_rate = total_sat / max(1, total_tokens)
    sat_rate_gt = total_sat_gt / max(1, total_tokens)

    metrics = {
        "supervised_loss": avg_loss,
        "satisfaction_rate_pred": sat_rate,
        "satisfaction_rate_gt": sat_rate_gt,
    }

    if track_unsafe and state_token_count > 0:
        metrics["unsafe_rate_gt_states"] = gt_unsafe_count / state_token_count
        metrics["unsafe_rate_pred_states"] = pred_unsafe_count / state_token_count

        # episode-level dataset reward and unsafe episode rate (independent of model)
        # reconstruct episodes using a fresh env with the same config
        if hasattr(dataset, "episodes_tokens"):
            env_cfg = dataset.env.cfg
            eval_env = NRMSafetyNavEnv(env_cfg)
            returns = []
            unsafe_episodes = 0
            for ep_tokens in dataset.episodes_tokens:
                obs, _ = eval_env.reset()
                total_ret = 0.0
                terminal_info = {}
                # skip last row which is the special end token
                for row in ep_tokens[:-1]:
                    a = int(row[1])
                    obs, r, done, info = eval_env.step(a)
                    total_ret += r
                    if done:
                        terminal_info = info
                        break
                returns.append(total_ret)
                if terminal_info.get("terminal_type") == "X":
                    unsafe_episodes += 1

            if returns:
                metrics["dataset_avg_return"] = float(np.mean(returns))
                metrics["dataset_return_std"] = float(np.std(returns))
                metrics["dataset_unsafe_episode_rate"] = unsafe_episodes / len(returns)

    return metrics


def rollout_nrm_nav_policy(
    model,
    adapter,
    dfa,
    env_cfg,
    num_episodes=100,
    max_steps=None,
    greedy=True,
    append_end_token=False,
):
    """
    Roll out a policy induced by the model in a fresh NRMSafetyNavEnv and
    collect episode-level reward and safety statistics, as well as DFA
    satisfaction on the generated trajectories.

    The model is used autoregressively over token sequences; at each step,
    the next action is chosen from the model's logits at the last position.
    """

    env = NRMSafetyNavEnv(env_cfg)
    model.eval()

    if max_steps is None:
        max_steps = env_cfg.max_steps

    episode_returns = []
    episode_lengths = []
    episode_unsafe = []
    episode_sat = []

    cum_reward_traces = []

    with torch.no_grad():
        for _ in range(num_episodes):
            obs, _ = env.reset()
            # start history with initial state token
            history = torch.tensor([[int(obs)]], dtype=torch.long, device=device)
            tokens_this_ep = [int(obs)]

            cum_reward = 0.0
            cum_rewards_ts = []
            unsafe = False

            for _ in range(max_steps):
                # ensure history length does not exceed model block size
                if history.shape[1] > model.block_size:
                    idx = history[:, -model.block_size:]
                else:
                    idx = history

                logits, _ = model(idx)
                last_logits = logits[:, -1, :]

                # restrict to valid discrete actions
                n_actions = env.action_space.n
                action_logits = last_logits[:, :n_actions]
                if greedy:
                    a = int(torch.argmax(action_logits, dim=-1).item())
                else:
                    probs = torch.softmax(action_logits, dim=-1)
                    a = int(torch.multinomial(probs[0], num_samples=1).item())

                next_obs, r, done, info = env.step(a)
                cum_reward += r
                cum_rewards_ts.append(cum_reward)

                cost = 1 if info.get("terminal_type") == "X" else 0
                if cost == 1:
                    unsafe = True

                # append tokens matching dataset convention: [action, 0, cost, next_state]
                new_tokens = [a, 0, cost, int(next_obs)]
                tokens_this_ep.extend(new_tokens)

                new_tokens_tensor = torch.tensor(new_tokens, dtype=torch.long, device=device).view(1, -1)
                history = torch.cat([history, new_tokens_tensor], dim=1)

                if done:
                    break

            episode_returns.append(float(cum_reward))
            episode_lengths.append(len(cum_rewards_ts))
            episode_unsafe.append(unsafe)
            cum_reward_traces.append(cum_rewards_ts)

            # DFA satisfaction on generated trajectory tokens
            seq_tensor = torch.tensor(tokens_this_ep, dtype=torch.long, device=device).view(1, -1)
            if append_end_token:
                seq_tensor = _append_end_token(seq_tensor, adapter, True)
            if isinstance(dfa, (list, tuple)):
                sats = [adapter.batch_check_dfa_sat(seq_tensor, d) for d in dfa]
                sat = torch.stack(sats, dim=0).min(dim=0).values
                sat_flag = float(sat[0].item())
            else:
                sat = adapter.batch_check_dfa_sat(seq_tensor, dfa)
                sat_flag = float(sat[0].item())
            episode_sat.append(sat_flag)

    # aggregate metrics
    avg_return = float(np.mean(episode_returns)) if episode_returns else 0.0
    std_return = float(np.std(episode_returns)) if episode_returns else 0.0
    avg_len = float(np.mean(episode_lengths)) if episode_lengths else 0.0
    unsafe_rate = float(np.mean(episode_unsafe)) if episode_unsafe else 0.0
    sat_rate = float(np.mean(episode_sat)) if episode_sat else 0.0

    # average cumulative reward vs time step
    max_len = max((len(tr) for tr in cum_reward_traces), default=0)
    avg_cum_reward_vs_time = []
    for t in range(max_len):
        vals = [tr[t] for tr in cum_reward_traces if len(tr) > t]
        if vals:
            avg_cum_reward_vs_time.append(float(np.mean(vals)))

    rollout_metrics = {
        "num_episodes": num_episodes,
        "avg_return": avg_return,
        "std_return": std_return,
        "avg_episode_length": avg_len,
        "unsafe_episode_rate": unsafe_rate,
        "satisfaction_rate_rollout": sat_rate,
        "cum_reward_vs_time": avg_cum_reward_vs_time,
        "returns": episode_returns,
    }

    return rollout_metrics


def analyze_dataset(args, dataset, adapter, raw_dfa):
    """
    Analyze the offline dataset in depth and save statistics and plots.

    Works for both cb and nrm_nav. If LTL constraints are provided, also
    reports satisfaction rates under the corresponding DFA(s).
    """

    save_root = args.save_path or f"{args.env}_dataset_analysis"
    os.makedirs(save_root, exist_ok=True)
    out_dir = os.path.join(save_root, "dataset_analysis")
    os.makedirs(out_dir, exist_ok=True)

    summary = {}
    summary["env"] = args.env
    summary["num_segments"] = len(dataset)

    # episodes_tokens is a list of per-episode token arrays (including end row)
    episodes = getattr(dataset, "episodes_tokens", [])
    summary["num_episodes"] = len(episodes)

    if episodes:
        ep_lengths = [int(ep.shape[0] - 1) for ep in episodes]  # minus end row
        summary["episode_length_min"] = int(np.min(ep_lengths))
        summary["episode_length_max"] = int(np.max(ep_lengths))
        summary["episode_length_mean"] = float(np.mean(ep_lengths))

        # Episode length histogram
        plt.figure(figsize=(6, 4))
        plt.hist(ep_lengths, bins=20)
        plt.xlabel("Episode length (transitions)")
        plt.ylabel("Count")
        plt.title(f"{args.env} episode length distribution")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "episode_length_hist.png"))
        plt.close()

        # State ID histogram (from episodes, ignoring end row)
        state_ids = []
        for ep in episodes:
            state_ids.extend(ep[:-1, 0].tolist())
        plt.figure(figsize=(6, 4))
        plt.hist(state_ids, bins=dataset.env.observation_space.n)
        plt.xlabel("State ID")
        plt.ylabel("Count")
        plt.title(f"{args.env} state visitation histogram")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "state_hist.png"))
        plt.close()

        if args.env == "nrm_nav":
            # Cost histogram for nrm_nav (last column)
            costs = []
            for ep in episodes:
                costs.extend(ep[:-1, 3].tolist())
            plt.figure(figsize=(4, 4))
            plt.hist(costs, bins=[-0.5, 0.5, 1.5])
            plt.xticks([0, 1])
            plt.xlabel("Cost")
            plt.ylabel("Count")
            plt.title("Cost distribution")
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, "cost_hist.png"))
            plt.close()

    # Position-wise token stats on a subset of segments
    sample_limit = min(len(dataset), 500)
    pos_stats = {}
    for i in range(sample_limit):
        x, _, _ = dataset[i]
        x_np = x.numpy()
        for pos in range(dataset.joined_dim):
            vals = x_np[pos :: dataset.joined_dim]
            st = pos_stats.setdefault(
                pos, {"min": float("inf"), "max": float("-inf"), "values": []}
            )
            st["min"] = min(st["min"], float(vals.min()))
            st["max"] = max(st["max"], float(vals.max()))
            st["values"].extend(vals.tolist())

    for pos, st in pos_stats.items():
        vals = np.array(st["values"])
        st["mean"] = float(vals.mean()) if vals.size else None
        st["std"] = float(vals.std()) if vals.size else None
        st["unique_count"] = int(len(np.unique(vals))) if vals.size else 0
        # drop raw values to keep JSON small
        st.pop("values", None)
    summary["pos_token_stats"] = pos_stats

    # Constraint satisfaction using provided LTL formula(s)
    dfa_list = raw_dfa if isinstance(raw_dfa, list) else [raw_dfa]
    seg_sat_results = []
    ep_sat_results = []

    for idx, dfa in enumerate(dfa_list):
        # satisfaction on segments (dataset items)
        n_seg = min(len(dataset), 1000)
        seg_sats = []
        for i in range(n_seg):
            x, _, _ = dataset[i]
            x_eval = _append_end_token(x.unsqueeze(0), adapter, getattr(args, "append_end_token_to_dfa", False))
            sat = adapter.batch_check_dfa_sat(x_eval, dfa)
            seg_sats.append(float(sat[0].item()))
        seg_sat_results.append(
            {
                "dfa_index": idx,
                "mean_sat_segments": float(np.mean(seg_sats)) if seg_sats else None,
                "num_segments_checked": n_seg,
            }
        )

        # satisfaction on full episodes
        if episodes:
            n_eps = min(len(episodes), 1000)
            ep_sats = []
            for ei in range(n_eps):
                ep = episodes[ei]
                flat = torch.from_numpy(ep.astype(np.int64).reshape(-1))
                flat = _append_end_token(flat.unsqueeze(0), adapter, getattr(args, "append_end_token_to_dfa", False))
                sat = adapter.batch_check_dfa_sat(flat, dfa)
                ep_sats.append(float(sat[0].item()))
            ep_sat_results.append(
                {
                    "dfa_index": idx,
                    "mean_sat_episodes": float(np.mean(ep_sats)) if ep_sats else None,
                    "num_episodes_checked": n_eps,
                }
            )

    summary["constraint_satisfaction_segments"] = seg_sat_results
    summary["constraint_satisfaction_episodes"] = ep_sat_results

    # Save summary JSON
    summary_path = os.path.join(out_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Dataset analysis saved to: {summary_path}")


def replay_dataset_episode(args, dataset):
    """
    Visually replay a single offline dataset episode in the environment.

    Uses dataset.episodes_tokens[replay_episode_index] and the associated env.
    """

    episodes = getattr(dataset, "episodes_tokens", [])
    if not episodes:
        print("Dataset has no episodes_tokens; nothing to replay.")
        return

    ep_idx = getattr(args, "replay_episode_index", 0)
    if ep_idx < 0 or ep_idx >= len(episodes):
        print(f"Invalid replay_episode_index {ep_idx}; dataset has {len(episodes)} episodes.")
        return

    env = dataset.env
    ep_tokens = episodes[ep_idx]

    print(f"Replaying dataset episode {ep_idx} in env '{args.env}'")
    obs, _ = env.reset(seed=getattr(args, "seed", None))
    print("Initial observation (state index):", obs)
    if args.env == "cb":
        print(env.render(mode="ansi"))
    else:
        print(env.render())

    total_reward = 0.0
    for t, row in enumerate(ep_tokens[:-1]):  # skip end row
        state_token = int(row[0])
        action = int(row[1])

        obs, r, done, info = env.step(action)
        total_reward += r

        print(f"\nStep {t}:")
        print(f"  dataset_state_token={state_token}, action={action}, reward={r:.3f}, done={done}, info={info}")
        if args.env == "cb":
            print(env.render(mode="ansi"))
        else:
            print(env.render())

        if done:
            break

    print(f"\nEpisode replay finished. Total reward (replayed): {total_reward:.3f}")


def get_arg_parser(add_help=True):
    p = argparse.ArgumentParser(add_help=add_help)

    p.add_argument("--num_episodes", type=int, default=2000)
    p.add_argument("--max_steps", type=int, default=200)
    p.add_argument("--block_size", type=int, default=128)
    p.add_argument("--discount", type=float, default=0.99)
    p.add_argument("--stochastic", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--env", type=str, choices=["cb", "nrm_nav"], default="cb")

    p.add_argument("--n_layer", type=int, default=4)
    p.add_argument("--n_head", type=int, default=4)
    p.add_argument("--n_embd", type=int, default=64)
    p.add_argument("--embd_pdrop", type=float, default=0.1)
    p.add_argument("--resid_pdrop", type=float, default=0.1)
    p.add_argument("--attn_pdrop", type=float, default=0.1)

    p.add_argument("--action_weight", type=float, default=1.0)
    p.add_argument("--reward_weight", type=float, default=0.0)
    p.add_argument("--value_weight", type=float, default=0.0)

    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--grad_clip", type=float, default=1.0)

    p.add_argument("--ltl_formula", type=str, default=None)
    p.add_argument("--ltl_formulas", type=str, nargs="+", default=None, help="List of LTL formulas")
    p.add_argument("--dfa_mode", type=str, choices=["single", "product", "multi"], default="product",
                   help="How to combine multiple formulas: single (first only), product DFA, or multi (separate DFAs with averaged loss)")
    p.add_argument("--use_safe_dfa", action="store_true", help="Build simple safety DFA for G(!unsafe) formulas")
    p.add_argument("--constraint_dims", type=int, nargs="+", default=[0])

    p.add_argument("--num_samples", type=int, default=10)
    p.add_argument("--temperature", type=float, default=0.5)
    p.add_argument("--alpha", type=float, default=0.4)

    p.add_argument(
        "--logic_eps",
        type=float,
        default=1e-10,
        help="Epsilon used to clamp acceptance probabilities before log in logic loss; <=0 disables clamping.",
    )
    p.add_argument(
        "--no_logic_clamp",
        action="store_true",
        help="Disable epsilon clamp in logic loss (allow log(0) with -inf).",
    )

    p.add_argument(
        "--inspect_dfa_only",
        action="store_true",
        help=(
            "Build adapter and DFA(s), print their sizes, and exit "
            "without training or evaluation."
        ),
    )
    p.add_argument(
        "--no_end_state_hack",
        action="store_true",
        help="Disable the generic DFA end-state hack (keep original accepting states).",
    )
    p.add_argument(
        "--append_end_token_to_dfa",
        action="store_true",
        help=(
            "Append an 'end' token/symbol to sequences before feeding them to the DFA "
            "(useful if the DFA expects to see the end symbol to accept)."
        ),
    )

    p.add_argument(
        "--analyze_dataset_only",
        action="store_true",
        help=(
            "Build dataset, adapter and DFA(s), run an in-depth dataset "
            "analysis (including constraint satisfaction), save plots, and "
            "exit without training."
        ),
    )
    p.add_argument(
        "--replay_dataset_episode",
        action="store_true",
        help=(
            "Replay a single offline dataset episode step by step in the "
            "environment and exit without training."
        ),
    )
    p.add_argument(
        "--replay_episode_index",
        type=int,
        default=0,
        help="Index of the dataset episode to replay when --replay_dataset_episode is set.",
    )

    p.add_argument("--save_path", type=str, default="cb_runs")

    return p


def parse_args():
    return get_arg_parser().parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
