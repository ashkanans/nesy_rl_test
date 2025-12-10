import argparse
import os
from pathlib import Path
import sys
from collections import deque
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).parent
sys.path.insert(0, str(REPO_ROOT / "trajectory-transformer"))
sys.path.insert(0, str(REPO_ROOT / "suffix-prediction"))

from trajectory.models.transformers import GPT
from dfa_adapter import TTDFAAdapter
from logic_loss_tt import LogicLossModule
from cb_dataset import CBSequenceDataset
from DeepAutoma import DeepDFA
from FiniteStateMachine import DFA

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
        adapter.create_dfa_from_ltl(f, f"cb_constraint_{i}") for i, f in enumerate(formulas)
    ]

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


def train(args, return_state=False):
    dataset = CBSequenceDataset(
        num_episodes=args.num_episodes, max_steps=args.max_steps,
        sequence_length=args.block_size, discount=args.discount,
        stochastic=args.stochastic, seed=args.seed
    )

    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, drop_last=True
    )

    adapter, deep_dfa, raw_dfa = build_adapter_and_dfa(args, dataset)
    # GPT expects vocab_size without the extra stop token it appends internally
    model = build_model(args, dataset, vocab_size=adapter.num_token_ids - 1)

    logic = LogicLossModule(
        deep_dfa=deep_dfa, adapter=adapter, mode='global',
        num_samples=args.num_samples, temperature=args.temperature, alpha=args.alpha,
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


def evaluate_model(model, adapter, dfa, dataset, batch_size=64):
    """
    Simple evaluation: supervised loss and DFA satisfaction rate on a dataset.
    """
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    model.eval()
    total_loss = 0.0
    total_batches = 0
    total_sat = 0.0
    total_tokens = 0

    with torch.no_grad():
        for batch in loader:
            x, y, mask = [b.to(device) for b in batch]
            logits, sup_loss = model(x, targets=y, mask=mask)
            preds = logits.argmax(dim=-1)
            if isinstance(dfa, (list, tuple)):
                sats = [adapter.batch_check_dfa_sat(preds, d) for d in dfa]
                sat = torch.stack(sats, dim=0).min(dim=0).values
            else:
                sat = adapter.batch_check_dfa_sat(preds, dfa)

            total_loss += sup_loss.item()
            total_batches += 1
            total_sat += sat.sum().item()
            total_tokens += sat.numel()

    avg_loss = total_loss / max(1, total_batches)
    sat_rate = total_sat / max(1, total_tokens)
    return {"supervised_loss": avg_loss, "satisfaction_rate": sat_rate}


def get_arg_parser(add_help=True):
    p = argparse.ArgumentParser(add_help=add_help)

    p.add_argument("--num_episodes", type=int, default=2000)
    p.add_argument("--max_steps", type=int, default=200)
    p.add_argument("--block_size", type=int, default=128)
    p.add_argument("--discount", type=float, default=0.99)
    p.add_argument("--stochastic", action="store_true")
    p.add_argument("--seed", type=int, default=0)

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
    p.add_argument("--constraint_dims", type=int, nargs="+", default=[0])

    p.add_argument("--num_samples", type=int, default=10)
    p.add_argument("--temperature", type=float, default=0.5)
    p.add_argument("--alpha", type=float, default=0.4)

    p.add_argument("--save_path", type=str, default="cb_runs")

    return p


def parse_args():
    return get_arg_parser().parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
