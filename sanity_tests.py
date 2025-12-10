import json
import torch

from colour_bomb import ColourBombGridworldV1Env
from cb_dataset import CBSequenceDataset
from nrm_nav_env import NRMSafetyNavEnv
from nrm_nav_dataset import NRMSafetySequenceDataset
from dfa_adapter import TTDFAAdapter
from FiniteStateMachine import DFA
from train_cb import build_product_dfa


def test_envs():
    cb = ColourBombGridworldV1Env()
    s0, _ = cb.reset(seed=0)
    assert cb._pos_to_state(cb.start_pos) == s0
    nav = NRMSafetyNavEnv()
    s1, _ = nav.reset(seed=0)
    assert nav._pos_to_state(nav.start_pos) == s1
    return {"cb_start": s0, "nav_start": s1}


def test_datasets():
    cb_ds = CBSequenceDataset(num_episodes=5, max_steps=5, sequence_length=8)
    nav_ds = NRMSafetySequenceDataset(num_episodes=5, max_steps=5, sequence_length=8)
    x, y, mask = cb_ds[0]
    x2, y2, mask2 = nav_ds[0]
    assert x.shape == y.shape == mask.shape
    assert x2.shape == y2.shape == mask2.shape
    xr = x.view(-1, 4)
    yr = y.view(-1, 4)
    assert (yr[:-1] == xr[1:]).all()
    xr2 = x2.view(-1, 4)
    yr2 = y2.view(-1, 4)
    assert (yr2[:-1] == xr2[1:]).all()
    return {
        "cb_len": len(cb_ds),
        "nav_len": len(nav_ds),
        "cb_seq_len": x.shape[0],
        "nav_seq_len": x2.shape[0],
        "cb_max_token": int(x.max().item()),
        "nav_max_token": int(x2.max().item()),
    }


def test_adapter_and_dfa():
    # small dummy adapter: 1 obs bin 3, 1 act bin 2, reward dim, value dim, stop token
    adapter = TTDFAAdapter(
        observation_dim=1,
        action_dim=1,
        num_bins=[3, 2, 1, 1],
        include_reward=True,
        include_value=True,
        use_stop_token=True,
    )
    # manual DFA: accept if first symbol is s0_bin0, otherwise reject
    sym_idx = adapter.symbolic_vocab.index("s0_bin0")
    num_syms = adapter.num_symbols
    transitions = {
        0: {s: (1 if s == sym_idx else 0) for s in range(num_syms)},
        1: {s: 1 for s in range(num_syms)},
    }
    acceptance = [False, True]
    dfa = DFA(transitions, acceptance, None, dictionary_symbols=adapter.symbolic_vocab)
    tokens = torch.tensor([[0]])  # first token s0_bin0
    sat = adapter.batch_check_dfa_sat(tokens, dfa)
    return {"adapter_num_tokens": adapter.num_token_ids, "sat": sat.item(), "tokens": tokens.tolist()}


def test_product_dfa():
    d1 = DFA({0: {0: 1}, 1: {0: 1}}, [False, True], None, dictionary_symbols=["a"])
    d2 = DFA({0: {0: 0}}, [True], None, dictionary_symbols=["a"])
    prod = build_product_dfa([d1, d2])
    return {"prod_states": prod.num_of_states, "prod_accept": prod.acceptance}


def main():
    results = {}
    results["envs"] = test_envs()
    results["datasets"] = test_datasets()
    results["adapter_dfa"] = test_adapter_and_dfa()
    results["product_dfa"] = test_product_dfa()
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
