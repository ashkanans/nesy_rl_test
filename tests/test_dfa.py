import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "suffix-prediction"))

from FiniteStateMachine import DFA

from dfa_adapter import TTDFAAdapter
from train_cb import build_product_dfa


def test_adapter_manual_dfa_accepts_expected_token():
    adapter = TTDFAAdapter(
        observation_dim=1,
        action_dim=1,
        num_bins=[3, 2, 1, 1],
        include_reward=True,
        include_value=True,
        use_stop_token=True,
    )

    sym_idx = adapter.symbolic_vocab.index("s0_bin0")
    num_syms = adapter.num_symbols
    transitions = {
        0: {s: (1 if s == sym_idx else 0) for s in range(num_syms)},
        1: {s: 1 for s in range(num_syms)},
    }
    acceptance = [False, True]
    dfa = DFA(transitions, acceptance, None, dictionary_symbols=adapter.symbolic_vocab)

    tokens = torch.tensor([[0]])
    sat = adapter.batch_check_dfa_sat(tokens, dfa)
    assert sat.item() == 1.0


def test_product_dfa_building():
    dfa_1 = DFA({0: {0: 1}, 1: {0: 1}}, [False, True], None, dictionary_symbols=["a"])
    dfa_2 = DFA({0: {0: 0}}, [True], None, dictionary_symbols=["a"])

    product = build_product_dfa([dfa_1, dfa_2])
    assert product.num_of_states == 2
    assert product.acceptance == [False, True]
