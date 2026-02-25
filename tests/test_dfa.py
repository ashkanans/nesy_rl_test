import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "suffix-prediction"))

from FiniteStateMachine import DFA

from dfa_adapter import TTDFAAdapter
from train_cb import build_product_dfa


def _build_manual_test_dfa():
    adapter = TTDFAAdapter(
        observation_dim=1,
        action_dim=1,
        num_bins=[3, 2, 1, 1],
        include_reward=True,
        include_value=True,
        use_stop_token=True,
    )

    end_idx = adapter.symbolic_vocab.index("end")
    sym_idx = adapter.symbolic_vocab.index("s0_bin0")
    num_syms = adapter.num_symbols
    transitions = {
        0: {s: (1 if s == sym_idx else 0) for s in range(num_syms)},
        1: {s: 1 for s in range(num_syms)},
    }
    transitions[0][end_idx] = 0
    transitions[1][end_idx] = 1
    acceptance = [False, True]
    dfa = DFA(transitions, acceptance, None, dictionary_symbols=adapter.symbolic_vocab)
    return adapter, dfa


def test_adapter_manual_dfa_accepts_expected_token():
    adapter, dfa = _build_manual_test_dfa()
    tokens = torch.tensor([[0, adapter.end_token_id]])
    sat = adapter.check_sat_token_ids(tokens, dfa)
    assert sat.item() is True


def test_missing_end_is_unsatisfied_in_canonical_mode():
    adapter, dfa = _build_manual_test_dfa()
    tokens = torch.tensor([[0]])
    sat = adapter.check_sat_token_ids(tokens, dfa)
    assert sat.item() is False


def test_token_prob_one_hot_matches_token_id_path():
    adapter, dfa = _build_manual_test_dfa()
    deep_dfa = dfa.return_deep_dfa()
    token_ids = torch.tensor([[0, adapter.end_token_id]], dtype=torch.long)

    hard_sat = adapter.check_sat_token_ids(token_ids, dfa).float()

    token_probs = torch.nn.functional.one_hot(token_ids, num_classes=adapter.num_token_ids).float()
    symbol_probs = adapter.token_probs_to_symbol_probs(token_probs)
    soft_sat = adapter.check_sat_symbol_probs(symbol_probs, deep_dfa)
    assert torch.allclose(soft_sat, hard_sat, atol=1e-6, rtol=1e-6)


def test_product_dfa_building():
    dfa_1 = DFA({0: {0: 1}, 1: {0: 1}}, [False, True], None, dictionary_symbols=["a"])
    dfa_2 = DFA({0: {0: 0}}, [True], None, dictionary_symbols=["a"])

    product = build_product_dfa([dfa_1, dfa_2])
    assert product.num_of_states == 2
    assert product.acceptance == [False, True]


def test_safe_dfa_parser_uses_exact_symbol_matching():
    adapter = TTDFAAdapter(
        observation_dim=1,
        action_dim=1,
        num_bins=[16, 4, 2, 2],
        include_reward=True,
        include_value=True,
        use_stop_token=True,
    )
    formula = "G(!(s0_bin5 | s0_bin7 | s0_bin11 | s0_bin12))"
    parsed = set(adapter._parse_unsafe_symbols(formula))

    assert "s0_bin5" in parsed
    assert "s0_bin7" in parsed
    assert "s0_bin11" in parsed
    assert "s0_bin12" in parsed
    assert "s0_bin1" not in parsed
