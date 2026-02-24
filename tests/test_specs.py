from types import SimpleNamespace

import pytest

from dfa_adapter import TTDFAAdapter
from specs import SPEC_REGISTRY
from train_cb import resolve_formulas


def test_all_spec_files_exist_in_registry():
    for env_name in ["cb", "nrm_nav", "frozenlake", "antmaze"]:
        assert env_name in SPEC_REGISTRY
        assert len(SPEC_REGISTRY[env_name]) > 0


def test_all_spec_formulas_compile():
    adapter = TTDFAAdapter(
        observation_dim=1,
        action_dim=0,
        num_bins=[128],
        include_reward=False,
        include_value=False,
        use_stop_token=True,
    )

    for env_name, presets in SPEC_REGISTRY.items():
        for preset_name, cfg in presets.items():
            for idx, formula in enumerate(cfg["formulas"]):
                dfa = adapter.create_dfa_from_ltl(
                    formula, formula_name=f"{env_name}_{preset_name}_{idx}", use_safe_dfa=False
                )
                assert dfa.num_of_states > 0


def test_resolve_formulas_accepts_runtime_spec_for_cb():
    args = SimpleNamespace(
        env="cb",
        spec="avoid_single_bomb_22",
        ltl_formula=None,
        ltl_formulas=None,
    )
    formulas = resolve_formulas(args)
    assert formulas == ["G(!(s0_bin22))"]


def test_resolve_formulas_rejects_spec_plus_manual_formula():
    args = SimpleNamespace(
        env="cb",
        spec="avoid_single_bomb_22",
        ltl_formula="G(!(s0_bin5))",
        ltl_formulas=None,
    )
    with pytest.raises(ValueError, match="either --spec or --ltl_formula"):
        resolve_formulas(args)


def test_resolve_formulas_rejects_placeholder_env_runtime():
    args = SimpleNamespace(
        env="frozenlake",
        spec="placeholder_safe",
        ltl_formula=None,
        ltl_formulas=None,
    )
    with pytest.raises(ValueError, match="only available for cb/nrm_nav"):
        resolve_formulas(args)
