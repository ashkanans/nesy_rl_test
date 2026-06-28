from types import SimpleNamespace
from collections import deque

import pytest

from dfa_adapter import TTDFAAdapter
from specs import SPEC_REGISTRY
from specs.cb_specs import SPECS_REQUIRING_SPOT
from train_cb import resolve_formulas


def test_all_spec_files_exist_in_registry():
    for env_name in ["cb", "nrm_nav", "frozenlake", "antmaze", "dsrl"]:
        assert env_name in SPEC_REGISTRY
        assert len(SPEC_REGISTRY[env_name]) > 0


def test_all_spec_formulas_compile():
    for env_name, presets in SPEC_REGISTRY.items():
        adapter = TTDFAAdapter(
            observation_dim=1,
            action_dim=1,
            num_bins=[128, 16, 2, 2],
            include_reward=True,
            include_value=True,
            use_stop_token=True,
        )
        for preset_name, cfg in presets.items():
            for idx, formula in enumerate(cfg["formulas"]):
                dfa = adapter.create_dfa_from_ltl(
                    formula, formula_name=f"{env_name}_{preset_name}_{idx}", use_safe_dfa=False
                )
                assert dfa.num_of_states > 0


def _has_reachable_accepting_state(dfa) -> bool:
    visited = set()
    queue = deque([0])
    while queue:
        state = queue.popleft()
        if state in visited:
            continue
        visited.add(state)
        if state < len(dfa.acceptance) and bool(dfa.acceptance[state]):
            return True
        for nxt in dfa.transitions.get(state, {}).values():
            if nxt not in visited:
                queue.append(nxt)
    return False


def test_all_spec_dfas_have_reachable_accepting_state():
    for env_name, presets in SPEC_REGISTRY.items():
        adapter = TTDFAAdapter(
            observation_dim=1,
            action_dim=1,
            num_bins=[128, 16, 2, 2],
            include_reward=True,
            include_value=True,
            use_stop_token=True,
        )
        for preset_name, cfg in presets.items():
            for idx, formula in enumerate(cfg["formulas"]):
                dfa = adapter.create_dfa_from_ltl(
                    formula, formula_name=f"{env_name}_{preset_name}_{idx}", use_safe_dfa=False
                )
                assert _has_reachable_accepting_state(dfa), (
                    f"DFA for {env_name}/{preset_name}[{idx}] has no reachable accepting state. "
                    f"Formula: {formula!r}"
                )


def test_resolve_formulas_accepts_runtime_spec_for_cb():
    args = SimpleNamespace(
        env="cb",
        spec="avoid_single_bomb_22",
        ltl_formula=None,
        ltl_formulas=None,
    )
    formulas = resolve_formulas(args)
    assert formulas == ["G(!(s0_bin22))"]


def test_cb_reach_goal_while_avoid_bombs_alias_matches_reach_goal_while_safe():
    cb_specs = SPEC_REGISTRY["cb"]
    assert "reach_goal_while_safe" in cb_specs
    assert "reach_goal_while_avoid_bombs" in cb_specs
    assert (
        cb_specs["reach_goal_while_avoid_bombs"]["formulas"]
        == cb_specs["reach_goal_while_safe"]["formulas"]
    )


def test_spot_required_specs_not_in_registry():
    for name in SPECS_REQUIRING_SPOT:
        assert name not in SPEC_REGISTRY.get("cb", {})


def test_resolve_formulas_rejects_spec_plus_manual_formula():
    args = SimpleNamespace(
        env="cb",
        spec="avoid_single_bomb_22",
        ltl_formula="G(!(s0_bin5))",
        ltl_formulas=None,
    )
    with pytest.raises(ValueError, match="either --spec or --ltl_formula"):
        resolve_formulas(args)


def test_resolve_formulas_accepts_runtime_spec_for_frozenlake():
    args = SimpleNamespace(
        env="frozenlake",
        spec="avoid_holes",
        ltl_formula=None,
        ltl_formulas=None,
        frozenlake_use_position_props=False,
    )
    formulas = resolve_formulas(args)
    assert len(formulas) == 1
    assert "s0_bin" in formulas[0]
