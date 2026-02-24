from types import SimpleNamespace

from dfa_adapter import TTDFAAdapter
from eval_runtime import apply_smoke_mode, spec_label_from_args, summarize_dfa_bundle


def test_apply_smoke_mode_caps_and_injects_default_spec_for_cb():
    args = SimpleNamespace(
        smoke=True,
        env="cb",
        num_episodes=999,
        max_steps=999,
        epochs=99,
        block_size=512,
        batch_size=256,
        n_layer=8,
        n_head=8,
        n_embd=512,
        eval_num_episodes=1000,
        beam_width=16,
        plan_horizon=8,
        spec=None,
        ltl_formula=None,
        ltl_formulas=None,
    )

    out = apply_smoke_mode(args)
    assert out.spec == "avoid_single_bomb_22"
    assert out.num_episodes <= 64
    assert out.max_steps <= 30
    assert out.epochs <= 1
    assert out.block_size <= 32
    assert out.batch_size <= 8
    assert out.n_layer <= 2
    assert out.n_head <= 2
    assert out.n_embd <= 64
    assert out.eval_num_episodes <= 16
    assert out.beam_width <= 4
    assert out.plan_horizon <= 2


def test_summarize_dfa_bundle_includes_spec_metadata():
    adapter = TTDFAAdapter(
        observation_dim=1,
        action_dim=0,
        num_bins=[8],
        include_reward=False,
        include_value=False,
        use_stop_token=True,
    )
    dfa = adapter.create_dfa_from_ltl("G(!(s0_bin1))", formula_name="x", use_safe_dfa=False)
    summary = summarize_dfa_bundle(
        dfa,
        spec_name="avoid_state_1",
        formulas=["G(!(s0_bin1))"],
        dfa_mode="single",
    )

    assert summary["spec"] == "avoid_state_1"
    assert summary["formulas"] == ["G(!(s0_bin1))"]
    assert summary["dfa_mode"] == "single"
    assert summary["num_states"] > 0


def test_spec_label_from_args_prefers_named_spec():
    args = SimpleNamespace(spec="avoid_unsafe", ltl_formulas=["F(s0_bin2)"], ltl_formula=None)
    assert spec_label_from_args(args) == "avoid_unsafe"


def test_apply_smoke_mode_sets_frozenlake_defaults():
    args = SimpleNamespace(
        smoke=True,
        env="frozenlake",
        num_episodes=9999,
        max_steps=999,
        epochs=10,
        block_size=128,
        batch_size=64,
        n_layer=4,
        n_head=4,
        n_embd=128,
        eval_num_episodes=99,
        beam_width=16,
        plan_horizon=8,
        policy_mix=0.6,
        spec=None,
        ltl_formula=None,
        ltl_formulas=None,
    )
    out = apply_smoke_mode(args)
    assert out.spec == "reach_goal_while_avoid_holes"
    assert out.num_episodes <= 200
    assert out.max_steps <= 30
    assert out.policy_mix == 0.0
