"""Named LTLf specification presets for DSRL-derived datasets."""

SPECS = {
    "avoid_unsafe": {
        "description": "Invariant safety over dataset cost signal (cost token must stay 0).",
        "formulas": ["G(!(v_bin1))"],
    },
    "reach_goal": {
        "description": "Eventually obtain a positive-reward transition (goal token).",
        "formulas": ["F(r_bin1)"],
    },
    "reach_goal_while_avoid_unsafe": {
        "description": "Reach a positive-reward transition while keeping cost token at 0.",
        "formulas": ["G(!(v_bin1)) & F(r_bin1)"],
    },
}
