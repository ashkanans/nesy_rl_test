"""Named LTLf specification presets for the NRM safety navigation environment."""

SPECS = {
    "avoid_unsafe": {
        "description": "Never visit unsafe cells X (state ids 11 and 18 in default grid).",
        "formulas": ["G(!(s0_bin11 | s0_bin18))"],
    },
    "avoid_state_11": {
        "description": "Never visit unsafe state 11.",
        "formulas": ["G(!(s0_bin11))"],
    },
    "reach_goal": {
        "description": "Eventually reach the goal cell.",
        "formulas": ["F(s0_bin4)"],
    },
    "reach_goal_while_avoid_unsafe": {
        "description": "Reach the goal while always avoiding unsafe cells.",
        "formulas": ["G(!(s0_bin11 | s0_bin18)) & F(s0_bin4)"],
    },
}
