"""Named LTLf specification presets for the ColourBomb environment."""

_BOMBS = "s0_bin22 | s0_bin27 | s0_bin43 | s0_bin78"
_GOALS = "s0_bin7 | s0_bin8 | s0_bin9 | s0_bin10 | s0_bin16 | s0_bin17 | s0_bin18 | s0_bin19 | s0_bin70 | s0_bin79"

SPECS = {
    "avoid_bombs": {
        "description": "Invariant safety: never visit known bomb states.",
        "formulas": [f"G(!({_BOMBS}))"],
    },
    "avoid_single_bomb_22": {
        "description": "Never visit bomb state 22.",
        "formulas": ["G(!(s0_bin22))"],
    },
    "reach_goal_while_safe": {
        "description": "Reach at least one goal state while always avoiding bombs.",
        "formulas": [f"G(!({_BOMBS})) & F({_GOALS})"],
    },
    "memory_sequence_yellow": {
        "description": (
            "Weaker memory-style sequencing proxy over existing state propositions."
        ),
        "formulas": [f"G(!({_BOMBS})) & F(s0_bin70) & F(s0_bin79)"],
    },
}
