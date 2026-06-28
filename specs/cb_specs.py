"""Named LTLf specification presets for the ColourBomb environment."""

_BOMBS = "s0_bin22 | s0_bin27 | s0_bin43 | s0_bin78"
_GOALS = "s0_bin7 | s0_bin8 | s0_bin9 | s0_bin10 | s0_bin16 | s0_bin17 | s0_bin18 | s0_bin19 | s0_bin70 | s0_bin79"
_BLUE_GOALS = "s0_bin9 | s0_bin10 | s0_bin18 | s0_bin19"
_WHITE_GOALS = "s0_bin7 | s0_bin8 | s0_bin16 | s0_bin17"
_YELLOW = "s0_bin70 | s0_bin79"
# Non-terminal states in the yellow corridor (adjacent to yellow terminals).
_YELLOW_APPROACH = "s0_bin61 | s0_bin71 | s0_bin80"

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
    "reach_goal_while_avoid_bombs": {
        "description": (
            "Alias of reach_goal_while_safe: reach at least one goal state while "
            "always avoiding bombs."
        ),
        "formulas": [f"G(!({_BOMBS})) & F({_GOALS})"],
    },
    "memory_sequence_yellow": {
        "description": (
            "Reach yellow terminal while avoiding bombs (approximation; full Until requires Spot)."
        ),
        "formulas": [f"G(!({_BOMBS})) & F({_YELLOW})"],
    },
    "reach_blue_while_safe": {
        "description": "Reach the blue goal zone (top-left) while avoiding all bombs.",
        "formulas": [f"G(!({_BOMBS})) & F({_BLUE_GOALS})"],
    },
    "reach_white_goal_while_safe": {
        "description": "Reach the white-P goal zone (top-right) while avoiding all bombs.",
        "formulas": [f"G(!({_BOMBS})) & F({_WHITE_GOALS})"],
    },
    "reach_yellow_while_safe": {
        "description": "Reach the yellow goal zone (bottom-right) while avoiding all bombs.",
        "formulas": [f"G(!({_BOMBS})) & F({_YELLOW})"],
    },
    "avoid_top_bombs": {
        "description": "Avoid the two upper-area bombs (states 22 and 43) while being free to use lower routes.",
        "formulas": ["G(!(s0_bin22 | s0_bin43))"],
    },
    "avoid_bottom_bombs": {
        "description": "Avoid the two lower-area bombs (states 27 and 78) while being free to use upper routes.",
        "formulas": ["G(!(s0_bin27 | s0_bin78))"],
    },
    "reach_goal_avoid_top_bombs": {
        "description": "Reach any goal while avoiding the upper bombs only.",
        "formulas": [f"G(!(s0_bin22 | s0_bin43)) & F({_GOALS})"],
    },
    "reach_goal_avoid_bottom_bombs": {
        "description": "Reach any goal while avoiding the lower bombs only.",
        "formulas": [f"G(!(s0_bin27 | s0_bin78)) & F({_GOALS})"],
    },
    "avoid_upward_action": {
        "description": "Never take the UP action. Tests action-space constraint in CB.",
        "formulas": ["G(!(a0_bin0))"],
    },
}

SPECS_REQUIRING_SPOT = {
    "memory_sequence_yellow_full": {
        "description": "Full Until-based yellow sequencing. Requires Spot for correct DFA compilation.",
        "formulas": [f"G(!({_BOMBS})) & ((!({_YELLOW})) U ({_YELLOW_APPROACH})) & F({_YELLOW})"],
    },
}
