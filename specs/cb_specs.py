"""Named LTLf specification presets for the ColourBomb environment."""

_BOMBS = "s0_bin22 | s0_bin27 | s0_bin43 | s0_bin78"
_GOALS = "s0_bin7 | s0_bin8 | s0_bin9 | s0_bin10 | s0_bin16 | s0_bin17 | s0_bin18 | s0_bin19 | s0_bin70 | s0_bin79"
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
            "Memory-style yellow-corridor sequencing proxy that is feasible under terminal-goal dynamics."
        ),
        # Prior formulation required visiting both yellow terminal states in one episode,
        # which is infeasible because hitting any goal state terminates the episode.
        # This replacement enforces: avoid bombs, visit a yellow-approach state first,
        # then eventually reach (at least one) yellow terminal.
        "formulas": [f"G(!({_BOMBS})) & ((!({_YELLOW})) U ({_YELLOW_APPROACH})) & F({_YELLOW})"],
    },
}
