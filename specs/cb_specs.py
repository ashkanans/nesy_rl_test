"""Named LTLf specification presets for the ColourBomb environment."""

SPECS = {
    "avoid_bombs": {
        "description": "Never visit known bomb states.",
        "formulas": ["G(!(s0_bin22 | s0_bin27 | s0_bin43 | s0_bin78))"],
    },
    "avoid_single_bomb_22": {
        "description": "Never visit bomb state 22.",
        "formulas": ["G(!(s0_bin22))"],
    },
}
