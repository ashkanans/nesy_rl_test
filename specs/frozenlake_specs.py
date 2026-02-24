"""Placeholder LTLf presets for future FrozenLake runtime integration."""

SPECS = {
    "placeholder_safe": {
        "description": "Compilable placeholder formula for FrozenLake integration.",
        "formulas": ["G(!(s0_bin5))"],
    },
    "placeholder_goal": {
        "description": "Compilable placeholder eventual-goal formula.",
        "formulas": ["F(s0_bin15)"],
    },
}
