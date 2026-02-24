"""Placeholder LTLf presets for future AntMaze runtime integration."""

SPECS = {
    "placeholder_safe": {
        "description": "Compilable placeholder safety formula for AntMaze integration.",
        "formulas": ["G(!(s0_bin1))"],
    },
    "placeholder_reach": {
        "description": "Compilable placeholder eventual target formula.",
        "formulas": ["F(s0_bin2)"],
    },
}
