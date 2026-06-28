"""FrozenLake LTLf presets with runtime expansion over hole/goal state ids."""

DEFAULT_HOLE_IDS_4x4 = [5, 7, 11, 12]
DEFAULT_GOAL_IDS_4x4 = [15]


def _or_states(state_ids):
    return " | ".join(f"s0_bin{int(s)}" for s in state_ids)


def build_frozenlake_formulas(preset, hole_state_ids, goal_state_ids, include_position_props=False):
    del include_position_props  # position bins are already encoded as s0_bin<state_id>
    holes = list(sorted(int(s) for s in hole_state_ids))
    goals = list(sorted(int(s) for s in goal_state_ids))
    if not holes or not goals:
        raise ValueError("FrozenLake formulas require non-empty hole and goal state ids.")

    hole_expr = _or_states(holes)
    goal_expr = _or_states(goals)

    if preset == "avoid_holes":
        return [f"G(!({hole_expr}))"]
    if preset == "reach_goal":
        return [f"F({goal_expr})"]
    if preset == "reach_goal_while_avoid_holes":
        return [f"G(!({hole_expr})) & F({goal_expr})"]
    if preset == "bounded_reach_goal_while_avoid_holes":
        # Weak bounded-like proxy: avoid holes and enforce eventually goal within finite trace.
        return [f"G(!({hole_expr})) & F({goal_expr})"]
    if preset == "never_move_left":
        return ["G(!(a0_bin0))"]
    if preset == "reach_goal_no_left":
        return [f"G(!(a0_bin0)) & F({goal_expr})"]
    raise ValueError(f"Unknown FrozenLake preset '{preset}'")


SPECS = {
    "avoid_holes": {
        "description": "Invariant safety: never step into a hole.",
        "formulas": build_frozenlake_formulas(
            "avoid_holes", DEFAULT_HOLE_IDS_4x4, DEFAULT_GOAL_IDS_4x4
        ),
        "propositions": {
            "hole": [f"s0_bin{x}" for x in DEFAULT_HOLE_IDS_4x4],
            "goal": [f"s0_bin{x}" for x in DEFAULT_GOAL_IDS_4x4],
        },
    },
    "reach_goal": {
        "description": "Eventually reach the goal.",
        "formulas": build_frozenlake_formulas(
            "reach_goal", DEFAULT_HOLE_IDS_4x4, DEFAULT_GOAL_IDS_4x4
        ),
        "propositions": {
            "hole": [f"s0_bin{x}" for x in DEFAULT_HOLE_IDS_4x4],
            "goal": [f"s0_bin{x}" for x in DEFAULT_GOAL_IDS_4x4],
        },
    },
    "reach_goal_while_avoid_holes": {
        "description": "Reach the goal while always avoiding holes.",
        "formulas": build_frozenlake_formulas(
            "reach_goal_while_avoid_holes", DEFAULT_HOLE_IDS_4x4, DEFAULT_GOAL_IDS_4x4
        ),
        "propositions": {
            "hole": [f"s0_bin{x}" for x in DEFAULT_HOLE_IDS_4x4],
            "goal": [f"s0_bin{x}" for x in DEFAULT_GOAL_IDS_4x4],
        },
    },
    "bounded_reach_goal_while_avoid_holes": {
        "description": "Finite-horizon proxy for bounded reach-while-safe.",
        "formulas": build_frozenlake_formulas(
            "bounded_reach_goal_while_avoid_holes", DEFAULT_HOLE_IDS_4x4, DEFAULT_GOAL_IDS_4x4
        ),
        "propositions": {
            "hole": [f"s0_bin{x}" for x in DEFAULT_HOLE_IDS_4x4],
            "goal": [f"s0_bin{x}" for x in DEFAULT_GOAL_IDS_4x4],
        },
    },
    "never_move_left": {
        "description": "Never take the LEFT action (a0_bin0). Tests action-level constraint without state grounding.",
        "formulas": build_frozenlake_formulas(
            "never_move_left", DEFAULT_HOLE_IDS_4x4, DEFAULT_GOAL_IDS_4x4
        ),
        "propositions": {
            "hole": [f"s0_bin{x}" for x in DEFAULT_HOLE_IDS_4x4],
            "goal": [f"s0_bin{x}" for x in DEFAULT_GOAL_IDS_4x4],
        },
    },
    "reach_goal_no_left": {
        "description": "Reach the goal using only RIGHT, DOWN, and UP. Feasible on the default 4x4 and 8x8 maps.",
        "formulas": build_frozenlake_formulas(
            "reach_goal_no_left", DEFAULT_HOLE_IDS_4x4, DEFAULT_GOAL_IDS_4x4
        ),
        "propositions": {
            "hole": [f"s0_bin{x}" for x in DEFAULT_HOLE_IDS_4x4],
            "goal": [f"s0_bin{x}" for x in DEFAULT_GOAL_IDS_4x4],
        },
    },
}
