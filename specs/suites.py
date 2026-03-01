"""Suite registry for multi-spec evaluation protocols."""

from __future__ import annotations

from specs import SPEC_REGISTRY

SUITE_REGISTRY = {
    "cb": {
        "v1": [
            "avoid_single_bomb_22",
            "avoid_bombs",
            "reach_goal_while_safe",
            "memory_sequence_yellow",
        ]
    },
    "frozenlake": {
        "v1": [
            "avoid_holes",
            "reach_goal",
            "reach_goal_while_avoid_holes",
        ]
    },
    "nrm_nav": {
        "v1": [
            "avoid_state_11",
            "avoid_unsafe",
        ]
    },
    "dsrl": {
        "v1": [
            "avoid_unsafe",
            "reach_goal",
            "reach_goal_while_avoid_unsafe",
        ]
    },
}


def get_suite(env: str, suite: str) -> list[str]:
    if env not in SUITE_REGISTRY:
        raise ValueError(
            f"Unknown env '{env}' for suite registry. Available: {sorted(SUITE_REGISTRY.keys())}"
        )
    env_suites = SUITE_REGISTRY[env]
    if suite not in env_suites:
        raise ValueError(
            f"Unknown suite '{suite}' for env '{env}'. Available: {sorted(env_suites.keys())}"
        )
    presets = list(env_suites[suite])
    missing = [p for p in presets if p not in SPEC_REGISTRY.get(env, {})]
    if missing:
        raise ValueError(
            f"Suite '{suite}' for env '{env}' references unknown presets: {missing}"
        )
    return presets
