"""Specification preset registry and loader helpers."""

from specs.antmaze_specs import SPECS as ANTMAZE_SPECS
from specs.cb_specs import SPECS as CB_SPECS
from specs.frozenlake_specs import SPECS as FROZENLAKE_SPECS
from specs.nrm_nav_specs import SPECS as NRM_NAV_SPECS

SPEC_REGISTRY = {
    "cb": CB_SPECS,
    "nrm_nav": NRM_NAV_SPECS,
    "frozenlake": FROZENLAKE_SPECS,
    "antmaze": ANTMAZE_SPECS,
}


def get_spec(env, preset):
    if env not in SPEC_REGISTRY:
        raise ValueError(
            f"Unknown env '{env}' for specs. Available: {sorted(SPEC_REGISTRY.keys())}"
        )

    env_specs = SPEC_REGISTRY[env]
    if preset not in env_specs:
        raise ValueError(
            f"Unknown spec preset '{preset}' for env '{env}'. "
            f"Available: {sorted(env_specs.keys())}"
        )

    return env_specs[preset]
