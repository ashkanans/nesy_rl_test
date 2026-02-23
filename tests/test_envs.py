from colour_bomb import ColourBombGridworldV1Env
from nrm_nav_env import NRMSafetyNavEnv


def test_colour_bomb_reset_starts_at_start_state():
    env = ColourBombGridworldV1Env()
    obs, _ = env.reset(seed=0)
    assert obs == env._pos_to_state(env.start_pos)


def test_nrm_nav_reset_starts_at_start_state():
    env = NRMSafetyNavEnv()
    obs, _ = env.reset(seed=0)
    assert obs == env._pos_to_state(env.start_pos)
