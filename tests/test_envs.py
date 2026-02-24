from colour_bomb import ColourBombGridworldV1Env
from frozenlake_env import FrozenLakeConfig, FrozenLakeEnv
from nrm_nav_env import NRMSafetyNavEnv


def test_colour_bomb_reset_starts_at_start_state():
    env = ColourBombGridworldV1Env()
    obs, _ = env.reset(seed=0)
    assert obs == env._pos_to_state(env.start_pos)


def test_nrm_nav_reset_starts_at_start_state():
    env = NRMSafetyNavEnv()
    obs, _ = env.reset(seed=0)
    assert obs == env._pos_to_state(env.start_pos)


def test_frozenlake_reset_starts_at_start_state():
    env = FrozenLakeEnv(FrozenLakeConfig(map_size="4x4", is_slippery=False, max_steps=30))
    obs, _ = env.reset(seed=0)
    assert obs == env._pos_to_state(env.start_pos)
