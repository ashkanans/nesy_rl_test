import numpy as np

from cb_dataset import CBSequenceDataset
from frozenlake_dataset import FrozenLakeSequenceDataset
from logic.token_schema import (
    get_schema_for_env,
    validate_episode_tokens,
)
from nrm_nav_dataset import NRMSafetySequenceDataset


def test_schema_field_indices_are_stable():
    cb = get_schema_for_env("cb")
    nrm = get_schema_for_env("nrm_nav")
    fl = get_schema_for_env("frozenlake")

    assert cb.schema_id == "cb_v1"
    assert cb.field_index("state") == 0
    assert cb.field_index("action") == 1
    assert cb.field_index("reward") == 2
    assert cb.field_index("aux") == 3

    assert nrm.schema_id == "nrm_nav_v1"
    assert nrm.field_index("state") == 0
    assert nrm.field_index("action") == 1
    assert nrm.field_index("reward") == 2
    assert nrm.field_index("safety_cost") == 3

    assert fl.schema_id == "frozenlake_v1"
    assert fl.field_index("state") == 0
    assert fl.field_index("action") == 1
    assert fl.field_index("reward") == 2
    assert fl.field_index("safety_cost") == 3


def test_cb_dataset_matches_schema_contract():
    dataset = CBSequenceDataset(num_episodes=4, max_steps=6, sequence_length=8, seed=0)
    schema = get_schema_for_env("cb")

    for episode in dataset.episodes_tokens:
        assert episode.shape[1] == schema.width
        assert np.issubdtype(episode.dtype, np.integer)
        validate_episode_tokens(
            episode,
            schema=schema,
            end_token_id=dataset.end_token_id,
            observation_space_n=dataset.env.observation_space.n,
            action_space_n=dataset.env.action_space.n,
        )


def test_nrm_dataset_matches_schema_contract():
    dataset = NRMSafetySequenceDataset(num_episodes=4, max_steps=6, sequence_length=8, seed=0)
    schema = get_schema_for_env("nrm_nav")

    for episode in dataset.episodes_tokens:
        assert episode.shape[1] == schema.width
        assert np.issubdtype(episode.dtype, np.integer)
        validate_episode_tokens(
            episode,
            schema=schema,
            end_token_id=dataset.end_token_id,
            observation_space_n=dataset.env.observation_space.n,
            action_space_n=dataset.env.action_space.n,
        )


def test_frozenlake_dataset_matches_schema_contract():
    dataset = FrozenLakeSequenceDataset(
        num_episodes=4,
        max_steps=6,
        sequence_length=8,
        seed=0,
        map_size="4x4",
        is_slippery=False,
        policy_mix=0.0,
    )
    schema = get_schema_for_env("frozenlake")

    for episode in dataset.episodes_tokens:
        assert episode.shape[1] == schema.width
        assert np.issubdtype(episode.dtype, np.integer)
        validate_episode_tokens(
            episode,
            schema=schema,
            end_token_id=dataset.end_token_id,
            observation_space_n=dataset.env.observation_space.n,
            action_space_n=dataset.env.action_space.n,
        )
