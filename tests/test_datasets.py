from cb_dataset import CBSequenceDataset
from frozenlake_dataset import FrozenLakeSequenceDataset
from nrm_nav_dataset import NRMSafetySequenceDataset


def _check_dataset_sample_shapes_and_shift(dataset):
    x, y, mask = dataset[0]

    assert x.shape == y.shape == mask.shape

    x_rows = x.view(-1, 4)
    y_rows = y.view(-1, 4)
    assert (y_rows[:-1] == x_rows[1:]).all()


def _check_exactly_one_end_per_episode(dataset):
    end_id = dataset.end_token_id
    for ep in dataset.episodes_tokens:
        assert int((ep.reshape(-1) == end_id).sum()) == 1


def test_cb_dataset_sample_shape_and_shift():
    dataset = CBSequenceDataset(num_episodes=5, max_steps=5, sequence_length=8)
    _check_dataset_sample_shapes_and_shift(dataset)
    _check_exactly_one_end_per_episode(dataset)


def test_nrm_nav_dataset_sample_shape_and_shift():
    dataset = NRMSafetySequenceDataset(num_episodes=5, max_steps=5, sequence_length=8)
    _check_dataset_sample_shapes_and_shift(dataset)
    _check_exactly_one_end_per_episode(dataset)


def test_frozenlake_dataset_sample_shape_and_shift():
    dataset = FrozenLakeSequenceDataset(
        num_episodes=8,
        max_steps=10,
        sequence_length=8,
        map_size="4x4",
        is_slippery=False,
        policy_mix=0.2,
    )
    _check_dataset_sample_shapes_and_shift(dataset)
    _check_exactly_one_end_per_episode(dataset)
