from cb_dataset import CBSequenceDataset
from nrm_nav_dataset import NRMSafetySequenceDataset


def _check_dataset_sample_shapes_and_shift(dataset):
    x, y, mask = dataset[0]

    assert x.shape == y.shape == mask.shape

    x_rows = x.view(-1, 4)
    y_rows = y.view(-1, 4)
    assert (y_rows[:-1] == x_rows[1:]).all()


def test_cb_dataset_sample_shape_and_shift():
    dataset = CBSequenceDataset(num_episodes=5, max_steps=5, sequence_length=8)
    _check_dataset_sample_shapes_and_shift(dataset)


def test_nrm_nav_dataset_sample_shape_and_shift():
    dataset = NRMSafetySequenceDataset(num_episodes=5, max_steps=5, sequence_length=8)
    _check_dataset_sample_shapes_and_shift(dataset)
