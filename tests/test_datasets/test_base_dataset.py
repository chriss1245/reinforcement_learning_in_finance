"""
This module is in charge of testing the base dataset module and its mixup functionality.
"""

import unittest

from portfolio_management_rl.datasets.dataset import MixUpConfig, StocksDataset
from portfolio_management_rl.utils.contstants import N_STOCKS, TESTS_DIR, WINDOW_SIZE
from portfolio_management_rl.utils.dtypes import Phase

DATA_DIR = TESTS_DIR / "media/test_dataset"


class TestDataset(unittest.TestCase):
    """
    Testing the Dataset basic behavior
    """

    def test_get_datasets(self):
        """
        Test that the static method get_datasets returns the correct datasets
        """
        _, _, _ = StocksDataset.get_datasets(data_dir=DATA_DIR)

    def test_get_item(self):
        """
        Test that the __getitem__ method returns the correct data
        """
        train_dataset = StocksDataset(data_dir=DATA_DIR, phase=Phase.TRAIN)

        x, y = train_dataset[0]

        self.assertEqual(x.shape, (N_STOCKS, WINDOW_SIZE))
        self.assertEqual(y.shape, (N_STOCKS, 1))

    def test_data_len(self):
        """
        Test that the __len__ method returns the correct data
        """
        train_dataset = StocksDataset(
            data_dir=DATA_DIR, phase=Phase.TRAIN, step_size_days=1
        )

        self.assertEqual(
            train_dataset[len(train_dataset) - 1][0].shape, (N_STOCKS, WINDOW_SIZE)
        )

        with self.assertRaises(IndexError):
            train_dataset[len(train_dataset)]

    def test_mixup(self):
        """
        Test that the mixup method returns the correct data
        """

        mixup_config = MixUpConfig(mixup_alpha=0.5, mixup_prob=1.0)

        train_dataset = StocksDataset(
            data_dir=DATA_DIR,
            phase=Phase.TRAIN,
            step_size_days=1,
            mixup_config=mixup_config,
        )
        no_mixup_dataset = StocksDataset(
            data_dir=DATA_DIR,
            phase=Phase.TRAIN,
            step_size_days=1,
            mixup_config=MixUpConfig(),
        )

        x, y = train_dataset[0]
        x_no_mixup, y_no_mixup = no_mixup_dataset[0]

        self.assertTrue((x != x_no_mixup).any())
        self.assertTrue((y != y_no_mixup).any())


if __name__ == "__main__":
    unittest.main()
