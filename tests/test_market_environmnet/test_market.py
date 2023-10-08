"""
This module is in charge of testing the market env module.
"""

import unittest

import numpy as np
import torch

from portfolio_management_rl.dataloaders.dataset import StocksDataset
from portfolio_management_rl.market_environment.brokers import Trading212
from portfolio_management_rl.market_environment.market_env import (
    MarketEnv,
    MarketEnvState,
)
from portfolio_management_rl.utils.contstants import N_STOCKS, TESTS_DIR

DATA_DIR = TESTS_DIR / "media" / "test_dataset"


class TestMarketEnv(unittest.TestCase):
    """
    Tests of the behavior of the market environment.
    """

    def setUp(self) -> None:
        datasets = StocksDataset.get_datasets(data_dir=DATA_DIR, return_dict=True)
        self.env = MarketEnv(dataset=datasets["train"])

    def test_initialization(self):
        self.assertIsInstance(self.env.broker, Trading212)
        self.assertEqual(self.env.initial_state.balance, 10000)
        self.assertEqual(self.env.iteration, 1)

    def test_reset(self):
        initial_state = self.env.initial_state
        self.env.iteration = 10
        reset_state = self.env.reset()
        self.assertEqual(reset_state, initial_state)
        self.assertEqual(self.env.iteration, 1)

    def test_step(self):
        action = {"distribution": np.zeros(shape=(N_STOCKS + 1,))}
        observation, reward, done, truncated, info = self.env.step(action)
        # Add assertions to check the behavior of step method.
        # It might include checking the types of return values,
        # the values themselves, or any changes to the internal state of the environment.

        # Example:
        self.assertIsInstance(observation, MarketEnvState)
        self.assertIsInstance(reward, float)
        self.assertIsInstance(done, bool)
        self.assertIsInstance(truncated, bool)
        self.assertIsInstance(info, dict)


if __name__ == "__main__":
    unittest.main()
