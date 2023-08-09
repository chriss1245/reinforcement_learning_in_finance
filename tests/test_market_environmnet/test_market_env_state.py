"""
This file constains the tests over the class MarketEnvState.
"""
from pathlib import Path
from unittest import TestCase

import numpy as np

from portfolio_management_rl.market_environment.market_env import MarketEnvState


class TestMarketEnvState(TestCase):
    """
    Testing MarketEnvState class.
    """

    def setUp(self) -> None:
        """
        Sets up the test.
        """
        self.history = np.ones(shape=(100, 2000))
        self.portfolio = np.random.dirichlet(np.ones(101), size=1)[0]

    def test_init(self):
        """
        Tests the initialization.
        """
        state = MarketEnvState(
            history=self.history,
            portfolio=self.portfolio,
            done=False,
        )

        self.assertEqual(state.history.shape, self.history.shape)
        self.assertEqual(state.portfolio.shape, self.portfolio.shape)
        self.assertEqual(state.done, False)

    def test_balance(self):
        """
        Tests the balance property.
        """
        state = MarketEnvState(
            history=self.history,
            portfolio=self.portfolio,
            done=False,
        )

        self.assertEqual(state.balance, state.portfolio[-1])

    def test_balance_setter(self):
        """
        Tests the balance setter.
        """
        state = MarketEnvState(
            history=self.history,
            portfolio=self.portfolio,
            done=False,
        )

        state.balance = 10
        self.assertEqual(state.balance, 10)
        self.assertEqual(state.portfolio[-1], 10)

    def test_prices(self):
        """
        Tests the prices property.
        """
        state = MarketEnvState(
            history=self.history,
            portfolio=self.portfolio,
            done=False,
        )

        self.assertEqual(state.prices.shape, (100,))
        self.assertTrue(np.all(state.prices == state.history[:, -1]))

    def test_shares(self):
        """
        Tests the shares property.
        """
        state = MarketEnvState(
            history=self.history,
            portfolio=self.portfolio,
            done=False,
        )

        self.assertEqual(state.shares.shape, (100,))
        self.assertTrue(np.all(state.shares == state.portfolio[:-1]))

    def test_shares_setter(self):
        """
        Tests the shares setter.
        """
        state = MarketEnvState(
            history=self.history,
            portfolio=self.portfolio,
            done=False,
        )

        state.shares = np.ones(shape=(100,))
        self.assertTrue(np.all(state.shares == np.ones(shape=(100,))))
        self.assertTrue(np.all(state.portfolio[:-1] == np.ones(shape=(100,))))

    def test_copy(self):
        """
        Tests the copy method.
        """
        state = MarketEnvState(
            history=self.history,
            portfolio=self.portfolio,
            done=False,
        )

        state_copy = state.copy()

        self.assertTrue(np.all(state.history == state_copy.history))
        self.assertTrue(np.all(state.portfolio == state_copy.portfolio))
        self.assertEqual(state.done, state_copy.done)

        state_copy.history[0, 0] = -1
        state_copy.portfolio[0] = -1

        self.assertFalse(np.all(state.history == state_copy.history))
        self.assertFalse(np.all(state.portfolio == state_copy.portfolio))
