"""
This module contains the tests for the Trading212 broker. The tests are done by creating a random portfolios
The broker objects are the responsible for the execution of the orders. They are the ones that update the
portfolio and the market state. The broker is the one that is responsible for the commissions and the
slippage. This broker in particular is the one that is used for the Trading212 platform which does not charge commissions unless
we buy stocks in other currencies. The slippage is the difference between the price that we see in the platform and the price
that we buy the stock at. In this case the slippage is 0.
"""

import unittest

import numpy as np
import torch

from portfolio_management_rl.market_environment.brokers import (
    MarketEnvState,
    Trading212,
)


class TestTrading212(unittest.TestCase):
    """
    Testing of the Trading212 broker. The tests are done by creating a random portfolios
    """

    def setUp(self):
        self.broker = Trading212()
        history = np.random.uniform(1, 100, size=(100, 5))
        self.prices = history[:, -1]

        portfolio = np.random.uniform(1, 100, size=(101,))
        portfolio[-1] = 8000

        self.market_state = MarketEnvState(history=history, portfolio=portfolio)

    def test_buy(self):
        """
        This function tests the buy function of the broker. It tests that the broker
        buys the correct amount of shares and that the balance and net worth are
        updated correctly.


        """
        # Test buying stocks

        net_worth = self.market_state.net_worth
        balance = self.market_state.balance
        shares = self.market_state.shares.copy()

        self.broker.buy(self.market_state, np.ones(shape=(100,)))

        np.testing.assert_array_almost_equal(shares + 1, self.market_state.shares)

        self.assertGreaterEqual(net_worth, self.market_state.net_worth)

        self.assertGreater(balance, self.market_state.balance)

        # Test buying stocks with insufficient balance
        with self.assertRaises(ValueError):
            self.broker.buy(self.market_state, np.ones(shape=(100,)) * 1000)

    def test_sell(self):
        """
        This tests the sell function of the broker. It tests that the broker
        sells the correct amount of shares and that the balance and net worth
        are updated correctly.

        Compares the shares quantity before and after the sell
        Makes sure the balance is greater after the sell
        Makes sure that the net_worth is less after the sell (because of commissions)
        """

        # Setup initial portfolio

        shares_before = self.market_state.shares.copy()  # shares before delta

        delta = np.ones(shape=(101,))
        delta[-1] = 0
        self.market_state.portfolio = self.market_state.portfolio + delta

        balance = self.market_state.balance
        net_worth = self.market_state.net_worth

        # Test selling stocks
        self.broker.sell(self.market_state, delta[:-1])

        np.testing.assert_array_almost_equal(shares_before, self.market_state.shares)
        self.assertGreater(self.market_state.balance, balance)
        self.assertGreater(net_worth, self.market_state.net_worth)

        # Test selling more stocks than available
        with self.assertRaises(ValueError):
            self.broker.sell(self.market_state, np.array([1, 1, 1]))


class TestTorchTrading212(unittest.TestCase):
    """
    This module tests the Trading212 broker when the market state is represented by torch tensors
    """

    def setUp(self):
        self.broker = Trading212()
        history = torch.tensor(
            np.random.uniform(1, 100, size=(100, 5)), dtype=torch.float32
        )
        self.prices = history[:, -1]

        portfolio = np.random.uniform(1, 100, size=(101,))
        portfolio[-1] = 8000
        portfolio = torch.from_numpy(portfolio).to(torch.float32)
        self.market_state = MarketEnvState(history=history, portfolio=portfolio)

    def test_buy(self):
        """
        This function tests the buy function of the broker. It tests that the broker
        buys the correct amount of shares and that the balance and net worth are
        updated correctly.

        """
        # Test buying stocks

        shares_before = self.market_state.shares.clone()
        delta = torch.ones(size=(101,), dtype=torch.float32)
        self.market_state.shares -= delta[:-1]
        net_worth = self.market_state.net_worth
        balance = self.market_state.balance

        self.broker.buy(self.market_state, delta[:-1])

        torch.testing.assert_close(shares_before, self.market_state.shares)

        self.assertGreaterEqual(net_worth, self.market_state.net_worth)
        self.assertGreaterEqual(balance, self.market_state.balance)
        # Test buying stocks with insufficient balance
        with self.assertRaises(ValueError):
            self.broker.buy(self.market_state, torch.ones(size=(100,)) * 1000)

    def test_sell(self):
        """
        This tests the sell function of the broker. It tests that the broker
        sells the correct amount of shares and that the balance and net worth
        are updated correctly.

        Compares the shares quantity before and after the sell
        Makes sure the balance is greater after the sell
        Makes sure that the net_worth is less after the sell (because of commissions)
        """

        # Setup initial portfolio

        shares_before = self.market_state.shares.clone()

        delta = torch.ones(size=(101,))
        delta[-1] = 0
        self.market_state.portfolio = self.market_state.portfolio + delta

        balance = self.market_state.balance
        net_worth = self.market_state.net_worth

        # Test selling stocks
        self.broker.sell(self.market_state, delta[:-1])

        torch.testing.assert_close(shares_before, self.market_state.shares)
        self.assertGreater(self.market_state.balance, balance)
        self.assertGreater(net_worth, self.market_state.net_worth)

        # Test selling more stocks than available
        with self.assertRaises(ValueError):
            self.broker.sell(self.market_state, torch.ones(size=(100,)) * 1000)


if __name__ == "__main__":
    unittest.main()
