"""
Tests for the rebalancing agent for the portfolio.
"""

import unittest

import numpy as np
import torch

from portfolio_management_rl.agents.equal_weight_agent import EqualWeightAgent
from portfolio_management_rl.market_environment.commons import MarketEnvState
from portfolio_management_rl.utils.contstants import N_STOCKS


class TestEqualWeightAgent(unittest.TestCase):
    """
    Test the portfolio rebalancing agent interface.
    """

    def setUp(self):
        """
        Set up the tests.
        """
        self.agent_rebalance = EqualWeightAgent()
        self.agent_not_rebalance = EqualWeightAgent(rebalance=False)

        self.portfolio = np.ones(N_STOCKS + 1, dtype=np.float32) * 10
        self.portfolio[0] = 100  # unbalanced portfolio
        self.history = np.ones((N_STOCKS, 100), dtype=np.float32)
        self.history[0] = np.arange(100, dtype=np.float32)

        self.torch_portfolio = torch.from_numpy(self.portfolio).to(torch.float32)
        self.torch_history = torch.from_numpy(self.history).to(torch.float32)

    def test_rebalance(self):
        """
        Test that the agent rebalances the portfolio to equal weights. If rebalance
        is set to true, the portfolio should be rebalanced to have a uniform weight
        """

        new_state = MarketEnvState(portfolio=self.portfolio, history=self.history)

        action = self.agent_rebalance.act(new_state)

        uniform = np.ones(N_STOCKS) / N_STOCKS
        uniform = np.append(uniform, 0)
        self.assertAlmostEqual(np.sum(uniform - action["distribution"]), 0)

    def test_not_rebalance(self):
        """
        Test that the agent does not rebalance the portfolio to equal weights. If rebalance
        is set to false, the portfolio should not be rebalanced.
        """

        new_state = MarketEnvState(portfolio=self.portfolio, history=self.history)

        old_distribution = new_state.net_distribution

        action = self.agent_not_rebalance.act(new_state)

        self.assertAlmostEqual(np.sum(old_distribution - action["distribution"]), 0)
