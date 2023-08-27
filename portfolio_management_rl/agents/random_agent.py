"""
This module implements a random agent. The purpose of this agent is to serve as a baseline to compare the performance of other agents.
Additianally, it can be helpful reducing the suvivorship bias of the dataset. Since the random agent does not learn, its performance 
will mark the performance of one of the worst agents possible. So we can correct the performance of other agents by substracting the
performance of the random agent.
"""

from typing import Union
from gym.spaces.dict import Dict
import numpy as np
from torch import Tensor
from .base import BaseAgent
from portfolio_management_rl.utils.contstants import N_STOCKS
from portfolio_management_rl.market_environment.market_env import MarketEnvState
from .efficient_frotier_agent import EfficientFrontierAgent
from .equal_weight_agent import EqualWeightAgent


class RandomAgent(BaseAgent):
    """
    This module implements a random agent. The purpose of this agent is to serve as a baseline to compare the performance of other agents.
    Additianally, it can be helpful reducing the suvivorship bias of the dataset. Since the random agent does not learn, its performance
    will mark the performance of one of the worst agents possible. So we can correct the performance of other agents by substracting the
    performance of the random agent.
    """

    def __init__(
        self,
        n_stocks: int = N_STOCKS,
        remember: bool = False,
    ):
        """
        Initialize the agent.

        Args:
            n_stocks (int, optional): Number of stocks. Defaults to N_STOCKS.
            include_cash (bool, optional): Whether to include cash when sampling the portfolio. Defaults to False. So the cash
                component of the portfolio will be 0.
            remember (bool, optional): Whether to remember the last portfolio. Defaults to False.
        """
        self.n_stocks = n_stocks
        self.remember = remember
        self.portfolio = np.ones(self.n_stocks + 1)

    @property
    def parameters(self) -> dict:
        """
        Returns a Dictionary of parameters
        """
        return {
            "n_stocks": self.n_stocks,
            "remember": self.remember,
        }

    def act(self, state: MarketEnvState) -> dict:
        """
        Act method of the agent. It returns a random portfolio.

        Args:
            state (MarketEnvState): Current state of the market.

        Returns:
            dict: Portfolio to execute.
        """
        rebalance = np.random.choice([True, False], p=[0.5, 0.5])

        if self.remember:
            # use the last portfolio as prior for the next portfolio

            port = np.random.dirichlet(self.portfolio + 1, size=1)[0]
            self.portfolio = port * 1000 // 1
            return {"distribution": port, "rebalance": rebalance}

        port = np.random.dirichlet(np.ones(self.n_stocks + 1), size=1)[0]
        self.portfolio = port * 1000 // 1

        return {"distribution": self.portfolio, "rebalance": rebalance}

    def reset(self) -> None:
        pass


class InsecureAgent(BaseAgent):
    def __init__(self, n_stocks: int = N_STOCKS):
        self.n_stocks = n_stocks

        self.agents = [
            RandomAgent(n_stocks=n_stocks),
            EqualWeightAgent(),
            EqualWeightAgent(rebalance=False),
            EfficientFrontierAgent(objective_function="sharpe_ratio"),
            EfficientFrontierAgent(objective_function="min_volatility"),
        ]

    @property
    def parameters(self) -> dict:
        """
        Returns a Dictionary of parameters
        """
        return {
            "n_stocks": self.n_stocks,
        }

    def act(self, state: MarketEnvState) -> dict:
        agent = np.random.choice(self.agents)
        return agent.act(state)

    def update(self, state: MarketEnvState):
        pass

    def reset(self) -> None:
        pass
