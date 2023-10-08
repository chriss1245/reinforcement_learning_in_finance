"""
This module implements a random agent. The purpose of this agent is to serve as 
a baseline to compare the performance of other agents.
Additianally, it can be helpful reducing the suvivorship bias of the dataset. 
"""
from typing import Dict, Union
from gym.spaces.dict import Dict
import numpy as np
from torch import Tensor

from portfolio_management_rl.market_environment.market_env import MarketEnvState
from portfolio_management_rl.utils.contstants import N_STOCKS

from .base import BaseAgent
from .efficient_frotier_agent import EfficientFrontierAgent
from .equal_weight_agent import EqualWeightAgent


class RandomAgent(BaseAgent):
    """
    This module implements a random agent. The purpose of this agent is to serve as a baseline
    to compare the performance of other agents. Additianally, it can be helpful reducing the suvivorship
    bias of the dataset. Since the random agent does not learn, its performance will mark the performance
    of one of the worst agents possible. So we can correct the performance of other agents by substracting the
    performance of the random agent.
    """

    _name = "RandomAgent"

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

    @property
    def name(self) -> str:
        """
        Returns the name of the agent
        """
        return self._name

    def act(self, state: MarketEnvState) -> dict:
        """
        Act method of the agent. It returns a random portfolio.

        Args:
            state (MarketEnvState): Current state of the market.

        Returns:
            dict: Portfolio to execute.
        """
        rebalance = np.random.uniform()
        if self.remember:
            # use the last portfolio as prior for the next portfolio

            port = np.random.dirichlet(self.portfolio + 1, size=1)[0]

            return {"distribution": port, "rebalance": rebalance}

        # mutlinomial
        port = np.random.dirichlet(np.ones(self.n_stocks + 1), size=1)[0]

        return {"distribution": port, "rebalance": rebalance}

    def reset(self) -> None:
        pass

    def update(
        self,
        state: MarketEnvState,
        action: Dict,
        reward: float,
        next_state: MarketEnvState,
    ) -> None:
        pass


class InsecureAgent(BaseAgent):
    """
    The insecure agent uses make decisions based on the performance of other agents. It is insecure because it does not know which
    agent is the best, so it randomly chooses one of the agents at each act step.

    It is very useful in order to make exploration. This can be interesting for generating replay buffers
    """

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

    @property
    def name(self) -> str:
        """
        Returns the name of the agent
        """
        return "InsecureAgent"

    def act(self, state: MarketEnvState) -> dict:
        agent = np.random.choice(self.agents)
        return agent.act(state)

    def update(
        self,
        state: MarketEnvState,
        action: Dict,
        reward: float,
        next_state: MarketEnvState,
    ) -> None:
        pass

    def reset(self) -> None:
        pass
