"""
This module implements an Equal Weight rebalancing agent. This means that we are 
going to rebalance our portfolio every time step to have equal weights for all
the companies in our portfolio.
"""

from typing import Optional, Union

import mlflow
import numpy as np
from gym.spaces.dict import Dict
from torch import Tensor

from portfolio_management_rl.market_environment.market_env import MarketEnvState
from portfolio_management_rl.utils.contstants import N_STOCKS

from .base import BaseAgent


class EqualWeightAgent(BaseAgent):
    """
    Equal Weight rebalancing agent. We are going to rebalance our portfolio every
    time step to have equal weights for all the companies in our portfolio.
    """

    _name = "EqualWeightAgent"

    def __init__(self, rebalance: float = 0.6):
        """
        Initializes the agent.

        Args:
            output_shape (tuple): Shape of the output action.
            include_balance (bool): Whether to divide the weights with the balance as well
                if true, (0.1, ..., 0.1, balance:=0.1) else (0.1, ..., 0.1, balance:=0)
        """
        self.weights = np.ones(shape=(N_STOCKS + 1,))
        self.weights[-1] = 0
        self.weights /= np.sum(self.weights)
        self.rebalance_prob = rebalance
        self.rebalance = rebalance > 0.5
        self.first_step = True

        if self.rebalance:
            self._name = "EqualWeightRebalanceAgent"

    @property
    def parameters(self) -> dict:
        """
        Returns a Dictionary of parameters
        """
        return {
            "weights": self.weights,
            "rebalance": self.rebalance,
        }

    @property
    def name(self) -> str:
        """
        Returns the name of the agent.
        """
        return self._name

    def reset(self) -> None:
        """
        Resets the agent to its initial state.
        """

        self.first_step = True

    def act(self, state: MarketEnvState) -> Dict:
        """
        Takes in a state and returns an action.

        Args:
            state (dict): State dictionary.

        Returns:
            dict: Action dictionary.
        """
        if state.balance > 500:
            return {
                "distribution": self.weights.copy(),
                "rebalance": 0.6,
            }
        if self.rebalance:
            self.first_step = False
            return {
                "distribution": self.weights.copy(),
                "rebalance": self.rebalance_prob,
            }

        if self.first_step:
            print("First step")
            self.first_step = False
            return {
                "distribution": self.weights.copy(),
                "rebalance": self.rebalance_prob,
            }
        else:
            return {
                "distribution": state.net_distribution.copy(),
                "rebalance": self.rebalance_prob,
            }

    def update(
        self,
        state: MarketEnvState,
        action: Dict,
        reward: Union[float, np.ndarray, Tensor],
        next_state: MarketEnvState,
    ) -> None:
        pass
