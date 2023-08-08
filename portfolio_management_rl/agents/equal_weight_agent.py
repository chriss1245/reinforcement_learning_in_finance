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

from .base import BaseAgent


class EqualWeightAgent(BaseAgent):
    """
    Equal Weight rebalancing agent. We are going to rebalance our portfolio every
    time step to have equal weights for all the companies in our portfolio.
    """

    def __init__(
        self,
        output_shape: tuple = (1, 101),
        include_balance: bool = True,
        rebalance: bool = True,
    ):
        """
        Initializes the agent.

        Args:
            output_shape (tuple): Shape of the output action.
            include_balance (bool): Whether to divide the weights with the balance as well
                if true, (0.1, ..., 0.1, balance:=0.1) else (0.1, ..., 0.1, balance:=0)
        """
        self.weights = np.ones(output_shape)
        if not include_balance:
            self.weights[-1] = 0
        self.weights /= np.sum(self.weights)

        self.output_shape = output_shape
        self.include_balance = include_balance
        self.rebalance = rebalance

    @property
    def parameters(self) -> dict:
        """
        Returns a Dictionary of parameters
        """
        return {
            "weights": self.weights,
            "include_balance": self.include_balance,
            "rebalance": self.rebalance,
        }

    def action(self, state: MarketEnvState) -> Dict:
        """
        Takes in a state and returns an action.

        Args:
            state (dict): State dictionary.

        Returns:
            dict: Action dictionary.
        """
        if self.rebalance:
            return {"distribution": self.weights}

        if self.include_balance:
            return {"distribution": state.net_distribution}

        return {"distribution": state.net_distribution[:-1]}

    def update(
        self,
        state: MarketEnvState,
        action: Dict,
        reward: Union[float, np.ndarray, Tensor],
        next_state: MarketEnvState,
    ) -> None:
        pass

    def log(self, metrics: Optional[dict] = None):
        """
        Logs the model in mlflow as a pyfunc
        """

        mlflow.log_metrics(metrics)
        mlflow.log_params(self.parameters)
