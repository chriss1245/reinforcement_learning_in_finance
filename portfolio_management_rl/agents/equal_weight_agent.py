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

    def __init__(
        self,
        rebalance: bool = True,
    ):
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

        self.rebalance = rebalance
        self.first_step = True

    @property
    def parameters(self) -> dict:
        """
        Returns a Dictionary of parameters
        """
        return {
            "weights": self.weights,
            "rebalance": self.rebalance,
        }

    def reset(self) -> None:
        """
        Resets the agent to its initial state.
        """
        self.weights = np.ones(shape=(N_STOCKS + 1,))
        self.weights[-1] = 0
        self.weights /= np.sum(self.weights)
        self.first_step = True

    def act(self, state: MarketEnvState) -> Dict:
        """
        Takes in a state and returns an action.

        Args:
            state (dict): State dictionary.

        Returns:
            dict: Action dictionary.
        """
        if self.rebalance:
            self.first_step = False
            return {"distribution": self.weights.copy(), "rebalance": True}

        else:
            if self.first_step:
                self.first_step = False
                return {"distribution": self.weights.copy(), "rebalance": True}
            else:
                return {
                    "distribution": state.net_distribution.copy(),
                    "rebalance": False,
                }

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
