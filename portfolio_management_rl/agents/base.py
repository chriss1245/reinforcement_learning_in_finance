"""
This module contains the base agent class that all other agents inherit from.
"""

from abc import ABC
from typing import Union

import numpy as np
from gym.spaces.dict import Dict
from torch import Tensor

from portfolio_management_rl.market_environment.market_env import MarketEnvState


class BaseAgent(ABC):
    """
    Base Agent. Defines the interface for all agents.
    """

    def action(self, state: MarketEnvState) -> Dict:
        """
        Takes in a state and returns an action.

        Args:
            state (dict): State dictionary.

        Returns:
            dict: Action dictionary.
        """
        raise NotImplementedError

    def update(
        self,
        state: MarketEnvState,
        action: Dict,
        reward: Union[float, np.ndarray, Tensor],
        next_state: MarketEnvState,
    ) -> None:
        """
        Updates the agent.

        Args:
            state (dict): State dictionary.
            action (dict): Action dictionary.
            reward (float): Reward value.
            next_state (dict): Next state dictionary.
        """
        raise NotImplementedError

    def log(self, **kwargs) -> None:
        """
        Logs the agent in mlflow.

        Args:
            path (str): Path to log the agent to.
        """
        raise NotImplementedError