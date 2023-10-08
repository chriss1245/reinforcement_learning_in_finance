"""
This module contains the base agent class that all other agents inherit from.
"""

from abc import ABC, abstractmethod
from typing import Union

import numpy as np
from gym.spaces.dict import Dict
from torch import Tensor

from typing import Dict, Optional
import mlflow
from portfolio_management_rl.market_environment.market_env import MarketEnvState


class BaseAgent(ABC):
    """
    Base Agent. Defines the interface for all agents.
    """

    @abstractmethod
    def act(self, state: MarketEnvState) -> Dict:
        """
        Takes in a state and returns an action.

        Args:
            state (dict): State dictionary.

        Returns:
            dict: Action dictionary.
        """

    @abstractmethod
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

    @abstractmethod
    def reset(self) -> None:
        """
        Resets the agent to its initial state.
        """

    def log(
        self,
        metrics: Optional[Dict[str, float]],
        tags: Optional[Dict[str, float]],
        plots: Optional[Dict[str, np.ndarray]] = None,
        **kwargs,
    ) -> None:
        # check that there is an active run
        if mlflow.active_run() is None:
            raise RuntimeError(
                "No active run found. Please create a run before logging anything."
            )

        if metrics is None:
            metrics = {}

        mlflow.log_params(self.parameters)

        if metrics:
            mlflow.log_metrics(metrics)

        tags = tags or {}
        tags["agent"] = self.name
        mlflow.set_tags(tags)

        for name, plot in plots.items():
            mlflow.log_figure(plot, name)

    @property
    @abstractmethod
    def parameters(self) -> dict:
        """
        Returns a Dictionary of parameters
        """

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Returns the name of the agent.
        """
