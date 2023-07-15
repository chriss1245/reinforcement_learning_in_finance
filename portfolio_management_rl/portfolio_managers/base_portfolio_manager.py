"""
This module contains the base agent class that all other agents inherit from.
"""

from abc import ABC


class BasePortfolioManager(ABC):
    """
    Base Agent. Defines the interface for all agents.
    """

    def __init__(self, config: dict):
        """
        Initializes the agent.

        Args:
            config (dict): Configuration dictionary.
        """
        self.config = config

    def action(self, state: dict) -> dict:
        """
        Takes in a state and returns an action.

        Args:
            state (dict): State dictionary.

        Returns:
            dict: Action dictionary.
        """
        raise NotImplementedError

    def update(
        self, state: dict, action: dict, reward: float, next_state: dict, done: bool
    ) -> None:
        """
        Updates the agent.

        Args:
            state (dict): State dictionary.
            action (dict): Action dictionary.
            reward (float): Reward value.
            next_state (dict): Next state dictionary.
            done (bool): Whether the episode is done.
        """
        raise NotImplementedError

    def save(self, **kwargs) -> None:
        """
        Saves the agent.

        Args:
            path (str): Path to save the agent to.
        """
        raise NotImplementedError

    def log(self, **kwargs) -> None:
        """
        Logs the agent.

        Args:
            path (str): Path to log the agent to.
        """
        raise NotImplementedError
