"""
This module emulates the market. It offers an interface to the agent to get the current state
of the market and to execute actions on the market.
"""

import datetime
from dataclasses import asdict, dataclass, field
from enum import Enum
from logging import getLogger
from pathlib import Path
from typing import Any, Optional, Tuple

import gym
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from portfolio_management_rl.datasets.utils import PortfolioDistributionSpace

from .brokers import Broker, Trading212


@dataclass()
class MarketEnvState:
    """
    State of the market environment.

    Attributes:
        balance (float): Current balance (in the base currency)
        history (np.ndarray): History of the market (in the base currency).
        portfolio (np.ndarray): Quantity of stocks.
        done (bool): Whether the episode is done or not.
    """

    history: np.ndarray  # (n stocks + 1 balance, n time steps)
    portfolio: np.ndarray  # (n stocks + 1 balance)
    done: bool = False

    def __iter__(self):
        """
        Iterates over the state. This is useful to convert the state to a dictionary.
        """
        return iter(asdict(self).items())

    def copy(self):
        """
        Returns a copy of the state.
        """
        return MarketEnvState(
            history=self.history.copy(),
            portfolio=self.portfolio.copy(),
            done=self.done,
        )

    @property
    def balance(self) -> float:
        """
        Returns the balance of the portfolio.

        Returns:
            float: Balance of the portfolio.
        """
        return self.portfolio[-1]

    @balance.setter
    def balance(self, value: float):
        """
        Sets the balance of the portfolio.

        Args:
            value (float): New balance.
        """
        self.portfolio[-1] = value

    @property
    def prices(self) -> np.ndarray:
        """
        Returns the prices of the market.

        Returns:
            np.ndarray: Prices of the market.
        """
        return self.history[:, -1]

    @property
    def shares(self) -> np.ndarray:
        """
        Returns the shares of the portfolio.

        Returns:
            np.ndarray: Shares of the portfolio.
        """
        return self.portfolio[:-1]

    @shares.setter
    def shares(self, value: np.ndarray):
        """
        Sets the shares of the portfolio.

        Args:
            value (np.ndarray): New shares.
        """
        self.portfolio[:-1] = value

    @property
    def investment(self) -> float:
        """
        Returns the investment of the portfolio.

        Returns:
            float: Investment of the portfolio.
        """
        return self.portfolio[:-1].dot(self.prices)

    @property
    def net_worth(self) -> float:
        """
        Returns the net worth of the portfolio.

        Returns:
            float: Net worth of the portfolio.
        """
        return self.balance + self.investment

    @property
    def net_distribution(self) -> np.ndarray:
        """
        Returns the distribution of the portfolio (including the balance).

        Returns:
            np.ndarray: Distribution of the portfolio (including the balance)
        """

        # add 1 to the prices vector
        prices = np.append(self.prices, 1)  # balance

        return (self.portfolio * prices) / self.net_worth


class MarketEnv(gym.Env):
    """
    Market environment. Simulates the market and offers an interface to the agent to get the current state
    of the market and to execute actions on the market.
    """

    def __init__(
        self,
        dataset: Dataset,
        initial_balance: float = 10000,
        broker: Broker = Trading212(),
        experiment_logdir: Optional[Path] = None,
    ):
        """
        Market environment. Simulates the market and offers an interface for
        portfolio management agents to get the current state of the market ands
        to execute actions on the market.

        Args:
            dataset (Dataset): Dataset containing the market data.
            initial_balance (float, optional): Initial balance. Defaults to 10000.
            broker (Broker, optional): Broker to use. Defaults to Trading212().
            experiment_logdir (Optional[str], optional): Path to store the experiments logs. Defaults to None.
        """

        # data generator
        self.dataset = dataset
        self.broker = broker
        action_space, observation_space = dataset.get_action_observation_space()

        history, _ = dataset[0]

        # Market States
        self.initial_state = MarketEnvState(
            history=history,  # type: ignore
            portfolio=np.zeros(shape=(101,)),
            done=False,
        )

        self.initial_state.balance = initial_balance

        self.last_state = None
        self.current_state = self.initial_state

        # History
        self.logdir = experiment_logdir

        # iterations
        self.iteration = 1

    def render(self, mode: str = "human") -> None:
        """
        Renders the current state of the market environment.

        Args:
            mode (str, optional): Mode of the rendering. Defaults to 'human'.
        """
        raise NotImplementedError

    def reset(self) -> MarketEnvState:
        """
        Resets the market environment to its initial state.
        """
        self.current_state = self.initial_state
        self.last_state = None
        self.iteration = 0

        return self.current_state

    def store(self, action: np.ndarray, reward: float, done: bool) -> None:
        """
        Stores the current state of the market environment if a logdir is provided.

        Args:
            action (np.ndarray): Action executed on the market.
            reward (float): Reward obtained by the action.
            done (bool): Whether the episode is finished or not.
        """
        raise NotImplementedError

    def get_reward(self) -> float:
        """
        Reward method for the market environment.

        Args:
            previous_state (MarketEnvState): Previous state of the market environment.
            next_state (MarketEnvState): Next state of the market environment.
            action (np.ndarray): Action executed on the market.
        """

        # MAPE profit prior to the update
        previous_gain = (
            self.last_state.net_worth - self.initial_state.net_worth
        ) / self.initial_state.net_worth

        # MAPE profit after the update
        next_gain = (
            self.current_state.net_worth - self.initial_state.net_worth
        ) / self.initial_state.net_worth

        # Regularization term (avoid that much changes in the portfolio)
        # regularization = np.mean(np.abs(next_state.portfolio_distribution - previous_state.portfolio_distribution))

        # If the episode has ended, the reward is the profit
        if self.current_state.done:
            return next_gain

        return next_gain - previous_gain  # - regularization

    def step(self, action: dict[str, Any]) -> Tuple[MarketEnvState, float, bool]:
        """
        Executes an action on the market. Returns the next observation, the reward
        and whether the episode has ended.

        reward = (future_balance - current_balance)/initial_balance - transaction_cost

        Args:
            action (np.ndarray): Dictionary containing the new action, and whether execute
                the action or not

        Returns:
            observation (np.array): The next observation.
            reward (float): The reward obtained by executing the action.
            done (bool): Whether the episode has ended.
        """

        history, _ = self.dataset[self.iteration]  # history, future_price_n_days

        # update the state
        self.last_state = self.current_state.copy()
        self.current_state = MarketEnvState(
            history=history,
            portfolio=self.current_state.portfolio.copy(),
            done=False,
        )

        # execute the action
        delta_distribution = (
            action["distribution"][:-1] - self.current_state.net_distribution[:-1]
        )

        # sell
        sell = np.zeros_like(delta_distribution)
        sell[delta_distribution < 0] = -delta_distribution[delta_distribution < 0]
        sell = (
            self.current_state.net_worth * sell / self.current_state.prices
        )  # covert to quantity
        self.broker.sell(self.current_state, sell)

        # buy
        buy = np.zeros_like(delta_distribution)

        buy[delta_distribution > 0] = delta_distribution[delta_distribution > 0]
        # normalize the buy vector to add up to 1
        # (because selling is alters the net worth   because of the transaction cost)
        buy /= np.sum(buy)

        buy = (
            self.current_state.balance * buy / self.current_state.prices
        )  # covert to quantity

        # truncate to  5 decimal places
        buy = np.trunc(buy * 100000) / 100000

        self.broker.buy(self.current_state, buy)

        # reward
        reward = self.get_reward()

        # Check if the episode has ended
        done = self.iteration == len(self.dataset) - 1

        if done:
            self.current_state.done = True
            reward = self.get_reward()

        self.iteration += 1

        return self.current_state, reward, done
