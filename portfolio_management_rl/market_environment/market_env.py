"""
This module emulates the market. It offers an interface to the agent to get the current state
of the market and to execute actions on the market.
"""

import logging
from pathlib import Path
from typing import Any, Optional, Tuple

import gym
import numpy as np

from portfolio_management_rl.datasets.dataset import StocksDataset
from portfolio_management_rl.utils.contstants import N_STOCKS
from portfolio_management_rl.utils.logger import get_logger

from .brokers import Broker, Trading212
from .commons import MarketEnvState

logger = get_logger(__file__, level=logging.DEBUG)


class MarketEnv(gym.Env):
    """
    Market environment. Simulates the market and offers an interface to the agent to get the current state
    of the market and to execute actions on the market.
    """

    def __init__(
        self,
        dataset: StocksDataset,
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

        history, _ = dataset[0]
        # Market States
        self.initial_state = MarketEnvState(
            history=history,  # type: ignore
            portfolio=np.zeros(shape=(N_STOCKS + 1,)),
            done=False,
        )

        self.initial_state.balance = initial_balance

        self.state = self.initial_state.copy()

        # History
        self.logdir = experiment_logdir

        # iterations
        self.iteration = 1

    def __len__(self) -> int:
        """
        Returns the length of the dataset.

        Returns:
            int: Length of the dataset.
        """
        return len(self.dataset - 1)  # we have n-1 transitions

    @staticmethod
    def get_state_action(state: MarketEnvState) -> dict[str, Any]:
        """
        Returns the action of the given state.

        Args:
            state (MarketEnvState): State of the market environment.

        Returns:
            dict[str, Any]: Action of the state.
        """

        distribution = state.net_distribution.copy()
        return {"distribution": distribution, "rebalance": False}

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
        self.state = self.initial_state

        self.iteration = 1

        return self.state

    def store(self, action: np.ndarray, reward: float, done: bool) -> None:
        """
        Stores the current state of the market environment if a logdir is provided.

        Args:
            action (np.ndarray): Action executed on the market.
            reward (float): Reward obtained by the action.
            done (bool): Whether the episode is finished or not.
        """
        raise NotImplementedError

    def get_reward(self, state: MarketEnvState, next_state: MarketEnvState) -> float:
        """
        Reward method for the market environment.

        Args:
            previous_state (MarketEnvState): Previous state of the market environment.
            next_state (MarketEnvState): Next state of the market environment.
            action (np.ndarray): Action executed on the market.
        """

        # MAPE profit prior to the update
        current_gain = (
            state.net_worth - self.initial_state.net_worth
        ) / self.initial_state.net_worth

        # MAPE profit after the update
        next_gain = (
            next_state.net_worth - self.initial_state.net_worth
        ) / self.initial_state.net_worth

        # Regularization term (avoid that much changes in the portfolio)
        # regularization = np.mean(np.abs(next_state.portfolio_distribution - previous_state.portfolio_distribution))

        # If the episode has ended, the reward is the profit
        if self.state.done:
            return next_gain

        return next_gain - current_gain  # - regularization

    def step(
        self, action: dict[str, Any]
    ) -> Tuple[MarketEnvState, float, bool, bool, dict[str, Any]]:
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
            truncated (bool): Whether the episode has been truncated.
            info (dict[str, Any]): Additional information.
        """

        current_state = self.state.copy()

        # execute the action
        delta_distribution = (
            action["distribution"][:-1] - current_state.net_distribution[:-1]
        )

        cashflow = 0.0
        # sell
        sell = np.zeros_like(delta_distribution)
        sell[delta_distribution < 0] = -delta_distribution[delta_distribution < 0]
        if np.sum(sell) > 0:
            sell = (
                current_state.net_worth * sell / current_state.prices
            )  # covert to quantity

            # if sell is greater than shares, sell all shares
            sell[sell > current_state.shares] = current_state.shares[
                sell > current_state.shares
            ]
            cashflow += self.broker.sell(current_state, sell)

        # buy
        buy = np.zeros_like(delta_distribution)

        buy[delta_distribution > 0] = delta_distribution[delta_distribution > 0]
        # normalize the buy vector to add up to 1
        # (because selling is alters the net worth   because of the transaction cost)
        if np.sum(buy) > 0:
            buy /= np.sum(buy)

            # covert to quantity
            buy_quantities = self.broker.get_quantity_to_buy(current_state, buy)
            cashflow += self.broker.buy(current_state, buy_quantities)

        # Create the next state
        done = self.iteration == len(self.dataset) - 1
        history, _ = self.dataset[self.iteration]  # history, future_price_n_days
        next_state = MarketEnvState(
            history=history, portfolio=current_state.portfolio.copy(), done=done
        )

        # reward
        reward = self.get_reward(current_state, next_state)

        self.iteration += 1
        self.state = next_state.copy()

        info = {"cashflow": cashflow}

        return next_state, reward, done, False, info
