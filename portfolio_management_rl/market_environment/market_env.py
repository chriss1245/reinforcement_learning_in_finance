"""
This module emulates the market. It offers an interface to the agent to get the current state
of the market and to execute actions on the market.
"""

from pathlib import Path
from tabnanny import verbose
from typing import Any, Optional, Tuple
import pandas as pd
import numpy as np
import datetime
import gym
from enum import Enum
from torch.utils.data import Dataset
from dataclasses import dataclass, field, asdict
from logging import getLogger

from portfolio_management_rl.datasets.utils import PortfolioDistributionSpace


@dataclass(slots=True)
class MarketEnvState():
    """
    State of the market environment.

    Attributes:
        balance (float): Current balance (in the base currency)
        history (np.ndarray): History of the market (in the base currency).
        portfolio (np.ndarray): Quantity of stocks.
        done (bool): Whether the episode is done or not.
    """
    balance: float
    history: np.ndarray
    portfolio: np.ndarray
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
        return MarketEnvState(**asdict(self))

    @staticmethod
    def from_dict(state_dict: dict):
        """
        Creates a state from a dictionary.

        Args:
            state_dict (dict): Dictionary containing the state.

        Returns:
            MarketEnvState: State of the market environment.
        """
        return MarketEnvState(**state_dict)

    @property
    def prices(self) -> np.ndarray:
        """
        Returns the prices of the stocks.

        Returns:
            np.ndarray: Prices of the stocks.
        """
        return self.history[:, -1]

    @property
    def investment(self) -> float:
        """
        Returns the investment of the portfolio.

        Returns:
            float: Investment of the portfolio.
        """
        return self.portfolio.dot(self.prices)

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

        temp = self.portfolio * self.prices
        return np.append(temp, self.balance) / self.net_worth

# Strategy pattern used to define the different Brokers (Trading212, Degiro, ...)
class Broker():
    """
    Base broker class
    """

    def buy(self, market_state: MarketEnvState, quantity: np.ndarray) -> np.ndarray:
        """
        Buys stocks.

        Args:
            balance (float): Current balance.

        Returns:
            np.ndarray: Bought stocks.
        """
        raise NotImplementedError

    def sell(self, market_state: MarketEnvState, quantity: np.ndarray) -> float:
        """
        Sell stocks by a given quatity

        Args:
            quamtity (np.ndarray): Quantity of stocks to sell.
        """
        raise NotImplementedError

class Trading212(Broker):
    """
    Trading 212 at july 1st does not charge comission for buying and selling stocks.
    However, it charges a 0.5% comission for converting the profit to the base currency.
    """

    def __init__(self, profit_commision_percent: float = 0.5):
        """
        Initializes the strategy.

        Args:
            proportion_profit (float): Proportion of the profit that is charged as comission.
        """
        self.profit_proportion = profit_commision_percent / 100

    def buy(self, market_state: MarketEnvState, quantity: np.ndarray) -> None:
        """
        Buys stocks. Adding the stocks to the portfolio and substracting the money from the balance.

        Args:
            balance (float): Current balance.

        Returns:
            np.ndarray: Bought stocks.
        """
        total_price = quantity.dot(market_state.prices) # total price of the stocks to buy

        if total_price > market_state.balance:
            raise ValueError("Not enough balance to buy the stocks.")

        market_state.balance -= total_price
        market_state.portfolio += quantity

    def sell(self, market_state: MarketEnvState, quantity: np.ndarray) -> None:
        """
        Sell stocks by a given quatity

        Args:
            quamtity (np.ndarray): Quantity of stocks to sell.
        
        Returns:
            float: Money earned by selling the stocks.
        """

        if np.any(quantity > market_state.portfolio):
            raise ValueError("Not enough stocks to sell.")

        total_price = quantity.dot(market_state.prices) # total price of the stocks to sell
        market_state.balance += total_price * (1 - self.profit_proportion)


class MarketEnv(gym.Env):
    """
    Market environment. Simulates the market and offers an interface to the agent to get the current state
    of the market and to execute actions on the market.
    """

    def __init__(self, dataset: Dataset,
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
        portfolio = np.zeros(shape=(self.dataset.action_space.shape[0],)) #type: ignore

        # Market States
        self.initial_state = MarketEnvState(
            balance=initial_balance,
            history=np.zeros(shape=self.dataset.observation_space.shape), #type: ignore
            portfolio=portfolio,
            done=False)

        self.last_state = None
        self.current_state = self.initial_state

        # History
        self.logdir = experiment_logdir

        #iterations
        self.iteration = 0

    def render(self, mode: str = 'human') -> None:
        """
        Renders the current state of the market environment.

        Args:
            mode (str, optional): Mode of the rendering. Defaults to 'human'.
        """
        raise NotImplementedError

    def reset(self):
        """
        Resets the market environment to its initial state.
        """
        self.current_state = self.initial_state
        self.last_state = None
        self.iteration = 0

    def store(self, action: np.ndarray, reward: float, done: bool) -> None:
        """
        Stores the current state of the market environment if a logdir is provided.

        Args:
            action (np.ndarray): Action executed on the market.
            reward (float): Reward obtained by the action.
            done (bool): Whether the episode is finished or not.
        """

        if self.logdir is not None:
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
        previous_gain = (self.last_state.net_worth - self.initial_state.net_worth) / self.initial_state.net_worth

        # MAPE profit after the update
        next_gain = (self.current_state.net_worth - self.initial_state.net_worth) / self.initial_state.net_worth

        # Regularization term (avoid that much changes in the portfolio)
        #regularization = np.mean(np.abs(next_state.portfolio_distribution - previous_state.portfolio_distribution))

        return next_gain - previous_gain # - regularization

    def step(self, action: dict[str, Any]) -> Tuple[np.ndarray, float, bool]:
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

        observation = self.dataset[self.iteration]

        # update the state
        self.last_state = self.current_state.copy()
        self.current_state = MarketEnvState(
            balance=self.current_state.balance,
            history=observation,
            portfolio=self.current_state.portfolio,
            done=False)

        # execute the action
        difference =  self.current_state.net_distribution - action['distribution']

        # sell
        sell = np.zeros_like(difference)
        sell[difference < 0] = -difference[difference < 0]
        sell = self.current_state.net_worth  * sell / self.current_state.prices # covert to quantity
        self.broker.sell(self.current_state, sell)

        # buy
        buy = np.zeros_like(difference)
        buy[difference > 0] = difference[difference > 0]
        # normalize the buy vector to add up to 1
        # (because selling is alters the net worth because of the transaction cost)
        buy = np.normalize(buy, norm=1)
        buy = self.current_state.balance * buy / self.current_state.prices # covert to quantity
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
