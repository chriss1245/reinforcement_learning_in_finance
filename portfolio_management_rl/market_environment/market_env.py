"""
This module emulates the market. It offers an interface to the agent to get the current state of the market and to execute
actions on the market.
"""

import pandas as pd
import numpy as np
import datetime
import gym
from enum import Enum
from torch.utils.data import Dataset


# Strategy pattern used to define the comission type
class BalanceUpdater():
    """
    Abstract class for the update balance strategy.
    """

    def update(self, balance: float, previous_proportion: np.ndarray, new_proportion: np.ndarray) -> float:
        """
        Updates the balance.

        Args:
            balance (float): Current balance.
            transaction_cost (float): Transaction cost.

        Returns:
            float: Updated balance.
        """
        raise NotImplementedError

class BalanceUpdaterTrading212(BalanceUpdater):
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

    def update(self, balance: float, previous_proportion: np.ndarray, new_proportion: np.ndarray) -> float:
        """
        Updates the balance.

        Args:
            balance (float): Current balance.
            transaction_cost (float): Transaction cost.

        Returns:
            float: Updated balance.
        """
        profit = new_proportion[:-1] - previous_proportion [:-1] # last element means not invests
        close_operations = profit < 0 # operations that are closing (selling)
        comissions = np.sum(np.abs(profit[close_operations]) * self.profit_proportion) # (defult 0.5%) comission for closing operations
        return balance + np.sum(profit) - np.sum(comissions)


class MarketEnvironmnet(gym.Env):
    """
    Market environment. Simulates the market and offers an interface to the agent to get the current state
    of the market and to execute actions on the market.
    """

    def __init__(self, dataset: Dataset, observation_space: gym.spaces.Box, action_space: gym.spaces.Box = PortfolioDistributionSpace(),
        initial_balance: float = 10000,
        balance_updater: BalanceUpdater =BalanceUpdaterTrading212())
        """
        Market environment. Simulates the market and offers an interface for
        portfolio management agents to get the current state of the market ands
        to execute actions on the market.

        Args:
            data (pd.DataFrame): Dataframe containing the market data.
            initial_date (datetime.datetime): Initial date of the market.
            step_size (int): Number of days to move forward in time when
                calling step().
            observation_horizon_days (int): Number of days to use as the
                observation horizon.
            
            initial_balance (float): Initial balance of the portfolio.
            transaction_cost (float): Transaction cost of the market.
        """

        # data generator
        self.dataset = dataset

        # balance
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.balance_updater = balance_updater

        # action space and observation space
        self.observation_space = observation_space
        self.action_space = action_space

        # initialize the last action to not invest
        self.last_action = np.zeros(shape=(self.action_space.shape[0],))
        self.last_action[-1] = 1 # last element means not invests
        


    def reset(self):
        """
        Resets the market environment to its initial state.
        """
        self.balance = self.initial_balance
        self.last_balance = self.initial_balance
        self.last_action = np.zeros(shape=(self.action_space.shape[0],))
        self.last_action[-1] = 1 # last element means not invests

        self.current_observation, self.next_observation = self.dataset[0]
        
        return self.current_observation


    def reward(self, balance: float,  action: dict) -> float:
        """
        Calculates the reward obtained by executing an action.

        """

        new_balance = self.balance_updater.update(self.balance, self.current_observation, action["action"])
        return new_balance/self.initial_balance # MAPE
        

    def step(self, action: dict):
        """
        Executes an action on the market. Returns the next observation, the reward
        and whether the episode has ended.

        reward = (future_balance - current_balance)/initial_balance - transaction_cost

        Args:
            action (dict): Dictionary containing the new action, and whether execute
                the action or not.
                {
                    "action": [0.1, 0.3, 0.6],
                    "execute": Trues
                }
        
        Returns:
            observation (np.array): The next observation.
            reward (float): The reward obtained by executing the action.
            done (bool): Whether the episode has ended.
        """

        if action["execute"]:


        else:
            reward = 0

        # Update the observation
        self.current_observation = self.next_observation
        self.next_observation = self.dataset[self.current_observation]

        # Check if the episode has ended
        done = self.current_observation == len(self.dataset) - 1

        return self.current_observation, reward, done