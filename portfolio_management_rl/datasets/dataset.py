"""
Datasets
"""
import gym
import pandas as pd
from torch.utils.data import Dataset

from .utils import PortfolioDistributionSpace


class StocksDataset(Dataset):
    """
    Stocks dataset generator. Generates a dataset of observation_horizon_days
    days of data. And the next (forecast_horizon_days) day as the target.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        observation_horizon_days: float = 730,
        forecast_horizon_days: float = 30,
        step_size_days: float = 30,
    ):
        """
        Stocks Dataset generator.

        Args:
            data: Dataframe with the data (each column is a company).
            observation_horizon_days: Number of days to use as input.
            forecast_horizon_days: Number of days to use as target.
            step_size_days: Number of days to jump between samples.
        """
        self.data = data
        self.observation_horizon_days = observation_horizon_days
        self.forecast_horizon_days = forecast_horizon_days
        self.step_size_days = step_size_days
        self.jump_days = observation_horizon_days + forecast_horizon_days

        self.companies = list(data.columns)

    def __len__(self):
        """
        Lenght of the data. taking into account the observation_horizon_days, the forecast_horizon_days
        and the step_size_days.
        """
        return (len(self.data) - self.jump_days) // self.step_size_days

    def __getitem__(self, idx):
        idx *= self.step_size_days
        x = self.data[
            idx : idx + self.observation_horizon_days
        ].values.T  # observation_horizon_days days
        y = self.data[idx + self.jump_days - 1 : idx + self.jump_days].values.T  # 1 day
        return x, y

    def get_compny_names(self):
        """
        Returns the names of the companies in the dataset.
        """
        return self.companies

    def get_action_observation_space(self) -> (gym.spaces.Box, gym.spaces.Box):
        """
        Returns the action and observation space for the environment.
        """
        # One proportion of capital per company + not invest
        action_space = PortfolioDistributionSpace(
            low=0, high=1, shape=(len(self.companies) + 1,)
        )

        # The observation space is the current price of each company (in dollars)
        observation_space = gym.spaces.Box(
            low=0,
            high=1e7,
            shape=(len(self.companies), self.observation_horizon_days),
        )
        return action_space, observation_space
