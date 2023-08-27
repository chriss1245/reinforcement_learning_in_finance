"""
Datasets
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Union

import gym
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from portfolio_management_rl.utils.contstants import (
    DATA_DIR,
    FORECAST_HORIZON,
    N_STOCKS,
    WINDOW_SIZE,
)
from portfolio_management_rl.utils.dtypes import Phase
from portfolio_management_rl.utils.logger import get_logger

from .utils import PortfolioDistributionSpace

logger = get_logger(__file__)


@dataclass
class MixUpConfig:
    """
    Configuration for the mixup augmentation.
    """

    mixup_sequences: tuple = ("close", "open", "mean", "mean_adj", "extreme_mean")
    sequences_prob: tuple = (0.25, 0.25, 0.13, 0.12, 0.25)
    mixup_alpha: float = 0.2
    mixup_prob: float = 0.5


class StocksDataset(Dataset):
    """
    Stocks dataset generator. Generates a dataset of observation_horizon_days
    days of data. And the next (forecast_horizon_days) day as the target.
    """

    n_stocks = N_STOCKS
    window_size = WINDOW_SIZE
    forecast_horizon = FORECAST_HORIZON

    def __init__(
        self,
        data_dir: Path = DATA_DIR / "sp500/processed",
        phase: Phase = Phase.TRAIN,
        step_size_days: float = FORECAST_HORIZON,
        offset: int = 0,
        mixup_config: Optional[MixUpConfig] = None,
    ):
        """
        Stocks Dataset generator.

        Args:
            data: Dataframe with the data (each column is a company).
            observation_horizon_days: Number of days to use as input.
            forecast_horizon_days: Number of days to use as target.
            step_size_days: Number of days to jump between samples.
        """

        if mixup_config and phase != Phase.TRAIN:
            raise ValueError(
                "Mixup can only be applied to the training dataset. "
                "Please set the phase to train."
            )

        if offset > step_size_days:
            logger.warning(
                f"The offset ({offset}) is greater than the step size ({step_size_days})."
            )

        data_dir = data_dir / phase.value

        # Main signal
        self.data = pd.read_csv(
            data_dir / "adj_close.csv", index_col=0, parse_dates=True
        )
        self.data = self.data.iloc[offset:]
        if self.data.isna().sum().sum() > 0:
            raise ValueError("The data contains nan values")

        self.step_size_days = step_size_days
        self.jump_days = self.window_size + self.forecast_horizon - 1

        self.companies = list(self.data.columns)

        self.len = (len(self.data) - self.jump_days) // self.step_size_days

        # Mixup signals
        self.mixup_data = {}
        self.mixup_config = mixup_config
        if mixup_config:
            for sequence in mixup_config.mixup_sequences:
                self.mixup_data[sequence] = pd.read_csv(
                    data_dir / f"{sequence}.csv", index_col=0, parse_dates=True
                )
                self.mixup_data[sequence] = self.mixup_data[sequence].iloc[offset:]
                if self.mixup_data[sequence].isna().sum().sum() > 0:
                    raise ValueError("The data contains nan values")

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        if idx >= self.len:
            raise IndexError("Index out of range")

        idx *= self.step_size_days
        x = self.data[
            idx : idx + self.window_size
        ].values.T  # observation_horizon_days days

        y = self.data[idx + self.jump_days - 1 : idx + self.jump_days].values.T  # 1 day

        if self.mixup_config and np.random.uniform() < self.mixup_config.mixup_prob:
            # Select a random sequence
            sequence = np.random.choice(
                self.mixup_config.mixup_sequences,
                p=self.mixup_config.sequences_prob,
            )
            x_mixup = self.mixup_data[sequence][idx : idx + self.window_size].values.T
            y_mixup = self.mixup_data[sequence][
                idx + self.jump_days - 1 : idx + self.jump_days
            ].values.T

            # Mixup
            x = (
                self.mixup_config.mixup_alpha * x
                + (1 - self.mixup_config.mixup_alpha) * x_mixup
            )
            y = (
                self.mixup_config.mixup_alpha * y
                + (1 - self.mixup_config.mixup_alpha) * y_mixup
            )

        return x, y

    def get_date(self, idx: int) -> pd.Timestamp:
        """
        Returns the date which  belongs to the idx sample.

        Args:
            idx: Index of the sample.

        Returns:
            Date of the sample.
        """
        return self.data.index[
            (idx * self.step_size_days + self.window_size - 1) : (
                idx * self.step_size_days + self.window_size
            )
        ][0]

    def get_dates(self) -> list[pd.Timestamp]:
        """
        Returns the dates of the samples.
        """
        return [self.get_date(idx) for idx in range(self.len - 1)]

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
            shape=(len(self.companies), self.window_size),
        )
        return action_space, observation_space

    @staticmethod
    def get_datasets(
        data_dir: Path = DATA_DIR / "sp500/processed",
        step_size_days: float = FORECAST_HORIZON,
        mixup_config: Optional[MixUpConfig] = MixUpConfig(),
        return_dict: bool = False,
    ) -> Union[Tuple[Dataset, Dataset, Dataset], dict[str, Dataset]]:
        """
        Returns the datasets for the train, validation and test phases.

        Args:
            data_dir: Directory where the data is stored.
            step_size_days: Number of days to jump between samples.
            mixup_config: Mixup configuration.
            return_dict: If True, returns a dictionary with the datasets.
        """
        train_dataset = StocksDataset(
            data_dir=data_dir,
            step_size_days=step_size_days,
            mixup_config=mixup_config,
            phase=Phase.TRAIN,
        )
        val_dataset = StocksDataset(
            data_dir=data_dir, phase=Phase.VAL, step_size_days=step_size_days
        )
        test_dataset = StocksDataset(
            data_dir=data_dir, phase=Phase.TEST, step_size_days=step_size_days
        )

        if return_dict:
            return {
                "train": train_dataset,
                "val": val_dataset,
                "test": test_dataset,
            }
        return train_dataset, val_dataset, test_dataset
