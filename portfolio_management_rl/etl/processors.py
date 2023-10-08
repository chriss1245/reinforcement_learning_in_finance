"""
This module contains tools for processing the historical data from stock prices.
"""

from pathlib import Path

import pandas as pd

from portfolio_management_rl.utils.contstants import (
    DATA_DIR,
    END_DATE,
    INITIAL_DATE,
    N_STOCKS,
    WINDOW_SIZE,
)
from portfolio_management_rl.utils.logger import get_logger

logger = get_logger(__file__)

PROCESSED_DATA_DIR = DATA_DIR / "sp500/processed"
DATA_DIR = DATA_DIR / "sp500/all"
YEAR_LEN = 252  # 252 trading days in a year
YEARS_TEST = 7
YEARS_VAL = 7


class Processor:
    """
    This class is in charge of processing the data. It creates a train, val and test dataset,
    removes the assets with nan values and saves the processed data.
    """

    def __init__(
        self,
        n_stocks: int = N_STOCKS,
        data_dir: Path = DATA_DIR,
        initial_date: str = INITIAL_DATE,
        end_date: str = END_DATE,
        years_test: int = YEARS_TEST,
        years_val: int = YEARS_VAL,
    ):
        self.n_stocks = n_stocks
        self.data_dir = data_dir
        self.initial_date = initial_date
        self.end_date = end_date
        self.years_test = years_test
        self.years_val = years_val

    def process(self):
        """
        Process the data. Creates a train, val and test dataset. The test dataset is over different companies.
        """
        PROCESSED_DATA_DIR.mkdir(exist_ok=True)
        all_nan = set()
        datasets = {}
        paths = [path for path in DATA_DIR.glob("*.csv") if path.stem != "companies"]
        for path in paths:
            df = pd.read_csv(path, index_col=0, parse_dates=True)
            df = df.loc[INITIAL_DATE:END_DATE]
            all_nan.update(df.columns[df.isna().sum() > 0])
            datasets[path.stem] = df

        # drop asses which have nan values
        for df in datasets.values():
            nan_cols = (df.isna().sum() > 0).values
            nan_df = df.loc[:, nan_cols]
            all_nan.update(nan_df.columns.to_list())

        # drop all nan columns
        for df in datasets.values():
            df.drop(columns=all_nan, inplace=True)

        logger.info(
            f"After dropping all nan columns, the companies left are: {len(datasets['close'].columns)}"
        )
        if len(datasets["close"].columns) < 2 * N_STOCKS:
            raise ValueError(
                f"After dropping all nan columns, the companies left are: {len(datasets['close'].columns)}. "
                f"Please increase the number of companies in the dataset."
            )

        # Create new datasets for augmented data
        datasets["extreme_mean"] = (datasets["low"] + datasets["high"]) / 2
        datasets["mean_adj"] = (datasets["open"] + datasets["adj_close"]) / 2
        datasets["mean"] = (datasets["open"] + datasets["close"]) / 2

        # Split the data into train, val and test
        logger.info("Splitting the data into train, val and test.")
        df = datasets["close"]
        final_date = df[-1:].index[0]
        initial_test_date = final_date - pd.DateOffset(years=7)
        initial_val_date = initial_test_date - pd.DateOffset(years=7)

        train_datasets = {}
        val_datasets = {}
        test_datasets = {}
        for name, df in datasets.items():
            train_datasets[name] = df.loc[:initial_val_date]

            # the val and test set have an overalp of window_size in order to have the historical for the first state
            val_datasets[name] = df.loc[
                initial_val_date - pd.DateOffset(years=3) : initial_test_date
            ]
            test_datasets[name] = df.loc[
                initial_test_date - pd.DateOffset(years=3) :
            ]  # 252 trading days in a year

        # Save the datasets
        (PROCESSED_DATA_DIR / "train").mkdir(exist_ok=True)
        (PROCESSED_DATA_DIR / "val").mkdir(exist_ok=True)
        (PROCESSED_DATA_DIR / "test").mkdir(exist_ok=True)

        for name, df in train_datasets.items():
            df.iloc[:, :N_STOCKS].to_csv(
                PROCESSED_DATA_DIR / f"train/{name}.csv", index=True
            )

        for name, df in val_datasets.items():
            df.iloc[:, :N_STOCKS].to_csv(
                PROCESSED_DATA_DIR / f"val/{name}.csv", index=True
            )

        # test dataset is over different companies
        for name, df in test_datasets.items():
            df.iloc[:, N_STOCKS : 2 * N_STOCKS].to_csv(
                PROCESSED_DATA_DIR / f"test/{name}.csv", index=True
            )

        logger.info("Processed data saved.")


if __name__ == "__main__":
    processor = Processor()
    processor.process()
