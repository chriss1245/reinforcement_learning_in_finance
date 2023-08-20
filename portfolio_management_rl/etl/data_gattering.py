"""
This module contains the required methods for data gathering
"""

import re
from abc import ABC
from enum import Enum
from pathlib import Path
from typing import Optional

import pandas as pd
import yfinance as yf

from portfolio_management_rl.utils.contstants import DATA_DIR
from portfolio_management_rl.utils.logging import get_logger

logger = get_logger(__file__)


class DownloadStrategy(Enum):
    SURVIVED = "SURVIVED"  # only companies with all the data in the time interval
    ALL = "ALL"  # all the companies


class Granularity(Enum):
    HOURLY = "1h"
    DAILY = "1d"
    MONTHLY = "1m"


class Downloader(ABC):
    """
    Common interface for Data Downloaders
    """

    __folder_name = "data"

    def download(self, path: Path) -> None:
        """
        Downloads the data given the specifications

        Args:
            path: The parent directory where the dataset will be downloaded
        """


class SP500Downloader(Downloader):
    """
    This class is in charge of downloading the data of the s&p500 companies
    """

    __folder_name = "sp500"
    __months = {
        "january": "01",
        "february": "02",
        "march": "03",
        "april": "04",
        "may": "05",
        "june": "06",
        "july": "07",
        "august": "08",
        "september": "09",
        "october": "10",
        "november": "11",
        "december": "12",
    }

    # Add as initial asset the USA bonds of  5years
    __risk_free_asset = {
        "ticker": "^FVX",
        "name": "USA treasury bond 10Y",
        "sector": "Bonds",
        "industry": None,
        "date_added": "1962-01-02",
        "date_removed": None,
        "is_in_index": False,
    }

    def __init__(
        self,
        initial_date="1989-12-31",
        end_date: Optional[str] = None,
        strategy: DownloadStrategy = DownloadStrategy.ALL,
        granularity: Granularity = Granularity.DAILY,
    ):
        """
        Historical Price Downloader for the s&p 500
        """

        self.initial_date = initial_date
        self.end_date = end_date
        self.strategy = strategy
        self.granularity = granularity

        # create the parent dir
        (DATA_DIR / self.__folder_name).mkdir(exist_ok=True)

    def download(self):
        """
        Downloads the data according to the givem initialization
        """

        companies_df, changes_df = pd.read_html(
            "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        )

        companies_dict = self.get_companies_dict(changes_df, companies_df)
        all_companies_df = pd.DataFrame.from_dict(companies_dict, orient="index")
        all_companies_df.reset_index(names=["ticker"], inplace=True)
        all_companies_df.date_added = all_companies_df.date_added.apply(
            SP500Downloader.sanitize_date
        )
        all_companies_df.date_added = pd.to_datetime(
            all_companies_df.date_added, format="mixed"
        )
        all_companies_df.date_removed = pd.to_datetime(
            all_companies_df.date_removed, format="mixed"
        )

        if self.strategy == DownloadStrategy.ALL:
            all_companies_df.to_csv(DATA_DIR / f"{self.__folder_name}/all.csv")

        # Filter by date
        temp = all_companies_df[all_companies_df.date_added <= self.initial_date]
        if self.end_date:
            temp = temp[temp.data_added >= self.end_date]

        # The first asset is the risk free
        new = pd.DataFrame.from_dict([self.__risk_free_asset])
        temp = pd.concat([new, temp], ignore_index=True)
        temp.to_csv(DATA_DIR / f"{self.__folder_name}/all_survived.csv")

        errors = 0
        companies_error = []
        names_error = []
        found_data = []

        landing_dir = (
            DATA_DIR / f"{self.__folder_name}/raw_prices_{self.granularity.value}"
        )
        landing_dir.mkdir(parents=True, exist_ok=True)

        for i, (company, name) in enumerate(
            zip(all_companies_df["ticker"], all_companies_df["name"])
        ):
            path = landing_dir / f"{company}.csv"

            # check if the data is already downloaded
            if path.exists():
                continue
            try:
                data = yf.download(company, start="1985-01-01", interval="1d")

                if len(data) == 0:
                    data = yf.download(
                        company.replace(".", "-"), start="1985-01-01", interval="1d"
                    )

                if len(data) > 0:
                    data.to_csv(path)
                    found_data.append(True)
                else:
                    errors += 1
                    companies_error.append(company)
                    names_error.append(name)
                    found_data.append(False)
                    logger.warning(
                        f"could not find company {company}: {name} in yfinance"
                    )
            except:
                errors += 1
                companies_error.append(company)
                names_error.append(name)
                found_data.append(False)
                logger.warning(f"could not find company {company}: {name} in yfinance")

        # write a txt file with the companies that could not be found
        with open(DATA_DIR / f"{self.__folder_name}companies_not_found.txt", "w") as f:
            for company, name in zip(companies_error, names_error):
                f.write(f"{company}: {name}\n")

    def get_companies_dict(
        self, changes_df: pd.DataFrame, companies_df: pd.DataFrame
    ) -> dict:
        """
        This method creates a dataframe of all the companies, their metada and if possible
        when they were added to the s&p500 when they were removed (if they were).

        Args:
            all_companies: a list with all the companies
            changes_df: a dataframe read from wikipedia which contains the historical changes
            companies_df: a dataframe from wikipedia which does contain the list of the companies of the s&p500

        Returns:

            A dictionary of standardized assets with their metadata and their in dates and out dates.
        """

        # all the companies that formed part in the sp500 index (wikipedia)
        companies = companies_df["Symbol"].tolist()
        logger.info("companies currently listed in the sp500 index: ", len(companies))

        temp = changes_df.Added.Ticker.to_list()
        temp += changes_df.Removed.Ticker.to_list()

        # all the companies that formed part in the sp500 index (wikipedia)
        all_companies = companies + temp

        # delete duplicates
        all_companies = list(dict.fromkeys(all_companies))
        logger.info(
            "all companies that formed part in the sp500 index: ", len(all_companies)
        )

        counter_missing_removed = 0

        companies_dict = {}
        exclude = ["FB", "WL"]

        ticker_names_dict = {
            "CMCSK": "CMCSA",
            "ENDP": "ENDPQ",
        }

        companies = companies_df["Symbol"].tolist()

        for company in all_companies:
            if company in exclude:
                continue
            if company in ticker_names_dict:
                company = ticker_names_dict[company]
            if company in companies:
                companies_dict[company] = {
                    "name": companies_df[companies_df["Symbol"] == company][
                        "Security"
                    ].values[0],
                    "sector": companies_df[companies_df["Symbol"] == company][
                        "GICS Sector"
                    ].values[0],
                    "industry": companies_df[companies_df["Symbol"] == company][
                        "GICS Sub-Industry"
                    ].values[0],
                    "date_added": companies_df[companies_df["Symbol"] == company][
                        "Date added"
                    ].values[0],
                    "date_removed": None,
                    "reason_removed": None,
                    "is_in_index": True,
                }
            else:
                try:
                    try:
                        name = changes_df.Removed[
                            changes_df.Removed["Ticker"] == company
                        ]["Security"].values[0]
                    except:
                        name = changes_df.Added[changes_df.Added["Ticker"] == company][
                            "Security"
                        ].values[0]

                    # standardaize date format to YYYY-MM-DD

                    companies_dict[company] = {
                        "name": name,
                        "sector": None,
                        "industry": None,
                        "date_added": None,
                        "date_removed": None,
                        "reason_removed": None,
                        "is_in_index": False,
                    }
                    try:
                        date = changes_df[changes_df["Removed"]["Ticker"] == company][
                            "Date"
                        ].values[0][0]
                        companies_dict[company]["date_removed"] = convert_date(date)
                        companies_dict[company]["reason_removed"] = changes_df[
                            changes_df["Removed"]["Ticker"] == company
                        ]["Reason"].values[0][0]
                    except Exception as e:
                        counter_missing_removed += 1
                        logger.info(
                            f"could not find date removed for {company}: current count: {counter_missing_removed}"
                        )

                    try:
                        date = changes_df[changes_df["Added"]["Ticker"] == company][
                            "Date"
                        ].values[0][0]
                        companies_dict[company][
                            "date_added"
                        ] = SP500Downloader.convert_date(date)
                    except Exception as e:
                        counter_missing_added += 1
                        logger.info(
                            f"could not find date added for {company} current count: {counter_missing_added}"
                        )
                except Exception as e:
                    logger.warning(f"could not find company {company} in changes_df")
                    logger.warning(e)
        return companies_dict

    @staticmethod
    def convert_date(date):
        date = date.lower()
        # take out month, day, year
        month, day, year = re.findall(r"\w+", date)

        # convert month to number
        if len(month) == 1:
            month = "0" + month
        if len(day) == 1:
            day = "0" + day
        return f"{year}-{SP500Downloader.__months[month]}-{day}"

    @staticmethod
    def sanitize_date(date):
        if date == "2001?":
            return "2001-01-01"
        return date


if __name__ == "__main__":
    downloader = SP500Downloader()
    downloader.download()
