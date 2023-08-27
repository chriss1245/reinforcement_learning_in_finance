"""
This module contains the required methods for data gathering
"""

import re
from abc import ABC
from pathlib import Path
from typing import Optional
import shutil

import pandas as pd
import yfinance as yf

from portfolio_management_rl.utils.contstants import DATA_DIR
from portfolio_management_rl.utils.logger import get_logger

logger = get_logger(__file__)


class Downloader(ABC):
    """
    Common interface for Data Downloaders
    """

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
    __risk_free_assets = [
        {
            "ticker": "^FVX",
            "name": "USA treasury bond 5Y (FVX)",
            "sector": "Bonds",
            "industry": None,
            "date_added": "1962-01-02",
            "date_removed": None,
            "is_in_index": False,
        },
        {
            "ticker": "^GSPC",
            "name": "S&P 500 index",
            "sector": "Index",
            "industry": None,
            "date_added": "1928-01-02",
            "date_removed": None,
            "is_in_index": False,
        },
    ]

    def __init__(
        self,
        initial_date: str = "1983-12-31",
        cleanup: bool = True,
    ):
        """
        Historical Price Downloader for the s&p 500

        Args:
            initial_date: The initial date to download the data
            cleanup: Whether to delete the raw data after the download
        """

        self.initial_date = initial_date
        self.cleanup = cleanup

        # create the parent dir
        root = DATA_DIR / self.__folder_name
        root.mkdir(exist_ok=True)
        (root / "all").mkdir(exist_ok=True)
        (root / "risk_free").mkdir(exist_ok=True)

    def download(self):
        """
        Downloads the data according to the givem initialization
        """

        # download the indexes
        indexes_df = pd.DataFrame.from_dict(self.__risk_free_assets)
        indexes_df.to_csv(DATA_DIR / f"{self.__folder_name}/risk_free/companies.csv")

        # download the companies using yfinance and wikipedia
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

        all_companies_df.to_csv(DATA_DIR / f"{self.__folder_name}/all/companies.csv")

        download_df = pd.concat([all_companies_df, indexes_df])

        errors = 0
        companies_error = []
        names_error = []

        landing_dir = DATA_DIR / f"{self.__folder_name}/raw_prices"
        landing_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Downloading the data")
        for _, (company, name) in enumerate(
            zip(download_df["ticker"], download_df["name"])
        ):
            path = landing_dir / f"{company}.csv"

            # check if the data is already downloaded
            if path.exists():
                continue
            try:
                ticker = yf.Ticker(company)
                data = yf.download(company, start=self.initial_date, interval="1d")

                if len(data) == 0:
                    data = yf.download(
                        company.replace(".", "-"),
                        start=self.initial_date,
                        interval="1d",
                    )

                if len(data) > 0:
                    data.to_csv(path)
                else:
                    errors += 1
                    companies_error.append(company)
                    names_error.append(name)

            except yf.exceptions.YFinanceException:
                errors += 1
                companies_error.append(company)
                names_error.append(name)

        logger.info(f"Errors: {errors}. Writting companies_not_found.txt")
        # write a txt file with the companies that could not be found
        with open(
            DATA_DIR / f"{self.__folder_name}/companies_not_found.txt",
            "w",
            encoding="utf-8",
        ) as f:
            for company, name in zip(companies_error, names_error):
                f.write(f"{company}: {name}\n")

        logger.info("Creating the dataframes")
        for df, name in zip([all_companies_df, indexes_df], ["all", "risk_free"]):
            close_df = pd.DataFrame(columns=df.ticker.tolist())
            adj_close_df = pd.DataFrame(columns=df.ticker.tolist())
            open_df = pd.DataFrame(columns=df.ticker.tolist())
            high_df = pd.DataFrame(columns=df.ticker.tolist())
            low_df = pd.DataFrame(columns=df.ticker.tolist())

            for ticker in df.ticker.tolist():
                try:
                    temp_df = pd.read_csv(
                        landing_dir / f"{ticker}.csv", parse_dates=True
                    )
                except FileNotFoundError:
                    logger.warning(f"File {ticker}.csv not found")
                    continue

                # filter out before
                temp_df = temp_df[temp_df["Date"] >= self.initial_date]

                temp_df = temp_df.set_index("Date")
                close_df[ticker] = temp_df["Close"]
                if not "Adj Close" in temp_df.columns:
                    temp_df["Adj Close"] = temp_df["Close"]
                adj_close_df[ticker] = temp_df["Adj Close"]
                open_df[ticker] = temp_df["Open"]
                high_df[ticker] = temp_df["High"]
                low_df[ticker] = temp_df["Low"]

            close_df.to_csv(DATA_DIR / f"{self.__folder_name}/{name}/close.csv")
            adj_close_df.to_csv(DATA_DIR / f"{self.__folder_name}/{name}/adj_close.csv")
            open_df.to_csv(DATA_DIR / f"{self.__folder_name}/{name}/open.csv")
            high_df.to_csv(DATA_DIR / f"{self.__folder_name}/{name}/high.csv")
            low_df.to_csv(DATA_DIR / f"{self.__folder_name}/{name}/low.csv")

        if self.cleanup:
            shutil.rmtree(landing_dir)

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
        logger.info(f"Companies currentlcy listed in the sp500 index: {len(companies)}")

        temp = changes_df.Added.Ticker.to_list()
        temp += changes_df.Removed.Ticker.to_list()

        # all the companies that formed part in the sp500 index (wikipedia)
        all_companies = companies + temp

        # delete duplicates
        all_companies = list(dict.fromkeys(all_companies))
        logger.info(
            f"All companies that formed part in the sp500 index: {len(all_companies)}"
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
                    except IndexError:
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
                        companies_dict[company][
                            "date_removed"
                        ] = SP500Downloader.convert_date(date)
                        companies_dict[company]["reason_removed"] = changes_df[
                            changes_df["Removed"]["Ticker"] == company
                        ]["Reason"].values[0][0]
                    except KeyError as e:
                        counter_missing_removed += 1
                        logger.debug(
                            f"Could not find date removed for {company}:"
                            + f"current count: {counter_missing_removed}, err: {e}"
                        )

                    try:
                        date = changes_df[changes_df["Added"]["Ticker"] == company][
                            "Date"
                        ].values[0][0]
                        companies_dict[company][
                            "date_added"
                        ] = SP500Downloader.convert_date(date)
                    except KeyError:
                        counter_missing_added += 1
                        logger.info(
                            f"could not find date added for {company} current count: {counter_missing_added}"
                        )
                except IndexError as e:
                    logger.debug(f"could not find company {company} in changes_df")
                    logger.debug(e)

        return companies_dict

    @staticmethod
    def convert_date(date):
        """
        It transfroms string dates to the format YYYY-MM-DD

        Args:
            date: a string date in the format MMM DD, YYYY where MMM is the month in letters

        Returns:
            A string date in the format YYYY-MM-DD
        """
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
        """
        Fixes issues with some dates

        Args:
            date: a string date in the format MMM DD, YYYY where MMM is the month in letters

        Returns:
            A string date in the format YYYY-MM-DD
        """
        if date == "2001?":
            return "2001-01-01"
        return date


if __name__ == "__main__":
    downloader = SP500Downloader()
    downloader.download()
