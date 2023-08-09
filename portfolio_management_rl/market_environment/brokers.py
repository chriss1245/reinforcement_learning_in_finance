"""
Class for brokers which are used to execute orders and calculate transaction costs.
"""

import numpy as np

from .commons import MarketEnvState


# Strategy pattern used to define the different Brokers (Trading212, Degiro, ...)
class Broker:
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
        total_price = quantity.dot(
            market_state.prices
        )  # total price of the stocks to buy

        if total_price > market_state.balance:
            raise ValueError("Not enough balance to buy the stocks.")

        market_state.balance -= total_price
        market_state.portfolio[:-1] += quantity

    def sell(self, market_state: MarketEnvState, quantity: np.ndarray) -> None:
        """
        Sell stocks by a given quatity

        Args:
            quamtity (np.ndarray): Quantity of stocks to sell.

        Returns:
            float: Money earned by selling the stocks.
        """

        if np.any(quantity > market_state.portfolio[:-1]):
            raise ValueError("Not enough stocks to sell.")

        total_price = quantity.dot(
            market_state.prices
        )  # total price of the stocks to sell
        market_state.portfolio[:-1] -= quantity
        market_state.balance += total_price * (1 - self.profit_proportion)
