"""
Class for brokers which are used to execute orders and calculate transaction costs.
"""

import numpy as np
import torch

from portfolio_management_rl.utils.contstants import PRICE_EPSILON

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

    # to avoid floating point errors  (the action price is greater than the current price as consequence)
    __epsilon__ = PRICE_EPSILON

    def __init__(self, profit_commision_percent: float = 0.5):
        """
        Initializes the strategy.

        Args:
            proportion_profit (float): Proportion of the profit that is charged as comission.
        """
        self.profit_proportion = profit_commision_percent / 100

    def buy(
        self, market_state: MarketEnvState, quantity: np.ndarray | torch.Tensor
    ) -> None:
        """
        Buys stocks. Adding the stocks to the portfolio and substracting the money from the balance.

        Args:
            balance (float): Current balance.

        Returns:
            np.ndarray: Bought stocks.
        """

        if isinstance(quantity, np.ndarray):
            total_price = quantity.dot(
                market_state.prices + self.__epsilon__
            )  # total price of the stocks to buy

            if total_price > market_state.balance:
                raise ValueError("Not enough balance to buy the stocks.")

            market_state.balance -= total_price
            market_state.shares += quantity

        else:
            total_price = torch.dot(
                quantity, market_state.prices + self.__epsilon__
            ).item()

            if total_price > market_state.balance:
                raise ValueError("Not enough balance to buy the stocks.")

            market_state.balance -= total_price
            market_state.shares += quantity

    def sell(
        self, market_state: MarketEnvState, quantity: np.ndarray | torch.Tensor
    ) -> None:
        """
        Sell stocks by a given quatity

        Args:
            quamtity (np.ndarray): Quantity of stocks to sell.

        Returns:
            float: Money earned by selling the stocks.
        """

        if isinstance(quantity, np.ndarray):
            if np.any(quantity > market_state.shares):
                raise ValueError("Not enough stocks to sell.")

            total_price = quantity.dot(
                (market_state.prices)
            )  # total price of the stocks to sell
            market_state.shares -= quantity
            market_state.balance += total_price * (1 - self.profit_proportion)
        else:
            if torch.any(quantity > market_state.shares):
                raise ValueError("Not enough stocks to sell.")

            total_price = torch.dot(quantity, market_state.prices)
            market_state.shares -= quantity
            market_state.balance += total_price * (1 - self.profit_proportion)
