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

    def buy(self, state: MarketEnvState, quantity: np.ndarray) -> np.ndarray:
        """
        Buys stocks.

        Args:
            balance (float): Current balance.

        Returns:
            np.ndarray: Bought stocks.
        """
        raise NotImplementedError

    def sell(self, state: MarketEnvState, quantity: np.ndarray) -> float:
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

    def __init__(self, fx_proportion: float = 0.7):
        """
        Initializes the strategy.

        Args:
            proportion_profit (float): Proportion of the profit that is charged as comission.
        """
        self.fx_proportion = fx_proportion / 100

    def get_quantity_to_buy(
        self, state: MarketEnvState, proportions: float
    ) -> np.ndarray:
        """
        Returns the quantity of stocks to buy.

        Args:
            proportion (float): Proportion of the balance to use to buy stocks.

        Returns:
            np.ndarray: Quantity of stocks to buy.
        """

        # takes into account the transaction cost and the fx rates
        quantity = (
            state.balance
            * proportions
            / (state.prices * (1 + self.fx_proportion) + PRICE_EPSILON + 1e-6)
        )
        return quantity

    def buy(self, state: MarketEnvState, quantity: np.ndarray | torch.Tensor) -> float:
        """
        Buys stocks. Adding the stocks to the portfolio and substracting the money from the balance.

        Args:
            balance (float): Current balance.

        Returns:
            cashflow (float): Money spent buying the stocks (negative).
        """

        if isinstance(quantity, torch.Tensor):
            total_price = torch.sum(
                quantity * (state.prices * (1 + self.fx_proportion) + PRICE_EPSILON)
            )

            if total_price > state.balance:
                raise ValueError("Not enough balance to buy the stocks.")

            state.balance = state.balance - total_price
            state.shares = state.shares + quantity

            return -total_price

        total_price = np.sum(
            quantity * (state.prices * (1 + self.fx_proportion) + PRICE_EPSILON)
        )

        if total_price > state.balance:
            raise ValueError("Not enough balance to buy the stocks.")

        state.balance = state.balance - total_price
        state.shares = state.shares + quantity

        return -total_price

    def sell(self, state: MarketEnvState, quantity: np.ndarray | torch.Tensor) -> float:
        """
        Sell stocks by a given quatity

        Args:
            quamtity (np.ndarray): Quantity of stocks to sell.

        Returns:
            float: Cashflow obtained by selling the stocks.
        """

        if isinstance(quantity, np.ndarray):
            if np.any(quantity > state.shares):
                raise ValueError("Not enough stocks to sell.")

            total_price = quantity.dot(
                (state.prices)
            )  # total price of the stocks to sell
            state.shares = state.shares - quantity
            state.balance = state.balance + total_price * (1 - self.fx_proportion)

            return total_price * (1 - self.fx_proportion)

        if isinstance(quantity, torch.Tensor):
            if torch.any(quantity > state.shares):
                raise ValueError("Not enough stocks to sell.")

            total_price = torch.dot(quantity, state.prices)
            state.shares -= quantity
            state.balance += total_price * (1 - self.fx_proportion)

            return total_price * (1 - self.fx_proportion)

        raise ValueError("Quantity must be a torch.Tensor or np.ndarray")
