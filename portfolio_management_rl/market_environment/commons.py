"""
Utility objects and functions for market environment.
"""

from dataclasses import asdict, dataclass

import numpy as np
import torch


@dataclass()
class MarketEnvState:
    """
    State of the market environment.

    Attributes:
        balance (float): Current balance (in the base currency)
        history (np.ndarray): History of the market (in the base currency).
        portfolio (np.ndarray): Quantity of stocks.
        done (bool): Whether the episode is done or not.
    """

    history: np.ndarray | torch.Tensor  # (n stocks + 1 balance, n time steps)
    portfolio: np.ndarray | torch.Tensor  # (n stocks + 1 balance)
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
        if isinstance(self.history, torch.Tensor):
            return MarketEnvState(
                history=self.history.clone(),
                portfolio=self.portfolio.clone(),
                done=self.done,
            )

        return MarketEnvState(
            history=self.history.copy(),
            portfolio=self.portfolio.copy(),
            done=self.done,
        )

    @property
    def balance(self) -> float:
        """
        Returns the balance of the portfolio.

        Returns:
            float: Balance of the portfolio.
        """
        res = self.portfolio[-1]
        if isinstance(res, torch.Tensor):
            return res.item()
        return res

    @balance.setter
    def balance(self, value: float):
        """
        Sets the balance of the portfolio.

        Args:
            value (float): New balance.
        """
        self.portfolio[-1] = value

    @property
    def prices(self) -> np.ndarray | torch.Tensor:
        """
        Returns the prices of the market.

        Returns:
            np.ndarray: Prices of the market.
        """
        return self.history[:, -1]

    @property
    def shares(self) -> np.ndarray | torch.Tensor:
        """
        Returns the shares of the portfolio.

        Returns:
            np.ndarray: Shares of the portfolio.
        """
        return self.portfolio[:-1]

    @shares.setter
    def shares(self, value: np.ndarray | torch.Tensor):
        """
        Sets the shares of the portfolio.

        Args:
            value (np.ndarray): New shares.
        """
        if value.shape != self.portfolio[:-1].shape:
            raise ValueError(
                "Shape of the value does not match the shape of the shares."
            )
        self.portfolio[:-1] = value

    @property
    def investment(self) -> float:
        """
        Returns the investment of the portfolio.

        Returns:
            float: Investment of the portfolio.
        """
        res = self.portfolio[:-1].dot(self.prices)
        if isinstance(res, torch.Tensor):
            return res.item()
        return res

    @property
    def net_worth(self) -> float:
        """
        Returns the net worth of the portfolio.

        Returns:
            float: Net worth of the portfolio.
        """
        if isinstance(self.portfolio, np.ndarray):
            prices = np.append(self.prices, 1)  # balance
            return np.sum(self.portfolio * prices)

        return self.portfolio[:-1].dot(self.prices).item() + self.balance

    @property
    def net_distribution(self) -> np.ndarray:
        """
        Returns the distribution of the portfolio (including the balance).

        Returns:
            np.ndarray: Distribution of the portfolio (including the balance)
        """

        # add 1 to the prices vector
        prices = np.append(self.prices, 1)  # balance

        return (self.portfolio * prices) / self.net_worth
