"""
This module contains the Efficient Frontier rebalancing agent. This agent applies markowitz classical portfolio theory 
in order to rebalance our portfolio every time step to have the weights that are on the
efficient frontier.
"""

from typing import Union

import mlflow
import numpy as np
from gym.spaces.dict import Dict
from pypfopt import expected_returns
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt.risk_models import (
    CovarianceShrinkage,
    fix_nonpositive_semidefinite,
    risk_matrix,
    semicovariance,
)
from torch import Tensor

from portfolio_management_rl.market_environment.market_env import MarketEnvState

from .base import BaseAgent


class EfficientFrontierAgent(BaseAgent):
    """
    This agent uses the Efficient Frontier in order to rebalance our portfolio every time step to have the weights
    that are on the efficient frontier. Also known as Markowitz Portfolio Optimization.
    """

    def __init__(
        self,
        output_shape: tuple = (1, 101),
        risk_free_rate: float = 0.02,
        expected_returns_method: str = "mean_historical_return",
        risk_matrix_method: str = "ledoit_wolf",
        objective_function: str = "sharpe_ratio",
    ):
        """
        Initializes the agent.

        Args:
            output_shape (tuple): Shape of the output action.
            risk_free_rate (float): Risk free rate. This is used to calculate the Sharpe ratio.
            expected_returns_method (str): Method to calculate the expected returns.
            fix_nonpositive_semidefinite (bool): Whether to fix the covariance matrix to be positive semidefinite.
            risk_matrix_method (str): Method to calculate the risk matrix.
            risk_matrix_shrinkage (float): Shrinkage intensity for the risk matrix.
        """
        self.output_shape = output_shape
        self.risk_free_rate = risk_free_rate
        self.expected_returns_method = expected_returns_method
        self.risk_matrix_method = risk_matrix_method
        self.objective_function = objective_function

        self.model: EfficientFrontier = None
        self.expected_returns = None
        self.risk_matrix = None  # usually covariance matrix
        self.weights = None

    @property
    def parameters(self) -> dict:
        """
        Returns a Dictionary of parameters
        """
        return {
            "output_shape": self.output_shape,
            "risk_free_rate": self.risk_free_rate,
            "expected_returns": self.expected_returns_method,
            "risk_matrix_method": self.risk_matrix_method,
            "objective_function": self.objective_function,
        }

    def action(self, state: MarketEnvState) -> Dict:
        """
        Takes in a state and returns an action.

        Args:
            state (dict): State dictionary.

        Returns:
            dict: Action dictionary.
        """

        self.fit(state)

        return {"distribution": self.weights}

    def update(
        self,
        state: MarketEnvState,
        action: Dict,
        reward: Union[float, np.ndarray, Tensor],
        next_state: MarketEnvState,
    ) -> None:
        pass

    def log(self, **kwargs) -> None:
        """
        Logs the agent in mlflow.

        Args:
            path (str): Path to log the agent to.
        """
        mlflow.log_params(self.parameters)

    def compute_risk_matrix(self, prices: np.ndarray) -> np.ndarray:
        """
        Calculates the risk matrix.

        Args:
            prices (np.ndarray): Prices array.

        Returns:
            np.ndarray: Risk matrix.
        """
        risk: np.ndarray = None
        match self.risk_matrix_method:
            case "ledoit_wolf":
                risk = CovarianceShrinkage(prices).ledoit_wolf()

            case "shrunk_covariance":
                risk = CovarianceShrinkage(prices).shrunk_covariance()

            case "semicovariance":
                risk = semicovariance(prices)

            case "risk_matrix":
                risk = risk_matrix(prices)

            case _:
                raise ValueError(
                    f"Risk matrix method {self.risk_matrix_method} not supported."
                )

        risk = fix_nonpositive_semidefinite(risk)
        return risk

    def compute_expected_returns(self, prices: np.ndarray) -> np.ndarray:
        """
        Calculates the expected returns.

        Args:
            prices (np.ndarray): Prices array.

        Returns:
            np.ndarray: Expected returns.
        """

        match self.expected_returns_method:
            case "mean_historical_return":
                return expected_returns.mean_historical_return(prices)

            case "ema_historical_return":
                return expected_returns.ema_historical_return(prices)

            case "capm_return":
                return expected_returns.capm_return(
                    prices, risk_free_rate=self.risk_free_rate
                )
            case _:
                raise ValueError(
                    f"Expected returns method {self.expected_returns_method} not supported."
                )

    def fit(self, state: MarketEnvState) -> None:
        """
        Fits the agent to the state.

        Args:
            state (dict): State dictionary.
        """

        self.expected_returns = self.compute_expected_returns(state.prices)
        self.risk_matrix = self.compute_risk_matrix(state.prices)

        self.model = EfficientFrontier(
            self.expected_returns,
            self.risk_matrix,
            weight_bounds=(0, 1),
        )

        match self.objective_function:
            case "sharpe_ratio":
                self.model.max_sharpe(risk_free_rate=self.risk_free_rate)

            case "min_volatility":
                self.model.min_volatility()

            case "quadratic_utility":
                self.model.max_quadratic_utility()

            case _:
                raise ValueError(
                    f"Objective function {self.objective_function} not supported."
                )

        self.weights = self.model.clean_weights()
