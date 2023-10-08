"""
Custom loss functions for neural networks.
"""

import torch
import torch.nn as nn


class WeightedMSE(nn.Module):
    """
    Weighted Mean Squared Error based on positive and negative predictions.
    """

    def __init__(self, positive_weight: float = 1.0, negative_weight: float = 3.0):
        super().__init__()
        self.positive_weight = positive_weight
        self.negative_weight = negative_weight

    def forward(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Calculates the weighted mean squared error.

        Args:
            y_hat (torch.Tensor): Predictions of the model.
            y (torch.Tensor): Ground truth values.

        Returns:
            torch.Tensor: Weighted mean squared error.
        """

        positive_idx = y > 0
        negative_idx = y < 0

        positive_mse = (
            torch.mean(torch.square(y_hat[positive_idx] - y[positive_idx]))
            * self.positive_weight
        )
        negative_mse = (
            torch.mean(torch.square(y_hat[negative_idx] - y[negative_idx]))
            * self.negative_weight
        )

        return positive_mse + negative_mse


class WeightedMAE(nn.Module):
    """
    Sign weighted Mean Absolute Error based on positive and negative predictions.
    """

    def __init__(self, positive_weight: float = 1.0, negative_weight: float = 3.0):
        super().__init__()
        self.positive_weight = positive_weight
        self.negative_weight = negative_weight

    def forward(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Calculates the weighted mean absolute error.

        Args:
            y_hat (torch.Tensor): Predictions of the model.
            y (torch.Tensor): Ground truth values.

        Returns:
            torch.Tensor: Weighted mean absolute error.
        """

        positive_idx = y > 0
        negative_idx = y < 0

        positive_mae = (
            torch.mean(torch.abs(y_hat[positive_idx] - y[positive_idx]))
            * self.positive_weight
        )
        negative_mae = (
            torch.mean(torch.abs(y_hat[negative_idx] - y[negative_idx]))
            * self.negative_weight
        )

        return positive_mae + negative_mae
