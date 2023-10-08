"""
Custom metrics for neural networks.

"""

import torch
import torchmetrics


class FlexibleMAPE(torchmetrics.Metric):
    """
    A flexible MAPE metric. It can be used to calculate the MAPE of all the errors, or only the positive or negative
    errors.
    """

    def __init__(self, mode="all"):
        super(FlexibleMAPE, self).__init__()
        self.mode = mode
        self.add_state("sum_error", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        if self.mode == "positive":
            idx = y_true > 0
        elif self.mode == "negative":
            idx = y_true < 0
        else:
            idx = torch.ones_like(y_true, dtype=torch.bool)

        percentage_error = 100 * torch.abs(
            (y_true[idx] - y_pred[idx]) / (y_true[idx] + torch.finfo(torch.float32).eps)
        )
        self.sum_error += torch.sum(percentage_error)
        self.count += len(percentage_error)

    def compute(self):
        return self.sum_error.float() / self.count


class FlexibleMAE(torchmetrics.Metric):
    """
    A flexible MAE metric. It can be used to calculate the MAE of all the errors, or only the positive or negative
    """

    def __init__(self, mode="all"):
        super().__init__()
        self.mode = mode
        self.add_state("sum_error", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        if self.mode == "positive":
            idx = y_true > 0
        elif self.mode == "negative":
            idx = y_true < 0
        else:
            idx = torch.ones_like(y_true, dtype=torch.bool)

        error = torch.abs(y_true[idx] - y_pred[idx])
        self.sum_error += torch.sum(error)
        self.count += len(error)

    def compute(self):
        return self.sum_error.float() / self.count
