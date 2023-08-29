"""
Callbacks for the neural network.
"""

from abc import ABC, abstractmethod
from typing import Optional, Union

from portfolio_management_rl.agents.deep_learning_agents.nn.dtypes import TrainerState
from portfolio_management_rl.agents.deep_learning_agents.nn.utils import (
    save_checkpoint,
    load_checkpoint,
)
from portfolio_management_rl.utils.contstants import LOGS_DIR
from torch.optim.lr_scheduler import ReduceLROnPlateau

import json

from pathlib import Path
import datetime as dt

from torch.utils.tensorboard import SummaryWriter


class BaseCallback(ABC):
    """
    Callback interface

    Methods:
        on_epoch_start: Called when the epoch begins
        on_epoch_end: Called when the epoch ends
    """

    @abstractmethod
    def on_epoch_start(self, state: TrainerState) -> None:
        """
        Called when the epoch begins

        Args:
            trainer_state: Current state of the trainer
        """

    @abstractmethod
    def on_epoch_end(self, state: TrainerState) -> None:
        """
        Called when the epoch ends

        Args:
            trainer_state: Current state of the trainer
        """


class EarlyStopExcpetion(Exception):
    """
    Early stopping exception to stop the training
    """


class EarlyStoping(BaseCallback):
    """
    Creates an early stoping callback which raises a EarlyStopException

    Args:
        monitor (str): The metric used as reference
        patience (int): How many epochs wait if the model does not improve
        is_loss (bool): If True, the model will be saved if the metric is greater than the best.
            If False, the model will be saved if the metric is lower than the best

    Raises:
        EarlyStopExcpetion: If the model does not improve for the specified number of epochs

    """

    def __init__(self, monitor: str, patience: int = 2, is_loss: bool = True):
        self.monitor = monitor
        self.patience = patience
        self.patience_counter = 0
        self.is_loss = is_loss
        self.best = float("inf") if self.is_loss else float("-inf")

    def on_epoch_start(self, state: TrainerState) -> None:
        pass

    def on_epoch_end(self, state: TrainerState) -> None:
        if self.is_loss:
            if state.metrics[self.monitor] < self.best:
                self.best = state.metrics[self.monitor]
                self.patience_counter = 0
            else:
                self.patience_counter += 1
        else:
            if state.metrics[self.monitor] > self.best:
                self.best = state.metrics[self.monitor]
                self.patience_counter = 0
            else:
                self.patience_counter += 1

        if self.patience_counter >= self.patience:
            raise EarlyStopExcpetion()


class Checkpoint(BaseCallback):
    """
    A callback to save one or more models during the training
    """

    checkpoint_dir = "m{metric:.4f}_e{epoch:03d}"

    def __init__(
        self,
        checkpoints_dir: Path,
        best_only: bool = True,
        monitor: Optional[str] = None,
        is_loss: Optional[bool] = True,
    ):
        checkpoints_dir.mkdir(exist_ok=True)

        self.checkpoints_dir = checkpoints_dir
        self.best_only = best_only
        self.monitor = monitor
        self.is_loss = is_loss
        self.best = float("inf") if self.is_loss else float("-inf")

    def on_epoch_start(self, state: TrainerState) -> None:
        pass

    def on_epoch_end(self, state: TrainerState) -> None:
        if self.monitor is not None:
            if self.is_loss:
                if state.metrics[self.monitor] < self.best:
                    self.best = state.metrics[self.monitor]
                else:
                    return
            else:
                if state.metrics[self.monitor] > self.best:
                    self.best = state.metrics[self.monitor]
                else:
                    return

        dir_ = self.checkpoints_dir / self.checkpoint_dir.format(
            epoch=state.epoch, metric=state.metrics[self.monitor]
        )

        dir_.mkdir(exist_ok=True)
        for name, model in state.networks.items():
            optimizer = state.optimizers.get(name)
            save_checkpoint(dir_ / f"{name}.pth", model, optimizer)

        # save the metadata of the trainer
        meta = {"epoch": state.epoch, "metrics": state.get_last_metrics()}

        with open(dir_ / "meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f)


class ReduceLROnPlateauCallback(BaseCallback):
    """
    Reduce the learning rate when the metric does not improve

    Args:
        monitor (str): The metric used as reference
        patience (int): How many epochs wait if the model does not improve
        factor (float): Factor to reduce the learning rate
        min_lr (float): Minimum learning rate
        is_loss (bool): If True, the model will be saved if the metric is greater than the best.
            If False, the model will be saved if the metric is lower than the best

    """

    def __init__(
        self,
        monitor: Union[str, dict[str, str]],
        patience: int = 2,
        factor: float = 0.1,
        min_lr: float = 1e-6,
        is_loss: bool = True,
    ):
        self.monitor = monitor
        self.patience = patience
        self.patience_counter = 0
        self.factor = factor
        self.min_lr = min_lr
        self.is_loss = is_loss

        self.schedulers = {}

    def on_epoch_start(self, state: TrainerState) -> None:
        for name, optimizer in state.optimizers.items():
            if name not in self.schedulers:
                self.schedulers[name] = ReduceLROnPlateau(
                    optimizer,
                    mode="min" if self.is_loss else "max",
                    patience=self.patience,
                    factor=self.factor,
                    min_lr=self.min_lr,
                )

    def on_epoch_end(self, state: TrainerState) -> None:
        for name, scheduler in self.schedulers.items():
            scheduler.step(state.metrics[self.monitor[name]])


class Tensorboard(BaseCallback):
    """
    Tensorboard callback to log the metrics
    """

    def __init__(self, log_dir: Path = LOGS_DIR, experiment_name: str = "experiment"):
        log_dir.mkdir(exist_ok=True)

        now = dt.datetime.now().strftime("%y%m%d%H%M")

        log_dir = log_dir / f"{experiment_name}_{now}"

        self.log_dir = log_dir
        self.writer = SummaryWriter(log_dir)

    def on_epoch_start(self, state: TrainerState) -> None:
        pass

    def on_epoch_end(self, state: TrainerState) -> None:
        for key, value in state.get_last_metrics().items():
            self.writer.add_scalar(key, value, state.epoch)
