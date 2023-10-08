"""
This module contains the forecasting tools for the project.

We aim to predict the future prices and imitate the behavior of the min volatilty agent
"""

from typing import Dict

from portfolio_management_rl.utils.contstants import (
    N_STOCKS,
    WINDOW_SIZE,
    FORECAST_HORIZON,
)
from portfolio_management_rl.utils.dtypes import Phase, Device

from portfolio_management_rl.dataloaders import StocksDataset

from portfolio_management_rl.nn.tcn import TemporalConvNet
from portfolio_management_rl.nn.metrics import FlexibleMAPE, FlexibleMAE
from portfolio_management_rl.nn.losses import WeightedMAE, WeightedMSE


from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    LearningRateMonitor,
)
from lightning.pytorch import LightningDataModule
import torch.nn as nn
import torch


class Forecaster(LightningModule):
    """
    Forecaster class. It is used to predict the future prices.
    """

    def __init__(
        self,
        n_stocks: int = N_STOCKS,
        window_size: int = WINDOW_SIZE,
        forecast_horizon: int = FORECAST_HORIZON,
        predict_sequence: bool = False,
        loss: str = "wmae",
    ):
        """
        Initialize the forecaster.

        Args:
            n_stocks (int, optional): Number of stocks. Defaults to N_STOCKS.
            window_size (int, optional): Window size. Defaults to WINDOW_SIZE.
            forecast_horizon (int, optional): Forecast horizon. Defaults to FORECAST_HORIZON.
        """
        super().__init__()
        self.n_stocks = n_stocks
        self.window_size = window_size
        self.forecast_horizon = forecast_horizon
        self.predict_sequence = predict_sequence

        losses = {
            "mae": nn.MSELoss,
            "mse": nn.L1Loss,
            "wmae": WeightedMAE,
            "wmse": WeightedMSE,
        }

        self.loss = losses[loss]()

        self.model = nn.Sequential()

        self.model = TemporalConvNet(
            input_size=n_stocks,
            kernel_size=3,
            num_filters=n_stocks,
            dilation_base=3,
            weight_norm=True,
            target_size=n_stocks + 1,
            dropout=0.35,
            window_size=window_size,
            # activation=nn.ReLU(),
        )

        self.positive_mape = FlexibleMAPE(mode="positive")
        self.negative_mape = FlexibleMAPE(mode="negative")
        self.all_mae = FlexibleMAE()
        self.positive_mae = FlexibleMAE(mode="positive")
        self.negative_mae = FlexibleMAE(mode="negative")

        self.all_mape = FlexibleMAPE(mode="all")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the forecaster.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        x = x.transpose(1, 2)
        if self.predict_sequence:
            y = self.model(x)  # (batch_size, seq_len, num_channels)
            # ignore the last channel
            return y[:, -self.forecast_horizon :, :-1].transpose(1, 2)
        else:
            y = self.model(x)[:, -1:, :]
            return y[:, :, :-1].transpose(1, 2)

    # (batch_size, 1, num_channels) -> (batch_size, num_channels)nn.L1Loss() if loss == "mae" else nn.MSELoss()

    def training_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """
        Training step of the forecaster.

        Args:
            batch (Dict[str, torch.Tensor]): Batch of data.
            batch_idx (int): Index of the batch.

        Returns:
            torch.Tensor: Loss of the training step.
        """
        x, y = batch
        y_hat = self(x)

        loss = self.loss(y_hat, y)

        self.log("train_loss", loss)
        self.log("train_positive_mape", self.positive_mape(y_hat, y))
        self.log("train_negative_mape", self.negative_mape(y_hat, y))
        self.log("train_all_mape", self.all_mape(y_hat, y))
        self.log("train_all_mae", self.all_mae(y_hat, y))
        self.log("train_positive_mae", self.positive_mae(y_hat, y))
        self.log("train_negative_mae", self.negative_mae(y_hat, y))

        return loss

    def validation_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """
        Validation step of the forecaster.

        Args:
            batch (Dict[str, torch.Tensor]): Batch of data.
            batch_idx (int): Index of the batch.

        Returns:
            torch.Tensor: Loss of the validation step.
        """
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        self.log("val_loss", loss)
        self.log("val_positive_mape", self.positive_mape(y_hat, y))
        self.log("val_negative_mape", self.negative_mape(y_hat, y))
        self.log("val_all_mape", self.all_mape(y_hat, y))
        self.log("val_all_mae", self.all_mae(y_hat, y))
        self.log("val_positive_mae", self.positive_mae(y_hat, y))
        self.log("val_negative_mae", self.negative_mae(y_hat, y))
        return loss

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """
        Test step of the forecaster.

        Args:
            batch (Dict[str, torch.Tensor]): Batch of data.
            batch_idx (int): Index of the batch.

        Returns:
            torch.Tensor: Loss of the test step.
        """
        x, y = batch

        # increasign weight of the last day
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        self.log("test_loss", loss)
        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """
        Configure the optimizer.

        Returns:
            torch.optim.Optimizer: Optimizer.
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=5, verbose=True, factor=0.5
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler,
            "monitor": "val_loss",
        }


from torch.utils.data import DataLoader
from portfolio_management_rl.dataloaders import MixUpConfig

# import a verbose callback


def main():
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=8),
        ModelCheckpoint(monitor="val_loss"),
        LearningRateMonitor(logging_interval="epoch", log_momentum=True),
    ]

    trainer = Trainer(
        max_epochs=100, callbacks=callbacks, deterministic=True, logger=True
    )

    train_dataset = StocksDataset(
        phase=Phase.TRAIN,
        step_size_days=1,
        y_sequence=True,
        relative=True,
        random_permute=True,
        # mixup_config=MixUpConfig(mixup_alpha=0.2),
    )

    val_dataset = StocksDataset(
        phase=Phase.VAL,
        step_size_days=1,
        y_sequence=True,
        relative=True,
    )

    trainloader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=8,
        prefetch_factor=10,
    )
    valloader = DataLoader(
        val_dataset, batch_size=64, shuffle=False, num_workers=8, prefetch_factor=10
    )

    forecaster = Forecaster(predict_sequence=True, loss="wmae")

    trainer.fit(
        forecaster,
        trainloader,
        valloader,
        # ckpt_path="/home/chris/Documents/temp_until_torch_rl11/lightning_logs/version_0/checkpoints/epoch=12-step=2314.ckpt",
    )


if __name__ == "__main__":
    # disable logging of da
    main()
