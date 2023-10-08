"""
This module contains the SAC agent class
"""

from typing import Dict, List, Optional
from collections import deque
from portfolio_management_rl.dataloaders.buffer import Buffer
from portfolio_management_rl.market_environment.market_env import (
    MarketEnv,
    MarketEnvState,
)
from portfolio_management_rl.nn.tcn import TemporalConvNet

import torch
from torch import Tensor
from typing import Tuple, Any

from portfolio_management_rl.utils.contstants import (
    N_STOCKS,
    WINDOW_SIZE,
    FORECAST_HORIZON,
)
from portfolio_management_rl.utils.dtypes import Phase, Device

from portfolio_management_rl.agents.sac.sac import (
    BilinearCriticNetwork,
    NormalActorNetwork,
    DirichletActorNetwork,
    AttentionCriticNetwork,
)
from portfolio_management_rl.agents.sac.trainers import SACTrainerV2
from portfolio_management_rl.agents.base import BaseAgent
from portfolio_management_rl.nn.callbacks import (
    Tensorboard,
    Checkpoint,
    EarlyStoping,
    ReduceLROnPlateauCallback,
)
from portfolio_management_rl.nn.utils import load_checkpoint
from pathlib import Path
from portfolio_management_rl.utils.contstants import PROJECT_DIR

CALLBACKS = [
    Tensorboard(),
    # ReduceLROnPlateauCallback("val_loss", patience=2,  is_loss=True),
]

# copy the networks deep copy
from copy import deepcopy

OPTIMIZER = "Adam"


class SACAgent(BaseAgent):
    """
    This module implements the SAC agent.
    """

    _name = "SACAgent"

    def __init__(
        self,
        n_stocks: int = N_STOCKS,
        window_size: int = WINDOW_SIZE,
        forecast_horizon: int = FORECAST_HORIZON,
        gamma: float = 0.99,
        tau: float = 0.05,
        alpha: float = 0.0004,
        lr: float = 0.001,
        batch_size: int = 256,
        actor: str = "normal",
        critic: str = "bilinear",
        training_device: Device = Device.GPU,
        infer_device: Device = Device.CPU,
        actor_noise: float = 0.2,
        dropout: float = 0.2,
        checkpoint_path: Optional[Path] = None,
        freeze_encoder: bool = False,
        encoder_path: Optional[Path] = PROJECT_DIR / "models/encoder.pt",
        name: str = "SACAgent",
    ):
        if actor == "dirichlet":
            self.actor = DirichletActorNetwork(
                seq_len=window_size,
                input_size=n_stocks,
                noise=actor_noise,
                dropout=dropout,
            )
        else:
            self.actor = NormalActorNetwork(
                seq_len=window_size,
                input_size=n_stocks,
                noise=actor_noise,
                dropout=dropout,
            )

        if critic == "bilinear":
            self.critic_1 = BilinearCriticNetwork(
                seq_len=window_size,
                input_size=n_stocks,
                dropout=dropout,
            )

            self.critic_2 = BilinearCriticNetwork(
                seq_len=window_size,
                input_size=n_stocks,
                dropout=dropout,
            )
        else:
            self.critic_1 = AttentionCriticNetwork(
                seq_len=window_size,
                input_size=n_stocks,
                dropout=dropout,
            )

            self.critic_2 = AttentionCriticNetwork(
                seq_len=window_size,
                input_size=n_stocks,
                dropout=dropout,
            )

        self._name = name

        self.actor_name = actor
        self.critic_name = critic

        if checkpoint_path is not None:
            load_checkpoint(checkpoint_path / "actor.pth", self.actor)
            load_checkpoint(checkpoint_path / "critic_1.pth", self.critic_1)
            load_checkpoint(checkpoint_path / "critic_2.pth", self.critic_2)

        if encoder_path is not None:
            load_checkpoint(encoder_path, self.actor.encoder.encoder)
            load_checkpoint(encoder_path, self.critic_1.encoder.encoder)
            load_checkpoint(encoder_path, self.critic_2.encoder.encoder)
            # freeze the encoder

        if freeze_encoder:
            for param in self.actor.encoder.encoder.parameters():
                param.requires_grad = False
            for param in self.critic_1.encoder.encoder.parameters():
                param.requires_grad = False
            for param in self.critic_2.encoder.encoder.parameters():
                param.requires_grad = False

        self.trainer = SACTrainerV2(
            actor=self.actor,
            critic_1=self.critic_1,
            critic_2=self.critic_2,
            optimizer=OPTIMIZER,
            callbacks=CALLBACKS,
            gamma=gamma,
            tau=tau,
            alpha=alpha,
            device=training_device,
            lr=lr,
        )

        self.dropout = dropout
        self.actor_noise = actor_noise
        self._training_device = training_device
        self._infer_device = infer_device

    def preprocess(self, state: MarketEnvState) -> Tuple[Tensor, Tensor]:
        """
        Preprocess the state of the environment to be used by the agent.

        Args:

            state (MarketEnvState): State of the environment.

        Returns:
            Returns a tuple of tensors containing the input and the target.
        """

        portfolio = (
            torch.from_numpy(state.net_distribution)
            .float()
            .to(self._infer_device.value)
        ).unsqueeze(0)
        relative_returns = state.history.copy()
        initial_prices = relative_returns[0, :]

        relative_returns = (relative_returns / initial_prices) - 1
        relative_returns = (
            torch.from_numpy(relative_returns).float().to(self._infer_device.value).T
        )
        # add the batch dimension
        relative_returns = relative_returns.unsqueeze(0)
        return relative_returns, portfolio

    @property
    def parameters(self) -> dict:
        """
        Agent tunable parameters.
        """

        parameters = {
            "n_stocks": self.actor.n_stocks,
            "window_size": self.actor.window_size,
            "forecast_horizon": self.actor.forecast_horizon,
            "gamma": self.actor.gamma,
            "tau": self.actor.tau,
            "alpha": self.actor.alpha,
            "lr": self.actor.lr,
            "batch_size": self.actor.batch_size,
            "actor_type": self.actor_name,
            "critic_type": self.critic_name,
            "dropout": self.dropout,
            "actor_noise": self.actor_noise,
        }

        return parameters

    @property
    def name(self) -> str:
        """
        Agent name.
        """
        return self._name

    def reset(self) -> None:
        pass

    def update(self) -> None:
        pass

    def act(self, state: MarketEnvState) -> Dict[str, Any]:
        """
        Act based on the state of the environment.

        Args:

            state (MarketEnvState): State of the environment.

        Returns:
            Returns a tuple of tensors containing the input and the target.
        """
        inputs = self.preprocess(state)
        with torch.no_grad():
            self.actor.eval()
            self.actor.to(self._infer_device.value)
            (action, rebalance), _ = self.actor.sample(inputs, reparametrize=False)
            self.actor.to(self._training_device.value)

        action = action.cpu().numpy().squeeze()
        rebalance = rebalance.cpu().squeeze().item()

        return {"distribution": action, "rebalance": rebalance}

    def learn(self, buffer: Buffer, validation_buffer: Optional[Buffer] = None) -> None:
        """
        Calls the trainer to learn from the buffer.
        """

        self.trainer.device = self._training_device
        self.trainer.fit(buffer, val_buffer=validation_buffer)
        self.trainer.device = self._infer_device


if __name__ == "__main__":
    from portfolio_management_rl.dataloaders import StocksDataset
    from portfolio_management_rl.backtester.backtester import MATRIX

    dataset = StocksDataset()

    agent = SACAgent()
    env = MarketEnv(dataset=dataset)

    matrix = MATRIX(market_env=env, verbose=True)
    state = env.reset()
    action = agent.act(state)

    import numpy as np

    print(np.sum(action["action"]))
    print(action["rebalance"])
