"""
Soft Actor Critic (SAC)  networks for continuous action spaces.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from typing import Tuple
from torch import Tensor
from portfolio_management_rl.agents.deep_learning_agents.nn.tcn import (
    TemporalConvNet,
    TemporalAttentionPooling,
)
from portfolio_management_rl.utils.contstants import (
    WINDOW_SIZE,
    FORECAST_HORIZON,
    N_STOCKS,
)


class StateEncoder(nn.Module):
    """
    Creates a state representation from the historical prices and the current portfolio.
    """

    def __init__(
        self,
        lr: float = 1e-3,
        seq_len: int = WINDOW_SIZE,
        input_size: int = N_STOCKS,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.encoder = TemporalConvNet(
            input_size=input_size,
            kernel_size=3,
            num_filters=input_size,
            dilation_base=2,
            weight_norm=True,
            target_size=input_size + 1,
            dropout=0.2,
            window_size=seq_len,
        )

        # pools the features of the sequence
        self.att1 = TemporalAttentionPooling(
            num_features=N_STOCKS + 1,
        )

        # combine the pooled features with the current portfolio
        # This is the state representation
        self.bilinear_state = nn.Bilinear(
            in1_features=N_STOCKS + 1,
            in2_features=N_STOCKS + 1,
            out_features=N_STOCKS + 1,
            bias=False,
        )

    def forward(self, state: Tuple[Tensor, Tensor]) -> Tensor:
        """
        Forward pass of the network given the state and action.

        Args:
            state (tuple): Tuple containing the historical prices and the current portfolio. (batch_size, seq_len, input_size), (batch_size, input_size), (batch_size, input_size)

        Returns:
            torch.Tensor: Q value.
        """

        history, portfolio = state

        # encode the historical prices
        history = self.encoder(history)

        # pool the features
        history = self.att1(history)

        # combine the pooled features with the current portfolio
        state = self.bilinear_state(history, portfolio)

        return state


class CriticNetwork(nn.Module):
    """
    Critic network for SAC: Q(s,a). Usually there will be two of these networks
    for stability reasons.

    The states contain two parts: the historical prices and the current portfolio
    The actions are the portfolio weights and a flag indicating whether rebalance or keep the
    current portfolio
    """

    def __init__(
        self,
        lr: float = 1e-3,
        seq_len: int = WINDOW_SIZE,
        input_size: int = N_STOCKS,
        dropout: float = 0.2,
    ):
        """
        Initialize the network.

        Args:
            lr (float): Learning rate.
            seq_len (int): Sequence length.
            input_size (int): Input size.
            output_size (int): Output size.
        """
        super().__init__()

        self.encoder = StateEncoder(
            lr=lr, seq_len=seq_len, input_size=input_size, dropout=dropout
        )

        # combine the state representation with the action if rebalance
        # This is the Q value of rebalance
        self.q_rebalance = nn.Bilinear(
            in1_features=N_STOCKS + 1,
            in2_features=N_STOCKS + 1,
            out_features=1,
            bias=False,
        )

        self.q_keep = nn.Bilinear(
            in1_features=N_STOCKS + 1,
            in2_features=N_STOCKS + 1,
            out_features=1,
            bias=False,
        )

        # soft gate for the two Q values
        self.rebalance = nn.Linear(1, 1, bias=False)

        self.sigmoid = nn.Sigmoid()
        self.swish = nn.SiLU()
        self.batch_norm = nn.BatchNorm1d(N_STOCKS + 1)
        self.dropout = nn.Dropout(dropout)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(
        self, state: Tuple[Tensor, Tensor], action: Tuple[Tensor, Tensor]
    ) -> Tensor:
        """
        Forward pass of the network given the state and action.

        Args:
            state (tuple): Tuple containing the historical prices and the current portfolio. (batch_size, seq_len, input_size), (batch_size, input_size)
            action (tuple): Tuple containing the portfolio weights and the rebalance flag. (batch_size, input_size), (batch_size, 1)

        Returns:
            torch.Tensor: Q value.
        """

        new_portfolio, rebalance_flag = action

        state_ = self.encoder(state)  # (tensor, tensor) -> tensor

        # combine the state representation with the action if rebalance
        q_rebalance = self.q_rebalance(state_, new_portfolio)
        q_keep = self.q_keep(state_, portfolio)

        # soft gate for the two Q values
        w = self.sigmoid(self.rebalance(rebalance_flag))
        # combine the two Q values
        q_val = w * q_rebalance + (1 - w) * q_keep

        return q_val


if __name__ == "__main__":
    critic1 = CriticNetwork().to("cuda")

    input("Press Enter to continue...")

    history = torch.rand(32, WINDOW_SIZE, N_STOCKS).to("cuda")
    portfolio = torch.rand(32, N_STOCKS + 1).to("cuda")
    new_portfolio = torch.rand(32, N_STOCKS + 1).to("cuda")
    rebalance_flag = torch.rand(32, 1).to("cuda")

    q_val = critic1((history, portfolio), (new_portfolio, rebalance_flag))
    print(q_val.shape)

    input("Press Enter to continue...")

    critic2 = CriticNetwork().to("cuda")

    q_val2 = critic2((history, portfolio), (new_portfolio, rebalance_flag))

    input("Press Enter to continue...")

    loss = F.mse_loss(q_val, q_val2)

    input("Press Enter to continue...")

    loss.backward()

    input("Press Enter to continue...")
