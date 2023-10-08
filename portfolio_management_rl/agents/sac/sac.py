"""
Soft Actor Critic (SAC)  networks for continuous action spaces.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Dirichlet, Beta
from typing import Tuple
from torch import Tensor
from portfolio_management_rl.nn.tcn import (
    TemporalConvNet,
    TemporalAttentionPooling,
)
from portfolio_management_rl.agents.sac.trainers import SACTrainerV2

from portfolio_management_rl.dataloaders.buffer import Buffer
from portfolio_management_rl.utils.contstants import (
    WINDOW_SIZE,
    FORECAST_HORIZON,
    N_STOCKS,
)

from portfolio_management_rl.utils.dtypes import Phase, Device

from typing import Optional


class StateEncoder(nn.Module):
    """
    Creates a state representation from the historical prices and the current portfolio.
    """

    def __init__(
        self,
        seq_len: int = WINDOW_SIZE,
        input_size: int = N_STOCKS,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.encoder = TemporalConvNet(
            input_size=input_size,
            kernel_size=3,
            num_filters=input_size,
            dilation_base=3,
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

        self.batch_norm = nn.BatchNorm1d(N_STOCKS + 1)
        self.swish = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

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

        state = self.swish(state)

        state = self.batch_norm(state)

        state = self.dropout(state)

        return state


class AttentionCriticNetwork(nn.Module):
    """
    Critic network for SAC: Q(s,a). Usually there will be two of these networks
    for stability reasons.

    The states contain two parts: the historical prices and the current portfolio
    The actions are the portfolio weights and a flag indicating whether rebalance or keep the
    current portfolio
    """

    def __init__(
        self,
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

        self.seq_len = seq_len
        self.input_size = input_size

        self.encoder = StateEncoder(
            seq_len=seq_len, input_size=input_size, dropout=dropout
        )

        # Attention mechanism where q is the current portfolio, k is the current state representation
        # and v is the new portfolio
        self.single_head_attention = nn.MultiheadAttention(
            embed_dim=N_STOCKS + 1,
            num_heads=1,
            dropout=dropout,
        )

        self.sigmoid = nn.Sigmoid()

        self.swish = nn.SiLU()
        self.batch_norm = nn.BatchNorm1d(N_STOCKS + 1)
        self.dropout = nn.Dropout(dropout)

    def get_input(
        self, device: Device = Device.GPU
    ) -> Tuple[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor]]:
        """
        Returns the input that the network expects.
        """

        history = torch.randn(1, self.seq_len, N_STOCKS)
        portfolio = torch.randn(1, N_STOCKS + 1)
        new_portfolio = torch.randn(1, N_STOCKS + 1)
        rebalance_prob = torch.randn(1, 1)

        if device == Device.GPU:
            history = history.to("cuda")
            portfolio = portfolio.to("cuda")
            new_portfolio = new_portfolio.to("cuda")
            rebalance_prob = rebalance_prob.to("cuda")

        return (history, portfolio), (new_portfolio, rebalance_prob)

    def forward(
        self, state: Tuple[Tensor, Tensor], action: Tuple[Tensor, Tensor]
    ) -> Tensor:
        """
        Forward pass of the network given the state and action.

        Args:
            state (tuple): Tuple containing the historical prices and the current portfolio. (batch_size, seq_len, input_size), (batch_size, input_size)
            action (tuple): Tuple containing the portfolio weights and the rebalance prob. (batch_size, input_size), (batch_size, 1)

        Returns:
            torch.Tensor: Q value.
        """

        history, portfolio = state
        new_portfolio, rebalance_prob = action

        state_ = self.encoder((history, portfolio))  # (tensor, tensor) -> tensor

        # Q(s,a) = Q(s, a_keep) * (1 - rebalance_flag) + Q(s, a_rebalance) * rebalance_flag

        # Q(s, a_keep)
        q_keep, attention_scores = self.single_head_attention(
            query=portfolio,
            key=state_,
            value=portfolio,
        )

        # Q(s, a_rebalance)
        q_rebalance, attention_scores = self.single_head_attention(
            query=portfolio,
            key=state_,
            value=new_portfolio,
        )

        q_val = rebalance_prob * q_rebalance + (1 - rebalance_prob) * q_keep

        return q_val


class BilinearCriticNetwork(nn.Module):
    """
    Critic network for SAC: Q(s,a). Using a bilinear layer to combine the state and action in order to get the Q value.
    The states contain two parts: the historical prices and the current portfolio
    The actions are the portfolio weights and a flag indicating whether rebalance or keep the
    current portfolio
    """

    def __init__(
        self,
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

        self.input_size = input_size
        self.seq_len = seq_len

        self.encoder = StateEncoder(
            seq_len=seq_len, input_size=input_size, dropout=dropout
        )

        # Attention mechanism where q is the current portfolio, k is the current state representation
        # and v is the new portfolio
        self.bilinear = nn.Bilinear(
            in1_features=N_STOCKS + 1,
            in2_features=N_STOCKS + 1,
            out_features=1,
            bias=True,
        )

        # soft gate for the two Q values(keep and rebalance)

        self.swish = nn.SiLU()

        self.batch_norm = nn.BatchNorm1d(1)
        self.dropout = nn.Dropout(dropout)

    def get_input(
        self, device: Device = Device.GPU
    ) -> Tuple[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor]]:
        """
        Returns the input that the network expects.
        """

        history = torch.randn(1, self.seq_len, N_STOCKS)
        portfolio = torch.randn(1, N_STOCKS + 1)
        new_portfolio = torch.randn(1, N_STOCKS + 1)
        rebalance_prob = torch.randn(1, 1)

        if device == Device.GPU:
            history = history.to("cuda")
            portfolio = portfolio.to("cuda")
            new_portfolio = new_portfolio.to("cuda")
            rebalance_prob = rebalance_prob.to("cuda")

        return (history, portfolio), (new_portfolio, rebalance_prob)

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

        _, portfolio = state
        new_portfolio, rebalance_prob = action

        state_ = self.encoder(state)  # (tensor, tensor) -> tensor

        # Q(s,a) = Q(s, a_keep) * (1 - rebalance_flag) + Q(s, a_rebalance) * rebalance_flag

        # Q(s, a_keep)
        q_keep = self.bilinear(state_, portfolio)
        q_keep = self.swish(q_keep)
        q_keep = self.batch_norm(q_keep)

        # Q(s, a_rebalance)
        q_rebalance = self.bilinear(state_, new_portfolio)
        q_rebalance = self.swish(q_rebalance)
        q_rebalance = self.batch_norm(q_rebalance)

        q_val = rebalance_prob * q_rebalance + (1 - rebalance_prob) * q_keep

        return q_val


class NormalActorNetwork(nn.Module):
    """
    Actor network for SAC: pi(s).
    """

    def __init__(
        self,
        seq_len: int = WINDOW_SIZE,
        input_size: int = N_STOCKS,
        noise: float = 0.001,
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

        self.noise = noise
        self.dropout = dropout
        self.seq_len = seq_len

        self.encoder = StateEncoder(
            seq_len=seq_len, input_size=input_size, dropout=dropout
        )

        self.prob = nn.Linear(N_STOCKS + 1, N_STOCKS + 1, bias=False)

        self.mu_p = nn.Linear(N_STOCKS + 1, N_STOCKS + 1, bias=False)
        self.sigma_p = nn.Linear(N_STOCKS + 1, N_STOCKS + 1, bias=False)

        self.mu_r = nn.Linear(N_STOCKS + 1, 1, bias=False)
        self.sigma_r = nn.Linear(N_STOCKS + 1, 1, bias=False)

        self.softmax = nn.Softmax(dim=-1)
        self.epsilon = torch.finfo(torch.float32).eps

    def get_input(self, device: Device = Device.GPU) -> Tuple[Tuple[Tensor, Tensor]]:
        """
        Returns the input that the network expects.
        """

        history = torch.randn(1, self.seq_len, N_STOCKS)
        portfolio = torch.randn(1, N_STOCKS + 1)

        if device == Device.GPU:
            history = history.to("cuda")
            portfolio = portfolio.to("cuda")

        return ((history, portfolio),)

    def forward(self, state: Tuple[Tensor, Tensor]) -> Tensor:
        """
        Forward pass of the network given the state.

        Args:
            state (tuple): Tuple containing the historical prices and the current portfolio. (batch_size, seq_len, input_size), (batch_size, input_size)

        Returns:
            torch.Tensor: Value.
        """

        state_ = self.encoder(state)  # (tensor, tensor) -> tensor

        hidden = self.prob(state_)
        mu = self.mu_p(hidden)
        sigma = self.sigma_p(hidden)

        # constrain the sigma to be between noise and 1
        sigma = torch.clamp(sigma, min=self.noise, max=1)

        mu_r = self.mu_r(state_)
        sigma_r = self.sigma_r(state_)
        sigma_r = torch.clamp(sigma_r, min=self.noise, max=1)

        return (mu, sigma), (mu_r, sigma_r)

    def sample(
        self, state: Tuple[Tensor, Tensor], reparametrize: bool = True
    ) -> Tensor:
        """
        Sample an action from the policy given the state. If reparametrize is True, then
        the action is sampled using the reparametrization trick which allows for backpropagation.
        Otherwise, the action is sampled using the policy distribution.

        """

        (mu, std), (mu_r, std_r) = self.forward(state)

        normal = Normal(mu, std)
        normal_r = Normal(mu_r, std_r)

        if reparametrize:
            action = normal.rsample()
            rebalance = normal_r.rsample()

        else:
            action = normal.sample()
            rebalance = normal_r.sample()

        action_ = self.softmax(action)
        rebalance = torch.tanh(rebalance)
        rebalance_logprobs = normal_r.log_prob(rebalance) - torch.log(
            1 - rebalance**2 + self.epsilon
        )

        rebalance = 0.5 * rebalance + 0.5  # between 0 and 1

        logprobs = torch.mean(normal.log_prob(action), -1, keepdim=True)  # log probs

        return (action_, rebalance), logprobs + rebalance_logprobs


class DirichletActorNetwork(nn.Module):
    """
    The policy network as a dirichlet distribution for probabilistic action selection.
    """

    def __init__(
        self,
        seq_len: int = WINDOW_SIZE,
        input_size: int = N_STOCKS,
        noise: float = 0.1,
        dropout: float = 0.2,
    ):
        """
        Initialize the network.

        Args:
            lr (float): Learning rate.
            seq_len (int): Sequence length.
            input_size (int): Input size.
            output_size (int): Output size.

            causal_beta (bool): Whether to use a causal beta distribution or not (a beta whose alpha, beta are a function of the dirichlet beliefs)
        """
        super().__init__()

        self.noise = noise
        self.dropout = dropout
        self.seq_len = seq_len

        # state encoder
        self.encoder = StateEncoder(
            seq_len=seq_len, input_size=input_size, dropout=dropout
        )

        # dense  hidden layer
        self.dense = nn.Sequential(
            nn.Linear(N_STOCKS + 1, N_STOCKS + 1),
            nn.SiLU(),
            nn.BatchNorm1d(N_STOCKS + 1),
        )

        # alpha vetor for the dirichlet distribution
        self.dirichlet_head = nn.Sequential(
            nn.Linear(N_STOCKS + 1, N_STOCKS + 1), nn.ReLU()
        )

        # mu and sigma for the shrunk normal distribution for the rebalance flag
        self.rebalance_head = nn.Sequential(nn.Linear(N_STOCKS + 1, 2), nn.ReLU())

        # small shift to avoid zero values
        self.epsilon = torch.finfo(torch.float32).eps

    def get_input(self, device: Device = Device.GPU) -> Tuple[Tuple[Tensor, Tensor]]:
        """
        Returns the input that the network expects.
        """

        history = torch.randn(1, self.seq_len, N_STOCKS)
        portfolio = torch.randn(1, N_STOCKS + 1)

        if device == Device.GPU:
            history = history.to("cuda")
            portfolio = portfolio.to("cuda")

        return ((history, portfolio),)

    def forward(self, state: Tuple[Tensor, Tensor]) -> Tensor:
        """
        Forward pass of the network given the state.

        Args:
            state (tuple): Tuple containing the historical prices and the current portfolio. (batch_size, seq_len, input_size), (batch_size, input_size)

        Returns:
            torch.Tensor: Value.
        """

        prices, portfolio = state

        # encode the state
        state_ = self.encoder(state)  # (tensor, tensor) -> tensor

        # pass the state through the dense hidden layer
        hidden = self.dense(state_)

        # pass the hidden layer through the alpha layer with relu activation and not normalization
        alpha_dirichlet = self.dirichlet_head(hidden) + self.epsilon

        normal_params = self.rebalance_head(hidden) + self.epsilon

        # split the beta parameters into alpha and beta
        mu = normal_params[:, 0].unsqueeze(-1)
        sigma = normal_params[:, 1].unsqueeze(-1)

        return alpha_dirichlet, (mu, sigma)

    def sample(
        self, state: Tuple[Tensor, Tensor], reparametrize: bool = True
    ) -> Tensor:
        """
        Sample an action from the policy given the state. If reparametrize is True, then
        the action is sampled using the reparametrization trick which allows for backpropagation.
        Otherwise, the action is sampled using the policy distribution.

        """

        alpha_dirichlet, (mu, sigma) = self.forward(state)

        dirichlet = Dirichlet(alpha_dirichlet)

        normal = Normal(mu, sigma)

        if reparametrize:
            portfolio = dirichlet.rsample()
            rebalance = normal.rsample()

        else:
            portfolio = dirichlet.sample()
            rebalance = normal.sample()

        rebalance_confidence = 0.5 * torch.tanh(rebalance) + 0.5  # between 0 and 1

        # assuming that the portfolio weights are independent of the rebalance flag
        # we can compute the log probabilities of the portfolio weights and the rebalance flag
        # separately and then sum them up

        # log probabilities of the confidence is given by the log probability of the normal distribution
        # plus the log probability of the 1/2 tanh function since the rebalance confidence is between 0 and 1
        reb_conf_logprobs = normal.log_prob(rebalance) - 0.5 * torch.log(
            1 - rebalance_confidence**2 + self.epsilon
        )  # 1- 0.999 close to 0. -> log(0) = -inf -> log_prob(normal) - log(1-0.999) = inf

        logprobs = dirichlet.log_prob(portfolio).unsqueeze(-1) + reb_conf_logprobs

        return (portfolio, rebalance), logprobs


if __name__ == "__main__":
    critic1 = AttentionCriticNetwork().to("cuda")
    critic2 = BilinearCriticNetwork().to("cuda")
    dirichlet_actor = DirichletActorNetwork().to("cuda")

    trainer = SACTrainerV2(dirichlet_actor, critic1, critic2)
    b_size = 8
    state = (
        torch.randn(b_size, WINDOW_SIZE, N_STOCKS).to("cuda"),
        torch.randn(b_size, N_STOCKS + 1).to("cuda"),
    )

    action = (
        torch.randn(b_size, N_STOCKS + 1).to("cuda"),
        torch.randn(b_size, 1).to("cuda"),
    )

    next_state = (
        torch.randn(b_size, WINDOW_SIZE, N_STOCKS).to("cuda"),
        torch.randn(b_size, N_STOCKS + 1).to("cuda"),
    )

    reward = torch.randn(b_size, 1).to("cuda")
    done = torch.randn(b_size, 1).to("cuda")

    input("Press Enter to continue...")
    trainer.step(state, action, next_state, reward, done)

    input("Done")
