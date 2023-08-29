"""
Soft Actor Critic (SAC)  networks for continuous action spaces.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Dirichlet, Beta
from typing import Tuple
from torch import Tensor
from portfolio_management_rl.agents.deep_learning_agents.nn.tcn import (
    TemporalConvNet,
    TemporalAttentionPooling,
)
from portfolio_management_rl.agents.deep_learning_agents.sac.trainers import (
    SACTrainerV1,
)

from portfolio_management_rl.market_environment.buffer import Buffer
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

        self.seq_len = seq_len

        self.encoder = StateEncoder(
            lr=lr, seq_len=seq_len, input_size=input_size, dropout=dropout
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

        # Attention mechanism where q is the current portfolio, k is the current state representation
        # and v is the new portfolio
        self.bilinear = nn.Bilinear(
            in1_features=N_STOCKS + 1,
            in2_features=N_STOCKS + 1,
            out_features=1,
            bias=False,
        )

        # soft gate for the two Q values(keep and rebalance)

        self.swish = nn.SiLU()

        self.batch_norm = nn.BatchNorm1d(N_STOCKS + 1)
        self.dropout = nn.Dropout(dropout)

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

        # Q(s, a_rebalance)
        q_rebalance = self.bilinear(state_, new_portfolio)

        q_val = rebalance_prob * q_rebalance + (1 - rebalance_prob) * q_keep

        return q_val


class ValueNetwork(nn.Module):
    """
    Value network for SAC: V(s). Usually there will be two of these networks
    for stability reasons.
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

        self.value = nn.Linear(N_STOCKS + 1, 1, bias=False)

        self.swish = nn.SiLU()

    def forward(self, state: Tuple[Tensor, Tensor]) -> Tensor:
        """
        Forward pass of the network given the state.

        Args:
            state (tuple): Tuple containing the historical prices and the current portfolio. (batch_size, seq_len, input_size), (batch_size, input_size)

        Returns:
            torch.Tensor: Value.
        """

        state_ = self.encoder(state)  # (tensor, tensor) -> tensor

        value = self.value(state_)

        value = self.swish(value)

        return value


class ActorNetwork(nn.Module):
    """
    Actor network for SAC: pi(s).
    """

    def __init__(
        self,
        lr: float = 1e-3,
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
        """
        super().__init__()

        self.noise = noise
        self.dropout = dropout
        self.seq_len = seq_len

        self.encoder = StateEncoder(
            lr=lr, seq_len=seq_len, input_size=input_size, dropout=dropout
        )

        self.prob = nn.Linear(N_STOCKS + 1, N_STOCKS + 1, bias=False)

        self.mu = nn.Linear(N_STOCKS + 1, N_STOCKS + 1, bias=False)
        self.sigma = nn.Linear(N_STOCKS + 1, N_STOCKS + 1, bias=False)

        self.logsoftmax = nn.LogSoftmax(dim=-1)

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
        mu = self.mu(hidden)
        sigma = self.sigma(hidden)

        # constrain the sigma to be between noise and 1
        sigma = torch.clamp(sigma, min=self.noise, max=1)

        return mu, sigma

    def sample(
        self, state: Tuple[Tensor, Tensor], reparametrize: bool = True
    ) -> Tensor:
        """
        Sample an action from the policy given the state. If reparametrize is True, then
        the action is sampled using the reparametrization trick which allows for backpropagation.
        Otherwise, the action is sampled using the policy distribution.

        """

        mu, std = self.forward(state)

        normal = Normal(mu, std)

        if reparametrize:
            action = normal.rsample()

        else:
            action = normal.sample()

        # originally the action is passed through a tanh activation function
        # but we need the softmax for the portfolio weights, so we are going to use
        # the softmax. in order to do that, we first compte the log softmax which will
        # give us the log probabilities of the portfolio weights (needed for the entropy regularization loss)
        # and then we exponentiate the log softmax to get the probabilities of the portfolio weights
        logprobs = self.logsoftmax(action)
        action = torch.exp(logprobs)  # probs

        return action, logprobs


class DirichletActorNetwork(nn.Module):
    """
    The policy network as a dirichlet distribution for probabilistic action selection.
    """

    def __init__(
        self,
        lr: float = 1e-3,
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
            lr=lr, seq_len=seq_len, input_size=input_size, dropout=dropout
        )

        # dense  hidden layer
        self.dense = nn.Sequential(
            nn.Linear(N_STOCKS + 1, N_STOCKS + 1),
            nn.ReLU(),
            nn.BatchNorm1d(N_STOCKS + 1),
        )

        # alpha vetor for the dirichlet distribution
        self.dirichlet_head = nn.Sequential(
            nn.Linear(N_STOCKS + 1, N_STOCKS + 1), nn.ReLU()
        )

        # alpha and beta parameters for the beta distribution
        self.beta_head = nn.Sequential(nn.Linear(N_STOCKS + 1, 2), nn.ReLU())

        # small shift to avoid zero values
        self.epsilon = torch.finfo(torch.float32).eps

    def forward(self, state: Tuple[Tensor, Tensor]) -> Tensor:
        """
        Forward pass of the network given the state.

        Args:
            state (tuple): Tuple containing the historical prices and the current portfolio. (batch_size, seq_len, input_size), (batch_size, input_size)

        Returns:
            torch.Tensor: Value.
        """

        # encode the state
        state_ = self.encoder(state)  # (tensor, tensor) -> tensor

        # pass the state through the dense hidden layer
        hidden = self.dense(state_)

        # pass the hidden layer through the alpha layer with relu activation and not normalization
        alpha_dirichlet = self.dirichlet_head(hidden) + self.epsilon

        beta_params = self.beta_head(hidden) + self.epsilon

        # split the beta parameters into alpha and beta
        alpha_beta = beta_params[:, 0].unsqueeze(-1)
        beta = beta_params[:, 1].unsqueeze(-1)

        return alpha_dirichlet, (alpha_beta, beta)

    def sample(
        self, state: Tuple[Tensor, Tensor], reparametrize: bool = True
    ) -> Tensor:
        """
        Sample an action from the policy given the state. If reparametrize is True, then
        the action is sampled using the reparametrization trick which allows for backpropagation.
        Otherwise, the action is sampled using the policy distribution.

        """

        alpha_dirichlet, (alpha_beta, beta) = self.forward(state)

        dirichlet = Dirichlet(alpha_dirichlet)

        beta = Beta(alpha_beta, beta)

        if reparametrize:
            portfolio = dirichlet.rsample()
            rebalance = beta.rsample()

        else:
            portfolio = dirichlet.sample()
            rebalance = beta.sample()

        # originally the action is distributed as normal distribution shrunk by a tanh activation function which makes the space a box
        # but our actions are actually multinomial distributions in a simplex, so we are going to use a dirichlet distribution.
        # also we include the rebalance flag as a beta distribution. This gives us a joint distribution of the portfolio weights and the rebalance flag
        # which is a dirichlet-beta distribution. Assuming they are not correlated, we can compute the log probability of the joint distribution
        # as the sum of the log probabilities of the marginal distributions.

        logprobs = dirichlet.log_prob(portfolio) + beta.log_prob(rebalance)  # indepen
        logprobs = torch.sum(logprobs, dim=-1, keepdim=True)

        return (portfolio, rebalance), logprobs


if __name__ == "__main__":
    critic1 = BilinearCriticNetwork().to("cuda")
    critic2 = BilinearCriticNetwork().to("cuda")
    value = ValueNetwork().to("cuda")
    dirichlet_actor = DirichletActorNetwork().to("cuda")

    trainer = SACTrainerV1(dirichlet_actor, critic1, critic2, value)
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
