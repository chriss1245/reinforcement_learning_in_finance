from tqdm import tqdm
import copy

import torch.nn as nn
import torch.nn.functional as F
import torch
from torch import Tensor
from typing import Tuple, Optional, List

from portfolio_management_rl.market_environment.buffer import Buffer
from portfolio_management_rl.utils.dtypes import Device, Phase
from portfolio_management_rl.agents.deep_learning_agents.nn.callbacks import (
    BaseCallback,
)

from portfolio_management_rl.agents.deep_learning_agents.nn.dtypes import TrainerState


class SACTrainerV1:
    """
    The trainer for the SAC agent V1

    Attributes:
        gamma (float): Discount factor.
        tau (float): Target network update factor.
        alpha (float): Entropy regularization factor.
        update_freq (int): Target network update frequency.
        device (str): Device to use for training.
        callbacks (list): List of callbacks to use.
        state (TrainerState): Trainer state

    """

    def __init__(
        self,
        actor: nn.Module,
        critic_1: nn.Module,
        critic_2: nn.Module,
        value: nn.Module,
        optimizer: str = "Adam",
        lr: float = 1e-4,
        callbacks: Optional[List[BaseCallback]] = None,
        gamma: float = 0.99,
        tau: float = 0.005,
        alpha: float = 0.2,
        target_network_update_freq: int = 1,
        verbose: bool = True,
        device: Device = Device.GPU,
    ):
        """
        The trainer for the SAC agent V1

        """

        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.update_freq = target_network_update_freq
        self.device = device.value
        self.callbacks = callbacks or []
        self.verbose = verbose

        target_value = copy.deepcopy(value).eval()
        for param in target_value.parameters():
            param.requires_grad = False

        networks = {
            "actor": actor.to(self.device),
            "critic_1": critic_1.to(self.device),
            "critic_2": critic_2.to(self.device),
            "value": value.to(self.device),
            "target_value": target_value.to(self.device),
        }

        optimizers = {
            key: getattr(torch.optim, optimizer)(value.parameters(), lr=lr)
            for key, value in networks.items()
            if key != "target_value"
        }

        self.state = TrainerState(
            epoch=0,
            metrics={},
            networks=networks,
            optimizers=optimizers,
        )

    @property
    def device(self) -> Device:
        """
        Returns the device used for training.
        """
        return Device(self._device)

    @device.setter
    def device(self, device: Device):
        self._device = device.value

        for key, value in self.state.networks.items():
            self.state.networks[key] = value.to(self.device)

    def update_target_network(self):
        """
        Update the different networks
        """

        # update the target value network softly with the value network
        for target_param, param in zip(
            self.target_value.parameters(), self.value.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )
            target_param.requires_grad = False

    def step(
        self,
        state: Tuple[Tensor, Tensor],
        action: Tuple[Tensor, Tensor],
        next_state: Tuple[Tensor, Tensor],
        reward: Tensor,
        done: Tensor,
        phase: Phase = Phase.TRAIN,
    ) -> dict:
        """
        A training or validation step.
        """

        metrics = {}

        # Value loss
        value = self.state.networks["value"](state).view(-1)
        next_value = self.state.networks["target_value"](next_state).view(-1)
        next_value[done.view(-1) == 1] = 0.0

        actions, logprobs = self.state.networks["actor"].sample(
            state, reparametrize=False
        )
        q1_new_policy = self.state.networks["critic_1"](state, actions)
        q2_new_policy = self.state.networks["critic_2"](state, actions)
        q_new_policy = torch.min(q1_new_policy, q2_new_policy)

        value_target = q_new_policy.view(-1) - self.alpha * logprobs.view(-1)
        value_loss = F.mse_loss(value, value_target.detach())  # * 0.5

        metrics[f"{phase.value}_value_loss"] = value_loss.item()

        if phase == Phase.TRAIN:
            self.state.optimizers["value"].zero_grad()
            value_loss.backward(retain_graph=True)
            self.state.optimizers["value"].step()

        # Actor loss
        actions, logprobs = self.state.networks["actor"].sample(
            state, reparametrize=True
        )
        q1_new_policy = self.state.networks["critic_1"](state, actions).view(-1)
        q2_new_policy = self.state.networks["critic_2"](state, actions).view(-1)
        q_new_policy = torch.min(q1_new_policy, q2_new_policy)

        actor_loss = torch.mean(self.alpha * logprobs - q_new_policy)

        metrics[f"{phase.value}_actor_loss"] = actor_loss.item()

        if phase == Phase.TRAIN:
            self.state.optimizers["actor"].zero_grad()
            actor_loss.backward(retain_graph=True)
            self.state.optimizers["actor"].step()

        # Critic loss
        q_hat = reward.view(-1) + self.gamma * next_value  # reward + gamma * next_value
        q1_old_policy = self.state.networks["critic_1"](state, action).view(-1)
        q2_old_policy = self.state.networks["critic_2"](state, action).view(-1)
        critic_1_loss = F.mse_loss(q1_old_policy, q_hat.detach())  # * 0.5
        critic_2_loss = F.mse_loss(q2_old_policy, q_hat.detach())  # * 0.5

        critic_loss = critic_1_loss + critic_2_loss

        metrics[f"{phase.value}_critic_1_loss"] = critic_1_loss.item()
        metrics[f"{phase.value}_critic_2_loss"] = critic_2_loss.item()
        metrics[f"{phase.value}_critic_loss"] = critic_loss.item()

        # total loss
        metrics[f"{phase.value}_loss"] = (
            value_loss.item() + actor_loss.item() + critic_loss.item()
        )

        if phase == Phase.TRAIN:
            self.state.optimizers["critic_1"].zero_grad()
            self.state.optimizers["critic_2"].zero_grad()
            critic_loss.backward()
            self.state.optimizers["critic_1"].step()
            self.state.optimizers["critic_2"].step()

        return metrics

    def run_epoch(
        self,
        buffer: Buffer,
        batch_size: int = 32,
        phase: Phase = Phase.TRAIN,
        steps: int = 100,
    ) -> dict:
        """
        Runs an epoch of training or validation.

        Args:
            buffer (Buffer): Buffer to sample the data from.
            batch_size (int): Batch size to use.
            phase (Phase): Phase of the epoch.
        """

        for callback in self.callbacks:
            callback.on_epoch_start(self.state)

        metrics = {}

        for i in tqdm(range(steps), desc=f"Epoch {self.state.epoch}: {phase.value}"):
            # sample a batch of data from the buffer
            state, action, next_state, reward, done = buffer.sample(
                batch_size, tensor=True
            )

            state[0].to(self.device)
            state[1].to(self.device)
            action[0].to(self.device)
            action[1].to(self.device)
            next_state[0].to(self.device)
            next_state[1].to(self.device)
            reward.to(self.device)
            done.to(self.device)

            # Run a step
            if phase == Phase.TRAIN:
                metrics_ = self.step(
                    state, action, next_state, reward, done, phase=phase
                )
                if i % self.update_freq == 0:
                    self.update_target_network()
            else:
                with torch.no_grad():
                    metrics_ = self.step(
                        state, action, next_state, reward, done, phase=phase
                    )

            for key, value in metrics_.items():
                if key not in metrics:
                    metrics[key] = 0
                metrics[key] += value / steps

        self.state.epoch += 1
        self.state.log_metrics(metrics)

        for callback in self.callbacks:
            callback.on_epoch_end(self.state)

        return metrics

    def print_metrics(self):
        """
        Print the metrics of the last epoch.
        """
        print(f"Epoch {self.state.epoch}:")
        for key, value in self.state.metrics.items():
            print(f"{key}: {value[-1]}")

    def fit(
        self,
        train_buffer: Buffer,
        val_buffer: Optional[Buffer],
        epochs: int = 10,
        steps: int = 100,
        val_steps: int = 100,
    ):
        """
        Learn the policy, the value function and the Q function.

        Args:
            train_buffer (Buffer): Buffer to sample the data from.
            val_buffer (Buffer): Buffer to sample the data from.
            epochs (int): Number of epochs to train for.
            steps (int): Number of steps per epoch.
            val_steps (int): Number of steps per validation epoch.
        """

        for _ in range(epochs):
            self.run_epoch(train_buffer, steps=steps)
            if val_buffer is not None:
                self.run_epoch(val_buffer, steps=val_steps, phase=Phase.VAL)

            if self.verbose:
                self.print_metrics()
