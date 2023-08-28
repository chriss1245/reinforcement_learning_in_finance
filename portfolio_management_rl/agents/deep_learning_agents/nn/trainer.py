from tqdm import tqdm
import copy
import numpy as np

import torch.nn as nn
import torch.nn.functional as F
import torch
from torch import Tensor
from typing import Tuple, Optional

from portfolio_management_rl.agents.deep_learning_agents.nn.utils import (
    save_checkpoints,
    load_checkpoint,
)

from portfolio_management_rl.market_environment.buffer import Buffer
from portfolio_management_rl.utils.dtypes import Device, Phase


class SACTrainerV1:
    """
    The trainer for the SAC agent V1 without the value network
    """

    def __init__(
        self,
        actor: nn.Module,
        critic_1: nn.Module,
        critic_2: nn.Module,
        value: nn.Module,
        optimizer: str = "Adam",
        lr: float = 1e-4,
        discount: float = 0.99,
        tau: float = 0.005,
        alpha: float = 0.2,
        entropy_tuning: bool = True,
        device: Device = Device.GPU,
        steps_per_experience_replay: int = 100,
        val_steps_per_experience_replay: int = 100,
        target_network_update_freq: int = 1,
    ):
        self.actor = actor
        self.critic_1 = critic_1
        self.critic_2 = critic_2
        self.value = value
        self.target_value = copy.deepcopy(value)
        self.target_value.eval()
        for param in self.target_value.parameters():
            param.requires_grad = False

        self.lr = lr
        self.discount = discount
        self.tau = tau
        self.alpha = alpha
        self.entropy_tuning = entropy_tuning
        self.steps = steps_per_experience_replay
        self.val_steps = val_steps_per_experience_replay
        self.update_freq = target_network_update_freq
        self.device = device.value

        self.actor_optimizer = getattr(torch.optim, optimizer)(
            self.actor.parameters(), lr=self.lr
        )
        self.critic_1_optimizer = getattr(torch.optim, optimizer)(
            self.critic_1.parameters(), lr=self.lr
        )
        self.critic_2_optimizer = getattr(torch.optim, optimizer)(
            self.critic_2.parameters(), lr=self.lr
        )
        self.value_optimizer = getattr(torch.optim, optimizer)(
            self.value.parameters(), lr=self.lr
        )

    def step_legacy(
        self,
        state: Tuple[Tensor, Tensor],
        action: Tuple[Tensor, Tensor],
        next_state: Tuple[Tensor, Tensor],
        reward: Tensor,
        done: Tensor,
    ):
        """
        A training step
        """

        # value loss
        value = self.value(state).view(-1)
        next_value = self.target_value(next_state).view(-1)
        next_value[done.view(-1) == 1] = 0.0

        # new policy
        actions, logprobs = self.actor.sample(state, reparametrize=False)
        q1_new_policy = self.critic_1(state, actions)
        q2_new_policy = self.critic_2(state, actions)
        q_new_policy = torch.min(q1_new_policy, q2_new_policy)

        self.value_optimizer.zero_grad()
        value_target = q_new_policy.view(-1) - logprobs.view(-1)
        value_loss = F.mse_loss(value, value_target.detach())  # * 0.5

        # do not lose the gradients of the value network
        value_loss.backward(retain_graph=True)
        self.value_optimizer.step()

        # actor loss
        actions, logprobs = self.actor.sample(state, reparametrize=True)
        q1_new_policy = self.critic_1(state, actions).view(-1)
        q2_new_policy = self.critic_2(state, actions).view(-1)
        q_new_policy = torch.min(q1_new_policy, q2_new_policy)

        actor_loss = torch.mean(logprobs - q_new_policy)

        self.actor_optimizer.zero_grad()
        # do not lose the gradients of the value network
        actor_loss.backward(retain_graph=True)

        self.actor_optimizer.step()

        # critic loss
        self.critic_1_optimizer.zero_grad()
        self.critic_2_optimizer.zero_grad()

        q_hat = self.alpha * reward.view(-1) + self.discount * next_value
        q1_old_policy = self.critic_1(state, action).view(-1)
        q2_old_policy = self.critic_2(state, action).view(-1)
        critic_1_loss = F.mse_loss(q1_old_policy, q_hat.detach())  # * 0.5
        critic_2_loss = F.mse_loss(q2_old_policy, q_hat.detach())  # * 0.5

        critic_loss = critic_1_loss + critic_2_loss

        critic_loss.backward()

        self.critic_1_optimizer.step()
        self.critic_2_optimizer.step()

    def step(
        self,
        state: Tuple[Tensor, Tensor],
        action: Tuple[Tensor, Tensor],
        next_state: Tuple[Tensor, Tensor],
        reward: Tensor,
        done: Tensor,
        phase: Phase = Phase.TRAIN,
    ):
        """
        A training or validation step.
        """
        metrics = {}

        # Value loss
        value = self.value(state).view(-1)
        next_value = self.target_value(next_state).view(-1)
        next_value[done.view(-1) == 1] = 0.0

        actions, logprobs = self.actor.sample(state, reparametrize=False)
        q1_new_policy = self.critic_1(state, actions)
        q2_new_policy = self.critic_2(state, actions)
        q_new_policy = torch.min(q1_new_policy, q2_new_policy)

        value_target = q_new_policy.view(-1) - logprobs.view(-1)
        value_loss = F.mse_loss(value, value_target.detach())

        metrics[f"{phase.value}_value_loss"] = value_loss.item()

        if phase == Phase.TRAIN:
            self.value_optimizer.zero_grad()
            value_loss.backward(retain_graph=True)
            self.value_optimizer.step()

        # Actor loss
        actions, logprobs = self.actor.sample(state, reparametrize=True)
        q1_new_policy = self.critic_1(state, actions).view(-1)
        q2_new_policy = self.critic_2(state, actions).view(-1)
        q_new_policy = torch.min(q1_new_policy, q2_new_policy)

        actor_loss = torch.mean(logprobs - q_new_policy)

        metrics[f"{phase.value}_actor_loss"] = actor_loss.item()

        if phase == Phase.TRAIN:
            self.actor_optimizer.zero_grad()
            actor_loss.backward(retain_graph=True)
            self.actor_optimizer.step()

        # Critic loss
        q_hat = self.alpha * reward.view(-1) + self.discount * next_value
        q1_old_policy = self.critic_1(state, action).view(-1)
        q2_old_policy = self.critic_2(state, action).view(-1)
        critic_1_loss = F.mse_loss(q1_old_policy, q_hat.detach())
        critic_2_loss = F.mse_loss(q2_old_policy, q_hat.detach())

        critic_loss = critic_1_loss + critic_2_loss

        metrics[f"{phase.value}_critic_1_loss"] = critic_1_loss.item()
        metrics[f"{phase.value}_critic_2_loss"] = critic_2_loss.item()
        metrics[f"{phase.value}_critic_loss"] = critic_loss.item()

        # total loss
        metrics[f"{phase.value}_loss"] = (
            value_loss.item() + actor_loss.item() + critic_loss.item()
        )

        if phase == Phase.TRAIN:
            self.critic_1_optimizer.zero_grad()
            self.critic_2_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_1_optimizer.step()
            self.critic_2_optimizer.step()

        return metrics

    def update_target_network(self):
        """
        Update the different networks
        """

        # update the target value network
        for target_param, param in zip(
            self.target_value.parameters(), self.value.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )
            target_param.requires_grad = False

    def run_epoch(
        self,
        buffer: Buffer,
        batch_size: int = 32,
        phase: Phase = Phase.TRAIN,
    ):
        """
        Run an epoch
        """

        metrics = {}

        for i in range(self.steps if phase == Phase.TRAIN else self.val_steps):
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
                metrics[key] += value / self.steps

        return metrics

    def train(
        self,
        train_buffer: Buffer,
        val_buffer: Optional[Buffer],
        episodes: int = 1,
        batch_size: int = 32,
        early_stopping: bool = False,
        monitor: str = "val_loss",
        patience: int = 2,
    ):
        """
        Learn the policy
        """

        metrics = {}
        if val_buffer is None:
            monitor = "train_loss"

        max_monitor = -np.inf
        patience_counter = 0

        for episode in range(episodes):
            for i in (pbar := tqdm(range(self.steps), desc="ExperienceReplay")):
                # sample a batch of data from the buffer
                state, action, next_state, reward, done = train_buffer.sample(
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

                # train the network
                metrics_ = self.step(
                    state, action, next_state, reward, done, phase=Phase.TRAIN
                )

                if i % self.update_freq == 0:
                    self.update_target_network()

                for key, value in metrics_.items():
                    if key not in metrics:
                        metrics[key] = []
                    metrics[key].append(value)

            if val_buffer is None:
                continue

            for i in (pbar := tqdm(range(self.steps), desc="ExperienceReplay")):
                # sample a batch of data from the buffer
                state, action, next_state, reward, done = val_buffer.sample(
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

                # train the network
                metrics_ = self.step(
                    state, action, next_state, reward, done, phase=Phase.VAL
                )

                for key, value in metrics_.items():
                    if key not in metrics:
                        metrics[key] = []
                    metrics[key].append(value)

            if early_stopping:
                if max_monitor < metrics[monitor][-1]:
                    max_monitor = metrics[monitor][-1]
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= patience:
                    break

            if save_checkpoints(episode):
                self.save_checkpoint(episode)
