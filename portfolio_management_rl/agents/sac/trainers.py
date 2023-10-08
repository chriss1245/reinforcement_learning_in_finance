from tqdm import tqdm
import copy

import torch.nn as nn
import torch.nn.functional as F
import torch
from torch import Tensor
from typing import Tuple, Optional, List

from portfolio_management_rl.dataloaders.buffer import Buffer
from portfolio_management_rl.utils.dtypes import Device, Phase
from portfolio_management_rl.nn.callbacks import (
    BaseCallback,
    EarlyStopExcpetion,
)

from portfolio_management_rl.nn.dtypes import TrainerState

from portfolio_management_rl.utils.logger import get_logger

logger = get_logger(__file__)
logger.setLevel("DEBUG")


class SACTrainerV2:
    """
    Soft actor critc optimization V2 without a value network and two target networks.
    """

    def __init__(
        self,
        actor: nn.Module,
        critic_1: nn.Module,
        critic_2: nn.Module,
        optimizer: str = "Adam",
        lr: float = 1e-4,
        callbacks: Optional[List[BaseCallback]] = None,
        gamma: float = 0.99,
        tau: float = 0.05,
        alpha: float = 0.000001,
        target_network_update_freq: int = 1,
        verbose: bool = True,
        device: Device = Device.GPU,
    ):
        """
        The trainer for the SAC agent V2

        """

        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.update_freq = target_network_update_freq
        self._device = device.value
        self.callbacks = callbacks or []
        self.verbose = verbose

        target_critc_1 = copy.deepcopy(critic_1).eval()
        for param in target_critc_1.parameters():
            param.requires_grad = False

        target_critc_2 = copy.deepcopy(critic_2).eval()
        for param in target_critc_2.parameters():
            param.requires_grad = False

        networks = {
            "actor": actor.to(self._device),
            "critic_1": critic_1.to(self._device),
            "critic_2": critic_2.to(self._device),
            "target_critic_1": target_critc_1.to(self._device),
            "target_critic_2": target_critc_2.to(self._device),
        }

        optimizers = {
            key: getattr(torch.optim, optimizer)(value.parameters(), lr=lr)
            for key, value in networks.items()
            if key not in ["target_critic_1", "target_critic_2"]
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
            self.state.networks[key] = value.to(self._device)
        # clean cache
        torch.cuda.empty_cache()

    def update_target_networks(self):
        """
        Update the different networks
        """

        # update the target critic networks softly with the critic networks
        for target_param, param in zip(
            self.state.networks["target_critic_1"].parameters(),
            self.state.networks["critic_1"].parameters(),
        ):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )
            target_param.requires_grad = False

        for target_param, param in zip(
            self.state.networks["target_critic_2"].parameters(),
            self.state.networks["critic_2"].parameters(),
        ):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )
            target_param.requires_grad = False

    def step(
        self,
        state: Tensor,
        action: Tensor,
        next_state: Tensor,
        reward: Tensor,
        done: Tensor,
        phase: Phase = Phase.TRAIN,
    ) -> dict:
        """
        A training or validation step.
        """

        metrics = {}

        with torch.no_grad():
            # critic loss
            next_action, logprobs = self.state.networks["actor"].sample(
                next_state, reparametrize=False
            )

            # avoid overestimation bias by taking the minimum of the two critic networks
            q1_new_policy = self.state.networks["target_critic_1"](
                next_state, next_action
            )
            q2_new_policy = self.state.networks["target_critic_2"](
                next_state, next_action
            )
            q_value = torch.min(q1_new_policy, q2_new_policy)

            # Target value to approximate
            # from q_value = reward + (1-done) * self.gamma * V(s')
            # to  q_value = reward +  (1-do ne) * self.gamma * (Q(s', a') - self.alpha * log(p(a')))
            logprobs = torch.clamp(logprobs, min=-100000, max=100000)
            q_value = reward + (1 - done) * self.gamma * (
                q_value - self.alpha * logprobs
            )

            # print(q_value)
            # input("continue?")
            # print(reward)
            # input("continue?")
            # print(logprobs)
            # input("continue?")
            # print(q_value)
            # print(q_value.shape)
            # input("continue?")

            # print(q_value.shape)
            # input("continue?")
            # print(q_value)
            # input("continue?")

        # Critic loss  time difference error averaged by two q-networks
        q1_old_policy = self.state.networks["critic_1"](state, action)
        q2_old_policy = self.state.networks["critic_2"](state, action)

        # print(q1_old_policy)
        # print(q1_old_policy.shape)
        # input("continue?")
        # print(q2_old_policy.shape)
        # input("continue?")
        # print(q1_old_policy)
        # input("continue?")

        critic_1_loss = F.mse_loss(q1_old_policy, q_value.detach())
        critic_2_loss = F.mse_loss(q2_old_policy, q_value.detach())
        critic_loss = (critic_1_loss + critic_2_loss) * 0.5
        # print(critic_loss.shape)
        # input("continue?")
        # print(critic_loss)
        # input("continue?")

        metrics[f"{phase.value}_critic_1_loss"] = critic_1_loss.item()
        metrics[f"{phase.value}_critic_2_loss"] = critic_2_loss.item()
        metrics[f"{phase.value}_critic_loss"] = critic_loss.item()

        if phase == Phase.TRAIN:
            self.state.optimizers["critic_1"].zero_grad()
            self.state.optimizers["critic_2"].zero_grad()
            critic_loss.backward(retain_graph=True)
            self.state.optimizers["critic_1"].step()
            self.state.optimizers["critic_2"].step()

        # actor loss
        new_actions, logprobs = self.state.networks["actor"].sample(
            state, reparametrize=True
        )
        # bound logprobs
        logprobs = torch.clamp(logprobs, min=-100000, max=100000)

        q1_new_policy = self.state.networks["critic_1"](state, new_actions)
        q2_new_policy = self.state.networks["critic_2"](state, new_actions)

        q_new_policy = torch.min(q1_new_policy, q2_new_policy)

        actor_loss = torch.mean(-q_new_policy + self.alpha * logprobs)  #

        metrics[f"{phase.value}_actor_loss"] = actor_loss.item()

        if phase == Phase.TRAIN:
            self.state.optimizers["actor"].zero_grad()
            actor_loss.backward()
            self.state.optimizers["actor"].step()

        # total loss
        metrics[f"{phase.value}_loss"] = actor_loss.item() + critic_loss.item()

        return metrics

    def run_epoch(
        self,
        buffer: Buffer,
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

        metrics = {}

        for i in tqdm(range(steps), desc=f"Epoch {self.state.epoch}: {phase.value}"):
            # sample a batch of data from the buffer
            state, action, next_state, reward, done = buffer.sample(
                tensor=True, device=self._device
            )

            # Run a step
            if phase == Phase.TRAIN:
                metrics_ = self.step(
                    state, action, next_state, reward, done, phase=phase
                )
                if i % self.update_freq == 0:
                    self.update_target_networks()
            else:
                with torch.no_grad():
                    metrics_ = self.step(
                        state, action, next_state, reward, done, phase=phase
                    )

            for key, value in metrics_.items():
                if key not in metrics:
                    metrics[key] = 0
                metrics[key] += value / steps

        self.state.log_metrics(metrics)

        return metrics

    def fit(
        self,
        train_buffer: Buffer,
        val_buffer: Optional[Buffer] = None,
        epochs: int = 15,
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

        try:
            for _ in range(epochs):
                for callback in self.callbacks:
                    callback.on_epoch_start(self.state)

                self.run_epoch(train_buffer, steps=steps)

                if val_buffer is not None:
                    self.run_epoch(val_buffer, steps=val_steps, phase=Phase.VAL)

                logger.info(f"Metrics: {self.state.get_last_metrics()}")
                for callback in self.callbacks:
                    callback.on_epoch_end(self.state)

                if self.verbose:
                    self.print_metrics()

                self.state.epoch += 1

        except EarlyStopExcpetion:
            if self.verbose:
                self.print_metrics()
            print("Early stopping criteria met.")

    def print_metrics(self):
        """
        Print the metrics of the last epoch.
        """
        print(f"Epoch {self.state.epoch}:")
        for key, value in self.state.metrics.items():
            print(f"{key}: {value[-1]}")
