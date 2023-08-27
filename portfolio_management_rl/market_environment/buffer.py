"""
Buffer class which stores a the list of transtions for later use in the replay buffer in a 
technique called experience replay for off-line training. This method is used to break the correlation 
between consecutive samples and to avoid the problem of forgetting previous experiences. Also is more 
efficient to train the network with a batch of samples than with a single sample.
"""
from __future__ import annotations

import datetime
from pathlib import Path
from typing import Optional

import h5py
import numpy as np

from portfolio_management_rl.market_environment.commons import MarketEnvState
from portfolio_management_rl.utils.contstants import N_STOCKS, WINDOW_SIZE

MAX_BUFFER_SIZE = 50000


class BufferFullError(Exception):
    """Raised when the buffer is full."""


class MaxBufferSizeError(Exception):
    """Raised when the maximum buffer size is reached."""


class Buffer:
    """
    Buffer for experience replay and imitation learning.
    """

    def __init__(
        self,
        buffer_size: int = 2000,
        batch_size: int = 32,
        n_stocks: int = N_STOCKS,
        window_size: int = WINDOW_SIZE,
    ):
        self.buffer_size = buffer_size
        self.window_size = window_size
        self.batch_size = batch_size
        self.n_stocks = n_stocks

        # actions bufferr composesd by the proposed portfolio distribution and a
        # flag indicating if do apply the proposed portfolio distribution or not
        self.actions_buffer = np.zeros(
            shape=(buffer_size, n_stocks + 1), dtype=np.float32
        )
        self.act_buffer = np.zeros(shape=(buffer_size, 1), dtype=np.float32)

        # states buffer composed by the last window_size days of the market
        # environment and the current portfolio (in number of shaes) (the shares of money worth 1)
        # and the current shares for each stock
        self.states_history_buffer = np.zeros(
            shape=(buffer_size, n_stocks, window_size), dtype=np.float32
        )
        self.states_portfolio_buffer = np.zeros(
            shape=(buffer_size, n_stocks + 1), dtype=np.float32
        )
        self.states_weights_buffer = np.zeros(
            shape=(buffer_size, n_stocks + 1), dtype=np.float32
        )

        # next states buffer composed by the last window_size days of the market
        # environment and the next portfolio distribution
        self.next_states_history_buffer = np.zeros(
            shape=(buffer_size, n_stocks, window_size), dtype=np.float32
        )
        self.next_states_portfolio_buffer = np.zeros(
            shape=(buffer_size, n_stocks + 1), dtype=np.float32
        )
        self.next_states_weights_buffer = np.zeros(
            shape=(buffer_size, n_stocks + 1), dtype=np.float32
        )

        self.rewards_buffer = np.zeros(shape=(buffer_size, 1), dtype=np.float32)
        self.dones_buffer = np.zeros(shape=(buffer_size, 1), dtype=np.int8)
        self.idx = 0

    def add(
        self,
        state: MarketEnvState,
        action: dict,
        next_state: MarketEnvState,
        reward: float,
    ):
        """
        Add a transition to the buffer.
        """

        if self.idx >= self.buffer_size:
            raise BufferFullError

        # add the current state
        self.states_history_buffer[self.idx] = state.history
        self.states_portfolio_buffer[self.idx] = state.portfolio
        self.states_weights_buffer[self.idx] = state.net_distribution

        # add the action
        self.actions_buffer[self.idx] = action["distribution"]
        self.act_buffer[self.idx] = action["rebalance"]

        # add the next state
        self.next_states_history_buffer[self.idx] = next_state.history
        self.next_states_portfolio_buffer[self.idx] = next_state.portfolio
        self.next_states_weights_buffer[self.idx] = next_state.net_distribution

        # add the reward
        self.rewards_buffer[self.idx] = reward

        # add the done flag
        self.dones_buffer[self.idx] = np.int8(next_state.done)

        self.idx += 1

    def extend(self, buffer: Buffer):
        """
        Appends the given buffer to the current buffer.
        """

        if len(self) + len(buffer) > self.buffer_size:
            raise BufferFullError("Not enough space in the buffer to extend it.")

        self.actions_buffer[self.idx : self.idx + len(buffer)] = buffer.actions_buffer
        self.act_buffer[self.idx : self.idx + len(buffer)] = buffer.act_buffer
        self.states_history_buffer[
            self.idx : self.idx + len(buffer)
        ] = buffer.states_history_buffer
        self.states_portfolio_buffer[
            self.idx : self.idx + len(buffer)
        ] = buffer.states_portfolio_buffer
        self.states_weights_buffer[
            self.idx : self.idx + len(buffer)
        ] = buffer.states_weights_buffer
        self.next_states_history_buffer[
            self.idx : self.idx + len(buffer)
        ] = buffer.next_states_history_buffer
        self.next_states_portfolio_buffer[
            self.idx : self.idx + len(buffer)
        ] = buffer.next_states_portfolio_buffer
        self.next_states_weights_buffer[
            self.idx : self.idx + len(buffer)
        ] = buffer.next_states_weights_buffer
        self.rewards_buffer[self.idx : self.idx + len(buffer)] = buffer.rewards_buffer
        self.dones_buffer[self.idx : self.idx + len(buffer)] = buffer.dones_buffer

        self.idx += len(buffer)

    def merge(self, buffer: Buffer):
        """
        Merges the given buffer with the current buffer.
        """

        if len(self) + len(buffer) > MAX_BUFFER_SIZE:
            raise MaxBufferSizeError("Maximun buffer size reached")

        self.actions_buffer = np.concatenate(
            [
                self.actions_buffer[: self.idx, :],
                buffer.actions_buffer[: len(buffer) - 1, :],
            ]
        )
        self.act_buffer = np.concatenate(
            [self.act_buffer[: self.idx], buffer.act_buffer[: len(buffer) - 1]]
        )
        self.states_history_buffer = np.concatenate(
            [
                self.states_history_buffer[: self.idx, :, :],
                buffer.states_history_buffer[: len(buffer) - 1, :, :],
            ]
        )
        self.states_portfolio_buffer = np.concatenate(
            [
                self.states_portfolio_buffer[: self.idx, :],
                buffer.states_portfolio_buffer[: len(buffer) - 1, :],
            ]
        )
        self.states_weights_buffer = np.concatenate(
            [
                self.states_weights_buffer[: self.idx, :],
                buffer.states_weights_buffer[: len(buffer) - 1, :],
            ]
        )
        self.next_states_history_buffer = np.concatenate(
            [
                self.next_states_history_buffer[: self.idx, :, :],
                buffer.next_states_history_buffer[: len(buffer) - 1, :, :],
            ]
        )

        self.next_states_portfolio_buffer = np.concatenate(
            [
                self.next_states_portfolio_buffer[: self.idx, :],
                buffer.next_states_portfolio_buffer[: len(buffer) - 1, :],
            ]
        )

        self.next_states_weights_buffer = np.concatenate(
            [
                self.next_states_weights_buffer[: self.idx, :],
                buffer.next_states_weights_buffer[: len(buffer) - 1, :],
            ]
        )

        self.rewards_buffer = np.concatenate(
            [self.rewards_buffer[: self.idx], buffer.rewards_buffer[: len(buffer) - 1]]
        )
        self.dones_buffer = np.concatenate(
            [self.dones_buffer[: self.idx], buffer.dones_buffer[: len(buffer) - 1]]
        )

        self.buffer_size += len(buffer)

        self.idx += len(buffer)

    def sample(self, batch_size: Optional[int] = None, prioritized: bool = False):
        """
        Sample a batch of transitions from the buffer.

        Args:
            batch_size: The size of the batch to sample.
            prioritized: If True, sample the batch with prioritized sampling
                (the latest transitions have more probability to be sampled).

        Returns:
            A tuple with the states, actions, next states, rewards and dones. represented as numpy arrays
            and dictionaries.
        """

        batch_size = batch_size or self.batch_size

        if prioritized:
            idxs_prob = np.arange(self.idx, 0, -1)
            idxs_prob = idxs_prob / np.sum(idxs_prob)
            idxs = np.random.choice(
                np.arange(self.idx), size=(batch_size,), p=idxs_prob
            )
        else:
            idxs = np.random.randint(
                low=0, high=self.idx, size=(batch_size,), dtype=np.int32
            )

        actions = {
            "distribution": self.actions_buffer[idxs],
            "rebalance": self.act_buffer[idxs],
        }
        states = {
            "history": self.states_history_buffer[idxs],
            "net_distribution": self.states_weights_buffer[idxs],
        }

        next_states = {
            "history": self.next_states_history_buffer[idxs],
            "net_distribution": self.next_states_portfolio_buffer[idxs],
        }

        return (
            states,
            actions,
            next_states,
            self.rewards_buffer[idxs],
            self.dones_buffer[idxs],
        )

    def reset(self):
        """
        Reset the buffer.
        """
        self.idx = 0

    def save(self, path: Path, prune: bool = True):
        """
        Save the buffer to a file as an hdf5 file.

        Args:
            path: The path to the file to save the buffer to.
        """

        path.parent.mkdir(exist_ok=True, parents=True)

        idx = self.buffer_size - 1
        if prune:
            idx = self.idx

        with h5py.File(str(path), "w") as f:
            f.create_dataset("actions_buffer", data=self.actions_buffer[:idx])
            f.create_dataset("act_buffer", data=self.act_buffer[:idx])
            f.create_dataset(
                "states_history_buffer", data=self.states_history_buffer[:idx]
            )
            f.create_dataset(
                "states_portfolio_buffer", data=self.states_portfolio_buffer[:idx]
            )
            f.create_dataset(
                "states_weights_buffer", data=self.states_weights_buffer[:idx]
            )
            f.create_dataset(
                "next_states_history_buffer", data=self.next_states_history_buffer[:idx]
            )
            f.create_dataset(
                "next_states_portfolio_buffer",
                data=self.next_states_portfolio_buffer[:idx],
            )
            f.create_dataset(
                "next_states_weights_buffer", data=self.next_states_weights_buffer[:idx]
            )
            f.create_dataset("rewards_buffer", data=self.rewards_buffer[:idx])
            f.create_dataset("dones_buffer", data=self.dones_buffer[:idx])

            # metada
            f.attrs["window_size"] = self.window_size
            f.attrs["buffer_size"] = len(self) if prune else self.buffer_size
            f.attrs["n_stocks"] = self.n_stocks
            f.attrs["batch_size"] = self.batch_size
            f.attrs["idx"] = self.idx
            f.attrs["date"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def load(self, path: Path):
        """
        Loads the buffer data from the hdf5 file.
        """

        with h5py.File(str(path), "r") as f:
            # check that this concides
            assert self.window_size == f.attrs["window_size"], "Window size mismatch"
            assert self.n_stocks == f.attrs["n_stocks"], "N stocks mismatch"

            self.actions_buffer = f["actions_buffer"][:]
            self.act_buffer = f["act_buffer"][:]
            self.states_history_buffer = f["states_history_buffer"][:]
            self.states_portfolio_buffer = f["states_portfolio_buffer"][:]
            self.states_weights_buffer = f["states_weights_buffer"][:]
            self.next_states_history_buffer = f["next_states_history_buffer"][:]
            self.next_states_portfolio_buffer = f["next_states_portfolio_buffer"][:]
            self.next_states_weights_buffer = f["next_states_weights_buffer"][:]
            self.rewards_buffer = f["rewards_buffer"][:]
            self.dones_buffer = f["dones_buffer"][:]

            self.idx = f.attrs["idx"]
            self.buffer_size = f.attrs["buffer_size"]
            self.batch_size = f.attrs["batch_size"]

    @staticmethod
    def load_from_file(path: Path):
        """
        Loads the buffer data from the hdf5 file.
        """
        with h5py.File(str(path), "r") as f:
            window_size = f.attrs["window_size"]
            buffer_size = f.attrs["buffer_size"]
            n_stocks = f.attrs["n_stocks"]
            batch_size = f.attrs["batch_size"]

            buffer = Buffer(
                buffer_size=buffer_size,
                batch_size=batch_size,
                n_stocks=n_stocks,
                window_size=window_size,
            )

            buffer.actions_buffer = f["actions_buffer"][:]
            buffer.act_buffer = f["act_buffer"][:]
            buffer.states_history_buffer = f["states_history_buffer"][:]
            buffer.states_portfolio_buffer = f["states_portfolio_buffer"][:]
            buffer.states_weights_buffer = f["states_weights_buffer"][:]
            buffer.next_states_history_buffer = f["next_states_history_buffer"][:]
            buffer.next_states_portfolio_buffer = f["next_states_portfolio_buffer"][:]
            buffer.next_states_weights_buffer = f["next_states_weights_buffer"][:]
            buffer.rewards_buffer = f["rewards_buffer"][:]
            buffer.dones_buffer = f["dones_buffer"][:]
            buffer.idx = f["idx"][:]
            return buffer

    def __len__(self):
        """
        Return the number of transitions in the buffer.
        """
        return self.idx + 1

    def __iter__(self):
        """
        Iterate over the buffer.
        """
        for idx in range(self.idx):
            actions = {
                "distribution": self.actions_buffer[idx],
                "rebalance": self.act_buffer[idx],
            }
            states = {
                "history": self.states_history_buffer[idx],
                "net_distribution": self.states_weights_buffer[idx],
            }

            next_states = {
                "history": self.next_states_history_buffer[idx],
                "net_distribution": self.next_states_portfolio_buffer[idx],
            }

            yield (
                states,
                actions,
                next_states,
                self.rewards_buffer[idx],
                self.dones_buffer[idx],
            )

    def __getitem__(self, idx):
        """
        Return the transition at the given index.
        """
        if idx >= self.idx:
            raise IndexError("Index out of bounds.")

        actions = {
            "distribution": self.actions_buffer[idx],
            "rebalance": self.act_buffer[idx],
        }
        states = {
            "history": self.states_history_buffer[idx],
            "net_distribution": self.states_weights_buffer[idx],
        }

        next_states = {
            "history": self.next_states_history_buffer[idx],
            "net_distribution": self.next_states_portfolio_buffer[idx],
        }

        return (
            states,
            actions,
            next_states,
            self.rewards_buffer[idx],
            self.dones_buffer[idx],
        )
