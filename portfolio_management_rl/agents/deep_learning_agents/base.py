"""
Base drl agent class
"""

import torch
from torch import nn
from torch.nn import functional as F

from portfolio_management_rl.agents.base import BaseAgent
from portfolio_management_rl.market_environment.buffer import Buffer
from abc import ABC, abstractmethod


class BaseDRLAgent(BaseAgent, ABC):
    """
    Base interface for deep reinforcement learning agents.
    """

    @abstractmethod
    def learn(self, buffer: Buffer):
        """
        Learn from the buffer.
        """
