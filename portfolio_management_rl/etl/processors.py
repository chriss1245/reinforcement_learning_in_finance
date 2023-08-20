"""
This module contains tools for processing the historical data from stock prices.
"""

from abc import ABC, abstractmethod


class Processor(ABC):
    """
    Abstract class for processing the historical data from stock prices.
    """


import torch


class NN(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, *args, **kwargs):
        pass


lol = NN()

lol.to("cuda")

input("Press Enter to continue...")
