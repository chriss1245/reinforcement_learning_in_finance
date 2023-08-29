"""
Utility functions for neural networks.
"""

from pathlib import Path
from typing import Optional

import torch


def save_checkpoint(
    path: Path,
    network: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> None:
    """
    Saves a checkpoint of the network.

    Args:
        network (torch.nn.Module): Network to save the checkpoint from.
        optimizer (torch.optim.Optimizer): Optimizer to save the checkpoint from.
        path (str): Path to save the network to.
        **metadata: Additional metadata to save.
    """
    torch.save(
        {
            "state_dict": network.state_dict(),
            "optimizer": optimizer.state_dict() if optimizer is not None else None,
        },
        path,
    )


def load_checkpoint(
    path: Path,
    network: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> None:
    """
    Loads a checkpoint of the network.

    Args:
        path (str): Path to load the network from.
        network (torch.nn.Module): Network to load the checkpoint into.
        optimizer (torch.optim.Optimizer): Optimizer to load the checkpoint into.
    """
    checkpoint = torch.load(path)

    network.load_state_dict(checkpoint["state_dict"])

    if optimizer is not None:
        opt_state = checkpoint["optimizer"]
        if opt_state is None:
            raise ValueError("Optimizer state is None")

        optimizer.load_state_dict(opt_state)
