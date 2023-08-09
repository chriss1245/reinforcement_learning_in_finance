"""
This  file contains utility functions for the datasets. And the modules that use them.
"""

import gym
import numpy as np


class PortfolioDistributionSpace(gym.spaces.Box):  # type: ignore
    """
    Custom gym action space for portfolio distributions. Uses a Dirichlet distribution
    to sample random portfolio distributions.
    """

    def sample(self):
        """
        Samples a random portfolio distribution from a Dirichlet distribution.
        such that sum_i x_i = 1

        Returns:
            np.ndarray: Random portfolio distribution.
        """
        return np.random.dirichlet(np.ones(self.shape), size=1)
