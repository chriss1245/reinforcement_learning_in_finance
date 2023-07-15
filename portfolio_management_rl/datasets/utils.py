import numpy as np
import gym


class PortfolioDistributionSpace(gym.spaces.Box):
    """
    Custom gym action space for portfolio distributions.
    """

    def sample(self):
        """
        Samples a random portfolio distribution.
        """
        samp = super().sample()
        return samp / np.sum(samp)