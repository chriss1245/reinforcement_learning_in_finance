"""
This file contains the backtester class.
"""

from portfolio_management_rl.agents.base import BaseAgent
from portfolio_management_rl.market_environment.market_env import MarketEnv
from portfolio_management_rl.market_environment.commons import MarketEnvState

from portfolio_management_rl.utils.logger import get_logger

from typing import List, Tuple, Dict, Any

logger = get_logger(__file__)


class Backtester:
    """
    Backtester class, it uses an agent to trade in a market environment and returns the final portfolio value.
    """

    def __init__(
        self, agent: BaseAgent, market_env: MarketEnv, episode_length: int = 2
    ):
        """
        Initialize the backtester.

        Args:
            agent: The agent to use.
            market_env: The market environment to use.
            episode_length: The length of the episode.
        """
        self.agent = agent
        self.market_env = market_env
        self.episode_length = episode_length

    def run(self) -> Tuple[List[MarketEnvState], List[Dict[str, Any]]]:
        """
        Run the backtester.

        Returns:
            A tuple containing the states and the actions taken.
        """
        logger.info("Running the backtester.")
        states = []
        actions = []
        rewards = []
        state = self.market_env.initial_state.copy()
        action = self.agent.act(state)
        state, reward, done = self.market_env.step(action)
        for _ in range(self.episode_length - 1):
            state = self.market_env.current_state.copy()
            action = self.agent.act(state)
            state, reward, done = self.market_env.step(action)
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            if done:
                break
        logger.info("Backtester finished.")
        return states, actions, rewards


if __name__ == "__main__":
    from portfolio_management_rl.datasets.dataset import StocksDataset

    dataset = StocksDataset()
    market_env = MarketEnv(dataset=dataset)
    from portfolio_management_rl.agents.equal_weight_agent import EqualWeightAgent

    agent = EqualWeightAgent(rebalance=False)
    backtester = Backtester(agent, market_env)
    states, actions, rewards = backtester.run()

    print("INITIAL NET WORTH: ", market_env.initial_state.net_worth)
    print("FINAL NET WORTH: ", states[-1].net_worth)
    print("REWARD: ", sum([r for r in rewards]))
