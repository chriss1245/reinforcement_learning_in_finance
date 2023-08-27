"""
This file contains the backtester class.
"""

from typing import Dict, List

import numpy as np
from tqdm import tqdm

from portfolio_management_rl.agents.base import BaseAgent
from portfolio_management_rl.market_environment.buffer import Buffer
from portfolio_management_rl.market_environment.market_env import MarketEnv
from portfolio_management_rl.utils.logger import get_logger

logger = get_logger(__file__)


class Backtester:
    """
    Backtester class, it uses an agent to trade in a market environment and returns the final portfolio value.
    """

    def __init__(
        self,
        agent: BaseAgent,
        market_env: MarketEnv,
        episode_length: int = -1,
        verbose: bool = False,
    ):
        """
        Initialize the backtester.

        Args:
            agent: The agent to use.
            market_env: The market environment to use.
            episode_length: The length of the episode.
            episodes: The number of episodes to run. if -1, it runs all the possible episodes
            buffer_relative_path: The relative path to the buffer.
            phase: The phase of the backtester.
        """
        self.agent = agent
        self.market_env = market_env
        self.episode_length = episode_length

        if self.episode_length == -1:
            self.episode_length = len(self.market_env.dataset) - 1

        self.verbose = verbose

    def run(self) -> Dict[str, float]:
        """
        Run the backtester.

        Returns:
            A tuple containing the states and the actions taken.
        """

        if self.verbose:
            logger.info("Running the backtester.")

        net_worths = []
        cashflows = []
        market_prices = []
        rewards = []

        state = self.market_env.initial_state.copy()

        # the cashflow is obtained later

        for _ in tqdm(range(1, self.episode_length), desc="Backtesting"):
            market_prices.append(np.sum(state.prices))
            net_worths.append(state.net_worth)

            action = self.agent.act(state)
            new_state, reward, done, _, info = self.market_env.step(action)

            cashflows.append(info["cashflow"])
            rewards.append(reward)

            if done:
                break
            state = new_state

        market_prices.append(np.sum(state.prices))
        net_worths.append(state.net_worth)
        cashflows.append(0.0)
        rewards.append(0.0)

        metrics = self.get_metrics(net_worths, cashflows, market_prices, rewards)

        if self.verbose:
            logger.info("Metrics:")
            for key, value in metrics.items():
                logger.info(f"{key}: {value}")

            logger.info("Backtester finished.")

        return metrics, net_worths, cashflows, market_prices, rewards

    def generate_buffer(self) -> Buffer:
        """
        Generates a buffer with the transitions of the backtester.
        """

        if self.episode_length == -1:
            self.episode_length = len(self.market_env)

        self.agent.reset()
        self.market_env.reset()

        buffer = Buffer(buffer_size=self.episode_length)

        state = self.market_env.initial_state.copy()

        for _ in tqdm(range(1, self.episode_length), desc="Generating buffer"):
            action = self.agent.act(state)
            next_state, reward, done, _, _ = self.market_env.step(action)

            buffer.add(state=state, action=action, reward=reward, next_state=next_state)
            if done:
                break
            state = next_state

        return buffer

    def get_metrics(
        self,
        net_worths: List[float],
        cashflows: List[float],
        market_prices: List[float],
        rewards: List[float],
    ) -> Dict[str, float]:
        """
        Calculate and return various financial metrics.

        Args:
            net_worths: List of net worth values over time.
            cashflows: List of cashflows over time.
            market_prices: List of market prices over time.

        Returns:
            A dictionary containing various financial metrics.
        """

        # Convert lists to numpy arrays for easier calculations
        net_worths = np.array(net_worths)
        cashflows = np.array(cashflows)
        market_prices = np.array(market_prices)
        rewards = np.array(rewards)

        # Calculate the average net worth
        avg_net_worth = np.mean(net_worths)

        # Calculate Turnover
        total_cashflow = np.sum(np.abs(cashflows))
        turnover = total_cashflow / avg_net_worth

        # Sharpe
        # Calculate daily returns
        daily_returns = (net_worths[1:] - net_worths[:-1]) / net_worths[:-1]
        # Calculate Sharpe Ratio (Assuming a risk-free rate of 0)
        sharpe_ratio = np.mean(daily_returns) / np.std(daily_returns)

        # Calculate Sortino Ratio (Assuming a risk-free rate of 0)
        downside_returns = daily_returns[daily_returns < 0]
        downside_deviation = np.std(downside_returns)
        sortino_ratio = np.mean(daily_returns) / downside_deviation

        # Drawdown
        # Compute the running maximum of the net worths
        running_max = np.maximum.accumulate(net_worths)
        # Compute rolling drawdown
        rolling_drawdown = (net_worths - running_max) / running_max
        # Maximum Drawdown is the minimum (most negative) rolling drawdown
        max_drawdown = np.min(rolling_drawdown)

        # Calculate Market Returns
        market_returns = (market_prices[1:] - market_prices[:-1]) / market_prices[:-1]
        # Calculate Portfolio Returns
        portfolio_returns = (net_worths[1:] - net_worths[:-1]) / net_worths[:-1]

        # Calculate Beta
        cov_matrix = np.cov(portfolio_returns, market_returns)
        beta = cov_matrix[0, 1] / cov_matrix[1, 1]

        # Calculate Alpha (Assuming a risk-free rate of 0)
        alpha = np.mean(portfolio_returns) - beta * np.mean(market_returns)

        # Calculate Market-Adjusted Returns
        market_adjusted_return = np.mean(portfolio_returns) - np.mean(market_returns)

        # Calculate Information Ratio
        excess_return = portfolio_returns - market_returns
        information_ratio = np.mean(excess_return) / np.std(excess_return)

        # Reward metrics
        total_reward = np.sum(rewards)
        avg_reward = np.mean(rewards)
        std_reward = np.std(rewards)
        sharpe_adjusted_reward = avg_reward / std_reward

        sortino_std = np.std(rewards[rewards < 0])
        sortino_adjusted_reward = avg_reward / sortino_std

        # Compile metrics into a dictionary
        metrics = {
            "sharpe": sharpe_ratio,
            "sortino": sortino_ratio,
            "turnover": turnover,
            "maximum_drawdown": max_drawdown,
            "beta": beta,
            "alpha": alpha,
            "market_djusted_return": market_adjusted_return,
            "information_ratio": information_ratio,
            "sharpe_adjusted_reward": sharpe_adjusted_reward,
            "sortino_adjusted_reward": sortino_adjusted_reward,
            "total_reward": total_reward,
            "avg_reward": avg_reward,
            "std_reward": std_reward,
        }

        return metrics


if __name__ == "__main__":
    from portfolio_management_rl.datasets.dataset import StocksDataset

    dataset = StocksDataset()
    market_env = MarketEnv(dataset=dataset)
    from portfolio_management_rl.agents.equal_weight_agent import EqualWeightAgent

    agent = EqualWeightAgent(rebalance=True)
    backtester = Backtester(agent, market_env, verbose=True)
    backtester.run()

    backtester.generate_buffer()
