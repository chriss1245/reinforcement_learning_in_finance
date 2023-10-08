"""
This file contains the backtester class.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

from portfolio_management_rl.agents.base import BaseAgent
from portfolio_management_rl.dataloaders.buffer import Buffer
from portfolio_management_rl.market_environment.market_env import MarketEnv
from portfolio_management_rl.utils.dtypes import Phase

from portfolio_management_rl.utils.logger import get_logger

# import garbarge collector
import gc

logger = get_logger(__file__)


class MATRIX:
    """
    MATRIX: Multi-Agent Training and Review for Investment eXecution
    Is the class which performs simulations for backtesting, and training agents.
    """

    def __init__(
        self,
        market_env: MarketEnv,
        validation_env: MarketEnv,
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
        self.market_env = market_env
        self.validation_env = validation_env
        self.episode_length = episode_length

        if self.episode_length == -1:
            self.episode_length = len(self.market_env.dataset) - 1

        self.verbose = verbose

    def backtest(
        self, agent: BaseAgent, phase: Phase = Phase.TRAIN, start: int = 0
    ) -> Dict[str, float]:
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
        portfolios = []
        rebalance_portfolios = []
        rebalances = []

        if phase == Phase.TRAIN:
            env = self.market_env
        elif phase == Phase.VAL:
            env = self.validation_env
        else:  # pragma: no cover
            raise ValueError("Invalid phase.")

        env.reset()
        env.change_start_point(start)
        state = env.initial_state.copy()

        agent.reset()

        # the cashflow is obtained later

        if self.episode_length == -1:
            episode_length = len(env)
        else:
            episode_length = self.episode_length

        for _ in tqdm(range(1, episode_length), desc="Backtesting"):
            market_prices.append(np.sum(state.prices))
            net_worths.append(state.net_worth)
            portfolios.append(state.net_distribution)

            action = agent.act(state)

            rebalance_portfolios.append(action["distribution"])
            rebalances.append(action["rebalance"])
            new_state, reward, done, _, info = env.step(action)

            cashflows.append(info["cashflow"])
            rewards.append(reward)

            state = new_state
            if done:
                break

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

        return (
            metrics,
            net_worths,
            cashflows,
            market_prices,
            rewards,
            portfolios,
            rebalances,
        )

    def experience_replay_training(
        self,
        agent: BaseAgent,
        example_agent: Optional[BaseAgent] = None,
        prefill: bool = False,
        n_iterations: int = 10,
    ) -> Dict[str, float]:
        """
        Train the agents using experience replay.

        Args:
            prefiling_agent: The agent used to generate the buffer initial buffer.
        """

        if self.verbose:
            logger.info("Running the backtester.")

        if example_agent is not None:
            logger.info("Generating validation buffer.")
            validation_buffer = self.generate_buffer(
                example_agent, n_episodes=10, random=True, phase=Phase.VAL
            )
        else:
            validation_buffer = None

        if prefill and example_agent is not None:
            if self.verbose:
                logger.info("Pretraining the agent.")

            prefilled_buffer = self.generate_buffer(
                example_agent, n_episodes=10, random=True
            )
            agent.learn(prefilled_buffer, validation_buffer=validation_buffer)

        gc.collect()

        for episode in (pbar := tqdm(range(n_iterations), desc="Training")):
            agent.reset()
            self.market_env.reset()
            pbar.set_postfix_str("Generating buffer.")
            buffer = self.generate_buffer(agent, n_episodes=5, random=True)
            pbar.set_postfix_str("Learning.")
            agent.learn(buffer, validation_buffer=validation_buffer)

            gc.collect()

    def generate_buffer(
        self, agent: BaseAgent, n_episodes=5, random=True, phase=Phase.TRAIN
    ) -> Buffer:
        """
        Generates a buffer with the transitions of the backtester.
        """

        if phase == Phase.TRAIN:
            env = self.market_env
        elif phase == Phase.VAL:
            env = self.validation_env
        else:
            raise ValueError("Invalid phase.")

        if self.episode_length == -1:
            episode_length = len(env)
        else:
            episode_length = self.episode_length

        env.reset()

        buffer_big = Buffer(buffer_size=1)

        state = self.market_env.initial_state.copy()

        for episode in range(n_episodes):
            starting_point = episode
            if random:
                starting_point = np.random.randint(episode, len(env))

            buffer = Buffer(buffer_size=episode_length - starting_point + 1)
            env.change_start_point(starting_point)
            agent.reset()
            done = False
            while not done:
                action = agent.act(state)
                next_state, reward, done, _, _ = env.step(action)

                buffer.add(
                    state=state, action=action, reward=reward, next_state=next_state
                )
                state = next_state

            buffer_big.merge(buffer)

        return buffer_big

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
            # "sharpe_adjusted_reward": sharpe_adjusted_reward,
            # "sortino_adjusted_reward": sortino_adjusted_reward,
            "avg_reward": avg_reward,
            # "std_reward": std_reward,
        }

        return metrics


if __name__ == "__main__":
    from portfolio_management_rl.dataloaders.dataset import StocksDataset, MixUpConfig

    dataset = StocksDataset()
    market_env = MarketEnv(dataset=dataset)

    validation_dataset = StocksDataset(phase=Phase.VAL)
    validation_env = MarketEnv(dataset=validation_dataset)
    from portfolio_management_rl.agents.equal_weight_agent import EqualWeightAgent
    from portfolio_management_rl.agents import RandomAgent
    from portfolio_management_rl.agents.efficient_frotier_agent import (
        EfficientFrontierAgent,
    )
    from portfolio_management_rl.agents.sac.sac_agent import SACAgent
    from portfolio_management_rl.utils.contstants import PROJECT_DIR

    example_agent = RandomAgent()
    # the second one does not serve
    agent = SACAgent(
        actor="normal",
        critic="bilinear",
        # encoder_path=None,
        freeze_encoder=True,
        # checkpoint_path=PROJECT_DIR / "logs/trial1/checkpoints/m1.4383_e000"
    )

    matrix = MATRIX(market_env, validation_env, verbose=True)
    matrix.experience_replay_training(agent, example_agent, prefill=False)
    matrix.backtest(agent)
