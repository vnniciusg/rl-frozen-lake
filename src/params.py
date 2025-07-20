"""
Configuration parameters for Reinforcement Learning on FrozenLake environment.

This module contains the parameter configuration class used to define hyperparameters
and environment settings for training reinforcement learning agents on the FrozenLake
environment from OpenAI Gym.

Classes:
    Params: Dataclass containing all configuration parameters for RL training.

Usage:
    from src.params import params

    # Access configured parameters
    episodes = params.total_episodes
    lr = params.learning_rate

Author: vnniciusg
"""

from dataclasses import dataclass


@dataclass
class Params:
    """
    Configuration parameters for reinforcement learning on FrozenLake environment.

    This dataclass encapsulates all hyperparameters and environment settings
    required for training and evaluating reinforcement learning agents on the
    FrozenLake environment.

    Attributes:
        total_episodes (int): Total number of training episodes to run.
        learning_rate (float): Step size parameter for Q-learning updates (typically 0.01-0.1).
        gamma (float): Discount factor for future rewards (typically 0.9-0.99).
        epsilon (float): Exploration probability for epsilon-greedy policy (0.0-1.0).
        map_size (int): Dimension of the square FrozenLake grid (e.g., 4 for 4x4 grid).
        seed (int): Random seed for reproducible results across runs.
        is_slippery (bool): Environment dynamics setting. If True, the agent moves in
            the intended direction with probability 1/3, and in perpendicular directions
            with probability 1/3 each. If False, actions are deterministic.
        n_runs (int): Number of independent training runs for statistical analysis.
        action_size (int): Number of possible actions in the environment (typically 4).
        state_size (int): Number of possible states in the environment (map_size^2).
        proba_frozen (float): Probability that a randomly generated tile is frozen
            (safe to walk on) rather than a hole (0.0-1.0).

    Example:
        >>> params = Params(
        ...     total_episodes=1000,
        ...     learning_rate=0.1,
        ...     gamma=0.95,
        ...     epsilon=0.1,
        ...     map_size=4,
        ...     seed=42,
        ...     is_slippery=True,
        ...     n_runs=10,
        ...     action_size=4,
        ...     state_size=16,
        ...     proba_frozen=0.8
        ... )
    """

    total_episodes: int
    learning_rate: float
    gamma: float
    epsilon: float
    map_size: int
    seed: int
    is_slippery: bool
    n_runs: int
    action_size: int
    state_size: int
    proba_frozen: float


params = Params(
    total_episodes=2000,
    learning_rate=0.8,
    gamma=0.95,
    epsilon=0.1,
    map_size=5,
    seed=123,
    is_slippery=False,
    n_runs=20,
    action_size=None,
    state_size=None,
    proba_frozen=0.9,
)
