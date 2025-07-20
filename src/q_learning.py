"""
Q-Learning algorithm implementation and action selection strategies.

This module contains the core components for implementing Q-Learning reinforcement
learning algorithm, including the Q-table management and epsilon-greedy action
selection strategy. These components work together to enable an agent to learn
optimal policies in discrete state-action environments like FrozenLake.

Classes:
    QLearning: Implements the Q-Learning algorithm with Q-table updates.
    EpsilonGreedy: Implements epsilon-greedy action selection strategy.

Dependencies:
    - numpy: For numerical computations and Q-table management
    - src.params: For global configuration parameters

Example:
    >>> from src.q_table import QLearning, EpsilonGreedy
    >>> q_learner = QLearning(learning_rate=0.1, gamma=0.95, state_size=16, action_size=4)
    >>> policy = EpsilonGreedy(epsilon=0.1)
    >>> action = policy.choose_action(env.action_space, state, q_learner.qtable)

Author: vnniciusg
"""

import gymnasium as gym
import numpy as np

from src.params import params


class QLearning:
    """
    Q-Learning algorithm implementation for reinforcement learning.

    This class implements the Q-Learning algorithm, maintaining a Q-table that
    stores state-action values and provides methods for updating these values
    based on the Bellman equation. The Q-table is used to learn the optimal
    policy for an agent in a discrete state-action environment.

    Attributes:
        state_size (int): Number of possible states in the environment.
        action_size (int): Number of possible actions in the environment.
        learning_rate (float): Step size for Q-value updates (0 < lr <= 1).
        gamma (float): Discount factor for future rewards (0 <= gamma <= 1).
        qtable (np.ndarray): Q-table storing state-action values.

    Example:
        >>> q_learner = QLearning(learning_rate=0.1, gamma=0.95, state_size=16, action_size=4)
        >>> new_q_value = q_learner.update(state=0, action=1, reward=1.0, new_state=4)
        >>> q_learner.reset_qtable()  # Reset for new training run
    """

    def __init__(
        self, learning_rate: float, gamma: float, state_size: int, action_size: int
    ) -> None:
        """
        Initialize the Q-Learning algorithm.

        Args:
            learning_rate (float): Step size for Q-value updates. Controls how much
                new information overrides old information (0 < lr <= 1).
            gamma (float): Discount factor for future rewards. Determines the
                importance of future rewards (0 <= gamma <= 1).
            state_size (int): Number of possible states in the environment.
            action_size (int): Number of possible actions in the environment.
        """
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.reset_qtable()

    def update(
        self,
        state: int,
        action: int,
        reward: float,
        new_state: int,
    ) -> float:
        """
        Update Q-value using the Q-Learning algorithm.

        Implements the Q-Learning update rule:
        Q(s,a) := Q(s,a) + lr * [R(s,a) + gamma * max Q(s',a') - Q(s,a)]

        Args:
            state (int): Current state index.
            action (int): Action taken in the current state.
            reward (float): Reward received for taking the action.
            new_state (int): Next state index after taking the action.

        Returns:
            float: Updated Q-value for the state-action pair.

        Note:
            This method calculates but does not store the updated Q-value.
            The caller is responsible for updating the Q-table with the returned value.
        """

        delta = (
            reward
            + self.gamma * np.max(self.qtable[new_state, :])
            - self.qtable[state, action]
        )

        return self.qtable[state, action] + self.learning_rate * delta

    def reset_qtable(self) -> None:
        """
        Reset the Q-table to zeros.

        Initializes or resets the Q-table with zeros for all state-action pairs.
        This is useful when starting a new training run or experiment.
        """
        self.qtable = np.zeros((self.state_size, self.action_size))


class EpsilonGreedy:
    """
    Epsilon-greedy action selection strategy for reinforcement learning.

    This class implements the epsilon-greedy policy, which balances exploration
    and exploitation by selecting random actions with probability epsilon and
    greedy actions (based on Q-values) with probability (1 - epsilon).

    Attributes:
        epsilon (float): Exploration probability (0 <= epsilon <= 1).
        rng (np.random.Generator): Random number generator for reproducible results.

    Example:
        >>> policy = EpsilonGreedy(epsilon=0.1)
        >>> action = policy.choose_action(env.action_space, state=0, qtable=q_table)
    """

    def __init__(self, epsilon: float) -> None:
        """
        Initialize the epsilon-greedy policy.

        Args:
            epsilon (float): Exploration probability. The probability of selecting
                a random action instead of the greedy action (0 <= epsilon <= 1).
        """
        self.epsilon = epsilon
        self.rng = np.random.default_rng(params.seed)

    def choose_action(
        self, action_space: gym.Space, state: int, qtable: np.ndarray
    ) -> int:
        """
        Choose an action using epsilon-greedy strategy.

        With probability epsilon, selects a random action for exploration.
        With probability (1 - epsilon), selects the action with the highest
        Q-value for the given state (exploitation). In case of ties in Q-values,
        randomly selects among the tied actions.

        Args:
            action_space (gym.Space): Action space of the environment for sampling
                random actions during exploration.
            state (int): Current state index for which to select an action.
            qtable (np.ndarray): Q-table containing state-action values with shape
                (state_size, action_size).

        Returns:
            int: Selected action index.

        Note:
            Uses the random number generator initialized with the global seed
            for reproducible action selection.
        """
        expor_exploit_tradeoff = self.rng.uniform(0, 1)

        if expor_exploit_tradeoff < self.epsilon:
            return action_space.sample()

        return self.rng.choice(np.where(qtable[state, :] == max(qtable[state, :]))[0])
