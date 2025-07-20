"""
Training execution module for Q-Learning reinforcement learning experiments.

This module contains the main training loop for running Q-Learning experiments
on FrozenLake environments. It handles multiple training runs, episode execution,
data collection, and returns comprehensive training statistics for analysis.

Functions:
    run_env: Execute multiple Q-Learning training runs and collect performance data.

Dependencies:
    - numpy: For numerical computations and data storage
    - tqdm: For progress bar visualization during training
    - src.params: For global configuration parameters
    - src.q_learning: For Q-Learning algorithm and exploration strategy

Example:
    >>> from src.run_env import run_env
    >>> from src.q_learning import QLearning, EpsilonGreedy
    >>> from src.environment import setup_environment
    >>>
    >>> learner = QLearning(learning_rate=0.1, gamma=0.95, state_size=16, action_size=4)
    >>> explorer = EpsilonGreedy(epsilon=0.1)
    >>> env = setup_environment()
    >>>
    >>> rewards, steps, episodes, qtables, states, actions = run_env(learner, explorer, env)

Author: vnniciusg
"""

import gymnasium as gym
import numpy as np
from tqdm import tqdm

from src.params import params
from src.q_learning import EpsilonGreedy, QLearning


def run_env(
    learner: QLearning, explorer: EpsilonGreedy, env: gym.Env
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[int], list[int]]:
    """
    Execute multiple Q-Learning training runs on the given environment.

    This function runs Q-Learning training for multiple independent runs, where each
    run consists of multiple episodes. It collects comprehensive statistics including
    rewards, episode lengths, final Q-tables, and state-action trajectories for
    analysis and evaluation purposes.

    Args:
        learner (QLearning): Q-Learning algorithm instance for updating Q-values.
        explorer (EpsilonGreedy): Epsilon-greedy policy for action selection.
        env (gym.Env): Gymnasium environment to train on (typically FrozenLake).

    Returns:
        Tuple containing:
            rewards (np.ndarray): Array of shape (total_episodes, n_runs) containing
                total rewards collected in each episode for each run.
            steps (np.ndarray): Array of shape (total_episodes, n_runs) containing
                number of steps taken in each episode for each run.
            episodes (np.ndarray): Array of episode indices from 0 to total_episodes-1.
            qtables (np.ndarray): Array of shape (n_runs, state_size, action_size)
                containing the final Q-table for each run.
            all_states (list[int]): List of all states visited during training
                across all runs and episodes.
            all_actions (list[int]): List of all actions taken during training
                across all runs and episodes, corresponding to all_states.

    Note:
        - The Q-table is reset at the beginning of each run for independent training
        - Progress is displayed using tqdm progress bars for each run
        - The environment is reset with the same seed for each episode to ensure
          consistent starting conditions within each run
        - State-action trajectories are collected across all runs for analysis

    Example:
        >>> rewards, steps, episodes, qtables, states, actions = run_env(
        ...     learner=q_learner,
        ...     explorer=epsilon_policy,
        ...     env=frozen_lake_env
        ... )
        >>> print(f"Average reward over all runs: {np.mean(rewards):.3f}")
        >>> print(f"Final Q-table shape: {qtables.shape}")
    """

    rewards = np.zeros((params.total_episodes, params.n_runs))
    steps = np.zeros((params.total_episodes, params.n_runs))
    episodes = np.arange(params.total_episodes)
    qtables = np.zeros((params.n_runs, params.state_size, params.action_size))

    all_states, all_actions = [], []

    for run in range(params.n_runs):
        learner.reset_qtable()

        for episode in tqdm(
            episodes, desc=f"Run {run}/{params.n_runs} - Episodes", leave=False
        ):
            state = env.reset(seed=params.seed)[0]
            step = 0
            done = False
            total_rewards = 0

            while not done:
                action = explorer.choose_action(
                    action_space=env.action_space, state=state, qtable=learner.qtable
                )

                all_states.append(state)
                all_actions.append(action)

                new_state, reward, terminated, truncated, _ = env.step(action)

                done = terminated or truncated

                learner.qtable[state, action] = learner.update(
                    state, action, reward, new_state
                )

                total_rewards += reward
                step += 1

                state = new_state

            rewards[episode, run] = total_rewards
            steps[episode, run] = step

        qtables[run, :, :] = learner.qtable

    return rewards, steps, episodes, qtables, all_states, all_actions
