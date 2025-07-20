"""
Visualization utilities for Q-Learning reinforcement learning experiments.

This module provides comprehensive visualization functions for analyzing and
presenting Q-Learning training results on FrozenLake environments. It includes
data processing utilities and plotting functions for Q-values, training metrics,
and agent behavior analysis.

The visualization functions in this module are adapted from the Gymnasium
FrozenLake Q-Learning tutorial, providing standardized ways to analyze
reinforcement learning experiments.

Functions:
    postprocess: Convert training results into pandas DataFrames for analysis.
    qtable_directions_map: Extract Q-value heatmap and policy directions from Q-table.
    plot_q_values_map: Visualize the learned Q-values and policy as a heatmap.
    plot_states_actions_distribution: Plot histograms of state and action visitation.
    plot_steps_and_rewards: Plot training progress metrics over episodes.

Dependencies:
    - numpy: For numerical computations and array operations
    - pandas: For data manipulation and DataFrame creation
    - matplotlib.pyplot: For plot creation and display
    - seaborn: For enhanced statistical visualizations
    - gymnasium: For environment rendering

Reference:
    This module is based on the Gymnasium FrozenLake Q-Learning tutorial:
    https://gymnasium.farama.org/tutorials/training_agents/frozenlake_q_learning/#sphx-glr-tutorials-training-agents-frozenlake-q-learning-py

Example:
    >>> import numpy as np
    >>> from src.visualization import postprocess, plot_q_values_map
    >>>
    >>> # Process training results
    >>> rewards_df, steps_df = postprocess(episodes, params, rewards, steps, map_size=4)
    >>>
    >>> # Visualize learned policy
    >>> plot_q_values_map(qtable, env, map_size=4)
"""

from typing import Any

import gymnasium as gym
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

matplotlib.use("Agg")


def postprocess(
    episodes: np.ndarray,
    params: Any,
    rewards: np.ndarray,
    steps: np.ndarray,
    map_size: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Convert training simulation results into pandas DataFrames for analysis.

    This function processes raw training data (rewards and steps) from multiple
    runs and episodes, converting them into structured DataFrames suitable for
    statistical analysis and visualization.

    Args:
        episodes (np.ndarray): Array of episode indices from the training runs.
        params (Any): Parameters object containing training configuration,
            specifically params.n_runs for the number of independent runs.
        rewards (np.ndarray): Array of shape (total_episodes, n_runs) containing
            rewards collected in each episode for each run.
        steps (np.ndarray): Array of shape (total_episodes, n_runs) containing
            number of steps taken in each episode for each run.
        map_size (int): Size of the square FrozenLake grid (e.g., 4 for 4x4).

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: A tuple containing:
            - rewards_df: DataFrame with columns ['Episodes', 'Rewards', 'Steps',
              'cum_rewards', 'map_size'] where each row represents one episode
              from one run, with cumulative rewards calculated.
            - steps_df: DataFrame with columns ['Episodes', 'Steps', 'map_size']
              containing episode-wise averaged steps across all runs.

    Note:
        - Data is flattened using Fortran order ('F') to maintain run grouping
        - Cumulative rewards are calculated across episodes within each run
        - Map size is added as a categorical variable for multi-size comparisons

    Example:
        >>> episodes = np.arange(1000)
        >>> rewards_df, steps_df = postprocess(episodes, params, rewards, steps, 4)
        >>> print(f"Total data points: {len(rewards_df)}")
        >>> print(f"Average episodes: {len(steps_df)}")
    """
    res = pd.DataFrame(
        data={
            "Episodes": np.tile(episodes, reps=params.n_runs),
            "Rewards": rewards.flatten(order="F"),
            "Steps": steps.flatten(order="F"),
        }
    )
    res["cum_rewards"] = rewards.cumsum(axis=0).flatten(order="F")
    res["map_size"] = np.repeat(f"{map_size}x{map_size}", res.shape[0])

    st = pd.DataFrame(data={"Episodes": episodes, "Steps": steps.mean(axis=1)})
    st["map_size"] = np.repeat(f"{map_size}x{map_size}", st.shape[0])
    return res, st


def qtable_directions_map(
    qtable: np.ndarray, map_size: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract Q-value heatmap and policy directions from a Q-table.

    This function processes a Q-table to create visualization-ready data by
    extracting maximum Q-values for each state and mapping the best actions
    to directional arrows for policy visualization.

    Args:
        qtable (np.ndarray): Q-table of shape (state_size, action_size) containing
            learned state-action values.
        map_size (int): Size of the square grid environment (e.g., 4 for 4x4).

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing:
            - qtable_val_max: 2D array of shape (map_size, map_size) with maximum
              Q-values for each state, reshaped to match the grid layout.
            - qtable_directions: 2D array of shape (map_size, map_size) with
              directional arrows ('←', '↓', '→', '↑') representing the best
              action for each state. Empty strings for states with no learning.

    Note:
        - Actions are mapped as: 0=LEFT(←), 1=DOWN(↓), 2=RIGHT(→), 3=UP(↑)
        - Only states with Q-values above machine epsilon get direction arrows
        - This prevents showing arrows for unvisited/unlearned states

    Example:
        >>> qtable = np.random.rand(16, 4)  # 4x4 grid with 4 actions
        >>> max_vals, directions = qtable_directions_map(qtable, map_size=4)
        >>> print(f"Policy grid shape: {directions.shape}")
        >>> print(f"Max Q-values shape: {max_vals.shape}")
    """
    qtable_val_max = qtable.max(axis=1).reshape(map_size, map_size)
    qtable_best_action = np.argmax(qtable, axis=1).reshape(map_size, map_size)
    directions = {0: "←", 1: "↓", 2: "→", 3: "↑"}
    qtable_directions = np.empty(qtable_best_action.flatten().shape, dtype=str)
    eps = np.finfo(float).eps  # Minimum float number on the machine
    for idx, val in enumerate(qtable_best_action.flatten()):
        if qtable_val_max.flatten()[idx] > eps:
            # Assign an arrow only if a minimal Q-value has been learned as best action
            # otherwise since 0 is a direction, it also gets mapped on the tiles where
            # it didn't actually learn anything
            qtable_directions[idx] = directions[val]
    qtable_directions = qtable_directions.reshape(map_size, map_size)
    return qtable_val_max, qtable_directions


def plot_q_values_map(
    qtable: np.ndarray, env: gym.Env, map_size: int, save_path: str = None
) -> None:
    """
    Visualize the learned Q-values and policy as side-by-side plots.

    This function creates a two-panel visualization showing the final environment
    state and the learned policy as a heatmap with directional arrows indicating
    the best action for each state.

    Args:
        qtable (np.ndarray): Q-table of shape (state_size, action_size) containing
            the learned state-action values to visualize.
        env (gym.Env): Gymnasium environment instance for rendering the final frame.
        map_size (int): Size of the square grid environment for reshaping Q-values.
        save_path (str, optional): Path to save the plot. If None, shows the plot.

    Returns:
        None: Displays or saves the plots.

    Note:
        - Left panel shows the last rendered frame of the environment
        - Right panel shows Q-values as a heatmap with policy arrows
        - Color intensity represents Q-value magnitude (Blues colormap)
        - Grid lines are added for better state separation
        - Only states with learned Q-values display directional arrows

    Example:
        >>> plot_q_values_map(qtable=final_qtable, env=environment, map_size=4,
        ...                   save_path="q_values_4x4.png")
        # Saves visualization to file
    """
    qtable_val_max, qtable_directions = qtable_directions_map(qtable, map_size)

    # Plot the last frame
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
    ax[0].imshow(env.render())
    ax[0].axis("off")
    ax[0].set_title("Last frame")

    # Plot the policy
    sns.heatmap(
        qtable_val_max,
        annot=qtable_directions,
        fmt="",
        ax=ax[1],
        cmap=sns.color_palette("Blues", as_cmap=True),
        linewidths=0.7,
        linecolor="black",
        xticklabels=[],
        yticklabels=[],
        annot_kws={"fontsize": "xx-large"},
    ).set(title="Learned Q-values\nArrows represent best action")
    for _, spine in ax[1].spines.items():
        spine.set_visible(True)
        spine.set_linewidth(0.7)
        spine.set_color("black")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_states_actions_distribution(
    states: list[int], actions: list[int], save_path: str = None
) -> None:
    """
    Plot histograms showing the distribution of visited states and taken actions.

    This function creates side-by-side histograms to analyze agent behavior by
    visualizing which states were visited most frequently and which actions
    were taken most often during training.

    Args:
        states (List[int]): List of all state indices visited during training
            across all runs and episodes.
        actions (List[int]): List of all action indices taken during training,
            corresponding to the states list.
        save_path (str, optional): Path to save the plot. If None, shows the plot.

    Returns:
        None: Displays or saves the histograms.

    Note:
        - Left panel shows state visitation frequency with KDE overlay
        - Right panel shows action selection frequency with labeled x-axis
        - Action labels: LEFT(0), DOWN(1), RIGHT(2), UP(3)
        - Helps identify exploration patterns and action preferences
        - Useful for debugging exploration strategies and environment bias

    Example:
        >>> plot_states_actions_distribution(
        ...     states=[0, 1, 2, 0, 1],
        ...     actions=[1, 2, 1, 0, 3],
        ...     save_path="states_actions_dist_4x4.png"
        ... )
        # Saves histograms to file
    """
    labels = {"LEFT": 0, "DOWN": 1, "RIGHT": 2, "UP": 3}

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
    sns.histplot(data=states, ax=ax[0], kde=True)
    ax[0].set_title("States")
    sns.histplot(data=actions, ax=ax[1])
    ax[1].set_xticks(list(labels.values()), labels=labels.keys())
    ax[1].set_title("Actions")
    fig.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_steps_and_rewards(
    rewards_df: pd.DataFrame, steps_df: pd.DataFrame, save_path: str = None
) -> None:
    """
    Plot training progress showing cumulative rewards and average steps over episodes.

    This function creates side-by-side line plots to visualize training progress,
    showing how the agent's performance evolves over episodes in terms of
    cumulative rewards collected and efficiency (steps taken).

    Args:
        rewards_df (pd.DataFrame): DataFrame containing training data with columns:
            - 'Episodes': Episode indices
            - 'cum_rewards': Cumulative rewards over episodes
            - 'map_size': Map size identifier for grouping/coloring
        steps_df (pd.DataFrame): DataFrame containing step data with columns:
            - 'Episodes': Episode indices
            - 'Steps': Average number of steps per episode
            - 'map_size': Map size identifier for grouping/coloring
        save_path (str, optional): Path to save the plot. If None, shows the plot.

    Returns:
        None: Displays or saves the line plots.

    Note:
        - Left panel shows cumulative rewards progression over episodes
        - Right panel shows average steps taken per episode over time
        - Lines are colored by map size for multi-size comparisons
        - Increasing cumulative rewards indicate learning progress
        - Decreasing steps often indicate more efficient policies
        - Useful for hyperparameter tuning and convergence analysis

    Example:
        >>> plot_steps_and_rewards(rewards_dataframe, steps_dataframe,
        ...                        save_path="training_progress.png")
        # Saves learning curves to file
    """
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
    sns.lineplot(
        data=rewards_df, x="Episodes", y="cum_rewards", hue="map_size", ax=ax[0]
    )
    ax[0].set(ylabel="Cumulated rewards")

    sns.lineplot(data=steps_df, x="Episodes", y="Steps", hue="map_size", ax=ax[1])
    ax[1].set(ylabel="Averaged steps number")

    for axi in ax:
        axi.legend(title="map size")
    fig.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()
