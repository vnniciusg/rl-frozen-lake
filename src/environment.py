"""
Environment setup utilities for reinforcement learning on FrozenLake.

This module provides functionality to create and configure FrozenLake environments
with appropriate wrappers for training and evaluation of reinforcement learning agents.
The environments are configured with video recording and episode statistics tracking
capabilities.

Functions:
    setup_environment: Creates a configured FrozenLake environment with wrappers.

Dependencies:
    - gymnasium: For the base FrozenLake environment
    - src.params: For configuration parameters

Example:
    >>> from src.environment import setup_environment
    >>> env = setup_environment()
    >>> observation, info = env.reset()
    >>> action = env.action_space.sample()
    >>> observation, reward, terminated, truncated, info = env.step(action)

Author: vnniciusg
"""

from typing import Literal

import gymnasium as gym
from gymnasium.envs.toy_text.frozen_lake import generate_random_map
from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo

from src.params import params


def setup_environment(
    env_name: str = "FrozenLake-v1",
    num_eval_episodes: int = params.total_episodes,
    render_mode: Literal["rgb_array", "human", "ansi"] = "rgb_array",
    training_period: int = 1000,
    video_folder: str = "frozen-lake-agent",
    name_prefix: str = "eval",
):
    """
    Create and configure a FrozenLake environment with recording capabilities.

    This function sets up a FrozenLake environment with the following features:
    - Random map generation based on configuration parameters
    - Video recording of episodes at specified intervals
    - Episode statistics tracking for performance monitoring
    - Configurable rendering modes for different use cases

    Args:
        env_name (str, optional): Name of the Gymnasium environment to create.
            Defaults to "FrozenLake-v1".
        num_eval_episodes (int, optional): Buffer length for episode statistics.
            Defaults to params.total_episodes.
        render_mode (Literal["rgb_array", "human", "ansi"], optional):
            Rendering mode for the environment:
            - "rgb_array": Returns RGB array for video recording
            - "human": Renders to screen for human observation
            - "ansi": Text-based rendering for terminal output
            Defaults to "rgb_array".
        training_period (int, optional): Interval between recorded episodes.
            Videos are recorded every `training_period` episodes. Defaults to 250.
        video_folder (str, optional): Directory path where recorded videos
            will be saved. Defaults to "frozen-lake-agent".
        name_prefix (str, optional): Prefix for video filenames. The recorded videos
            will be named as "{name_prefix}-episode-{episode_number}.mp4".
            Defaults to "eval".

    Returns:
        gymnasium.Env: A wrapped FrozenLake environment with the following wrappers:
            - RecordVideo: Records videos at specified intervals
            - RecordEpisodeStatistics: Tracks episode rewards, lengths, and times

    Note:
        The environment configuration (map size, slipperiness, frozen tile probability,
        and random seed) is automatically loaded from the global params object.

    Example:
        >>> env = setup_environment(
        ...     render_mode="human",
        ...     training_period=100,
        ...     video_folder="my_videos"
        ... )
        >>> obs, info = env.reset()
        >>> # Environment is ready for training/evaluation
    """

    _env = gym.make(
        env_name,
        is_slippery=params.is_slippery,
        render_mode=render_mode,
        desc=generate_random_map(
            size=params.map_size, p=params.proba_frozen, seed=params.seed
        ),
    )

    _env = RecordVideo(
        _env,
        video_folder=video_folder,
        name_prefix=name_prefix,
        episode_trigger=lambda x: x % training_period == 0,
    )

    return RecordEpisodeStatistics(env=_env, buffer_length=num_eval_episodes)
