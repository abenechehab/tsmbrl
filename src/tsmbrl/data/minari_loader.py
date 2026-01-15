"""Minari dataset loading utilities for TSMBRL."""

from dataclasses import dataclass
from typing import Any, Dict, Iterator, Optional

import minari
import numpy as np
from minari import MinariDataset


@dataclass
class Episode:
    """
    Processed episode data from a Minari dataset.

    Contains observations, actions, rewards, and episode metadata
    in a format suitable for TSFM evaluation.

    Attributes:
        id: Episode identifier
        observations: Array of shape (steps+1, obs_dim) - includes initial and final obs
        actions: Array of shape (steps, act_dim)
        rewards: Array of shape (steps,)
        terminations: Boolean array of shape (steps,)
        truncations: Boolean array of shape (steps,)

    Note:
        observations has one more element than actions because it includes
        both the initial observation (from reset) and the final observation
        (after the last action).
    """

    id: int
    observations: np.ndarray  # (steps+1, obs_dim)
    actions: np.ndarray  # (steps, act_dim)
    rewards: np.ndarray  # (steps,)
    terminations: np.ndarray  # (steps,)
    truncations: np.ndarray  # (steps,)

    @property
    def states(self) -> np.ndarray:
        """
        States s_t (excludes final observation).

        Returns:
            Array of shape (steps, obs_dim)
        """
        return self.observations[:-1]

    @property
    def next_states(self) -> np.ndarray:
        """
        Next states s_{t+1} (excludes initial observation).

        Returns:
            Array of shape (steps, obs_dim)
        """
        return self.observations[1:]

    @property
    def num_steps(self) -> int:
        """Number of timesteps (actions) in the episode."""
        return len(self.actions)

    @property
    def dones(self) -> np.ndarray:
        """Episode done flags (termination OR truncation)."""
        return self.terminations | self.truncations


class MinariDataLoader:
    """
    Load and manage Minari offline RL datasets.

    Handles downloading, loading, and iterating through episodes from
    Minari datasets. Automatically flattens Dict observation and action
    spaces for compatibility with TSFMs.

    Attributes:
        dataset_id: Minari dataset identifier
        dataset: Loaded MinariDataset instance

    Example:
        >>> loader = MinariDataLoader("D4RL/door/human-v2", download=True)
        >>> print(f"Obs dim: {loader.obs_dim}, Act dim: {loader.act_dim}")
        >>> for episode in loader.iterate_episodes(max_episodes=10):
        ...     print(f"Episode {episode.id}: {episode.num_steps} steps")
    """

    def __init__(self, dataset_id: str, download: bool = True):
        """
        Initialize loader for a Minari dataset.

        Args:
            dataset_id: Minari dataset identifier (e.g., "D4RL/door/human-v2")
            download: Whether to download the dataset if not present locally
        """
        self.dataset_id = dataset_id
        self.dataset: MinariDataset = minari.load_dataset(dataset_id, download=download)
        self._metadata = self._extract_metadata()

    def _extract_metadata(self) -> Dict[str, Any]:
        """Extract and compute dataset metadata."""
        spec = self.dataset.spec
        return {
            "dataset_id": self.dataset_id,
            "total_episodes": self.dataset.total_episodes,
            "total_steps": self.dataset.total_steps,
            "observation_space": spec.observation_space,
            "action_space": spec.action_space,
            "obs_dim": self._get_space_dim(spec.observation_space),
            "act_dim": self._get_space_dim(spec.action_space),
        }

    def _get_space_dim(self, space: Any) -> int:
        """
        Get flattened dimension of a Gymnasium space.

        Args:
            space: Gymnasium space object

        Returns:
            Flattened dimension

        Raises:
            ValueError: If space type is not supported
        """
        from gymnasium.spaces import Box, Dict as DictSpace, Discrete, MultiBinary, MultiDiscrete

        if isinstance(space, Box):
            return int(np.prod(space.shape))
        elif isinstance(space, Discrete):
            return 1
        elif isinstance(space, MultiBinary):
            return space.n
        elif isinstance(space, MultiDiscrete):
            return len(space.nvec)
        elif isinstance(space, DictSpace):
            return sum(self._get_space_dim(s) for s in space.spaces.values())
        else:
            raise ValueError(f"Unsupported space type: {type(space)}")

    def _flatten_array(self, arr: Any, expected_len: int) -> np.ndarray:
        """
        Flatten observation or action array.

        Handles Dict spaces, nested arrays, and ensures proper shape.

        Args:
            arr: Input array (can be dict, nested array, or regular array)
            expected_len: Expected first dimension (number of timesteps)

        Returns:
            Flattened 2D array of shape (expected_len, dim)
        """
        if isinstance(arr, dict):
            # Concatenate all values along last axis
            flattened_parts = []
            for key in sorted(arr.keys()):  # Sort for consistent ordering
                part = self._flatten_array(arr[key], expected_len)
                flattened_parts.append(part)
            return np.concatenate(flattened_parts, axis=-1)

        arr = np.asarray(arr)

        # Ensure 2D
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        elif arr.ndim > 2:
            # Flatten all dimensions except first (timesteps)
            arr = arr.reshape(arr.shape[0], -1)

        return arr.astype(np.float32)

    def iterate_episodes(
        self,
        max_episodes: Optional[int] = None,
        min_length: int = 0,
    ) -> Iterator[Episode]:
        """
        Iterate through episodes in the dataset.

        Args:
            max_episodes: Maximum number of episodes to yield (None for all)
            min_length: Minimum episode length to include (skip shorter episodes)

        Yields:
            Episode objects with processed data
        """
        count = 0

        for ep_data in self.dataset.iterate_episodes():
            if max_episodes is not None and count >= max_episodes:
                break

            # Skip short episodes
            if ep_data.total_timesteps < min_length:
                continue

            # Flatten observations: shape (steps+1, obs_dim)
            obs = self._flatten_array(ep_data.observations, ep_data.total_timesteps + 1)

            # Flatten actions: shape (steps, act_dim)
            actions = self._flatten_array(ep_data.actions, ep_data.total_timesteps)

            yield Episode(
                id=ep_data.id,
                observations=obs,
                actions=actions,
                rewards=ep_data.rewards.astype(np.float32),
                terminations=ep_data.terminations,
                truncations=ep_data.truncations,
            )
            count += 1

    def sample_episodes(self, n_episodes: int, seed: Optional[int] = None) -> list:
        """
        Sample random episodes from the dataset.

        Args:
            n_episodes: Number of episodes to sample
            seed: Random seed for reproducibility

        Returns:
            List of Episode objects
        """
        if seed is not None:
            self.dataset.set_seed(seed)

        episodes = []
        for ep_data in self.dataset.sample_episodes(n_episodes=n_episodes):
            obs = self._flatten_array(ep_data.observations, ep_data.total_timesteps + 1)
            actions = self._flatten_array(ep_data.actions, ep_data.total_timesteps)

            episodes.append(
                Episode(
                    id=ep_data.id,
                    observations=obs,
                    actions=actions,
                    rewards=ep_data.rewards.astype(np.float32),
                    terminations=ep_data.terminations,
                    truncations=ep_data.truncations,
                )
            )

        return episodes

    @property
    def metadata(self) -> Dict[str, Any]:
        """Dataset metadata dictionary."""
        return self._metadata.copy()

    @property
    def obs_dim(self) -> int:
        """Flattened observation dimension."""
        return self._metadata["obs_dim"]

    @property
    def act_dim(self) -> int:
        """Flattened action dimension."""
        return self._metadata["act_dim"]

    @property
    def total_episodes(self) -> int:
        """Total number of episodes in the dataset."""
        return self._metadata["total_episodes"]

    @property
    def total_steps(self) -> int:
        """Total number of steps across all episodes."""
        return self._metadata["total_steps"]

    def __repr__(self) -> str:
        return (
            f"MinariDataLoader('{self.dataset_id}', "
            f"episodes={self.total_episodes}, "
            f"obs_dim={self.obs_dim}, "
            f"act_dim={self.act_dim})"
        )
