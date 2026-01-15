"""Data formatting utilities for TSFM input."""

from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch


@dataclass
class ForecastWindow:
    """
    A single forecasting window with context and targets.

    Represents a lookback window of observations and actions, along with
    the corresponding future observations to predict and known future actions.

    Attributes:
        context_observations: Historical observations, shape (lookback, obs_dim)
        context_actions: Historical actions, shape (lookback, act_dim)
        target_observations: Future observations to predict, shape (horizon, obs_dim)
        future_actions: Known future actions (covariates), shape (horizon, act_dim)
        episode_id: Source episode identifier
        start_idx: Starting index within the episode
    """

    context_observations: np.ndarray  # (lookback, obs_dim)
    context_actions: np.ndarray  # (lookback, act_dim)
    target_observations: np.ndarray  # (horizon, obs_dim)
    future_actions: np.ndarray  # (horizon, act_dim)
    episode_id: int
    start_idx: int

    @property
    def lookback(self) -> int:
        """Length of context window."""
        return len(self.context_observations)

    @property
    def horizon(self) -> int:
        """Length of prediction horizon."""
        return len(self.target_observations)


class TimeSeriesFormatter:
    """
    Format RL trajectories for TSFM inference.

    Converts episode data into sliding windows suitable for time series
    forecasting models. Supports both tensor and DataFrame output formats.

    Attributes:
        lookback: Number of past timesteps for context
        horizon: Number of future timesteps to predict
        obs_dim: Observation dimension
        act_dim: Action dimension

    Example:
        >>> formatter = TimeSeriesFormatter(lookback=50, horizon=10, obs_dim=39, act_dim=28)
        >>> windows = list(formatter.create_windows_from_episode(obs, actions))
        >>> tensors = formatter.to_tensor_format(windows)
    """

    def __init__(
        self,
        lookback: int,
        horizon: int,
        obs_dim: int,
        act_dim: int,
    ):
        """
        Initialize formatter.

        Args:
            lookback: Number of past timesteps for context
            horizon: Number of future timesteps to predict
            obs_dim: Observation dimension
            act_dim: Action dimension
        """
        self.lookback = lookback
        self.horizon = horizon
        self.obs_dim = obs_dim
        self.act_dim = act_dim

    def create_windows_from_episode(
        self,
        observations: np.ndarray,
        actions: np.ndarray,
        episode_id: int = 0,
        stride: int = 1,
    ) -> Iterator[ForecastWindow]:
        """
        Create sliding windows from a single episode.

        Args:
            observations: Shape (steps+1, obs_dim) - includes initial and final obs
            actions: Shape (steps, act_dim)
            episode_id: Episode identifier
            stride: Step between consecutive windows

        Yields:
            ForecastWindow objects

        Note:
            Uses observations[:-1] (states) for forecasting, not the final observation.
            This gives us states aligned with actions where state[t] -> action[t] -> state[t+1].
        """
        # Use states (exclude final observation)
        states = observations[:-1]  # (steps, obs_dim)
        num_steps = len(states)

        # Minimum episode length needed
        min_length = self.lookback + self.horizon
        if num_steps < min_length:
            return  # Episode too short

        # Create sliding windows
        for start in range(0, num_steps - min_length + 1, stride):
            context_end = start + self.lookback
            target_end = context_end + self.horizon

            yield ForecastWindow(
                context_observations=states[start:context_end].copy(),
                context_actions=actions[start:context_end].copy(),
                target_observations=states[context_end:target_end].copy(),
                future_actions=actions[context_end:target_end].copy(),
                episode_id=episode_id,
                start_idx=start,
            )

    def to_tensor_format(
        self,
        windows: List[ForecastWindow],
        include_actions_in_context: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Convert windows to tensor format.

        Args:
            windows: List of ForecastWindow objects
            include_actions_in_context: Whether to concatenate actions to context

        Returns:
            Dictionary with:
                - 'context': Shape (n_windows, lookback, features)
                - 'targets': Shape (n_windows, horizon, obs_dim)
                - 'future_actions': Shape (n_windows, horizon, act_dim)
        """
        contexts = []
        targets = []
        future_actions = []

        for w in windows:
            if include_actions_in_context:
                # Concatenate observations and actions
                ctx = np.concatenate([w.context_observations, w.context_actions], axis=-1)
            else:
                ctx = w.context_observations

            contexts.append(ctx)
            targets.append(w.target_observations)
            future_actions.append(w.future_actions)

        return {
            "context": torch.from_numpy(np.stack(contexts)).float(),
            "targets": torch.from_numpy(np.stack(targets)).float(),
            "future_actions": torch.from_numpy(np.stack(future_actions)).float(),
        }

    def to_univariate_tensor_batch(
        self,
        windows: List[ForecastWindow],
    ) -> Dict[str, torch.Tensor]:
        """
        Convert to batch of univariate time series.

        Each observation dimension is treated as a separate 1D series.
        This format works best with Chronos tensor API.

        Args:
            windows: List of ForecastWindow objects

        Returns:
            Dictionary with:
                - 'context': Shape (n_windows * obs_dim, lookback)
                - 'targets': Shape (n_windows * obs_dim, horizon)
                - 'metadata': List of dicts with window_idx and obs_dim
        """
        contexts = []
        targets = []
        metadata = []

        for w_idx, w in enumerate(windows):
            for dim in range(self.obs_dim):
                # 1D context for this dimension
                contexts.append(w.context_observations[:, dim])
                targets.append(w.target_observations[:, dim])
                metadata.append({"window_idx": w_idx, "obs_dim": dim})

        return {
            "context": torch.from_numpy(np.stack(contexts)).float(),
            "targets": torch.from_numpy(np.stack(targets)).float(),
            "metadata": metadata,
        }

    def to_dataframe_format(
        self,
        windows: List[ForecastWindow],
        include_past_actions: bool = True,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
        """
        Convert windows to DataFrame format for Chronos-2 DataFrame API.

        Creates separate time series for each observation dimension,
        with actions as covariates.

        Args:
            windows: List of ForecastWindow objects
            include_past_actions: Whether to include actions as past covariates

        Returns:
            Tuple of (context_df, future_df, ground_truth_targets)
                - context_df: Historical data with target and covariate columns
                - future_df: Future covariate values (actions)
                - ground_truth_targets: Shape (n_windows, horizon, obs_dim)
        """
        context_rows = []
        future_rows = []
        ground_truth = []

        for window_idx, w in enumerate(windows):
            ground_truth.append(w.target_observations)

            # For each observation dimension, create a separate series
            for dim in range(self.obs_dim):
                series_id = f"w{window_idx}_d{dim}"

                # Context data (historical)
                for t in range(self.lookback):
                    row = {
                        "timestamp": t,
                        "window_id": window_idx,
                        "obs_dim": dim,
                        "id": series_id,
                        "target": w.context_observations[t, dim],
                    }

                    # Add past actions as covariates
                    if include_past_actions:
                        for a in range(self.act_dim):
                            row[f"action_{a}"] = w.context_actions[t, a]

                    context_rows.append(row)

                # Future data (for covariates only)
                for t in range(self.horizon):
                    future_row = {
                        "timestamp": self.lookback + t,
                        "window_id": window_idx,
                        "obs_dim": dim,
                        "id": series_id,
                    }

                    # Add future actions as known covariates
                    for a in range(self.act_dim):
                        future_row[f"action_{a}"] = w.future_actions[t, a]

                    future_rows.append(future_row)

        context_df = pd.DataFrame(context_rows)
        future_df = pd.DataFrame(future_rows)
        ground_truth_array = np.stack(ground_truth)

        return context_df, future_df, ground_truth_array

    def windows_to_arrays(
        self,
        windows: List[ForecastWindow],
    ) -> Dict[str, np.ndarray]:
        """
        Convert windows to simple numpy arrays.

        Args:
            windows: List of ForecastWindow objects

        Returns:
            Dictionary with:
                - 'context_obs': Shape (n_windows, lookback, obs_dim)
                - 'context_actions': Shape (n_windows, lookback, act_dim)
                - 'target_obs': Shape (n_windows, horizon, obs_dim)
                - 'future_actions': Shape (n_windows, horizon, act_dim)
        """
        return {
            "context_obs": np.stack([w.context_observations for w in windows]),
            "context_actions": np.stack([w.context_actions for w in windows]),
            "target_obs": np.stack([w.target_observations for w in windows]),
            "future_actions": np.stack([w.future_actions for w in windows]),
        }


def prepare_evaluation_data(
    loader,  # MinariDataLoader
    formatter: TimeSeriesFormatter,
    max_episodes: Optional[int] = None,
    windows_per_episode: int = 20,
    min_episode_length: int = 0,
    seed: Optional[int] = None,
) -> List[ForecastWindow]:
    """
    Prepare evaluation windows from a Minari dataset.

    Iterates through episodes and creates sliding windows for evaluation.
    Optionally subsamples windows per episode to limit total count.

    Args:
        loader: MinariDataLoader instance
        formatter: TimeSeriesFormatter instance
        max_episodes: Maximum episodes to process (None for all)
        windows_per_episode: Maximum windows to keep per episode
        min_episode_length: Minimum episode length to include
        seed: Random seed for window subsampling

    Returns:
        List of ForecastWindow objects

    Example:
        >>> loader = MinariDataLoader("D4RL/door/human-v2")
        >>> formatter = TimeSeriesFormatter(50, 10, loader.obs_dim, loader.act_dim)
        >>> windows = prepare_evaluation_data(loader, formatter, max_episodes=100)
    """
    if seed is not None:
        np.random.seed(seed)

    all_windows = []
    min_length = formatter.lookback + formatter.horizon

    for episode in loader.iterate_episodes(
        max_episodes=max_episodes,
        min_length=max(min_length, min_episode_length),
    ):
        episode_windows = list(
            formatter.create_windows_from_episode(
                observations=episode.observations,
                actions=episode.actions,
                episode_id=episode.id,
            )
        )

        # Subsample if too many windows
        if len(episode_windows) > windows_per_episode:
            indices = np.linspace(0, len(episode_windows) - 1, windows_per_episode, dtype=int)
            episode_windows = [episode_windows[i] for i in indices]

        all_windows.extend(episode_windows)

    return all_windows


def create_single_window(
    observations: np.ndarray,
    actions: np.ndarray,
    lookback: int,
    horizon: int,
    start_idx: int = 0,
) -> ForecastWindow:
    """
    Create a single forecast window from arrays.

    Convenience function for creating a window without using the full
    TimeSeriesFormatter class.

    Args:
        observations: Shape (steps+1, obs_dim)
        actions: Shape (steps, act_dim)
        lookback: Context length
        horizon: Prediction horizon
        start_idx: Starting index in the arrays

    Returns:
        ForecastWindow object
    """
    states = observations[:-1]  # Exclude final observation
    context_end = start_idx + lookback
    target_end = context_end + horizon

    return ForecastWindow(
        context_observations=states[start_idx:context_end].copy(),
        context_actions=actions[start_idx:context_end].copy(),
        target_observations=states[context_end:target_end].copy(),
        future_actions=actions[context_end:target_end].copy(),
        episode_id=0,
        start_idx=start_idx,
    )
