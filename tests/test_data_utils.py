"""Tests for data loading and formatting utilities."""

import numpy as np
import pytest

from tsmbrl.data.formatters import (
    ForecastWindow,
    TimeSeriesFormatter,
    create_single_window,
)
from tsmbrl.data.minari_loader import Episode


class TestEpisode:
    """Tests for Episode dataclass."""

    def test_episode_properties(self):
        """Test Episode dataclass properties."""
        n_steps = 100
        obs_dim = 10
        act_dim = 4

        obs = np.random.randn(n_steps + 1, obs_dim).astype(np.float32)
        actions = np.random.randn(n_steps, act_dim).astype(np.float32)
        rewards = np.random.randn(n_steps).astype(np.float32)

        episode = Episode(
            id=0,
            observations=obs,
            actions=actions,
            rewards=rewards,
            terminations=np.zeros(n_steps, dtype=bool),
            truncations=np.zeros(n_steps, dtype=bool),
        )

        assert episode.states.shape == (n_steps, obs_dim)
        assert episode.next_states.shape == (n_steps, obs_dim)
        assert episode.num_steps == n_steps
        np.testing.assert_array_equal(episode.states, obs[:-1])
        np.testing.assert_array_equal(episode.next_states, obs[1:])

    def test_episode_dones(self):
        """Test done flags computation."""
        n_steps = 10
        obs = np.random.randn(n_steps + 1, 4).astype(np.float32)
        actions = np.random.randn(n_steps, 2).astype(np.float32)
        rewards = np.random.randn(n_steps).astype(np.float32)

        terminations = np.array([False] * 9 + [True])
        truncations = np.array([False] * 10)

        episode = Episode(
            id=0,
            observations=obs,
            actions=actions,
            rewards=rewards,
            terminations=terminations,
            truncations=truncations,
        )

        expected_dones = terminations | truncations
        np.testing.assert_array_equal(episode.dones, expected_dones)


class TestTimeSeriesFormatter:
    """Tests for TimeSeriesFormatter."""

    def test_create_windows(self):
        """Test window creation from episode."""
        lookback = 10
        horizon = 5
        obs_dim = 4
        act_dim = 2
        n_steps = 50

        formatter = TimeSeriesFormatter(
            lookback=lookback,
            horizon=horizon,
            obs_dim=obs_dim,
            act_dim=act_dim,
        )

        obs = np.random.randn(n_steps + 1, obs_dim)  # +1 for final obs
        actions = np.random.randn(n_steps, act_dim)

        windows = list(formatter.create_windows_from_episode(obs, actions))

        # Expected number of windows: n_steps - lookback - horizon + 1
        expected_count = n_steps - lookback - horizon + 1
        assert len(windows) == expected_count

        # Check first window shapes
        w0 = windows[0]
        assert w0.context_observations.shape == (lookback, obs_dim)
        assert w0.context_actions.shape == (lookback, act_dim)
        assert w0.target_observations.shape == (horizon, obs_dim)
        assert w0.future_actions.shape == (horizon, act_dim)

    def test_minimum_length_episode(self):
        """Test that short episodes produce no windows."""
        formatter = TimeSeriesFormatter(
            lookback=20,
            horizon=10,
            obs_dim=4,
            act_dim=2,
        )

        # Episode too short: only 25 steps, need 30 minimum
        obs = np.random.randn(26, 4)  # 25 steps + final obs
        actions = np.random.randn(25, 2)

        windows = list(formatter.create_windows_from_episode(obs, actions))
        assert len(windows) == 0

    def test_exact_minimum_length(self):
        """Test episode with exact minimum length."""
        lookback = 10
        horizon = 5
        n_steps = lookback + horizon  # Exactly minimum

        formatter = TimeSeriesFormatter(
            lookback=lookback,
            horizon=horizon,
            obs_dim=2,
            act_dim=1,
        )

        obs = np.random.randn(n_steps + 1, 2)
        actions = np.random.randn(n_steps, 1)

        windows = list(formatter.create_windows_from_episode(obs, actions))
        assert len(windows) == 1

    def test_tensor_format(self):
        """Test conversion to tensor format."""
        formatter = TimeSeriesFormatter(
            lookback=10,
            horizon=5,
            obs_dim=4,
            act_dim=2,
        )

        windows = [
            ForecastWindow(
                context_observations=np.random.randn(10, 4),
                context_actions=np.random.randn(10, 2),
                target_observations=np.random.randn(5, 4),
                future_actions=np.random.randn(5, 2),
                episode_id=0,
                start_idx=0,
            )
            for _ in range(5)
        ]

        tensors = formatter.to_tensor_format(windows)

        assert tensors["context"].shape == (5, 10, 4)
        assert tensors["targets"].shape == (5, 5, 4)
        assert tensors["future_actions"].shape == (5, 5, 2)

    def test_tensor_format_with_actions(self):
        """Test tensor format with actions concatenated."""
        formatter = TimeSeriesFormatter(
            lookback=10,
            horizon=5,
            obs_dim=4,
            act_dim=2,
        )

        windows = [
            ForecastWindow(
                context_observations=np.random.randn(10, 4),
                context_actions=np.random.randn(10, 2),
                target_observations=np.random.randn(5, 4),
                future_actions=np.random.randn(5, 2),
                episode_id=0,
                start_idx=i,
            )
            for i in range(3)
        ]

        tensors = formatter.to_tensor_format(windows, include_actions_in_context=True)

        # Context should have obs + actions = 4 + 2 = 6 features
        assert tensors["context"].shape == (3, 10, 6)

    def test_univariate_batch(self):
        """Test univariate batch conversion."""
        formatter = TimeSeriesFormatter(
            lookback=10,
            horizon=5,
            obs_dim=4,
            act_dim=2,
        )

        windows = [
            ForecastWindow(
                context_observations=np.random.randn(10, 4),
                context_actions=np.random.randn(10, 2),
                target_observations=np.random.randn(5, 4),
                future_actions=np.random.randn(5, 2),
                episode_id=0,
                start_idx=0,
            )
            for _ in range(3)
        ]

        batch = formatter.to_univariate_tensor_batch(windows)

        # Should have 3 windows * 4 dims = 12 univariate series
        assert batch["context"].shape == (12, 10)
        assert batch["targets"].shape == (12, 5)
        assert len(batch["metadata"]) == 12


class TestForecastWindow:
    """Tests for ForecastWindow dataclass."""

    def test_window_properties(self):
        """Test window properties."""
        window = ForecastWindow(
            context_observations=np.random.randn(20, 4),
            context_actions=np.random.randn(20, 2),
            target_observations=np.random.randn(10, 4),
            future_actions=np.random.randn(10, 2),
            episode_id=5,
            start_idx=100,
        )

        assert window.lookback == 20
        assert window.horizon == 10
        assert window.episode_id == 5
        assert window.start_idx == 100


class TestCreateSingleWindow:
    """Tests for create_single_window helper."""

    def test_create_single_window(self):
        """Test single window creation."""
        obs = np.random.randn(101, 4)  # 100 steps + final obs
        actions = np.random.randn(100, 2)

        window = create_single_window(
            observations=obs,
            actions=actions,
            lookback=20,
            horizon=10,
            start_idx=50,
        )

        assert window.context_observations.shape == (20, 4)
        assert window.context_actions.shape == (20, 2)
        assert window.target_observations.shape == (10, 4)
        assert window.future_actions.shape == (10, 2)
        assert window.start_idx == 50
