"""Tests for evaluation metrics."""

import numpy as np
import pytest

from tsmbrl.metrics.forecasting_metrics import (
    mae,
    mse,
    multi_step_metrics,
    per_dimension_metrics,
    rmse,
    single_step_metrics,
    normalized_mse,
)
from tsmbrl.metrics.probabilistic_metrics import (
    calibration_error,
    compute_all_probabilistic_metrics,
    coverage,
    crps_from_quantiles,
    interval_width,
    winkler_score,
)


class TestForecastingMetrics:
    """Tests for forecasting metrics."""

    def test_mse_perfect(self):
        """Test MSE with perfect predictions."""
        pred = np.array([1.0, 2.0, 3.0])
        target = np.array([1.0, 2.0, 3.0])

        assert mse(pred, target) == 0.0

    def test_mse_error(self):
        """Test MSE with known error."""
        pred = np.array([1.0, 2.0, 3.0])
        target = np.array([2.0, 3.0, 4.0])

        # Error is always 1, so MSE = mean([1, 1, 1]) = 1.0
        assert mse(pred, target) == 1.0

    def test_mae(self):
        """Test MAE computation."""
        pred = np.array([1.0, 2.0, 3.0])
        target = np.array([2.0, 1.0, 5.0])

        # Errors: [1, 1, 2], MAE = 4/3
        expected = np.mean([1, 1, 2])
        assert mae(pred, target) == pytest.approx(expected)

    def test_rmse(self):
        """Test RMSE computation."""
        pred = np.array([1.0, 2.0, 3.0])
        target = np.array([2.0, 3.0, 4.0])

        expected = np.sqrt(1.0)  # MSE = 1.0
        assert rmse(pred, target) == pytest.approx(expected)

    def test_single_step_metrics(self):
        """Test single-step metrics."""
        pred = np.random.randn(100)
        target = pred + 0.1  # Small error

        metrics = single_step_metrics(pred, target)

        assert "mse" in metrics
        assert "mae" in metrics
        assert "rmse" in metrics
        assert metrics["mse"] == pytest.approx(0.01, abs=0.001)

    def test_multi_step_metrics(self):
        """Test multi-step metric computation."""
        # (batch=10, horizon=5, features=2)
        pred = np.random.randn(10, 5, 2)
        target = pred + 0.1

        metrics = multi_step_metrics(pred, target)

        assert "mse" in metrics
        assert "mae" in metrics
        assert "rmse" in metrics
        assert "mse_per_step" in metrics
        assert len(metrics["mse_per_step"]) == 5
        assert metrics["horizon"] == 5

        # All errors should be approximately 0.01
        assert metrics["mse"] == pytest.approx(0.01, abs=0.001)

    def test_multi_step_metrics_2d(self):
        """Test multi-step metrics with 2D input."""
        pred = np.random.randn(10, 5)  # No feature dimension
        target = pred + 0.1

        metrics = multi_step_metrics(pred, target)

        assert metrics["horizon"] == 5
        assert len(metrics["mse_per_step"]) == 5

    def test_per_dimension_metrics(self):
        """Test per-dimension metrics."""
        pred = np.random.randn(10, 5, 3)  # 3 features
        target = pred.copy()
        # Add different errors to each dimension
        target[:, :, 0] += 0.1
        target[:, :, 1] += 0.2
        target[:, :, 2] += 0.3

        metrics = per_dimension_metrics(pred, target)

        assert "mse_per_dim" in metrics
        assert len(metrics["mse_per_dim"]) == 3
        assert metrics["n_features"] == 3

        # Errors should increase with dimension
        assert metrics["mse_per_dim"][0] < metrics["mse_per_dim"][1]
        assert metrics["mse_per_dim"][1] < metrics["mse_per_dim"][2]

    def test_normalized_mse(self):
        """Test normalized MSE."""
        pred = np.array([0.0, 0.0, 0.0])
        target = np.array([1.0, 2.0, 3.0])

        # MSE = mean([1, 4, 9]) = 14/3
        # Var = var([1, 2, 3]) = 2/3
        # NMSE = (14/3) / (2/3) = 7
        nmse = normalized_mse(pred, target)
        assert nmse == pytest.approx(7.0, abs=0.01)


class TestProbabilisticMetrics:
    """Tests for probabilistic metrics."""

    def test_crps(self):
        """Test CRPS computation."""
        # Simple case: predictions match targets
        quantiles = np.array([[1.0, 2.0, 3.0]])  # (1, 3)
        quantile_levels = [0.1, 0.5, 0.9]
        targets = np.array([2.0])  # Matches median

        crps = crps_from_quantiles(quantiles, quantile_levels, targets)
        assert crps >= 0  # CRPS is non-negative

    def test_coverage_all_within(self):
        """Test coverage when all targets are within interval."""
        lower = np.array([0.0, 0.0, 0.0])
        upper = np.array([1.0, 1.0, 1.0])
        targets = np.array([0.5, 0.5, 0.5])

        assert coverage(lower, upper, targets) == 1.0

    def test_coverage_none_within(self):
        """Test coverage when no targets are within interval."""
        lower = np.array([0.0, 0.0, 0.0])
        upper = np.array([1.0, 1.0, 1.0])
        targets = np.array([2.0, 2.0, 2.0])

        assert coverage(lower, upper, targets) == 0.0

    def test_coverage_partial(self):
        """Test coverage with some targets within interval."""
        lower = np.array([0.0, 0.0, 0.0])
        upper = np.array([1.0, 1.0, 1.0])
        targets = np.array([0.5, 2.0, 0.5])

        assert coverage(lower, upper, targets) == pytest.approx(2 / 3)

    def test_interval_width(self):
        """Test interval width computation."""
        lower = np.array([0.0, 1.0, 2.0])
        upper = np.array([1.0, 2.0, 3.0])

        # All intervals have width 1
        assert interval_width(lower, upper) == 1.0

    def test_winkler_score_within(self):
        """Test Winkler score when targets are within interval."""
        lower = np.array([0.0])
        upper = np.array([1.0])
        targets = np.array([0.5])

        # Score should equal interval width when target is within
        score = winkler_score(lower, upper, targets, alpha=0.2)
        assert score == 1.0

    def test_winkler_score_outside(self):
        """Test Winkler score when target is outside interval."""
        lower = np.array([0.0])
        upper = np.array([1.0])
        targets = np.array([2.0])  # Above upper bound

        # Score = width + penalty
        # Penalty = (2/alpha) * (target - upper) = (2/0.2) * 1 = 10
        score = winkler_score(lower, upper, targets, alpha=0.2)
        assert score == 1.0 + 10.0

    def test_calibration_error(self):
        """Test calibration error computation."""
        n = 1000
        np.random.seed(42)
        targets = np.random.randn(n)

        # Well-calibrated quantiles
        quantile_levels = [0.1, 0.5, 0.9]
        quantiles = np.percentile(targets, [10, 50, 90]).reshape(1, -1)
        quantiles = np.repeat(quantiles, n, axis=0)

        cal = calibration_error(quantiles, quantile_levels, targets)

        # Should be close to 0 for all levels
        for level in quantile_levels:
            assert abs(cal[level]) < 0.05

    def test_compute_all_probabilistic_metrics(self):
        """Test comprehensive probabilistic metrics."""
        n_samples = 100
        horizon = 5
        n_quantiles = 3

        quantiles = np.random.randn(n_samples, horizon, n_quantiles)
        # Sort to ensure proper ordering
        quantiles = np.sort(quantiles, axis=-1)
        quantile_levels = [0.1, 0.5, 0.9]
        targets = np.random.randn(n_samples, horizon)

        metrics = compute_all_probabilistic_metrics(
            quantiles, quantile_levels, targets
        )

        assert "crps" in metrics
        assert "calibration" in metrics
        assert "mean_abs_calibration_error" in metrics
        assert "coverage_80" in metrics
        assert "interval_width_80" in metrics
        assert "winkler_80" in metrics
