"""Probabilistic forecasting metrics for TSMBRL."""

from typing import Any, Dict, List, Optional

import numpy as np


def crps_from_quantiles(
    quantiles: np.ndarray,
    quantile_levels: List[float],
    targets: np.ndarray,
) -> float:
    """
    Compute CRPS approximation from quantile predictions.

    Uses the quantile-weighted pinball loss approximation.

    Args:
        quantiles: Quantile predictions
            Shape: (n_samples, horizon, n_quantiles) or
                   (n_samples, horizon, n_quantiles, features)
        quantile_levels: List of quantile levels (e.g., [0.1, 0.5, 0.9])
        targets: Ground truth values
            Shape: (n_samples, horizon) or (n_samples, horizon, features)

    Returns:
        Average CRPS score (lower is better)

    Note:
        This is an approximation of CRPS using quantile predictions.
        For exact CRPS, you would need the full predictive distribution.
    """
    # Flatten spatial dimensions
    if quantiles.ndim == 4:
        n_samples, horizon, n_q, features = quantiles.shape
        quantiles = quantiles.reshape(-1, n_q)
        targets = targets.reshape(-1)
    elif quantiles.ndim == 3:
        quantiles = quantiles.reshape(-1, quantiles.shape[-1])
        targets = targets.flatten()

    n_points = len(targets)
    n_q = len(quantile_levels)

    # Compute pinball loss for each quantile
    total_loss = 0.0
    for i, tau in enumerate(quantile_levels):
        pred = quantiles[:, i]
        error = targets - pred

        # Pinball loss: tau * max(error, 0) + (1-tau) * max(-error, 0)
        # Equivalently: error * (tau - (error < 0))
        loss = np.where(error >= 0, tau * error, (tau - 1) * error)
        total_loss += np.mean(np.abs(loss))

    # Average over quantiles and scale
    crps = 2 * total_loss / n_q
    return float(crps)


def calibration_error(
    quantiles: np.ndarray,
    quantile_levels: List[float],
    targets: np.ndarray,
) -> Dict[float, float]:
    """
    Compute calibration error for each quantile level.

    Well-calibrated predictions should have:
    P(Y < predicted_quantile_tau) = tau

    Args:
        quantiles: Quantile predictions
        quantile_levels: List of quantile levels
        targets: Ground truth values

    Returns:
        Dictionary mapping quantile level to calibration error
        (observed coverage - expected coverage)
    """
    # Flatten
    if quantiles.ndim == 4:
        quantiles = quantiles.reshape(-1, quantiles.shape[2])
        targets = targets.flatten()
    elif quantiles.ndim == 3:
        quantiles = quantiles.reshape(-1, quantiles.shape[-1])
        targets = targets.flatten()

    calibration = {}
    for i, tau in enumerate(quantile_levels):
        pred_quantile = quantiles[:, i]
        # Empirical coverage: fraction of targets below predicted quantile
        observed = np.mean(targets < pred_quantile)
        calibration[tau] = float(observed - tau)  # Error: observed - expected

    return calibration


def coverage(
    lower_quantile: np.ndarray,
    upper_quantile: np.ndarray,
    targets: np.ndarray,
) -> float:
    """
    Compute empirical coverage of prediction intervals.

    Args:
        lower_quantile: Lower bound predictions (e.g., 0.1 quantile)
        upper_quantile: Upper bound predictions (e.g., 0.9 quantile)
        targets: Ground truth values

    Returns:
        Fraction of targets within [lower, upper]
    """
    lower = lower_quantile.flatten()
    upper = upper_quantile.flatten()
    targets = targets.flatten()

    within = (targets >= lower) & (targets <= upper)
    return float(np.mean(within))


def interval_width(
    lower_quantile: np.ndarray,
    upper_quantile: np.ndarray,
) -> float:
    """
    Compute average prediction interval width.

    Narrower intervals with good coverage are better.

    Args:
        lower_quantile: Lower bound predictions
        upper_quantile: Upper bound predictions

    Returns:
        Average interval width
    """
    width = upper_quantile - lower_quantile
    return float(np.mean(width))


def winkler_score(
    lower_quantile: np.ndarray,
    upper_quantile: np.ndarray,
    targets: np.ndarray,
    alpha: float = 0.2,
) -> float:
    """
    Compute Winkler score for prediction intervals.

    Rewards narrow intervals that contain the target.
    Penalizes when target falls outside interval.

    Args:
        lower_quantile: Lower bound (should be alpha/2 quantile)
        upper_quantile: Upper bound (should be 1-alpha/2 quantile)
        targets: Ground truth values
        alpha: Nominal miscoverage rate (e.g., 0.2 for 80% interval)

    Returns:
        Average Winkler score (lower is better)
    """
    lower = lower_quantile.flatten()
    upper = upper_quantile.flatten()
    targets = targets.flatten()

    width = upper - lower

    # Penalty for targets below lower bound
    below = targets < lower
    penalty_below = (2 / alpha) * (lower - targets) * below

    # Penalty for targets above upper bound
    above = targets > upper
    penalty_above = (2 / alpha) * (targets - upper) * above

    score = width + penalty_below + penalty_above
    return float(np.mean(score))


def mean_absolute_calibration_error(
    calibration: Dict[float, float],
) -> float:
    """
    Compute mean absolute calibration error.

    Args:
        calibration: Dictionary from calibration_error()

    Returns:
        Mean of absolute calibration errors
    """
    return float(np.mean(np.abs(list(calibration.values()))))


def compute_all_probabilistic_metrics(
    quantiles: np.ndarray,
    quantile_levels: List[float],
    targets: np.ndarray,
) -> Dict[str, Any]:
    """
    Compute all probabilistic metrics.

    Args:
        quantiles: Quantile predictions
            Shape: (n_samples, horizon, n_quantiles) or
                   (n_samples, horizon, n_quantiles, features)
        quantile_levels: List of quantile levels (e.g., [0.1, 0.5, 0.9])
        targets: Ground truth values

    Returns:
        Dictionary with all probabilistic metrics
    """
    results: Dict[str, Any] = {
        "crps": crps_from_quantiles(quantiles, quantile_levels, targets),
        "calibration": calibration_error(quantiles, quantile_levels, targets),
    }

    # Compute mean absolute calibration error
    results["mean_abs_calibration_error"] = mean_absolute_calibration_error(
        results["calibration"]
    )

    # Find lower/upper quantile indices for interval metrics
    # Typically use 0.1 and 0.9 for 80% interval
    lower_idx = None
    upper_idx = None

    for i, q in enumerate(quantile_levels):
        if abs(q - 0.1) < 0.01:
            lower_idx = i
        if abs(q - 0.9) < 0.01:
            upper_idx = i

    if lower_idx is not None and upper_idx is not None:
        # Extract lower/upper quantiles
        if quantiles.ndim == 4:
            lower = quantiles[:, :, lower_idx, :].flatten()
            upper = quantiles[:, :, upper_idx, :].flatten()
        elif quantiles.ndim == 3:
            lower = quantiles[:, :, lower_idx].flatten()
            upper = quantiles[:, :, upper_idx].flatten()
        else:
            lower = quantiles[:, lower_idx].flatten()
            upper = quantiles[:, upper_idx].flatten()

        targets_flat = targets.flatten()

        results["coverage_80"] = coverage(lower, upper, targets_flat)
        results["interval_width_80"] = interval_width(lower, upper)
        results["winkler_80"] = winkler_score(lower, upper, targets_flat, alpha=0.2)

    return results


def sharpness(
    quantiles: np.ndarray,
    quantile_levels: List[float],
) -> float:
    """
    Compute sharpness (average interval width) for given quantiles.

    Args:
        quantiles: Quantile predictions
        quantile_levels: List of quantile levels

    Returns:
        Average sharpness (interval width)
    """
    # Find symmetric quantile pair closest to (0.1, 0.9)
    lower_idx = None
    upper_idx = None

    for i, q in enumerate(quantile_levels):
        if q < 0.5:
            if lower_idx is None or q > quantile_levels[lower_idx]:
                lower_idx = i
        elif q > 0.5:
            if upper_idx is None or q < quantile_levels[upper_idx]:
                upper_idx = i

    if lower_idx is None or upper_idx is None:
        return 0.0

    if quantiles.ndim >= 3:
        lower = quantiles[..., lower_idx].flatten()
        upper = quantiles[..., upper_idx].flatten()
    else:
        lower = quantiles[:, lower_idx].flatten()
        upper = quantiles[:, upper_idx].flatten()

    return float(np.mean(upper - lower))
