"""Standard forecasting metrics for TSMBRL."""

from typing import Dict, List, Optional, Union

import numpy as np


def mse(
    predictions: np.ndarray,
    targets: np.ndarray,
    axis: Optional[int] = None,
) -> Union[float, np.ndarray]:
    """
    Mean Squared Error.

    Args:
        predictions: Predicted values
        targets: Ground truth values
        axis: Axis to compute mean over (None = all elements)

    Returns:
        MSE value(s)
    """
    return np.mean((predictions - targets) ** 2, axis=axis)


def mae(
    predictions: np.ndarray,
    targets: np.ndarray,
    axis: Optional[int] = None,
) -> Union[float, np.ndarray]:
    """
    Mean Absolute Error.

    Args:
        predictions: Predicted values
        targets: Ground truth values
        axis: Axis to compute mean over (None = all elements)

    Returns:
        MAE value(s)
    """
    return np.mean(np.abs(predictions - targets), axis=axis)


def rmse(
    predictions: np.ndarray,
    targets: np.ndarray,
    axis: Optional[int] = None,
) -> Union[float, np.ndarray]:
    """
    Root Mean Squared Error.

    Args:
        predictions: Predicted values
        targets: Ground truth values
        axis: Axis to compute mean over (None = all elements)

    Returns:
        RMSE value(s)
    """
    return np.sqrt(mse(predictions, targets, axis=axis))


def single_step_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
) -> Dict[str, float]:
    """
    Compute metrics for single-step predictions.

    Args:
        predictions: Shape (batch,) or (batch, 1) or (batch, 1, features)
        targets: Same shape as predictions

    Returns:
        Dictionary with MSE, MAE, RMSE
    """
    pred_flat = predictions.flatten()
    tgt_flat = targets.flatten()

    return {
        "mse": float(mse(pred_flat, tgt_flat)),
        "mae": float(mae(pred_flat, tgt_flat)),
        "rmse": float(rmse(pred_flat, tgt_flat)),
    }


def multi_step_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
) -> Dict[str, Union[float, List[float]]]:
    """
    Compute metrics for multi-step predictions.

    Args:
        predictions: Shape (batch, horizon) or (batch, horizon, features)
        targets: Same shape as predictions

    Returns:
        Dictionary with:
            - Aggregated metrics (mse, mae, rmse)
            - Per-step metrics (mse_per_step, mae_per_step, rmse_per_step)
            - horizon: Number of prediction steps
    """
    # Ensure 3D: (batch, horizon, features)
    if predictions.ndim == 2:
        predictions = predictions[:, :, np.newaxis]
        targets = targets[:, :, np.newaxis]

    batch_size, horizon, n_features = predictions.shape

    # Per-step metrics (average over batch and features)
    mse_per_step = []
    mae_per_step = []
    rmse_per_step = []

    for h in range(horizon):
        pred_h = predictions[:, h, :].flatten()
        tgt_h = targets[:, h, :].flatten()

        mse_per_step.append(float(mse(pred_h, tgt_h)))
        mae_per_step.append(float(mae(pred_h, tgt_h)))
        rmse_per_step.append(float(rmse(pred_h, tgt_h)))

    # Aggregated metrics
    pred_flat = predictions.flatten()
    tgt_flat = targets.flatten()

    return {
        "mse": float(mse(pred_flat, tgt_flat)),
        "mae": float(mae(pred_flat, tgt_flat)),
        "rmse": float(rmse(pred_flat, tgt_flat)),
        "mse_per_step": mse_per_step,
        "mae_per_step": mae_per_step,
        "rmse_per_step": rmse_per_step,
        "horizon": horizon,
    }


def per_dimension_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
) -> Dict[str, Union[List[float], int]]:
    """
    Compute metrics for each observation dimension separately.

    Args:
        predictions: Shape (batch, horizon, features) or (batch, features)
        targets: Same shape as predictions

    Returns:
        Dictionary with per-dimension MSE, MAE, RMSE
    """
    # Flatten time dimension if present
    if predictions.ndim == 3:
        batch, horizon, features = predictions.shape
        predictions = predictions.reshape(-1, features)
        targets = targets.reshape(-1, features)

    n_features = predictions.shape[-1]

    return {
        "mse_per_dim": [
            float(mse(predictions[:, f], targets[:, f])) for f in range(n_features)
        ],
        "mae_per_dim": [
            float(mae(predictions[:, f], targets[:, f])) for f in range(n_features)
        ],
        "rmse_per_dim": [
            float(rmse(predictions[:, f], targets[:, f])) for f in range(n_features)
        ],
        "n_features": n_features,
    }


def normalized_mse(
    predictions: np.ndarray,
    targets: np.ndarray,
) -> float:
    """
    Normalized MSE (by target variance).

    Useful for comparing across different scales. A value of 1.0 means
    the model performs as well as predicting the mean.

    Args:
        predictions: Predicted values
        targets: Ground truth values

    Returns:
        Normalized MSE (lower is better, <1 is better than mean prediction)
    """
    var = np.var(targets)
    if var < 1e-8:
        return 0.0
    return float(mse(predictions, targets) / var)


def mape(
    predictions: np.ndarray,
    targets: np.ndarray,
    epsilon: float = 1e-8,
) -> float:
    """
    Mean Absolute Percentage Error.

    Args:
        predictions: Predicted values
        targets: Ground truth values (must not be zero)
        epsilon: Small value to avoid division by zero

    Returns:
        MAPE as a fraction (multiply by 100 for percentage)
    """
    return float(np.mean(np.abs((targets - predictions) / (np.abs(targets) + epsilon))))


def smape(
    predictions: np.ndarray,
    targets: np.ndarray,
    epsilon: float = 1e-8,
) -> float:
    """
    Symmetric Mean Absolute Percentage Error.

    Args:
        predictions: Predicted values
        targets: Ground truth values
        epsilon: Small value to avoid division by zero

    Returns:
        sMAPE as a fraction (multiply by 100 for percentage)
    """
    numerator = np.abs(targets - predictions)
    denominator = np.abs(targets) + np.abs(predictions) + epsilon
    return float(np.mean(2 * numerator / denominator))


def compute_all_forecasting_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
) -> Dict[str, Union[float, List[float]]]:
    """
    Compute all forecasting metrics.

    Args:
        predictions: Shape (batch, horizon, features) or (batch, horizon)
        targets: Same shape as predictions

    Returns:
        Dictionary with all metrics
    """
    result = multi_step_metrics(predictions, targets)

    # Add normalized metrics
    result["normalized_mse"] = normalized_mse(predictions, targets)

    # Add per-dimension metrics
    dim_metrics = per_dimension_metrics(predictions, targets)
    result.update(dim_metrics)

    return result
