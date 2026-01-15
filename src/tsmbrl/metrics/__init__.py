"""Evaluation metrics for TSMBRL."""

from tsmbrl.metrics.forecasting_metrics import (
    mse,
    mae,
    rmse,
    single_step_metrics,
    multi_step_metrics,
    per_dimension_metrics,
    normalized_mse,
)
from tsmbrl.metrics.probabilistic_metrics import (
    crps_from_quantiles,
    calibration_error,
    coverage,
    interval_width,
    winkler_score,
    compute_all_probabilistic_metrics,
)
from tsmbrl.metrics.metric_utils import aggregate_metrics_across_datasets, rank_models

__all__ = [
    # Forecasting metrics
    "mse",
    "mae",
    "rmse",
    "single_step_metrics",
    "multi_step_metrics",
    "per_dimension_metrics",
    "normalized_mse",
    # Probabilistic metrics
    "crps_from_quantiles",
    "calibration_error",
    "coverage",
    "interval_width",
    "winkler_score",
    "compute_all_probabilistic_metrics",
    # Utilities
    "aggregate_metrics_across_datasets",
    "rank_models",
]
