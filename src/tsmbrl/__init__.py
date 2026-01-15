"""
TSMBRL: Time Series Foundation Models for Model-Based Reinforcement Learning.

A testbed for evaluating TSFMs on MBRL dynamics modeling tasks.
"""

__version__ = "0.1.0"
__author__ = "TSMBRL Team"

from tsmbrl.data.minari_loader import MinariDataLoader
from tsmbrl.data.formatters import TimeSeriesFormatter, ForecastWindow
from tsmbrl.models.model_registry import get_model, list_models
from tsmbrl.metrics.forecasting_metrics import mse, mae, rmse, multi_step_metrics
from tsmbrl.metrics.probabilistic_metrics import crps_from_quantiles, calibration_error

__all__ = [
    "MinariDataLoader",
    "TimeSeriesFormatter",
    "ForecastWindow",
    "get_model",
    "list_models",
    "mse",
    "mae",
    "rmse",
    "multi_step_metrics",
    "crps_from_quantiles",
    "calibration_error",
]
