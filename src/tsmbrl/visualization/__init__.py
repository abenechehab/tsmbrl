"""Visualization tools for TSMBRL."""

from tsmbrl.visualization.figure_config import (
    apply_style,
    COLORS,
    MODEL_COLORS,
    get_model_color,
)
from tsmbrl.visualization.plot_predictions import (
    plot_trajectory_predictions,
    plot_multi_horizon_mse,
)
from tsmbrl.visualization.plot_uncertainty import (
    plot_prediction_intervals,
    plot_calibration,
)
from tsmbrl.visualization.plot_comparisons import (
    plot_model_comparison_bar,
    plot_aggregated_results,
)

__all__ = [
    # Configuration
    "apply_style",
    "COLORS",
    "MODEL_COLORS",
    "get_model_color",
    # Prediction plots
    "plot_trajectory_predictions",
    "plot_multi_horizon_mse",
    # Uncertainty plots
    "plot_prediction_intervals",
    "plot_calibration",
    # Comparison plots
    "plot_model_comparison_bar",
    "plot_aggregated_results",
]
