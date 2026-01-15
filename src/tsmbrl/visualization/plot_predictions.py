"""Prediction visualization functions for TSMBRL."""

from pathlib import Path
from typing import Dict, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np

from .figure_config import COLORS, apply_style, get_model_color, save_figure


def plot_trajectory_predictions(
    ground_truth: np.ndarray,
    predictions: Dict[str, np.ndarray],
    context_length: int,
    dimension: int = 0,
    title: Optional[str] = None,
    xlabel: str = "Timestep",
    ylabel: Optional[str] = None,
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True,
) -> Optional[plt.Figure]:
    """
    Plot predictions from multiple models on a single trajectory.

    Args:
        ground_truth: Full trajectory, shape (total_steps, obs_dim)
        predictions: Dict mapping model names to predictions, shape (horizon, obs_dim)
        context_length: Length of context used
        dimension: Which observation dimension to plot
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        save_path: Path to save figure
        show: Whether to display the plot

    Returns:
        Figure object if show=False
    """
    apply_style()

    fig, ax = plt.subplots(figsize=(12, 6))

    # Time indices
    total_steps = len(ground_truth)
    horizon = list(predictions.values())[0].shape[0]

    t_full = np.arange(total_steps)
    t_context = np.arange(context_length)
    t_pred = np.arange(context_length, context_length + horizon)

    # Plot full ground truth
    ax.plot(
        t_full,
        ground_truth[:, dimension] if ground_truth.ndim > 1 else ground_truth,
        color=COLORS["ground_truth"],
        linewidth=2,
        label="Ground Truth",
        alpha=0.7,
    )

    # Plot predictions
    for i, (model_name, pred) in enumerate(predictions.items()):
        color = get_model_color(model_name)
        pred_dim = pred[:, dimension] if pred.ndim > 1 else pred

        ax.plot(
            t_pred,
            pred_dim,
            color=color,
            linewidth=2,
            linestyle="--",
            marker="o",
            markersize=4,
            label=model_name,
        )

    # Mark context/prediction boundary
    ax.axvline(
        x=context_length - 0.5,
        color="gray",
        linestyle=":",
        linewidth=1.5,
        label="Prediction Start",
    )

    # Shade context region
    ax.axvspan(0, context_length - 0.5, alpha=0.1, color="blue", label="_nolegend_")

    # Formatting
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel or f"Observation (dim {dimension})")
    ax.set_title(title or "Trajectory Predictions")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        save_figure(fig, save_path)
        if not show:
            return None

    if show:
        plt.show()
        return None

    return fig


def plot_multi_horizon_mse(
    mse_per_step: Dict[str, List[float]],
    title: Optional[str] = None,
    xlabel: str = "Prediction Horizon",
    ylabel: str = "MSE",
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True,
) -> Optional[plt.Figure]:
    """
    Plot MSE vs prediction horizon for multiple models.

    Args:
        mse_per_step: Dict mapping model name to list of MSE values per step
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        save_path: Path to save figure
        show: Whether to display the plot

    Returns:
        Figure object if show=False
    """
    apply_style()

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, (model_name, mse_values) in enumerate(mse_per_step.items()):
        color = get_model_color(model_name)
        horizons = np.arange(1, len(mse_values) + 1)

        ax.plot(
            horizons,
            mse_values,
            color=color,
            linewidth=2,
            marker="o",
            markersize=6,
            label=model_name,
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title or "MSE vs Prediction Horizon")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Integer x-axis ticks
    ax.set_xticks(range(1, max(len(v) for v in mse_per_step.values()) + 1))

    plt.tight_layout()

    if save_path:
        save_figure(fig, save_path)
        if not show:
            return None

    if show:
        plt.show()
        return None

    return fig


def plot_per_dimension_metrics(
    metrics_per_dim: Dict[str, List[float]],
    metric_name: str = "MSE",
    title: Optional[str] = None,
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True,
) -> Optional[plt.Figure]:
    """
    Plot metrics for each observation dimension.

    Args:
        metrics_per_dim: Dict mapping model name to per-dimension metric values
        metric_name: Name of the metric
        title: Plot title
        save_path: Path to save figure
        show: Whether to display

    Returns:
        Figure object if show=False
    """
    apply_style()

    fig, ax = plt.subplots(figsize=(12, 6))

    n_dims = len(list(metrics_per_dim.values())[0])
    x = np.arange(n_dims)
    width = 0.8 / len(metrics_per_dim)

    for i, (model_name, values) in enumerate(metrics_per_dim.items()):
        offset = (i - len(metrics_per_dim) / 2 + 0.5) * width
        color = get_model_color(model_name)

        ax.bar(
            x + offset,
            values,
            width=width,
            label=model_name,
            color=color,
            edgecolor="black",
            linewidth=0.5,
        )

    ax.set_xlabel("Observation Dimension")
    ax.set_ylabel(metric_name)
    ax.set_title(title or f"{metric_name} per Observation Dimension")
    ax.set_xticks(x)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()

    if save_path:
        save_figure(fig, save_path)
        if not show:
            return None

    if show:
        plt.show()
        return None

    return fig
