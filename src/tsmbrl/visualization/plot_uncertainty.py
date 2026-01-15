"""Uncertainty visualization functions for TSMBRL."""

from pathlib import Path
from typing import Dict, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np

from .figure_config import COLORS, apply_style, save_figure


def plot_prediction_intervals(
    ground_truth: np.ndarray,
    mean_predictions: np.ndarray,
    lower_quantile: np.ndarray,
    upper_quantile: np.ndarray,
    context: Optional[np.ndarray] = None,
    dimension: int = 0,
    quantile_level: float = 0.8,
    title: Optional[str] = None,
    xlabel: str = "Timestep",
    ylabel: Optional[str] = None,
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True,
) -> Optional[plt.Figure]:
    """
    Plot predictions with uncertainty bands.

    Args:
        ground_truth: Target values, shape (horizon,) or (horizon, obs_dim)
        mean_predictions: Mean predictions, same shape as ground_truth
        lower_quantile: Lower bound predictions
        upper_quantile: Upper bound predictions
        context: Optional context observations to show
        dimension: Which dimension to plot (if multivariate)
        quantile_level: Nominal coverage level (for legend)
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

    # Extract dimension if multivariate
    if ground_truth.ndim > 1:
        ground_truth = ground_truth[:, dimension]
        mean_predictions = mean_predictions[:, dimension]
        lower_quantile = lower_quantile[:, dimension]
        upper_quantile = upper_quantile[:, dimension]
        if context is not None:
            context = context[:, dimension]

    horizon = len(ground_truth)

    # Time indices
    if context is not None:
        context_len = len(context)
        t_context = np.arange(context_len)
        t_pred = np.arange(context_len, context_len + horizon)

        # Plot context
        ax.plot(
            t_context,
            context,
            color=COLORS["context"],
            linewidth=2,
            label="Context",
        )

        # Mark boundary
        ax.axvline(
            x=context_len - 0.5,
            color="gray",
            linestyle=":",
            linewidth=1.5,
        )
    else:
        t_pred = np.arange(horizon)

    # Plot uncertainty band first (background)
    ax.fill_between(
        t_pred,
        lower_quantile,
        upper_quantile,
        color=COLORS["uncertainty_fill"],
        alpha=0.3,
        label=f"{int(quantile_level * 100)}% Prediction Interval",
    )

    # Plot ground truth
    ax.plot(
        t_pred,
        ground_truth,
        color=COLORS["ground_truth"],
        linewidth=2,
        linestyle="--",
        marker="s",
        markersize=5,
        label="Ground Truth",
    )

    # Plot mean prediction
    ax.plot(
        t_pred,
        mean_predictions,
        color=COLORS["prediction"],
        linewidth=2,
        marker="o",
        markersize=5,
        label="Prediction (mean)",
    )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel or f"Observation (dim {dimension})")
    ax.set_title(title or "Prediction with Uncertainty")
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


def plot_calibration(
    calibration_errors: Dict[float, float],
    title: Optional[str] = None,
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True,
) -> Optional[plt.Figure]:
    """
    Plot calibration diagram.

    Args:
        calibration_errors: Dict mapping quantile level to calibration error
        title: Plot title
        save_path: Path to save figure
        show: Whether to display the plot

    Returns:
        Figure object if show=False
    """
    apply_style()

    fig, ax = plt.subplots(figsize=(8, 8))

    expected = sorted(calibration_errors.keys())
    observed = [q + calibration_errors[q] for q in expected]

    # Perfect calibration line
    ax.plot(
        [0, 1],
        [0, 1],
        "k--",
        linewidth=1.5,
        label="Perfect Calibration",
    )

    # Observed calibration
    ax.scatter(
        expected,
        observed,
        s=100,
        color=COLORS["prediction"],
        zorder=5,
        edgecolor="black",
        linewidth=1,
    )
    ax.plot(
        expected,
        observed,
        color=COLORS["prediction"],
        linewidth=2,
        label="Observed",
    )

    ax.set_xlabel("Expected Quantile Level")
    ax.set_ylabel("Observed Coverage")
    ax.set_title(title or "Calibration Plot")
    ax.legend(loc="lower right")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_aspect("equal")
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


def plot_coverage_vs_interval(
    coverages: List[float],
    interval_widths: List[float],
    nominal_levels: List[float],
    title: Optional[str] = None,
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True,
) -> Optional[plt.Figure]:
    """
    Plot coverage vs interval width for different confidence levels.

    Args:
        coverages: Empirical coverage values
        interval_widths: Average interval widths
        nominal_levels: Nominal coverage levels
        title: Plot title
        save_path: Path to save figure
        show: Whether to display

    Returns:
        Figure object if show=False
    """
    apply_style()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Coverage plot
    ax1.scatter(
        nominal_levels,
        coverages,
        s=100,
        color=COLORS["prediction"],
        zorder=5,
    )
    ax1.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect")
    ax1.set_xlabel("Nominal Coverage")
    ax1.set_ylabel("Empirical Coverage")
    ax1.set_title("Coverage Calibration")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Width plot
    ax2.bar(
        [str(n) for n in nominal_levels],
        interval_widths,
        color=COLORS["prediction"],
        edgecolor="black",
    )
    ax2.set_xlabel("Nominal Coverage Level")
    ax2.set_ylabel("Average Interval Width")
    ax2.set_title("Interval Width by Coverage Level")
    ax2.grid(True, alpha=0.3, axis="y")

    fig.suptitle(title or "Uncertainty Quantification Analysis", fontsize=14)
    plt.tight_layout()

    if save_path:
        save_figure(fig, save_path)
        if not show:
            return None

    if show:
        plt.show()
        return None

    return fig
