"""Model comparison visualization functions for TSMBRL."""

from pathlib import Path
from typing import Dict, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np

from .figure_config import apply_style, get_model_color, save_figure


def plot_model_comparison_bar(
    results: Dict[str, Dict[str, float]],
    metric: str = "mse",
    title: Optional[str] = None,
    ylabel: Optional[str] = None,
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True,
) -> Optional[plt.Figure]:
    """
    Bar chart comparing models on a single metric.

    Args:
        results: Dict mapping model name to metrics dict
        metric: Which metric to plot
        title: Plot title
        ylabel: Y-axis label
        save_path: Path to save figure
        show: Whether to display

    Returns:
        Figure object if show=False
    """
    apply_style()

    models = list(results.keys())
    values = [results[m].get(metric, 0) for m in models]
    colors = [get_model_color(m) for m in models]

    fig, ax = plt.subplots(figsize=(10, 6))

    bars = ax.bar(
        models,
        values,
        color=colors,
        edgecolor="black",
        linewidth=1,
    )

    ax.set_ylabel(ylabel or metric.upper())
    ax.set_title(title or f"Model Comparison: {metric.upper()}")
    ax.grid(True, alpha=0.3, axis="y")

    # Rotate x labels if many models
    if len(models) > 4:
        plt.xticks(rotation=45, ha="right")

    # Add value labels on bars
    max_val = max(values) if values else 1
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02 * max_val,
            f"{val:.4f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    plt.tight_layout()

    if save_path:
        save_figure(fig, save_path)
        if not show:
            return None

    if show:
        plt.show()
        return None

    return fig


def plot_aggregated_results(
    results: Dict[str, Dict[str, float]],
    metrics: List[str] = ["mse", "mae", "crps"],
    title: Optional[str] = None,
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True,
) -> Optional[plt.Figure]:
    """
    Multi-bar chart comparing models across multiple metrics.

    Args:
        results: Dict mapping model name to metrics dict
        metrics: List of metrics to include
        title: Plot title
        save_path: Path to save figure
        show: Whether to display

    Returns:
        Figure object if show=False
    """
    apply_style()

    models = list(results.keys())
    n_models = len(models)
    n_metrics = len(metrics)

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(n_metrics)
    width = 0.8 / n_models

    for i, model in enumerate(models):
        values = [results[model].get(m, 0) for m in metrics]
        offset = (i - n_models / 2 + 0.5) * width

        ax.bar(
            x + offset,
            values,
            width=width,
            label=model,
            color=get_model_color(model),
            edgecolor="black",
            linewidth=0.5,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([m.upper() for m in metrics])
    ax.set_ylabel("Value")
    ax.set_title(title or "Model Comparison Across Metrics")
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


def plot_dataset_comparison(
    results: Dict[str, Dict[str, float]],
    metric: str = "mse",
    title: Optional[str] = None,
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True,
) -> Optional[plt.Figure]:
    """
    Compare a single model across multiple datasets.

    Args:
        results: Dict mapping dataset name to metrics dict
        metric: Which metric to plot
        title: Plot title
        save_path: Path to save figure
        show: Whether to display

    Returns:
        Figure object if show=False
    """
    apply_style()

    datasets = list(results.keys())
    values = [results[d].get(metric, 0) for d in datasets]

    fig, ax = plt.subplots(figsize=(12, 6))

    bars = ax.bar(
        datasets,
        values,
        color="#3498DB",
        edgecolor="black",
        linewidth=1,
    )

    ax.set_ylabel(metric.upper())
    ax.set_xlabel("Dataset")
    ax.set_title(title or f"{metric.upper()} Across Datasets")
    ax.grid(True, alpha=0.3, axis="y")

    # Rotate x labels
    plt.xticks(rotation=45, ha="right")

    # Add value labels
    max_val = max(values) if values else 1
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02 * max_val,
            f"{val:.4f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()

    if save_path:
        save_figure(fig, save_path)
        if not show:
            return None

    if show:
        plt.show()
        return None

    return fig


def plot_actions_ablation(
    with_actions: Dict[str, float],
    without_actions: Dict[str, float],
    metrics: List[str] = ["mse", "mae", "rmse"],
    title: Optional[str] = None,
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True,
) -> Optional[plt.Figure]:
    """
    Compare model performance with and without action covariates.

    Args:
        with_actions: Metrics when using actions as covariates
        without_actions: Metrics without actions (baseline)
        metrics: List of metrics to compare
        title: Plot title
        save_path: Path to save figure
        show: Whether to display

    Returns:
        Figure object if show=False
    """
    apply_style()

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(metrics))
    width = 0.35

    values_with = [with_actions.get(m, 0) for m in metrics]
    values_without = [without_actions.get(m, 0) for m in metrics]

    bars1 = ax.bar(
        x - width / 2,
        values_without,
        width,
        label="Without Actions",
        color="#95A5A6",
        edgecolor="black",
    )
    bars2 = ax.bar(
        x + width / 2,
        values_with,
        width,
        label="With Actions",
        color="#E74C3C",
        edgecolor="black",
    )

    ax.set_ylabel("Value")
    ax.set_title(title or "Action Covariate Ablation")
    ax.set_xticks(x)
    ax.set_xticklabels([m.upper() for m in metrics])
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    # Add improvement percentage
    for i, (v_with, v_without) in enumerate(zip(values_with, values_without)):
        if v_without > 0:
            improvement = (v_without - v_with) / v_without * 100
            sign = "+" if improvement < 0 else ""
            ax.annotate(
                f"{sign}{-improvement:.1f}%",
                xy=(i, max(v_with, v_without) * 1.05),
                ha="center",
                fontsize=9,
                color="green" if improvement > 0 else "red",
            )

    plt.tight_layout()

    if save_path:
        save_figure(fig, save_path)
        if not show:
            return None

    if show:
        plt.show()
        return None

    return fig
