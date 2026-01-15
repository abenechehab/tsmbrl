"""Utility functions for metric computation and aggregation."""

from typing import Any, Dict, List

import numpy as np


def aggregate_metrics_across_datasets(
    results: List[Dict[str, Any]],
) -> Dict[str, float]:
    """
    Aggregate metrics across multiple datasets/experiments.

    Args:
        results: List of metric dictionaries from individual experiments

    Returns:
        Aggregated statistics (mean, std, median) for each numeric metric

    Example:
        >>> results = [{"mse": 0.1}, {"mse": 0.2}, {"mse": 0.15}]
        >>> agg = aggregate_metrics_across_datasets(results)
        >>> print(agg["mse_mean"])  # 0.15
    """
    metric_values: Dict[str, List[float]] = {}

    for r in results:
        for key, value in r.items():
            if isinstance(value, (int, float)) and not np.isnan(value):
                if key not in metric_values:
                    metric_values[key] = []
                metric_values[key].append(float(value))

    aggregated = {}
    for key, values in metric_values.items():
        values_arr = np.array(values)
        aggregated[f"{key}_mean"] = float(np.mean(values_arr))
        aggregated[f"{key}_std"] = float(np.std(values_arr))
        aggregated[f"{key}_median"] = float(np.median(values_arr))
        aggregated[f"{key}_min"] = float(np.min(values_arr))
        aggregated[f"{key}_max"] = float(np.max(values_arr))

    return aggregated


def rank_models(
    results: Dict[str, Dict[str, float]],
    metric: str = "mse",
    lower_is_better: bool = True,
) -> List[str]:
    """
    Rank models by a specific metric.

    Args:
        results: Dictionary mapping model name to metrics dictionary
        metric: Metric to rank by
        lower_is_better: Whether lower values are better

    Returns:
        List of model names sorted by performance (best first)

    Example:
        >>> results = {"model_a": {"mse": 0.1}, "model_b": {"mse": 0.2}}
        >>> rank_models(results, metric="mse")
        ['model_a', 'model_b']
    """
    model_scores = [
        (name, metrics.get(metric, float("inf"))) for name, metrics in results.items()
    ]

    model_scores.sort(key=lambda x: x[1], reverse=not lower_is_better)

    return [name for name, _ in model_scores]


def compare_models(
    results: Dict[str, Dict[str, float]],
    metrics: List[str] = ["mse", "mae", "crps"],
) -> Dict[str, Dict[str, Any]]:
    """
    Compare models across multiple metrics.

    Args:
        results: Dictionary mapping model name to metrics
        metrics: List of metrics to compare

    Returns:
        Dictionary with comparison statistics

    Example:
        >>> results = {
        ...     "model_a": {"mse": 0.1, "mae": 0.2},
        ...     "model_b": {"mse": 0.15, "mae": 0.18}
        ... }
        >>> compare_models(results)
    """
    comparison: Dict[str, Dict[str, Any]] = {}

    for metric in metrics:
        values = []
        for name, m in results.items():
            if metric in m:
                values.append((name, m[metric]))

        if not values:
            continue

        # Sort by metric value
        values.sort(key=lambda x: x[1])

        comparison[metric] = {
            "ranking": [name for name, _ in values],
            "values": {name: val for name, val in values},
            "best": values[0][0],
            "worst": values[-1][0],
            "spread": values[-1][1] - values[0][1] if len(values) > 1 else 0,
        }

    return comparison


def compute_relative_improvement(
    baseline: Dict[str, float],
    model: Dict[str, float],
    metrics: List[str] = ["mse", "mae", "rmse"],
) -> Dict[str, float]:
    """
    Compute relative improvement of model over baseline.

    Args:
        baseline: Baseline model metrics
        model: Model metrics to compare
        metrics: List of metrics to compare

    Returns:
        Dictionary with relative improvement for each metric
        (positive = model is better, negative = model is worse)
    """
    improvements = {}

    for metric in metrics:
        if metric in baseline and metric in model:
            base_val = baseline[metric]
            model_val = model[metric]

            if abs(base_val) > 1e-8:
                # Relative improvement (lower is better for these metrics)
                improvement = (base_val - model_val) / abs(base_val)
                improvements[f"{metric}_improvement"] = float(improvement)
                improvements[f"{metric}_improvement_pct"] = float(improvement * 100)

    return improvements


def format_metrics_table(
    results: Dict[str, Dict[str, float]],
    metrics: List[str] = ["mse", "mae", "rmse", "crps"],
    precision: int = 4,
) -> str:
    """
    Format results as a text table.

    Args:
        results: Dictionary mapping model name to metrics
        metrics: List of metrics to include
        precision: Decimal places for values

    Returns:
        Formatted table string
    """
    # Header
    header = ["Model"] + [m.upper() for m in metrics]
    col_widths = [max(len(h), 15) for h in header]

    lines = []

    # Header row
    header_str = " | ".join(h.ljust(w) for h, w in zip(header, col_widths))
    lines.append(header_str)
    lines.append("-" * len(header_str))

    # Data rows
    for model_name, model_metrics in results.items():
        row = [model_name]
        for metric in metrics:
            if metric in model_metrics:
                row.append(f"{model_metrics[metric]:.{precision}f}")
            else:
                row.append("N/A")

        row_str = " | ".join(str(v).ljust(w) for v, w in zip(row, col_widths))
        lines.append(row_str)

    return "\n".join(lines)
