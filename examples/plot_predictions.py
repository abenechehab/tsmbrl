#!/usr/bin/env python3
"""
Visualization example for TSMBRL.

Demonstrates how to create publication-quality plots of predictions.
"""

import numpy as np
from pathlib import Path


def create_synthetic_results():
    """Create synthetic results for demonstration."""
    np.random.seed(42)

    # Synthetic trajectory
    n_context = 50
    n_horizon = 10
    obs_dim = 4

    t = np.linspace(0, 4 * np.pi, n_context + n_horizon)
    ground_truth = np.column_stack(
        [
            np.sin(t),
            np.cos(t),
            np.sin(2 * t) + 0.5,
            np.cos(2 * t) - 0.5,
        ]
    )
    ground_truth += 0.1 * np.random.randn(n_context + n_horizon, obs_dim)

    # Simulated predictions (with some error)
    predictions = {
        "chronos2": ground_truth[n_context:] + 0.15 * np.random.randn(n_horizon, obs_dim),
        "baseline": ground_truth[n_context:] + 0.3 * np.random.randn(n_horizon, obs_dim),
    }

    # Quantile predictions
    mean_pred = predictions["chronos2"]
    lower = mean_pred - 0.3
    upper = mean_pred + 0.3

    return {
        "ground_truth": ground_truth,
        "predictions": predictions,
        "context_length": n_context,
        "mean": mean_pred,
        "lower": lower,
        "upper": upper,
    }


def example_trajectory_plot(data, save_dir):
    """Plot trajectory predictions."""
    print("\n[1] Creating trajectory plot...")

    from tsmbrl.visualization.plot_predictions import plot_trajectory_predictions

    plot_trajectory_predictions(
        ground_truth=data["ground_truth"],
        predictions=data["predictions"],
        context_length=data["context_length"],
        dimension=0,
        title="Observation Dimension 0: Predictions vs Ground Truth",
        save_path=save_dir / "trajectory_predictions.png",
        show=False,
    )
    print(f"  Saved to {save_dir / 'trajectory_predictions.png'}")


def example_uncertainty_plot(data, save_dir):
    """Plot predictions with uncertainty bands."""
    print("\n[2] Creating uncertainty plot...")

    from tsmbrl.visualization.plot_uncertainty import plot_prediction_intervals

    horizon = data["mean"].shape[0]
    context = data["ground_truth"][: data["context_length"], 0]
    ground_truth = data["ground_truth"][data["context_length"] :, 0]

    plot_prediction_intervals(
        ground_truth=ground_truth,
        mean_predictions=data["mean"][:, 0],
        lower_quantile=data["lower"][:, 0],
        upper_quantile=data["upper"][:, 0],
        context=context,
        dimension=0,
        quantile_level=0.8,
        title="Predictions with 80% Confidence Interval",
        save_path=save_dir / "uncertainty_bands.png",
        show=False,
    )
    print(f"  Saved to {save_dir / 'uncertainty_bands.png'}")


def example_horizon_mse_plot(save_dir):
    """Plot MSE vs prediction horizon."""
    print("\n[3] Creating MSE vs horizon plot...")

    from tsmbrl.visualization.plot_predictions import plot_multi_horizon_mse

    # Simulated MSE values that increase with horizon
    mse_per_step = {
        "chronos2": [0.01 * (1 + 0.1 * h) ** 2 for h in range(10)],
        "baseline": [0.02 * (1 + 0.15 * h) ** 2 for h in range(10)],
    }

    plot_multi_horizon_mse(
        mse_per_step=mse_per_step,
        title="MSE Growth Over Prediction Horizon",
        save_path=save_dir / "mse_vs_horizon.png",
        show=False,
    )
    print(f"  Saved to {save_dir / 'mse_vs_horizon.png'}")


def example_model_comparison(save_dir):
    """Plot model comparison bar chart."""
    print("\n[4] Creating model comparison plot...")

    from tsmbrl.visualization.plot_comparisons import plot_aggregated_results

    results = {
        "chronos2": {"mse": 0.015, "mae": 0.10, "crps": 0.08},
        "chronos2-small": {"mse": 0.018, "mae": 0.12, "crps": 0.09},
        "baseline": {"mse": 0.025, "mae": 0.15, "crps": 0.12},
    }

    plot_aggregated_results(
        results=results,
        metrics=["mse", "mae", "crps"],
        title="Model Comparison Across Metrics",
        save_path=save_dir / "model_comparison.png",
        show=False,
    )
    print(f"  Saved to {save_dir / 'model_comparison.png'}")


def example_calibration_plot(save_dir):
    """Plot calibration diagram."""
    print("\n[5] Creating calibration plot...")

    from tsmbrl.visualization.plot_uncertainty import plot_calibration

    # Simulated calibration errors (well-calibrated)
    calibration_errors = {
        0.1: -0.02,  # Observed 0.08 instead of 0.10
        0.25: 0.01,
        0.5: -0.01,
        0.75: 0.02,
        0.9: -0.01,
    }

    plot_calibration(
        calibration_errors=calibration_errors,
        title="Quantile Calibration",
        save_path=save_dir / "calibration.png",
        show=False,
    )
    print(f"  Saved to {save_dir / 'calibration.png'}")


def main():
    """Run all visualization examples."""
    print("=" * 60)
    print("TSMBRL Visualization Examples")
    print("=" * 60)

    # Create output directory
    save_dir = Path("results/figures")
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving figures to: {save_dir.absolute()}")

    # Generate synthetic data
    data = create_synthetic_results()

    # Run examples
    try:
        example_trajectory_plot(data, save_dir)
    except Exception as e:
        print(f"  Failed: {e}")

    try:
        example_uncertainty_plot(data, save_dir)
    except Exception as e:
        print(f"  Failed: {e}")

    try:
        example_horizon_mse_plot(save_dir)
    except Exception as e:
        print(f"  Failed: {e}")

    try:
        example_model_comparison(save_dir)
    except Exception as e:
        print(f"  Failed: {e}")

    try:
        example_calibration_plot(save_dir)
    except Exception as e:
        print(f"  Failed: {e}")

    print("\n" + "=" * 60)
    print("Visualization examples complete!")
    print(f"Check {save_dir.absolute()} for output files.")
    print("=" * 60)


if __name__ == "__main__":
    main()
