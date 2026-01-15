#!/usr/bin/env python3
"""
Basic usage example for TSMBRL.

This script demonstrates:
1. Loading a Minari dataset
2. Preparing data for TSFM inference
3. Running Chronos predictions
4. Computing and printing metrics

Note: Requires Minari and Chronos to be installed, and the model to be downloaded.
"""

import numpy as np


def main():
    """Run basic TSMBRL example."""
    print("=" * 60)
    print("TSMBRL Basic Usage Example")
    print("=" * 60)

    # Configuration
    DATASET = "D4RL/door/human-v2"
    MODEL = "chronos2"
    LOOKBACK = 50
    HORIZON = 10
    MAX_EPISODES = 5  # Use few episodes for demo

    print(f"\nConfiguration:")
    print(f"  Dataset: {DATASET}")
    print(f"  Model: {MODEL}")
    print(f"  Lookback: {LOOKBACK}")
    print(f"  Horizon: {HORIZON}")

    # Import TSMBRL components
    from tsmbrl.data.minari_loader import MinariDataLoader
    from tsmbrl.data.formatters import TimeSeriesFormatter, prepare_evaluation_data
    from tsmbrl.models.model_registry import get_model
    from tsmbrl.metrics.forecasting_metrics import multi_step_metrics
    from tsmbrl.metrics.probabilistic_metrics import compute_all_probabilistic_metrics

    # 1. Load dataset
    print(f"\n[1/5] Loading dataset: {DATASET}")
    loader = MinariDataLoader(DATASET, download=True)
    print(f"  Total episodes: {loader.total_episodes}")
    print(f"  Obs dim: {loader.obs_dim}, Act dim: {loader.act_dim}")

    # 2. Create formatter
    print(f"\n[2/5] Creating formatter")
    formatter = TimeSeriesFormatter(
        lookback=LOOKBACK,
        horizon=HORIZON,
        obs_dim=loader.obs_dim,
        act_dim=loader.act_dim,
    )

    # 3. Prepare evaluation windows
    print(f"\n[3/5] Preparing evaluation windows")
    windows = prepare_evaluation_data(
        loader,
        formatter,
        max_episodes=MAX_EPISODES,
        windows_per_episode=10,
    )
    print(f"  Created {len(windows)} windows")

    if len(windows) == 0:
        print("ERROR: No windows created. Episodes may be too short.")
        return

    # 4. Load model
    print(f"\n[4/5] Loading model: {MODEL}")
    try:
        model = get_model(MODEL, device="cuda")
    except Exception as e:
        print(f"  Failed to load on CUDA, trying CPU: {e}")
        model = get_model(MODEL, device="cpu")

    print(f"  Supports covariates: {model.supports_covariates}")
    print(f"  Is probabilistic: {model.is_probabilistic}")

    # 5. Run predictions
    print(f"\n[5/5] Running predictions...")
    all_preds = []
    all_targets = []
    all_quantiles = []

    for i, window in enumerate(windows[:10]):  # Limit for demo
        if i % 5 == 0:
            print(f"  Processing window {i + 1}/{min(10, len(windows))}")

        result = model.predict_with_actions(
            context_obs=window.context_observations,
            context_actions=window.context_actions,
            future_actions=window.future_actions,
            prediction_length=HORIZON,
            quantile_levels=[0.1, 0.5, 0.9],
        )

        all_preds.append(result["mean"])
        all_targets.append(window.target_observations)
        if "quantiles" in result:
            all_quantiles.append(result["quantiles"])

    # Stack and compute metrics
    predictions = np.stack(all_preds)
    targets = np.stack(all_targets)

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    forecast_metrics = multi_step_metrics(predictions, targets)
    print(f"\nForecasting Metrics:")
    print(f"  MSE:  {forecast_metrics['mse']:.6f}")
    print(f"  MAE:  {forecast_metrics['mae']:.6f}")
    print(f"  RMSE: {forecast_metrics['rmse']:.6f}")
    print(f"  MSE per step: {[f'{m:.4f}' for m in forecast_metrics['mse_per_step']]}")

    if all_quantiles:
        try:
            quantiles = np.stack(all_quantiles)
            prob_metrics = compute_all_probabilistic_metrics(
                quantiles, [0.1, 0.5, 0.9], targets
            )
            print(f"\nProbabilistic Metrics:")
            print(f"  CRPS: {prob_metrics['crps']:.6f}")
            if "coverage_80" in prob_metrics:
                print(f"  Coverage (80%): {prob_metrics['coverage_80']:.3f}")
            print(f"  Calibration: {prob_metrics['calibration']}")
        except Exception as e:
            print(f"\nProbabilistic metrics failed: {e}")

    print("\n" + "=" * 60)
    print("Example complete!")


if __name__ == "__main__":
    main()
