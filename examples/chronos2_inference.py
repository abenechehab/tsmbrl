#!/usr/bin/env python3
"""
Chronos-2 specific example showing different usage patterns.

This demonstrates:
1. Point predictions (no quantiles)
2. Probabilistic predictions with quantiles
3. Predictions with covariates (actions for MBRL)
"""

import numpy as np
import torch


def example_point_prediction():
    """Demonstrate point predictions without quantiles."""
    print("=" * 50)
    print("Example 1: Point Predictions")
    print("=" * 50)

    from tsmbrl.models.chronos2_wrapper import Chronos2TSFM

    # Load model
    print("Loading Chronos-2...")
    try:
        model = Chronos2TSFM(device="cuda")
    except Exception:
        print("CUDA not available, using CPU")
        model = Chronos2TSFM(device="cpu")

    # Create synthetic time series
    t = np.linspace(0, 4 * np.pi, 100)
    series = np.sin(t) + 0.1 * np.random.randn(100)

    # Predict (no quantiles = point prediction only)
    context = series[:80]
    result = model.predict(
        context=context,
        prediction_length=20,
        quantile_levels=None,  # No quantiles
    )

    print(f"Context shape: {context.shape}")
    print(f"Mean predictions shape: {result['mean'].shape}")
    print(f"Contains quantiles: {'quantiles' in result}")

    # Compute error on ground truth
    ground_truth = series[80:100]
    mse = np.mean((result["mean"].flatten() - ground_truth) ** 2)
    print(f"MSE on held-out data: {mse:.6f}")

    return result


def example_probabilistic():
    """Demonstrate probabilistic predictions with quantiles."""
    print("\n" + "=" * 50)
    print("Example 2: Probabilistic Predictions")
    print("=" * 50)

    from tsmbrl.models.chronos2_wrapper import Chronos2TSFM

    try:
        model = Chronos2TSFM(device="cuda")
    except Exception:
        model = Chronos2TSFM(device="cpu")

    # Create synthetic time series
    t = np.linspace(0, 4 * np.pi, 100)
    series = np.sin(t) + 0.1 * np.random.randn(100)

    # Predict with quantiles
    context = series[:80]
    result = model.predict(
        context=context,
        prediction_length=20,
        quantile_levels=[0.1, 0.5, 0.9],  # With quantiles
    )

    print(f"Context shape: {context.shape}")
    print(f"Mean predictions shape: {result['mean'].shape}")
    print(f"Quantiles shape: {result['quantiles'].shape}")
    print(f"Quantile levels: {result['quantile_levels']}")

    return result


def example_multivariate():
    """Demonstrate multivariate prediction."""
    print("\n" + "=" * 50)
    print("Example 3: Multivariate Prediction")
    print("=" * 50)

    from tsmbrl.models.chronos2_wrapper import Chronos2TSFM

    try:
        model = Chronos2TSFM(device="cuda")
    except Exception:
        model = Chronos2TSFM(device="cpu")

    # Create multivariate synthetic data
    n_steps = 100
    n_features = 4
    t = np.linspace(0, 4 * np.pi, n_steps)

    data = np.column_stack(
        [
            np.sin(t),
            np.cos(t),
            np.sin(2 * t),
            np.cos(2 * t),
        ]
    )
    data += 0.1 * np.random.randn(n_steps, n_features)

    # Predict (handles multivariate by looping over dimensions)
    context = data[:80]
    result = model.predict(
        context=context,
        prediction_length=20,
        quantile_levels=[0.1, 0.5, 0.9],
    )

    print(f"Context shape: {context.shape}")
    print(f"Mean predictions shape: {result['mean'].shape}")
    print(f"Quantiles shape: {result['quantiles'].shape}")

    # Compute per-dimension MSE
    ground_truth = data[80:100]
    for d in range(n_features):
        mse = np.mean((result["mean"][:, d] - ground_truth[:, d]) ** 2)
        print(f"  Dimension {d} MSE: {mse:.6f}")

    return result


def example_with_covariates():
    """Demonstrate prediction with covariates (actions for MBRL)."""
    print("\n" + "=" * 50)
    print("Example 4: Prediction with Covariates (MBRL)")
    print("=" * 50)

    from tsmbrl.models.chronos2_wrapper import Chronos2TSFM

    try:
        model = Chronos2TSFM(device="cuda")
    except Exception:
        model = Chronos2TSFM(device="cpu")

    # Simulate MBRL-like data
    lookback = 50
    horizon = 10
    obs_dim = 4
    act_dim = 2

    # Generate synthetic trajectory
    np.random.seed(42)
    context_obs = np.random.randn(lookback, obs_dim)
    future_actions = np.random.randn(horizon, act_dim)
    target_obs = np.random.randn(horizon, obs_dim)  # Ground truth

    # Predict with future_covariates (actions)
    result = model.predict(
        context=context_obs,
        prediction_length=horizon,
        future_covariates=future_actions,  # Actions as covariates
        quantile_levels=[0.1, 0.5, 0.9],
    )

    print(f"Context observations: {context_obs.shape}")
    print(f"Future actions (covariates): {future_actions.shape}")
    print(f"Predictions: {result['mean'].shape}")
    print(f"Quantiles: {result['quantiles'].shape}")

    # Compute metrics
    mse = np.mean((result["mean"] - target_obs) ** 2)
    print(f"\nMSE vs ground truth: {mse:.6f}")

    return result


def main():
    """Run all examples."""
    print("Chronos-2 Inference Examples")
    print("============================\n")

    # Check if model loading works
    try:
        from tsmbrl.models.model_registry import get_model

        model = get_model("chronos2-small", device="cpu")
        print("Model loading test: SUCCESS\n")
    except Exception as e:
        print(f"Model loading test failed: {e}")
        print("Make sure chronos-forecasting is installed.\n")
        return

    # Run examples
    try:
        example_point_prediction()
    except Exception as e:
        print(f"Point prediction example failed: {e}")

    try:
        example_probabilistic()
    except Exception as e:
        print(f"Probabilistic example failed: {e}")

    try:
        example_multivariate()
    except Exception as e:
        print(f"Multivariate example failed: {e}")

    try:
        example_with_covariates()
    except Exception as e:
        print(f"Covariates example failed: {e}")

    print("\n" + "=" * 50)
    print("All examples complete!")
    print("=" * 50)


if __name__ == "__main__":
    main()
