#!/usr/bin/env python3
"""
Chronos-2 specific example showing different usage patterns.

This demonstrates:
1. Tensor API for univariate prediction
2. Multivariate prediction
3. Action-conditioned prediction for MBRL
"""

import numpy as np
import torch


def example_tensor_api():
    """Demonstrate tensor API usage."""
    print("=" * 50)
    print("Example 1: Tensor API")
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

    # Predict
    context = torch.from_numpy(series[:80]).float()
    result = model.predict_probabilistic(
        context=context,
        prediction_length=20,
        quantile_levels=[0.1, 0.5, 0.9],
    )

    print(f"Context shape: {context.shape}")
    print(f"Mean predictions shape: {result['mean'].shape}")
    print(f"Quantiles shape: {result['quantiles'].shape}")

    # Compute error on ground truth
    ground_truth = series[80:100]
    mse = np.mean((result["mean"].flatten() - ground_truth) ** 2)
    print(f"MSE on held-out data: {mse:.6f}")

    return result


def example_multivariate():
    """Demonstrate multivariate prediction."""
    print("\n" + "=" * 50)
    print("Example 2: Multivariate Prediction")
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

    # Predict
    context = data[:80]
    result = model.predict_multivariate(
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


def example_with_actions():
    """Demonstrate action-conditioned prediction for MBRL."""
    print("\n" + "=" * 50)
    print("Example 3: Action-Conditioned Prediction (MBRL)")
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
    context_actions = np.random.randn(lookback, act_dim)
    future_actions = np.random.randn(horizon, act_dim)
    target_obs = np.random.randn(horizon, obs_dim)  # Ground truth

    # Predict
    result = model.predict_with_actions(
        context_obs=context_obs,
        context_actions=context_actions,
        future_actions=future_actions,
        prediction_length=horizon,
        quantile_levels=[0.1, 0.5, 0.9],
    )

    print(f"Context observations: {context_obs.shape}")
    print(f"Context actions: {context_actions.shape}")
    print(f"Future actions: {future_actions.shape}")
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

        model = get_model("chronos2-tiny", device="cpu")
        print("Model loading test: SUCCESS\n")
    except Exception as e:
        print(f"Model loading test failed: {e}")
        print("Make sure chronos-forecasting is installed.\n")
        return

    # Run examples
    try:
        example_tensor_api()
    except Exception as e:
        print(f"Tensor API example failed: {e}")

    try:
        example_multivariate()
    except Exception as e:
        print(f"Multivariate example failed: {e}")

    try:
        example_with_actions()
    except Exception as e:
        print(f"Action-conditioned example failed: {e}")

    print("\n" + "=" * 50)
    print("All examples complete!")
    print("=" * 50)


if __name__ == "__main__":
    main()
