#!/usr/bin/env python3
"""
Main inference script for TSMBRL experiments.

Usage:
    python -m tsmbrl.inference --dataset door-human --model chronos2 --horizon 10

    # With actions as covariates
    python -m tsmbrl.inference --dataset door-human --model chronos2 --with-actions

    # Save results to file
    python -m tsmbrl.inference --dataset door-human -o results/experiment.json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
from tqdm import tqdm

from tsmbrl.config.experiment_config import ExperimentConfig
from tsmbrl.data.dataset_registry import get_dataset_id, list_datasets
from tsmbrl.data.formatters import TimeSeriesFormatter, prepare_evaluation_data
from tsmbrl.data.minari_loader import MinariDataLoader
from tsmbrl.metrics.forecasting_metrics import multi_step_metrics, per_dimension_metrics
from tsmbrl.metrics.probabilistic_metrics import compute_all_probabilistic_metrics
from tsmbrl.models.model_registry import get_model, list_models
from tsmbrl.utils.file_utils import save_results
from tsmbrl.utils.logging_utils import setup_logger

logger = setup_logger("tsmbrl.inference")


def run_inference(config: ExperimentConfig) -> Dict[str, Any]:
    """
    Run a complete inference experiment.

    Args:
        config: Experiment configuration

    Returns:
        Dictionary with predictions, metrics, and metadata
    """
    logger.info(f"Starting experiment: {config.dataset_name} / {config.model_name}")
    logger.info(f"  Lookback: {config.lookback}, Horizon: {config.horizon}")
    logger.info(f"  Use actions: {config.use_actions_as_covariates}")

    # 1. Load dataset
    logger.info(f"Loading dataset: {config.dataset_name}")
    dataset_id = get_dataset_id(config.dataset_name)
    loader = MinariDataLoader(dataset_id, download=True)

    logger.info(f"  Total episodes: {loader.total_episodes}")
    logger.info(f"  Obs dim: {loader.obs_dim}, Act dim: {loader.act_dim}")

    # 2. Create formatter
    formatter = TimeSeriesFormatter(
        lookback=config.lookback,
        horizon=config.horizon,
        obs_dim=loader.obs_dim,
        act_dim=loader.act_dim,
    )

    # 3. Prepare evaluation windows
    logger.info("Creating evaluation windows...")
    windows = prepare_evaluation_data(
        loader,
        formatter,
        max_episodes=config.max_episodes,
        windows_per_episode=config.windows_per_episode,
        seed=config.seed,
    )
    logger.info(f"  Created {len(windows)} evaluation windows")

    if len(windows) == 0:
        raise ValueError(
            "No valid evaluation windows created. "
            "Check lookback/horizon settings relative to episode lengths."
        )

    # 4. Load model
    logger.info(f"Loading model: {config.model_name}")
    model = get_model(config.model_name, device=config.device)
    logger.info(f"  Supports covariates: {model.supports_covariates}")
    logger.info(f"  Is probabilistic: {model.is_probabilistic}")

    # 5. Run predictions
    logger.info("Running predictions...")
    all_predictions = []
    all_targets = []
    all_quantiles = []

    for window in tqdm(windows, desc="Predicting"):
        # Use action-conditioned prediction if enabled
        result = model.predict_with_actions(
            context_obs=window.context_observations,
            context_actions=window.context_actions,
            future_actions=window.future_actions,
            prediction_length=config.horizon,
            quantile_levels=config.quantile_levels,
        )

        # Collect predictions
        mean_pred = result["mean"]  # Shape: (horizon, obs_dim) or similar

        # Ensure consistent shape: (horizon, obs_dim)
        if mean_pred.ndim == 1:
            mean_pred = mean_pred.reshape(-1, 1)

        all_predictions.append(mean_pred)
        all_targets.append(window.target_observations)

        if "quantiles" in result:
            all_quantiles.append(result["quantiles"])

    # 6. Stack and compute metrics
    predictions = np.stack(all_predictions)  # (n_windows, horizon, obs_dim)
    targets = np.stack(all_targets)

    logger.info("Computing metrics...")

    # Forecasting metrics
    forecast_metrics = multi_step_metrics(predictions, targets)
    dim_metrics = per_dimension_metrics(predictions, targets)

    results: Dict[str, Any] = {
        "metrics": {
            **forecast_metrics,
            **dim_metrics,
        },
        "metadata": {
            "dataset": config.dataset_name,
            "dataset_id": dataset_id,
            "model": config.model_name,
            "lookback": config.lookback,
            "horizon": config.horizon,
            "use_actions": config.use_actions_as_covariates,
            "n_windows": len(windows),
            "obs_dim": loader.obs_dim,
            "act_dim": loader.act_dim,
            "quantile_levels": config.quantile_levels,
            "seed": config.seed,
        },
    }

    # Probabilistic metrics (if available)
    if all_quantiles and config.compute_probabilistic_metrics:
        try:
            quantiles = np.stack(all_quantiles)
            prob_metrics = compute_all_probabilistic_metrics(
                quantiles, config.quantile_levels, targets
            )
            results["metrics"].update(prob_metrics)
        except Exception as e:
            logger.warning(f"Failed to compute probabilistic metrics: {e}")

    # Optionally save predictions
    if config.save_predictions:
        results["predictions"] = predictions.tolist()
        results["targets"] = targets.tolist()

    # Log summary
    logger.info("=" * 50)
    logger.info("RESULTS SUMMARY")
    logger.info("=" * 50)
    logger.info(f"  MSE:  {results['metrics']['mse']:.6f}")
    logger.info(f"  MAE:  {results['metrics']['mae']:.6f}")
    logger.info(f"  RMSE: {results['metrics']['rmse']:.6f}")
    if "crps" in results["metrics"]:
        logger.info(f"  CRPS: {results['metrics']['crps']:.6f}")
    logger.info("=" * 50)

    return results


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Run TSFM inference on MBRL datasets",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Dataset arguments
    parser.add_argument(
        "--dataset",
        "-d",
        type=str,
        required=True,
        help=f"Dataset name. Available: {', '.join(list_datasets())}",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="Maximum episodes to process (None for all)",
    )
    parser.add_argument(
        "--windows-per-episode",
        type=int,
        default=20,
        help="Maximum windows per episode",
    )

    # Model arguments
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="chronos2",
        help=f"Model name. Available: {', '.join(list_models())}",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device for inference",
    )

    # Forecasting arguments
    parser.add_argument(
        "--lookback",
        "-l",
        type=int,
        default=50,
        help="Lookback window size (context length)",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=10,
        help="Prediction horizon",
    )
    parser.add_argument(
        "--with-actions",
        action="store_true",
        help="Use actions as covariates (experimental)",
    )
    parser.add_argument(
        "--no-actions",
        action="store_true",
        help="Do not use actions (baseline)",
    )

    # Output arguments
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output file path for results JSON",
    )
    parser.add_argument(
        "--save-predictions",
        action="store_true",
        help="Save raw predictions in output (increases file size)",
    )

    # Misc arguments
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--quantiles",
        type=float,
        nargs="+",
        default=[0.1, 0.5, 0.9],
        help="Quantile levels for probabilistic forecasting",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose output",
    )

    args = parser.parse_args()

    # Set random seed
    np.random.seed(args.seed)

    # Build config
    config = ExperimentConfig(
        dataset_name=args.dataset,
        max_episodes=args.max_episodes,
        windows_per_episode=args.windows_per_episode,
        model_name=args.model,
        device=args.device,
        lookback=args.lookback,
        horizon=args.horizon,
        use_actions_as_covariates=args.with_actions and not args.no_actions,
        quantile_levels=args.quantiles,
        save_predictions=args.save_predictions,
        seed=args.seed,
    )

    # Run experiment
    try:
        results = run_inference(config)
    except Exception as e:
        logger.error(f"Experiment failed: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)

    # Save or print results
    if args.output:
        output_path = Path(args.output)
        save_results(results, output_path)
        logger.info(f"Results saved to: {output_path}")
    else:
        # Print to stdout
        print(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    main()
