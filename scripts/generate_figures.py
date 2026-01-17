#!/usr/bin/env python3
"""
Generate publication-quality figures from experiment results.

Usage:
    python scripts/generate_figures.py --results results/summary.csv
        --output results/figures/
"""

import argparse
from pathlib import Path

import pandas as pd
# import numpy as np


def load_results(results_path: Path) -> pd.DataFrame:
    """Load results from CSV."""
    return pd.read_csv(results_path)


def plot_horizon_comparison(df: pd.DataFrame, output_dir: Path):
    """Plot MSE vs horizon for different configurations."""
    from tsmbrl.visualization.plot_predictions import plot_multi_horizon_mse

    # Group by model and horizon
    horizon_results = {}

    for model in df["model"].unique():
        model_df = df[df["model"] == model]
        horizons = sorted(model_df["horizon"].unique())

        mse_values = []
        for h in horizons:
            mse = model_df[model_df["horizon"] == h]["mse"].mean()
            mse_values.append(mse)

        horizon_results[model] = mse_values

    if horizon_results:
        plot_multi_horizon_mse(
            mse_per_step=horizon_results,
            title="MSE vs Prediction Horizon",
            save_path=output_dir / "mse_vs_horizon.png",
            show=False,
        )
        print("Saved: mse_vs_horizon.png")


def plot_dataset_comparison(df: pd.DataFrame, output_dir: Path):
    """Plot comparison across datasets."""
    from tsmbrl.visualization.plot_comparisons import plot_dataset_comparison

    # Average MSE per dataset
    dataset_results = {}
    for dataset in df["dataset"].unique():
        mse = df[df["dataset"] == dataset]["mse"].mean()
        dataset_results[dataset] = {"mse": mse}

    if dataset_results:
        plot_dataset_comparison(
            results=dataset_results,
            metric="mse",
            title="MSE Across Datasets",
            save_path=output_dir / "dataset_comparison.png",
            show=False,
        )
        print("Saved: dataset_comparison.png")


def plot_actions_ablation(df: pd.DataFrame, output_dir: Path):
    """Plot with/without actions comparison."""
    from tsmbrl.visualization.plot_comparisons import plot_actions_ablation

    if "use_actions" not in df.columns:
        print("  Skipping actions ablation: no use_actions column")
        return

    # Aggregate metrics
    with_actions = (
        df[df["use_actions"] is True][["mse", "mae", "rmse"]].mean().to_dict()
    )
    without_actions = (
        df[df["use_actions"] is False][["mse", "mae", "rmse"]].mean().to_dict()
    )

    if with_actions and without_actions:
        plot_actions_ablation(
            with_actions=with_actions,
            without_actions=without_actions,
            metrics=["mse", "mae", "rmse"],
            title="Effect of Action Covariates",
            save_path=output_dir / "actions_ablation.png",
            show=False,
        )
        print("Saved: actions_ablation.png")


def plot_model_comparison(df: pd.DataFrame, output_dir: Path):
    """Plot comparison across models."""
    from tsmbrl.visualization.plot_comparisons import plot_aggregated_results

    # Aggregate metrics per model
    model_results = {}
    for model in df["model"].unique():
        model_df = df[df["model"] == model]
        model_results[model] = {
            "mse": model_df["mse"].mean(),
            "mae": model_df["mae"].mean(),
            "rmse": model_df["rmse"].mean(),
        }

        if "crps" in model_df.columns:
            model_results[model]["crps"] = model_df["crps"].mean()

    if len(model_results) > 1:
        metrics = ["mse", "mae", "rmse"]
        if all("crps" in v for v in model_results.values()):
            metrics.append("crps")

        plot_aggregated_results(
            results=model_results,
            metrics=metrics,
            title="Model Comparison",
            save_path=output_dir / "model_comparison.png",
            show=False,
        )
        print("Saved: model_comparison.png")


def create_summary_table(df: pd.DataFrame, output_dir: Path):
    """Create summary statistics table."""
    # Group by key variables
    summary = df.groupby(["dataset", "model", "use_actions"]).agg(
        {
            "mse": ["mean", "std"],
            "mae": ["mean", "std"],
        }
    )

    summary.columns = ["_".join(col) for col in summary.columns]
    summary = summary.round(6)

    # Save as CSV
    summary.to_csv(output_dir / "summary_table.csv")
    print("Saved: summary_table.csv")

    # Also save as LaTeX
    latex_str = summary.to_latex(
        float_format="%.4f",
        caption="Experiment Results Summary",
        label="tab:results",
    )
    with open(output_dir / "summary_table.tex", "w") as f:
        f.write(latex_str)
    print("Saved: summary_table.tex")


def main():
    parser = argparse.ArgumentParser(
        description="Generate figures from experiment results"
    )
    parser.add_argument(
        "--results",
        "-r",
        type=str,
        default="results/summary.csv",
        help="Path to results CSV file",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="results/figures",
        help="Output directory for figures",
    )

    args = parser.parse_args()

    # Load results
    results_path = Path(args.results)
    if not results_path.exists():
        print(f"Error: Results file not found: {results_path}")
        print("Run experiments first with: bash scripts/run_experiments.sh")
        return

    print(f"Loading results from: {results_path}")
    df = load_results(results_path)
    print(f"Loaded {len(df)} experiment results")

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving figures to: {output_dir}")

    # Generate figures
    print("\nGenerating figures...")

    try:
        plot_horizon_comparison(df, output_dir)
    except Exception as e:
        print(f"  Failed horizon plot: {e}")

    try:
        plot_dataset_comparison(df, output_dir)
    except Exception as e:
        print(f"  Failed dataset plot: {e}")

    try:
        plot_actions_ablation(df, output_dir)
    except Exception as e:
        print(f"  Failed ablation plot: {e}")

    try:
        plot_model_comparison(df, output_dir)
    except Exception as e:
        print(f"  Failed model comparison: {e}")

    try:
        create_summary_table(df, output_dir)
    except Exception as e:
        print(f"  Failed summary table: {e}")

    print("\nFigure generation complete!")


if __name__ == "__main__":
    main()
