#!/usr/bin/env python3
"""
Aggregate results from multiple experiments into a summary CSV.

Usage:
    python scripts/aggregate_results.py --input results/raw/
        --output results/summary.csv
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Any

import pandas as pd


def load_experiment_results(results_dir: Path) -> List[Dict[str, Any]]:
    """Load all JSON result files from a directory."""
    results = []

    for json_file in results_dir.glob("*.json"):
        try:
            with open(json_file, "r") as f:
                data = json.load(f)

            # Extract metadata and metrics
            result = {
                "file": json_file.name,
                **data.get("metadata", {}),
                **data.get("metrics", {}),
            }

            # Flatten nested structures
            if "calibration" in result:
                cal = result.pop("calibration")
                for q, err in cal.items():
                    result[f"calibration_{q}"] = err

            # Remove list-valued metrics (per-step, per-dim)
            result = {k: v for k, v in result.items() if not isinstance(v, list)}

            results.append(result)

        except Exception as e:
            print(f"Warning: Failed to load {json_file}: {e}")

    return results


def create_summary_dataframe(results: List[Dict[str, Any]]) -> pd.DataFrame:
    """Create a summary DataFrame from results."""
    df = pd.DataFrame(results)

    # Reorder columns
    priority_cols = [
        "dataset",
        "model",
        "horizon",
        "lookback",
        "use_actions",
        "mse",
        "mae",
        "rmse",
        "crps",
        "coverage_80",
        "n_windows",
    ]

    # Get columns that exist
    ordered_cols = [c for c in priority_cols if c in df.columns]
    other_cols = [c for c in df.columns if c not in priority_cols]

    df = df[ordered_cols + other_cols]

    return df


def compute_aggregated_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Compute aggregated statistics across configurations."""
    # Group by dataset, model, use_actions
    group_cols = ["dataset", "model", "use_actions"]

    if all(c in df.columns for c in group_cols):
        agg_df = (
            df.groupby(group_cols)
            .agg(
                {
                    "mse": ["mean", "std"],
                    "mae": ["mean", "std"],
                    "rmse": ["mean", "std"],
                }
            )
            .round(6)
        )

        # Flatten column names
        agg_df.columns = ["_".join(col).strip() for col in agg_df.columns.values]
        agg_df = agg_df.reset_index()

        return agg_df

    return df


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate experiment results into summary CSV"
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        default="results/raw",
        help="Directory containing JSON result files",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="results/summary.csv",
        help="Output CSV file path",
    )
    parser.add_argument(
        "--aggregated",
        "-a",
        type=str,
        default=None,
        help="Output aggregated statistics CSV (optional)",
    )
    parser.add_argument(
        "--print",
        "-p",
        action="store_true",
        help="Print summary to console",
    )

    args = parser.parse_args()

    # Load results
    results_dir = Path(args.input)
    if not results_dir.exists():
        print(f"Error: Results directory not found: {results_dir}")
        return

    print(f"Loading results from: {results_dir}")
    results = load_experiment_results(results_dir)
    print(f"Loaded {len(results)} experiments")

    if len(results) == 0:
        print("No results found!")
        return

    # Create DataFrame
    df = create_summary_dataframe(results)

    # Save full results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved summary to: {output_path}")

    # Save aggregated stats if requested
    if args.aggregated:
        agg_df = compute_aggregated_stats(df)
        agg_path = Path(args.aggregated)
        agg_df.to_csv(agg_path, index=False)
        print(f"Saved aggregated stats to: {agg_path}")

    # Print summary
    if args.print:
        print("\n" + "=" * 60)
        print("RESULTS SUMMARY")
        print("=" * 60)
        print(df.to_string())

        # Print key metrics comparison
        if "use_actions" in df.columns:
            print("\n" + "-" * 60)
            print("With Actions vs Without Actions (MSE)")
            print("-" * 60)

            for dataset in df["dataset"].unique():
                subset = df[df["dataset"] == dataset]
                with_actions = subset[subset["use_actions"] is True]["mse"].mean()
                without_actions = subset[subset["use_actions"] is False]["mse"].mean()

                if pd.notna(with_actions) and pd.notna(without_actions):
                    improvement = (
                        (without_actions - with_actions) / without_actions * 100
                    )
                    print(
                        f"  {dataset}: "
                        f"with={with_actions:.6f}, "
                        f"without={without_actions:.6f}, "
                        f"improvement={improvement:+.1f}%"
                    )


if __name__ == "__main__":
    main()
