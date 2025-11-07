#!/usr/bin/env python3
"""Create ensemble predictions using mean of positive predictions.

This script reads prediction parquet files from multiple models and creates
an ensemble by taking the mean of positive predictions. If all predictions
for a given row are negative or zero, the ensemble predicts 0.
"""

import argparse
from pathlib import Path

import polars as pl


def create_mean_positive_ensemble(
    input_paths: list[Path],
    output_path: Path,
    ensemble_name: str = "ensemble",
) -> None:
    """Create ensemble using mean of positive predictions.

    For each row:
    - Filter to positive predictions only
    - Take mean of positive values
    - If no positive values exist, predict 0

    Args:
        input_paths: List of paths to prediction parquet files
        output_path: Path where ensemble predictions will be saved
        ensemble_name: Name for the ensemble model (default: "ensemble")

    Raises:
        ValueError: If input_paths is empty or files have mismatched schemas
    """
    if not input_paths:
        raise ValueError("At least one input path must be provided")

    # Read all prediction files
    dfs = [pl.read_parquet(path) for path in input_paths]

    # Verify all dataframes have the same shape
    shapes = [df.shape for df in dfs]
    if len(set(shapes)) > 1:
        raise ValueError(f"Input files have different shapes: {shapes}")

    # Take all columns from first dataframe
    result = dfs[0].clone()

    # Stack all predictions into a matrix (rows x models)
    # We'll process row by row to apply the median-of-positives logic
    n_rows = dfs[0].height
    n_models = len(dfs)

    # Create a DataFrame with one column per model
    stacked = pl.DataFrame(
        {f"model_{i}": df["prediction"] for i, df in enumerate(dfs)}
    )

    # For each row, filter positive values and take mean (or 0 if none positive)
    def mean_of_positives(row: dict) -> float:
        """Calculate mean of positive values, or 0 if all negative/zero."""
        values = [v for v in row.values() if v is not None and v > 0]
        if not values:
            return 0.0
        return float(pl.Series(values).mean())

    # Apply row-wise
    ensemble_predictions = stacked.select(
        pl.struct([f"model_{i}" for i in range(n_models)])
        .map_elements(mean_of_positives, return_dtype=pl.Float64)
        .alias("prediction")
    )

    # Replace prediction column with ensemble values
    result = result.with_columns(ensemble_predictions)

    # Update model_name to ensemble name
    result = result.with_columns(pl.lit(ensemble_name).alias("model_name"))

    # Create output directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save ensemble predictions
    result.write_parquet(output_path)

    # Calculate stats
    n_zeros = (result["prediction"] == 0.0).sum()
    pct_zeros = 100 * n_zeros / n_rows

    print(f"✓ Created mean-of-positives ensemble from {n_models} models")
    print(f"✓ Saved to: {output_path}")
    print(f"✓ Shape: {result.shape}")
    print(f"✓ Predictions set to 0 (all models negative): {n_zeros} ({pct_zeros:.2f}%)")


def main() -> None:
    """Parse arguments and create ensemble predictions."""
    parser = argparse.ArgumentParser(
        description="Create ensemble using mean of positive predictions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python scripts/create_ensemble_median_positive.py \\
    --input results/eval_*/model_name=model1/seed=42/predictions.parquet \\
            results/eval_*/model_name=model2/seed=42/predictions.parquet \\
            results/eval_*/model_name=model3/seed=42/predictions.parquet \\
    --output results/eval_*/model_name=ensemble/seed=42/predictions.parquet
        """,
    )

    parser.add_argument(
        "--input",
        nargs="+",
        required=True,
        type=Path,
        help="Input prediction parquet files (space-separated)",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Output path for ensemble predictions",
    )
    parser.add_argument(
        "--ensemble-name",
        default="ensemble",
        help="Name for the ensemble model (default: ensemble)",
    )

    args = parser.parse_args()

    create_mean_positive_ensemble(
        input_paths=args.input,
        output_path=args.output,
        ensemble_name=args.ensemble_name,
    )


if __name__ == "__main__":
    main()
