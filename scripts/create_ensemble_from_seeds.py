#!/usr/bin/env python3
"""Create ensemble predictions from multiple seed runs of the same model.

This script reads prediction parquet files from multiple seed runs and creates
an ensemble by taking the mean of positive predictions. If all predictions
for a given row are negative or zero, the ensemble predicts 0.

Adapted from create_ensemble_median_positive.py to work with seeds instead of models.
"""

import argparse
from pathlib import Path

import polars as pl


def create_seed_ensemble(
    eval_dir: Path,
    model_name: str,
    seeds: list[int],
    output_dir: Path | None = None,
    ensemble_name: str = "ensemble",
) -> None:
    """Create ensemble using mean of positive predictions across seeds.

    For each row:
    - Filter to positive predictions only
    - Take mean of positive values
    - If no positive values exist, predict 0

    Args:
        eval_dir: Base evaluation directory (e.g., results/evaluation/eval_*)
        model_name: Name of the model to ensemble
        seeds: List of seed numbers to include in ensemble
        output_dir: Optional custom output directory (default: same as eval_dir)
        ensemble_name: Name for the ensemble model (default: "ensemble")

    Raises:
        ValueError: If seeds list is empty or prediction files not found
        FileNotFoundError: If evaluation directory doesn't exist
    """
    if not seeds:
        raise ValueError("At least one seed must be provided")

    if not eval_dir.exists():
        raise FileNotFoundError(f"Evaluation directory not found: {eval_dir}")

    # Construct paths to seed prediction files
    input_paths = []
    for seed in seeds:
        path = eval_dir / f"model_name={model_name}" / f"seed={seed}" / "predictions.parquet"
        if not path.exists():
            raise FileNotFoundError(
                f"Prediction file not found for seed {seed}: {path}"
            )
        input_paths.append(path)

    print(f"Loading predictions from {len(input_paths)} seeds...")

    # Read all prediction files
    dfs = [pl.read_parquet(path) for path in input_paths]

    # Verify all dataframes have the same shape
    shapes = [df.shape for df in dfs]
    if len(set(shapes)) > 1:
        raise ValueError(f"Input files have different shapes: {shapes}")

    # Take all columns from first dataframe
    result = dfs[0].clone()

    # Stack all predictions into a matrix (rows x seeds)
    n_rows = dfs[0].height
    n_seeds = len(dfs)

    print(f"Computing mean-of-positives ensemble for {n_rows} predictions...")

    # Create a DataFrame with one column per seed
    stacked = pl.DataFrame(
        {f"seed_{i}": df["prediction"] for i, df in enumerate(dfs)}
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
        pl.struct([f"seed_{i}" for i in range(n_seeds)])
        .map_elements(mean_of_positives, return_dtype=pl.Float64)
        .alias("prediction")
    )

    # Replace prediction column with ensemble values
    result = result.with_columns(ensemble_predictions)

    # Update model_name to ensemble name
    result = result.with_columns(pl.lit(ensemble_name).alias("model_name"))

    # Determine output path
    if output_dir is None:
        output_dir = eval_dir

    output_path = (
        output_dir / f"model_name={ensemble_name}" / "seed=ensemble" / "predictions.parquet"
    )

    # Create output directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save ensemble predictions
    result.write_parquet(output_path)

    # Calculate stats
    n_zeros = (result["prediction"] == 0.0).sum()
    pct_zeros = 100 * n_zeros / n_rows

    print(f"✓ Created mean-of-positives ensemble from {n_seeds} seeds")
    print(f"✓ Source model: {model_name}")
    print(f"✓ Seeds: {seeds}")
    print(f"✓ Saved to: {output_path}")
    print(f"✓ Shape: {result.shape}")
    print(f"✓ Predictions set to 0 (all seeds negative): {n_zeros} ({pct_zeros:.2f}%)")


def discover_eval_dir(base_dir: Path = Path("results/evaluation")) -> Path:
    """Find the most recent evaluation directory.

    Args:
        base_dir: Base directory containing evaluation results

    Returns:
        Path to most recent eval_* directory

    Raises:
        FileNotFoundError: If no evaluation directories found
    """
    eval_dirs = sorted(base_dir.glob("eval_*"))
    if not eval_dirs:
        raise FileNotFoundError(f"No evaluation directories found in {base_dir}")

    # Return most recent (last when sorted)
    return eval_dirs[-1]


def discover_seeds(eval_dir: Path, model_name: str) -> list[int]:
    """Discover available seeds for a given model.

    Args:
        eval_dir: Evaluation directory
        model_name: Model name

    Returns:
        Sorted list of seed numbers

    Raises:
        FileNotFoundError: If no seeds found for model
    """
    model_dir = eval_dir / f"model_name={model_name}"
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    # Find all seed directories
    seed_dirs = list(model_dir.glob("seed=*"))
    if not seed_dirs:
        raise FileNotFoundError(f"No seed directories found for {model_name}")

    # Extract seed numbers
    seeds = []
    for seed_dir in seed_dirs:
        seed_str = seed_dir.name.replace("seed=", "")
        try:
            seeds.append(int(seed_str))
        except ValueError:
            # Skip non-numeric seeds (e.g., "seed=ensemble")
            continue

    if not seeds:
        raise FileNotFoundError(f"No numeric seeds found for {model_name}")

    return sorted(seeds)


def main() -> None:
    """Parse arguments and create ensemble predictions."""
    parser = argparse.ArgumentParser(
        description="Create ensemble from multiple seed runs of the same model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:

  # Auto-discover most recent evaluation and all seeds
  python scripts/create_ensemble_from_seeds.py --model-name tft_kyrgyzstan

  # Specify evaluation directory and seeds
  python scripts/create_ensemble_from_seeds.py \\
    --eval-dir results/evaluation/eval_2025-01-23_143052 \\
    --model-name tft_kyrgyzstan \\
    --seeds 42,43,44,45,46,47,48,49,50,51

  # Custom ensemble name
  python scripts/create_ensemble_from_seeds.py \\
    --model-name tft_kyrgyzstan \\
    --ensemble-name tft_mean_ensemble
        """,
    )

    parser.add_argument(
        "--eval-dir",
        type=Path,
        help="Evaluation directory (default: auto-discover most recent)",
    )
    parser.add_argument(
        "--model-name",
        required=True,
        help="Model name to create ensemble from",
    )
    parser.add_argument(
        "--seeds",
        help="Comma-separated list of seeds (default: auto-discover all seeds)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory (default: same as eval-dir)",
    )
    parser.add_argument(
        "--ensemble-name",
        default="ensemble",
        help="Name for the ensemble model (default: ensemble)",
    )

    args = parser.parse_args()

    # Auto-discover evaluation directory if not provided
    if args.eval_dir is None:
        print("Auto-discovering evaluation directory...")
        eval_dir = discover_eval_dir()
        print(f"✓ Found: {eval_dir}")
    else:
        eval_dir = args.eval_dir

    # Auto-discover seeds if not provided
    if args.seeds is None:
        print(f"Auto-discovering seeds for {args.model_name}...")
        seeds = discover_seeds(eval_dir, args.model_name)
        print(f"✓ Found seeds: {seeds}")
    else:
        seeds = [int(s.strip()) for s in args.seeds.split(",")]

    # Create ensemble
    create_seed_ensemble(
        eval_dir=eval_dir,
        model_name=args.model_name,
        seeds=seeds,
        output_dir=args.output_dir,
        ensemble_name=args.ensemble_name,
    )


if __name__ == "__main__":
    main()
