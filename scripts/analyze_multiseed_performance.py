#!/usr/bin/env python3
"""Analyze multi-seed model performance and ensemble comparison.

This script analyzes the performance distribution across multiple seed runs
and compares individual seeds to the ensemble model. It produces:

1. CDF plots of NSE across basins for lead times 1, 5, and 10
2. Rolling forecast plot showing all seeds + ensemble for a median-performing basin

Usage:
    python scripts/analyze_multiseed_performance.py \\
        --eval-dir results/evaluation/eval_2025-01-23_143052 \\
        --model-name tft_kyrgyzstan \\
        --ensemble-name ensemble
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
from transfer_learning_publication.evaluation import MetricCalculator

# Set style
sns.set_context("paper", font_scale=1.3)
sns.set_style("whitegrid")


def load_all_predictions(
    eval_dir: Path, model_name: str, ensemble_name: str
) -> tuple[pl.LazyFrame, list[int]]:
    """Load all seed predictions and ensemble predictions.

    Args:
        eval_dir: Evaluation directory
        model_name: Base model name
        ensemble_name: Ensemble model name

    Returns:
        Tuple of (all predictions LazyFrame, list of seed numbers)
    """
    # Extract available seeds
    seed_dirs = list((eval_dir / f"model_name={model_name}").glob("seed=*"))
    seeds = []
    for seed_dir in seed_dirs:
        seed_str = seed_dir.name.replace("seed=", "")
        try:
            seeds.append(int(seed_str))
        except ValueError:
            continue
    seeds = sorted(seeds)

    # Load each seed file individually and add seed column
    seed_frames = []
    for seed in seeds:
        path = eval_dir / f"model_name={model_name}" / f"seed={seed}" / "predictions.parquet"
        df = pl.scan_parquet(path).with_columns(pl.lit(seed).cast(pl.Int64).alias("seed"))
        seed_frames.append(df)

    # Concatenate all seed data
    seed_data = pl.concat(seed_frames)

    # Load ensemble predictions
    ensemble_path = (
        eval_dir / f"model_name={ensemble_name}" / "seed=ensemble" / "predictions.parquet"
    )
    ensemble_data = pl.scan_parquet(ensemble_path)

    # Add seed column to ensemble (use -1 to distinguish)
    ensemble_data = ensemble_data.with_columns(pl.lit(-1).cast(pl.Int64).alias("seed"))

    # Combine all data
    all_data = pl.concat([seed_data, ensemble_data])

    return all_data, seeds


def compute_metrics_by_seed(
    data: pl.LazyFrame, lead_times: list[int] | None = None
) -> pl.DataFrame:
    """Compute NSE metrics grouped by model, seed, basin, and lead_time.

    Args:
        data: LazyFrame with predictions
        lead_times: Optional list of lead times to filter

    Returns:
        DataFrame with NSE metrics
    """
    calc = MetricCalculator(data)

    metrics = calc.compute_metrics(
        metrics=["nse"],
        group_by=["model_name", "seed", "group_identifier", "lead_time"],
        exclude_filled=True,
        lead_times=lead_times,
    )

    return metrics


def plot_boxplot_comparison(
    metrics: pl.DataFrame,
    lead_time: int,
    seeds: list[int],
    ensemble_name: str,
    output_path: Path,
    color: str = "steelblue",
) -> None:
    """Plot boxplots of NSE values across basins for a given lead time.

    Args:
        metrics: DataFrame with NSE metrics
        lead_time: Lead time to plot
        seeds: List of seed numbers
        ensemble_name: Name of ensemble model
        output_path: Path to save figure
        color: Color for the boxes
    """
    # Filter to specific lead time
    lead_data = metrics.filter(pl.col("lead_time") == lead_time)

    # Prepare data for boxplot
    # Each model (10 seeds + 1 ensemble) gets NSE values across all basins
    plot_data = []
    labels = []

    # Add individual seeds
    for i, seed in enumerate(seeds):
        seed_subset = lead_data.filter(pl.col("seed") == seed)
        nse_values = seed_subset["NSE"].to_numpy()
        # Remove NaN values
        nse_values = nse_values[~np.isnan(nse_values)]
        plot_data.append(nse_values)
        labels.append(str(i + 1))  # Label as 1, 2, 3, ... for seeds

    # Add ensemble
    ensemble_subset = lead_data.filter(pl.col("seed") == -1)
    ensemble_nse = ensemble_subset["NSE"].to_numpy()
    ensemble_nse = ensemble_nse[~np.isnan(ensemble_nse)]
    plot_data.append(ensemble_nse)
    labels.append("Ens")  # Label as "Ens" for ensemble

    # Create plot
    fig, ax = plt.subplots(figsize=(12, 6))

    # Create boxplots
    bp = ax.boxplot(
        plot_data,
        labels=labels,
        patch_artist=True,
        widths=0.6,
        showfliers=False,  # Don't show outliers for cleaner plot
    )

    # Color boxes
    for i, patch in enumerate(bp['boxes']):
        if i < len(seeds):
            # Individual seeds - lighter color
            patch.set_facecolor(color)
            patch.set_alpha(0.5)
        else:
            # Ensemble - darker color
            patch.set_facecolor(color)
            patch.set_alpha(1.0)
            patch.set_edgecolor('black')
            patch.set_linewidth(2)

    # Color whiskers, caps, medians
    for element in ['whiskers', 'caps', 'medians']:
        for item in bp[element]:
            item.set_color('black')
            item.set_linewidth(1.5 if element == 'medians' else 1)

    # Add median value annotations
    for i, (data, label) in enumerate(zip(plot_data, labels)):
        if len(data) > 0:
            median_val = np.median(data)
            # Position text slightly above the median line
            y_pos = median_val + 0.003
            ax.text(
                i + 1,
                y_pos,
                f'{median_val:.3f}',
                ha='center',
                va='bottom',
                fontsize=9,
                fontweight='bold' if label == 'Ens' else 'normal',
            )

    # Set y-axis limits based on lead time
    if lead_time == 1:
        ylim = (0.95, 1.0)
    elif lead_time == 5:
        ylim = (0.88, 1.0)
    else:  # lead_time == 10
        ylim = (0.78, 1.0)

    # Formatting
    ax.set_xlabel("Model", fontsize=14)
    ax.set_ylabel("NSE", fontsize=14)
    ax.set_title(f"NSE Distribution - {lead_time}-day Forecast", fontsize=16)
    ax.set_ylim(ylim)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"✓ Saved boxplot: {output_path}")


def select_median_basin(
    metrics: pl.DataFrame, model_name: str, seeds: list[int]
) -> str:
    """Select basin with median NSE performance averaged across seeds.

    Args:
        metrics: DataFrame with NSE metrics
        model_name: Model name
        seeds: List of seed numbers

    Returns:
        Basin ID with median performance
    """
    # Filter to individual seeds only (exclude ensemble)
    seed_metrics = metrics.filter(
        (pl.col("model_name") == model_name) & (pl.col("seed").is_in(seeds))
    )

    # Average NSE across all seeds and lead times for each basin
    basin_avg = (
        seed_metrics.group_by("group_identifier")
        .agg(pl.col("NSE").mean().alias("NSE_mean"))
        .sort("NSE_mean")
    )

    # Remove NaN basins
    basin_avg = basin_avg.filter(~pl.col("NSE_mean").is_nan())

    # Select median basin
    n_basins = len(basin_avg)
    median_idx = n_basins // 2
    median_basin = basin_avg[median_idx]["group_identifier"][0]
    median_nse = basin_avg[median_idx]["NSE_mean"][0]
    print(f"✓ Selected basin: {median_basin} (median NSE: {median_nse:.3f})")

    return median_basin


def plot_rolling_forecast(
    eval_dir: Path,
    model_name: str,
    ensemble_name: str,
    basin_id: str,
    seeds: list[int],
    output_path: Path,
    window_size: int = 10,
    max_windows: int | None = None,
    color: str = "steelblue",
) -> None:
    """Plot rolling forecast with non-overlapping windows on continuous date axis.

    Args:
        eval_dir: Evaluation directory
        model_name: Base model name
        ensemble_name: Ensemble model name
        basin_id: Basin to plot
        seeds: List of seed numbers
        output_path: Path to save figure
        window_size: Size of each forecast window (default: 10)
        max_windows: Maximum number of windows to plot (default: None = unlimited)
        color: Color for the curves
    """
    # Load all predictions for this basin
    all_predictions = []

    # Load individual seeds
    for seed in seeds:
        path = (
            eval_dir
            / f"model_name={model_name}"
            / f"seed={seed}"
            / "predictions.parquet"
        )
        df = pl.read_parquet(path).filter(pl.col("group_identifier") == basin_id)
        df = df.with_columns(pl.lit(seed).alias("seed"))
        all_predictions.append(df)

    # Load ensemble
    ensemble_path = (
        eval_dir
        / f"model_name={ensemble_name}"
        / "seed=ensemble"
        / "predictions.parquet"
    )
    ensemble_df = pl.read_parquet(ensemble_path).filter(
        pl.col("group_identifier") == basin_id
    )
    ensemble_df = ensemble_df.with_columns(pl.lit(-1).alias("seed"))
    all_predictions.append(ensemble_df)

    # Combine all predictions
    combined = pl.concat(all_predictions)

    # Check if we have date columns
    if "prediction_date" not in combined.columns:
        print(
            "Warning: No prediction_date column found. Cannot create rolling forecast plot."
        )
        return

    # Sort by issue date and lead time
    combined = combined.sort(["issue_date", "lead_time"])

    # Get unique issue dates for non-overlapping windows
    unique_issue_dates = combined.select("issue_date").unique().sort("issue_date")["issue_date"].to_list()

    # Select issue dates that are 10 days apart (non-overlapping windows)
    window_issue_dates = []
    if len(unique_issue_dates) > 0:
        window_issue_dates.append(unique_issue_dates[0])
        last_selected = unique_issue_dates[0]

        for issue_date in unique_issue_dates[1:]:
            # Check if at least window_size days have passed
            days_diff = (issue_date - last_selected).days
            if days_diff >= window_size:
                window_issue_dates.append(issue_date)
                last_selected = issue_date

                if max_windows is not None and len(window_issue_dates) >= max_windows:
                    break

    print(f"  Plotting {len(window_issue_dates)} forecast windows...")

    # Create plot
    fig, ax = plt.subplots(figsize=(14, 6))

    # Plot observations once (continuous black line)
    # Get all unique prediction dates with observations
    obs_data = (
        ensemble_df.select(["prediction_date", "observation"])
        .unique()
        .sort("prediction_date")
    )
    ax.plot(
        obs_data["prediction_date"],
        obs_data["observation"],
        color="black",
        linewidth=1.5,
        label="Observations",
    )

    # Plot forecasts for each selected issue date
    for issue_date in window_issue_dates:
        # Get all predictions for this issue date
        window_data = combined.filter(pl.col("issue_date") == issue_date).sort("lead_time")

        if len(window_data) == 0:
            continue

        # Plot individual seed predictions (light blue lines)
        for seed in seeds:
            seed_data = window_data.filter(pl.col("seed") == seed).sort("lead_time")
            if len(seed_data) > 0:
                ax.plot(
                    seed_data["prediction_date"],
                    seed_data["prediction"],
                    color=color,
                    alpha=0.3,
                    linewidth=1,
                )

        # Plot ensemble prediction (bold blue line)
        ensemble_subset = window_data.filter(pl.col("seed") == -1).sort("lead_time")
        if len(ensemble_subset) > 0:
            ax.plot(
                ensemble_subset["prediction_date"],
                ensemble_subset["prediction"],
                color=color,
                alpha=1.0,
                linewidth=2.5,
            )

    # Formatting
    ax.set_xlabel("Date", fontsize=14)
    ax.set_ylabel("Streamflow", fontsize=14)
    ax.set_title(f"Rolling {window_size}-Day Forecasts - Basin {basin_id}", fontsize=16)
    ax.grid(True, alpha=0.3)

    # Rotate x-axis labels for better readability
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Custom legend
    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D([0], [0], color="black", linewidth=1.5, label="Observations"),
        Line2D([0], [0], color=color, linewidth=2.5, alpha=1.0, label="Ensemble"),
        Line2D([0], [0], color=color, linewidth=1, alpha=0.3, label="Individual Seeds"),
    ]
    ax.legend(handles=legend_elements, fontsize=12, loc="upper right")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"✓ Saved rolling forecast plot: {output_path}")


def main() -> None:
    """Main analysis workflow."""
    parser = argparse.ArgumentParser(
        description="Analyze multi-seed performance and ensemble comparison",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--eval-dir",
        type=Path,
        required=True,
        help="Evaluation directory containing predictions",
    )
    parser.add_argument(
        "--model-name",
        required=True,
        help="Base model name (e.g., tft_kyrgyzstan)",
    )
    parser.add_argument(
        "--ensemble-name",
        default="ensemble",
        help="Ensemble model name (default: ensemble)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("figures"),
        help="Output directory for plots (default: figures/)",
    )
    parser.add_argument(
        "--basin-id",
        help="Specific basin for rolling forecast (default: auto-select median)",
    )
    parser.add_argument(
        "--color",
        default="steelblue",
        help="Color for plots (default: steelblue)",
    )

    args = parser.parse_args()

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Multi-Seed Performance Analysis")
    print("=" * 60)

    # Load all predictions
    print("\n1. Loading predictions...")
    all_data, seeds = load_all_predictions(
        args.eval_dir, args.model_name, args.ensemble_name
    )
    print(f"✓ Loaded data from {len(seeds)} seeds: {seeds}")

    # Compute metrics
    print("\n2. Computing NSE metrics...")
    metrics = compute_metrics_by_seed(all_data)
    print(f"✓ Computed metrics for {metrics.height} combinations")

    # Create boxplot plots for lead times 1, 5, 10
    print("\n3. Creating boxplots...")
    for lead_time in [1, 5, 10]:
        output_path = args.output_dir / f"boxplot_nse_lead{lead_time}.png"
        plot_boxplot_comparison(
            metrics=metrics,
            lead_time=lead_time,
            seeds=seeds,
            ensemble_name=args.ensemble_name,
            output_path=output_path,
            color=args.color,
        )

    # Select basin for rolling forecast
    print("\n4. Selecting basin for rolling forecast...")
    if args.basin_id:
        basin_id = args.basin_id
        print(f"✓ Using user-specified basin: {basin_id}")
    else:
        basin_id = select_median_basin(metrics, args.model_name, seeds)

    # Create rolling forecast plot
    print("\n5. Creating rolling forecast plot...")
    rolling_output = args.output_dir / f"rolling_forecast_{basin_id}.png"
    plot_rolling_forecast(
        eval_dir=args.eval_dir,
        model_name=args.model_name,
        ensemble_name=args.ensemble_name,
        basin_id=basin_id,
        seeds=seeds,
        output_path=rolling_output,
        color=args.color,
    )

    # Save metrics to parquet for further analysis
    metrics_output = args.output_dir / "metrics_by_seed.parquet"
    metrics.write_parquet(metrics_output)
    print(f"\n✓ Saved metrics to: {metrics_output}")

    print("\n" + "=" * 60)
    print("Analysis complete!")
    print("=" * 60)
    print(f"\nOutput files:")
    print(f"  - {args.output_dir}/boxplot_nse_lead1.png")
    print(f"  - {args.output_dir}/boxplot_nse_lead5.png")
    print(f"  - {args.output_dir}/boxplot_nse_lead10.png")
    print(f"  - {args.output_dir}/rolling_forecast_{basin_id}.png")
    print(f"  - {args.output_dir}/metrics_by_seed.parquet")


if __name__ == "__main__":
    main()
