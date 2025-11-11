"""
Create animated GIF showing forecast evolution over time.

This script reads prediction data from a parquet file and generates an animation
showing how forecasts evolve as new issue dates are added.
"""

import argparse
import logging
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.animation import FuncAnimation

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# Set matplotlib rcParams for professional styling
plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 10
plt.rcParams["axes.titlesize"] = 11
plt.rcParams["axes.labelsize"] = 10
plt.rcParams["xtick.labelsize"] = 9
plt.rcParams["ytick.labelsize"] = 9
plt.rcParams["legend.fontsize"] = 9


def create_forecast_gif(
    df: pl.DataFrame,
    group_id: str,
    start_date: str,
    end_date: str,
    output_path: str = "forecast_animation.gif",
    forecast_window_days: int = 10,
) -> None:
    """
    Create an animated GIF showing forecast evolution over time.

    Optimized version with professional styling: pre-computes all data upfront,
    normalizes by max observation, and uses visual hierarchy to distinguish
    current vs historical forecasts.

    Parameters:
    -----------
    df : pl.DataFrame
        DataFrame with forecast data (columns: group_identifier, issue_date,
        prediction_date, prediction, observation, lead_time)
    group_id : str
        The group_identifier to filter for
    start_date : str
        Start date for animation (format: 'YYYY-MM-DD')
    end_date : str
        End date for animation (format: 'YYYY-MM-DD')
    output_path : str
        Path to save the GIF
    forecast_window_days : int
        Number of days to show ahead of current issue date
    """

    # Convert string dates to datetime
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")

    # Filter and sort data once
    filtered_df = df.filter(
        (pl.col("group_identifier") == group_id)
        & (pl.col("issue_date") >= start_dt)
        & (pl.col("issue_date") <= end_dt)
    ).sort("issue_date", "lead_time")

    if len(filtered_df) == 0:
        raise ValueError(
            f"No data found for group_id='{group_id}' between {start_date} and {end_date}"
        )

    # ===========================================================================
    # DATA NORMALIZATION
    # ===========================================================================

    # Find max observation value for normalization
    max_observation = filtered_df.select(pl.col("observation").max()).item()

    if max_observation > 0:
        logger.info(f"Normalizing data by max observation: {max_observation:.2f}")
        filtered_df = filtered_df.with_columns(
            [
                (pl.col("observation") / max_observation).alias("observation"),
                (pl.col("prediction") / max_observation).alias("prediction"),
            ]
        )
    else:
        logger.warning("Max observation is 0, skipping normalization")

    # ===========================================================================
    # PRE-COMPUTE ALL DATA UPFRONT (avoid DataFrame ops in animation loop)
    # ===========================================================================

    # Get unique issue dates (already sorted) and strip timezone
    issue_dates_series = filtered_df.select("issue_date").unique().sort("issue_date")
    issue_dates = [
        dt.replace(tzinfo=None) if hasattr(dt, "replace") else dt
        for dt in issue_dates_series["issue_date"].to_list()
    ]

    logger.info(f"Animating through {len(issue_dates)} issue dates")

    # Pre-compute all forecast lines (one per issue date)
    # Store as list of (issue_date, dates, predictions) tuples
    all_forecasts: list[tuple[datetime, list[datetime], list[float]]] = []

    for issue_dt in issue_dates:
        forecast_data = filtered_df.filter(pl.col("issue_date") == issue_dt).sort(
            "lead_time"
        )

        dates = [
            d.replace(tzinfo=None) if hasattr(d, "replace") else d
            for d in forecast_data["prediction_date"].to_list()
        ]
        preds = forecast_data["prediction"].to_list()

        all_forecasts.append((issue_dt, dates, preds))

    # Pre-compute observations (unique prediction_date + observation pairs)
    obs_data = (
        filtered_df.select(["prediction_date", "observation"])
        .unique()
        .sort("prediction_date")
    )

    all_obs_dates = [
        d.replace(tzinfo=None) if hasattr(d, "replace") else d
        for d in obs_data["prediction_date"].to_list()
    ]
    all_obs_values = obs_data["observation"].to_list()

    # Pre-compute which observations are visible at each frame
    # (only show observations up to current issue date)
    obs_cutoff_indices = []
    for issue_dt in issue_dates:
        cutoff_idx = 0
        for idx, obs_date in enumerate(all_obs_dates):
            if obs_date <= issue_dt:
                cutoff_idx = idx + 1
            else:
                break
        obs_cutoff_indices.append(cutoff_idx)

    # Pre-compute global y-axis limits (fixed scale for entire animation)
    all_prediction_values = [pred for _, _, preds in all_forecasts for pred in preds]
    all_values = all_prediction_values + all_obs_values

    y_min = min(all_values)
    y_max = max(all_values)
    # Add 10% padding to make it look nicer
    y_range = y_max - y_min
    y_min_padded = y_min - 0.1 * y_range
    y_max_padded = y_max + 0.1 * y_range

    # ===========================================================================
    # ANIMATION LOOP (now just slicing pre-computed data)
    # ===========================================================================

    fig, ax = plt.subplots(figsize=(12, 6))

    def update(frame_idx: int) -> None:
        ax.clear()
        current_issue_date = issue_dates[frame_idx]

        # Plot historical forecasts (faded, thin lines)
        for i in range(frame_idx):
            _, dates, preds = all_forecasts[i]
            if len(dates) > 0:
                ax.plot(dates, preds, "r-", alpha=0.15, linewidth=1, zorder=2)

        # Plot current forecast (bold, with scatter points)
        if frame_idx < len(all_forecasts):
            _, current_dates, current_preds = all_forecasts[frame_idx]
            if len(current_dates) > 0:
                ax.plot(
                    current_dates,
                    current_preds,
                    "r-",
                    alpha=0.8,
                    linewidth=2.5,
                    zorder=8,
                    label="Forecast",
                )
                ax.scatter(
                    current_dates,
                    current_preds,
                    c="red",
                    s=40,
                    alpha=0.8,
                    zorder=9,
                    edgecolors="darkred",
                    linewidths=0.5,
                )

        # Plot observations up to current issue date (always on top)
        cutoff_idx = obs_cutoff_indices[frame_idx]
        if cutoff_idx > 0:
            obs_dates_slice = all_obs_dates[:cutoff_idx]
            obs_values_slice = all_obs_values[:cutoff_idx]
            ax.plot(
                obs_dates_slice,
                obs_values_slice,
                "k-",
                linewidth=2.5,
                zorder=10,
                label="Observations",
            )

        # Dynamic x-axis: show history + forecast window + buffer
        min_date = issue_dates[0]
        buffer_days = 5
        max_date = current_issue_date + timedelta(days=forecast_window_days + buffer_days)
        ax.set_xlim(min_date, max_date)

        # Set FIXED y-axis limits (pre-computed)
        ax.set_ylim(y_min_padded, y_max_padded)

        # Labels and formatting
        ax.set_xlabel("Date")
        ax.set_ylabel("Scaled Discharge (-)")
        ax.set_title(
            f'{group_id} - Issue Date: {current_issue_date.strftime("%Y-%m-%d")}'
        )

        # Fixed legend - always show
        ax.legend(loc="upper left", framealpha=0.9)

        # Clean look - no grid
        plt.xticks(rotation=45)
        plt.tight_layout()

    # Create animation
    logger.info(f"Creating animation with {len(issue_dates)} frames...")
    anim = FuncAnimation(
        fig, update, frames=len(issue_dates), interval=200, repeat=True, blit=False
    )

    # Save as GIF with high DPI
    logger.info(f"Saving GIF to {output_path}...")
    anim.save(output_path, writer="pillow", fps=5, dpi=200)
    plt.close()

    logger.info(f"✓ Animation saved to {output_path}")


def main() -> None:
    """CLI entrypoint for creating forecast animations."""
    parser = argparse.ArgumentParser(
        description="Create animated GIF showing forecast evolution over time",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  uv run python scripts/create_forecast_animation.py \\
    results/evaluation/model/predictions.parquet \\
    --group-id kyrgyzstan_15215 \\
    --start-date 2019-04-30 \\
    --end-date 2019-10-31

  # With expm1 transform (hotfix for log1p bug)
  uv run python scripts/create_forecast_animation.py \\
    results/evaluation/model/predictions.parquet \\
    --group-id kyrgyzstan_15215 \\
    --start-date 2019-04-30 \\
    --end-date 2019-10-31 \\
    --expm1
        """,
    )

    parser.add_argument(
        "data_path",
        type=str,
        help="Path to the predictions parquet file",
    )
    parser.add_argument(
        "--group-id",
        type=str,
        required=True,
        help="Group identifier to filter for (e.g., 'kyrgyzstan_15215')",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        required=True,
        help="Start date for animation (format: YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        required=True,
        help="End date for animation (format: YYYY-MM-DD)",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="forecast.gif",
        help="Path to save the GIF (default: forecast.gif)",
    )
    parser.add_argument(
        "--forecast-window",
        type=int,
        default=10,
        help="Number of days to show ahead of current issue date (default: 10)",
    )
    parser.add_argument(
        "--expm1",
        action="store_true",
        help=(
            "Apply expm1 (inverse log1p) transform to predictions and observations. "
            "Use this as a hotfix if evaluation pipeline applied log1p transform "
            "but didn't inverse it. This flag will be removed once pipeline is fixed."
        ),
    )

    args = parser.parse_args()

    # Validate data path exists
    data_path = Path(args.data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {args.data_path}")

    # Read data
    logger.info(f"Reading data from {args.data_path}...")
    df = pl.read_parquet(args.data_path)

    # Apply expm1 transform if requested (HOTFIX for log1p bug)
    if args.expm1:
        logger.warning(
            "⚠️  Applying expm1 transform to observations and predictions "
            "(HOTFIX for inverse log1p transform bug)"
        )
        df = df.with_columns(
            [
                (pl.col("observation").exp() - 1).alias("observation"),
                (pl.col("prediction").exp() - 1).alias("prediction"),
            ]
        )

    # Create animation
    create_forecast_gif(
        df=df,
        group_id=args.group_id,
        start_date=args.start_date,
        end_date=args.end_date,
        output_path=args.output_path,
        forecast_window_days=args.forecast_window,
    )


if __name__ == "__main__":
    main()
