#!/usr/bin/env python
"""
Prepare streamflow data for basin clustering.

Reads basin IDs from quality_basins_filtered.txt and creates 52-week standardized
hydrographs for clustering analysis.

Output:
    - streamflow_52wk.npy: shape (n_basins, 52, 1)
    - basin_ids.csv: gauge_id column matching .npy row order
    - processing_log.csv: detailed processing log for audit trail
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

from transfer_learning_publication.data import CaravanDataSource

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def process_basin(
    caravan: CaravanDataSource, gauge_id: str, hemisphere: str
) -> tuple[np.ndarray | None, str | None]:
    """
    Process a single basin to create 52-week standardized hydrograph.

    Args:
        caravan: CaravanDataSource instance
        gauge_id: Basin gauge ID
        hemisphere: "northern" or "southern"

    Returns:
        (weekly_data, error_msg): weekly_data is (52,) array or None if failed
    """
    try:
        # Convert gauge_id to lowercase for GRDC basins
        query_id = gauge_id.lower() if gauge_id.startswith("GRDC_") else gauge_id

        # Get streamflow data
        ts_data = caravan.get_timeseries(
            gauge_ids=[query_id], columns=["streamflow"]
        ).collect()

        if len(ts_data) == 0:
            return None, "no_data"

        # Check for all-null streamflow
        if ts_data["streamflow"].null_count() == len(ts_data):
            return None, "all_null"

        # Determine water year start based on hemisphere
        if hemisphere == "northern":
            wy_start_month = 10  # October
        else:  # southern
            wy_start_month = 7  # July

        # Add water year column
        ts_data = ts_data.with_columns(
            [
                pl.when(pl.col("date").dt.month() >= wy_start_month)
                .then(pl.col("date").dt.year() + 1)
                .otherwise(pl.col("date").dt.year())
                .alias("water_year")
            ]
        )

        # Calculate day of year
        ts_data = ts_data.with_columns([pl.col("date").dt.ordinal_day().alias("doy")])

        # Compute day of water year
        if hemisphere == "northern":
            # Oct 1 is day 1, Sep 30 is day 365
            ts_data = ts_data.with_columns(
                [
                    pl.when(pl.col("date").dt.month() >= 10)
                    .then(pl.col("doy") - pl.lit(273) + 1)  # Oct 1 = day 274
                    .otherwise(pl.col("doy") + pl.lit(92))  # Days from Oct to end
                    .alias("day_of_wy")
                ]
            )
        else:  # southern
            # Jul 1 is day 1, Jun 30 is day 365
            ts_data = ts_data.with_columns(
                [
                    pl.when(pl.col("date").dt.month() >= 7)
                    .then(pl.col("doy") - pl.lit(181) + 1)  # Jul 1 = day 182
                    .otherwise(pl.col("doy") + pl.lit(184))  # Days from Jul to end
                    .alias("day_of_wy")
                ]
            )

        # Create mean annual daily hydrograph
        daily_avg = (
            ts_data.group_by("day_of_wy")
            .agg(pl.col("streamflow").mean().alias("streamflow_avg"))
            .sort("day_of_wy")
        )

        # Convert to pandas for gap filling
        daily_df = daily_avg.to_pandas()

        # Forward fill gaps < 7 days
        streamflow_vals = daily_df["streamflow_avg"].values
        for i in range(len(streamflow_vals)):
            if np.isnan(streamflow_vals[i]):
                # Look back up to 7 days
                for j in range(1, 8):
                    if i - j >= 0 and not np.isnan(streamflow_vals[i - j]):
                        streamflow_vals[i] = streamflow_vals[i - j]
                        break

        # Check for remaining NaNs
        if np.isnan(streamflow_vals).any():
            return None, "too_many_gaps"

        # Ensure we have at least 364 days
        if len(streamflow_vals) < 364:
            return None, "insufficient_days"

        # Aggregate to 52 weeks
        weekly_data = []
        for week in range(52):
            start_day = week * 7
            end_day = min(start_day + 7, len(streamflow_vals))
            week_mean = np.mean(streamflow_vals[start_day:end_day])
            weekly_data.append(week_mean)

        weekly_data = np.array(weekly_data)

        # Z-score standardize
        if np.std(weekly_data) > 0:
            weekly_data = (weekly_data - np.mean(weekly_data)) / np.std(weekly_data)
        else:
            return None, "constant_values"

        return weekly_data, None

    except Exception as e:
        return None, f"error: {str(e)}"


def main() -> None:
    """Main execution function."""
    # Define paths
    base_dir = Path("/Users/nicolaslazaro/Desktop/work/publication-experiments")
    basin_ids_file = base_dir / "basin_selection/quality_basins_filtered.txt"
    output_dir = base_dir / "basin_selection/clustering"
    caravan_path = Path("/Users/nicolaslazaro/Desktop/LSH_hive_data")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    # Read basin IDs
    logger.info(f"Reading basin IDs from {basin_ids_file}")
    with open(basin_ids_file) as f:
        basin_ids = [line.strip() for line in f if line.strip()]
    logger.info(f"Loaded {len(basin_ids)} basin IDs")

    # Initialize CARAVAN data source
    logger.info(f"Initializing CaravanDataSource at {caravan_path}")
    caravan = CaravanDataSource(str(caravan_path))

    # Get static attributes (lat/lon for hemisphere determination)
    logger.info("Loading basin metadata...")
    static_attrs = caravan.get_static_attributes(columns=["gauge_lat"]).collect()

    # Create basin metadata with hemisphere
    basin_metadata = static_attrs.with_columns(
        [
            pl.when(pl.col("gauge_lat") >= 0)
            .then(pl.lit("northern"))
            .otherwise(pl.lit("southern"))
            .alias("hemisphere")
        ]
    )

    # Convert to dict for fast lookup
    hemisphere_map = dict(
        zip(
            basin_metadata["gauge_id"].to_list(),
            basin_metadata["hemisphere"].to_list(),
        )
    )

    logger.info(f"Loaded metadata for {len(hemisphere_map)} basins")
    logger.info("=" * 60)

    # Process all basins
    logger.info(f"Processing {len(basin_ids)} basins...")
    valid_basins = []
    valid_data = []
    processing_log = []

    for idx, gauge_id in enumerate(basin_ids, 1):
        # Get hemisphere
        hemisphere = hemisphere_map.get(gauge_id)
        if hemisphere is None:
            logger.warning(f"Basin {gauge_id} not found in CARAVAN metadata")
            processing_log.append(
                {
                    "gauge_id": gauge_id,
                    "status": "failed",
                    "reason": "not_in_caravan",
                    "hemisphere": None,
                }
            )
            continue

        # Process basin
        weekly_data, error_msg = process_basin(caravan, gauge_id, hemisphere)

        if weekly_data is not None:
            valid_basins.append(gauge_id)
            valid_data.append(weekly_data)
            processing_log.append(
                {
                    "gauge_id": gauge_id,
                    "status": "success",
                    "reason": None,
                    "hemisphere": hemisphere,
                }
            )
        else:
            logger.warning(f"Basin {gauge_id} failed: {error_msg}")
            processing_log.append(
                {
                    "gauge_id": gauge_id,
                    "status": "failed",
                    "reason": error_msg,
                    "hemisphere": hemisphere,
                }
            )

        # Progress tracking
        if idx % 100 == 0:
            logger.info(
                f"Progress: {idx}/{len(basin_ids)} | "
                f"Valid: {len(valid_basins)} | "
                f"Failed: {len(processing_log) - len(valid_basins)}"
            )

    logger.info("=" * 60)
    logger.info("Processing complete!")
    logger.info(f"Valid basins: {len(valid_basins)}")
    logger.info(f"Failed basins: {len(basin_ids) - len(valid_basins)}")

    # Save outputs
    logger.info("\nSaving outputs...")

    # 1. Save streamflow array
    X = np.array(valid_data)  # Shape: (n_basins, 52)
    X = X.reshape(X.shape[0], 52, 1)  # Shape: (n_basins, 52, 1)
    npy_path = output_dir / "streamflow_52wk.npy"
    np.save(npy_path, X)
    logger.info(f"✓ Saved {npy_path} (shape: {X.shape}, dtype: {X.dtype})")

    # 2. Save basin IDs
    basin_ids_df = pd.DataFrame({"gauge_id": valid_basins})
    csv_path = output_dir / "basin_ids.csv"
    basin_ids_df.to_csv(csv_path, index=False)
    logger.info(f"✓ Saved {csv_path} ({len(valid_basins)} basins)")

    # 3. Save processing log for audit trail
    log_df = pd.DataFrame(processing_log)
    log_path = output_dir / "processing_log.csv"
    log_df.to_csv(log_path, index=False)
    logger.info(f"✓ Saved {log_path}")

    # Summary statistics
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Input basins: {len(basin_ids)}")
    logger.info(f"Successfully processed: {len(valid_basins)}")
    logger.info(f"Failed: {len(basin_ids) - len(valid_basins)}")
    logger.info(f"Success rate: {len(valid_basins) / len(basin_ids) * 100:.1f}%")

    # Show failure reasons
    if len(basin_ids) > len(valid_basins):
        logger.info("\nFailure reasons:")
        failure_df = log_df[log_df["status"] == "failed"]
        for reason, count in failure_df["reason"].value_counts().items():
            logger.info(f"  {reason}: {count}")

    logger.info(f"\nOutput directory: {output_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
