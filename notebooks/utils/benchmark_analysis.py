"""Analysis functions for comparing benchmark and challenger models."""

import polars as pl
from transfer_learning_publication.evaluation import MetricCalculator


def load_and_compute_metrics(
    path: str,
    version_label: str,
    metric: str = "nse",
    lead_times: list[int] | None = None,
) -> pl.DataFrame:
    """Load predictions, compute metrics, and tag with version label.

    Args:
        path: Glob pattern for prediction parquet files
        version_label: Label to identify this version (e.g., 'Benchmark', 'Natural Finetuned')
        metric: Metric to compute
        lead_times: Optional list of lead times to filter to

    Returns:
        DataFrame with columns: model_name, group_identifier, lead_time, {METRIC}, model_version
    """
    # Load predictions and compute metrics
    calc = MetricCalculator.from_parquet(path)

    metrics = calc.compute_metrics(
        metrics=[metric],
        group_by=["model_name", "group_identifier", "lead_time"],
        exclude_filled=True,
    )

    # Add version label
    metrics = metrics.with_columns(pl.lit(version_label).alias("model_version"))

    # Filter to specified lead times if provided
    if lead_times is not None:
        metrics = metrics.filter(pl.col("lead_time").is_in(lead_times))

    # Remove NaN values
    metrics = metrics.filter(~pl.col(metric.upper()).is_nan())

    return metrics


def analyze_by_performance_tier(
    comparison_df: pl.DataFrame,
    model_type: str,
    challenger_label: str,
    metric: str = "nse",
    lead_times: list[int] | None = None,
) -> pl.DataFrame:
    """Stratified analysis: bin basins by per-horizon percentiles of baseline performance.

    Args:
        comparison_df: DataFrame with remaining_skill_captured column
        model_type: Specific model type to analyze
        challenger_label: Specific challenger to compare
        metric: Metric name (lowercase)
        lead_times: Lead times to include (defaults to all)

    Returns:
        Summary DataFrame with statistics per tier and lead time
    """
    # Filter to specific model_type and challenger
    filtered = comparison_df.filter(
        (pl.col("model_type") == model_type)
        & (pl.col("model_version") == challenger_label)
    )

    if lead_times is not None:
        filtered = filtered.filter(pl.col("lead_time").is_in(lead_times))

    # Remove nulls
    filtered = filtered.filter(~pl.col("remaining_skill_captured").is_null())

    # Compute per-lead-time percentiles and assign tiers
    filtered = filtered.with_columns(
        [
            pl.col(f"{metric.upper()}_benchmark")
            .quantile(0.33)
            .over("lead_time")
            .alias("p33_threshold"),
            pl.col(f"{metric.upper()}_benchmark")
            .quantile(0.66)
            .over("lead_time")
            .alias("p66_threshold"),
        ]
    )

    # Assign performance tier based on per-horizon percentiles
    filtered = filtered.with_columns(
        [
            pl.when(
                pl.col(f"{metric.upper()}_benchmark") < pl.col("p33_threshold")
            )
            .then(pl.lit("Low"))
            .when(
                pl.col(f"{metric.upper()}_benchmark") <= pl.col("p66_threshold")
            )
            .then(pl.lit("Medium"))
            .otherwise(pl.lit("High"))
            .alias("performance_tier")
        ]
    )

    # Compute statistics per tier and lead time
    tier_stats = (
        filtered.group_by(["lead_time", "performance_tier"])
        .agg(
            [
                pl.col(f"{metric.upper()}_benchmark")
                .mean()
                .alias("baseline_mean"),
                pl.col("remaining_skill_captured")
                .mean()
                .alias("mean_skill_captured"),
                pl.col("remaining_skill_captured")
                .median()
                .alias("median_skill_captured"),
                pl.col("remaining_skill_captured")
                .std()
                .alias("std_skill_captured"),
                (pl.col("remaining_skill_captured") > 0)
                .sum()
                .alias("n_improved"),
                (pl.col("remaining_skill_captured") < 0)
                .sum()
                .alias("n_degraded"),
                pl.len().alias("n_total"),
            ]
        )
        .with_columns(
            [(pl.col("n_improved") / pl.col("n_total") * 100).alias("pct_improved")]
        )
        .sort(["lead_time", "performance_tier"])
    )

    return tier_stats


def compute_skill_correlation(
    comparison_df: pl.DataFrame,
    model_type: str,
    challenger_label: str,
    metric: str = "nse",
    lead_times: list[int] | None = None,
) -> pl.DataFrame:
    """Compute correlation between baseline performance and skill captured.

    Args:
        comparison_df: DataFrame with remaining_skill_captured column
        model_type: Specific model type to analyze
        challenger_label: Specific challenger to compare
        metric: Metric name (lowercase)
        lead_times: Lead times to include (defaults to all)

    Returns:
        DataFrame with correlation statistics per lead time
    """
    # Filter to specific model_type and challenger
    filtered = comparison_df.filter(
        (pl.col("model_type") == model_type)
        & (pl.col("model_version") == challenger_label)
    )

    if lead_times is not None:
        filtered = filtered.filter(pl.col("lead_time").is_in(lead_times))

    # Remove nulls
    filtered = filtered.filter(~pl.col("remaining_skill_captured").is_null())

    # Convert to pandas for correlation computation (easier with pandas)
    df_pd = filtered.to_pandas()

    # Compute correlations per lead time
    correlations = []
    for lt in sorted(df_pd["lead_time"].unique()):
        df_lt = df_pd[df_pd["lead_time"] == lt]

        if len(df_lt) > 2:
            # Pearson correlation
            corr_pearson = df_lt[f"{metric.upper()}_benchmark"].corr(
                df_lt["remaining_skill_captured"]
            )

            # Spearman correlation (rank-based, more robust)
            corr_spearman = df_lt[f"{metric.upper()}_benchmark"].corr(
                df_lt["remaining_skill_captured"], method="spearman"
            )

            correlations.append(
                {
                    "lead_time": lt,
                    "pearson_r": corr_pearson,
                    "spearman_r": corr_spearman,
                    "n_basins": len(df_lt),
                }
            )

    return pl.DataFrame(correlations)
