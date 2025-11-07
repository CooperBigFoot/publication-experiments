"""Visualization functions for benchmark vs challenger analysis."""

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
from matplotlib.colors import hsv_to_rgb, rgb_to_hsv


def create_color_palette(
    model_types: list[str],
    version_labels: list[str],
    benchmark_label: str = "Benchmark",
) -> dict[tuple[str, str], tuple[float, float, float]]:
    """Create color palette where benchmark is saturated and challengers are lighter.

    Args:
        model_types: List of unique model types (e.g., ['ealstm', 'tft'])
        version_labels: List of all version labels including benchmark
        benchmark_label: The label used for benchmark models

    Returns:
        Dictionary mapping (model_type, version_label) to RGB color tuple
    """
    # Get base colors for each model type using seaborn palette
    base_colors = sns.color_palette("Set2", n_colors=len(model_types))
    model_type_colors = dict(zip(model_types, base_colors))

    # Create color mapping
    color_map = {}

    # Determine saturation levels for challengers
    challenger_labels = [v for v in version_labels if v != benchmark_label]
    n_challengers = len(challenger_labels)

    for model_type in model_types:
        base_rgb = model_type_colors[model_type]

        # Benchmark gets full saturation
        color_map[(model_type, benchmark_label)] = base_rgb

        # Challengers get progressively lighter versions
        # Convert to HSV to adjust saturation and value
        base_hsv = rgb_to_hsv(np.array(base_rgb).reshape(1, 1, 3))[0, 0]

        for i, challenger_label in enumerate(challenger_labels):
            # Reduce saturation and increase value (brightness) for challengers
            # saturation: 1.0 (benchmark) -> 0.4 (lightest challenger)
            # value: original -> 0.95 (lightest challenger)
            factor = (i + 1) / (n_challengers + 1)
            new_saturation = base_hsv[1] * (1 - 0.6 * factor)
            new_value = min(base_hsv[2] + 0.3 * factor, 0.95)

            new_hsv = np.array([base_hsv[0], new_saturation, new_value]).reshape(
                1, 1, 3
            )
            new_rgb = hsv_to_rgb(new_hsv)[0, 0]

            color_map[(model_type, challenger_label)] = tuple(new_rgb)

    return color_map


def plot_improvement_stacked_bar(
    comparison_df: pl.DataFrame,
    model_type: str,
    challenger_label: str,
    metric: str = "nse",
    lead_times: list[int] | None = None,
    figsize: tuple[int, int] = (8, 5),
):
    """Plot stacked bar chart showing % of basins that improved, worsened, or stayed same.

    Args:
        comparison_df: DataFrame from the comparison join (with delta column)
        model_type: Specific model type to plot (e.g., 'ealstm')
        challenger_label: Specific challenger to compare (e.g., 'Natural Finetuned')
        metric: Metric name (lowercase)
        lead_times: Lead times to include (defaults to all)
        figsize: Figure size tuple
    """
    # Filter to specific model_type and challenger
    filtered = comparison_df.filter(
        (pl.col("model_type") == model_type)
        & (pl.col("model_version") == challenger_label)
    )

    if lead_times is not None:
        filtered = filtered.filter(pl.col("lead_time").is_in(lead_times))

    # Round metrics to 3 decimals for comparison
    filtered = filtered.with_columns(
        [pl.col(f"{metric}_delta").round(3).alias(f"{metric}_delta_rounded")]
    )

    # Calculate percentages for each lead time
    stats = (
        filtered.group_by("lead_time")
        .agg(
            [
                (pl.col(f"{metric}_delta_rounded") > 0).sum().alias("n_improved"),
                (pl.col(f"{metric}_delta_rounded") < 0).sum().alias("n_degraded"),
                (pl.col(f"{metric}_delta_rounded") == 0).sum().alias("n_unchanged"),
                pl.len().alias("n_total"),
            ]
        )
        .with_columns(
            [
                (pl.col("n_improved") / pl.col("n_total") * 100).alias(
                    "pct_improved"
                ),
                (pl.col("n_degraded") / pl.col("n_total") * 100).alias(
                    "pct_degraded"
                ),
                (pl.col("n_unchanged") / pl.col("n_total") * 100).alias(
                    "pct_unchanged"
                ),
            ]
        )
        .sort("lead_time")
    )

    # Convert to pandas for plotting
    stats_pd = stats.to_pandas()

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # X positions
    x_pos = np.arange(len(stats_pd))
    lead_times_list = stats_pd["lead_time"].tolist()

    # Stacked bar chart
    # Bottom: improved (green)
    # Middle: unchanged (gray)
    # Top: degraded (red)
    ax.bar(
        x_pos, stats_pd["pct_improved"], color="green", alpha=0.7, label="Improved"
    )

    ax.bar(
        x_pos,
        stats_pd["pct_unchanged"],
        bottom=stats_pd["pct_improved"],
        color="gray",
        alpha=0.5,
        label="Unchanged",
    )

    ax.bar(
        x_pos,
        stats_pd["pct_degraded"],
        bottom=stats_pd["pct_improved"] + stats_pd["pct_unchanged"],
        color="red",
        alpha=0.7,
        label="Degraded",
    )

    # Add percentage labels on each segment
    for i, row in stats_pd.iterrows():
        # Improved label
        if row["pct_improved"] > 5:  # Only show if segment is large enough
            ax.text(
                i,
                row["pct_improved"] / 2,
                f'{row["pct_improved"]:.1f}%',
                ha="center",
                va="center",
                fontsize=10,
                color="white",
                fontweight="bold",
            )

        # Unchanged label
        if row["pct_unchanged"] > 5:
            ax.text(
                i,
                row["pct_improved"] + row["pct_unchanged"] / 2,
                f'{row["pct_unchanged"]:.1f}%',
                ha="center",
                va="center",
                fontsize=10,
                color="black",
                fontweight="bold",
            )

        # Degraded label
        if row["pct_degraded"] > 5:
            ax.text(
                i,
                row["pct_improved"] + row["pct_unchanged"] + row["pct_degraded"] / 2,
                f'{row["pct_degraded"]:.1f}%',
                ha="center",
                va="center",
                fontsize=10,
                color="white",
                fontweight="bold",
            )

    # Customize plot
    ax.set_xlabel("Lead Time (days)", fontsize=12)
    ax.set_ylabel("Percentage of Basins (%)", fontsize=12)
    ax.set_title(
        f"{model_type.upper()}: {challenger_label} vs Benchmark",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xticks(x_pos)
    ax.set_xticklabels(lead_times_list)
    ax.set_ylim(0, 100)
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    sns.despine()
    plt.show()

    # Print summary table
    print(f"\n{model_type.upper()} - {challenger_label} vs Benchmark:")
    print(f"{'Lead Time':<12} {'Improved':<12} {'Unchanged':<12} {'Degraded':<12}")
    print("-" * 48)
    for _, row in stats_pd.iterrows():
        print(
            f"{row['lead_time']:<12} "
            f"{row['n_improved']:>4} ({row['pct_improved']:>5.1f}%)  "
            f"{row['n_unchanged']:>4} ({row['pct_unchanged']:>5.1f}%)  "
            f"{row['n_degraded']:>4} ({row['pct_degraded']:>5.1f}%)"
        )


def plot_skill_scatter(
    comparison_df: pl.DataFrame,
    model_type: str,
    challenger_label: str,
    metric: str = "nse",
    lead_times: list[int] | None = None,
    figsize: tuple[int, int] = (12, 4),
):
    """Plot scatter: baseline NSE vs remaining skill captured.

    Shows whether finetuning helps low vs high performers differently.

    Args:
        comparison_df: DataFrame with remaining_skill_captured column
        model_type: Specific model type to plot (e.g., 'ealstm')
        challenger_label: Specific challenger to compare (e.g., 'Natural Finetuned')
        metric: Metric name (lowercase)
        lead_times: Lead times to include (defaults to all)
        figsize: Figure size tuple
    """
    # Filter to specific model_type and challenger
    filtered = comparison_df.filter(
        (pl.col("model_type") == model_type)
        & (pl.col("model_version") == challenger_label)
    )

    if lead_times is not None:
        filtered = filtered.filter(pl.col("lead_time").is_in(lead_times))
        plot_lead_times = lead_times
    else:
        plot_lead_times = sorted(filtered["lead_time"].unique().to_list())

    # Remove nulls
    filtered = filtered.filter(~pl.col("remaining_skill_captured").is_null())

    # Convert to pandas
    df_plot = filtered.to_pandas()

    # Create subplots for each lead time
    n_plots = len(plot_lead_times)
    fig, axes = plt.subplots(1, n_plots, figsize=figsize, sharey=True)

    if n_plots == 1:
        axes = [axes]

    for i, lead_time in enumerate(plot_lead_times):
        ax = axes[i]
        df_lt = df_plot[df_plot["lead_time"] == lead_time]

        # Scatter plot
        scatter = ax.scatter(
            df_lt[f"{metric.upper()}_benchmark"],
            df_lt["remaining_skill_captured"] * 100,  # Convert to percentage
            alpha=0.6,
            s=50,
            c=df_lt["remaining_skill_captured"],
            cmap="RdYlGn",
            vmin=-0.5,
            vmax=0.5,
            edgecolors="black",
            linewidth=0.5,
        )

        # Add horizontal line at 0
        ax.axhline(y=0, color="black", linestyle="--", alpha=0.5, linewidth=1)

        # Compute correlation
        if len(df_lt) > 2:
            corr = df_lt[f"{metric.upper()}_benchmark"].corr(
                df_lt["remaining_skill_captured"]
            )
            ax.text(
                0.05,
                0.95,
                f"r = {corr:.3f}",
                transform=ax.transAxes,
                fontsize=10,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
            )

        # Customize
        ax.set_xlabel(f"Baseline {metric.upper()}", fontsize=11)
        ax.set_title(
            f"Lead Time: {lead_time} days", fontsize=12, fontweight="bold"
        )
        ax.grid(alpha=0.3)

        if i == 0:
            ax.set_ylabel("Remaining Skill Captured (%)", fontsize=11)

    # Add colorbar
    cbar = fig.colorbar(
        scatter, ax=axes, orientation="vertical", fraction=0.02, pad=0.04
    )
    cbar.set_label("Remaining Skill Captured", fontsize=10)

    # Main title
    fig.suptitle(
        f"{model_type.upper()}: {challenger_label} vs Benchmark - Skill Analysis",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )

    plt.tight_layout()
    sns.despine()
    plt.show()

    # Print interpretation
    print(f"\n{model_type.upper()} - {challenger_label} Skill Capture Analysis:")
    print("=" * 70)
    print("Interpretation:")
    print("  - Negative correlation → Finetuning helps low performers more")
    print("  - Positive correlation → Finetuning helps high performers more")
    print("  - Points above 0 → Captured remaining skill (improved)")
    print("  - Points below 0 → Lost skill (degraded)")
    print("=" * 70)


def plot_tier_skill_analysis(
    tier_stats: pl.DataFrame,
    model_type: str,
    challenger_label: str,
    figsize: tuple[int, int] = (12, 5),
):
    """Plot skill captured by performance tier across lead times.

    Args:
        tier_stats: Output from analyze_by_performance_tier()
        model_type: Model type name for title
        challenger_label: Challenger label for title
        figsize: Figure size
    """
    df_plot = tier_stats.to_pandas()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Plot 1: Mean skill captured by tier
    tier_order = ["Low", "Medium", "High"]
    colors = {"Low": "#d62728", "Medium": "#ff7f0e", "High": "#2ca02c"}

    for tier in tier_order:
        df_tier = df_plot[df_plot["performance_tier"] == tier]
        ax1.plot(
            df_tier["lead_time"],
            df_tier["mean_skill_captured"] * 100,
            marker="o",
            linewidth=2,
            markersize=8,
            label=f"{tier} Performers",
            color=colors[tier],
        )

    ax1.axhline(y=0, color="black", linestyle="--", alpha=0.5, linewidth=1)
    ax1.set_xlabel("Lead Time (days)", fontsize=12)
    ax1.set_ylabel("Mean Skill Captured (%)", fontsize=12)
    ax1.set_title("Skill Captured by Performance Tier", fontsize=13, fontweight="bold")
    ax1.legend(loc="best", fontsize=10)
    ax1.grid(alpha=0.3)

    # Plot 2: % Improved by tier
    for tier in tier_order:
        df_tier = df_plot[df_plot["performance_tier"] == tier]
        ax2.plot(
            df_tier["lead_time"],
            df_tier["pct_improved"],
            marker="s",
            linewidth=2,
            markersize=8,
            label=f"{tier} Performers",
            color=colors[tier],
        )

    ax2.axhline(y=50, color="black", linestyle="--", alpha=0.5, linewidth=1)
    ax2.set_xlabel("Lead Time (days)", fontsize=12)
    ax2.set_ylabel("% Basins Improved", fontsize=12)
    ax2.set_title("% of Basins that Improved", fontsize=13, fontweight="bold")
    ax2.legend(loc="best", fontsize=10)
    ax2.grid(alpha=0.3)
    ax2.set_ylim(0, 100)

    fig.suptitle(
        f"{model_type.upper()}: {challenger_label} - Performance Tier Analysis",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )

    plt.tight_layout()
    sns.despine()
    plt.show()

    # Print summary
    print(f"\n{model_type.upper()} - {challenger_label}: Key Findings")
    print("=" * 80)

    for lt in sorted(df_plot["lead_time"].unique()):
        print(f"\nLead Time = {lt} days:")
        df_lt = df_plot[df_plot["lead_time"] == lt].sort_values(
            "performance_tier", key=lambda x: x.map({"Low": 0, "Medium": 1, "High": 2})
        )
        for _, row in df_lt.iterrows():
            print(
                f"  {row['performance_tier']:>6} performers: "
                f"Mean skill = {row['mean_skill_captured']*100:>6.2f}%  |  "
                f"{row['pct_improved']:>5.1f}% improved  |  "
                f"n={int(row['n_total'])}"
            )
