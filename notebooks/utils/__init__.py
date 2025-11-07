"""Utility functions for benchmark vs challenger analysis."""

from .benchmark_analysis import (
    load_and_compute_metrics,
    analyze_by_performance_tier,
    compute_skill_correlation,
)

from .visualization import (
    create_color_palette,
    plot_improvement_stacked_bar,
    plot_skill_scatter,
    plot_tier_skill_analysis,
)

__all__ = [
    # Data loading and analysis
    "load_and_compute_metrics",
    "analyze_by_performance_tier",
    "compute_skill_correlation",
    # Visualization
    "create_color_palette",
    "plot_improvement_stacked_bar",
    "plot_skill_scatter",
    "plot_tier_skill_analysis",
]
