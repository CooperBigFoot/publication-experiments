"""Plot training and validation loss curves from metrics CSV."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_metrics(metrics_path: Path, output_path: Path | None = None) -> None:
    """Plot training and validation loss curves.

    Args:
        metrics_path: Path to metrics.csv file
        output_path: Optional path to save the plot. If None, displays interactively.
    """
    # Read metrics
    metrics_df = pd.read_csv(metrics_path)

    # Create figure with subplots
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Plot 1: Loss vs Step
    ax1 = axes[0]
    train_steps = metrics_df[metrics_df["train_loss"].notna()]
    val_steps = metrics_df[metrics_df["val_loss"].notna()]

    ax1.plot(
        train_steps["step"],
        train_steps["train_loss"],
        label="Train Loss",
        alpha=0.7,
        linewidth=1,
    )
    ax1.plot(
        val_steps["step"],
        val_steps["val_loss"],
        label="Val Loss",
        alpha=0.9,
        linewidth=2,
        color="orange",
    )
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training and Validation Loss vs Step")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Loss vs Epoch
    ax2 = axes[1]
    train_epochs = (
        metrics_df[metrics_df["train_loss"].notna()].groupby("epoch")["train_loss"].mean()
    )
    val_epochs = (
        metrics_df[metrics_df["val_loss"].notna()].groupby("epoch")["val_loss"].mean()
    )

    ax2.plot(
        train_epochs.index,
        train_epochs.values,
        label="Train Loss (avg per epoch)",
        alpha=0.7,
        linewidth=2,
        marker="o",
    )
    ax2.plot(
        val_epochs.index,
        val_epochs.values,
        label="Val Loss",
        alpha=0.9,
        linewidth=2,
        marker="s",
        color="orange",
    )
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.set_title("Training and Validation Loss vs Epoch")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save or show
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved to: {output_path}")
    else:
        plt.show()

    # Print summary statistics
    print("\n=== Training Summary ===")
    print(f"Total epochs: {int(metrics_df['epoch'].max())}")
    print(f"Total steps: {int(metrics_df['step'].max())}")
    print(f"\nFinal train loss: {train_steps['train_loss'].iloc[-1]:.6f}")
    print(f"Final val loss: {val_steps['val_loss'].iloc[-1]:.6f}")
    print(
        f"\nBest val loss: {val_steps['val_loss'].min():.6f} "
        f"(at step {val_steps.loc[val_steps['val_loss'].idxmin(), 'step']:.0f})"
    )
    print(f"Min train loss: {train_steps['train_loss'].min():.6f}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot training and validation loss curves from metrics CSV"
    )
    parser.add_argument(
        "metrics_path",
        type=Path,
        help="Path to metrics.csv file",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Path to save the plot (default: display interactively)",
    )

    args = parser.parse_args()

    if not args.metrics_path.exists():
        raise FileNotFoundError(f"Metrics file not found: {args.metrics_path}")

    plot_metrics(args.metrics_path, args.output)


if __name__ == "__main__":
    main()
