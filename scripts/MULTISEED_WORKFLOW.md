# Multi-Seed Ensemble Analysis Workflow

This document outlines the complete workflow for training TFT on multiple seeds, creating an ensemble, and analyzing performance.

## Overview

The workflow consists of 4 steps:
1. **Train** - Train model on 10 different seeds
2. **Evaluate** - Evaluate all 10 seeds on test data
3. **Ensemble** - Create ensemble predictions from the 10 seeds
4. **Analyze** - Generate CDF plots and rolling forecast visualizations

---

## Step 1: Train Models (10 seeds)

Train the TFT model on Kyrgyzstan data with 10 different random seeds:

```bash
uv run tl-train experiments/kyrgyzstan_benchmark.yaml \
  --models tft_kyrgyzstan \
  --n-runs 10 \
  --start-seed 42
```

**Expected output:**
```
checkpoints/training/model_name=tft_282k_kyrgyzstan_benchmark/
├── run_2025-*_seed42/
├── run_2025-*_seed43/
├── run_2025-*_seed44/
├── run_2025-*_seed45/
├── run_2025-*_seed46/
├── run_2025-*_seed47/
├── run_2025-*_seed48/
├── run_2025-*_seed49/
├── run_2025-*_seed50/
└── run_2025-*_seed51/
```

**Time estimate:** ~10-20 hours (depends on GPU, early stopping)

---

## Step 2: Evaluate Models

Evaluate all 10 trained models on the test set:

```bash
uv run tl-evaluate experiments/kyrgyzstan_benchmark.yaml \
  --eval-config configs/evaluation/kyrgyzstan_benchmark.yaml \
  --models tft_kyrgyzstan \
  --seeds 42,43,44,45,46,47,48,49,50,51
```

**Expected output:**
```
results/evaluation/eval_2025-*/
└── model_name=tft_282k_kyrgyzstan_benchmark/
    ├── seed=42/predictions.parquet
    ├── seed=43/predictions.parquet
    ├── seed=44/predictions.parquet
    ├── seed=45/predictions.parquet
    ├── seed=46/predictions.parquet
    ├── seed=47/predictions.parquet
    ├── seed=48/predictions.parquet
    ├── seed=49/predictions.parquet
    ├── seed=50/predictions.parquet
    └── seed=51/predictions.parquet
```

**Time estimate:** ~1-2 hours

---

## Step 3: Create Ensemble

Create ensemble predictions using mean-of-positives across all 10 seeds:

### Option A: Auto-discover (recommended)

```bash
uv run python scripts/create_ensemble_from_seeds.py \
  --model-name tft_282k_kyrgyzstan_benchmark
```

This will:
- Auto-discover the most recent evaluation directory
- Auto-discover all available seeds
- Create ensemble predictions

### Option B: Explicit paths

```bash
uv run python scripts/create_ensemble_from_seeds.py \
  --eval-dir results/evaluation/eval_2025-01-23_143052 \
  --model-name tft_282k_kyrgyzstan_benchmark \
  --seeds 42,43,44,45,46,47,48,49,50,51 \
  --ensemble-name tft_ensemble
```

**Expected output:**
```
results/evaluation/eval_2025-*/
└── model_name=tft_ensemble/  # or model_name=ensemble/
    └── seed=ensemble/
        └── predictions.parquet
```

**Output:**
```
✓ Created mean-of-positives ensemble from 10 seeds
✓ Source model: tft_282k_kyrgyzstan_benchmark
✓ Seeds: [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]
✓ Saved to: results/evaluation/.../model_name=tft_ensemble/seed=ensemble/predictions.parquet
✓ Shape: (12340, 7)
✓ Predictions set to 0 (all seeds negative): 234 (1.89%)
```

---

## Step 4: Analyze Performance

Generate CDF plots and rolling forecast visualization:

```bash
uv run python scripts/analyze_multiseed_performance.py \
  --eval-dir results/evaluation/eval_2025-01-23_143052 \
  --model-name tft_282k_kyrgyzstan_benchmark \
  --ensemble-name tft_ensemble \
  --output-dir figures/multiseed_analysis
```

**Expected output:**
```
figures/multiseed_analysis/
├── cdf_nse_lead1.png          # CDF for 1-day forecasts
├── cdf_nse_lead5.png          # CDF for 5-day forecasts
├── cdf_nse_lead10.png         # CDF for 10-day forecasts
├── rolling_forecast_{basin_id}.png  # Rolling forecast for median basin
└── metrics_by_seed.parquet    # Detailed metrics for further analysis
```

**Output:**
```
============================================================
Multi-Seed Performance Analysis
============================================================

1. Loading predictions...
✓ Loaded data from 10 seeds: [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]

2. Computing NSE metrics...
✓ Computed metrics for 5390 combinations

3. Creating CDF plots...
✓ Saved CDF plot: figures/multiseed_analysis/cdf_nse_lead1.png
✓ Saved CDF plot: figures/multiseed_analysis/cdf_nse_lead5.png
✓ Saved CDF plot: figures/multiseed_analysis/cdf_nse_lead10.png

4. Selecting basin for rolling forecast...
✓ Selected basin: kyrgyz_12345 (median NSE: 0.723)

5. Creating rolling forecast plot...
✓ Saved rolling forecast plot: figures/multiseed_analysis/rolling_forecast_kyrgyz_12345.png

✓ Saved metrics to: figures/multiseed_analysis/metrics_by_seed.parquet

============================================================
Analysis complete!
============================================================
```

---

## Understanding the Outputs

### CDF Plots

Each CDF plot shows:
- **Light blue curves** (alpha=0.3): Individual seeds - empirical CDF of NSE across basins
- **Bold blue curve** (alpha=1.0): Ensemble - empirical CDF of ensemble NSE across basins
- **Red dashed line**: NSE=0 baseline

**Interpretation:**
- Higher curves = better performance
- Ensemble should generally outperform or match individual seeds
- Spread of seed curves shows variance due to random initialization

### Rolling Forecast Plot

The plot shows non-overlapping 10-day forecast windows:
- **Black line**: Ground truth observations
- **Light blue lines** (alpha=0.3): Individual seed predictions
- **Bold blue line** (alpha=1.0): Ensemble predictions

**Interpretation:**
- Ensemble should smooth out individual seed variability
- Shows how forecasts evolve over multiple 10-day horizons
- Basin selected automatically based on median NSE performance

### Metrics Parquet

The `metrics_by_seed.parquet` file contains:
- NSE values for each (model, seed, basin, lead_time) combination
- Can be loaded for custom analysis:

```python
import polars as pl

metrics = pl.read_parquet("figures/multiseed_analysis/metrics_by_seed.parquet")

# Compare ensemble vs individual seeds
summary = (
    metrics
    .group_by("model_name")
    .agg([
        pl.col("NSE").mean().alias("NSE_mean"),
        pl.col("NSE").std().alias("NSE_std"),
        pl.col("NSE").median().alias("NSE_median")
    ])
)
```

---

## Quick Reference

### Full pipeline in one go

```bash
# 1. Train (takes longest)
uv run tl-train experiments/kyrgyzstan_benchmark.yaml \
  --models tft_kyrgyzstan --n-runs 10 --start-seed 42

# 2. Evaluate
uv run tl-evaluate experiments/kyrgyzstan_benchmark.yaml \
  --eval-config configs/evaluation/kyrgyzstan_benchmark.yaml \
  --models tft_kyrgyzstan --seeds 42,43,44,45,46,47,48,49,50,51

# 3. Create ensemble
uv run python scripts/create_ensemble_from_seeds.py \
  --model-name tft_282k_kyrgyzstan_benchmark

# 4. Analyze
uv run python scripts/analyze_multiseed_performance.py \
  --eval-dir results/evaluation/eval_2025-* \
  --model-name tft_282k_kyrgyzstan_benchmark \
  --ensemble-name ensemble \
  --output-dir figures/multiseed_analysis
```

---

## Troubleshooting

### "No evaluation directories found"

Run step 2 (tl-evaluate) first.

### "Prediction file not found for seed X"

Check that all seeds completed successfully in step 2. Use `--seeds` to specify only available seeds.

### "No prediction_date column found"

Your model config must have `include_dates: true` in the `data_preparation` section for rolling forecasts.

### Custom basin for rolling forecast

```bash
uv run python scripts/analyze_multiseed_performance.py \
  --eval-dir results/evaluation/eval_2025-* \
  --model-name tft_282k_kyrgyzstan_benchmark \
  --ensemble-name ensemble \
  --basin-id kyrgyz_12345  # Specify your basin
```

---

## Notes

- The ensemble uses **mean-of-positives**: averages only positive predictions; if all seeds predict ≤0, ensemble predicts 0
- CDF plots remove NaN NSE values (basins with constant observations)
- Rolling forecast shows first 10 non-overlapping windows by default
- All scripts support `--help` for detailed options
