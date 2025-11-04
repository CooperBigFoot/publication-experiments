# Central Asia Benchmark Configuration Setup

**Date:** 2025-11-04
**Purpose:** Configure benchmark models for Kyrgyzstan and Tajikistan basins

## Summary

Created complete configuration setup for training and evaluating 4 model architectures on 2 countries:
- **8 model configs** (4 architectures × 2 countries)
- **2 experiment configs** (1 per country for batch training)
- **2 evaluation configs** (1 per country for test evaluation)

## Model Configurations Created

### Kyrgyzstan (49 basins)
- `configs/models/ealstm_kyrgyzstan_benchmark.yaml` (256,007 params)
- `configs/models/tide_kyrgyzstan_benchmark.yaml` (263,967 params)
- `configs/models/tsmixer_kyrgyzstan_benchmark.yaml` (255,380 params)
- `configs/models/tft_kyrgyzstan_benchmark.yaml` (248,131 params)

### Tajikistan (14 basins)
- `configs/models/ealstm_tajikistan_benchmark.yaml` (256,007 params)
- `configs/models/tide_tajikistan_benchmark.yaml` (263,967 params)
- `configs/models/tsmixer_tajikistan_benchmark.yaml` (255,380 params)
- `configs/models/tft_tajikistan_benchmark.yaml` (248,131 params)

## Experiment Configurations

- `experiments/kyrgyzstan_benchmark.yaml` - All 4 models for Kyrgyzstan
- `experiments/tajikistan_benchmark.yaml` - All 4 models for Tajikistan

## Evaluation Configurations

- `configs/evaluation/kyrgyzstan_benchmark.yaml` - Test evaluation for Kyrgyzstan
- `configs/evaluation/tajikistan_benchmark.yaml` - Test evaluation for Tajikistan

## Configuration Details

**Data:**
- Base path: `../publication_data_transformed/central_asia`
- Pipeline: `../publication_data_transformed/central_asia/ts_pipeline.joblib`
- Basin stats: `configs/basin_stats/basin_stats_central_asia.json`

**Model Setup:**
- Mode: `forecast` (autoregressive with past streamflow)
- Sequence: `input_length=150`, `output_length=10`
- Loss function: `basin_nse` with `epsilon=0.1`
- Scheduler: `cosine_annealing` (T_max=150, eta_min=0.00001)

**Training:**
- Mode: `evaluation` (60/20/20 train/val/test split)
- Max epochs: 150
- Early stopping: enabled
- Batch size: 2048
- Learning rate: 0.0001 (all models)

## Usage

### Training

**Train all models for one country:**
```bash
uv run tl-train experiments/kyrgyzstan_benchmark.yaml
uv run tl-train experiments/tajikistan_benchmark.yaml
```

**Train specific models:**
```bash
uv run tl-train experiments/kyrgyzstan_benchmark.yaml --models ealstm,tide
```

**Multi-seed training:**
```bash
uv run tl-train experiments/kyrgyzstan_benchmark.yaml --n-runs 5 --start-seed 42
```

**Single model:**
```bash
uv run tl-train configs/models/ealstm_kyrgyzstan_benchmark.yaml
```

### Evaluation

**Evaluate all trained models:**
```bash
uv run tl-evaluate experiments/kyrgyzstan_benchmark.yaml \
  --eval-config configs/evaluation/kyrgyzstan_benchmark.yaml

uv run tl-evaluate experiments/tajikistan_benchmark.yaml \
  --eval-config configs/evaluation/tajikistan_benchmark.yaml
```

**Evaluate specific models:**
```bash
uv run tl-evaluate experiments/kyrgyzstan_benchmark.yaml \
  --eval-config configs/evaluation/kyrgyzstan_benchmark.yaml \
  --models ealstm,tide
```

**Evaluate specific seeds:**
```bash
uv run tl-evaluate experiments/kyrgyzstan_benchmark.yaml \
  --eval-config configs/evaluation/kyrgyzstan_benchmark.yaml \
  --seeds 42,43,44
```

## Output Structure

**Training checkpoints:**
```
checkpoints/training/
├── model_name=ealstm/
│   ├── run_YYYY-MM-DD_seed42/
│   │   ├── checkpoints/best_val_loss_*.ckpt
│   │   └── metrics.csv
│   └── run_YYYY-MM-DD_seed43/
├── model_name=tide/
├── model_name=tsmixer/
└── model_name=tft/
```

**Evaluation results:**
```
results/evaluation/eval_YYYY-MM-DD_HHMMSS/
├── model_name=ealstm/
│   └── seed=42/
│       ├── predictions.parquet
│       └── eval_metadata.json
├── model_name=tide/
├── model_name=tsmixer/
└── model_name=tft/
```

## Notes

- All models use ~250k parameters for fair comparison
- Basin NSE loss ensures equal weighting across basins of different scales
- Cosine annealing scheduler smoothly reduces learning rate over 150 epochs
- Evaluation mode provides validation metrics for model comparison
- Pipeline path is consistent across training and evaluation for proper inverse transforms
