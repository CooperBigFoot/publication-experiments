# Transfer Learning Configuration Setup

**Date:** 2025-11-05
**Purpose:** Create configurations for training "challenger" models on CARAVAN data for transfer learning to Tajikistan/Kyrgyzstan

## Overview

Created 8 model configs (4 architectures × 2 data sources) for pre-training models that will later be finetuned on country-specific data. All configs use identical hyperparameters to the benchmark models but differ only in the `data:` section.

## Model Configs Created

### Natural Basins Set (4 configs)
Train on natural, unregulated basins from CARAVAN dataset:
- `configs/models/ealstm_30k_natural_basins.yaml`
- `configs/models/tsmixer_162k_natural_basins.yaml`
- `configs/models/tft_282k_natural_basins.yaml`
- `configs/models/tide_613k_natural_basins.yaml`

**Data configuration:**
```yaml
data:
  base_path: ../publication_data_transformed/caravan
  gauge_ids_file: configs/basin_ids_files/natural_basins_filtered.txt
  pipeline_path: ../publication_data_transformed/caravan/ts_pipeline.joblib
```

### Clusters 2&4 Set (4 configs)
Train on climatically similar basins (PCA clusters 2 & 4) from CARAVAN dataset:
- `configs/models/ealstm_30k_clusters_2_4.yaml`
- `configs/models/tsmixer_162k_clusters_2_4.yaml`
- `configs/models/tft_282k_clusters_2_4.yaml`
- `configs/models/tide_613k_clusters_2_4.yaml`

**Data configuration:**
```yaml
data:
  base_path: ../publication_data_transformed/caravan
  gauge_ids_file: configs/basin_ids_files/clusters_2_4_basin_ids.txt
  pipeline_path: ../publication_data_transformed/caravan/ts_pipeline.joblib
```

## Experiment Configs Created

### Training Experiments (2)
- `experiments/natural_basins_transfer_learning.yaml` - Train all 4 models on natural basins
- `experiments/clusters_2_4_transfer_learning.yaml` - Train all 4 models on clusters 2&4

Both use:
- Evaluation mode (60/20/20 train/val/test split)
- 150 max epochs
- Early stopping enabled
- Same hyperparameters as Central Asia tuned models

## Evaluation Configs Created

### Stage 1: In-Domain Evaluation (2 configs)
Evaluate models on same data they were trained on:
- `configs/evaluation/natural_basins_transfer_learning.yaml`
- `configs/evaluation/clusters_2_4_transfer_learning.yaml`

### Stage 2: Zero-Shot Transfer Evaluation (2 configs)
Evaluate pretrained models on target domains **before finetuning**:
- `configs/evaluation/tajikistan_zero_shot.yaml`
- `configs/evaluation/kyrgyzstan_zero_shot.yaml`

Both use:
```yaml
data:
  base_path: ../publication_data_transformed/central_asia
  gauge_ids_file: configs/basin_ids_files/{country}_basins.txt
  pipeline_path: ../publication_data_transformed/central_asia/ts_pipeline.joblib
```

## Workflow

### 1. Train Base Models
```bash
# Train natural basins models
uv run tl-train experiments/natural_basins_transfer_learning.yaml --n-runs 5

# Train clusters 2&4 models
uv run tl-train experiments/clusters_2_4_transfer_learning.yaml --n-runs 5
```

### 2. Evaluate In-Domain Performance
```bash
# Evaluate on source domain
uv run tl-evaluate experiments/natural_basins_transfer_learning.yaml \
  --eval-config configs/evaluation/natural_basins_transfer_learning.yaml

uv run tl-evaluate experiments/clusters_2_4_transfer_learning.yaml \
  --eval-config configs/evaluation/clusters_2_4_transfer_learning.yaml
```

### 3. Evaluate Zero-Shot Transfer
```bash
# Natural basins → Tajikistan (zero-shot)
uv run tl-evaluate experiments/natural_basins_transfer_learning.yaml \
  --eval-config configs/evaluation/tajikistan_zero_shot.yaml

# Natural basins → Kyrgyzstan (zero-shot)
uv run tl-evaluate experiments/natural_basins_transfer_learning.yaml \
  --eval-config configs/evaluation/kyrgyzstan_zero_shot.yaml

# Clusters 2&4 → Tajikistan (zero-shot)
uv run tl-evaluate experiments/clusters_2_4_transfer_learning.yaml \
  --eval-config configs/evaluation/tajikistan_zero_shot.yaml

# Clusters 2&4 → Kyrgyzstan (zero-shot)
uv run tl-evaluate experiments/clusters_2_4_transfer_learning.yaml \
  --eval-config configs/evaluation/kyrgyzstan_zero_shot.yaml
```

### 4. Finetune on Target Domains
```bash
# After evaluating zero-shot performance, finetune on country-specific data
uv run tl-finetune experiments/tajikistan_benchmark.yaml \
  --checkpoint-source training \
  --lr-reduction 25

uv run tl-finetune experiments/kyrgyzstan_benchmark.yaml \
  --checkpoint-source training \
  --lr-reduction 25
```

## Key Differences from Benchmarks

| Aspect | Benchmark Models | Transfer Learning Models |
|--------|-----------------|-------------------------|
| Data source | `central_asia` | `caravan` |
| Basin selection | Country-specific (14-18 basins) | Natural basins or clusters 2&4 (hundreds of basins) |
| Purpose | Direct training on target domain | Pre-training for finetuning |
| Pipeline | Central Asia pipeline | CARAVAN pipeline |

## Notes

- All model hyperparameters match the Central Asia tuned values
- Features, sequences, and data preparation settings are identical to benchmarks
- Only the `data:` section differs between benchmarks and transfer learning configs
- Evaluation configs handle pipeline switching automatically (CARAVAN → Central Asia)
