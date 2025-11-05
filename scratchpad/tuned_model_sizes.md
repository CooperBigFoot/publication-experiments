# Tuned Model Parameter Counts

Model sizes after hyperparameter tuning on Central Asian data (63 basins).

## Parameter Counts

| Model | Total Parameters | Benchmark Configs |
|-------|-----------------|-------------------|
| EA-LSTM | 29,654 | `ealstm_30k_{country}_benchmark.yaml` |
| TSMixer | 162,283 | `tsmixer_162k_{country}_benchmark.yaml` |
| TFT | 281,695 | `tft_282k_{country}_benchmark.yaml` |
| TiDE | 613,192 | `tide_613k_{country}_benchmark.yaml` |

## Benchmark Configurations

8 total benchmark configs (4 models × 2 countries):

**Kyrgyzstan (49 basins):**
- `ealstm_30k_kyrgyzstan_benchmark.yaml`
- `tsmixer_162k_kyrgyzstan_benchmark.yaml`
- `tft_282k_kyrgyzstan_benchmark.yaml`
- `tide_613k_kyrgyzstan_benchmark.yaml`

**Tajikistan (14 basins):**
- `ealstm_30k_tajikistan_benchmark.yaml`
- `tsmixer_162k_tajikistan_benchmark.yaml`
- `tft_282k_tajikistan_benchmark.yaml`
- `tide_613k_tajikistan_benchmark.yaml`

## Summary

- EA-LSTM is the smallest model (~30K parameters)
- TiDE is the largest model (~613K parameters)
- All models have 100% trainable parameters (no frozen layers)
- All configs use cosine annealing LR scheduler (T_max: 150, eta_min: 0.00001)
- Hyperparameters are from tuning on all 63 Central Asia basins

## Source

Generated using: `uv run python scripts/count_model_params.py configs/models/<model>_tuned_2025-11-04.yaml`
