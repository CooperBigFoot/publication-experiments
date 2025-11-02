# Setup Guide for Hyperparameter Tuning with tl-tune

This repository is now configured for hyperparameter optimization using the `tl-tune` CLI from the transfer-learning-publication package.

## Repository Structure

```
publication-experiments/
├── configs/
│   ├── basin_ids_files/          # Put your basin ID files here
│   ├── models/                    # Model configuration templates
│   │   ├── ealstm_base.yaml
│   │   ├── tide_base.yaml
│   │   ├── tsmixer_base.yaml
│   │   └── tft_base.yaml
│   └── search_spaces/             # Hyperparameter search spaces
│       ├── ealstm_search.yaml
│       ├── tide_search.yaml
│       ├── tsmixer_search.yaml
│       └── tft_search.yaml
├── experiments/
│   └── baseline.yaml              # Experiment configuration
└── SETUP_GUIDE.md                 # This file
```

## Before Running tl-tune

You need to update the model configuration files with your specific data paths and features.

### Step 1: Prepare Your Basin IDs File

Create a text file with one basin/gauge ID per line:

```bash
# Example: configs/basin_ids_files/my_basins.txt
basin_001
basin_002
basin_003
```

### Step 2: Update Model Configurations

Edit each model config file (`configs/models/*_base.yaml`) and update the following sections marked with `# TODO:`:

#### Data Section
```yaml
data:
  base_path: ../path/to/your/data  # Update this
  gauge_ids_file: configs/basin_ids_files/your_basins.txt  # Update this
  pipeline_path: ../path/to/your/data/ts_pipeline.joblib  # Update this
```

#### Features Section
```yaml
features:
  forcing:
    # Update with YOUR meteorological forcing features
    - temperature_2m_max
    - temperature_2m_min
    # ... add your features
    - streamflow  # Target variable (can be anywhere in the list)

  static:
    # Update with YOUR catchment attributes
    - area
    - elevation
    # ... add your attributes

  future:
    # Specify which forcing features are known in the future
    # Required for forecast mode - typically ALL meteorological variables
    - temperature_2m_max
    - temperature_2m_min
    # ... add your future-known features
    # Note: streamflow should NOT be in future features

  target: streamflow
```

#### Sequence Section
```yaml
sequence:
  input_length: 150  # Adjust for your data (e.g., 365 for daily data)
  output_length: 1   # Forecast horizon
```

#### Scheduler T_max
```yaml
model:
  overrides:
    scheduler_kwargs:
      T_max: 150  # Should match trainer.max_epochs in experiment config
```

### Step 3: Verify Experiment Configuration

The experiment config (`experiments/baseline.yaml`) is ready to use. You can adjust:
- `max_epochs`: Number of epochs per trial (currently 150)
- `early_stopping_patience`: Patience for early stopping (currently 15)
- `gradient_clip_val`: Gradient clipping threshold (currently 1.0)

## Forecast Mode Configuration

All models are configured for **forecast mode**:

✅ **What this means:**
- Predicts **future** streamflow (at t+1, t+2, ...)
- Uses past streamflow values (is_autoregressive: true)
- Uses future meteorological forcing (specified in `features.future`)
- EA-LSTM uses bidirectional architecture (bidirectional: true)

✅ **Key settings:**
```yaml
data_preparation:
  mode: forecast              # Forecast mode (default)
  is_autoregressive: true     # Use past streamflow
  include_dates: true         # Include timestamps

# For EA-LSTM only:
model:
  overrides:
    bidirectional: true       # Required for forecast mode
```

## Running Hyperparameter Tuning

Once you've updated the model configs with your data paths and features:

### Tune Individual Models

```bash
# Tune EA-LSTM
uv run tl-tune experiments/baseline.yaml \
  --search-space configs/search_spaces/ealstm_search.yaml \
  --model ealstm \
  --n-trials 50 \
  --max-epochs 30

# Tune TiDE
uv run tl-tune experiments/baseline.yaml \
  --search-space configs/search_spaces/tide_search.yaml \
  --model tide \
  --n-trials 50 \
  --max-epochs 30

# Tune TSMixer
uv run tl-tune experiments/baseline.yaml \
  --search-space configs/search_spaces/tsmixer_search.yaml \
  --model tsmixer \
  --n-trials 50 \
  --max-epochs 30

# Tune TFT
uv run tl-tune experiments/baseline.yaml \
  --search-space configs/search_spaces/tft_search.yaml \
  --model tft \
  --n-trials 50 \
  --max-epochs 30
```

### Tuning Tips

1. **Start with fewer trials for testing:**
   ```bash
   --n-trials 10 --max-epochs 10
   ```

2. **Use a data subset for faster tuning:**
   Add to your model configs:
   ```yaml
   tuning_data:
     base_path: ../path/to/your/data
     gauge_ids_file: configs/basin_ids_files/tuning_subset_100.txt
   ```

3. **Monitor progress:**
   - Real-time updates show trial results
   - Results saved to `checkpoints/tuning/study_*/`
   - Best parameters saved to `checkpoints/tuning/study_*/best_params.yaml`

## After Tuning

`tl-tune` automatically generates optimized configs:

```
configs/models/
├── ealstm_tuned_2024-11-02.yaml
├── tide_tuned_2024-11-02.yaml
├── tsmixer_tuned_2024-11-02.yaml
└── tft_tuned_2024-11-02.yaml
```

Use these for training:

```bash
# Train with optimized hyperparameters
uv run tl-train configs/models/ealstm_tuned_2024-11-02.yaml --n-runs 5
```

## Search Space Details

### Learning Rate Range
All models: **0.000001 to 0.001** (log scale)

### Model-Specific Parameters

**EA-LSTM:**
- hidden_size: 32-256
- num_layers: 1-3
- future_hidden_size: 32-128 (for bidirectional)
- future_layers: 1-2
- bidirectional_fusion: concat/add/average

**TiDE:**
- hidden_size: 64-256
- num_encoder_layers: 1-4
- num_decoder_layers: 1-4
- decoder_output_size: 8-64
- Feature projections: 0, 16, 32, 64

**TSMixer:**
- hidden_size: 32-256
- num_mixing_layers: 2-8
- static_embedding_size: 4-32
- fusion_method: add/concat

**TFT:**
- hidden_size: 32-128
- lstm_layers: 1-3
- num_attention_heads: 1, 2, 4, 8
- encoder_layers: 1-3

## Troubleshooting

### "Missing required config section"
Make sure all TODO items in model configs are updated.

### "No basins found"
Check that:
- `base_path` points to valid data directory
- `gauge_ids_file` exists and contains valid basin IDs
- Data structure matches expected hive-partitioned format

### "Feature not found"
Ensure all features listed in `forcing` and `static` exist in your data files.

### "BiEALSTM requires future forcing features"
For EA-LSTM with bidirectional=true, you must specify `features.future`.

## Questions?

Refer to the comprehensive documentation:
- CLI Guide: `/Users/nicolaslazaro/Desktop/work/transfer-learning-publication/docs/cli_guide.md`
- Configuration Guide: `/Users/nicolaslazaro/Desktop/work/transfer-learning-publication/docs/configuration_guide.md`
