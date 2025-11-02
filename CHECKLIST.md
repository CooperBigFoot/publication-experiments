# Pre-Tuning Checklist

Before running `tl-tune`, complete these tasks:

## 1. Data Preparation
- [ ] Basin IDs file created in `configs/basin_ids_files/`
- [ ] Data directory exists with train/val/test splits
- [ ] Preprocessing pipeline (ts_pipeline.joblib) exists
- [ ] Data follows hive-partitioned structure

## 2. Model Config Updates (Do for ALL 4 models)

### configs/models/ealstm_base.yaml
- [ ] Update `data.base_path`
- [ ] Update `data.gauge_ids_file`
- [ ] Update `data.pipeline_path`
- [ ] Update `features.forcing` list
- [ ] Update `features.static` list
- [ ] Update `features.future` list (exclude streamflow!)
- [ ] Update `sequence.input_length`
- [ ] Update `sequence.output_length`
- [ ] Verify `scheduler_kwargs.T_max` matches experiment max_epochs

### configs/models/tide_base.yaml
- [ ] Update `data.base_path`
- [ ] Update `data.gauge_ids_file`
- [ ] Update `data.pipeline_path`
- [ ] Update `features.forcing` list
- [ ] Update `features.static` list
- [ ] Update `features.future` list
- [ ] Update `sequence.input_length`
- [ ] Update `sequence.output_length`
- [ ] Verify `scheduler_kwargs.T_max` matches experiment max_epochs

### configs/models/tsmixer_base.yaml
- [ ] Update `data.base_path`
- [ ] Update `data.gauge_ids_file`
- [ ] Update `data.pipeline_path`
- [ ] Update `features.forcing` list
- [ ] Update `features.static` list
- [ ] Update `features.future` list
- [ ] Update `sequence.input_length`
- [ ] Update `sequence.output_length`
- [ ] Verify `scheduler_kwargs.T_max` matches experiment max_epochs

### configs/models/tft_base.yaml
- [ ] Update `data.base_path`
- [ ] Update `data.gauge_ids_file`
- [ ] Update `data.pipeline_path`
- [ ] Update `features.forcing` list
- [ ] Update `features.static` list
- [ ] Update `features.future` list
- [ ] Update `sequence.input_length`
- [ ] Update `sequence.output_length`
- [ ] Verify `scheduler_kwargs.T_max` matches experiment max_epochs

## 3. Experiment Config
- [ ] Review `experiments/baseline.yaml` trainer settings
- [ ] Adjust `max_epochs` if needed (default: 150)
- [ ] Adjust `early_stopping_patience` if needed (default: 15)

## 4. Optional: Data Subset for Faster Tuning
- [ ] Create smaller basin subset file (e.g., 100 basins)
- [ ] Add `tuning_data` section to model configs

## 5. Verify Setup
```bash
# Test that configs load correctly
uv run python -c "
from pathlib import Path
import yaml

# Test each model config
for model in ['ealstm', 'tide', 'tsmixer', 'tft']:
    config_path = Path(f'configs/models/{model}_base.yaml')
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    print(f'✓ {model}_base.yaml loaded successfully')

    # Check for TODO markers
    with open(config_path) as f:
        content = f.read()
    if 'TODO:' in content:
        print(f'  ⚠️  Contains TODO items - please update!')
    else:
        print(f'  ✓ No TODO items found')
"
```

## 6. Ready to Tune!

Once all checkboxes are complete:

```bash
# Quick test (10 trials, 10 epochs)
uv run tl-tune experiments/baseline.yaml \
  --search-space configs/search_spaces/ealstm_search.yaml \
  --model ealstm \
  --n-trials 10 \
  --max-epochs 10

# Full tuning (50 trials, 30 epochs)
uv run tl-tune experiments/baseline.yaml \
  --search-space configs/search_spaces/ealstm_search.yaml \
  --model ealstm \
  --n-trials 50 \
  --max-epochs 30
```

## Common Issues

**"FileNotFoundError: pipeline not found"**
→ Check `data.pipeline_path` in model config

**"ValueError: Target not in forcing features"**
→ Add `streamflow` to `features.forcing`

**"ValueError: BiEALSTM requires future forcing features"**
→ Add `features.future` list to EA-LSTM config

**"No basins found"**
→ Check `data.gauge_ids_file` path and contents
