# Bug: `tl-evaluate` Doesn't Support New Dict Format for Model Configs

## Problem

`tl-evaluate` fails when experiment configs use the new dict format introduced for `tl-finetune`:

```yaml
# This works with tl-finetune but breaks tl-evaluate
models:
  ealstm_natural_kyrgyzstan:
    config: configs/models/ealstm_30k_natural_finetune_kyrgyzstan.yaml
    source_model: ealstm_natural
```

**Error:**
```
TypeError: argument should be a str or an os.PathLike object where __fspath__ returns a str, not 'dict'
```

## Current Workaround

Create separate evaluation experiment configs using the old string format:

```yaml
# evaluate_natural_kyrgyzstan.yaml
models:
  ealstm_natural_kyrgyzstan: configs/models/ealstm_30k_natural_finetune_kyrgyzstan.yaml
```

This duplicates configuration and creates maintenance burden.

## Desired Solution

`tl-evaluate` should support both formats:

**String format (backward compatible):**
```yaml
models:
  tide: configs/models/tide.yaml
```

**Dict format (for consistency with tl-finetune):**
```yaml
models:
  tide_finetuned:
    config: configs/models/tide_finetuned.yaml
    source_model: tide_baseline  # Ignored by tl-evaluate
```

When dict format is used, `tl-evaluate` should:
- Extract the `config` field and use it as the model config path
- Ignore the `source_model` field (only relevant for tl-finetune)

## Implementation Notes

In `orchestrator.py`, the model config path extraction should handle both types:

```python
# Current (breaks on dict)
config_path = Path(self.models_config_paths[model_name])

# Proposed
model_spec = self.models_config_paths[model_name]
if isinstance(model_spec, dict):
    config_path = Path(model_spec['config'])
else:
    config_path = Path(model_spec)
```

## Impact

**Current:** Users must maintain separate experiment configs for fine-tuning and evaluation

**Fixed:** Users can use the same experiment config for both `tl-finetune` and `tl-evaluate`

```bash
# Single experiment config for both operations
uv run tl-finetune experiments/finetune_natural_tajikistan.yaml --seed 42
uv run tl-evaluate experiments/finetune_natural_tajikistan.yaml \
  --eval-config configs/evaluation/tajikistan.yaml \
  --checkpoint-source finetuning
```
