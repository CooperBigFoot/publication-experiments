# Feature Request: Separate Source and Target Model Names in `tl-finetune`

## Problem

Currently, `tl-finetune` uses the experiment model name for **both**:

1. Finding training checkpoints to load
2. Naming the fine-tuned checkpoint directories

This prevents fine-tuning the same base model on multiple target datasets in separate runs.

## Example Scenario

Training produces:

```
checkpoints/training/model_name=ealstm_natural/run_2025-11-05_seed42/best_val_loss_0.0994.ckpt
```

We want to fine-tune this model on two different regions:

- Tajikistan (13 basins)
- Kyrgyzstan (49 basins)

Current behavior forces both to save to the same location:

```
checkpoints/finetuning/model_name=ealstm_natural/run_2025-11-06_seed42/
```

This causes:

- Second run overwrites first run
- No way to distinguish which region the fine-tuned model was trained on
- Must manually rename/move outputs between runs

## Desired Solution

Allow specifying **different names** for source and target:

```yaml
# experiments/finetune_natural_tajikistan.yaml
models:
  ealstm_natural_tajikistan:  # Target name (for saving fine-tuned checkpoints)
    config: configs/models/ealstm_30k_natural_finetune_tajikistan.yaml
    base_model: ealstm_natural  # Source name (for finding training checkpoints)
```

This would:

- Load from: `checkpoints/training/model_name=ealstm_natural/...`
- Save to: `checkpoints/finetuning/model_name=ealstm_natural_tajikistan/...`

## Alternative Solutions

**Option A: Add `base_model` field to experiment config**

```yaml
models:
  ealstm_natural_tajikistan:
    config: configs/models/ealstm_30k_natural_finetune_tajikistan.yaml
    base_model: ealstm_natural  # NEW: explicitly specify source model name
```

**Option B: Add CLI flag**

```bash
tl-finetune experiments/finetune_tajikistan.yaml \
  --base-model-mapping ealstm_natural_tajikistan:ealstm_natural
```

**Option C: Infer from checkpoint path**
Add optional `checkpoint_path` to experiment config that overrides checkpoint discovery

## Use Case Frequency

Common for transfer learning experiments:

- Fine-tuning on multiple downstream tasks
- Fine-tuning same base model on different domains/regions
- A/B testing fine-tuning strategies

## Workaround

Currently requires sequential runs with manual file organization:

```bash
tl-finetune experiment_tajikistan.yaml
mv checkpoints/finetuning/model_name=X checkpoints/finetuning/model_name=X_tajikistan
tl-finetune experiment_kyrgyzstan.yaml
```
