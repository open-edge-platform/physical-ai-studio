# CLI Usage Guide

GetiAction provides a powerful command-line interface built on PyTorch
Lightning CLI and jsonargparse. This CLI supports multiple configuration
patterns and provides full flexibility for training policies.

## Features

✅ **Multiple Config Patterns**

- Dataclass/Pydantic configs (type-safe)
- jsonargparse `class_path` pattern (dynamic)
- Mixed approach

✅ **Full jsonargparse Support**

- YAML/JSON config files
- Command-line argument overrides
- Config composition and inheritance

✅ **PyTorch Lightning Integration**

- All Lightning Trainer features
- Callbacks, loggers, and plugins
- Distributed training support

✅ **Type Safety**

- Automatic type validation from type hints
- Dataclass support
- Pydantic model support

## Installation

```bash
# Install with jsonargparse support
pip install getiaction

# Or from source
cd library
pip install -e "."
```

## Basic Usage

### 1. Train with YAML Config

```bash
python -m getiaction fit --config configs/train_dummy_class_path.yaml
```

### 2. Generate Config Template

```bash
# See all available options
python -m getiaction fit --help

# Print default config
python -m getiaction fit --print_config
```

### 3. Override Config from CLI

```bash
python -m getiaction fit \
    --config configs/train_dummy_class_path.yaml \
    --trainer.max_epochs 200 \
    --data.init_args.train_batch_size 64 \
    --model.init_args.optimizer.init_args.lr 0.0001
```

### 4. Train without Config File

```bash
python -m getiaction fit \
    --model.class_path getiaction.policies.dummy.policy.Dummy \
    --model.init_args.model.class_path getiaction.policies.dummy.model.Dummy \
    --model.init_args.model.init_args.action_shape=[7] \
    --model.init_args.optimizer.class_path torch.optim.Adam \
    --model.init_args.optimizer.init_args.lr=0.001 \
    --data.class_path getiaction.data.lerobot.LeRobotDataModule \
    --data.init_args.repo_id=lerobot/pusht \
    --trainer.max_epochs=100
```

## Configuration Patterns

### Pattern 1: Dataclass/Pydantic (Type-Safe)

Create strongly-typed configs using Python dataclasses or Pydantic:

```python
from dataclasses import dataclass
from getiaction.policies.dummy.config import DummyConfig, DummyModelConfig, OptimizerConfig

@dataclass
class TrainConfig:
    seed: int = 42
    model: DummyConfig
    max_epochs: int = 100
```

### Pattern 2: jsonargparse class_path (Dynamic)

Use `class_path` for maximum flexibility:

```yaml
model:
  class_path: getiaction.policies.dummy.policy.Dummy
  init_args:
    model:
      class_path: getiaction.policies.dummy.model.Dummy
      init_args:
        action_shape: [7]
    optimizer:
      class_path: torch.optim.Adam
      init_args:
        lr: 0.001
```

### Pattern 3: Mixed Approach

Combine both patterns for flexibility + type safety:

```yaml
# Use dataclass for structured configs
model_config:
  action_shape: [7]
  n_action_steps: 4

# Use class_path for dynamic components
model:
  class_path: getiaction.policies.dummy.policy.Dummy
  init_args:
    model: ${model_config}
```

## Advanced Features

### Config Composition

```yaml
# base_config.yaml
trainer:
  max_epochs: 100
  accelerator: auto

# experiment.yaml
__base__: base_config.yaml  # Inherit from base
trainer:
  max_epochs: 200  # Override specific values
```

### Environment Variables

```bash
export GETI_ACTION_EPOCHS=200
python -m getiaction fit \
    --config configs/train.yaml \
    --trainer.max_epochs=${GETI_ACTION_EPOCHS}
```

### Multiple Configs

```bash
# Merge multiple config files
python -m getiaction fit \
    --config configs/base.yaml \
    --config configs/experiment.yaml
```

### Validation

```bash
# Validate config without training
python -m getiaction fit \
    --config configs/train.yaml \
    --trainer.fast_dev_run=true
```

## Examples

### Example 1: Quick Start

```bash
# Train Dummy policy on PushT dataset
python -m getiaction fit \
    --config configs/train_dummy_class_path.yaml
```

### Example 2: GPU Training with Mixed Precision

```yaml
trainer:
  accelerator: gpu
  devices: 1
  precision: 16-mixed
  max_epochs: 200
```

### Example 3: Distributed Training

```bash
# Multi-GPU training
python -m getiaction fit \
    --config configs/train.yaml \
    --trainer.accelerator=gpu \
    --trainer.devices=4 \
    --trainer.strategy=ddp
```

### Example 4: Custom Callbacks

```yaml
trainer:
  callbacks:
    - class_path: lightning.pytorch.callbacks.EarlyStopping
      init_args:
        patience: 10
        monitor: train/loss

    - class_path: lightning.pytorch.callbacks.ModelCheckpoint
      init_args:
        save_top_k: 3
        monitor: train/loss
```

### Example 5: Custom Optimizer and Scheduler

```yaml
model:
  init_args:
    optimizer:
      class_path: torch.optim.AdamW
      init_args:
        lr: 0.001
        weight_decay: 0.00001

lr_scheduler:
  class_path: torch.optim.lr_scheduler.CosineAnnealingLR
  init_args:
    T_max: 200
    eta_min: 0.00001
```

## CLI Commands

### fit

Train a model:

```bash
python -m getiaction fit --config CONFIG_PATH
```

### validate

Run validation only:

```bash
python -m getiaction validate \
    --config CONFIG_PATH \
    --ckpt_path CHECKPOINT_PATH
```

### test

Run testing:

```bash
python -m getiaction test \
    --config CONFIG_PATH \
    --ckpt_path CHECKPOINT_PATH
```

### predict

Run predictions:

```bash
python -m getiaction predict \
    --config CONFIG_PATH \
    --ckpt_path CHECKPOINT_PATH
```

## Tips and Best Practices

1. **Start with example configs**: Use provided configs as templates
2. **Use `--print_config`**: Generate full config with all defaults
3. **Validate first**: Use `fast_dev_run` to catch errors quickly
4. **Version control configs**: Keep configs in git for reproducibility
5. **Use type hints**: jsonargparse automatically validates types
6. **Leverage inheritance**: Create base configs and extend them

## Troubleshooting

### Config Validation Errors

```bash
# Check config structure
python -m getiaction fit --config CONFIG_PATH --parser.error_handler=ignore

# Print parsed config
python -m getiaction fit --config CONFIG_PATH --print_config
```

### Import Errors

Ensure all classes are importable:

```python
# Test imports
python -c "from getiaction.policies.dummy.policy import Dummy"
```

### Type Validation

jsonargparse validates types automatically. If you get type errors:

1. Check your config matches the class signature
2. Use `--parser.default_meta=false` to disable strict validation
3. Check type hints in your classes
