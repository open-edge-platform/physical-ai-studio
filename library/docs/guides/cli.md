# CLI Guide

Train policies using the command-line interface built on PyTorch Lightning and jsonargparse.

## Features

- YAML/JSON config files with CLI overrides
- Type-safe configuration (dataclasses, Pydantic)
- Dynamic class loading (`class_path` pattern)
- Full PyTorch Lightning features (callbacks, loggers, distributed training)

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
    --data.train_batch_size 64 \
    --model.optimizer.init_args.lr 0.0001
```

### 4. Train without Config File

```bash
python -m getiaction fit \
    --model.class_path getiaction.policies.dummy.policy.Dummy \
    --model.model.class_path getiaction.policies.dummy.model.Dummy \
    --model.model.action_shape=[7] \
    --model.optimizer.class_path torch.optim.Adam \
    --model.optimizer.init_args.lr=0.001 \
    --data.class_path getiaction.data.lerobot.LeRobotDataModule \
    --data.repo_id=lerobot/pusht \
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

Validate config before full training:

```bash
python -m getiaction fit --config configs/train.yaml --trainer.fast_dev_run=true
```

## Examples

### Quick Start

```bash
python -m getiaction fit --config configs/train_dummy_class_path.yaml
```

### GPU Training

```yaml
trainer:
  accelerator: gpu
  devices: 1
  precision: 16-mixed
  max_epochs: 200
```

### Multi-GPU Training

```bash
python -m getiaction fit --config configs/train.yaml --trainer.strategy=ddp --trainer.devices=4
```

### Custom Callbacks

```yaml
trainer:
  callbacks:
    - class_path: lightning.pytorch.callbacks.EarlyStopping
      init_args:
        patience: 10
        monitor: train/loss
```

### Custom Optimizer

```yaml
model:
  init_args:
    optimizer:
      class_path: torch.optim.AdamW
      init_args:
        lr: 0.001
        weight_decay: 0.00001
```

## Commands

```bash
# Train
python -m getiaction fit --config CONFIG_PATH

# Validate
python -m getiaction validate --config CONFIG_PATH --ckpt_path CHECKPOINT

# Test
python -m getiaction test --config CONFIG_PATH --ckpt_path CHECKPOINT

# Predict
python -m getiaction predict --config CONFIG_PATH --ckpt_path CHECKPOINT
```

## Tips

- Use example configs as templates
- Run `--print_config` to see all defaults
- Validate with `fast_dev_run` before full training
- Version control your configs

## Troubleshooting

### Config errors

Run `--print_config` to see parsed values

### Import errors

Test imports manually:

```bash
python -c "from getiaction.policies.dummy.policy import Dummy"
```

### Type errors

Check config matches class signature
