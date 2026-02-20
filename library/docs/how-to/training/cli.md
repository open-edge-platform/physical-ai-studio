# CLI Guide

Train policies using the command-line interface built on PyTorch Lightning CLI.

## Features

- YAML/JSON config files with CLI overrides
- Type-safe configuration (dataclasses, Pydantic)
- Dynamic class loading (`class_path` pattern)
- Full PyTorch Lightning features (callbacks, loggers, distributed training)

## Basic Usage

### Train with Config File

```bash
physicalai fit --config configs/physicalai/act.yaml
```

### Generate Config Template

```bash
# See all options
physicalai fit --help

# Print default config
physicalai fit --print_config
```

### Override Config from CLI

```bash
physicalai fit \
    --config configs/train.yaml \
    --trainer.max_epochs 200 \
    --data.train_batch_size 64 \
    --model.optimizer.init_args.lr 0.0001
```

### Train without Config File

```bash
physicalai fit \
    --model.class_path physicalai.policies.dummy.policy.Dummy \
    --model.model.class_path physicalai.policies.dummy.model.Dummy \
    --model.model.action_shape=[7] \
    --model.optimizer.class_path torch.optim.Adam \
    --model.optimizer.init_args.lr=0.001 \
    --data.class_path physicalai.data.lerobot.LeRobotDataModule \
    --data.repo_id=lerobot/pusht \
    --trainer.max_epochs=100
```

## Configuration Patterns

### Pattern 1: class_path (Dynamic)

Use `class_path` for maximum flexibility:

```yaml
model:
  class_path: physicalai.policies.dummy.policy.Dummy
  init_args:
    model:
      class_path: physicalai.policies.dummy.model.Dummy
      init_args:
        action_shape: [7]
    optimizer:
      class_path: torch.optim.Adam
      init_args:
        lr: 0.001
```

### Pattern 2: Dataclass/Pydantic (Type-Safe)

Create strongly-typed configs:

```python test="skip" reason="illustrative pattern, not runnable as-is"
from dataclasses import dataclass

@dataclass
class TrainConfig:
    seed: int = 42
    max_epochs: int = 100
```

### Config Composition

```yaml
# base_config.yaml
trainer:
  max_epochs: 100
  accelerator: auto

# experiment.yaml
__base__: base_config.yaml
trainer:
  max_epochs: 200  # Override
```

### Multiple Configs

```bash
physicalai fit \
    --config configs/base.yaml \
    --config configs/experiment.yaml
```

## Commands

| Command    | Description         |
| ---------- | ------------------- |
| `fit`      | Train a model       |
| `validate` | Run validation      |
| `test`     | Run test evaluation |
| `predict`  | Run inference       |

```bash
physicalai fit --config CONFIG_PATH
physicalai validate --config CONFIG_PATH --ckpt_path CHECKPOINT
physicalai test --config CONFIG_PATH --ckpt_path CHECKPOINT
physicalai predict --config CONFIG_PATH --ckpt_path CHECKPOINT
```

## Examples

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
physicalai fit --config configs/physicalai/act.yaml --trainer.strategy=ddp --trainer.devices=4
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

### Validate Before Training

```bash
physicalai fit --config configs/physicalai/act.yaml --trainer.fast_dev_run=true
```

## Tips

- Use example configs as templates
- Run `--print_config` to see all defaults
- Validate with `fast_dev_run` before full training
- Version control your configs

## Troubleshooting

**Config errors**: Run `--print_config` to see parsed values

**Import errors**: Test imports manually:

```bash
python -c "from physicalai.policies.pi0.policy import Pi0"
```

**Type errors**: Check config matches class signature
