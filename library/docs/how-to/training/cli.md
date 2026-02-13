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
getiaction fit --config configs/getiaction/act.yaml
```

### Generate Config Template

```bash
# See all options
getiaction fit --help

# Print default config
getiaction fit --print_config
```

### Override Config from CLI

```bash
getiaction fit \
    --config configs/train.yaml \
    --trainer.max_epochs 200 \
    --data.train_batch_size 64 \
    --model.optimizer.init_args.lr 0.0001
```

### Train without Config File

```bash
getiaction fit \
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

### Pattern 1: class_path (Dynamic)

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
getiaction fit \
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
getiaction fit --config CONFIG_PATH
getiaction validate --config CONFIG_PATH --ckpt_path CHECKPOINT
getiaction test --config CONFIG_PATH --ckpt_path CHECKPOINT
getiaction predict --config CONFIG_PATH --ckpt_path CHECKPOINT
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
getiaction fit --config configs/getiaction/act.yaml --trainer.strategy=ddp --trainer.devices=4
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
getiaction fit --config configs/getiaction/act.yaml --trainer.fast_dev_run=true
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
python -c "from getiaction.policies.pi0.policy import Pi0"
```

**Type errors**: Check config matches class signature
