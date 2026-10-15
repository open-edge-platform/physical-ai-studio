# CLI Guide

Train policies using the shared `physicalai` command and studio's jsonargparse subcommands.

## Features

- YAML/JSON config files with CLI overrides
- Type-safe configuration (dataclasses, Pydantic)
- Dynamic class loading (`class_path` pattern)
- Full PyTorch Lightning features (callbacks, loggers, distributed training)

The runtime package owns the top-level `physicalai` executable. Studio contributes
the training subcommands through the `physicalai.cli.subcommands` entry-point group.

## Basic Usage

### Train with Config File

```bash
physicalai fit --config configs/physicalai/act/pusht/default.yaml
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

## Native Framework Training

`physicalai train <framework>` delegates directly to an installed native engine. The Studio options must precede the framework name; every argument after the name is passed through unchanged, including native `--help` and native flags:

```bash
physicalai train lerobot --policy.type=act --policy.device=cuda --dataset.repo_id=lerobot/pusht --steps=20000
physicalai train lerobot --policy.type=act --policy.device=xpu --dataset.repo_id=lerobot/pusht
physicalai train --interpreter /opt/lerobot/bin/python lerobot --help
```

`pai train` accepts Studio execution options such as `--interpreter` before the framework name; every argument after the name is passed through unchanged. LeRobot's native `--policy.device` chooses both the training device and the matching managed worker torch extra (`cuda` → `cu128`, `xpu` → `xpu`, `cpu` → `cpu`). When no device is set, the engine auto-detects XPU, then CUDA, then CPU from available hardware, even if the host has a CPU-only torch build. `pai engine prepare --accelerator` below only pre-provisions an environment; it does not override the native training device.

The native command keeps its standard streams and exit status. `--interpreter` skips managed-environment preparation, so the supplied Python must already have the torch build required by the native device. `physicalai train physicalai` aliases `physicalai fit`; first-party Lightning options remain on `fit` (`--trainer.*`, `--model.*`, and `--data.*`), including `--trainer.accelerator`.

Managed worker environments are provisioned by the shared engine manager:

```bash
physicalai engine prepare lerobot --accelerator xpu
physicalai engine list
physicalai engine prune
```

Each framework plugin supplies its own locked worker project. `--interpreter` skips managed-environment provisioning and uses a pre-installed framework environment.

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

| Command    | Description                                    |
| ---------- | ---------------------------------------------- |
| `fit`      | Train a model                                  |
| `validate` | Run validation                                 |
| `test`     | Run test evaluation                            |
| `predict`  | Run inference                                  |
| `train`    | Run a native framework trainer                 |
| `engine`   | Prepare, inspect, or prune worker environments |

```bash
physicalai fit --config CONFIG_PATH
physicalai fit --config CONFIG_PATH --fit.ckpt_path CHECKPOINT       # resume training
physicalai validate --config CONFIG_PATH --validate.ckpt_path CHECKPOINT
physicalai test --config CONFIG_PATH --test.ckpt_path CHECKPOINT
physicalai predict --config CONFIG_PATH --predict.ckpt_path CHECKPOINT
```

Arguments belonging to the `Trainer` method itself (`ckpt_path`, `weights_only`,
`verbose`) are namespaced under the subcommand name, so they do not collide with
`--trainer.*` constructor arguments. `physicalai benchmark` and
`physicalai export` use a bare `--ckpt_path` because they are not
`Trainer`-backed.

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
physicalai fit --config configs/physicalai/act/pusht/default.yaml --trainer.strategy=ddp --trainer.devices=4
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
physicalai fit --config configs/physicalai/act/pusht/default.yaml --trainer.fast_dev_run=true
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
python -c "from physicalai.policies.pi05.policy import Pi05"
```

**Type errors**: Check config matches class signature
