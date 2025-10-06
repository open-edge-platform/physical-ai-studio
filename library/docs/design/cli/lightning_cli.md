# LightningCLI Integration

The GetiAction CLI is built on top of PyTorch Lightning's `LightningCLI`,
which provides robust argument parsing and configuration management through
`jsonargparse`.

## Class Structure

```mermaid
classDiagram
    class LightningCLI {
        +model_class: Type[LightningModule]
        +datamodule_class: Type[LightningDataModule]
        +subclass_mode_model: bool
        +subclass_mode_data: bool
        +save_config_callback: SaveConfigCallback|None
    }

    class GetiActionCLI {
        +cli()
    }

    LightningCLI <|-- GetiActionCLI : uses
```

## Implementation

The CLI implementation in `cli.py` is intentionally minimal:

```python
from lightning.pytorch.cli import LightningCLI
from getiaction.data import DataModule
from getiaction.policies.base import Policy

def cli() -> None:
    """Main CLI entry point."""
    LightningCLI(
        model_class=Policy,
        datamodule_class=DataModule,
        save_config_callback=None,
        subclass_mode_model=True,  # Allow any Policy subclass
        subclass_mode_data=True,   # Allow any DataModule subclass
    )
```

## Key Features

### 1. Subclass Mode

Enabling `subclass_mode_model=True` and `subclass_mode_data=True` allows
dynamic instantiation of any Policy or DataModule subclass via
configuration:

```yaml
model:
  class_path: getiaction.policies.dummy.policy.Dummy
  init_args:
    model:
      class_path: getiaction.policies.dummy.model.Dummy
```

### 2. Configuration Parsing

`jsonargparse` automatically:

- Parses YAML/JSON configuration files
- Validates types against class signatures
- Merges CLI arguments with config files
- Supports nested configuration structures

### 3. Command Support

The CLI automatically provides standard Lightning commands:

```bash
getiaction fit          # Train model
getiaction validate     # Run validation
getiaction test         # Run testing
getiaction predict      # Run predictions
```

## Configuration Flow

```mermaid
sequenceDiagram
    participant User
    participant CLI
    participant jsonargparse
    participant LightningCLI
    participant Trainer

    User->>CLI: getiaction fit --config train.yaml
    CLI->>jsonargparse: Parse config + args
    jsonargparse->>jsonargparse: Validate types
    jsonargparse->>LightningCLI: Instantiate components
    LightningCLI->>Trainer: trainer.fit(model, datamodule)
    Trainer-->>User: Training results
```

## Benefits

1. **No Custom Parsing**: Lightning handles all argument parsing
2. **Type Safety**: Automatic validation from type hints
3. **Ecosystem Integration**: Works with all Lightning features (callbacks,
   loggers, plugins)
4. **Maintainability**: Less code to maintain, battle-tested
   implementation
5. **Extensibility**: Easy to add new commands and options

## Example Usage

### Basic Training

```bash
getiaction fit \
    --model.class_path getiaction.policies.dummy.policy.Dummy \
    --data.class_path getiaction.data.lerobot.LeRobotDataModule \
    --trainer.max_epochs 100
```

### With Config File

```bash
getiaction fit --config configs/train.yaml
```

### Override Config Values

```bash
getiaction fit \
    --config configs/train.yaml \
    --trainer.max_epochs 200 \
    --data.train_batch_size 64
```

### Print Full Configuration

```bash
getiaction fit --print_config
```

## Integration Points

The CLI integrates with:

- **Policies** via `Policy` base class
- **DataModules** via `DataModule` base class
- **Trainers** via Lightning `Trainer` class
- **Callbacks** via Lightning callback system
- **Loggers** via Lightning logger system

This design ensures the CLI remains lightweight while providing full
access to the Lightning ecosystem.
