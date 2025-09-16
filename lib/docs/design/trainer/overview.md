# Trainer

This is the implemenation of a `Lightning` trainer,
with a callback for policy / datamodule interaction:

```mermaid
classDiagram
    class Callback
    class L.Trainer
    class L.LightningModule
    class ActionTrainerModule
    class ActionDataModule

    class PolicyDatasetInteractionCallback {
        +__init__(hook_fn: Callable[[L.Trainer, L.LightningModule], None])
        +on_fit_start(trainer: L.Trainer, pl_module: L.LightningModule)
        -hook_fn: Callable
    }

    class LightningActionTrainer {
        +__init__(num_sanity_val_steps: int = 0, callbacks: list | None = None, **trainer_kwargs)
        +fit(model: ActionTrainerModule, datamodule: ActionDataModule, **kwargs)
        +test(*args, **kwargs)
        +predict(*args, **kwargs)
        +validate(*args, **kwargs)
        -trainer: L.Trainer
    }

    PolicyDatasetInteractionCallback --|> Callback : inherits
    LightningActionTrainer --> L.Trainer : uses
    LightningActionTrainer --> ActionTrainerModule : uses in fit()
    LightningActionTrainer --> ActionDataModule : uses in fit()
    PolicyDatasetInteractionCallback ..> L.Trainer : interacts with
    PolicyDatasetInteractionCallback ..> L.LightningModule : interacts with
```
