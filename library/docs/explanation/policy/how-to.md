# Implement a Policy

Build the config, model, processors, and policy first. Add optional features later.

## 1. Config

The config owns ordered features, model settings, and action horizons.

```python
@dataclass(frozen=True, kw_only=True)
class MyModelConfig(Config):
    input_features: list[Feature]
    output_features: list[Feature]
    action_dim: int
    hidden_size: int = 512
    chunk_size: int = 32
    n_action_steps: int = 32
```

Do not include optimizer settings, artifact paths, normalization values, or export
destinations.

## 2. Model

Use a flat constructor. Implement loss and full-chunk prediction.

```python
class MyModel(TemplateModel):
    def __init__(
        self,
        *,
        action_dim: int,
        hidden_size: int = 512,
        chunk_size: int = 32,
    ) -> None:
        super().__init__()
        self.chunk_size = chunk_size
        self.action_dim = action_dim
        self.backbone = build_backbone(hidden_size)
        self.action_head = nn.Linear(hidden_size, chunk_size * action_dim)

    def compute_loss(self, batch):
        prediction = self._predict(batch)
        loss = F.mse_loss(prediction, batch["action"])
        return loss, {"loss": loss}

    @torch.no_grad()
    def predict_action_chunk(self, batch):
        return self._predict(batch)
```

Return `(batch_size, chunk_size, action_dim)`. Processors handle normalization,
padding, and `n_action_steps`.

Override temporal indices only when the model needs temporal context. See
[Required Interfaces](interfaces.md#temporal-indices).

## 3. Policy Constructor

Show all lifecycle state in the constructor.

```python
class MyPolicy(TemplatePolicy):
    def __init__(
        self,
        input_features: list[Feature] | None = None,
        output_features: list[Feature] | None = None,
        pretrained_name_or_path: str | Path | None = None,
        *,
        n_action_steps: int = 32,
        optimizer_lr: float = 1e-4,
    ) -> None:
        super().__init__(n_action_steps=n_action_steps)
        self.model = None
        self._config: MyModelConfig | None = None
        self._preprocessor = None
        self._postprocessor = None

        self._input_features = input_features
        self._output_features = output_features
        self._pretrained_name_or_path = pretrained_name_or_path
        self.optimizer_lr = optimizer_lr

        if input_features is not None and output_features is not None:
            self.configure_model()
```

Keep deprecated arguments together at the end of the signature. Translate them before
config resolution. See [Policy Migration](migration.md#legacy-arguments).

## 4. Config Construction

An explicit config uses the same materialization path.

```python
@classmethod
def from_config(cls, config: MyModelConfig, **options) -> "MyPolicy":
    policy = cls(n_action_steps=config.n_action_steps, **options)
    policy._config = config
    policy.configure_model()
    return policy
```

A pretrained resolver returns data only:

```python
def _resolve_config_from_hf(path: str | Path) -> tuple[MyModelConfig, Path]:
    return read_artifact_config(path), resolve_weights(path)
```

It does not create the model.

## 5. Dataset Setup

`setup()` is the data-policy boundary.

```python
def setup(self, stage: str) -> None:
    if stage != "fit":
        return

    input_features = list(self.trainer.datamodule.train_dataset.observation_features.values())
    output_features = list(self.trainer.datamodule.train_dataset.action_features.values())
    self._validate_or_set_features(input_features, output_features)
```

Read features directly from the dataset. Do not rebuild them from `dataset_stats`.

## 6. Materialization

Keep `configure_model()` linear.

```python
def configure_model(self) -> None:
    if self.model is not None:
        return

    config, weights_path = self._resolve_config_and_weights()
    self._config = config
    self.model = MyModel.from_config(config)
    self._preprocessor, self._postprocessor = make_policy_processors(config)

    if weights_path is not None:
        self.model.load_weights(weights_path)
```

## 7. Policy Flow

Keep preprocessing, model calls, and postprocessing visible.

```python
def forward(self, batch: Observation):
    prepared = self._prepare_batch(batch, require_actions=self.training)
    if self.training:
        return self.model.compute_loss(prepared)
    return self.predict_action_chunk(batch)

def predict_action_chunk(self, batch: Observation) -> Tensor:
    prepared = self._prepare_batch(batch, require_actions=False)
    chunk = self.model.predict_action_chunk(prepared)
    return self._postprocessor(chunk)

def training_step(self, batch: Observation, batch_idx: int) -> Tensor:
    loss, metrics = self.forward(batch)
    self.log_dict(metrics)
    return loss

def configure_optimizers(self):
    return torch.optim.AdamW(self.model.parameters(), lr=self.optimizer_lr)
```

## 8. Validate

Test:

1. explicit config construction;
2. repeated `configure_model()` calls;
3. one loss and one prediction;
4. output shape and feature order;
5. checkpoint restore without pretrained lookup;
6. optional capabilities and export only when supported.

See [Advanced Patterns](advanced.md) for optional behavior.
