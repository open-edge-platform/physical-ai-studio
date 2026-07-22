# Configuration System Design

## Goal

PhysicalAI should support the same workflows from Python, YAML, CLI, and Studio
APIs.

The configuration system makes these paths equivalent:

```python
trainer = Trainer.from_config("training.yaml")
trainer.fit()
```

```bash
physicalai fit --config training.yaml
```

```python
Trainer.from_config(studio_payload_dict).fit()
```

```python
config = trainer.to_config()
config.to_yaml("training.reproduced.yaml")
```

The same model should also work for inference and robot runtime:

```python
model = InferenceModel.from_config("inference.yaml")
runtime = PolicyRuntime.from_config("runtime.yaml")
```

```bash
physicalai infer --config inference.yaml
physicalai run --config runtime.yaml
```

## Direction

The configuration backbone is **jsonargparse**. Constructor signatures are the
schema. Typed dataclasses are optional ergonomic wrappers, not a requirement.

- Each class is its own schema. jsonargparse introspects constructors directly;
  no inheritance, decorators, or special types are required for a class to be
  configurable.
- jsonargparse handles YAML loading, CLI parsing, recursive instantiation, and
  subclass resolution (including short-name lookup against the typed base
  class).
- Typed config dataclasses (e.g. `Pi05Config`, `TrainingConfig`) are added per
  class **only** when the public Python SDK or Studio benefits from typed
  construction. They do not replace constructor introspection.
- Workflow orchestrators (`Trainer`, `PolicyRuntime`, `InferenceModel`) own
  `from_config(...)` and `to_config()` entry points. Internally they wrap a
  jsonargparse parser.
- `physicalai.config.FromConfig` extends `jsonargparse.FromConfigMixin` with
  dataclass and Pydantic adapters. It provides the `cls.from_config(...)`
  classmethod surface for known target classes.
- The custom `instantiate_obj` / `instantiate_obj_from_dict` helpers are
  removed. All programmatic instantiation goes through jsonargparse (directly,
  or via `cls.from_config(...)`).
- `library/src/physicalai/config/base.py::Config` is removed. Every method it
  provided (`save`, `load`, `to_dict`, `from_dict`, `to_jsonargparse`) has a
  one-line jsonargparse or stdlib replacement, and the unified base no longer
  earns its keep.
- `ComponentSpec` stays in `physicalai.inference.manifest` for now. The
  long-term goal (Phase 8) is to delete it: the manifest becomes a typed
  dataclass whose fields use jsonargparse subclass typing, and short-name
  resolution replaces the bespoke `type` registry.

## Big Picture

The system has three data layers and one execution layer.

```text
Class schema (constructor signature, optionally with a typed dataclass)
  typed constructor arguments for one class

Workflow config (YAML / dict / dataclass / jsonargparse Namespace)
  user-authored workflow graph before execution

Manifest (exported package metadata)
  concrete artifacts + runtime descriptors after build/export

Orchestrator
  live object that consumes configs/manifests and runs the workflow
```

Examples:

```text
Pi05 / Pi05Config        -> constructor args for Pi05 (Pi05Config optional)
TrainingConfig           -> model + data + trainer (YAML, dict, or dataclass)
Manifest                 -> exported artifacts + runtime metadata
Trainer                  -> orchestrates training
InferenceModel           -> loads model artifacts and computes actions
PolicyRuntime            -> runs robot control loop
```

Passive data objects do not execute workflows. Orchestrators do.

```python
trainer = Trainer.from_config("training.yaml")
trainer.fit()
```

## Authoring Tiers

A single jsonargparse parser supports three authoring tiers. All three produce
the same instantiated objects.

### Tier 0 — Bare class, no config wrapper

The default for slots whose concrete class is fixed by the parser. jsonargparse
introspects the constructor; no dataclass needed.

```python
class Trainer:
    def __init__(self, max_epochs: int = 10, accelerator: str = "auto") -> None:
        ...

parser.add_class_arguments(Trainer, "trainer")
```

YAML (flat, no `class_path` / `init_args`):

```yaml
trainer:
  max_epochs: 10
  accelerator: gpu
```

CLI:

```bash
--trainer.max_epochs 10 --trainer.accelerator gpu
```

Dict:

```python
parser.parse_object({"trainer": {"max_epochs": 10}})
```

Use Tier 0 for slots that always resolve to one concrete class — in physicalai
that is essentially just `trainer`. Slots that select between alternative
implementations (e.g. `model`, `data`, `callbacks`, `runner`, `preprocessors`)
are Tier 2, not Tier 0.

### Tier 1 — Typed config dataclass (icing)

For classes that are part of the public Python API or that Studio constructs
programmatically, add a typed config dataclass.

```python
@dataclass
class Pi05Config:
    chunk_size: int = 50
    n_action_steps: int = 50

class Pi05(FromConfig):
    def __init__(self, chunk_size: int = 50, n_action_steps: int = 50) -> None:
        ...
```

All three forms now work:

```python
Pi05.from_config(Pi05Config(chunk_size=50))   # typed dataclass
Pi05.from_config({"chunk_size": 50})           # dict
Pi05.from_config("pi05.yaml")                  # YAML
```

Tier 1 adds:

- IDE autocomplete and static type checking at call sites.
- A canonical object Studio backends can construct without dict gymnastics.
- A shared schema for SDK examples and Studio payload mapping.

Tier 1 is **independent of Tier 0 vs Tier 2 slot selection**. A `Pi05Config`
dataclass is useful for typed Python construction (`Pi05.from_config(cfg)`),
but the orchestrator's `model` slot is still Tier 2 because the parser must be
able to choose between Pi05, ACT, SmolVLA, etc. The dataclass is a convenience
for direct class consumption, not a way to make polymorphic slots flat.

It does not change the YAML / CLI / dict paths, which still work via Tier 0
constructor introspection.

### Tier 2 — Polymorphic selection

When the choice of class is part of the configuration, type the slot with the
**base class** and let jsonargparse resolve the concrete class by short name.

```python
parser.add_subclass_arguments(LightningModule, "model")
```

When jsonargparse knows the expected base type, it walks importable subclasses
and resolves a bare class name automatically. Two forms are supported.

**Tier 2a — bare short name (when no init args are needed).**

```yaml
model: Pi05
```

```bash
--model Pi05
```

The most compact form. Works whenever no init arguments need to be set.

**Tier 2b — nested form (when init args are needed).**

```yaml
model:
  class_path: Pi05            # short name, not dotted path
  init_args:
    chunk_size: 50
```

```bash
--model Pi05 --model.chunk_size 50
```

`class_path` accepts a bare short name (`Pi05`) as long as exactly one
importable subclass of `LightningModule` matches. The dotted path
(`physicalai.policies.pi05.Pi05`) is required only to disambiguate when two
subclasses share the same short name (e.g. a vendor fork).

YAML does not allow a scalar value to also carry child keys, so a flat shape
like `model: Pi05` with sibling `chunk_size: 50` indented underneath is not
expressible without a custom YAML tag. The Tier 2b nested form is the canonical
way to combine class selection with init args.

For short-name resolution to work, the subclass must be importable at the time
the parser runs. Built-in classes are imported eagerly from package
`__init__.py`; third-party classes register through Python entry points or are
imported explicitly by the caller.

### Tier choice per slot

| Slot kind                                                              | Tier   |
| ---------------------------------------------------------------------- | ------ |
| Concrete class fixed by the parser (e.g. `trainer`)                    | Tier 0 |
| Same as Tier 0 but exposed for typed Python construction               | Tier 0 + Tier 1 dataclass |
| Slot selecting between alternative subclasses (e.g. `model`, `data`, `callbacks`, `runner`, `preprocessors`) | Tier 2 |

In physicalai, the only training-side Tier 0 slot is `trainer`. Every policy
slot (`model`), datamodule slot (`data`), callback list, runner, preprocessor,
and postprocessor is Tier 2 because the codebase ships multiple
implementations.

Tier 1 is orthogonal: a Tier 2 slot can still have per-class typed dataclasses
(`Pi05Config`, `ACTConfig`, ...) that Studio and SDK users construct directly.

## Core Concepts

### Class Schemas

A class is its own schema. There is no requirement to wrap every class in a
`Config` object.

```python
class Pi05:
    def __init__(self, chunk_size: int = 50, n_action_steps: int = 50) -> None:
        if n_action_steps > chunk_size:
            raise ValueError("n_action_steps must be <= chunk_size")
        self.chunk_size = chunk_size
        self.n_action_steps = n_action_steps
```

Validation lives in `__init__` / `__post_init__`. There is no project-wide
`Config` base class for typed constructor schemas.

### FromConfig

`FromConfig` is a thin extension of `jsonargparse.FromConfigMixin` that adds
dataclass and Pydantic adapters on top of jsonargparse's built-in
`from_config(path | dict)` support.

```python
# library/src/physicalai/config/mixin.py
from jsonargparse import FromConfigMixin

class FromConfig(FromConfigMixin):
    """jsonargparse FromConfigMixin + dataclass/Pydantic adapters."""

    @classmethod
    def from_config(cls, config):
        if dataclasses.is_dataclass(config) and not isinstance(config, type):
            config = dataclasses.asdict(config)
        elif isinstance(config, BaseModel):
            config = config.model_dump()
        return super().from_config(config)   # parent handles str | PathLike | dict
```

What the parent (`jsonargparse.FromConfigMixin`) provides:

- `cls.from_config(path | dict)` classmethod.
- Subclass dispatch via `{class_path, init_args}` (including bare short names).
- Recursive nested instantiation.
- `__from_config_init_defaults__` for overriding constructor defaults from a
  config file at subclass creation time.

What this extension adds:

- Dataclass argument support (Tier 1 typed configs).
- Pydantic model argument support.

Usage:

```python
Pi05.from_config({"chunk_size": 50})
Pi05.from_config("pi05.yaml")
Pi05.from_config(Pi05Config(chunk_size=50))
Pi05.from_config({"class_path": "Pi05Plus", "init_args": {...}})
```

Used by LeRobot wrappers, `Pi05`, `SmolVLA`, and similar policy classes, plus
any orchestrator that wants a single ergonomic Python entry point.

### Checkpoint Serialization

There is no shared `Config` base class. Classes that need checkpoint
round-tripping (e.g. for torch `load(..., weights_only=True)`) implement their
own two-line `to_dict` / `from_dict`, or use `dataclasses.asdict` and
`cls(**data)` directly.

For YAML / dict / dataclass I/O, jsonargparse owns serialization:

```python
cfg = parser.parse_path("training.yaml")
parser.dump(cfg, "training.reproduced.yaml")
```

`parser.dump` correctly handles nested `class_path` / `init_args` shapes that
`dataclasses.asdict` cannot.

### Workflow Configs

A workflow config is a user-authored configuration before execution. It is
expressed in YAML / dict / dataclass and parsed by jsonargparse.

Training config:

```yaml
# training.yaml
trainer:                              # Tier 0 — concrete Trainer class
  max_epochs: 10
  accelerator: gpu
  devices: 1
  callbacks:                          # Tier 2 — Callback subclasses
    - ProgressCallback                # 2a: short name, no init args
    - class_path: ModelCheckpoint     # 2b: short name + init args
      init_args:
        save_top_k: 3

model:                                # Tier 2 — LightningModule subclass
  class_path: Pi05                    # one of: Pi05, ACT, SmolVLA, Pi0, GR00T, ...
  init_args:
    chunk_size: 50
    n_action_steps: 50

data:                                 # Tier 2 — LightningDataModule subclass
  class_path: LeRobotDataModule       # one of: LeRobotDataModule, custom modules
  init_args:
    repo_id: physical-ai/example
```

Only `trainer` is flat (Tier 0). `model`, `data`, and `callbacks` are
polymorphic slots and carry `class_path:`. Class names are short — jsonargparse
resolves them against the typed base class (`LightningModule`,
`LightningDataModule`, `Callback`) by walking importable subclasses.

Python equivalent (preferred entry point):

```python
Trainer.from_config("training.yaml").fit()
```

Typed dataclass equivalent (Tier 1):

```python
config = TrainingConfig(
    trainer=TrainerConfig(max_epochs=10),
    model=Pi05Config(chunk_size=50),
    data=LeRobotDataConfig(repo_id="physical-ai/example"),
)
Trainer.from_config(config).fit()
```

The typed form carries class identity in the Python type system itself
(`Pi05Config` ↔ `Pi05`); the YAML form carries it in `class_path`. Both reach
the same parser.

Dict equivalent (e.g. from Studio):

```python
Trainer.from_config(
    {
        "trainer": {"max_epochs": 10},
        "model": {
            "class_path": "Pi05",
            "init_args": {"chunk_size": 50},
        },
        "data": {
            "class_path": "LeRobotDataModule",
            "init_args": {"repo_id": "physical-ai/example"},
        },
    }
).fit()
```

All four forms hit the same parser and produce the same instantiated objects.

### Polymorphic Children in Tier 1 (Callbacks, Loggers)

Tier 1 typed configs hold polymorphic children (`callbacks: list[Callback]`,
`logger: Logger | list[Logger]`) as **live instances**, not as `{class_path,
init_args}` wrappers.

```python
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger
from physicalai.train.callbacks import IterationTimer

config = TrainingConfig(
    trainer=TrainerConfig(
        max_epochs=10,
        callbacks=[
            ModelCheckpoint(save_top_k=3, monitor="val/loss"),
            EarlyStopping(monitor="val/loss", patience=5),
            IterationTimer(),
        ],
        logger=TensorBoardLogger(save_dir="logs", name="exp1"),
    ),
    model=Pi05Config(chunk_size=50),
    data=LeRobotDataConfig(repo_id="physical-ai/example"),
)
Trainer.from_config(config).fit()
```

jsonargparse recognizes already-instantiated objects at class-typed leaves and
passes them through `parser.instantiate()` unchanged. No wrapper type, no
`class_path` strings in Python. Python users write Python; YAML users write
`class_path` / `init_args`. Each path uses its native idiom.

This matches Lightning's `Trainer(callbacks=..., logger=...)` ergonomics and
the imperative assembly the Studio backend already does in
`application/backend/src/workers/training_worker.py`.

**Serialization rule.** `to_config()` MUST use `parser.dump(...)` to round-trip
typed configs. Do NOT use `dataclasses.asdict(config)`:

- `asdict` recursively flattens dataclass fields and strips class identity from
  non-dataclass children. A `ModelCheckpoint` instance becomes an opaque value
  with no `class_path`, breaking subclass dispatch on reload.
- `parser.dump` walks the parser schema and emits `class_path` / `init_args`
  for every polymorphic leaf, preserving round-trip identity.

```python
trainer = Trainer.from_config(config)
trainer.to_config("training.reproduced.yaml")   # parser.dump under the hood
```

Runtime config:

```yaml
# runtime.yaml
runtime:
  fps: 30
  robot:
    port: /dev/ttyACM0
  model:
    path: ./exports/act_policy
  execution:
    mode: chunk
```

```python
PolicyRuntime.from_config("runtime.yaml").run(duration_s=60)
```

Each orchestrator owns its workflow config shape. There is no shared
`WorkflowConfig` base.

### Manifest

A manifest describes an exported package and lives in
`physicalai.inference.manifest`.

Today, the manifest uses a custom `ComponentSpec` schema with `type` /
`class_path` / `init_args` / `artifact` fields. The long-term target (Phase 8)
is a typed dataclass whose fields use jsonargparse subclass typing, with bare
short names in YAML:

```yaml
# manifest.yaml — Phase 8 target shape
format: policy_package
version: "1.0"

policy:
  name: pi05
  source:
    class_path: Pi05

model:
  artifacts:
    openvino: pi05.xml
  runner:
    class_path: ActionChunkingRunner
    init_args:
      chunk_size: 50
  preprocessors:
    - class_path: Normalize
      init_args:
        stats_path: stats.safetensors

hardware:
  robots:
    - class_path: SO101
      init_args:
        port: /dev/ttyACM0
```

No `type:` field. No dotted paths (unless disambiguation is needed). The same
parser code instantiates the manifest as instantiates training / runtime
configs.

Workflow config and manifest remain conceptually distinct:

```text
Workflow config
  desired workflow before running/building/exporting

Manifest
  concrete exported package after build/export
```

The manifest can also act as an inference-time config because it already
contains the necessary class references:

```python
model = InferenceModel.from_config("export/manifest.json")
```

## Parser Construction

A single generic builder produces parsers for all subcommands, mirroring the
LightningCLI pattern: shared base + per-method customization.

```python
def _build_parser_base() -> ArgumentParser:
    """Shared schema: trainer + model + data + callbacks."""
    parser = ArgumentParser()
    parser.add_class_arguments(Trainer, "trainer")
    parser.add_subclass_arguments(LightningModule, "model")
    parser.add_subclass_arguments(LightningDataModule, "data")
    return parser

def _build_subcommand_parser(method: str) -> ArgumentParser:
    """Shared base + method-specific arguments via add_method_arguments."""
    parser = _build_parser_base()
    parser.add_method_arguments(Trainer, method, skip=_SKIP_BY_METHOD[method])
    return parser

def _build_main_parser() -> ArgumentParser:
    main = ArgumentParser()
    subs = main.add_subcommands()
    for method in ("fit", "validate", "test", "predict"):
        subs.add_subcommand(method, _build_subcommand_parser(method))
    return main
```

`Trainer.from_config(source, method="fit")` reuses the same builder:

```python
@classmethod
def from_config(cls, source, method: str = "fit") -> "Trainer":
    parser = _build_subcommand_parser(method)
    cfg = _parse(parser, source)              # path | dict | dataclass | Namespace
    instantiated = parser.instantiate(cfg)
    trainer = instantiated.trainer
    trainer._configured_model = instantiated.model
    trainer._configured_datamodule = instantiated.data
    return trainer
```

`_parse` dispatches by source type:

```python
def _parse(parser, source):
    if isinstance(source, (str, Path)):
        return parser.parse_path(str(source))
    if dataclasses.is_dataclass(source) and not isinstance(source, type):
        return parser.parse_object(dataclasses.asdict(source))
    if isinstance(source, BaseModel):
        return parser.parse_object(source.model_dump())
    if isinstance(source, dict):
        return parser.parse_object(source)
    return source                             # already a Namespace
```

### Per-Distribution Parsers

`physicalai` and `physicalai` ship as two distributions with two main
parsers. Each follows the same shared-base + per-subcommand pattern.

| Distribution                  | Subcommands                                    | Builder                          |
| ----------------------------- | ---------------------------------------------- | -------------------------------- |
| `library` (torch + Lightning) | `fit`, `validate`, `test`, `predict`, `export` | `_build_training_subparser(...)` |
| `physicalai` (torch-free)     | `run`, `serve`, `infer`, `inspect-manifest`    | `_build_runtime_subparser(...)`  |

Each has its own `_build_parser_base()` because the shared schema differs
(training: trainer + model + data; runtime: robot + model + execution).

## End-to-End Pipelines

### Training From YAML

```python
trainer = Trainer.from_config("training.yaml")
trainer.fit()
```

Internally:

```python
parser = _build_subcommand_parser("fit")
cfg = parser.parse_path("training.yaml")
instantiated = parser.instantiate(cfg)
trainer = instantiated.trainer
trainer._configured_model = instantiated.model
trainer._configured_datamodule = instantiated.data
return trainer
```

`fit()` defaults to the bound `model` / `datamodule` when not given explicit
arguments.

### Training From CLI

```bash
physicalai fit --config training.yaml
physicalai fit \
  --trainer.max_epochs 10 \
  --model Pi05 \
  --model.chunk_size 50 \
  --data LeRobotDataModule \
  --data.repo_id physical-ai/example
```

The CLI uses the same parser as `Trainer.from_config`. There is one definition
of the training schema.

### Training From Studio

The application layer converts the API payload to the form
`Trainer.from_config` accepts (dict, or a typed `TrainingConfig` if available).

```python
def handle_train_job(payload: TrainJobPayload) -> None:
    config = studio_payload_to_training_config(payload)
    Trainer.from_config(config).fit()
```

The worker should not manually assemble model, data module, callbacks, and
trainer if the library can own that orchestration.

```text
Studio payload
    -> application mapper (dict or TrainingConfig)
    -> Trainer.from_config(...)
    -> Trainer.fit
```

### Training Round Trip

For ordinary components, `to_config()` describes that component. For
orchestrators, `to_config()` returns the highest-level reproducible workflow
config available.

```python
trainer = Trainer.from_config("training.yaml")
trainer.fit()

config = trainer.to_config()
config.to_yaml("training.reproduced.yaml")
```

Expected behavior:

```python
Trainer(max_epochs=10).to_config()
# returns trainer-only config (model/data omitted)

Trainer.from_config("training.yaml").to_config()
# returns full training config (trainer + model + data)
```

The serialized output uses the same shape as the input — flat for Tier 0 / 1
classes, `class_path` / `init_args` for Tier 2 selections.

### Inference From Manifest

```python
model = InferenceModel.from_config("export/manifest.json")
action = model.select_action(observation)
```

Before Phase 8 (current state):

```python
manifest = Manifest.load("export/manifest.json")
runner = instantiate_component(manifest.model.runner)
pre = [instantiate_component(s) for s in manifest.model.preprocessors]
post = [instantiate_component(s) for s in manifest.model.postprocessors]

model = InferenceModel(
    artifacts=manifest.model.artifacts,
    runner=runner,
    preprocessors=pre,
    postprocessors=post,
)
```

After Phase 8 (target state):

```python
parser = _build_inference_parser()
cfg = parser.parse_path("export/manifest.json")
instantiated = parser.instantiate(cfg)
model = InferenceModel(**instantiated.model.as_dict())
```

`ComponentSpec`, `instantiate_component`, and `ComponentRegistry` are deleted.
Manifest YAML uses jsonargparse-native subclass typing with bare short names.

### Robot Runtime From YAML

```python
PolicyRuntime.from_config("runtime.yaml").run(duration_s=60)
```

```bash
physicalai run --config runtime.yaml --duration-s 60
```

## API Surface

### FromConfig

`FromConfig` is class-level construction sugar (extends
`jsonargparse.FromConfigMixin`).

```python
class FromConfig(jsonargparse.FromConfigMixin):
    @classmethod
    def from_config(cls, config) -> Self: ...
```

Accepted `config` types:

- `str | PathLike` — YAML / JSON config file path.
- `dict` — already-parsed config dict.
- `dataclass instance` — Tier 1 typed config.
- `pydantic.BaseModel` — Tier 1 typed config.

Used by:

- LeRobot wrappers (noisy kwargs + YAML files).
- `Pi05`, `SmolVLA`, and similar policy classes.
- Any orchestrator that wants a single ergonomic Python entry point.

Orchestrators override `from_config` to bind subcomponents through the shared
parser (see [Parser Construction](#parser-construction)).

`fit()` remains the public execution API:

```python
trainer.fit(model=model, datamodule=data)
Trainer.from_config("training.yaml").fit()
```

### ToConfig

`to_config()` is opt-in serialization.

```python
class Trainer:
    def to_config(self) -> TrainingConfig | dict[str, Any]: ...
```

Objects decide what state is reproducible constructor config and what state is
runtime-only. Outbound serialization is explicit; inbound instantiation is
generic.

For policies, export goes through the existing `policy.export(...)` path and
produces a manifest, not a workflow config.

```python
policy.export("export/", backend="openvino")
```

### Instantiation

There is no custom `instantiate_obj` / `instantiate_obj_from_dict` helper.
Programmatic instantiation goes through one of two paths:

1. **Class-targeted** — `cls.from_config(config)` for code that knows the
   target class.
2. **Parser-driven** — `parser.parse_object(data); parser.instantiate(cfg)` for
   code that builds a multi-component workflow.

Both paths use jsonargparse internally. `physicalai.config` re-exports:

```python
from physicalai.config import FromConfig
```

## CLI Parity

The CLI uses the same schema as the Python API.

```bash
physicalai run --config runtime.yaml
```

```python
PolicyRuntime.from_config("runtime.yaml").run()
```

These construct the same runtime.

`physicalai` is the torch-free runtime package. Its CLI depends on
`jsonargparse` but must not depend on torch or Lightning.

```text
physicalai
  infer
  run
  serve
  inspect-manifest

physicalai (library distribution)
  fit
  validate
  test
  predict
  export
```

Training commands plug into the runtime CLI via entry points, but importing
`physicalai` alone does not import training dependencies.

```toml
[project.entry-points."physicalai.cli.subcommands"]
run = "physicalai.cli.run:register"
serve = "physicalai.cli.serve:register"

[project.entry-points."physicalai.cli.subcommands"]
fit = "physicalai.train.cli:register_fit"
```

## Package Boundaries

```text
physicalai/                                    (torch-free)
  config/
    FromConfig                  (extends jsonargparse.FromConfigMixin)
  inference/
    manifest.py                 (ComponentSpec, Manifest — until Phase 8)
    component_factory.py        (instantiate_component — until Phase 8)
    InferenceModel
  runtime/
    PolicyRuntime
  cli/                          (jsonargparse-based)

library/                                       (torch + Lightning)
  config/
    mixin.py                    (FromConfig re-export)
  train/
    Trainer
    parser.py                   (_build_parser_base, _build_subcommand_parser)
    cli.py                      (jsonargparse fit/validate/test/predict)
  export/
    manifest builders
```

Note: `library/src/physicalai/config/base.py` is **removed** in Phase 1. There
is no shared `Config` base class on the library side.

## Design Rules

1. Constructor signatures are the schema. Typed config dataclasses are optional
   ergonomic wrappers, not a requirement.
2. jsonargparse is the parsing + instantiation backbone for YAML, CLI, dict,
   and dataclass inputs.
3. A single generic parser builder (`_build_parser_base` +
   `_build_subcommand_parser(method)`) feeds every subcommand. No per-method
   bespoke parsers.
4. `FromConfig` extends `jsonargparse.FromConfigMixin`. No parallel
   instantiation backend.
5. Orchestrators (`Trainer`, `PolicyRuntime`, `InferenceModel`) own
   `from_config(...)` and `to_config()`.
6. Polymorphic slots (`model`, `data`, `callbacks`, `runner`, `preprocessors`,
   `postprocessors`, `robots`, `cameras`) carry `class_path:` (Tier 2). Flat
   YAML applies only to slots bound to a single concrete class (`trainer`).
   Short names are used in `class_path`; dotted paths are reserved for
   disambiguation.
7. No custom `instantiate_obj` / `instantiate_obj_from_dict`. Programmatic
   instantiation goes through `cls.from_config(...)` or a jsonargparse parser.
8. There is no shared `Config` base class. Per-class checkpoint serialization is
   handled with two-line `to_dict` / `from_dict` or `dataclasses.asdict` +
   `cls(**data)`.
9. Keep torch and Lightning out of the `physicalai` runtime package.
10. Inbound config is generic; outbound config is explicit.
11. Application payload translation lives in the application layer; reusable
    workflow logic lives in the library / runtime layer.
12. Built-in subclasses of base-typed slots (models, runners, preprocessors,
    callbacks, robots) are imported eagerly from their package `__init__.py`
    so jsonargparse short-name resolution works without user setup.

## Migration Plan

### Phase 1: Adopt `jsonargparse.FromConfigMixin`; Delete `Config` Base

- Reimplement `library/src/physicalai/config/mixin.py::FromConfig` as a thin
  extension of `jsonargparse.FromConfigMixin` that adds dataclass / Pydantic
  adapters.
- Delete `library/src/physicalai/config/base.py` (`Config` class).
- For any class that still needs checkpoint round-tripping, inline a two-line
  `to_dict` / `from_dict` on that class, or use `dataclasses.asdict` +
  `cls(**data)`.
- Remove `Config` from `physicalai.config.__init__` re-exports.
- Delete `library/tests/unit/config/test_config.py` (covers the removed base).

### Phase 2: Delete Custom Instantiation Helpers

- Delete `library/src/physicalai/config/instantiate.py`
  (`instantiate_obj`, `instantiate_obj_from_dict`).
- Migrate any remaining callers to `cls.from_config(...)` or a direct
  jsonargparse parser call.
- Remove these symbols from `physicalai.config.__init__`.

### Phase 3: Generic Parser Builder + `Trainer.from_config` / `to_config`

- Implement `_build_parser_base()` and `_build_subcommand_parser(method)` in
  `library/src/physicalai/train/parser.py` using jsonargparse.
- Implement `Trainer.from_config(source, method="fit")` accepting path / dict /
  dataclass / Pydantic / Namespace.
- Implement `Trainer.to_config()` returning the highest-level reproducible
  config: full workflow when `from_config` constructed it, trainer-only
  otherwise.
- `fit()` defaults to bound `model` / `datamodule` when none is passed.

### Phase 4: YAML Migration For Existing Configs

- Convert `library/configs/*.yaml` (training, benchmark) to the canonical
  shape:
  - Flat (Tier 0) for `trainer`.
  - Tier 2 with short-name `class_path:` for `model`, `data`, `callbacks`,
    and any other polymorphic slot.
  - Drop dotted paths; rely on short-name resolution against the typed base
    class.

### Phase 5: Studio Integration

- Decide between dict mapper (`studio_payload_to_training_config`) and typed
  mapper (Tier 1 `TrainingConfig`). Implement chosen path in the application
  layer.
- Refactor `application/backend/src/workers/training_worker.py` to call
  `Trainer.from_config(config).fit()`.

### Phase 6: CLI

- Add jsonargparse-based CLI entry points in `physicalai/`
  (`run`, `serve`, `infer`, `inspect-manifest`) using
  `_build_runtime_subparser(method)`.
- Add `physicalai` CLI in `library/`
  (`fit`, `validate`, `test`, `predict`, `export`) sharing the parsers used by
  `Trainer.from_config`.

### Phase 7: Runtime / Inference `from_config`

- Implement `PolicyRuntime.from_config(...)` and
  `InferenceModel.from_config(...)` using the same jsonargparse pattern.
- Confirm `Manifest` can be loaded directly by `InferenceModel.from_config`
  (still going through `ComponentSpec` at this stage).

### Phase 8: Delete `ComponentSpec`; Unify Manifest Schema

Deferred. Requires a versioned manifest schema bump (manifests are an exported
artifact format).

- Replace `Manifest` / `ComponentSpec` with typed dataclasses whose
  runner / preprocessor / postprocessor / robot fields are typed by their base
  class (e.g. `runner: ActionRunner`, `preprocessors: list[Preprocessor]`).
- Manifest YAML uses jsonargparse-native subclass syntax with bare short names
  (`class_path: ActionChunkingRunner`, no dotted path, no `type:` field).
- Delete `ComponentSpec`, `instantiate_component`, `ComponentRegistry`.
- `InferenceModel.from_config` loads manifests via `parser.instantiate`.
- Ensure built-in runners / preprocessors / postprocessors / robots are
  imported eagerly from their package `__init__.py` so short-name resolution
  succeeds.
- Document the manifest schema version bump and provide a migration path for
  existing exported packages.

### Phase 9 (optional): Typed Workflow Configs (Tier 1)

- Add per-class config dataclasses (`Pi05Config`, `ACTConfig`,
  `LeRobotDataConfig`, `TrainerConfig`) and aggregate `TrainingConfig` /
  `RuntimeConfig` / `InferenceConfig` where the public SDK or Studio benefits
  from typed Python construction.
- These wrap the same jsonargparse parser; YAML / dict / dataclass forms all
  remain valid.

## Open Questions

- Should `Trainer.to_config()` return a typed dataclass or the raw
  jsonargparse-compatible dict?
- Where does `studio_payload_to_training_config` live: application layer only,
  or as a `Trainer.from_studio_payload(...)` classmethod in the library?
- Which classes warrant Tier 1 typed configs from day one
  (recommendation: every public policy + `Trainer` + datamodules used by
  Studio)?
- For Phase 8: should the manifest schema version bump be coordinated with a
  broader exported-package format change, or shipped as a standalone breaking
  change?

