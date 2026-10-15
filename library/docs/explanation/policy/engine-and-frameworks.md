# Native Framework Engines and Policy Plugins

> **Status:** Engine API, first-party `Trainer` integration, LeRobot engine, and locked CUDA worker environment are implemented in the Engine POC. Studio catalog/UI integration and XPU validation remain follow-ups. This document describes the architecture and its current evidence, not a complete support commitment for every policy or framework.

## Purpose

Robotics policies are developed in different frameworks, each with its own model configuration, training loop, checkpoint format, device options, and inference processing. Reimplementing those recipes inside a single training loop can produce behavior that differs from the original framework.

This design lets a framework or model repository retain its own training behavior while Physical AI provides a common place to discover engines, launch native commands, manage isolated environments, and load packaged models through Runtime.

The design supports four stages of model enablement:

1. A **framework plugin** exposes policies already supported by a training framework.
2. A **repository plugin** provides early support for a policy from its own repository.
3. An **enablement** adds a capability, such as accelerator training or an optimized export. The preferred route is to contribute changes upstream; downstream overlays may bridge the gap.
4. A **first-party policy** is implemented in Studio when an external integration is too fragile or cannot meet product requirements.

The system should stop at the earliest stage that meets the required quality and deployment needs.

## Architecture

```text
Python API                     pai CLI                       Studio (follow-up)
     │                             │                         extends policy catalog
     └──────────────┬──────────────┴─────────────────────────┘
                    ▼
             Engine registry (`physicalai.engines`)
               ├── Trainer: Engine + Lightning Trainer, in-process
               ├── LeRobotEngine: native LeRobot config/CLI, locked worker
               └── repository or framework engines, added as plugins
                    │
           shared worker environments and process support
                    ▼
             native checkpoint / Runtime package
                    ▼
        Runtime InferenceModel ← plugin RuntimeAdapter
                                owns model-specific preprocessing,
                                inference, postprocessing, and reset
```

Dependencies point down. Engine and policy plugins do not import Studio UI or backend types. Studio owns presentation, project persistence, training targets, and job orchestration. Runtime loads packages through the plugins' registered inference adapters.

## Engine API

`physicalai.engine.Engine` is a lightweight base class and registry. It defines the operations common to frameworks, but it does not prescribe a constructor, a settings type, or a single training method signature.

```python
class Engine(ABC):
    framework: ClassVar[str]

    # Implemented per engine, with framework-specific signatures as needed:
    def cli(self, argv: Sequence[str], *, interpreter: Path | None = None) -> int: ...
    def run(self, settings, *, dataset, output_dir, device, hooks=None, interpreter=None) -> Run: ...
    def export(self, source, output_path, backend=None) -> Path: ...
    def policies(self) -> list[str]: ...
    def config_schema(self, policy: str) -> dict: ...
    def capabilities(self, policy: str) -> Capabilities: ...
```

The base also provides `Engine.create(name, **options)`, `Engine.get(name)`, `Engine.names()`, and `Engine.load(package)`. `Engine.get()` returns the engine class; `Engine.create()` constructs an instance. Use the concrete class directly when the framework is already known and strong IDE/type-checker support is useful.

`Run` describes a completed or in-progress native run. `TrainHooks` is a framework-neutral progress and cooperative-cancellation interface for headless runs. Engines may add framework-specific methods without changing the base interface.

### First party: Trainer

`physicalai.train.Trainer` is both the first-party engine and the existing Studio Trainer. It inherits Lightning's `fit`, `validate`, `test`, and `predict` methods rather than wrapping them. Its existing constructor and first-party training API remain available.

```python
from physicalai.data import LeRobotDataModule
from physicalai.policies import ACT
from physicalai.train import Trainer

trainer = Trainer(max_epochs=10, precision="bf16-mixed", accelerator="gpu")
trainer.fit(ACT(), datamodule=LeRobotDataModule(repo_id="lerobot/pusht"))
trainer.validate(...)
trainer.test(...)
trainer.predict(...)
```

The same engine can be selected dynamically. `Engine.create()` forwards constructor options to the selected engine; `Engine.get()` returns the class to instantiate:

```python
from physicalai.engine import Engine

trainer = Engine.create("physicalai", max_epochs=10, precision="bf16-mixed", accelerator="gpu")
trainer.fit(policy, datamodule=data)

TrainerClass = Engine.get("physicalai")
trainer = TrainerClass(max_epochs=10, precision="bf16-mixed", accelerator="gpu")
```

`Trainer.export()` forwards first-party policy export options, including `input_sample`, post-export hooks, and backend-specific options. The first-party Studio job runner remains `run_training_job()`; `Trainer.run()` is not yet wired to that Studio-specific path.

### LeRobot: native config and methods

`LeRobotEngine` retains LeRobot's own configuration and trainer. Python callers pass `TrainPipelineConfig`; the engine serializes it and runs LeRobot in its locked worker environment.

```python
from lerobot.configs.default import DatasetConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.policies.act.configuration_act import ACTConfig
from physicalai.engine import Engine

engine = Engine.create("lerobot")
config = TrainPipelineConfig(
    dataset=DatasetConfig(repo_id="lerobot/pusht"),
    policy=ACTConfig(device="cuda", push_to_hub=False),
    steps=20_000,
)
run = engine.train(config)
package = engine.export(run, "exports/act", backend="native")
model = Engine.load(package)
```

The engine can also be imported directly:

```python
from physicalai_lerobot_policy_plugin.engine import LeRobotEngine

engine = LeRobotEngine(interpreter="/opt/lerobot/bin/python")  # optional user-managed environment
run = engine.train(config)
```

The host and worker LeRobot versions must match when Python config objects cross the process boundary. Otherwise, use LeRobot's native flags or a config file.

### Other frameworks and model repositories

A framework plugin can expose every policy type available in that framework. A repository plugin can expose one policy from an independent research repository. Each defines its own constructor and methods:

```python
class AcmeEngine(Engine):
    framework = "acme"

    def __init__(self, *, interpreter: Path | None = None): ...
    def train(self, config: AcmeTrainConfig) -> Run: ...
    def evaluate(self, config: AcmeEvalConfig) -> Metrics: ...  # optional framework-specific operation
```

Plugins register their engine class through the `physicalai.engines` entry-point group. The plugin's base install registers its Runtime adapter; installing `physicalai-train` is needed only for training-side Engine discovery and execution.

## Model enablement

The goal is to make a new model usable early without permanently taking ownership of its implementation. Enablement proceeds in stages; a project can stop at any stage that meets its requirements.

| Stage              | Scope                                              | Typical use                                                                                           |
| ------------------ | -------------------------------------------------- | ----------------------------------------------------------------------------------------------------- |
| Framework plugin   | Multiple policies provided by one framework        | Use LeRobot's native trainer and policies through one `LeRobotEngine`.                                |
| Repository plugin  | One model or family from an independent repository | Support a newly released repo on day 0 or 1, before considering a first-party port.                   |
| Enablement         | An additional capability for selected policies     | Add XPU training or an optimized export to a framework or repository plugin.                          |
| First-party policy | A policy implemented and maintained in Studio      | Use when an external integration cannot meet reliability, compatibility, or performance requirements. |

### Upstream-first maintenance

Changes that make a framework work with a device or export format should be contributed upstream and maintained there whenever possible. A plugin then selects the upstream release or commit that contains the capability. A downstream enablement is a bridge while upstream work is pending; once accepted, it should be reduced to an upstream version update and the downstream workaround removed.

If an upstream contribution is not accepted or cannot land in time, an enablement can provide the capability without changing the default framework plugin. The preferred fallback is our own overlay code (for example, a wrapper or a scoped export-time adaptation). A published downstream patch series from another team may also be referenced, subject to SDL approval. The plugin should identify the source repository, exact revision, selected patch files, and checksums; it should not copy the patch series into the plugin or maintain a fork as the normal path. If applying another team's patches is not approved, use upstream changes or our own overlay code instead.

[Robotics AI Suite's model pipelines](https://github.com/open-edge-platform/robotics-ai-suite/tree/main/src/pipelines) illustrate both approaches: some wrap an unmodified framework for conversion, while others pin an upstream repository and apply downstream patch series. They are examples of the enablement pattern, not dependencies of this project.

### Enablement declaration and selection

An Engine may declare which additional capabilities have been validated for each policy. The current `Enablement` type records a name, upstream reference, and per-policy capabilities:

```python
Enablement(
    name="intel-xpu",
    upstream="acme-policy@<commit>",
    provides={"act": {"train": ("xpu",)}},
)

engine.capabilities("act")
# Capabilities(training_accelerators=("cpu", "xpu"), ...)
```

The declaration is evidence about tested support, not an automatic promise that every policy in the framework supports that backend. A capability should be offered only after a corresponding test passes. For training, that means the native run uses the expected device and recipe. For export, it includes loading the generated package through the plugin's Runtime adapter and comparing its actions with the native model on the same observations.

Each enablement that needs different framework code or dependencies should use its own pinned worker project and lock. The environment identity should include the upstream revision, patch-set reference (when approved), accelerator profile, and lock hash. If an enablement uses a different framework revision than training did, checkpoint compatibility must be tested; otherwise, train and export in the same enablement environment.

The framework continues to own model-device settings. For LeRobot, `policy.device` determines the selected worker profile; `pai engine prepare --accelerator` is only an explicit environment pre-provisioning override. The Engine must not rewrite the framework's native training arguments.

### Runtime boundary

The installed plugin's Runtime adapter owns the full inference path for its package, including the framework's preprocessing, inference, postprocessing, action queues, and reset behavior. Runtime-specific processors are not required for a plugin package. The exported package contains model data and provenance; executable code remains in the installed, registered plugin.

If an enablement changes preprocessing or postprocessing, the robot needs the matching plugin/framework version. The package should record the enablement and upstream revision, and parity tests should exercise the same adapter and dependency version that deployment will use. Framework/runtime dependency conflicts remain an inference-isolation concern; an optimized model does not remove the plugin's framework dependencies if its adapter still uses them for preprocessing.

### Current implementation boundary

| Capability                                                    | Status in this POC                                                                                       |
| ------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------- |
| Engine-level `Enablement` metadata and `capabilities(policy)` | Implemented; used to report declared support.                                                            |
| Native LeRobot training, export, and Runtime adapter          | Implemented and validated for selected ACT/Diffusion paths.                                              |
| Automatic worker profile from native device or hardware       | Implemented for LeRobot; explicit pre-provisioning is available with `pai engine prepare --accelerator`. |
| Separate locked environment for each enablement               | Design direction; not yet implemented.                                                                   |
| Downloading and applying another team's patch series          | Not implemented; requires SDL approval.                                                                  |
| Concrete OpenVINO/ONNX/ExecuTorch enablement recipes          | Not implemented in this PR.                                                                              |

## CLI

First-party commands such as `pai fit`, `pai validate`, `pai test`, `pai predict`, `pai export`, and `pai benchmark` remain. The native-framework command passes everything after the framework identifier to that framework's CLI:

```bash
pai train lerobot --policy.type=act --dataset.repo_id=lerobot/pusht --steps=20000
pai train lerobot --policy.type=act --policy.device=xpu --dataset.repo_id=lerobot/pusht
pai train lerobot --help
pai train --interpreter /opt/lerobot/bin/python lerobot --config_path=./train_config.json
```

The native command receives the original flags, standard streams, help output, and exit status. `--interpreter` is a Physical AI execution option and comes before the framework name; it is not passed to LeRobot.

Device selection stays framework-native. For LeRobot, `--policy.device` passes through to `lerobot-train`; the engine uses its value to select the matching worker environment. If LeRobot does not specify a device, the engine detects available hardware. `pai engine prepare --accelerator` selects a worker profile when pre-provisioning an environment.

## Worker environments and dependencies

Conflicting framework and Runtime dependencies should not be installed into one interpreter. For example, LeRobot's Pi extra and Runtime currently require incompatible Transformers versions. Each worker-based engine therefore ships a small uv project and lock with its plugin; the host Engine prepares a cached environment for it.

The LeRobot plugin declares its supported LeRobot range in its main `pyproject.toml`. Its worker `env/pyproject.toml` describes training dependencies and CPU, CUDA (`cu128`), and XPU PyTorch indexes; `env/uv.lock` pins exact transitive versions. A test verifies that the locked LeRobot version satisfies the plugin's declared range.

```text
physicalai-lerobot-policy-plugin/
├── engine.py       LeRobotEngine; no Studio imports
├── _worker.py      worker-side protocol and native training entry point
├── env/
│   ├── pyproject.toml
│   └── uv.lock
├── inference.py    lazy Runtime adapter registration
└── runtime.py      LeRobot inference adapter
```

The environment lifecycle is explicit and inspectable:

```bash
pai engine prepare lerobot --accelerator cu128  # pre-provision; training device still comes from LeRobot config
pai engine list                                  # show cached environments
pai engine prune                                 # remove environments with obsolete locks
```

Conceptually, core copies the plugin's worker project outside the user's checkout, then runs `uv sync --frozen --project <cached-project> --extra <profile>`. The cache is keyed by engine, profile, and lock hash. `interpreter=/path/to/python` bypasses provisioning and uses a user-managed environment; the worker handshake checks the framework version.

For integration tests that compare native LeRobot with `LeRobotEngine`, the host also needs the matching PyTorch extra. The tests select XPU/CUDA/CPU from available host PyTorch and reject a detected GPU that the host interpreter cannot use, rather than silently falling back:

```bash
cd library
uv sync --extra cu128 --extra tests --extra lerobot --frozen  # use xpu on an XPU host
PHYSICALAI_RUN_LEROBOT_INTEGRATION=1 PHYSICALAI_LEROBOT_TEST_STEPS=5 \
  uv run --no-sync pytest tests/integration/test_engine_lerobot_hf_training.py::test_native_lerobot_training_matches_engine_on_pinned_pusht -q
```

## Runtime packaging

The LeRobot engine packages the native `pretrained_model/` folder with a Runtime manifest. The installed plugin's `RuntimeAdapter` handles LeRobot preprocessing, stateful `select_action()`, postprocessing, and reset. Runtime's own generic preprocessors are not used for this plugin package. Exported packages contain model data; executable code stays in installed and registered plugins.

The LeRobot GR00T policy is not advertised or loaded: its loader enables `trust_remote_code=True`. It remains blocked until remote code is reviewed and pinned.

## Studio integration and enablements

Studio owns labels, forms, dataset snapshots, training targets, and job persistence. Extending the existing #1294 policy catalog to discover `policies()`, `config_schema()`, and `capabilities()` is a follow-up; it is not implemented in this PR. Studio plugins must not import Studio-specific UI code.

Enablements are upstream-first. A plugin may declare validated per-policy capabilities such as XPU training or an export format. Downstream patch materialization and concrete OpenVINO/ONNX/ExecuTorch recipes are out of scope pending SDL review. The final fallback for a brittle or unsupported integration is a first-party policy implementation.

## Validation

| Area                         | Result         | Evidence / limit                                                                                                                                           |
| ---------------------------- | -------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------- |
| First-party Lightning API    | **Pass**       | `fit`, `validate`, `test`, `predict` are exact inherited Lightning methods/signatures; `Trainer.export()` option forwarding is unit-tested.                |
| First-party GPU train/eval   | **Pass**       | ACT `fit`, `validate`, `test`: 3 E2E cases passed on RTX A6000.                                                                                            |
| First-party export/inference | **Pass**       | ACT OpenVINO export-to-Runtime + numerical consistency: 2 passed; OpenVINO selected CPU for inference. `Trainer.export()` → Runtime is not yet E2E-tested. |
| LeRobot GPU training parity  | **Pass**       | Native vs Engine ACT/PushT, 5 steps on RTX A6000; config, losses, final weights match (`rtol=1e-6`, `atol=1e-7`) with deterministic cuDNN.                 |
| LeRobot Runtime inference    | **Pass**       | Pinned pretrained ACT and Diffusion on CUDA; 3 observations plus reset/replay; max action difference 0.                                                    |
| Managed CUDA environment     | **Pass**       | Native `policy.device=cuda` selected cu128; worker lock prepared with `uv sync --frozen`; LeRobot 0.6.1 / PyTorch 2.11.0+cu128.                            |
| CLI passthrough              | **Pass**       | Native `--help`: exit 0, 386 stdout lines; invalid native flag: exit 2.                                                                                    |
| Unit/hooks/package           | **Pass**       | Engine 58 focused tests; 6 gated skips; plugin tests 2; backend was not changed; library hooks, plugin Ruff/mypy, lock checks, wheel/sdist checks passed.  |
| XPU                          | **Not tested** | No XPU hardware run.                                                                                                                                       |
| Full library suite           | **Incomplete** | No full-suite pass claimed; a full run was interrupted with first-party E2E failures/errors.                                                               |

With cuDNN's default nondeterministic mode, the first cross-process comparison exceeded the tight tolerance (maximum loss delta about 0.0021). The parity test now sets LeRobot's deterministic cuDNN option for both runs rather than loosening the tolerance.

## Follow-ups

- Extend Studio's #1294 policy catalog/UI/job path to consume engine policies, schemas, and capabilities.
- Exercise Lightning `Trainer.predict()` and first-party `Trainer.export()` → Runtime end to end.
- Validate XPU worker provisioning and training.
- Implement enablement patch materialization and optimized export recipes only after SDL review.
