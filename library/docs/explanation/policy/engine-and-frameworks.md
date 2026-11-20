# Native Framework Engines and Policy Plugins

> **Status:** The Engine API, first-party `Trainer` integration, and the LeRobot engine with its locked worker environments are implemented. Repository plugins (GR00T N1.7, StarVLA), Studio catalog/UI integration, and XPU validation are follow-ups. This document describes the architecture and its current evidence, not a complete support commitment for every policy or framework.

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
               ├── Trainer: TrainingEngine + Lightning Trainer, in-process
               ├── LeRobotEngine: native LeRobot config/CLI, bundled locked worker
               └── repository or framework engines, added as plugins
                    │
           declarative worker environments (bundled or Git + verified patches)
                    ▼
             native checkpoint / Runtime package
                    ▼
        Runtime InferenceModel ← plugin RuntimeAdapter
                                owns model-specific preprocessing,
                                inference, postprocessing, and reset
```

Dependencies point down. Engine and policy plugins do not import Studio UI or backend types. Studio owns presentation, project persistence, training targets, and job orchestration. Runtime loads packages through the plugins' registered inference adapters.

## Engine API

`physicalai.engine` separates what every engine does from what only some engines do. `Engine` packages models for Runtime and reports its capabilities. `TrainingEngine` adds native training. A repository plugin that only packages or exports models subclasses `Engine` and needs no training stubs; `pai train` rejects it with a clear error.

```python
class Engine(ABC):
    framework: ClassVar[str]
    env: ClassVar[WorkerEnv | None] = None               # the framework's own environment
    enablements: ClassVar[tuple[Enablement, ...]] = ()   # each may declare its own environment
    stock_export_formats: ClassVar[tuple[str, ...]] = ()

    def export(self, source, output_path, backend=None) -> Path: ...   # abstract
    def policies(self) -> list[str]: ...                               # abstract
    def capabilities(self, policy: str) -> Capabilities: ...
    def environments(self) -> tuple[WorkerEnv, ...]: ...
    def environment(self, name: str | None = None) -> WorkerEnv: ...


class TrainingEngine(Engine):
    training_accelerators: ClassVar[tuple[str, ...]] = ()

    def cli(self, argv: Sequence[str], *, interpreter: Path | None = None) -> int: ...   # abstract
    def run(self, settings, *, dataset, output_dir, device, hooks=None, interpreter=None) -> Run: ...
    def config_schema(self, policy: str) -> dict: ...                                   # abstract
```

`capabilities()` reports training accelerators only for `TrainingEngine` subclasses. The base also provides `Engine.create(name, **options)`, `Engine.get(name)`, `Engine.names()`, and `Engine.load(package)`. `Engine.get()` returns the engine class; `Engine.create()` constructs an instance. Use the concrete class directly when the framework is already known and strong IDE/type-checker support is useful.

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

If an upstream contribution is not accepted or cannot land in time, an enablement can provide the capability without changing the default framework plugin. The preferred fallback is our own overlay code (for example, a wrapper or a scoped export-time adaptation). A published downstream patch series from another team may also be used, subject to SDL and license review. If applying another team's patches is not approved, use upstream changes or our own overlay code.

A plugin declares its patches as one or more `PatchSeries` on a `GitProject`. The series type records who owns the patches:

| Series             | Patches live                                   | Content pinned by                                   | Use when                                                       |
| ------------------ | ---------------------------------------------- | --------------------------------------------------- | -------------------------------------------------------------- |
| `PublishedPatches` | In another repository at an immutable revision | Per-file SHA-256 digests and the resulting Git tree | Another team owns the patches, such as Robotics AI Suite today |
| `BundledPatches`   | In the plugin, reviewed in this repository     | Digests computed from the files; optional tree pin  | The patch maintainers contribute here                          |

A published series can also name a local `mirror` for offline builds; a mirror changes where bytes come from, not who owns them. Several series can be layered, for example the Suite's published series plus fixes owned by this repository, or a series moved into the plugin a few patches at a time.

Runtime isolate applies patches with `git am` only to a clean, cached checkout of the pinned upstream commit, never to a user's checkout or model package. It may create that checkout while preparing an inference worker. When the last series pins its resulting tree, the source identity is the commit plus that tree; isolate combines it with the sync profile to derive the cache key. Moving the same patches between repositories therefore keeps the source identity stable while provenance records still show where each patch came from. Runtime owns this shared materialization logic, so each plugin only declares its patch series. Once a change lands upstream, the plugin drops the series and pins the upstream release instead.

[Robotics AI Suite's model pipelines](https://github.com/open-edge-platform/robotics-ai-suite/tree/main/src/pipelines) illustrate both approaches: some wrap an unmodified framework for conversion, while others pin an upstream repository and apply downstream patch series. They are examples of the enablement pattern, not dependencies of this project.

### Enablement declaration and selection

An Engine may declare which additional capabilities have been validated for each policy. `Enablement` records a name, an upstream reference, per-policy capabilities, and, when the capability needs different code, its own environment:

```python
Enablement(
    name="intel-xpu",
    upstream="acme-policy@<commit>",
    provides={"act": {"train": ("xpu",)}},
)

Enablement(
    name="openvino",
    provides={"acme_policy": {"export": ("openvino",)}},
    env=WorkerEnv(..., name="openvino", project=GitProject(upstream, commit, patches=series)),
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

| Capability                                                    | Status                                                                                                               |
| ------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------- |
| Engine-level `Enablement` metadata and `capabilities(policy)` | Implemented; used to report declared support.                                                                        |
| Native LeRobot training, export, and Runtime adapter          | Implemented and validated for selected ACT/Diffusion paths.                                                          |
| Automatic worker profile from native device or hardware       | Implemented for LeRobot; explicit pre-provisioning is available with `pai engine prepare --accelerator`.             |
| Separate locked environment for each enablement               | Runtime isolate manages source/profile caches; Engine retains `Enablement.env` and framework metadata.               |
| Downloading and applying another team's patch series          | Implemented in Runtime isolate (`GitProject`, `PatchSeries`) with pinned digests and tree check; SDL review pending. |
| Concrete OpenVINO/ONNX/ExecuTorch enablement recipes          | Not implemented here; an OpenVINO enablement for GR00T N1.7 is a follow-up.                                          |

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

Conflicting framework and Runtime dependencies should not be installed into one interpreter. For example, Isaac-GR00T N1.7 pins `transformers==4.57.3`, while Runtime requires a newer release. Runtime owns project materialization, cache management, synchronization, and process launch in `physicalai.isolate`. The optional Engine layer maps `WorkerEnv` metadata to isolate environments and adds accelerator selection, lock/version checks, and the training worker protocol.

A `WorkerEnv` names its uv project source and how to synchronize it:

| Field              | Purpose                                                                                                                                                                                                                                       |
| ------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `project`          | `BundledProject` (a project and lock shipped in the plugin, as for LeRobot) or `GitProject` (a framework's own project at a commit, optionally with `PublishedPatches` or `BundledPatches` layers). A plain directory means `BundledProject`. |
| `name`             | Distinguishes the engine's own environment (`default`) from enablement environments.                                                                                                                                                          |
| `python`, `extras` | uv `--python` version and the uv extras for each accelerator profile (`cpu`, `cuda`, `xpu`), for example `{"cuda": ("cu132",)}`; a single-profile environment needs no hardware selection.                                                    |
| `worker`           | Default worker script; `run_worker(..., worker=...)` selects another script, for example an export worker.                                                                                                                                    |

Runtime isolate caches environments by the source and selected `SyncProfile` (Python version and extras) at `~/.cache/physicalai/isolate/<key>`; `PHYSICALAI_ISOLATE_CACHE` overrides the root. Engine preparation uses this same cache and records framework-specific metadata for `physicalai engine list` and `prune`. The `PHYSICALAI_ENGINE_CACHE` override remains available to Engine-only deployments. Engine passes `PHYSICALAI_WORKER_PROJECT` to workers so scripts in a framework checkout are reachable.

The worker handshake checks the framework package against the environment's exact lock. A plugin that depends on a PyPI framework also declares its supported range, which the handshake checks as before; frameworks pinned by Git commit declare no range.

The LeRobot plugin declares its supported LeRobot range in its main `pyproject.toml`. Its worker `env/pyproject.toml` describes training dependencies and CPU, CUDA 13.2 (`cu132`), and XPU PyTorch indexes, matching Studio's host build; `env/uv.lock` pins exact transitive versions. Policy families with their own extras (`smolvla`, the `pi` family, `molmoact2`) train in their own projects under `envs/`; every other policy uses `env/`. Engine verifies that each locked LeRobot version satisfies the plugin's declared range.

```text
physicalai-lerobot-policy-plugin/
├── engine.py       LeRobotEngine; no Studio imports
├── _worker.py      worker-side protocol and native training entry point
├── env/            default worker project
│   ├── pyproject.toml
│   └── uv.lock
├── envs/           per-family projects: smolvla/, pi/, molmoact2/
├── inference.py    lazy Runtime adapter registration
└── runtime.py      LeRobot inference adapter
```

The environment lifecycle is explicit and inspectable:

```bash
pai engine prepare lerobot --accelerator cuda                        # pre-provision; device still comes from LeRobot config
pai engine prepare lerobot --policy smolvla                          # a policy family's own environment
pai engine list                                                      # show cached environments
pai engine prune                                                     # remove environments no engine or enablement declares
```

`physicalai.isolate.prepare` materializes the source outside the user's checkout and runs `uv sync --locked --project <cached-project>` (`--frozen` for an upstream `GitProject`) with the selected Python and extras. `physicalai.engine.prepare_environment` adapts `WorkerEnv` to `Environment(source, SyncProfile(...))`, then preserves Engine metadata and lock checks. Managed workers launch through `physicalai.isolate.start` with the declared or overridden entrypoint; `interpreter=/path/to/python` bypasses preparation and uses a user-managed environment.

For integration tests that compare native LeRobot with `LeRobotEngine`, the host also needs the matching PyTorch extra. The tests select XPU/CUDA/CPU from available host PyTorch and reject a detected GPU that the host interpreter cannot use, rather than silently falling back:

```bash
cd library
uv sync --extra cu132 --extra tests --extra lerobot --frozen  # use xpu on an XPU host
PHYSICALAI_RUN_LEROBOT_INTEGRATION=1 PHYSICALAI_LEROBOT_TEST_STEPS=5 \
  uv run --no-sync pytest tests/integration/test_engine_lerobot_hf_training.py::test_native_lerobot_training_matches_engine_on_pinned_pusht -q
```

## Runtime packaging

The LeRobot engine packages the native `pretrained_model/` folder with a Runtime manifest. The installed plugin's `RuntimeAdapter` handles LeRobot preprocessing, stateful `select_action()`, postprocessing, and reset. Runtime's own generic preprocessors are not used for this plugin package. Exported packages contain model data; executable code stays in installed and registered plugins.

The LeRobot GR00T policy is not advertised or loaded: its loader enables `trust_remote_code=True`. It remains blocked until remote code is reviewed and pinned. NVIDIA's own GR00T N1.7 is planned as a separate repository plugin.

## Studio integration and enablements

Studio owns labels, forms, dataset snapshots, training targets, and job persistence. Extending the existing #1294 policy catalog to discover `policies()`, `config_schema()`, and `capabilities()` is a follow-up; it is not implemented in this PR. Studio plugins must not import Studio-specific UI code.

Enablements are upstream-first. A plugin may declare validated per-policy capabilities such as XPU training or an export format. No plugin here declares a downstream patch enablement yet; SDL review of Runtime isolate patch materialization is required before one ships. The final fallback for a brittle or unsupported integration is a first-party policy implementation.

## Validation

| Area                         | Result         | Evidence / limit                                                                                                                                                                               |
| ---------------------------- | -------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| First-party Lightning API    | **Pass**       | `fit`, `validate`, `test`, `predict` are exact inherited Lightning methods/signatures; `Trainer.export()` option forwarding is unit-tested.                                                    |
| First-party GPU train/eval   | **Pass**       | ACT `fit`, `validate`, `test`: 3 E2E cases passed on RTX A6000.                                                                                                                                |
| First-party export/inference | **Pass**       | ACT OpenVINO export-to-Runtime + numerical consistency: 2 passed; OpenVINO selected CPU for inference. `Trainer.export()` → Runtime is not yet E2E-tested.                                     |
| LeRobot GPU training parity  | **Pass**       | Native vs Engine ACT/PushT, 20 steps on RTX 3090, both in the LeRobot worker environment; config, losses, and final weights match exactly (`rtol=1e-6`, `atol=1e-7`) with deterministic cuDNN. |
| LeRobot Runtime inference    | **Pass**       | Pinned pretrained ACT (ALOHA transfer cube) and Diffusion (PushT) on RTX 3090: Runtime actions match native LeRobot before and after reset.                                                    |
| Managed CUDA environment     | **Pass**       | Native `policy.device=cuda` selected the `cuda` profile (`cu132` extra); worker project prepared with `uv sync --locked`; LeRobot 0.7.0 / PyTorch 2.15.0+cu132 on a CUDA 13.0 driver.          |
| LeRobot policy environments  | **Pass**       | SmolVLA (from `lerobot/smolvla_base`) and Pi0.5 fine-tune for 20 steps on RTX 3090 in their own worker environments and package. MolmoAct2 is not trained.                                     |
| CLI passthrough              | **Pass**       | Native `--help`: exit 0, 704 stdout lines; invalid native flag: exit 2.                                                                                                                        |
| Engine core and plugins      | **Pass**       | 120 Engine unit tests and 12 LeRobot plugin tests.                                                                                                                                             |
| Unit/hooks/package           | **Pass**       | `uv lock --check`, Ruff on touched production modules, and mypy on changed Python modules pass. The full library suite is tracked separately below.                                            |
| XPU                          | **Not tested** | No XPU hardware run.                                                                                                                                                                           |
| Full library suite           | **Incomplete** | The full run timed out after 30 minutes without a final summary; the focused Engine and plugin suites above passed.                                                                            |

With cuDNN's default nondeterministic mode, the first cross-process comparison exceeded the tight tolerance (maximum loss delta about 0.0021). The parity test now sets LeRobot's deterministic cuDNN option for both runs rather than loosening the tolerance.

## Follow-ups

- Extend Studio's #1294 policy catalog/UI/job path to consume engine policies, schemas, and capabilities.
- Exercise Lightning `Trainer.predict()` and first-party `Trainer.export()` → Runtime end to end.
- Validate XPU worker provisioning and training.
- Complete SDL review of Runtime isolate patch materialization (`GitProject`, `PatchSeries`).
- Land the Runtime `close()` lifecycle; managed project isolation now lives in Runtime while training progress hooks remain Engine-side.
- Add repository plugins for GR00T N1.7 and StarVLA.
