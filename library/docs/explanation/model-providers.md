# Engine and Upstream Provider Design

- **Status:** Proposal for discussion; not an implemented API.
- **Scope:** Library-first training, benchmarking, and export orchestration in `physicalai-train`, consumed by Python, CLI, and GUI.
- **Decision requested:** Adopt an additive `Engine` facade with both Lightning integrations and upstream-native providers.
- **Public API decision:** Do not expose `ModelRef` or `DatasetRef` as primary user-facing entities. Concrete provider policy objects, existing `Policy`/`DataModule` objects, dataset paths, and native recipes are the preferred public inputs. Serializable references remain internal workflow/provenance representations.

All new API names, package names, entry-point groups, configuration schemas, and provider examples below are proposed pseudocode. Existing APIs are identified explicitly. Placeholder revisions must be replaced with approved immutable revisions before execution. Source inspection informs this proposal; no upstream training, benchmarking, or export was executed to validate it.

## Reading Guide

- [Recommendation and trade-offs](#1-recommendation)
- [Domain model](#4-domain-model) and [architecture](#5-architecture)
- [First-party API pipeline](#6-public-api-first-party-and-lightning-workflows)
- [LeRobot, GR00T, and StarVLA API examples](#7-public-api-upstream-providers)
- [Configuration](#8-configuration-contract) and [API/CLI/GUI parity](#9-api-cli-and-gui-parity)
- [Provider internals and delegation](#10-how-providers-delegate)
- [Environments](#11-execution-environments) and [data/inference boundaries](#12-data-and-inference-boundaries)
- [AllenAI benchmark integration](#13-benchmark-providers-and-allenai)
- [Migration](#16-compatibility-and-migration), [validation plan](#17-validation-and-rollout-plan), and [open decisions](#19-open-decisions)

## 1. Recommendation

**Keep the existing Lightning integration and add upstream-native providers behind a framework-neutral `Engine`. Prefer native delegation for new independent repositories; retain wrappers where deep Lightning integration justifies their maintenance.**

The alternatives are not simply "wrapper versus plugin":

- **Packaging/discovery:** built-in implementation or externally installed plugin.
- **Execution ownership:** PhysicalAI/Lightning owns training, or the original framework owns training.
- **Process boundary:** same process or isolated execution environment.

A plugin can provide a Lightning wrapper, a native workflow, or both. LeRobot is a concrete case where both routes are useful. Moving a Lightning wrapper into another package does not remove training adaptation or dependency conflicts.

### 1.1 Engineering trade-offs

| Criterion                               | Lightning wrapper                                                            | Upstream-native provider                                                  |
| --------------------------------------- | ---------------------------------------------------------------------------- | ------------------------------------------------------------------------- |
| Initial integration with current Studio | Strong fit; existing trainer, data, and checkpoint conventions               | Requires shared orchestration and result infrastructure first             |
| Adding many independent repositories    | Each new training stack requires adaptation                                  | Reuse upstream workflows through repository-specific adapters             |
| Native recipe fidelity                  | Must validate lifecycle, scheduler, precision, data, and checkpoint behavior | Preserves upstream behavior if invocation and inputs remain equivalent    |
| Deep customization                      | Direct access to modules, gradients, callbacks, and distributed strategies   | Limited to upstream extension points and declared provider capabilities   |
| Dependency compatibility                | Usually requires one compatible dependency graph                             | Separate interpreters or containers accommodate incompatible graphs       |
| Debugging                               | Familiar single-process Lightning debugging                                  | Multiple processes, upstream logs, readiness, and environment diagnostics |
| Configuration UX                        | Uniform Lightning settings                                                   | Common operation envelope with provider-specific settings                 |
| Operational cost                        | Lower for already compatible models                                          | Process startup, environment administration, artifact and event bridges   |
| Upstream maintenance                    | Model plus training integration must track upstream                          | Launcher/configuration/artifact contracts must track upstream             |
| XPU and export work                     | Convenient access to model internals; still requires validation              | Can use upstream code/tooling, but needs separate hardware/export work    |
| Long-term role                          | Deep integration for selected workflows                                      | Broad access and preservation of original framework behavior              |

Neither route guarantees correctness, performance, deployment support, or zero maintenance. Native delegation reduces the code we translate; it does not remove data, action-semantics, dependency, or artifact responsibilities.

### 1.2 Selection rule

1. Reuse a validated existing integration when it satisfies the requested workflow.
2. For a new independent repository, establish the upstream baseline and add a native provider first.
3. Add a Lightning route when callbacks, training customization, hardware work, or product requirements justify it.
4. Propose first-party model code only when measured limitations of the upstream/adapter route justify ownership.
5. Do not rewrite the working LeRobot wrapper merely to make every implementation look alike.

The dependency boundary is concrete, not hypothetical: at the inspected GR00T revision, upstream pins `peft==0.17.1` and `transformers==4.57.3`, while this Studio checkout requires `peft>=0.20.0` and its VLA extras use Transformers 5.5.x. The current first-party GR00T config describes N1.5, not the requested N1.7. A new native provider avoids conflating those implementations or forcing an incompatible shared environment.

This aligns with the draft [Model Enablement Guidelines](https://github.com/open-edge-platform/physical-ai-studio/blob/docs/design-docs/library/docs/design/model-guidelines/model-implementation-guidelines.md): integration route, support level, capability evidence, and portfolio investment are independent attributes.

## 2. Goals and Non-goals

### Goals

- Provide a simple `Engine` API for first-party and upstream-backed workflows.
- Preserve the existing Lightning Python API and CLI configurations.
- Keep model architecture and native training logic in their original repositories.
- Make serializable workflow specifications behave consistently across API, CLI, and GUI.
- Consume environments created with `uv`, Conda, pip, or container tooling.
- Expose native checkpoints as legitimate artifacts without forcing Lightning conversion.
- Add benchmark providers without replacing authoritative upstream evaluation protocols.
- Make unsupported operations and incompatible combinations fail before expensive execution.

### Non-goals

- Automatically integrate any arbitrary GitHub URL without provider code.
- Replace Lightning, package managers, application job scheduling, or Runtime inference runners.
- Make all native trainers accept Lightning settings or reproduce Lightning results.
- Guarantee every model/benchmark pairing, hardware target, or export format.
- Build environment provisioning, cluster scheduling, or a plugin marketplace in the first version.
- Turn Python callbacks or live model objects into arbitrary cross-process serialized code.

## 3. Current Implementation and Ownership

The current library is Lightning-centered:

- `library/src/physicalai/train/trainer.py`: `Trainer` subclasses `lightning.Trainer`.
- `library/src/physicalai/policies/base/policy.py`: `Policy` subclasses `LightningModule`.
- `library/src/physicalai/policies/lerobot/policy.py`: the wrapper delegates model/loss/processor behavior while adapting training, data interaction, and checkpoints to Lightning.
- `library/src/physicalai/policies/__init__.py`: `get_policy()` dispatches the `physicalai` and `lerobot` sources and returns a `Policy`.
- `library/src/physicalai/cli/fit.py`: the existing fit command dispatches to Lightning.
- `library/src/physicalai/benchmark/gyms/benchmark.py`: the built-in benchmark owns gym rollouts and aggregation.
- `library/src/physicalai/cli/benchmark.py`: loading and cleanup assume the current policy/gym interfaces.
- `application/backend/src/training/job.py`: execution assumes Lightning settings, datamodules, and a `model.ckpt` artifact.
- `application/backend/src/services/training_backends/`: local and remote application execution already have abstractions; reuse their scheduling/transport roles rather than building a parallel scheduler.

Runtime owns the `physicalai` executable, `pai`, `physicalai.config`, `InferenceModel`, and inference runners. Studio contributes CLI subcommands and owns the export side of the Runtime contract. Do not create another `physicalai/config/` package or call the new orchestrator `Runner`.

## 4. Domain Model

| Entity                 | Meaning                                                                   | Not equivalent to                                                 |
| ---------------------- | ------------------------------------------------------------------------- | ----------------------------------------------------------------- |
| `Engine`               | Public orchestration facade for supported operations                      | A neural network, package manager, scheduler, or inference runner |
| Provider               | Adapter for a particular implementation repository/framework              | Architecture name or support guarantee                            |
| Provider policy object | Concrete first-party or provider-backed policy used by `Engine`           | A guarantee that it is a local `torch.nn.Module`                  |
| Dataset input          | Existing `DataModule`, dataset path, or provider-native recipe            | A universal dataset/dataloader abstraction                        |
| `Policy`               | Existing loaded Lightning-facing policy                                   | A framework-neutral checkpoint reference                          |
| `Model`                | Existing neural-network implementation abstraction                        | A framework-neutral provider policy                               |
| DataModule             | Existing Lightning data lifecycle and loaders                             | A native dataset path or recipe                                   |
| Framework route        | `lightning` or provider-relative `native` training                        | Hardware backend or process location                              |
| Execution target       | Interpreter/launcher, working directory, and execution constraints        | Training implementation or model provider                         |
| Workflow specification | Serializable operation inputs and options                                 | A backend database job                                            |
| Execution plan         | Validated, resolved invocation and artifact/event expectations            | An arbitrary user-supplied shell command                          |
| Artifact reference     | Native checkpoint, weights bundle, or validated deployment export         | Guaranteed full training-state resume                             |
| Policy session         | Loaded inference resource with reset, prediction, and close semantics     | A training engine                                                 |
| Benchmark provider     | Adapter for an evaluation harness and its protocol                        | Just a Gym environment                                            |

Do not expose a universal `Model` or `Dataset` reference layer in the first public API. It would obscure the distinction between a loaded Lightning policy and an upstream execution handle. Provider packages should expose concrete policy objects such as `Groot` or `StarVLA`, while `Engine.fit()` accepts existing `Policy`/`DataModule` objects, dataset paths, or native recipes. The Engine may create private serializable references during planning and provenance collection. Resolve and validate immutable source/checkpoint identities during explicit preparation/preflight; require approved pinned Hub revisions before downloading.

## 5. Architecture

```text
Python API -------------------------------+
                                          |
Runtime-hosted physicalai CLI ------------+--> Engine (physicalai-train)
                                          |       |
GUI --> FastAPI --> application worker ---+       +-- specifications and validation
                                                  +-- provider discovery
                                                  +-- operation planning and events
                                                  |
                           +----------------------+--------------------+
                           |                                           |
                    Lightning route                             Native provider route
                           |                                           |
                    Existing Trainer                         Process/API/CLI execution
                           |                                           |
                    Policy + DataModule                     Original framework workflow
                           |                                           |
                           +----------------------+--------------------+
                                                  |
                                      Results and artifact references
```

Dependencies point downward: the Engine does not import FastAPI, backend database models, or GUI code. The backend owns authorization, persistence, scheduling, and storage. It calls library operations; it does not own the only implementation of external model training.

Model and benchmark frameworks can run in different environments. Library orchestration in one process does not require model computation in that process.

## 6. Public API: First-party and Lightning Workflows

### 6.1 Existing direct API remains unchanged

This is the existing public style, not a proposed replacement:

```python
from physicalai.data import LeRobotDataModule
from physicalai.policies import ACT
from physicalai.train import Trainer

policy = ACT()
datamodule = LeRobotDataModule(repo_id="lerobot/pusht", train_batch_size=8)
trainer = Trainer(max_epochs=10)
trainer.fit(model=policy, datamodule=datamodule)
```

Users retain direct access to callbacks, modules, gradients, strategies, and the Lightning trainer. `get_policy()` continues to return a `Policy`; it must not start returning remote handles.

### 6.2 Engine around the existing objects

```python
from physicalai.engine import Engine
from physicalai.data import LeRobotDataModule
from physicalai.policies import ACT

policy = ACT()
datamodule = LeRobotDataModule(repo_id="lerobot/pusht", train_batch_size=8)

engine = Engine(
    framework="lightning",
    trainer={"max_epochs": 10, "accelerator": "auto"},
    output_dir="./runs/act",
)
result = engine.fit(model=policy, datamodule=datamodule)
```

Keep Lightning-specific arguments under `trainer` rather than accepting arbitrary Lightning keywords on a supposedly framework-neutral Engine. Reject `trainer` for a native route. Output/checkpoint settings must have one documented owner; reject conflicting output directories rather than silently overriding them.

The Engine delegates to the existing `Trainer`, then collects declared artifacts. It does not automatically export every backend as a side effect of successful training.

### 6.3 Simple provider-facing workflow

```python
from physicalai.engine import Engine
from physicalai.policies import ACT

policy = ACT()
engine = Engine(
    framework="lightning",
    trainer={"max_epochs": 10},
    output_dir="./runs/act",
)
result = engine.fit(
    model=policy,
    dataset="./datasets/pusht",
    data_options={"format": "lerobot-v3", "train_batch_size": 8, "num_workers": 4},
)
```

For Lightning, `model` can be an existing `Policy` and `dataset` can be an existing `DataModule`. For native providers, the provider exposes a concrete policy object and accepts a path or native recipe. Private references may be created during planning, but users do not construct them.

### 6.4 Train, benchmark, export, and load

```python
from physicalai.workflows import BenchmarkRef
from physicalai.inference import InferenceModel

benchmark = BenchmarkRef(provider="physicalai", suite="pusht")
scores = engine.benchmark(
    model=result.model,
    benchmark=benchmark,
    options={"num_episodes": 20, "seed": 42},
)

artifact = engine.export(
    model=result.model,
    backend="openvino",
    output_dir="./exports/act",
)

# Existing Runtime entry point, used only for a compatible validated export.
inference_model = InferenceModel(artifact.path)
inference_model.reset()
action = inference_model.select_action(observation)
```

`result.model` is a provider-backed trained policy/artifact handle whose concrete type depends on the provider; it is not promised to be a resident Torch module. `framework` chooses the fit route; benchmark and export dispatch by their own provider/capability contracts. The complete pipeline above is a target experience, not evidence that the new orchestration path already works.

## 7. Public API: Upstream Providers

### 7.1 LeRobot: same implementation, two training routes

```python
from physicalai.engine import Engine
from physicalai_lerobot import LeRobotPolicy

model = LeRobotPolicy.from_pretrained(
    "lerobot/smolvla_base",
    revision="<approved-40-character-checkpoint-SHA>",
)
dataset = "./datasets/my_robot"

# Existing wrapper and PhysicalAI/Lightning training lifecycle.
wrapped = Engine(
    framework="lightning",
    trainer={"max_steps": 10_000},
    output_dir="./runs/lerobot-lightning",
).fit(model=model, dataset=dataset, data_options={"train_batch_size": 32})

# LeRobot's own training lifecycle in a configured upstream environment.
native = Engine(
    framework="native",
    execution="lerobot-local",
    output_dir="./runs/lerobot-native",
).fit(
    model=model,
    dataset=dataset,
    native_config={"steps": 10_000, "batch_size": 32},
)
```

The provider owns checkpoint-loading and dataset mappings for each route. The routes are not claimed to produce identical optimization trajectories. Common configuration shape does not prove native training parity.

The current wrapper lists GR00T among named policies, but its declared generic equivalence-tested subset is ACT, Diffusion, and SmolVLA. Do not promote all discoverable models to validated support.

### 7.2 Isaac-GR00T: native fine-tuning

```python
from physicalai.engine import Engine
from physicalai_isaac_groot import Groot, PythonEnvironment

model = Groot.from_pretrained(
    "nvidia/GR00T-N1.7-3B",
    revision="<approved-40-character-checkpoint-SHA>",
)

# This example deliberately starts with an already-prepared GR00T dataset.
dataset = "/work/Isaac-GR00T/demo_data/cube_to_bowl_5"

environment = PythonEnvironment(
    python="/work/Isaac-GR00T/.venv/bin/python",
    working_dir="/work/Isaac-GR00T",
    expected_revision="51d4c89f72fda44cbf77285c6a8114b52676b8a1",
)

engine = Engine(
    framework="native",
    execution=environment,
    output_dir="./runs/gr00t",
)

result = engine.fit(
    model=model,
    dataset=dataset,
    native_config={
        "embodiment_tag": "NEW_EMBODIMENT",
        "modality_config_path": "/work/Isaac-GR00T/examples/SO100/so100_config.py",
        "num_gpus": 1,
        "max_steps": 10_000,
        "global_batch_size": 32,
        "save_steps": 1_000,
    },
)
```

NVIDIA creates the model, processors, dataset loaders, optimizer, distributed execution, and native checkpoints. No Lightning `Policy` is constructed on this route. The modality Python file is executable trusted code, not safe user data; hosted GUI requests must select approved recipes rather than arbitrary Python files.

A shorter everyday call can use the saved `execution="gr00t-local"` profile. Missing profiles/dependencies produce setup guidance; they do not trigger implicit installations.

### 7.3 StarVLA: preserve the original recipe

```python
from physicalai_starvla import StarVLA  # illustrative provider package
from physicalai.workflows import NativeConfig

model = StarVLA(variant="QwenOFT")
dataset = "/data/LEROBOT_LIBERO_DATA"

result = Engine(
    framework="native",
    execution="starvla-local",
    output_dir="./runs/starvla-oft",
).fit(
    model=model,
    dataset=dataset,
    native_config=NativeConfig.file(
        "./recipes/starvla-libero.yaml",
        overrides={"trainer.max_train_steps": 10_000},
    ),
)
```

The StarVLA provider preserves its upstream YAML and supported `accelerate launch` recipe. A recipe can initialize a backbone without a full pretrained policy checkpoint; the provider's `StarVLA(variant=...)` constructor therefore differs from `from_pretrained()`.

StarVLA's inspected guide uses `starVLA/training/train_starvla.py`, `--config_yaml`, framework variants, and named dataset mixtures. These are not interchangeable with GR00T's launcher flags. The adapter must explicitly bind the selected dataset to the recipe's root and mixture fields; it must not discard additional co-training datasets.

A provider should initially approve specific recipes, not claim every script in the repository. StarVLA recommends its stable `starVLA` branch for verified results; the inspected default `starVLA_dev` revision is research evidence, not a proposed production pin.

## 8. Configuration Contract

Separate the PhysicalAI envelope from native configuration:

| Input                                                    | Owner and behavior                                                               |
| -------------------------------------------------------- | -------------------------------------------------------------------------------- |
| Provider, operation, execution profile, output directory | PhysicalAI                                                                       |
| Model/checkpoint and dataset references                  | PhysicalAI identity; provider binds them into the native config                  |
| `trainer`                                                | Lightning-only settings                                                          |
| `data_options`                                           | Selected data integration; provider validates support                            |
| `native_config` mapping                                  | Provider-versioned native option schema                                          |
| `NativeConfig.file(...)`                                 | Explicit upstream file mode, supported only where the upstream loader accepts it |
| `resume_from`                                            | Artifact reference with an explicit supported resume mode                        |

Do not overload a plain string to mean either "mapping YAML to CLI flags" or "pass this native YAML file upstream." Use a typed file form. GR00T's inspected fine-tuning launcher parses `FinetuneConfig` through Tyro; it does not directly consume an arbitrary PhysicalAI YAML workflow.

Resolution rules:

1. Resolve the workflow with Runtime-owned `Config` facilities. Define workflow schemas in Studio, not a second configuration framework or package.
2. Validate the installed provider and supported native schema/recipe version.
3. Load the native recipe with its declared upstream semantics in the target environment when necessary.
4. Apply explicit native overrides to the recipe; reject unknown keys unless a clearly marked experimental mode is supported.
5. Bind model, data, and output references. Reject competing explicit values in native config. Recipe placeholders must be declared as bindings, not silently overwritten values.
6. Resolve relative paths against their source file/project as documented, stage referenced files when needed, and record content hashes.
7. Record the resolved native config, source versions, and launch specification. Redact credentials and sensitive paths from logs; store necessary provenance only in access-controlled artifacts.

Do not convert epochs to steps, reinterpret global/per-device batch sizes, or translate precision modes without a documented provider mapping. Native metrics retain their original names and meaning.

## 9. API, CLI, and GUI Parity

### 9.1 Versioned workflow configuration

Proposed `gr00t-fit.yaml`:

```yaml
workflow_version: 1
operation: fit

engine:
  framework: native
  execution: gr00t-local
  output_dir: ./runs/gr00t

model:
  provider: isaac-groot
  checkpoint: nvidia/GR00T-N1.7-3B
  revision: "<approved-40-character-checkpoint-SHA>"

dataset:
  path: /work/Isaac-GR00T/demo_data/cube_to_bowl_5
  format: gr00t-lerobot-v2

native_config:
  embodiment_tag: NEW_EMBODIMENT
  modality_config_path: /work/Isaac-GR00T/examples/SO100/so100_config.py
  num_gpus: 1
  max_steps: 10000
  global_batch_size: 32
  save_steps: 1000
```

The equivalent first-party `act-fit.yaml` uses the same envelope, but selects Lightning and its own settings:

```yaml
workflow_version: 1
operation: fit

engine:
  framework: lightning
  output_dir: ./runs/act
  trainer:
    max_epochs: 10

model:
  provider: physicalai
  name: act

dataset:
  path: ./datasets/pusht
  format: lerobot-v3

data_options:
  train_batch_size: 8
  num_workers: 4
```

A LeRobot workflow sets `model.provider: lerobot` and a pinned checkpoint, then selects either the Lightning `trainer`/`data_options` form or the native `native_config` form. This is the same choice as the Python examples, not a CLI-only integration.

Execution profiles contain machine-local interpreter/project locations and supported stack identities. They are not model configuration. Resolve and record the chosen profile for each run so that identically named profiles on different machines do not imply identical environments.

### 9.2 Python API from the same specification

```python
from physicalai.engine import Engine
from physicalai.workflows import FitWorkflow

workflow = FitWorkflow.from_file("gr00t-fit.yaml")
engine = Engine.from_config(workflow.engine)

# Optional inspection: no installation, download, or accelerator initialization.
plan = engine.plan(workflow)
plan.checks.raise_for_errors()

# Performs preparation, native execution, and result collection.
result = engine.run(workflow)
```

`Engine.fit(...)` is the ergonomic constructor/dispatcher for the same operation. `run(workflow)` dispatches by the declared operation. A plan can report that a later target-side preparation check is still required; it must not claim full validation from metadata alone.

### 9.3 CLI

```bash
# Existing direct Lightning configuration remains supported.
physicalai fit --config library/configs/physicalai/act.yaml

# Proposed first-party and native workflow paths, hosted by the same CLI.
physicalai fit --config act-fit.yaml
physicalai fit --config gr00t-fit.yaml

# Inspect a redacted launch plan without running training.
physicalai fit --config gr00t-fit.yaml --dry-run

# Override a provider-validated native option.
physicalai fit --config gr00t-fit.yaml --native_config.max_steps 20000

# The other operations use their own versioned workflow specifications.
physicalai benchmark --config benchmark.yaml
physicalai export --config export.yaml
```

The command must match the configuration's operation. Do not let `physicalai fit` execute an export specification.

Legacy configs without `workflow_version` continue through the existing Lightning parser. Detect the workflow discriminator before selecting the full parser; reject mixed legacy/workflow forms and unknown versions. Existing `--trainer.*`, class-path configs, and direct Lightning commands retain their behavior.

### 9.4 GUI/backend

```text
GUI discovers installed/available providers and operation schemas
  -> user selects model, dataset, execution target, and native recipe/settings
  -> backend authorizes references and persists the workflow specification
  -> existing application worker resolves machine-local resources
  -> worker calls Engine.run(workflow)
  -> typed events update progress and results populate the model catalog
```

The GUI renders common fields plus a provider-schema-based advanced form/editor. It need not hardcode a form per model. The backend maps uploaded config/assets to authorized worker-local references; it does not accept arbitrary host paths or executable recipes from an untrusted request.

**Parity means the same serializable specification, validation, behavior, events, and artifacts.** Arbitrary live Python callbacks and model objects remain advanced local API features; the GUI cannot reproduce arbitrary Python code. No hidden GUI-only training implementation is allowed.

## 10. How Providers Delegate

### 10.1 CLI delegation: initial GR00T route

The provider resolves the approved checkpoint revision to a local snapshot and translates the request to NVIDIA's existing launcher. Resolving the checkpoint locally prevents an upstream loader from silently downloading moving Hub HEAD; transitive backbone/tokenizer assets also need approved revisions or a controlled offline cache.

```bash
/work/Isaac-GR00T/.venv/bin/python \
  gr00t/experiment/launch_finetune.py \
  --base-model-path /cache/approved-gr00t-checkpoint \
  --dataset-path /data/prepared-gr00t-dataset \
  --embodiment-tag NEW_EMBODIMENT \
  --modality-config-path /configs/approved-so100.py \
  --num-gpus 1 \
  --max-steps 10000 \
  --global-batch-size 32 \
  --save-steps 1000 \
  --output-dir /runs/experiment-001/native
```

This is the shape of the inspected upstream CLI, not a new PhysicalAI training implementation. Execute a validated argv list without `shell=True`.

Illustrative adapter logic:

```python
class IsaacGrootFitAdapter:
    def plan_fit(self, request: ResolvedFitRequest) -> CommandPlan:
        options = validate_finetune_options(request.native_config)
        reject_reserved_bindings(options)
        return CommandPlan(
            python=request.execution.python,
            cwd=request.execution.project,
            script="gr00t/experiment/launch_finetune.py",
            argv=[
                "--base-model-path", str(request.checkpoint.path),
                "--dataset-path", str(request.dataset.path),
                "--output-dir", str(request.native_output_dir),
                *encode_known_finetune_flags(options),
            ],
            collector="isaac-groot-checkpoints-v1",
        )
```

The real adapter also owns dataset/modality validation, distributed-launch expectations, licensing/access checks, checkpoint verification, and a supported-stack declaration. The common executor owns process lifecycle; the adapter does not invent its own job scheduler.

### 10.2 Native Python API delegation

Prefer a Python entry point when it is stable and exposes useful structured hooks. Invoke it inside the upstream environment, not by importing incompatible libraries into the Engine process.

```python
# Lightweight provider worker running under the upstream interpreter.
request = read_validated_request()
config = build_config_with_upstream_loader(request)
attach_supported_reporting_callbacks(config)
call_upstream_training_api(config)
write_artifact_manifest(collect_native_artifacts(request))
```

For GR00T, `launch_finetune.py` constructs and modifies configuration before calling `gr00t.experiment.experiment.run(config)`. Bypassing the launcher requires reproducing those semantics. Start with CLI delegation rather than copying that preparation into Studio; adopt a direct API only with equivalence evidence or an upstream-supported integration function.

For StarVLA, preserve `accelerate launch`, the approved distributed config, and `--config_yaml` entry point. For LeRobot-native, prefer its documented train entry point and pinned config schema. Do not guess one generic flag translator for all repositories.

### 10.3 Protocol and discovery

Use entry points for installed provider discovery, not for environment solving or arbitrary URL execution:

```toml
# Proposed external distribution, not an existing package/group.
[project.entry-points."physicalai.train.providers"]
isaac-groot = "physicalai_isaac_groot:register"
```

A registration function returns lightweight descriptors and operation factories. Discovery must not import upstream Torch/Transformers, allocate devices, or download weights. Third-party entry-point loading still executes trusted installed code; metadata is not a sandbox.

Split capabilities instead of requiring one giant plugin class:

```python
class FitProvider(Protocol):
    def describe(self) -> FitCapabilities: ...
    def validate(self, request: FitRequest) -> ValidationReport: ...
    def plan_fit(self, request: ResolvedFitRequest) -> ExecutionPlan: ...
    def collect(self, completed: CompletedExecution) -> TrainingResult: ...

class BenchmarkProvider(Protocol):
    def validate(self, request: BenchmarkRequest) -> ValidationReport: ...
    def plan_evaluation(self, request: ResolvedBenchmarkRequest) -> ExecutionPlan: ...
    def collect(self, completed: CompletedExecution) -> BenchmarkResult: ...
```

Policy sessions and exports have separate contracts. Providers implement only supported operations. Descriptors include schema versions, framework routes, native checkpoint formats, tested stacks, and evidence references. Reject duplicate provider IDs and incompatible protocol majors deterministically.

An external worker must not depend on all of `physicalai-train`: that would reinstall the conflicting training stack. Start with versioned JSON request/event/result schemas and a small provider-local bridge. Extract a minimal dependency-free protocol distribution if multiple adapters justify it; do not prematurely create a large second SDK.

### 10.4 Engine dispatch and Lightning adapter

The facade centralizes lifecycle behavior without centralizing framework internals:

```python
class Engine:
    def run(self, workflow: Workflow) -> OperationResult:
        adapter = self.providers.operation_for(
            workflow.operation,
            workflow.provider_identity,
            framework=self.framework if workflow.operation == "fit" else None,
        )
        adapter.validate(workflow).raise_for_errors()
        resolved = self.prepare(workflow, adapter)
        plan = adapter.plan(resolved)
        completed = self.executor.execute(plan, events=self.events)
        return adapter.collect(completed)
```

Here `plan` is the internal operation-dispatch abstraction over `plan_fit` or `plan_evaluation`. Execution can be an approved in-process call or an isolated command; it is not always a subprocess.

The Lightning adapter's in-process operation remains small:

```python
def execute_lightning_fit(request: ResolvedFitRequest) -> CompletedExecution:
    policy = resolve_live_policy_or_reference(request.model)
    datamodule = resolve_live_datamodule_or_reference(request.data)
    trainer = Trainer(**request.resolved_trainer_options)
    trainer.fit(
        model=policy,
        datamodule=datamodule,
        ckpt_path=request.validated_lightning_resume_path,
    )
    return collect_lightning_execution(trainer, policy, request)
```

Collection must respect existing best-checkpoint selection, distinguish last versus best artifacts, and report when checkpointing was disabled. The generic Engine does not assume `Trainer` exists on native routes or execute a nested native trainer inside a Lightning step.

## 11. Execution Environments

### 11.1 Consume existing tools

| Concern                                        | Owner                                            |
| ---------------------------------------------- | ------------------------------------------------ |
| Environment creation and dependency resolution | `uv`, Conda, pip, Docker, or user infrastructure |
| Supported stack and setup instructions         | Provider                                         |
| Environment selection and preflight            | Engine/execution adapter                         |
| Job authorization and scheduling               | Application or external caller                   |
| Native training/inference computation          | Upstream framework                               |

Users follow upstream platform-specific installation instructions. A typical prepared `uv` checkout exposes `.venv/bin/python`; PhysicalAI can invoke that interpreter without activating its environment in the host process.

```bash
# Illustrative launch into an already prepared uv project; no dependency sync.
uv run --project /work/Isaac-GR00T --no-sync python <provider-worker> <request-file>
```

The placeholder worker is supplied by the provider. Some platforms require native-library activation variables; execution profiles must capture those explicitly. For Conda, support a trusted `conda run` launcher where directly invoking Python omits required setup.

Do not implicitly run `uv sync`, upgrade packages, clone repositories, or install plugins during `fit()`. An optional later preparation command can delegate to package managers with explicit consent. Container execution is optional, not the only isolation mechanism.

### 11.2 Versioning and isolation

Record provider version, upstream commit, dirty-tree status/hash, Python and dependency fingerprint, device/driver stack, lockfile or image digest, and resolved execution profile. Check compatible stacks before execution and label nonvalidated stacks experimental rather than silently claiming support.

A changed checkpoint should not rebuild an unchanged environment. Keep environment, weights, and derived dataset caches separate, with identities and access control appropriate to each.

A virtualenv isolates dependencies, not malicious code. Containers also need scoped mounts, credentials, permissions, and network access. Protect the host service from untrusted executable configurations.

### 11.3 Initial scope

Implement trusted existing-interpreter execution first. Allow validated launcher profiles where needed. Defer automatic provisioning, arbitrary remote execution, and universal container recipes. Integrate with the backend's existing local/remote worker placement rather than implementing a second placement layer in the Engine.

## 12. Data and Inference Boundaries

A shared file format does not imply shared semantics. Validate:

- Dataset version, episode boundaries, sampling rate, splits, and instruction fields.
- Camera naming/order, image dtype/range/layout, resolution, and observation history.
- Embodiment, state/action feature order, dimensions, units, and coordinate frames.
- Relative versus absolute actions, rotation representation, and gripper conventions.
- Normalization ownership, statistics source, action horizon, and execution horizon.

For GR00T, support an already-prepared v2 dataset first. A later v3 conversion is an explicit, cached preparation operation that preserves the original snapshot and records converter revision, mapping, statistics, and output fingerprint. Never infer an unknown robot's action semantics from dimensions alone.

Native inference uses a persistent session, not a subprocess per observation:

```python
with engine.open_policy(model=result.model) as session:
    session.reset()
    action_chunk = session.predict(observation)
```

The session contract specifies observation/action schemas, state/reset scope, batching, ordering, ownership of action queues, deadlines, and close behavior. Do not normalize twice or apply action chunking in both the bridge and upstream policy. Remote failures must invalidate stale responses; robot-side safe behavior remains a Runtime/controller concern.

Use the upstream serving protocol behind a provider bridge when available. GR00T has a policy API and ZMQ serving; this does not automatically make it compatible with every benchmark protocol or with Runtime deployment.

## 13. Benchmark Providers and AllenAI

A benchmark provider can either adapt an environment to PhysicalAI's existing rollout loop or delegate the complete upstream evaluation harness. Prefer the latter when task sampling, execution, termination, scoring, and aggregation define the benchmark's authoritative protocol.

A concrete candidate is [AllenAI's vla-evaluation-harness](https://github.com/allenai/vla-evaluation-harness), linked from StarVLA. Its inspected README documents:

- `vla-eval serve --config ...` for model servers.
- `vla-eval run --config ...` for benchmarks.
- Benchmark containers by default, with a local-development mode.
- Model environments using standalone `uv` scripts.
- Existing StarVLA and LeRobot model bridges, including a LeRobot N1.7 path.

This is useful infrastructure to reuse, not functionality to recreate in the Engine. Its inspected matrix distinguishes direct GR00T N1.6 from LeRobot GR00T N1.7; do not claim direct Isaac-GR00T N1.7 compatibility without a bridge and tests.

```python
from physicalai.workflows import BenchmarkRef, NativeConfig

benchmark = BenchmarkRef(
    provider="allenai-vla-eval",
    suite="libero-spatial",
    native_config=NativeConfig.file("./recipes/libero-spatial.yaml"),
)

scores = engine.benchmark(
    model=result.model,
    benchmark=benchmark,
    execution="vla-eval-local",
)
```

Preflight must establish a compatible model-server route, embodiment, observation/action schema, benchmark assets, and supported checkpoints. Two installed plugins do not imply a valid pairing.

```text
Engine benchmark request
  -> AllenAI provider resolves benchmark and compatible model-server config
  -> provider connects to an existing server OR launches one approved server
  -> wait for readiness, then invoke upstream evaluator
  -> retain native recordings/results and build shared result envelope
  -> close only processes/resources owned by this operation
```

Choose one server lifecycle owner: reuse the harness's existing model bridge where suitable, or supply a PhysicalAI policy-session bridge. Do not start duplicate model servers. The harness remains responsible for its benchmark containers and scoring; the Engine records their identities and outcomes.

Retain upstream metric names, units, denominators, aggregation rules, and per-episode evidence. Do not reduce every benchmark to a fabricated `success_rate`. Distinguish failed infrastructure/tasks, missing results, and genuine task failures.

## 14. Results, Lifecycle, and Failure Behavior

A proposed training result includes:

```python
result.status          # succeeded, failed, or canceled
result.model           # optional provider-backed trained policy/artifact handle
result.checkpoints     # typed artifact references with resume capabilities
result.metrics         # native metrics, not universally comparable losses
result.provenance      # resolved workflow and execution identities
result.artifacts       # native reports, processor metadata, recordings, etc.
```

Represent weights-only initialization and full-state resume separately. GR00T and other native bundles can contain several metadata/processor files. Never rename an arbitrary directory to `model.ckpt` and claim Lightning compatibility. Custom live policies without a reconstructible registered artifact contract remain local-only; report that limitation explicitly.

Suggested run layout:

```text
run/
  workflow.json          # resolved PhysicalAI specification
  provenance.json        # immutable identities and environment evidence
  events.jsonl           # versioned structured events
  manifest.json          # status and typed artifact references
  native/                # upstream-owned files, preserved as a bundle
```

Lifecycle:

```text
created -> validating -> preparing -> running -> collecting -> succeeded
                 |          |           |           |
                 +----------+-----------+-----------+-> failed/canceled
```

- Preparation and execution emit typed state events; human logs are not control signals.
- Prefer supported upstream callbacks for metric events. A CLI-only provider can start with coarse progress and raw logs; do not invent a percentage or scrape unstable logs as the sole source of truth.
- A successful exit requires validated expected artifacts; partial files do not imply success.
- Exceptions are the default synchronous API failure signal and carry structured diagnostics/result references. Cancellation is distinct from failure.
- Cancel process groups, including distributed children, with a grace period and bounded escalation. Surface whether a usable checkpoint survived; do not promise save-on-cancel if upstream lacks it.
- Keep diagnostic artifacts on failure; publish completed manifests atomically. Do not delete an existing model directory to start a new run.
- Do not retry training blindly after a lost connection. Remote reattachment belongs to the application job layer and requires persistent operation identity.

The first-party Lightning route can emit richer progress than an initial CLI adapter. Surface the capability difference identically in Python, CLI, and GUI.

## 15. Export and Runtime Contract

Export is optional and independently validated. An upstream ONNX file or TensorRT engine is not automatically a Runtime artifact.

The inspected GR00T tooling exports multiple components and retains some host-side PyTorch operations. Assess reuse of these exports before considering a first-party port, but do not equate their existence with OpenVINO compatibility.

A validated export must include the required model files, metadata, preprocessing/postprocessing and state semantics, numerical/task parity evidence, and a matching Runtime load path. Coordinate any new native/remote inference adapter with Runtime rather than defining a competing inference loader in Studio.

Track at least training/fine-tuning, eager inference, benchmarking, each export backend, XPU workflow support, and deployment support separately. Provider presence is not proof of any of these.

## 16. Compatibility and Migration

1. Add `Engine`, private workflow/provenance references, versioned workflows, and provider discovery without replacing existing classes.
2. Keep `physicalai.train.Trainer`, all `Policy`/`Model` classes, datamodules, `get_policy()`, and existing CLI configs working.
3. Implement the Lightning route by composition, not by changing `Trainer` into a remote facade.
4. Add the native provider route behind the new workflow discriminator.
5. Refactor reusable execution logic out of backend-specific orchestration incrementally. Preserve job identity, scheduling, authorization, and persistence in the application.
6. Extend backend artifact models/import/storage to accept native bundles and optional exports; a library plugin alone does not update these assumptions.
7. Extend CLI benchmark loading/cleanup to provider-owned sessions and resources rather than assuming `benchmark.gyms`.
8. Generate GUI/API schemas from library operation/provider contracts through the backend's established API-generation process.

Avoid a flag-day migration. Lightning remains an equally supported direct API, even though it sits below the Engine in the high-level architecture.

## 17. Validation and Rollout Plan

### Phase 1: Prove the orchestration boundary

- Introduce private workflow/provenance schemas, result schemas, and a thin Engine.
- Implement first-party Lightning delegation and preserve direct API behavior.
- Add an isolated fake provider for deterministic CPU tests of planning, events, failures, and artifacts.
- Add one GR00T-native CLI integration on an approved CUDA stack and prepared dataset.
- Validate upstream baseline versus delegated invocation; do not claim task quality from a one-step smoke test.

### Phase 2: Prove reuse rather than a GR00T abstraction

- Add LeRobot-native alongside the existing Lightning wrapper.
- Exercise a config-file/launcher integration with one StarVLA recipe before freezing the plugin contract.
- Validate settings rejection, checkpoint loading, cancellation, and declared resume semantics.
- Add schema-driven CLI/GUI integration and native artifact handling.

### Phase 3: Benchmark and deployment capabilities

- Integrate one compatible AllenAI harness model/benchmark pair.
- Reuse upstream protocol/server implementations and preserve benchmark scoring.
- Assess upstream export reuse and validate specific Runtime/OpenVINO paths separately.
- Consider managed setup helpers or containers only after existing-environment UX is proven.

### Acceptance tests

| Area                   | Required evidence                                                                                                       |
| ---------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| Backward compatibility | Existing direct Trainer calls, Lightning configs, policy factories, and checkpoint loading retain behavior              |
| Surface parity         | API/CLI/backend worker produce equivalent resolved plans and result schemas from the same versioned specification       |
| Discovery              | No heavy imports/downloads/device allocation; duplicate IDs and incompatible versions fail clearly                      |
| Isolation              | Conflicting upstream dependency graphs execute in different interpreters without modifying the host environment         |
| Native fidelity        | Pinned config, dataset mapping, launcher, preprocessing, and checkpoint reload agree with upstream baseline             |
| Configuration          | Unknown options, conflicting bindings, path resolution, and unsupported routes fail before execution                    |
| Data semantics         | Tests cover feature order, units, normalization, relative actions, histories, and chunk ownership                       |
| Lifecycle              | Cancellation cleans up distributed children; failures and missing artifacts cannot become successful jobs               |
| Resume                 | Weights-only and full-state paths are distinct and tested only where supported                                          |
| Benchmark              | Native task/episode protocol and aggregation remain intact; infrastructure failures are not scored as robot failures    |
| Export                 | Runtime round trip and numerical/task parity for each claimed backend                                                   |
| Security               | Trusted plugin/recipe boundaries, authorized paths, redacted diagnostics, bounded messages, and pinned asset provenance |

## 18. Security and Operational Constraints

Follow `library/docs/development/security.md` and Runtime's shared contracts:

- Execute approved provider code and registered recipes, not arbitrary imports/commands from HTTP or YAML.
- Keep dynamically selectable class paths allowlisted; plugin registration is a separate explicit trust decision.
- Use argv lists, validated paths, scoped mounts, and credentials injected through approved secret channels.
- Treat native Python configs as executable code. Do not let an untrusted uploaded config bypass this boundary.
- Use versioned JSON for cross-process control; prohibit untrusted pickle/object deserialization and bound payload sizes.
- Validate manifests and artifact containment before publishing them; never follow untrusted artifact paths outside the allowed run root.
- Pin model and transitive asset revisions and use allowlisted downloads. Do not silently enable remote model code.
- Do not log tokens, environment dumps, or checkpoint paths. Keep necessary path-bearing provenance restricted and redact user-facing plans/errors.
- Do not expose upstream policy servers publicly by default. Authenticate/secure remote transport and validate request size and schema.
- Record licenses, gated-model access requirements, and approved exceptions for unsupported stacks.

## 19. Open Decisions

- Final namespace: `physicalai.engine.Engine` and `physicalai.workflows` are proposed, not committed.
- Default route: start explicit; any later default must be deterministic, documented, and never a failure fallback.
- Whether richer typed provider options merit provider-specific Python helper classes in addition to JSON schemas.
- The minimum worker-protocol packaging needed without importing the full training distribution upstream.
- How to persist execution profiles across library-only and application-managed installations without duplicating Runtime configuration ownership.
- Which StarVLA recipe and AllenAI model/benchmark pair become the second compatibility tests.
- Which GR00T capabilities receive a support commitment after the initial native CUDA baseline.

## 20. References and Evidence Boundaries

- [Model Enablement Guidelines (draft)](https://github.com/open-edge-platform/physical-ai-studio/blob/docs/design-docs/library/docs/design/model-guidelines/model-implementation-guidelines.md).
- [Model Enablement Reference (draft)](https://github.com/open-edge-platform/physical-ai-studio/blob/docs/design-docs/library/docs/design/model-guidelines/model-implementation-guidelines-reference.md).
- [Existing Trainer](trainer/README.md) and [LeRobot wrapper](policy/lerobot.md).
- [Isaac-GR00T inspected revision](https://github.com/NVIDIA/Isaac-GR00T/tree/51d4c89f72fda44cbf77285c6a8114b52676b8a1): fine-tuning launcher, policy API/server, dependency pins, and deployment tooling.
- [StarVLA inspected guide](https://github.com/starVLA/starVLA/blob/2f17402a5ccaa09907516ae5e542b0fa6ee5d155/docs/starVLA_guideline.md): native YAML, model variants, dataset mixtures, and Accelerate launcher. This is a development-branch observation, not validation of a supported stack.
- [AllenAI harness inspected revision](https://github.com/allenai/vla-evaluation-harness/tree/4aeb4369640e8019d46af9534ce9b957e486ad38): native serve/run commands, benchmark isolation, and model bridge matrix. Upstream reproduction claims do not transfer automatically to a PhysicalAI plugin.
- [Current LeRobot GR00T documentation](https://github.com/huggingface/lerobot/blob/main/docs/source/groot.mdx): N1.7 availability is not proof of Studio wrapper validation.
- [Python packaging entry points](https://packaging.python.org/en/latest/specifications/entry-points/): discovery mechanism, not environment management.
- [Runtime inference runners](https://github.com/openvinotoolkit/physicalai/tree/main/src/physicalai/inference/runners): existing meaning of Runner that this design avoids.
