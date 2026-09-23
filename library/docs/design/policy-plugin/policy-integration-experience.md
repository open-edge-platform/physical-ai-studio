# Policy integration: proposed user experience

**Status:** Design interview complete through round 7. Decisions are recorded in [ADR 0001](0001-library-first-policy-orchestration.md) through [ADR 0004](0004-official-checkpoints-and-runtime-inference.md); the remaining items are designed during the first proof. New classes, commands, and recipe identifiers below are illustrative and do not exist yet.

The agreed foundations are in [ADR 0001](0001-library-first-policy-orchestration.md) and [ADR 0002](0002-upstream-first-policy-integration.md). Source evidence is in the [research note](policy-integration-research.md). This proposal does not authorize implementation or changes to the GR00T issue.

## Recommendation

One Studio installation should orchestrate several compatible Python environments. Existing direct Lightning code stays in-process and unchanged. Users selecting managed workflows choose a policy and operation, not a Python interpreter; advanced users can attach an existing environment.

The agreed parameter selecting a Python environment is `python_env`. Avoid `execution`: Runtime already defines `Execution` strategies, and `execution_horizon` is an action-chunking term.

This separates three experiences rather than making one object behave unpredictably:

| Experience | Intended user | Behavior |
| --- | --- | --- |
| Direct Lightning | Researcher customizing training | Existing Policy/Trainer objects, callbacks, debugging, and live tensors in the caller's environment |
| Managed workflow | CLI/GUI/notebook user running a reproducible recipe | Serializable request; execution in an explicitly resolved environment; structured events and artifact results |
| Existing-environment workflow | Contributor or infrastructure owner | Same request/result interface, but execution uses an explicitly attached upstream checkout/interpreter or approved launcher |

The same first-party implementation can participate in the first two experiences. This does not promise two incompatible Transformers versions can coexist in a single interpreter.

## 1. A small optional facade, not another Trainer

### Preserve direct usage

The existing style remains valid when its dependencies are installed:

```python
from physicalai.data import LeRobotDataModule
from physicalai.policies import ACT
from physicalai.train import Trainer

policy = ACT()
data = LeRobotDataModule(repo_id="lerobot/pusht", train_batch_size=8)
Trainer(max_epochs=10).fit(policy, datamodule=data)
```

Do not silently turn these calls into process launches or make the policy a remote proxy. Live callbacks and model mutations retain their existing meaning.

### Programmatic inputs use the official API: agreed

The user requires model, dataset, and configuration inputs rather than a YAML-only facade, and expects an upstream workflow to use the upstream framework's own API. Agreed rule, recorded in [ADR 0003](0003-upstream-apis-as-source-of-truth.md):

> Always use the official class. Pass a live instance when that class is importable in the caller's Python environment; otherwise pass a reference to the same class.

Do not create Studio mirror classes such as `GrootDataset` or `GrootFitConfig`. They would duplicate upstream schemas, drift with upstream releases, and increase contributor work. The previous round's mirror-class example is withdrawn.

GR00T's official `FinetuneConfig` is a plain dataclass whose module imports only standard-library modules. That makes it straightforward to describe as data and reconstruct faithfully in the GR00T Python environment. The import is lightweight, but installing the `gr00t` distribution still installs its pinned dependency set. [U7]

Where the official package is importable:

```python
from gr00t.configs.finetune_config import FinetuneConfig  # official GR00T API
from physicalai.engine import Engine                      # proposed

config = FinetuneConfig(
    base_model_path="nvidia/GR00T-N1.7-3B",
    dataset_path="/datasets/so101-gr00t",
    embodiment_tag="NEW_EMBODIMENT",
    max_steps=10_000,
    global_batch_size=32,
)
result = Engine().fit(config)
```

Where it is not importable—for example, in Studio's Python environment—use the same official class by reference. Runtime's existing `Config` already represents objects as `class_path` plus `init_args` without instantiating them:

```python
from physicalai.config import Config
from physicalai.engine import Engine  # proposed

config = Config(
    "gr00t.configs.finetune_config.FinetuneConfig",
    {
        "base_model_path": "nvidia/GR00T-N1.7-3B",
        "embodiment_tag": "NEW_EMBODIMENT",
        "max_steps": 10_000,
        "global_batch_size": 32,
    },
)
result = Engine().fit(config, dataset=studio_dataset)
```

The plugin constructs the official object inside the GR00T Python environment, where the official constructor validates it. Studio adds only a small envelope, such as `model`, `dataset`, `output_dir`, and `python_env`. Envelope values bind to their official fields. Supplying both an envelope value and the corresponding official field, such as `dataset` and `dataset_path`, is an error rather than an override.

The official class path identifies the plugin. This matters because GR00T N1.7 has an official NVIDIA training API and a LeRobot integration. The namespace distinguishes those lifecycles without an extra provider flag and forms the allowlist for classes a worker may instantiate. [U1, U9, library security rules]

A plugin is therefore a thin adapter, not a wrapper class hierarchy. It declares the official classes and operations it handles, binds the Studio envelope to official fields, invokes the official entry point, translates events and artifacts, and supplies dataset converters.

The inspected GR00T fine-tuning launcher builds its complete training configuration inside the module's `__main__` block. There is no official function that accepts `FinetuneConfig` and starts training. Contribute that small refactor upstream; until it is available, the plugin can invoke the official launcher or carry a tracked downstream patch. [U2]

Passing a live official object from a notebook requires the official package in that notebook's Python environment. Running Engine inside the GR00T Python environment would require an orchestration package installable there. Current `physicalai-train` requires `peft>=0.20.0`, while inspected GR00T pins `peft==0.17.1`. [S12, U3] Packaging that lightweight option follows the Q15 decision.

For first-party policies, Studio's classes are the official API: pass a live `ACT()` in-process or `Config("physicalai.policies.ACT", ...)` for isolated execution.

For the in-process Lightning route, accept existing objects without reconstructing them from configuration:

```python
engine = Engine(trainer=Trainer(max_epochs=10))
result = engine.fit(policy, datamodule=data)
```

The `trainer` input applies only to this route. Conflicting output/checkpoint ownership must be rejected rather than silently overridden. Direct `Trainer.fit()` remains equally supported and does not require the facade.

A live object cannot move between Python environments with different dependency stacks. Do not pickle arbitrary models, dataloaders, GPU state, or callbacks into another interpreter. Do not silently discard in-memory modifications by reloading the original checkpoint. Use an explicit artifact-save/import path or run the operation in a compatible process.

### YAML is another input route

```python
engine = Engine()
engine.prepare("gr00t-fit.yaml")  # explicit setup for a reviewed workflow
result = engine.run("gr00t-fit.yaml")
```

The serializable Python inputs, YAML workflow, and GUI request resolve through the same validation and dispatch. In YAML, official objects use the same `class_path` and `init_args` form. Programmatic callers do not need to create a temporary YAML file. An adapter can still materialize a native configuration file when its upstream launcher requires one.

Arbitrary live Python objects/callbacks remain local features; they are not promised to round-trip through YAML or the GUI. Preparation must also accept programmatic inputs, with its exact method shape to be designed. `fit()` and `run()` use prepared environments; neither silently installs or upgrades packages.

The facade hides provider discovery, environment resolution, request validation, execution, event handling, and artifact collection. It must earn its interface through these shared responsibilities, not merely rename an upstream call. A non-mutating plan can explain preparation and execution; target-side checks still run before execution.

Operation results refer to artifacts, metrics, and provenance, not necessarily a resident Torch model. Export, benchmarking, and inference remain independently supported operations rather than automatic training side effects. The concrete result interface follows the object-semantics decision.

### Alternative considered

Provider-specific functions with shared internal execution utilities were considered. The user accepted the common facade on the condition that it has a proper programmatic interface in addition to file-driven execution. Do not postpone that interface as optional future ergonomics, and do not freeze the illustrative class/method names before the contracts are agreed.

## 2. One workflow across Python, CLI, and GUI

A workflow records the plugin, operation, selected recipe, immutable model identity, data input, output location, and official configuration. Machine-specific Python environment selections remain separate from portable model/data configuration.

- Existing Lightning configuration files and `physicalai fit --config ...` retain their behavior.
- New workflows have an explicit version discriminator, so parsing never depends on guessing from an incidental key.
- A new preparation operation can be exposed through the Runtime-hosted CLI; its exact command name requires coordination with Runtime.
- The GUI persists the same workflow specification and calls the library through the backend worker. Authorization, job scheduling, storage, and persistence remain application responsibilities.
- The backend maps user-visible dataset/model identifiers to authorized worker resources. It does not accept arbitrary filesystem paths or executable native configurations from untrusted requests.

Official settings keep their upstream names and semantics. A common envelope must not equate epochs with steps, global with per-device batch size, or weights-only initialization with full-state resume. Library workflow configuration consumes Runtime-owned Config facilities rather than introducing another configuration package.

## 3. Friendly setup without a second package manager

The user accepted two explicit setup paths. Prepared-environment-only execution remains a possible internal engineering slice, not the complete user-facing setup experience:

### Default: prepare a curated environment

```text
Select policy and supported workflow
    -> inspect compatibility and setup requirements
    -> choose Set up once
    -> approve downloads, licenses, and reviewed setup actions
    -> create/verify the environment with existing tools
    -> Ready to run
```

Studio should delegate to a provider's pinned setup recipe using existing tooling such as uv or containers. It should not implement its own dependency solver or maintain a speculative universal dependency set.

Only reviewed recipes for documented platforms receive managed setup. Do not claim arbitrary upstream installation can be automated, silently install host drivers/system packages, or execute an arbitrary GitHub URL. Setup reports unsupported platforms and manual prerequisites clearly.

The CLI/GUI can preview the setup and request confirmation. The Python library has explicit preparation calls and never waits for an interactive prompt in a worker. Credentials and model-access approvals are handled explicitly; they are not embedded in the recipe or logged.

### Advanced: attach an existing environment

A contributor can select an upstream checkout and interpreter, or an approved container/launcher profile. Studio probes its identity and compatibility without modifying it. Validated profiles and experimental local modifications are visibly different.

### Setup lifecycle

- Cache environments by compatible recipe/software/platform identity, not by model display name or checkpoint filename.
- Share an environment only when the resolved stack is compatible. Keep model and dataset caches separate.
- Publish a prepared environment only after setup/preflight succeeds; interrupted setup must not look ready.
- Create a new environment identity for an upgrade; do not mutate an environment used by an active job.
- Record the resolved environment and upstream/patch identities with each run.
- Distinguish ready-to-run from validated capability. A successful import or installation does not establish policy quality.

Curated setup plus attach-existing support is agreed for the user-facing experience, starting with a bounded supported recipe rather than universal installation automation. Exact recipe identity, environment selection, and implementation sequencing remain to be specified.

## 4. First-party isolation without a flag-day packaging migration

### User-facing behavior

| Call | Execution |
| --- | --- |
| Direct `Trainer.fit(policy, datamodule)` | Existing interpreter; caller installs a compatible stack |
| Workflow selecting first-party policy A | Resolved environment A, running the existing Policy and Trainer |
| Workflow selecting first-party policy B | Resolved environment B when requirements differ, using the same core Lightning interface |
| Workflow selecting GR00T/Xiaomi | Upstream-compatible environment, preserving its own training lifecycle |

Choosing isolated execution is explicit in the workflow's Python environment selection; no route switches silently after an import failure. A displayed default can keep everyday usage simple, but its resolved identity is recorded and ambiguity is an error.

### Share environments, not necessarily worker processes

An environment is a reusable software stack, not a model instance. Multiple compatible policies can use the same prepared environment, sequentially or from separate workers. Sharing an environment does not grant concurrent access to a GPU or require the models to share Python globals; resource scheduling remains with the caller/application.

The host launches the selected interpreter/launcher. Users do not activate and deactivate environments between workflow calls, and Studio does not change a running notebook's interpreter or `sys.path`.

Compatibility includes the complete resolved stack: Python, Torch/Transformers and other packages, upstream/bridge versions, native libraries, and platform/backend constraints. Similar version ranges alone are not sufficient. A shared Python environment must satisfy every participating workflow and its declared validation scope; discovery does not grant a support claim.

Agreed selection rule: an explicit Python environment wins; otherwise use the plugin's configured default and reuse its compatible prepared environment. If several candidates remain without a designated default, require selection rather than guessing. Never merge requirements into or upgrade an active environment automatically. Preparing a new shared stack is an explicit setup operation, followed by validation for the participating workflows.

### Incremental packaging approach

1. Preserve existing public import paths, direct training behavior, and currently supported installation/configuration workflows.
2. Make policy-family imports lazy so selecting one family does not import incompatible siblings. The current policy package imports all first-party families, and individual families import Transformers at module scope.
3. Put genuinely policy-specific requirements in selectable dependency sets. Use separate, validated environment recipes for incompatible sets; do not widen version bounds without testing.
4. For uv-based development, declare genuinely conflicting extras rather than forcing them into one resolution. uv supports separate resolution of explicit conflicts, but does not install incompatible versions together.
5. Do not add incompatible new families to an existing all-policies extra and pretend they can coexist. Preserve its currently supported compatible behavior; any change to install semantics needs an explicit compatibility plan.
6. If unavoidable mandatory dependencies prevent isolated installation, narrow those dependencies or split the affected implementation distribution deliberately. Do not split every existing policy package before demonstrating that need.

A policy-specific environment is useful only if the distribution installed there has satisfiable metadata. A subprocess alone is not the fix. Existing imports/checkpoints/configuration and supported install paths need regression tests through any packaging change.

Sources: [current policy factory](../../../../library/src/physicalai/policies/__init__.py), [Pi0.5 imports](../../../../library/src/physicalai/policies/pi05/model.py), [library dependency declarations](../../../../library/pyproject.toml), and [uv conflicting dependencies](https://docs.astral.sh/uv/concepts/resolution/#conflicting-dependencies).

## 5. Small contributor contract

The user agreed that a plugin can expose one supported operation initially, such as inference or fine-tuning. Optional operations should be absent from capability discovery, not methods that exist but fail with `NotImplementedError` after a job starts.

A contributor supplies repository-specific knowledge:

- Lightweight identity, capability, and configuration descriptions.
- A supported upstream/environment recipe, or instructions/profile for an existing environment.
- An operation adapter that invokes the approved upstream interface and interprets its artifacts.
- Real-workflow acceptance evidence for the capability claimed.

Shared library infrastructure provides process lifecycle, event transport, setup orchestration, discovery, and result validation. Contributors should not write custom React, modify the Studio factory, implement a package manager, or reconstruct another trainer merely to add a policy.

A training-only plugin can use upstream inference to validate that its saved artifact reloads without exposing a separate Studio inference capability. An inference-only plugin need not invent training support. Artifact interchange with export/benchmark plugins still requires explicit compatibility validation.

Agreed listing rules: anyone may install a plugin manually. Studio's GUI lists only curated policy plugins: reviewed plugins with a named maintainer, a pinned upstream revision, and evidence for each claimed capability. Capabilities without evidence are labeled experimental. Studio-maintained and community-maintained plugins are distinguished, and a listed plugin that stops meeting these requirements is marked unsupported or removed.

## 6. Physical validation without blocking every contribution on robot access

The agreed evidence levels are independent, not interchangeable:

1. **Real software workflow:** train, save, reload, infer; verify stateful behavior and resume where claimed, on the supported compute stack.
2. **Simulation/task evaluation:** measure the policy's closed-loop behavior for the stated task protocol rather than just tensor shapes or training loss.
3. **Physical deployment:** test the actual robot/controller, observation/action mapping, timing, reset, and failure behavior for the named deployment scenario, with appropriate supervision and safety controls.

Require physical evidence before a real-robot deployment claim. Do not require robot testing for every training-only contribution, or imply a successful test on one robot validates every embodiment. Conversely, simulation-only evidence must never be labeled physical deployment validation.

Hardware access is an explicit test resource. A project-owned or contributor-supplied reproducible hardware report can be reviewed for its supported scenario. Repeat relevant validation when deployment behavior changes. The recurring schedule and scenario matrix are deferred; the team's current robots are SO-101 and the Seeed Studio reBot B601-RS. This does not require unrelated training-only changes or every contributor PR to have robot access.

## 7. Dataset conversion: agreed

Q15 decides which types describe the work. Q16 concerns transforming the data itself between Studio's dataset contract and an official dataset contract. Conversion is needed even when the official API is used.

The user agreed to plugin-supplied converters as library code, so Python, CLI, and GUI share one implementation. The existing backend dataset-import flow detects formats, drafts a manifest, requests review, and commits. It currently registers only a LeRobot v3 adapter and rejects LeRobot v2 archives, the format family GR00T extends. The import flow should call library converters rather than reimplementing upstream formats. [S16]

- **Studio to official:** first priority, so upstream policies can train on Studio recordings.
- **Official to Studio:** support where the mapping is well defined, so official datasets can be viewed, benchmarked, or used to train first-party policies.
- **Reuse official tooling where practical:** GR00T ships its own LeRobot v3-to-v2 conversion script, but its Python constraint prevents direct reuse; see below. Studio's recorded camera and joint names can supply much of GR00T's modality metadata; the plugin must still validate every required semantic field. [U8, S16]

Not every conversion is mechanical or lossless:

- Xiaomi requires three specific camera views and complete bimanual arm, waist, and base arrays. A single-arm dataset cannot satisfy that contract without inventing data. [X6]
- Converting joint actions to relative or end-effector actions can require kinematics, not reshaping.
- GR00T's default video path supports H.264 but does not guarantee AV1 decoding. Normalization statistics must be recomputed for converted data. [U1]

Use exact mappings automatically where they are defined. Require an explicit, reviewed mapping when a semantic choice is necessary. Reject an impossible mapping with a clear diagnostic. Preserve the original dataset, cache the derived dataset with conversion provenance, and do not promise lossless round trips.

A shared dataset source does not mean a universal dataloader. An arbitrary live loader, custom transform, or sampler cannot silently cross into a worker or be ignored. Preserve its behavior through a supported explicit adapter or report the incompatibility before execution.

### Where conversion code lives: agreed

The user expects LeRobot-to-GR00T conversion in the library, alongside `Dataset`, `LeRobotDataModule`, and `FormatConverter`, because Studio stores recordings as LeRobot datasets.

`FormatConverter` converts individual batches between Studio's `Observation` and LeRobot dictionaries in memory. GR00T conversion writes a new dataset on disk in another project's contract: files, metadata, videos, and statistics. Treat it as dataset conversion rather than another `DataFormat` value or a `FormatConverter` mode.

GR00T's official v3-to-v2 converter requires Python `>=3.10,<3.12` and pins LeRobot to a Git commit. Studio's library requires Python 3.12 or later, and GR00T's training environment requires Python 3.12. Reusing that script would therefore require a third Python environment. [U8, U12, S12, U3]

Agreed: library-native conversion running in Studio's Python environment. Studio can already read LeRobot v3 datasets; the converter writes GR00T's documented format and `modality.json`. Conformance tests load the output with GR00T's official loader in the GR00T Python environment, so GR00T's contract remains the source of truth. Offer upstream a converter compatible with current Python.

Put the general conversion interface in core `physicalai.data`. Put GR00T-specific conversion in the GR00T plugin's lightweight part, which installs in Studio's environment without GR00T's dependencies. The plugin's worker part runs in the GR00T Python environment.

## 8. Results and inference: agreed

The backend currently expects a trained model directory to contain `model.ckpt` and an `exports/` directory. [S9] Agreed: training returns status, the official checkpoint in its original format, metrics under their upstream names, and provenance: plugin, upstream revision, downstream patches, Python environment, and dataset conversion. Studio's model list shows the plugin and each validated capability.

Runtime already discovers inference adapters from installed packages; an adapter implements `load(model_path)` and `predict(inputs)`. [R4] Agreed: a plugin registers a Runtime adapter delegating to the upstream inference API in its Python environment, such as GR00T's policy server. Benchmarks and robot control then use Runtime's existing `InferenceModel` and `PolicySource`. Export to OpenVINO or ONNX remains a separately validated capability. This requires Runtime coordination, a single owner for normalization and action queues, and fail-safe behavior for timeouts or stale responses. See [ADR 0004](0004-official-checkpoints-and-runtime-inference.md).

## 9. First proof: agreed

1. Convert a Studio LeRobot dataset to GR00T's format with the library converter, starting from NVIDIA's SO-100 or reBot B601 DM example as the baseline.
2. Set up GR00T's Python environment.
3. Fine-tune GR00T N1.7 through the GR00T plugin.
4. Reload the official checkpoint.
5. Run inference.

In the same proof, run one first-party policy, such as SmolVLA, through `Engine` in its own Python environment. Physical robot testing follows later.

## Deferred to the first proof

- Export and quantization details.
- A lightweight orchestration package usable inside upstream Python environments.
- Specific changes to propose upstream, such as a programmatic GR00T fine-tuning function.
- Concrete SO-101 and reBot B601-RS dataset mappings.
- Where Studio-maintained plugins live.
- Physical-test cadence and the robot/scenario matrix.

## Sources

- [S9 — Studio training job](https://github.com/open-edge-platform/physical-ai-studio/blob/main/application/backend/src/training/job.py)
- [S12 — Studio library dependencies](../../../../library/pyproject.toml)
- [S16 — Studio dataset import flow](../../../../application/backend/src/services/dataset_import/README.md) and [registered adapters](../../../../application/backend/src/services/dataset_import/adapters/__init__.py)
- [U1 — GR00T README](https://github.com/NVIDIA/Isaac-GR00T/blob/51d4c89f72fda44cbf77285c6a8114b52676b8a1/README.md)
- [U2 — GR00T fine-tuning launcher](https://github.com/NVIDIA/Isaac-GR00T/blob/51d4c89f72fda44cbf77285c6a8114b52676b8a1/gr00t/experiment/launch_finetune.py)
- [U3 — GR00T dependencies](https://github.com/NVIDIA/Isaac-GR00T/blob/51d4c89f72fda44cbf77285c6a8114b52676b8a1/pyproject.toml)
- [U7 — GR00T FinetuneConfig](https://github.com/NVIDIA/Isaac-GR00T/blob/51d4c89f72fda44cbf77285c6a8114b52676b8a1/gr00t/configs/finetune_config.py)
- [U8 — GR00T LeRobot conversion script](https://github.com/NVIDIA/Isaac-GR00T/blob/51d4c89f72fda44cbf77285c6a8114b52676b8a1/scripts/lerobot_conversion/convert_v3_to_v2.py)
- [U9 — LeRobot GR00T N1.7 integration](https://github.com/huggingface/lerobot/blob/9a6bb61043bac8c14353fcb6ea513b7473c118e3/src/lerobot/policies/groot/groot_n1_7.py)
- [U12 — GR00T converter dependencies](https://github.com/NVIDIA/Isaac-GR00T/blob/51d4c89f72fda44cbf77285c6a8114b52676b8a1/scripts/lerobot_conversion/pyproject.toml)
- [R4 — Runtime inference adapter interface](https://github.com/openvinotoolkit/physicalai/blob/2fafa820bb81798da92c9ed84f23d8542bc2f3c0/src/physicalai/inference/adapters/base.py)
- [X6 — Xiaomi data format](https://github.com/XiaomiRobotics/Xiaomi-Robotics-1/blob/0dd7aef8dc87296246aae812a1f59ccb708e5546/xr1/docs/data_format.md)
- [Library security rules](../../../../library/docs/development/security.md)
