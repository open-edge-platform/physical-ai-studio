# Policy integration research

**Status:** Research and design record for a completed design interview. Decisions are recorded in [ADR 0001](0001-library-first-policy-orchestration.md) through [ADR 0004](0004-official-checkpoints-and-runtime-inference.md); remaining details are designed during the first proof.

**Inspected:** 2026-09-22. Source inspection only: no upstream installation, model download, training, export, or benchmark execution. CI configuration was inspected, not recent CI results.

This research led to the decisions recorded in this directory. It does not change [GR00T issue #1197](https://github.com/open-edge-platform/physical-ai-studio/issues/1197).

## Recommendation in brief

Standardize workflow requests, lifecycle reporting, and artifact contracts, not every model's training loop. Preserve the existing Lightning workflow. For independent upstream repositories, prefer a provider that delegates to the upstream workflow in a compatible Python environment. Add deeper Lightning adaptation or first-party model code when a measured product requirement justifies it.

The user agreed to independent capability validation, library-first orchestration, separate environments, an upstream hardware baseline, and preservation of the core Lightning pipeline. They subsequently chose upstream-first integration, first-party code only when required, external contributor packages, temporary downstream patches, and real end-to-end validation. They also accepted a common facade with a proper Python interface, compatible Python environments shared across policies, curated setup, the official-API rule, dataset converters, preflight checks, official checkpoints, and inference through Runtime adapters. Native delegation reduces translation of upstream behavior; it does not eliminate integration testing, data adaptation, deployment work, or maintenance.

## Evidence snapshot

| Repository | Inspected revision |
| --- | --- |
| Physical AI Studio | `2c1b5777feba818b0205152bae43ae19f58c88dd` |
| NVIDIA Isaac-GR00T | `51d4c89f72fda44cbf77285c6a8114b52676b8a1` |
| Xiaomi-Robotics-1 | `0dd7aef8dc87296246aae812a1f59ccb708e5546` |
| Physical AI Runtime, local checkout | `2fafa820bb81798da92c9ed84f23d8542bc2f3c0` |

Upstream links below are pinned to these revisions. Dependency requirements describe these snapshots, not permanent requirements for a policy family.

## 1. The alternatives are not mutually exclusive

Separate four decisions:

| Decision | Choices |
| --- | --- |
| Model implementation ownership | Studio-owned implementation or upstream implementation |
| Training lifecycle ownership | Studio's Lightning workflow or upstream's workflow |
| Packaging and discovery | Built-in adapter or separately installed plugin |
| Execution environment | Compatible host process or separate interpreter/container |

A plugin can deliver a Lightning adapter, an upstream-native workflow, or a first-party implementation. Moving an adapter into a plugin package does not itself resolve dependency conflicts or preserve an upstream training recipe. Python entry points advertise installed implementations and load their Python objects; they do not provide dependency isolation or a security sandbox. [P1]

Proposed vocabulary for the interview:

- **Policy integration:** the supported connection between a policy implementation and Studio workflows.
- **Provider:** the adapter that owns repository/framework-specific integration knowledge. It is not a model architecture or a support guarantee.
- **Lightning adapter:** an integration that retains upstream model code but gives Studio's Lightning workflow ownership of training.
- **Upstream-native route:** an integration that retains the upstream training lifecycle, even when upstream itself uses Lightning.
- **Capability:** a specific operation, format, or hardware workflow that an integration exposes; validation is a separate evidence claim.

The agreed distinction between implementation ownership, upstream-native routes, external policy plugins, and validated capabilities is recorded in the [project glossary](../../../../CONTEXT.md). Concrete provider class names and public method names remain proposed. Avoid using “native” alone: it currently risks meaning both Studio-owned code and the original upstream workflow.

## 2. What Studio implements today

### Lightning is a real implementation constraint

`Policy` inherits `LightningModule` and includes inference/action-queue and evaluation behavior. `Trainer` inherits Lightning's trainer and injects Studio callbacks. `get_policy()` returns a `Policy` and dispatches only the first-party and LeRobot sources. An upstream process handle cannot replace that return value without changing its interface. [S1–S3]

The application already separates local, remote, and SSH execution. This is a useful placement/transport seam, not a framework-neutral model integration. Its shared training context describes a checkpoint, logs, and exports; `run_training_job()` constructs a LeRobot datamodule and Lightning trainer, publishes the checkpoint, then attempts export. Supporting native checkpoint bundles therefore requires changes beyond registering a policy name. [S7–S9]

### LeRobot has tests, but discovery is broader than demonstrated support

The wrapper is substantial: it owns configuration reconstruction, feature extraction, normalization processors, optimizer construction, Lightning checkpoint handling, and inference delegation. It is not just a forwarding call. [S4]

The package exposes eight named wrappers, but its declared `VALIDATED_EQUIVALENCE_POLICIES` contains ACT, Diffusion, and SmolVLA. The equivalence suite marks GR00T, Pi0, Pi0.5, Pi0 Fast, and XVLA as expected failures for documented reasons; large VLAs can also skip without an accelerator. The checked-in GR00T reason mentions Eagle/Flash Attention and must not be assumed to describe a newly integrated N1.7 stack. [S5–S6]

The equivalence test compares Lightning execution with a hand-written replay loop over the captured batches, using shared model/configuration/processor inputs. This is useful adapter-level loss/gradient/weight evidence, not proof of equivalence to the complete upstream training command, scheduler, distributed recipe, or resume lifecycle. The CI workflow invokes non-slow integration tests on a GPU-labeled runner; its existence does not establish that every case runs and passes. [S6, S10]

**Design implication:** distinguish discoverable, runnable, and validated. A registry entry, import, or skipped/expected-failure test must not become a supported-workflow claim.

### Reuse the robot-plugin pattern selectively

Robot plugins contribute stable identifiers, typed payloads, builders, and optional capabilities through entry-point registration. Studio provides persistence, schema-driven forms, and orchestration. The existing SDK is specifically a robot-catalog SDK. [S11, R1]

Reuse the discovery, schema, ownership, and explicit optional-capability ideas. Do not copy the requirement that every provider's heavy dependencies install into the backend environment: robot-plugin documentation explicitly requires compatible dependencies in that environment, whereas the model stacks below conflict. [S11, U3, X3]

### Existing design material is not an accepted decision

The Engine proposal already recommends library-first orchestration with both Lightning and upstream-native routes. Some facts have drifted: it refers to native GR00T configuration that is now removed, and older Studio Transformers pins. Keep its architectural arguments separate from its dated implementation observations. [S3, S12–S13]

The separate `docs/design-docs` branch contains draft Model Enablement Guidelines. They explicitly separate integration route, support level, validated capabilities, and portfolio investment, and recommend attempting export through upstream/adapters before a first-party port. These are useful proposals, not evidence of team approval. [D1]

## 3. Upstream findings

| Concern | Isaac-GR00T N1.7 | Xiaomi-Robotics-1 |
| --- | --- | --- |
| Training lifecycle | Tyro fine-tuning launcher constructs configuration; experiment invokes a custom Hugging Face-style trainer | Hydra/MMEngine setup builds a Lightning module and datamodule; `torchrun` launches Lightning with a DeepSpeed recipe |
| Inspected dependency pins | Python 3.12, Torch 2.9.0, Transformers 4.57.3, PEFT 0.17.1 | Torch 2.8.0, Lightning 2.5.3, Transformers 4.57.1, DeepSpeed 0.18.9 |
| Dataset interface | LeRobot-v2-style files plus modality/embodiment metadata | Episode JSON plus three synchronized camera videos and explicit state/action layout |
| Checkpoint/inference interface | Native model bundle, processor/statistics/configuration, `Gr00tPolicy`; ZMQ serving documented | Training output plus resolved config; inspected deployment loads DeepSpeed checkpoint contents and normalization configuration; client-server inference |
| Export evidence inspected | ONNX/TensorRT tooling exists, with several engines and residual host-side Torch operations | No export claim established by this inspection |

Sources: GR00T [U1–U6]; Xiaomi [X1–X7].

### Xiaomi is not a simple “port to Lightning” example

The upstream training command already constructs `lightning.Trainer`, and its model runner already inherits `LightningModule`. Replacing its recipe with Studio's trainer/datamodule is a separate behavior change, not a necessary consequence of using Lightning. Its data preparation, optimizer/scheduler settings, checkpoint layout, and async action behavior still need integration. [X1–X7]

### Dependency isolation solves a demonstrated problem

Studio requires Lightning `>=2.6.6` and PEFT `>=0.20.0`; its current VLA extras use Transformers 5.15.x. The backend's hardware extras select Torch 2.11.x. These declared requirements conflict with the inspected upstream pins. Merely installing a model plugin into Studio cannot satisfy the unchanged dependency sets together. [S12, S14, U3, X3]

A lightweight provider descriptor in the host and a compatible external worker/interpreter is a plausible solution. Installing all of `physicalai-train` into that worker would reintroduce conflicting requirements; keep any bridge/protocol dependencies small.

The user also identified dependency conflicts among first-party policies. Separate interpreters cannot fix mutually incompatible requirements if each environment must install the same conflicting package metadata. Extending isolation to first-party policies also requires compatible per-policy dependency sets and import behavior. The user accepted the incremental direction and requested sharing compatible environments across policies; the selection rule is in ADR 0001, and exact packaging is designed during the first proof.

The factory package currently imports all first-party families, and Pi0.5 imports Transformers at module scope. Selecting one policy must not import incompatible siblings if per-policy environments are to work. Lazy public exports can preserve import names while changing when implementations load. [S3, S15]

uv resolves optional extras together unless conflicts are declared. Explicit conflicts permit separate resolutions, not simultaneous installation of incompatible versions; mandatory conflicting base dependencies still need restructuring. This supports trying per-policy extras and compatible environment recipes before splitting every policy into a new distribution. [P2]

### Export does not require owning the whole training implementation

GR00T already has export tooling. Investigate reusing or extending that tooling before rewriting the model. However, its export consists of multiple components and retains host Torch operations; it is not automatically a Runtime/OpenVINO-compatible artifact. [U6]

Keep training ownership independent from export/quantization ownership. A Studio-owned export adapter can consume an upstream checkpoint or model and produce a deployment artifact, subject to model-specific conversion, licensing, dependency, and correctness work. Runtime remains the owner of artifact loading and inference execution. [R2]

## 4. Design direction

```text
Python / CLI / GUI
        |
shared workflow orchestration
        |
        +-- existing Lightning workflow
        |      +-- first-party policies
        |      +-- selected upstream-model adapters
        |
        +-- upstream-native providers
               +-- GR00T workflow in its environment
               +-- Xiaomi workflow in its environment
        |
validated results + native checkpoint bundles + provenance
        |
optional benchmark / export / quantization operations
        |
Runtime-compatible artifacts only where separately validated
```

### Selection rule: agreed in round 2

1. Reuse a validated upstream integration when it meets the required capability target.
2. For a new independent repository, establish the upstream baseline and start by delegating to its supported workflow.
3. Add deeper Lightning adaptation only for a concrete capability or customization requirement, not interface uniformity alone.
4. Own first-party model code only when it is required to meet an approved requirement that the upstream/adapter route cannot reasonably satisfy.
5. Support external contributor packages without changes to Studio core. Packaging does not decide training ownership.

[ADR 0002](0002-upstream-first-policy-integration.md) records this direction; it does not require migration away from existing first-party policies.

Native delegation has an upfront platform cost: process lifecycle, configuration validation, artifact collection, diagnostics, and cancellation. It becomes valuable across several distinct repositories. Do not call a shell script launcher a complete integration.

### Keep the shared interface small

The orchestration module should hide provider-specific preparation and invocation while exposing capability/schema discovery, validated operation requests, lifecycle events, and typed results. Keep inference sessions separate from long-running training operations; keep export optional and backend-specific. Do not require one giant abstract class implementing every Studio operation.

Keep upstream configuration semantics rather than translating all trainers into Lightning keywords. In particular, preserve global versus per-device batch sizes, schedule units, full-state resume versus weights-only initialization, and precision meaning. Native checkpoints remain native bundles; they are not renamed to `model.ckpt`.

Reuse the application's existing scheduling and worker placement. A library-level orchestration module must not become a second scheduler, package manager, Runtime loader, or universal remote execution system.

### Validation is part of the interface

Require evidence for each claimed model-version/operation/software-stack/hardware combination:

- Deterministic CPU tests for validation, errors, lifecycle events, cancellation, and artifact containment.
- A real upstream baseline and delegated train → save → reload → inference test on a supported accelerator stack.
- Explicit full-state-resume testing where promised; weight reload alone is insufficient.
- Dataset/action semantics: camera order, units, frames, normalization, history, action horizon, execution horizon, and reset behavior.
- Task-quality/benchmark evidence beyond one-step loss or tensor-shape checks.
- Separate export/Runtime round trips and measured quality/latency/memory evidence for each claimed format or quantization mode.

Avoid duplicating normalization or action queues across upstream and Studio. A server transport is not proof of interchangeable policy semantics.

### Trust and operations

Entry-point plugins and executable upstream configuration are trusted code. A virtual environment isolates dependencies, not malicious code. Use reviewed revisions/recipes, validated argv and paths, bounded structured request/event/result messages, explicit resource ownership, redacted diagnostics, and process-group cancellation for distributed workers. Follow the [library security rules](../../../../library/docs/development/security.md).

The user accepted explicit curated setup as well as attaching an existing environment, without silently installing or upgrading packages during training. The [experience proposal](policy-integration-experience.md) now also covers programmatic inputs and shared-environment behavior. Upstream pins require a security and licensing review before they become Studio-supported stack definitions.

## 5. Design tree and current frontier

### Round 1: agreed foundations

- **Q1 — Product promise:** capabilities are independently validated. Upstream capability work should be contributed back to pursue day-zero support.
- **Q2 — Public interfaces:** orchestration belongs in the library; the GUI consumes the library interface.
- **Q3 — Execution constraints:** separate environments are accepted. The user explicitly notes first-party policies also suffer dependency conflicts.
- **Q4 — Hardware release bar:** establish an upstream-supported baseline and contribute hardware enablement upstream for off-the-shelf support.
- **Q5 — Compatibility commitment:** retain `Policy`, `Trainer`, and the core Lightning pipeline; this is essential, not a temporary compatibility layer.

These foundations are recorded in [ADR 0001](0001-library-first-policy-orchestration.md). Implementation is deferred until the first proof of concept.

### Round 2: agreements and unresolved questions

- **Q6 — Agreed:** upstream route first; first-party model implementation only when required for customization or another concrete requirement.
- **Q7 — Agreed:** external model integrations without Studio-core edits; make contributor effort small.
- **Q8 — Open:** the user provisionally sees value in a common facade but requests a concrete design before deciding.
- **Q9 — Open:** first-party isolation must be evaluated for elegance, user experience, and backward compatibility.
- **Q10 — Open:** no preference on environment provisioning; the user requests a concrete, user-friendly design.
- **Q11 — Agreed:** permit downstream patches while upstream PRs are pending.
- **Q12 — Agreed:** use real end-to-end workflows; physical-hardware testing is an important goal. The exact hardware admission/release gates remain open.

See [ADR 0002](0002-upstream-first-policy-integration.md) for the settled trade-offs.

### Round 3: agreements and clarification requests

- **Q8 — Agreed with requirement:** the facade must have a proper programmatic interface taking model, dataset, and configuration objects. `Engine.run(yaml)` alone is insufficient. The meaning of an upstream model object still needs agreement.
- **Q9 — Agreed with clarification:** support first-party workflow isolation while preserving direct Lightning behavior. Compatible policies should share an environment; users should not switch shell environments per policy.
- **Q10 — Agreed:** explicit curated setup plus attach-existing support, with no silent dependency changes during training.
- **Q13 — Agreed:** physical tests gate real-robot deployment claims and relevant deployment changes, not every training-only contribution or contributor PR.
- **Q14 — Agreed:** one independently validated capability is enough to contribute a plugin.

The user reaffirmed Q11 and Q12 as part of agreeing to Q10–Q14. [ADR 0001](0001-library-first-policy-orchestration.md) and [ADR 0002](0002-upstream-first-policy-integration.md) now reflect these agreements.

### Round 4: user feedback

- **Q15 — Clarified:** the user expects upstream workflows to use the official framework's model, dataset, and configuration API, delegated by Engine, and questioned whether Studio should wrap upstream classes.
- **Q16 — Clarified:** the user expects conversion between Studio's dataset interface and official dataset contracts.
- **Q17 — Agreed with naming concern:** the user accepts the selection rule for a Python environment but rejects `execution` as ambiguous with Runtime.

Facts gathered: GR00T's `FinetuneConfig` is a plain dataclass with standard-library-only imports, but installing its distribution still installs conflicting pins. The launcher builds the full training configuration in `__main__`. GR00T N1.7 is also integrated in LeRobot. GR00T ships a LeRobot v3-to-v2 converter, while Studio's dataset import currently registers only a LeRobot v3 adapter and rejects v2 archives. Runtime defines `Execution` strategies, and Studio defines both gym environments and an Environment entity containing robots and cameras. [U2, U7–U9, S16–S17, R3]

### Round 5: current frontier

- **Q15 — Official-API rule:** official class always; live instance when importable and `class_path` reference otherwise; no Studio mirror classes; namespace selects the plugin.
- **Q16 — Dataset converters:** plugin-owned library converters; Studio-to-official first and official-to-Studio where well defined; reject or require review for semantic gaps.
- **Q17 — Naming:** `python_env` and “Python environment”, avoiding `execution`, `environment`/`env`, `runtime`, `backend`, `target`, and `sandbox`.
- **Q18 — Physical validation cadence:** reference robot/scenario matrix and when physical tests run. Unblocked by agreed Q12 and Q13.
- **Q19 — Plugin listing and maintenance:** requirements for curated GUI listing versus manual installation, trust, labeling, and maintenance ownership. Unblocked by agreed Q7, Q10, Q11, and Q14.

These questions do not depend on each other. The lightweight orchestration package for use inside upstream Python environments, upstream API requests, result and inference-session interfaces, and the proof-of-design plan follow Q15. Converter implementation scope follows Q16.

### Round 5: answers

- **Q15, Q16 — Awaiting confirmation:** the user did not answer the refined recommendations.
- **Q17 — Agreed:** `python_env`; the glossary term is now “Python environment”.
- **Q18 — Deferred:** the team currently uses SO-101 and the Seeed Studio reBot B601-RS.
- **Q19 — Agreed:** curated GUI listing with a named maintainer, pinned upstream revision, and per-capability evidence; manual installation remains possible.

Facts gathered: GR00T documents fine-tuning for SO100 and for the Seeed reBot Arm B601 DM, including LeRobot v3-to-v2 conversion, modality configuration, and real-robot evaluation. The team's B601 is the RS variant, and Studio's curated manifest lists only the B601 DM follower type. [U10–U11, S18]

### Round 6: current frontier

- **Q15, Q16 — Confirmation:** official-API rule and dataset converters.
- **Q20 — Python environment location:** resolve and prepare the environment on the machine running the job, including remote and SSH training targets. Unblocked by Q2, Q9, Q10, and Q17.
- **Q21 — First proof:** GR00T N1.7 fine-tuning on a Studio recording plus a first-party workflow through the same interface. Unblocked by Q6, Q12, and Q18's robot information; its detailed design follows Q15 and Q16.
- **Q22 — LeRobot wrapper:** label unvalidated wrappers experimental and defer redesign, or decide the LeRobot integration route now. Unblocked by Q1, Q6, Q12, and Q19.

After this round: result/artifact and model-catalog interface; inference and robot deployment of upstream-trained policies; export; lightweight Engine inside upstream environments; upstream contributions; and concrete converter mappings.

### Round 6: answers

- **Q15, Q16 — Agreed:** recorded in [ADR 0003](0003-upstream-apis-as-source-of-truth.md).
- **Q20 — Partly agreed:** the user asked why an explicit check is needed when Python already raises `ModuleNotFoundError`.
- **Q21 — Agreed:** the first proof converts a Studio LeRobot dataset to GR00T's format, with conversion handled in the library alongside `Dataset`, `LeRobotDataModule`, and `FormatConverter`. The first-party part of the proof was not addressed.
- **Q22 — Open evaluation:** the user wants to determine whether the wrapper or plugin route is better for LeRobot.

Facts gathered: GR00T's official converter requires Python `>=3.10,<3.12` and a LeRobot Git commit, while Studio and GR00T training require Python 3.12. `FormatConverter` converts batches in memory rather than datasets on disk. Runtime inference adapters implement `load` and `predict` and are discovered from installed packages. The backend expects `model.ckpt` and `exports/`. GR00T documents FFmpeg and `CUDA_HOME` failures that surface only when the affected code runs. [U1, U12, S9, S12, R4]

### Round 7: current frontier

- **Q20 — Preflight:** run the import early in the target Python environment, then check versions, upstream revision, downstream patches, and declared prerequisites before expensive work.
- **Q22 — LeRobot evaluation:** compare both routes on ACT and SmolVLA after the release; label unvalidated wrappers experimental meanwhile.
- **Q23 — Converter placement:** library-native LeRobot-to-GR00T conversion in Studio's environment, verified with GR00T's official loader; general interface in core, GR00T specifics in the plugin.
- **Q24 — Second proof implementation:** one first-party policy through Engine in its own Python environment.
- **Q25 — Training results:** official checkpoint bundles with provenance and validated capabilities in the model list.
- **Q26 — Inference and deployment:** a plugin-provided Runtime adapter delegating to upstream inference, with export as a separate capability.

These questions are independent of each other. Q25 and Q26 were unblocked by Q15. Later branches: export and quantization details, a lightweight Engine inside upstream environments, upstream contributions, concrete robot mappings, and where Studio-maintained plugins live.

### Round 7: answers

The user agreed to every Round 7 recommendation, and to designing the remaining branches during the first proof:

- **Q20:** fast check inside the job's Python environment before expensive work, on the machine that runs the job. Recorded in [ADR 0001](0001-library-first-policy-orchestration.md).
- **Q22:** label untested LeRobot wrappers experimental; compare wrapper and plugin approaches after the release. Recorded in [ADR 0002](0002-upstream-first-policy-integration.md).
- **Q23:** library-native LeRobot-to-GR00T conversion verified with GR00T's loader; plugins split into a lightweight part and a worker part. Recorded in [ADR 0003](0003-upstream-apis-as-source-of-truth.md).
- **Q24:** the first proof also runs one first-party policy through Engine in its own Python environment.
- **Q25, Q26:** official checkpoints with provenance; inference through plugin-provided Runtime adapters, with export validated separately. Recorded in [ADR 0004](0004-official-checkpoints-and-runtime-inference.md).

### Status: interview complete

No design question remains open. Deferred to the first proof: export and quantization details, a lightweight orchestration package for upstream Python environments, specific upstream contributions, concrete robot dataset mappings, where Studio-maintained plugins live, and physical-test cadence. The user has not requested implementation; this record is documentation only.

## Sources

### Studio

- [S1 — Policy](../../../../library/src/physicalai/policies/base/policy.py)
- [S2 — Trainer](../../../../library/src/physicalai/train/trainer.py)
- [S3 — Policy factory](../../../../library/src/physicalai/policies/__init__.py)
- [S4 — LeRobot wrapper](../../../../library/src/physicalai/policies/lerobot/policy.py)
- [S5 — Named and equivalence-validated policy sets](../../../../library/src/physicalai/policies/lerobot/__init__.py)
- [S6 — Equivalence tests and native replay](../../../../library/tests/integration/test_lerobot_wrapper_equivalence.py)
- [S7 — Application training interface](../../../../application/backend/src/services/training_backends/base.py)
- [S8 — Local training adapter](../../../../application/backend/src/services/training_backends/local.py)
- [S9 — Shared training job](https://github.com/open-edge-platform/physical-ai-studio/blob/main/application/backend/src/training/job.py)
- [S10 — Library CI](../../../../.github/workflows/library.yml)
- [S11 — Robot plugin architecture](https://github.com/open-edge-platform/physical-ai-studio/blob/main/application/docs/explanation/robot-plugin-architecture.md)
- [S12 — Library dependency declarations](../../../../library/pyproject.toml)
- [S14 — Backend dependency declarations](../../../../application/backend/pyproject.toml)
- [S15 — Pi0.5 model imports](../../../../library/src/physicalai/policies/pi05/model.py)
- [S16 — Dataset import flow](../../../../application/backend/src/services/dataset_import/README.md) and [registered adapters](../../../../application/backend/src/services/dataset_import/adapters/__init__.py)
- [S17 — Studio Environment setup](../../../../application/docs/04-environment-setup.md)
- [S18 — Curated robot plugin manifest](https://github.com/open-edge-platform/physical-ai-studio/blob/main/application/backend/src/plugins/manifest.json)
- [D1 — Model Enablement Guidelines, draft branch](https://github.com/open-edge-platform/physical-ai-studio/blob/docs/design-docs/library/docs/design/model-guidelines/model-implementation-guidelines.md)

### GR00T

- [U1 — Release, training, dataset, and serving documentation](https://github.com/NVIDIA/Isaac-GR00T/blob/51d4c89f72fda44cbf77285c6a8114b52676b8a1/README.md)
- [U2 — Fine-tuning launcher](https://github.com/NVIDIA/Isaac-GR00T/blob/51d4c89f72fda44cbf77285c6a8114b52676b8a1/gr00t/experiment/launch_finetune.py)
- [U3 — Dependencies](https://github.com/NVIDIA/Isaac-GR00T/blob/51d4c89f72fda44cbf77285c6a8114b52676b8a1/pyproject.toml)
- [U4 — Training implementation](https://github.com/NVIDIA/Isaac-GR00T/blob/51d4c89f72fda44cbf77285c6a8114b52676b8a1/gr00t/experiment/experiment.py)
- [U5 — Policy interface and observation/action semantics](https://github.com/NVIDIA/Isaac-GR00T/blob/51d4c89f72fda44cbf77285c6a8114b52676b8a1/getting_started/policy.md)
- [U6 — Export/deployment tooling and remaining Torch operations](https://github.com/NVIDIA/Isaac-GR00T/blob/51d4c89f72fda44cbf77285c6a8114b52676b8a1/scripts/deployment/README.md)
- [U7 — FinetuneConfig](https://github.com/NVIDIA/Isaac-GR00T/blob/51d4c89f72fda44cbf77285c6a8114b52676b8a1/gr00t/configs/finetune_config.py)
- [U8 — LeRobot v3-to-v2 converter](https://github.com/NVIDIA/Isaac-GR00T/blob/51d4c89f72fda44cbf77285c6a8114b52676b8a1/scripts/lerobot_conversion/convert_v3_to_v2.py)
- [U9 — LeRobot GR00T N1.7 integration](https://github.com/huggingface/lerobot/blob/9a6bb61043bac8c14353fcb6ea513b7473c118e3/src/lerobot/policies/groot/groot_n1_7.py)
- [U10 — GR00T SO100 example](https://github.com/NVIDIA/Isaac-GR00T/blob/51d4c89f72fda44cbf77285c6a8114b52676b8a1/examples/SO100/README.md)
- [U11 — GR00T reBot Arm B601 DM example](https://github.com/NVIDIA/Isaac-GR00T/blob/51d4c89f72fda44cbf77285c6a8114b52676b8a1/examples/rebot-arm-dm/README.md)
- [U12 — GR00T converter dependencies](https://github.com/NVIDIA/Isaac-GR00T/blob/51d4c89f72fda44cbf77285c6a8114b52676b8a1/scripts/lerobot_conversion/pyproject.toml)

### Xiaomi

- [X1 — Post-training and deployment guide](https://github.com/XiaomiRobotics/Xiaomi-Robotics-1/blob/0dd7aef8dc87296246aae812a1f59ccb708e5546/xr1/README.md)
- [X2 — Training entry point](https://github.com/XiaomiRobotics/Xiaomi-Robotics-1/blob/0dd7aef8dc87296246aae812a1f59ccb708e5546/xr1/tools/train.py)
- [X3 — Dependencies](https://github.com/XiaomiRobotics/Xiaomi-Robotics-1/blob/0dd7aef8dc87296246aae812a1f59ccb708e5546/xr1/assets/requirements.txt)
- [X4 — Distributed launcher](https://github.com/XiaomiRobotics/Xiaomi-Robotics-1/blob/0dd7aef8dc87296246aae812a1f59ccb708e5546/xr1/scripts/train.sh)
- [X5 — Lightning runner](https://github.com/XiaomiRobotics/Xiaomi-Robotics-1/blob/0dd7aef8dc87296246aae812a1f59ccb708e5546/xr1/mibot/models/runner/base_runner.py)
- [X6 — Training data format](https://github.com/XiaomiRobotics/Xiaomi-Robotics-1/blob/0dd7aef8dc87296246aae812a1f59ccb708e5546/xr1/docs/data_format.md)
- [X7 — Deployment checkpoint loading](https://github.com/XiaomiRobotics/Xiaomi-Robotics-1/blob/0dd7aef8dc87296246aae812a1f59ccb708e5546/xr1/mibot/server/deploy.py)

### Shared contracts and packaging

- [R1 — Robot catalog SDK](https://github.com/openvinotoolkit/physicalai/blob/2fafa820bb81798da92c9ed84f23d8542bc2f3c0/packages/physicalai-studio-plugin/README.md)
- [R2 — Runtime export/load contract](https://github.com/openvinotoolkit/physicalai/blob/2fafa820bb81798da92c9ed84f23d8542bc2f3c0/skills/inference/physicalai-runtime-loading-exported-policies/references/export-load-contract.md)
- [R3 — Runtime execution strategies](https://github.com/openvinotoolkit/physicalai/blob/2fafa820bb81798da92c9ed84f23d8542bc2f3c0/docs/explanation/runtime.md)
- [R4 — Runtime inference adapter interface](https://github.com/openvinotoolkit/physicalai/blob/2fafa820bb81798da92c9ed84f23d8542bc2f3c0/src/physicalai/inference/adapters/base.py)
- [P1 — PyPA entry-points specification](https://packaging.python.org/en/latest/specifications/entry-points/)
- [P2 — uv resolution: conflicting dependencies](https://docs.astral.sh/uv/concepts/resolution/#conflicting-dependencies)
