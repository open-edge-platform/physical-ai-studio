# Model Enablement Reference

**Status**: Draft for team review \
**Scope**: Engineering records and validation checklists for [Model Enablement Guidelines](./model-implementation-guidelines.md)

The guideline defines policy and support claims. This reference defines the minimum implementation evidence.

## 1. Model Record

Create one record in the model tracker before model-specific implementation begins.

| Field       | Required content                                                                                   |
| ----------- | -------------------------------------------------------------------------------------------------- |
| Identity    | Model, upstream repository and revision, license, checkpoint, and host framework                   |
| Decision    | Portfolio status, integration route, support-level target, capability target, owner, and rationale |
| Status      | Each capability marked not evaluated, planned, in progress, validated, blocked, or excepted        |
| Evidence    | Tests, benchmarks, artifacts, supported stack, and known limitations                               |
| Maintenance | Downstream patches, upstream links, removal conditions, and review date                            |

For `Available`, record the generic adapter smoke test and upstream identifier. For `Studio-supported` or `Deployment-supported`, complete the relevant checklists below.

## 2. Baseline

- [ ] Run the upstream model before modifying it.
- [ ] Record source revision, packages, checkpoint, seeds, hardware, and dependencies.
- [ ] Save representative fixed inputs, outputs, and action behavior.
- [ ] Record the applicable task-quality metric.
- [ ] Record training curve or eager performance when part of the capability target.

Use this baseline for integration, XPU, OpenVINO, quantization, and first-party equivalence checks.

## 3. Studio Integration

- [ ] Configuration, checkpoint loading, preprocessing, policy call, and postprocessing validated.
- [ ] Integration behavior agrees with the upstream baseline within the approved tolerance.
- [ ] Supported workflows, hardware, software, and limitations documented.
- [ ] Named integration and regression tests added for a Studio-supported claim.

Prefer a reusable framework adapter. A standalone repository may use the same contract directly. Do not require migration to a shared framework.

## 4. XPU Capabilities

Validate and report training, fine-tuning, and eager inference separately.

| Capability      | Minimum evidence                                                                                                  |
| --------------- | ----------------------------------------------------------------------------------------------------------------- |
| Training        | Training metric, baseline comparison, throughput, memory use, hardware, and software stack                        |
| Fine-tuning     | Resulting task quality, baseline comparison, throughput, memory use, hardware, and software stack                 |
| Eager inference | Fixed-input and action equivalence, task quality, latency or throughput, memory use, hardware, and software stack |

| Failure class                       | Default action                                                          |
| ----------------------------------- | ----------------------------------------------------------------------- |
| Hardcoded CUDA or device assumption | Fix locally and offer the reusable fix upstream                         |
| CUDA-only dependency                | Substitute behind a capability check without changing expected behavior |
| Missing or slow XPU operation       | File a minimal reproducer and use a narrow fallback only when required  |

## 5. OpenVINO and Quantization

Attempt OpenVINO through the adapter or upstream policy before proposing first-party code. Prefer direct PyTorch export when reliable; use ONNX as an interchange path when needed.

### OpenVINO Feasibility Assessment

- [ ] Record the attempted export path, versions, shapes, and failure or result.
- [ ] Identify model code, dependency, PyTorch export, OpenVINO, or Runtime ownership for blockers.
- [ ] Link a minimal reproducer and upstream issue where applicable.
- [ ] Estimate the workaround and maintenance cost.

### `OpenVINO-enabled`

- [ ] Artifact, metadata, and `InferenceModel.load(...)` round trip validated.
- [ ] Numerical and task-quality budgets met.
- [ ] End-to-end latency, throughput, memory use, and startup time measured for the target scenario.
- [ ] Results compared with the applicable eager baseline.

### Quantization Evaluation

- [ ] Method, calibration data, precision, and hardware recorded.
- [ ] Task quality and end-to-end performance compared with the non-quantized artifact.
- [ ] Decision recorded: adopt, reject with evidence, or blocked.

## 6. First-Party Proposal

Before approving a first-party implementation, record:

- Adapter or upstream limitation and failed export evidence.
- Dependency and license constraints.
- Expected capability or performance benefit.
- Initial and ongoing maintenance cost.
- Equivalence-test and upstream synchronization plan.

## 7. Exceptions and Patches

An exception includes the affected capability and scenario, reproducer, attempted versions and workarounds, rationale, owner, approval, and review date.

Every downstream patch records affected models, upstream issue or pull request when one exists, removal condition, and owner. Remove the patch after the supported upstream fix is adopted.
