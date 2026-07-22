# Model Enablement Guidelines

**Status**: Draft for team review

**Audience**: Physical AI Studio and Runtime teams

**Scope**: How Studio selects, integrates, validates, optimizes, and maintains robot-learning policies

Implementation checklists are in [Model Enablement Reference](./model-implementation-guidelines-reference.md). Cross-team platform and ecosystem work is covered in [Intel Hardware Enablement for Robot Learning](./intel-enablement-strategy.md).

## Executive Summary

Studio has two goals: make many useful models available and deliver production-quality Intel deployment for selected models. These goals require different levels of investment and must not be reported as one model count.

Track four independent attributes for each model:

| Attribute              | Values                                                                                             | Question answered                         |
| ---------------------- | -------------------------------------------------------------------------------------------------- | ----------------------------------------- |
| Integration route      | Generic adapter, named integration, first-party implementation                                     | How is the model connected to Studio?     |
| Support level          | Available, Studio-supported, Deployment-supported                                                  | What does the team maintain and support?  |
| Validated capabilities | Studio integration; XPU training, fine-tuning, or inference; Runtime/OpenVINO; quantized inference | What has been demonstrated?               |
| Portfolio status       | Committed, candidate, monitor                                                                      | How much investment is currently planned? |

The operating policy is:

- Use generic adapters to increase availability without creating a model-specific support commitment.
- Add a named integration when Studio commits to documented workflows, tests, and maintenance.
- Attempt OpenVINO through the adapter or upstream model before building a first-party implementation.
- Assess OpenVINO feasibility for every Studio-supported model. OpenVINO is required for Deployment-supported status.
- Evaluate quantization for every OpenVINO-enabled model. Adopt it when measured deployment benefit justifies the quality and maintenance cost.
- Build first-party implementations only when adapters cannot meet product, dependency, export, performance, or maintenance requirements.
- Deliver downstream when needed, while contributing reusable fixes to frameworks, model repositories, PyTorch XPU, OpenVINO, and related projects.

This approach separates breadth from depth: generic availability expands coverage, while named and first-party integrations receive explicit validation and Intel optimization investment.

## 1. Support Model

### 1.1 Integration Route

| Route                      | Use when                                                                                           | Studio responsibility                                             |
| -------------------------- | -------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------- |
| Generic adapter            | A framework or repository can be exposed through a common plugin contract                          | Maintain the adapter; no model-specific guarantee                 |
| Named integration          | Studio supports specific workflows, configuration, aliases, tests, and documentation for the model | Maintain the documented model behavior and capability evidence    |
| First-party implementation | An adapter cannot meet approved requirements at acceptable cost                                    | Maintain model code, compatibility, and approved deployment paths |

An integration route is not a quality claim. A model can reach production deployment through an adapter, and a first-party implementation is not automatically Deployment-supported.

### 1.2 Support Level

| Level                  | Claim                                                                         | Minimum requirement                                                                          |
| ---------------------- | ----------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------- |
| `Available`            | The model is accessible through a generic adapter; upstream limitations apply | Adapter smoke test and upstream model identifier documented                                  |
| `Studio-supported`     | Studio maintains documented model workflows                                   | Named integration, pinned baseline, capability tests, supported stack, and known limitations |
| `Deployment-supported` | Studio and Runtime maintain an Intel production deployment path               | Studio-supported plus validated Runtime/OpenVINO artifact and deployment benchmark           |

Support claims apply only to documented workflows, hardware, and software versions. `Available` models must not be counted or described as Studio-supported models.

### 1.3 Portfolio Status

| Status      | Meaning                                                                                 |
| ----------- | --------------------------------------------------------------------------------------- |
| `Committed` | Customer, product, benchmark, or roadmap need has an owner and funded capability target |
| `Candidate` | The model is being evaluated; investment and target are not approved                    |
| `Monitor`   | No current implementation commitment; revisit when evidence changes                     |

Portfolio status sets investment priority; it does not change technical support claims.

## 2. Capability Milestones

Capabilities are tracked independently because training, fine-tuning, eager inference, export, and quantization do not always progress in sequence. Record each as `not evaluated`, `planned`, `in progress`, `validated`, `blocked`, or `excepted`.

| Milestone                   | Claim                                                                    | Required evidence                                                                                             |
| --------------------------- | ------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------- |
| `Model-enabled`             | A named or first-party integration supports a documented Studio workflow | Pinned baseline, Studio integration test, configuration, supported stack, and known limitations               |
| `XPU training validated`    | Training works on documented Intel GPU hardware                          | Baseline comparison, training metric, throughput, memory use, and software stack                              |
| `XPU fine-tuning validated` | Fine-tuning works on documented Intel GPU hardware                       | Baseline comparison, task-quality metric, throughput, memory use, and software stack                          |
| `XPU inference validated`   | Eager Torch inference works on documented Intel GPU hardware             | Fixed-input and action equivalence, task quality, latency or throughput, and memory use                       |
| `OpenVINO-enabled`          | Runtime loads a validated OpenVINO artifact for the target scenario      | Artifact and metadata, `InferenceModel.load(...)`, numerical and task parity, end-to-end deployment benchmark |
| `Quantized`                 | A quantized artifact provides an accepted deployment tradeoff            | Task-quality result and measured latency, throughput, or memory benefit over the non-quantized artifact       |

Do not use `XPU-enabled` without naming the validated workflow. Validation of eager inference does not imply training or fine-tuning support.

A model is `Target complete` when all capabilities in its approved target are validated or have an approved exception. Target completion is an execution status, not a permanent support level; targets can expand as product needs change.

## 3. OpenVINO and Quantization Policy

OpenVINO is Intel's preferred production deployment path because it can materially improve latency, throughput, memory use, and deployment portability.

- Every Studio-supported model receives an OpenVINO feasibility assessment.
- Every Deployment-supported model requires an OpenVINO-enabled path through Runtime.
- Every OpenVINO-enabled model receives a quantization evaluation after the graph is stable.
- A quantized artifact is adopted only when task quality remains within its approved budget and deployment benefit is measurable.

An assessment can identify a blocker without delaying Model-enabled or XPU capability delivery. For committed deployment targets, blocked OpenVINO work remains owned and tracked until resolved or excepted. An exception records the reproducer, affected scenario, attempted versions and workarounds, rationale, owner, and review date.

Measure end-to-end behavior for the target robot workload, including critical preprocessing, model execution, state handling, and postprocessing. Tensor parity alone is not production evidence.

## 4. Model Selection and Implementation

Evaluate models using product demand, adoption, license, architecture and dependency risk, framework maturity, reuse across model families, expected Intel deployment value, and maintenance cost.

Use this implementation order:

1. Expose the model through an existing generic adapter when possible.
2. Create a named integration when Studio-supported behavior is approved.
3. Attempt XPU and OpenVINO enablement through the adapter and upstream model.
4. Propose a first-party implementation only when evidence shows the adapter path cannot meet the approved target.

A first-party proposal must include the adapter limitation, export attempt, dependency and license constraints, expected performance or capability benefit, ongoing maintenance cost, and upstream synchronization plan.

For a new framework, build one reusable adapter rather than independent model ports. For a standalone repository, use the same contract where practical; do not require the model to move into a framework before Studio can support it.

## 5. Enablement Process

1. **Prioritize**: assign portfolio status, owner, support-level target, capability target, and integration route.
2. **Baseline**: run the upstream model and record source revision, checkpoint, fixed inputs, seeds, metrics, dependencies, and hardware.
3. **Integrate**: add the adapter, named integration, or first-party implementation and validate Model-enabled evidence where applicable.
4. **Validate XPU capabilities**: test training, fine-tuning, and inference separately according to the approved target.
5. **Enable deployment**: export through the existing integration, validate Runtime/OpenVINO correctness, and benchmark the target scenario.
6. **Evaluate quantization**: record whether it is adopted, rejected with benchmark evidence, or blocked.
7. **Maintain**: publish supported capabilities, track blockers and patches, and review target completion.

The model tracker is the source of truth. Each record links to evidence, issues, artifacts, downstream patches, and upstream work.

## 6. Validation and Evidence

The model owner and reviewer agree on test tolerances and task-quality budgets before validation. Record the test suite, model and checkpoint version, software stack, hardware, and measurement method.

Use evidence appropriate to the capability:

- Fixed-input outputs and action behavior.
- Training or fine-tuning curves and resulting task quality.
- Product-relevant task or benchmark success.
- End-to-end latency, throughput, memory use, and startup time.
- Runtime artifact load and metadata validation.

Report OpenVINO results against the applicable eager baseline and quantized results against the non-quantized artifact. A performance claim must name the workload, hardware, precision, input shape, and measurement boundary.

## 7. Upstream and Maintenance Policy

Studio and Runtime own end-to-end product behavior. They may ship downstream fixes without waiting for upstream review, but reusable fixes should be contributed to the appropriate framework, model repository, PyTorch XPU, OpenVINO, or dependency project.

1. Link each downstream patch to an upstream issue or pull request when an upstream path exists.
2. Record every affected model so shared blockers can be prioritized by impact.
3. Remove the downstream patch when the supported upstream version contains the fix.
4. Retain Studio's integration and Runtime contract even when implementation details move upstream.

## 8. Portfolio Review

Review the portfolio quarterly using separate breadth, support, and deployment measures:

- Models Available through generic adapters.
- Models with Studio-supported workflows.
- Models with each validated XPU capability.
- Models with Deployment-supported OpenVINO paths.
- Time from model release to Model-enabled and OpenVINO-enabled.
- OpenVINO and quantization improvements in target scenarios.
- Shared blockers by affected model count and product impact.
- Downstream patches, upstream status, and removal rate.
- Maintenance cost by integration route.

Raw model count is not a sufficient success measure. The portfolio should show both coverage and the depth of validated Intel deployment support.
