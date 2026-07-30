# Intel Hardware Enablement for Robot Learning

**Status**: Draft for team review \
**Audience**: Physical AI Studio, Runtime, PyTorch XPU, oneAPI, OpenVINO, infrastructure, and ecosystem teams \
**Scope**: Cross-team ownership and investment needed to scale robot-learning support on Intel hardware \

The Studio and Runtime model process is defined in [Model Enablement Guidelines](./model-implementation-guidelines.md).

## Executive Summary

Studio and Runtime need broad model coverage and production-quality Intel deployment. Per-model work can deliver both, but it will not scale if the same XPU, dependency, and OpenVINO gaps are solved repeatedly downstream.

Intel needs three coordinated work areas:

| Work area | Accountable outcome                                                    | Primary owner                                            |
| --------- | ---------------------------------------------------------------------- | -------------------------------------------------------- |
| Product   | Supported Studio workflows and performant Runtime/OpenVINO deployment  | Studio and Runtime                                       |
| Platform  | Reusable XPU, oneAPI, OpenVINO, and quantization capabilities          | Platform component teams                                 |
| Ecosystem | Intel support in priority frameworks and standalone model repositories | Named ecosystem owner with product and partnership support |

LeRobot is currently a high-leverage framework because it hosts many relevant policies. It is not a required destination or the only collaboration target. Future frameworks and high-value standalone model repositories use the same selection criteria and engagement model.

This proposal asks leadership to establish:

- A named owner for framework and model-repository engagement.
- CI capacity for priority external projects on Intel hardware.
- A recurring triage with PyTorch XPU, OpenVINO, and related component owners.
- Robot learning as a recognized workload in platform planning.

These investments do not gate initial model availability. They determine how quickly Intel can reach optimized deployment and whether support cost per model decreases over time.

## 1. Operating Model

### Product: Studio and Runtime

Studio and Runtime own end-to-end customer outcomes:

- Model selection, support level, and capability targets.
- Framework adapters, named integrations, and first-party implementations.
- Training, fine-tuning, inference, export, and task-level validation.
- Runtime artifact and metadata contracts.
- OpenVINO deployment and quantization evaluation.
- Downstream delivery when upstream fixes are not yet available.

Product work generates complete-workload evidence. It also identifies shared blockers that should move into platform or ecosystem ownership.

### Platform

Platform teams own reusable capabilities that should not be solved separately for each model:

- Missing or slow PyTorch XPU operations and kernels.
- oneAPI capabilities needed by robot-learning workloads.
- Supported alternatives to common CUDA-only dependencies.
- OpenVINO conversion, dynamic-shape, runtime, and performance gaps.
- Quantization tooling and accuracy or performance issues.

Studio supplies a minimal reproducer, affected models and workflows, target hardware, measured impact, and downstream workaround when one exists. Component teams provide triage, disposition, and a supported fix or documented limitation.

### Ecosystem

Ecosystem work reduces downstream maintenance by improving projects where models originate. Targets include:

- Shared frameworks such as LeRobot.
- Future frameworks that gain relevant model or customer adoption.
- Standalone repositories containing important models.
- Common dependencies used across multiple frameworks or models.

Intel can engage directly with an important standalone model; the model does not need to move into a shared framework first. Prioritize projects based on customer and product impact, expected reuse, and Intel deployment value, whether the project is part of a framework or maintained in its own repository.

## 2. Ecosystem Prioritization

Prioritize external projects using:

- Customer, product, benchmark, or roadmap demand.
- Number and relevance of models affected.
- Reuse across model families and repositories.
- Maintainer activity and willingness to collaborate.
- Feasibility of Intel hardware CI.
- XPU and OpenVINO deployment impact.
- License, dependency, and long-term maintenance risk.

Use the lowest engagement level that can produce a durable result:

| Level                   | Commitment                                                      | Typical use                                               |
| ----------------------- | --------------------------------------------------------------- | --------------------------------------------------------- |
| Contribution            | Issues, minimal reproducers, and focused fixes                  | Isolated XPU, dependency, or export gap                   |
| Integration             | Maintained adapter, compatibility tests, and CI                 | Framework or repository used by multiple supported models |
| Strategic collaboration | Named contacts, shared roadmap, Intel CI, and agreed interfaces | High-impact project with sustained mutual investment      |

LeRobot is a current candidate for strategic collaboration. Other frameworks or standalone repositories can enter the same level when evidence supports the investment.

## 3. Shared Blocker Process

Studio owns one blocker backlog grouped by root cause rather than by model. Each blocker records:

- Owning component or external project.
- Minimal reproducer and affected workflow.
- Affected models and support targets.
- Hardware and software versions.
- Correctness, performance, or schedule impact.
- Downstream workaround and maintenance cost.
- Upstream issue or pull request.
- Owner, disposition, and next review date.

Use the same operating rule across product, platform, and ecosystem work:

1. Ship a narrow downstream fix when needed for an approved product target.
2. Offer the reusable fix to the responsible upstream project in parallel.
3. Link downstream and upstream work through the blocker record.
4. Remove the downstream patch after the supported upstream fix is adopted.

One dependency blocking six models is one shared blocker with six-model impact, not six unrelated support requests.

## 4. Collaboration Interfaces

For priority frameworks and repositories, seek interfaces that make Intel enablement reusable:

- Device-neutral training and inference code.
- Capability checks around vendor-specific dependencies.
- Exportable model cores separated from host-side state and action selection.
- Declared dynamic shapes and stable input/output contracts.
- Checkpoint and feature metadata needed by Studio and Runtime.
- XPU and export smoke tests on Intel hardware.

Studio retains its product contract for training integration, export metadata, parity, quantization, and Runtime loading. Upstream implementation support should remove downstream shims, not fragment the product contract.

## 5. Decisions and Commitments

| Decision             | Accountable group                                   | Required outcome                                                                        |
| -------------------- | --------------------------------------------------- | --------------------------------------------------------------------------------------- |
| Ecosystem owner      | Product leadership with partnership or DevRel       | Named owner for framework and standalone-repository relationships                       |
| CI allocation        | Infrastructure leadership                           | Capacity and maintenance owner for priority-project XPU and export tests                |
| Platform triage      | PyTorch XPU, OpenVINO, and relevant component leads | Recurring review and disposition of ranked shared blockers                              |
| Workload recognition | Platform roadmap owners                             | Robot-learning evidence included in operator, kernel, export, and quantization planning |

Each commitment needs a named individual, start date, and review cadence. Team names alone are not sufficient ownership.

## 6. Measures

Track outcomes rather than contribution volume:

| Measure                                                                 | Target direction |
| ----------------------------------------------------------------------- | ---------------- |
| Time from model release to Studio-supported integration                 | Down             |
| Time from Studio-supported to OpenVINO-enabled                          | Down             |
| Models with validated Runtime/OpenVINO deployment                       | Up               |
| OpenVINO latency, throughput, or memory improvement in target scenarios | Up               |
| Priority external policies passing XPU and export smoke tests           | Up               |
| Models affected by unresolved shared blockers                           | Down             |
| Downstream patches carried per supported model                          | Down             |
| Upstream fixes adopted relative to downstream patches added             | Up               |

Review the measures and ranked blocker backlog quarterly. Use the results to adjust platform priorities, ecosystem investment, and Studio model commitments.

## 7. Risks

| Risk                                                       | Mitigation                                                                     |
| ---------------------------------------------------------- | ------------------------------------------------------------------------------ |
| External maintainers do not accept Intel work              | Continue normal contributions and carry narrow downstream patches              |
| Collaboration over-focuses on one framework                | Re-rank frameworks and standalone repositories using the same criteria         |
| Platform backlogs treat models as unrelated issues         | Group by root cause and report affected model and product impact               |
| Generic model availability is mistaken for product support | Report Available, Studio-supported, and Deployment-supported models separately |
| Downstream patches become permanent                        | Assign removal conditions and review dates to every patch                      |
| CI is added without maintenance ownership                  | Require an owner and supported test scope before allocation                    |
