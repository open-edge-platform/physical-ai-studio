# Intel Hardware Enablement for Robot Learning

**Status**: Draft for team review \
**Audience**: Physical AI Studio, Runtime, PyTorch XPU, OpenVINO, infrastructure, and ecosystem teams \
**Scope**: Cross-team ownership and investment needed to scale robot-learning support on Intel hardware \

The Studio and Runtime model process is defined in [Model Enablement Guidelines](./model-implementation-guidelines.md).

## Executive Summary

Studio and Runtime need broad model coverage and production-quality Intel deployment. Per-model work can deliver both, but it will not scale if the same XPU, dependency, and OpenVINO gaps are solved repeatedly downstream.

This proposal recommends three coordinated work areas:

| Work area | Accountable outcome                                                    | Proposed owner                                           |
| --------- | ---------------------------------------------------------------------- | -------------------------------------------------------- |
| Product   | Supported Studio workflows and performant Runtime/OpenVINO deployment  | Studio and Runtime                                       |
| Platform  | Reusable XPU, OpenVINO, and quantization capabilities                 | Platform component teams                                 |
| Ecosystem | Intel support in priority frameworks and standalone model repositories | Named ecosystem owner with product and partnership support |

LeRobot is currently a high-leverage framework because it hosts many relevant policies. It is not a required destination or the only collaboration target. Future frameworks and high-value standalone model repositories use the same selection criteria and engagement model.

This proposal asks leadership to establish clear ownership for framework and model-repository engagement, allocate CI capacity for priority external projects on Intel hardware, create recurring triage with PyTorch XPU, OpenVINO, and related component owners, and recognize robot learning as a platform-planning workload.

These investments do not gate initial model availability. They determine how quickly Intel can reach optimized deployment and whether support cost per model decreases over time.

## 1. Proposed Operating Model

### Product: Studio and Runtime

Under this proposal, Studio and Runtime would own end-to-end customer outcomes. This would include model selection, support levels, capability targets, framework adapters, named integrations, and first-party implementations. It would also include training, fine-tuning, inference, export, task-level validation, Runtime artifact and metadata contracts, OpenVINO deployment, quantization evaluation, and downstream delivery when upstream fixes are not yet available.

Product work would generate complete-workload evidence and identify shared blockers that should move into platform or ecosystem ownership.

### Platform

Under this proposal, platform teams would own reusable capabilities that should not be solved separately for each model. These could include PyTorch XPU operations and kernels, supported alternatives to common CUDA-only dependencies, OpenVINO conversion and runtime support, and quantization tooling and performance.

Studio would supply a minimal reproducer, affected models and workflows, target hardware, measured impact, and any downstream workaround. Component teams would provide triage, disposition, and a supported fix or documented limitation.

### Ecosystem

Ecosystem work would reduce downstream maintenance by improving the projects where models originate. Targets would include shared frameworks such as LeRobot, future frameworks with relevant model or customer adoption, standalone repositories containing important models, and common dependencies used across multiple frameworks or models.

Intel should be able to engage directly with an important standalone model; the model would not need to move into a shared framework first. Projects should be prioritized based on customer and product impact, expected reuse, and Intel deployment value, whether the project is part of a framework or maintained in its own repository.

## 2. Ecosystem Prioritization

Prioritize external projects according to customer, product, benchmark, and roadmap demand; the number and relevance of affected models; reuse across model families and repositories; maintainer activity and willingness to collaborate; the feasibility of Intel hardware CI; XPU and OpenVINO deployment impact; and license, dependency, and long-term maintenance risk.

The proposed approach is to use the lowest engagement level that could produce a durable result:

| Level                   | Commitment                                                      | Typical use                                               |
| ----------------------- | --------------------------------------------------------------- | --------------------------------------------------------- |
| Contribution            | Issues, minimal reproducers, and focused fixes                  | Isolated XPU, dependency, or export gap                   |
| Integration             | Maintained adapter, compatibility tests, and CI                 | Framework or repository used by multiple supported models |
| Strategic collaboration | Named contacts, shared roadmap, Intel CI, and agreed interfaces | High-impact project with sustained mutual investment      |

LeRobot is a current candidate for strategic collaboration. Other frameworks such as StarVLA and Dexbotic or standalone repositories can enter the same level when evidence supports the investment.

## 3. Shared Blockers

The proposed approach would treat a common XPU, OpenVINO, or dependency limitation as a shared blocker rather than as separate problems for each affected model. It should be prioritized and reported according to its total product impact and reuse potential.

Downstream workarounds may be necessary to meet product commitments, but durable fixes should be pursued with the responsible platform team or upstream project. This would allow model-specific enablement to contribute to reusable platform and ecosystem capabilities.

## 4. Collaboration Interfaces

For priority frameworks and repositories, the teams should seek interfaces that make Intel enablement reusable. These could include device-neutral training and inference code, capability checks around vendor-specific dependencies, exportable model cores separated from host-side state and action selection, declared dynamic shapes, stable input/output contracts, checkpoint and feature metadata needed by Studio and Runtime, and XPU and export smoke tests on Intel hardware.

Studio would retain its product contract for training integration, export metadata, parity, quantization, and Runtime loading. Upstream implementation support should remove downstream shims, not fragment the product contract.

## 5. Decisions and Commitments

| Decision             | Accountable group                                   | Required outcome                                                                        |
| -------------------- | --------------------------------------------------- | --------------------------------------------------------------------------------------- |
| Ecosystem owner      | Product leadership with partnership or DevRel       | Named owner for framework and standalone-repository relationships                       |
| CI allocation        | Infrastructure leadership                           | Capacity and maintenance owner for priority-project XPU and export tests                |
| Platform triage      | PyTorch XPU, OpenVINO, and relevant component leads | Recurring review and disposition of ranked shared blockers                              |
| Workload recognition | Platform roadmap owners                             | Robot-learning evidence included in operator, kernel, export, and quantization planning |

Each commitment would need a named individual, start date, and review cadence. Team names alone should not be considered sufficient ownership.

## 6. Measures

Track outcomes rather than contribution volume.

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

The measures and ranked blocker backlog should be reviewed quarterly, with results used to adjust platform priorities, ecosystem investment, and Studio model commitments.

## 7. Risks

| Risk                                                       | Mitigation                                                                     |
| ---------------------------------------------------------- | ------------------------------------------------------------------------------ |
| External maintainers do not accept Intel work              | Continue normal contributions and carry narrow downstream patches              |
| Collaboration over-focuses on one framework                | Re-rank frameworks and standalone repositories using the same criteria         |
| Platform backlogs treat models as unrelated issues         | Group by root cause and report affected model and product impact               |
| Generic model availability is mistaken for product support | Report Available, Studio-supported, and Deployment-supported models separately |
| Downstream patches become permanent                        | Assign removal conditions and review dates to every patch                      |
| CI is added without maintenance ownership                  | Require an owner and supported test scope before allocation                    |
