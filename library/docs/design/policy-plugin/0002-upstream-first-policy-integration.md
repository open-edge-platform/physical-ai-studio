---
status: accepted
---

# Prefer upstream workflows and independently contributed policy integrations

For new policy integrations, preserve the upstream workflow first and introduce Studio-owned model code only when a concrete requirement cannot reasonably be met through the upstream route or an adapter. Enable external contributors to distribute policy integrations without modifying Studio core, so adding a model does not require ownership of another model implementation. Keep Studio's existing first-party policies and Lightning pipeline intact.

## Consequences

- Reuse a validated upstream integration when it meets the target; do not add a parallel first-party implementation for uniformity alone.
- Make contributor integration effort a design criterion for the provider interface. A plugin can start with one independently validated capability, its configuration/recipe, operation adapter, artifact/result interpretation, and real-workflow test. The precise interface and package layout remain to be decided.
- Anyone may install a policy plugin manually. List a plugin in Studio's GUI only after review, with a named maintainer, a pinned upstream revision, and evidence for each claimed capability. Label capabilities without evidence as experimental, distinguish Studio-maintained from community-maintained plugins, and mark a listed plugin unsupported or remove it when it stops meeting these requirements.
- Contribute reusable changes upstream, but permit tested, pinned downstream patches while upstream work is pending. Retain ownership, provenance, regression coverage, and a plan to remove those patches.
- Label LeRobot wrappers outside the equivalence-tested set of ACT, Diffusion, and SmolVLA as experimental. After the release, compare the wrapper and plugin approaches for LeRobot on ACT and SmolVLA by agreement with LeRobot's own training, maintenance cost across a LeRobot upgrade, remaining Studio features such as LoRA, SnapFlow, export, and XPU, and dependency impact. LeRobot remains Studio's dataset library under either approach.
- Validate claimed workflows end to end using their real dependencies. Synthetic tests and mocks remain useful but do not establish model support.
- Require physical-hardware evidence before claiming support for a real-robot deployment scenario, and repeat relevant validation when deployment behavior changes. Simulation does not substitute for this evidence. Training-only integration and every contributor pull request do not require physical robot access; the recurring test schedule and robot/scenario matrix are deferred. The team's current robots are SO-101 and the Seeed Studio reBot B601-RS.

See the [research record](policy-integration-research.md) and [proposed user experience](policy-integration-experience.md). No implementation is authorized by this decision record.
