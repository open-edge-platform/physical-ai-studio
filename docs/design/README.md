# Design Documentation

Architecture and design documents for **Physical AI Studio** — the training-side package (`physicalai-train`) that collects data, trains policies, benchmarks, and exports artifacts for the `physicalai` runtime.

Design docs are organized next to the code they describe, with this file as the single discovery index. There are three locations:

| Location | Scope | Index |
| --- | --- | --- |
| `docs/design/` | Cross-cutting: strategy, architecture, deployment stack, team plans | this file |
| [`library/docs/design/`](../../library/docs/design/README.md) | Library module & component designs (`physicalai-train`) | library README |
| `application/docs/designs/` | Backend / UI application designs | placeholder (none yet) |

---

## Start Here

**[Strategy](./strategy.md)** — the two core architectural decisions:

1. **Library-first:** the library owns all components; the application is UI/orchestration.
2. **Layered deployment stack:** `physicalai` (universal physical-AI engine, inference core, unified `manifest.json`, built-in runners) → external plugins (exotic patterns only). Vision (`model_api`) may share the inference core.

---

## Team Plans

| Document | Description |
| --- | --- |
| [Q3 Strategy](./q3-strategy/studio-strategy.md) | Execution items for `physicalai-train` and `physicalai` runtime in Q3 |
| [Summer Plan / Paper Submission](./q3-strategy/feature-improvements.md) | Feature work needed for ICRA / ICLR / CVPR submissions |
| [Planning Epics & Tasks](./q3-strategy/planning/README.md) | Breakdown of epics and tasks supporting the Q3 strategy |
| [Model Coverage Tracker](./q3-strategy/policies.csv) | Per-policy coverage state (CSV) |
| [Benchmark Data](./q3-strategy/benchmark.csv) | Benchmark tracking data (CSV) |

---

## Model Enablement

Per-model process for selecting, integrating, validating, optimizing, and maintaining robot-learning policies, plus the policy extension/plugin mechanism. Lives under `library/docs/design/` because it governs library-owned code paths.

| Document | Description |
| --- | --- |
| [Model Enablement Guidelines](../../library/docs/design/model-guidelines/model-implementation-guidelines.md) | Per-model process for Studio training and Runtime deployment |
| [Model Enablement Reference](../../library/docs/design/model-guidelines/model-implementation-guidelines-reference.md) | Implementation checklists and reference details |
| [Intel Hardware Enablement for Robot Learning](../../library/docs/design/model-guidelines/intel-enablement-strategy.md) | Cross-team platform & upstream enablement work |
| [Guidelines Slides](../../library/docs/design/model-guidelines/model-implementation-guidelines-slides.md) | Slide deck companion to the guidelines |
| [Policy Plugin and Model Enablement Strategy](../../library/docs/design/policy-plugin/plugin-options.md) | Proposal making an installed `Policy` subclass the unit of extension |

---

## Deployment Stack

The inference and deployment architecture: `physicalai` runtime → external plugins. Vision remains a separate domain layer (`model_api`) that can share the inference core.

| Component | Document | Description |
| --- | --- | --- |
| Packaging Strategy | [Two-Repo Options](./deployment/physical-ai-two-repo-options.md) | Two-repo, two-distribution packaging with PEP 420 namespace split |
| Inference Core | [Inference Core](./deployment/inferencekit.md) | Domain-agnostic inference layer design (internal to framework) |
| Deployment Engine | [Deployment Shell](./deployment/deployment-shell.md) | Runtime engine, CLI, manifest loading |
| LeRobot Integration | [LeRobot Integration](./deployment/lerobot.md) | Built-in format loader for LeRobot `manifest.json` (proposed) |

---

## Library Component Interfaces

Runtime-side component interface designs. Live under `library/docs/design/components/` because they describe library-owned abstractions.

| Component | Document | Description |
| --- | --- | --- |
| Robot Interface | [Robot Interface](../../library/docs/design/components/robot-interface.md) | Robot ABC, leader/follower wrappers, SDK integration (LeRobot, UR, ABB) |
| Camera Interface | [Camera Interface](../../library/docs/design/components/camera-interface.md) | `physicalai.capture` — camera classes, 3-tier reads, timestamped frames |
| Benchmarking | [Benchmarking API](../../library/docs/design/components/benchmarking.md) | NumPy-only benchmark protocols, runner, latency metrics |
| Teleoperation | [Teleoperation API](../../library/docs/design/components/teleoperation.md) | Leader/follower semantics, session lifecycle, safety primitives |
| Data Collection | [Data Collection API](../../library/docs/design/components/data-collection.md) | DatasetWriter, episode management, HF Hub upload |

---

## Internal Notes

| Document | Description |
| --- | --- |
| [LeRobot Export Suggestions](./internal/lerobot-export-suggestions.md) | Improvement suggestions for the LeRobot export API |

---

## Application Designs

Backend and UI design documents live in [`application/docs/designs/`](../../application/docs/designs/). None are merged yet; when app-side designs land (e.g. remote-training, trainer-service), they will be indexed here.

---

## Repository Ownership (Split Plan)

The design set spans two repos. Use this as the source of truth when copying docs into `physicalai` (runtime) and `physicalai-studio` (training).

**Physical-AI runtime repo (`physical-ai`):**

| Document | Purpose |
| --- | --- |
| [Strategy](./strategy.md) | Architecture principles, runtime/training split |
| [Packaging Strategy](./deployment/physical-ai-two-repo-options.md) | Two-repo, two-distribution plan |
| [Inference Core](./deployment/inferencekit.md) | Domain-agnostic inference core design |
| [Deployment Shell](./deployment/deployment-shell.md) | Runtime engine + CLI |
| [Robot Interface](../../library/docs/design/components/robot-interface.md) | Runtime robot API |
| [Camera Interface](../../library/docs/design/components/camera-interface.md) | Runtime capture API |
| [Benchmarking API](../../library/docs/design/components/benchmarking.md) | Runtime benchmark protocols + runner |
| [LeRobot Integration](./deployment/lerobot.md) | Runtime loader integration |

**Studio repo (`physical-ai-studio` / `physicalai-train`):**

| Document | Purpose |
| --- | --- |
| [Teleoperation API](../../library/docs/design/components/teleoperation.md) | Training-side teleop workflows |
| [Data Collection API](../../library/docs/design/components/data-collection.md) | Dataset recording and upload |
| [Model Enablement Guidelines](../../library/docs/design/model-guidelines/model-implementation-guidelines.md) | Per-model integration process |
| [Q3 Strategy](./q3-strategy/studio-strategy.md) | Team execution plan |

**Archive / legacy context:**

| Document | Purpose |
| --- | --- |
| [Architecture Pitch](./architecture-pitch.md) | Historical proposal context |
| [Framework Presentation](./deployment/physical-ai-framework-presentation.md) | Legacy framing before packaging split |

---

## Directory Structure

```
docs/design/                              # Cross-cutting designs + this index
├── README.md                            # This file
├── strategy.md                         # Architecture vision & key decisions
├── architecture-pitch.md               # Historical proposal (archive)
├── deployment/                          # Deployment stack designs
│   ├── physical-ai-two-repo-options.md
│   ├── inferencekit.md
│   ├── deployment-shell.md
│   ├── lerobot.md
│   └── physical-ai-framework-presentation.md  # (archive)
├── internal/
│   └── lerobot-export-suggestions.md
└── q3-strategy/                         # Team plans
    ├── studio-strategy.md
    ├── feature-improvements.md
    ├── policies.csv
    ├── benchmark.csv
    └── planning/{epics,tasks}/

library/docs/design/                     # Library module designs (physicalai-train)
├── README.md                           # Library design index
├── cli/ config/ data/ eval/ execution/  # Module designs
├── export/ gyms/ inference/ policy/ trainer/
├── components/                          # Runtime component interface designs
│   ├── robot-interface.md
│   ├── camera-interface.md
│   ├── benchmarking.md
│   ├── teleoperation.md
│   └── data-collection.md
├── model-guidelines/                    # Per-model integration process
│   ├── model-implementation-guidelines.md
│   ├── model-implementation-guidelines-reference.md
│   ├── intel-enablement-strategy.md
│   └── model-implementation-guidelines-slides.{md,html}
└── policy-plugin/                       # Policy extension mechanism (proposal)
    └── plugin-options.md

application/docs/designs/                # Application (backend/UI) designs
```

---

## Reading Guide

### For New Readers

1. Start with **[Strategy](./strategy.md)** — the two core decisions
2. Read **[Inference Core](./deployment/inferencekit.md)** for the inference foundation
3. Browse [library design index](../../library/docs/design/README.md) for module-level detail

### For Library Development

1. **[Strategy § Part 1](./strategy.md)** — ownership boundaries
2. Module design for the area you're working on (see [library README](../../library/docs/design/README.md))
3. For new models: **[Model Enablement Guidelines](../../library/docs/design/model-guidelines/model-implementation-guidelines.md)**

### For Deployment

1. **[Strategy § Part 2](./strategy.md)** — layered architecture
2. **[Packaging Strategy](./deployment/physical-ai-two-repo-options.md)** — two-repo split
3. **[Inference Core](./deployment/inferencekit.md)** — domain-agnostic inference layer
4. **[Deployment Shell](./deployment/deployment-shell.md)** — CLI and configuration

---

## Contributing

Design docs follow the PR-based review workflow:

1. **WIP staging:** develop on the `docs/design-docs` branch (or a feature branch off it) and push to your fork.
2. **Review:** open a PR to `upstream/main` carrying just the doc(s) being finalized.
3. **Merge:** only after team approval — at that point the doc is live for the team.

When updating these documents:

- Keep each document focused on a single component or topic.
- Maintain cross-references between related documents using relative paths.
- Put library-owned module designs under `library/docs/design/`, deployment and cross-cutting docs under `docs/design/`, and application designs under `application/docs/designs/`.
- **Update this README** whenever you add, remove, rename, or move a document — it is the discovery surface for the whole design set.