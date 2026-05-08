# Design Documentation

Architecture and design documents for the physical‑AI runtime.

---

## Start Here

- **[Strategy](./architecture/strategy.md)** — architecture vision, scope, and key decisions
- **[Architecture](./architecture/architecture.md)** — physicalai runtime architecture and CLI
- **[Packaging Strategy](./packaging/physical-ai-two-repo-options.md)** — two‑repo, two‑distribution plan
- **[Modular Packages In One Repo](./packaging/modular-packages-in-one-repo.md)** — how to publish reusable packages without an early repo split

---

## Components

| Component          | Document                                                         | Purpose                                        |
| ------------------ | ---------------------------------------------------------------- | ---------------------------------------------- |
| **Runtime System** | [Runtime System](./components/runtime-system/README.md)          | Policy runtime, execution, and CLI             |
| Inference Core     | [Inference Core](./components/inferencekit.md)                   | Domain‑agnostic inference layer                |
| Robot Interface    | [Robot Interface](./components/robot-interface.md)               | Robot Protocol and hardware integration        |
| Camera Interface   | [Camera Interface](./components/camera-interface.md)             | Capture API and camera backends                |
| Benchmarking       | [Benchmarking API](./components/benchmarking.md)                 | Benchmark protocols + runner                   |

## Integrations

| Integration | Document                                         | Purpose                                |
| ----------- | ------------------------------------------------ | -------------------------------------- |
| LeRobot     | [LeRobot Integration](./integrations/lerobot.md) | Loader integration for LeRobot exports |

---

## Document Structure

```text
docs/design/
├── README.md
├── architecture/
│   ├── strategy.md
│   └── architecture.md
├── components/
│   ├── runtime-system/
│   │   ├── README.md
│   │   ├── policy_runtime_design.md
│   │   ├── policy_server_design.md
│   │   ├── design_review_summary.md
│   │   ├── design_review_deck.md
│   │   └── inference_comparison_report.md
│   ├── inferencekit.md
│   ├── robot-interface.md
│   ├── camera-interface.md
│   └── benchmarking.md
├── integrations/
│   └── lerobot.md
└── packaging/
    └── physical-ai-two-repo-options.md
```

---

## Reading Guide

1. **[Strategy](./architecture/strategy.md)** — baseline architecture and scope
2. **[Architecture](./architecture/architecture.md)** — runtime architecture and boundaries
3. **[Runtime System](./components/runtime-system/README.md)** — policy runtime design and phases
4. **[Inference Core](./components/inferencekit.md)** — inference foundation

---

_Last Updated: 2026-05-06_
