# Strategy: Physical‑AI Runtime

This document defines the architecture and scope of the **physicalai runtime**. It is the top‑level blueprint for what belongs in this repo and how it relates to `physicalai-train` and Studio.

---

## Executive Summary

**physicalai** is the lightweight runtime distribution for physical‑AI inference and deployment. It owns:

- Inference runtime and CLI
- Camera and robot interfaces
- Benchmarking runner and protocols

Training, data collection, teleoperation, and policy implementations live in **physicalai-train** (Studio repo). The runtime never imports training code at install time.

**Key decisions:**

1. **Runtime-first packaging**: `physicalai` is a small, dependency‑light runtime. `physicalai-train` depends on it.
2. **Layered inference stack**: a domain‑agnostic inference core (`physicalai.inference`) sits inside the runtime and is reused by the runtime and external domain layers.

---

## Scope and Ownership

### In Scope (this repo)

- `physicalai.inference` — base inference core
- `physicalai.runtime` — runtime orchestration and CLI
- `physicalai.capture` — camera interface and backends
- `physicalai.robot` — robot interface and drivers
- `physicalai.benchmark` — evaluation runner and protocols (NumPy‑only)

### Out of Scope (Studio / physicalai-train)

- Teleoperation sessions
- Data collection / DatasetWriter
- Policies, training, and evaluation loops
- Simulation environments (gyms)

---

## Dependency Direction

```text
physicalai-train  →  physicalai  →  numpy, opencv, etc.
```

`physicalai` never imports training modules at install time. Heavy dependencies stay in `physicalai-train`.

---

## Runtime Architecture (High Level)

```text
physicalai (runtime)
├── runtime      # orchestration + CLI + config
├── inference    # domain‑agnostic core
├── capture      # camera interfaces
├── robot        # robot interfaces
├── benchmark    # benchmark runner + protocols
```

The runtime exposes a unified `InferenceModel` API, backed by the inference core. The runtime owns execution patterns (runners), device lifecycle, safety, and CLI wiring.

---

## Repository Boundaries

The **Studio repo** consumes the runtime but does not define runtime contracts. This repo is the canonical source of public runtime APIs.

See [Packaging Strategy](../packaging/physical-ai-two-repo-options.md) for the split plan and [Architecture](./architecture.md) for the full runtime architecture.

---

## Component References

- [Architecture](./architecture.md) — runtime orchestration and CLI
- [Inference Core](../components/inferencekit.md)
- [Robot Interface](../components/robot-interface.md)
- [Camera Interface](../components/camera-interface.md)
- [Benchmarking API](../components/benchmarking.md)

---

_Last Updated: 2026-02-23_
