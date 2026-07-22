---
marp: true
theme: default
paginate: true
backgroundColor: #fff
---

# Geti Action Architecture Proposal

**Library-First Design & Three-Layer Deployment Stack**

---

## What We're Proposing

Two architectural decisions that simplify our codebase and clarify our product story:

1. **Library-first** — The library owns all core components.
   The application becomes UI/orchestration only.

2. **Three-layer deployment** — Clean separation between
   generic inference, robotics, and the deployment product.

---

## Today's Problem

Some core workflows exist only in the **application** and are not mirrored in the **library**. This breaks GUI/CLI parity:

| Scenario                       | Problem                                                 |
| ------------------------------ | ------------------------------------------------------- |
| Teleoperation via CLI          | Not possible — teleop lives in the application          |
| Data collection via CLI        | Not possible — data collection lives in the application |
| Robot/camera access in scripts | Requires application build/run                          |
| Consistent APIs across UI/CLI  | Drift between app logic and library logic               |

**Result: the library trains models but core runtime workflows are gated by the application.**

---

## Part 1: Library-First Architecture

```
┌─────────────────────────────────────────────────────────┐
│                 APPLICATION (Studio)                    │
│                                                         │
│   React UI  │  FastAPI Backend  │  Workflows  │  Calib  │
│                                                         │
│      Orchestrates library components — no core logic    │
└──────────────────────────┬──────────────────────────────┘
                           │ imports
                           ▼
┌─────────────────────────────────────────────────────────┐
│                   LIBRARY (getiaction)                  │
│                                                         │
│  Robots │ Cameras │ Teleop │ Data │ Policies │ Infer    │
│  Training │ Export │ Eval │ CLI │ Config                │
└─────────────────────────────────────────────────────────┘
         │              │               │
         ▼              ▼               ▼
   Robot SDKs      Camera SDKs    LeRobot / HF Hub
 (optional extras)
```

---

## What the Library Owns

| Component                           | Library | Application |
| ----------------------------------- | :-----: | :---------: |
| Robot ABC + SDK drivers             |  **✓**  |   imports   |
| Camera ABC + SDK drivers            |  **✓**  |   imports   |
| Teleoperation sessions              |  **✓**  |   imports   |
| Data collection / episodes          |  **✓**  |   imports   |
| Policies (ACT, Pi0, SmolVLA, GR00T) |  **✓**  |      —      |
| Inference, Training, Export         |  **✓**  |    uses     |
| CLI (`getiaction`)                  |  **✓**  |      —      |
| Calibration, GUI, Workflows         |    —    |    **✓**    |

---

## Current State → Target State

```
  CURRENT                              TARGET
  ────────────────────                 ────────────────────

  application/                         application/
    robots/  ← drivers here              (imports from library)
    workers/ ← camera, teleop            (imports from library)

  library/                             library/
    policies/                            robots/    ← MOVED
    inference/                           cameras/   ← MOVED
    data/                                teleop/    ← MOVED
                                         data/      (already here)
                                         policies/  (already here)
                                         inference/ (already here)
```

---

## Why Library-First Matters

| Benefit                    | Impact                                                                   |
| -------------------------- | ------------------------------------------------------------------------ |
| **Edge deployment**        | `pip install getiaction[inference]` + 10 lines of Python. No web server. |
| **Faster adoption**        | Researchers start with `pip install`, not a full stack                   |
| **Single source of truth** | No version skew between library and application                          |
| **Consistent API**         | Same Python API from CLI, scripts, and application                       |
| **Testable in isolation**  | Unit test hardware interfaces without application                        |

---

## Modular Installation

Hardware SDKs are optional extras — core stays lightweight:

```bash
pip install getiaction                # Core API base (only pyyaml, numpy, PIL)
pip install getiaction[lerobot]       # LeRobot
pip install getiaction[trossen]       # Trossen robots
pip install getiaction[ur]            # Universal Robots
pip install getiaction[realsense]     # Intel RealSense cameras
```

<!-- v0.1.0 scope: **SO101 + Trossen** only. Others via extras post-v0.1.0. -->

---

## Part 2: Deployment Stack

After data collection and training, you have an exported model.
**How do you run it on a robot?**

Three clean layers, each with a single responsibility:

---

## Three-Layer Architecture

```
┌──────────────────────────────────────────────────────┐
│          physical-ai-framework (shell)               │
│                                                      │
│  Deployment product. CLI + packaging + docs.         │
│  Can use getiaction, LeRobot, or other frameworks.   │
│  phyai run --config deploy.yaml                      │
└────────────────────────┬─────────────────────────────┘
                         │ depends on
                         ▼
┌──────────────────────────────────────────────────────┐
│               getiaction (library)                   │
│                                                      │
│  Robotics inference: policies, action chunking,      │
│  robot control, preprocessing, LeRobot compat.       │
└────────────────────────┬─────────────────────────────┘
                         │ depends on
                         ▼
┌──────────────────────────────────────────────────────┐
│              inferencekit (core)                     │
│                                                      │
│  Domain-agnostic: InferenceModel, RuntimeAdapter,    │
│  OpenVINO / ONNX / TensorRT / Torch backends.        │
│  No robotics. Reusable across all ML domains.        │
└──────────────────────────────────────────────────────┘
```

---

## Layer Responsibilities

| Layer                     | Owns                                                         | Does NOT own                     |
| ------------------------- | ------------------------------------------------------------ | -------------------------------- |
| **inferencekit**          | Model loading, backend adapters, runners, callbacks          | Robotics, policies, domain logic |
| **getiaction**            | Robot/camera APIs, policies, action chunking, LeRobot compat | Deployment CLI, packaging        |
| **physical-ai-framework** | CLI (`phyai`), packaging, deployment docs                    | Core implementations             |

**Dependencies flow upward only.** Never the reverse.

---

## physical‑ai‑framework as a Façade

A **façade** is a thin front‑door API that re‑exports capabilities without re‑implementing them.

To keep the product self‑contained without duplicating logic:

- **Re-export stable APIs** from getiaction/inferencekit (InferenceModel, Robot, Camera)
- **Provide a small, stable surface** optimized for deployment
- **Host adapters** for other frameworks (LeRobot, future integrations)

Result: a single product entrypoint, multiple backends, no core duplication.

```
          physical‑ai‑framework (façade)
                     │
     ┌───────────────┼────────────────┐
     │               │                │
 getiaction      lerobot         future
   adapter        adapter        adapters
     │               │                │
  getiaction      lerobot          other
   + infkit        runtime       frameworks
```

---

## Why Interfaces Stay in getiaction

Moving robot/camera/inference interfaces into the deployment shell creates real risks:

- **Circular dependencies** (getiaction needs those APIs for training/eval)
- **DX regression** (researchers must install the deployment product)
- **Shell stops being thin** (it becomes the primary library)
- **Release coupling** (core APIs tied to deployment cadence)

**Preferred approach:** keep interfaces in getiaction and expose a stable, versioned API that physical‑ai‑framework wraps.

---

## Why Three Layers?

| Layer                     | Rationale                                                                                           |
| ------------------------- | --------------------------------------------------------------------------------------------------- |
| **inferencekit**          | Reusable beyond robotics (CV, NLP, anomaly detection). Keeps domain logic out of generic inference. |
| **getiaction**            | Adds robotics intelligence on top. Action chunking, policy runners, hardware control.               |
| **physical-ai-framework** | Product boundary for leadership. Marketable independently. No engineering duplication.              |

---

## The User Experience (Inference + Deployment)

**Inference API** (library):

```python
from getiaction.inference import InferenceModel
from getiaction.robots import SO101
from getiaction.cameras import Webcam

robot = SO101.from_config("robot.yaml")
camera = Webcam.from_config("camera.yaml")
policy = InferenceModel("./exports/act_policy")

with robot, camera:
    obs = robot.get_observation(format="lerobot")
    obs["images"] = {"wrist": camera.read()}
    action = policy.select_action(obs)
    robot.send_action(action)
```

**Deployment CLI** (physical-ai-framework):

```bash
phyai run --model ./exports/act_policy --robot robot.yaml --episodes 10
phyai serve --model ./exports/act_policy --robot robot.yaml
```

---

## Key Decisions

| Decision              | Choice                  | Rationale                             |
| --------------------- | ----------------------- | ------------------------------------- |
| Core logic location   | Library                 | Single source of truth                |
| Application role      | UI + orchestration only | Library-first                         |
| Deployment packaging  | Thin shell repo         | Product boundary, no engineering risk |
| Robot/camera API home | Library                 | Edge deployment without web server    |
| Installation model    | Optional extras per SDK | Lightweight core                      |
| Dataset format        | LeRobot (v0.1.0)        | VLA use case requirement              |

---

## Guardrails

- No hardware SDKs in core install (optional extras only)
- No application dependency for edge deployment
- No core logic in physical-ai-framework (thin wrappers only)
- No version skew between library and deployment
- No heuristic robot detection

---

## Summary

|                          | Before                | After                       |
| ------------------------ | --------------------- | --------------------------- |
| **Robot control**        | Application only      | Library (+ app imports)     |
| **Edge deployment**      | Full stack required   | `pip install getiaction`    |
| **Inference stack**      | Monolithic            | 3 clean layers              |
| **Product story**        | Single repo           | Distinct deployment product |
| **Developer experience** | Start with full stack | Start with `pip install`    |

---

## Next Steps

1. Move robot/camera/teleop drivers from application → library
2. Define library public API surface (`getiaction.robots`, `getiaction.cameras`, `getiaction.teleop`)
3. Set up inferencekit as standalone package
4. Wire physical-ai-framework shell to depend on getiaction
5. Migrate application to import from library

---

<!-- _class: lead -->

# Questions?

Design docs: `docs/design/strategy.md`
