# Physical-AI Packaging Strategy

Two repositories. Two PyPI distributions. One namespace.

**Repos:**

- [`physical-ai`](https://github.com/openvinotoolkit/physical-ai) — runtime/inference library (currently private)
- [`physical-ai-studio`](https://github.com/open-edge-platform/physical-ai-studio) — training SDK + application (going public)

**PyPI distributions:**

- `physicalai` — lightweight runtime (inference, capture, robot, export, benchmark runner)
- `physicalai-train` — training SDK (policies, data, eval, gyms, benchmark presets)

Both distributions share the `physicalai` namespace via PEP 420 implicit namespace packaging.

---

## Constraints

- Only **two repos** available (no third repo option)
- **No circular dependencies** — dependency flows upward only
- **Lightweight base install** — `pip install physicalai` must not pull torch/lightning
- **No `__init__.py` at namespace root** — PEP 420 requirement for namespace split
- Core APIs return plain `dict`/`np.ndarray` only
- Studio is **not** a PyPI package — built via npm or Docker only
- Training SDK **is** published to PyPI as `physicalai-train`

---

## Phased Rollout

### Phase 1 — Open-Source Launch

Timeline: immediate (3 days).

`physical-ai-studio` goes public with the codebase renamed from `getiaction` to `physicalai`. All Python modules ship inside a single distribution: `physicalai-train`.

```
physical-ai-studio/                    # goes public
├── application/                       # FastAPI + React (not packaged)
└── library/
    ├── pyproject.toml                 # name = "physicalai-train"
    └── src/physicalai/
        ├── train/                     # Trainer, Lightning integration
        ├── policies/                  # ACT, Pi0, SmolVLA, GR00T
        ├── data/                      # DataModule, Observation
        ├── benchmark/                 # Benchmarking
        ├── eval/                      # Rollout evaluation
        ├── gyms/                      # Sim environments
        ├── inference/                 # InferenceModel (temporary)
        ├── export/                    # ONNX, OpenVINO export
        ├── transforms/                # Shared transforms
        ├── cli/                       # CLI
        ├── config/                    # Configuration
        └── devices/                   # Device utilities
        # NO physicalai/__init__.py

physical-ai/                           # stays private/empty
```

**Install experience:**

```bash
pip install physicalai-train           # gets everything
```

```python
from physicalai.train import Trainer
from physicalai.policies import ACT
from physicalai.inference import InferenceModel
```

**Why this works:** Shipping everything as `physicalai-train` avoids claiming the `physicalai` PyPI name prematurely. The `physicalai` name is reserved for the lightweight runtime that ships from `physical-ai` once that repo goes public. PEP 420 compliance (no namespace `__init__.py`) is enforced from day one so that Phase 2 is a clean split, not a rewrite.

---

### Phase 2 — Runtime Extraction

Timeline: when `physical-ai` repo goes public.

Extract runtime modules from studio into the `physical-ai` repo. Publish `physicalai` (lightweight) from there. `physicalai-train` stays in studio and declares `physicalai>=0.1.0` as a dependency.

```
physical-ai/                           # now public
├── pyproject.toml                     # name = "physicalai"
└── src/physicalai/
    ├── inference/                     # InferenceModel
    ├── capture/                       # Camera interfaces
    ├── robot/                         # Robot ABC
    ├── benchmark/                     # BenchmarkRunner, protocols (numpy-only)
    ├── export/                        # Model export
    └── transforms/                    # Shared transforms
    # NO physicalai/__init__.py

physical-ai-studio/                    # training stays here
├── application/                       # FastAPI + React
└── library/
    ├── pyproject.toml                 # name = "physicalai-train"
    │                                  # depends on: physicalai>=0.1.0
    └── src/physicalai/
        ├── train/
        ├── policies/
        ├── data/
        ├── eval/
        ├── gyms/
        ├── cli/
        └── config/
        # NO physicalai/__init__.py
```

**Install experience:**

```bash
pip install physicalai                 # runtime only — no torch
pip install physicalai-train           # auto-pulls physicalai
```

```python
# Runtime only
from physicalai.inference import InferenceModel
from physicalai.capture import Camera

# Training (requires physicalai-train)
from physicalai.train import Trainer
from physicalai.policies import ACT
```

**What moves:**

| Module        | From                 | To            |
| ------------- | -------------------- | ------------- |
| `inference/`  | `physical-ai-studio` | `physical-ai` |
| `capture/`    | `physical-ai-studio` | `physical-ai` |
| `robot/`      | `physical-ai-studio` | `physical-ai` |
| `benchmark/`  | `physical-ai-studio` | `physical-ai` |
| `export/`     | `physical-ai-studio` | `physical-ai` |
| `transforms/` | `physical-ai-studio` | `physical-ai` |

**Prerequisite:** Clean up cross-module imports. Currently `inference/` imports from `data/`, `export/`, and `policies/` — these dependencies must be broken before extraction.

---

### Phase 3 — Training Consolidation (Pending Stakeholder Alignment)

Timeline: requires executive approval.

Move training modules from studio into `physical-ai`. Both distributions publish from a single repo. Studio becomes a pure application (FastAPI + React).

```
physical-ai/                           # owns all Python code
├── packages/
│   ├── physicalai/                    # runtime dist
│   │   ├── pyproject.toml
│   │   └── src/physicalai/
│   │       ├── inference/
│   │       ├── capture/
│   │       ├── robot/
│   │       ├── benchmark/
│   │       ├── export/
│   │       └── transforms/
│   │       # NO physicalai/__init__.py
│   └── physicalai-train/              # training dist
│       ├── pyproject.toml
│       └── src/physicalai/
│           ├── train/
│           ├── policies/
│           ├── data/
│           ├── eval/
│           ├── gyms/
│           ├── cli/
│           └── config/
│           # NO physicalai/__init__.py

physical-ai-studio/                    # pure application
├── application/
│   ├── backend/                       # FastAPI
│   └── ui/                            # React
└── pyproject.toml                     # depends on physicalai-train
```

**Why this is the end-state:** Library code belongs in a library repo. Studio becomes a consumer, not a publisher. One repo for all Python packaging simplifies CI, versioning, and release coordination.

**Why it needs approval:** Moving training out of studio changes repo ownership boundaries. This is an organizational decision, not a technical one. If rejected, Phase 2 is a stable long-term state — training stays in studio, runtime lives in `physical-ai`, namespace split works either way.

---

## Package Architecture

### Namespace Split

Both distributions contribute subpackages under the `physicalai` namespace. Neither distribution owns the namespace root.

```
physicalai/                            # PEP 420 namespace — no __init__.py
├── inference/    → physicalai dist    # runtime
├── capture/      → physicalai dist
├── robot/        → physicalai dist
├── export/       → physicalai dist
├── transforms/   → physicalai dist
├── benchmark/    → physicalai dist    # evaluation mechanism (numpy-only)
├── train/        → physicalai-train   # training
├── policies/     → physicalai-train
├── data/         → physicalai-train
├── eval/         → physicalai-train   # rollout loop, video recording (torch)
├── gyms/         → physicalai-train   # simulation environments (heavy deps)
├── cli/          → physicalai-train
└── config/       → physicalai-train
```

Training subpackages (`train/`, `policies/`, `data/`, etc.) are siblings under `physicalai`, not nested under `physicalai.train.*`. This keeps imports flat and natural.

### Dependency Direction

```
physicalai-train  →  physicalai  →  (numpy, opencv, etc.)
     │                    │
     │                    └── no torch, no lightning
     └── torch, lightning, policies, data
```

`physicalai-train` depends on `physicalai`. Never the reverse.

### Install Matrix

| User          | Install                        | Gets                                               |
| ------------- | ------------------------------ | -------------------------------------------------- |
| Edge deployer | `pip install physicalai`       | Inference, camera, robot, export, benchmark runner |
| ML researcher | `pip install physicalai-train` | Everything (auto-pulls physicalai)                 |
| Studio user   | Docker / npm build             | Application with physicalai-train baked in         |

### Import Examples

```python
# Runtime only (pip install physicalai)
from physicalai.inference import InferenceModel
from physicalai.capture import RealSenseCamera
from physicalai.robot import Robot
from physicalai.benchmark import BenchmarkRunner, BenchmarkResults

# Training (pip install physicalai-train)
from physicalai.train import Trainer
from physicalai.policies import ACT
from physicalai.data import LeRobotDataModule
from physicalai.gyms import LiberoGym
```

### Critical Rule

**No `__init__.py` at `src/physicalai/` in either distribution.** This is the PEP 420 namespace mechanism. If either package adds a `physicalai/__init__.py`, the namespace breaks — only one distribution's subpackages will be visible.

### IDE Configuration (PEP 420 Namespace Packages)

**End users** (`pip install` into the same virtualenv): No configuration needed. Pyright/Pylance, PyCharm, and mypy correctly discover both distributions under the `physicalai` namespace. Go-to-definition, autocomplete, and type checking work out of the box.

**Phase 1 & Phase 3 developers** (single repo): No issues. All source files are in one workspace.

**Phase 2 developers** (two repos, editable installs): Requires a one-line IDE config so the type checker sees both source trees.

**VS Code (Pylance)**:

```json
// .vscode/settings.json
{
  "python.analysis.extraPaths": ["../physical-ai/library/src"]
}
```

**PyCharm**: Right-click the second repo's `src/` directory → Mark Directory As → Sources Root.

#### Known Issues & Mitigations

| Issue                                                                                                           | Impact                                                      | Status                                                                                         |
| --------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------- | ---------------------------------------------------------------------------------------------- |
| Pylance 2025.10.1 regression ([pylance-release#7737](https://github.com/microsoft/pylance-release/issues/7737)) | PEP 420 namespace detection broke temporarily               | **Fixed** in 2025.10.2                                                                         |
| Namespace packages flagged as untyped ([pyright#10203](https://github.com/microsoft/pyright/issues/10203))      | Pyright incorrectly required type stubs                     | **Fixed** April 2025                                                                           |
| `pkgutil.extend_path()` resolution ([pyright#2882](https://github.com/microsoft/pyright/issues/2882))           | Only first namespace instance resolved                      | **Does not affect us** — we use pure PEP 420, not `pkgutil`                                    |
| Mixed implicit/explicit namespace ([pyright#3430](https://github.com/microsoft/pyright/issues/3430))            | Mixing `__init__.py` and no-`__init__.py` breaks resolution | **Avoided** by our "no `__init__.py` at namespace root" rule                                   |
| Legacy editable installs ([pip#7265](https://github.com/pypa/pip/issues/7265))                                  | `setup.py develop` breaks namespace packages                | **Avoided** — use PEP 660 editable installs (`pip install -e .` with hatchling/setuptools ≥64) |

#### Why PEP 420 Over `pkgutil.extend_path()`

Google Cloud (`google.cloud.*`) and Azure SDK (`azure.*`) use `pkgutil.extend_path()` with explicit `__init__.py` files. Pyright only resolves the first namespace instance with this approach ([pyright#2882](https://github.com/microsoft/pyright/issues/2882) — closed "as designed"). Pure PEP 420 (no `__init__.py`) is explicitly supported by Pyright and avoids this limitation entirely.

---

## PoC Validation

A proof-of-concept validates PEP 420 namespace packaging across both single-repo and two-repo scenarios (22/22 tests pass). See `poc/` directory to reproduce.

---

## Risks

| Risk                                       | Mitigation                                                                          |
| ------------------------------------------ | ----------------------------------------------------------------------------------- |
| PEP 420 unfamiliarity                      | PoC validates both scenarios (see `poc/`); document the critical `__init__.py` rule |
| Accidental `__init__.py` at namespace root | CI lint rule: fail if `src/physicalai/__init__.py` exists                           |
| Cross-distribution import leaks            | CI import boundary check: `physicalai` dist must not import from training modules   |
| Phase 3 blocked by stakeholders            | Phase 2 is a stable long-term fallback                                              |
| Version coordination between two dists     | Pin `physicalai>=X.Y.Z` in `physicalai-train`; release runtime first                |
| `transforms/` ownership ambiguity          | Assign to `physicalai`; training dist uses it as a dependency                       |
| IDE support for Phase 2 multi-repo dev     | One-line `extraPaths` config for VS Code; Sources Root for PyCharm (documented)     |

---

## Alternatives Considered

The chosen approach (two separate distributions with PEP 420 namespace) is documented above. Two alternatives were evaluated and rejected:

### Option A — Single Package With Extras

Ship everything as one distribution. Use extras for heavy dependencies (`pip install physicalai[train]`).

```python
pip install physicalai              # runtime only
pip install physicalai[train]       # adds torch, lightning
```

**Why rejected:**

- Requires strict lazy-import discipline across the entire codebase
- Risk of accidental heavy imports at runtime — one `import torch` at module level breaks lightweight install
- CI must test both base and extras installs
- Current codebase has module-level torch imports in `data/`, `export/`, `policies/` — would need extensive refactoring to make extras safe
- No hard dependency boundary — discipline is the only guard

### Option C — Training Under Studio Branding

Keep training SDK in `physical-ai-studio` repo published under a different package name (not `physicalai.*`).

```bash
pip install physicalai              # runtime
pip install physical-ai-studio-sdk  # training
```

**Why rejected:**

- Training lives under "studio" branding, which is confusing — studio is a UI application
- Breaks the `physicalai.*` namespace unity — users need to know two unrelated package names
- ML researchers searching PyPI for "physicalai" won't discover the training SDK
- Studio repo mixes UI application and library concerns permanently

---

_Last Updated: 2026-02-20_
