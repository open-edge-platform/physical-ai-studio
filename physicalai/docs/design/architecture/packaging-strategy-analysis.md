# Physical AI: Packaging, Naming, and Market Strategy

> **Date**: 2026-03-02 | **Status**: Decision document — draft for discussion

---

## Core Identity

We are an edge team. That single fact anchors every naming, packaging, and positioning decision in
this document.

```text
pip install physicalai        → edge runtime, no torch, runs exported models anywhere
pip install physicalai-train  → full robotics ML SDK: policies, training, export, benchmarking
```

**Hardware story**: agnostic by design. Works on CUDA, AMD, Intel XPU, and CPU. Works best on
Intel because of OpenVINO advantages — but nobody is locked in. TensorRT, ExecuTorch, and ONNX
Runtime are all supported. This is a strength, not a caveat.

---

## What We Have

Two repos, two PyPI packages, one shared `physicalai.*` namespace.

**`physicalai-train`** (ships from `physical-ai-studio`) is a complete robotics ML SDK:

- Export-friendly, hardware-agnostic reimplementations of SOTA policies: ACT, Pi0, Pi0.5, GR00T,
  SmolVLA — with XAI, LoRA, KV-cache, and no Flash Attention hard dependency
- Thin PyTorch Lightning wrappers around 10+ LeRobot policies — zero reimplementation, full
  compatibility with existing LeRobot workflows
- Export to four backends: ONNX, OpenVINO, Torch Export IR, plain Torch
- Lightning `Trainer` subclass, benchmark runners (LIBERO, PushT), gym wrappers, rollout eval

**`physicalai`** (ships from `physicalai` repo) is the lightweight edge runtime:

- `InferenceModel`: auto-detects backend from file extension, manages action queues, runs
  inference without torch
- Multi-backend adapters: OpenVINO, ONNX Runtime, Torch Export IR, TensorRT, ExecuTorch
- Camera and robot interfaces, NumPy-only benchmark runner

**Current state**: the `physicalai` repo is scaffolding — `InferenceModel` still lives in
`physical-ai-studio`. Moving it is the first priority action (see Repo Structure).

---

## Why Two Packages, Not `physicalai[train]`

The extras approach (`pip install physicalai[train]`) was evaluated and rejected. The reason is
structural, not stylistic.

The SDK has 47 files with top-level `import torch` across policies, data, gyms, eval, devices, and
transforms. Making a lightweight extras install safe would require lazy-importing torch in every
one of those files — a permanent maintenance burden where one accidental module-level import
silently breaks the edge install, with CI as the only guard.

The PEP 420 namespace split makes the boundary **structural**:

- `physicalai` cannot import torch — torch is not in its dependency tree
- `physicalai-train` can import anything it needs
- No discipline required. The constraint is enforced by the package boundary, not by developers.

Despite being two packages, the import namespace is unified. Users see a single `physicalai.*`
namespace regardless of which package provides the submodule:

```python
from physicalai.inference import InferenceModel  # from physicalai
from physicalai.train import Trainer             # from physicalai-train
from physicalai.policies import ACT              # from physicalai-train
```

This works via PEP 420 implicit namespace packages — no `__init__.py` at `src/physicalai/` in
either distribution. A proof-of-concept validates this across single-repo and two-repo scenarios
(22/22 tests pass).

---

## Market Context

### The robotics ML landscape

| Project              | Training         | Export             | Edge deployment             | Hardware       |
| -------------------- | ---------------- | ------------------ | --------------------------- | -------------- |
| **LeRobot**          | ✓ PyTorch native | ✗ none             | ✗ none                      | CUDA-first     |
| **Isaac Lab**        | ✓ Isaac          | Limited            | Isaac Runtime only          | NVIDIA only    |
| **ROS2**             | ✗                | ✗                  | ✓ via nodes                 | Agnostic       |
| **ONNX Runtime**     | ✗                | ✗                  | ✓ general purpose           | Agnostic       |
| **OpenVINO**         | ✗                | ✓ model conversion | ✓ Intel-optimized           | Intel + others |
| **TorchServe**       | ✗                | ✗                  | ✓ HTTP serving              | CUDA-first     |
| **physicalai**       | ✗                | ✗                  | ✓ multi-backend, edge-first | Agnostic       |
| **physicalai-train** | ✓ Lightning      | ✓ 4 backends       | ✓ InferenceModel            | Agnostic       |

### The gap we fill

**LeRobot has no export path.** Every LeRobot user eventually hits the same wall:

```text
Train with LeRobot                          ✓
Export to ONNX / OpenVINO / TensorRT        ✗  dead end
Deploy on non-CUDA hardware                 ✗  dead end
```

Nobody else fills steps 2 and 3 specifically for learned robotics policies. That is the strategic
opening.

### Where we fit

We are not competing with LeRobot on training. We complete the pipeline they leave unfinished:

```text
Data collection → LeRobot training → physicalai-train export → physicalai edge runtime
```

The first-party policies (ACT, Pi0, Pi0.5, GR00T, SmolVLA) are not LeRobot alternatives. They
exist because LeRobot's implementations are not designed for ONNX/OpenVINO export or
hardware-agnostic deployment. Export-friendly, hardware-agnostic implementations were a
requirement from day one. Future roadmap policies will diverge further — architectures LeRobot
will not have.

---

## User Stories

### 1 — The LeRobot Deployer (primary funnel)

> "I trained an ACT policy with LeRobot. Now I need to run it on an edge device without a GPU.
> I cannot ship a CUDA environment to production."

1. Trains with LeRobot on a CUDA workstation
2. Hits the export wall — LeRobot has no export path
3. Finds physicalai via LeRobot docs (after the upstream export PR merges) or via search
4. `pip install physicalai` on the edge device — no torch, lightweight
5. Exports the model once with `physicalai-train` on the training machine
6. Runs on edge: `physicalai run --model ./exports/act_policy --robot robot.yaml`

**Entry point**: `pip install physicalai`
**What unblocks this user**: the upstream LeRobot export PR + clear deployment docs
**What they never need to know**: `physicalai-train` exists

---

### 2 — The LeRobot Power User

> "I use LeRobot for training. I need multi-GPU via Lightning, OpenVINO export for production,
> and attention visualisation to debug my ACT policy."

1. Already using LeRobot, hits its limits: no Lightning, no export, no XAI
2. Finds `physicalai-train` — wraps existing LeRobot policies, no retraining needed
3. `LeRobotPolicy("act", ...)` + `Trainer` — drop-in Lightning integration
4. `policy.export(output_path, backend="openvino")` — exports to production backend
5. Deploys with `physicalai` on edge

**Entry point**: `pip install physicalai-train`
**Key message**: works with your existing LeRobot workflow, no retraining required
**What they gain**: Lightning multi-GPU, 4 export backends, XAI, benchmark runners

---

### 3 — The Edge OEM / Hardware Deployer

> "We build industrial robots. Our ML team sends us trained model files. We need deterministic
> inference latency on edge hardware. We do not want torch in production."

1. ML team trains and exports using any framework (physicalai-train, LeRobot, custom)
2. OEM receives an exported model directory
3. `pip install physicalai` — no torch, no Lightning, numpy + runtime only
4. `model = InferenceModel("./exported_policy")` — backend auto-detected from file extension
5. `action = model.select_action(observation)` — returns numpy array, no torch in the loop

**Entry point**: `pip install physicalai`
**Key message**: one install, any exported model, any backend
**What they gain**: torch-free production runtime, multi-backend fallback, action queue
management, safety and logging callbacks

---

### 4 — The Robotics ML Researcher

> "I want Pi0.5 with LoRA fine-tuning on my custom dataset. LeRobot does not have Pi0.5.
> I also need to export the best checkpoint for edge deployment."

1. Searching for a Pi0.5 implementation with export support
2. Finds `physicalai-train` first-party policies
3. `pip install physicalai-train[pi0]`
4. `Pi05(variant="pi05", lora_rank=16)` + `Trainer` — trains with Lightning
5. `policy.export(output_path, backend="openvino")` — exports best checkpoint
6. Deploys with `physicalai` or runs benchmarks

**Entry point**: `pip install physicalai-train[pi0]`
**Key message**: export-friendly, hardware-agnostic SOTA implementations
**What they gain**: Pi0.5, LoRA, KV-cache denoising loop, XAI — capabilities LeRobot does not
ship
**Positioning discipline**: never "better than LeRobot's ACT" — always "export-friendly,
hardware-agnostic, with XAI"

---

### 5 — The Studio Application User

> "I am a robotics engineer, not an ML engineer. I want a UI to manage training runs and
> deploy to my robot fleet."

1. Deploys Physical AI Studio via Docker or npm
2. Configures and launches training jobs from the React UI
3. Studio calls `physicalai-train` under the hood
4. Exports models from the UI
5. Deploys via `physicalai` on edge devices

**Entry point**: Docker / npm (not PyPI)
**Key message**: complete platform — UI + training SDK + edge runtime, no Python required

---

## LeRobot Strategy

### Positioning: complementary, not competitive

LeRobot is the training standard. We do not compete on training. We provide what LeRobot lacks:
export, hardware-agnostic deployment, and a production runtime.

**Message hierarchy for LeRobot users — in order:**

1. "physicalai runs your LeRobot models at the edge"
2. "physicalai-train wraps LeRobot with Lightning and adds export"
3. Never lead with: "we have ACT, Pi0, GR00T too"

The first-party policies are a technical necessity — LeRobot's implementations are not designed
for ONNX/OV export or Intel XPU. They are not the pitch to LeRobot users.

### The upstream PR is the highest-leverage action

Getting the export PR merged into LeRobot upstream is worth more than any other marketing action:

- Every LeRobot README, tutorial, and HF notebook covering deployment will reference `physicalai`
- LeRobot maintainers become advocates, not bystanders
- The manifest format becomes a shared standard, not a competing one
- The entire LeRobot user base gains awareness of physicalai without further effort

Everything else is secondary to that relationship.

### Actions ranked by strategic value

| Action                                                  | Effort                       | Impact                                     |
| ------------------------------------------------------- | ---------------------------- | ------------------------------------------ |
| Get export PR merged upstream                           | High (LeRobot team decision) | Highest — we become their deployment story |
| Contribute export format docs to LeRobot                | Medium                       | High — establishes us as collaborator      |
| Add physicalai to LeRobot README as deployment option   | Low (one PR)                 | Medium — permanent visibility              |
| Publish LeRobot → physicalai tutorial on Hugging Face   | Low                          | Medium — direct user funnel                |
| Lead physicalai-train docs with "works with LeRobot"    | Very low                     | Medium — captures searching users          |
| Publish benchmark: LeRobot model on edge via physicalai | Medium                       | Medium — concrete proof point              |

### What not to do

- Do not market first-party policies as alternatives to LeRobot's implementations
- Do not claim "better training" — that signals competition regardless of intent
- Do not ignore LeRobot in docs — silence reads as competition too
- Do not go quiet after the upstream PR — maintain the relationship actively

---

## Naming Decisions

### Prime name: who owns `physicalai`?

The prime name is the most valuable asset on PyPI. The question is whether it should mean "edge
runtime" or "full pipeline."

| Configuration | Runtime package   | SDK package        | Prime name means |
| ------------- | ----------------- | ------------------ | ---------------- |
| A (current)   | `physicalai`      | `physicalai-train` | Edge runtime     |
| B (flipped)   | `physicalai-edge` | `physicalai`       | Full pipeline    |

**Recommendation: Option A** — `physicalai` = edge runtime.

Rationale:

1. **Accidental installs are asymmetric.** If a deployer accidentally installs the full SDK on
   edge, they get torch, Lightning, and 2GB+ of dependencies on a constrained device — a real
   problem. If a researcher accidentally installs only the runtime, they get a clear import error
   and `pip install physicalai-train` fixes it in 5 seconds. Option A's failure mode is benign;
   Option B's is expensive.

2. **The prime name sets brand perception.** `pip install physicalai` pulling in torch makes us
   look like another training framework. The whole "we complete LeRobot's pipeline" positioning
   depends on the first touch being lightweight. If `physicalai` means "everything including
   torch," we lose the differentiation in the first 10 seconds.

3. **Deployers are harder to reach.** Researchers find packages through papers, Twitter, and
   HuggingFace. Deployers find packages through pip, READMEs, and Stack Overflow. The prime name
   matters more for the audience with less discovery surface.

4. **`physicalai-edge` is fine but unnecessary.** It is the best of the alternative runtime
   names — explicit, self-explanatory, not too narrow. But it only becomes necessary if we need
   the prime name for something else, and there is no compelling case for that.

If the prime name must change in the future, `physicalai-edge` is the fallback for the runtime:

| Edge runtime name      | Pros                                                  | Cons                                                    |
| ---------------------- | ----------------------------------------------------- | ------------------------------------------------------- |
| `physicalai` (runtime) | Strongest brand recall; shortest; cleanest for deploy | Constrains future if prime name must mean full pipeline |
| `physicalai-edge`      | Explicit positioning; clear audience                  | "Edge" can feel narrow for server inference; longer     |
| `physicalai-deploy`    | Signals production; aligns with export + deploy story | "Deploy" implies infra tooling, not runtime API         |
| `physicalai-inference` | Precise about function; hardware-neutral              | Longer; sounds like generic model serving               |

### SDK name: `physicalai-train` vs alternatives

`physicalai-train` undersells the scope of what the package is. The SDK contains policies, export,
eval, benchmarks, gym wrappers, CLI, and config — "train" covers roughly 30% of the
functionality. This analysis evaluates alternatives.

| Name                  | Scope accuracy | Repo-agnostic | LeRobot signal | Survives Phase 2 | User clarity                         |
| --------------------- | -------------- | ------------- | -------------- | ---------------- | ------------------------------------ |
| `physicalai-train`    | Low (30%)      | Yes           | Low            | Yes              | Misleads about scope                 |
| `physicalai-sdk`      | High           | Yes           | Medium         | Yes              | Generic but accurate                 |
| `physicalai-studio`   | Low            | No            | Low            | **No**           | Implies GUI, not Python library      |
| `physicalai-robotics` | Medium         | Yes           | Medium-high    | Yes              | Vague — doesn't say what it does     |
| `physicalai-ml`       | Medium         | Yes           | Medium         | Yes              | Doesn't capture export or benchmarks |

**Recommendation: Keep `physicalai-train` now. Rename to `physicalai-sdk` at Phase 2.**

Rationale:

1. **`physicalai-train` is the safest name during the LeRobot relationship-building period.** It
   sounds like a training utility, not a competing framework. After the relationship is cemented,
   renaming to `physicalai-sdk` is a non-event.

2. **`physicalai-sdk` is the most accurate long-term name.** It is honest about scope, follows
   convention (`aws-sdk`, `google-cloud-sdk`), and pairs cleanly with the runtime: you develop
   with the SDK, you deploy with the runtime.

3. **Renaming now means renaming twice.** Once now, once at Phase 2 consolidation. Do it once.
   Until then, fix the `pyproject.toml` description — that is the immediate action:

   ```toml
   # current — inaccurate
   description = "A package for training vision-action models"

   # correct
   description = "Robotics ML SDK: policies, training, export, and deployment for physical AI"
   ```

4. **The transition at rename time is clean.** We could publish `physicalai-sdk` as the new
   package, then make `physicalai-train` a thin metapackage that depends on `physicalai-sdk`.
   Existing installs would keep working. Docs point to the new name. No broken environments.

---

## Repo Structure

### Where the SDK lives: Studio vs physicalai repo

The SDK (`physicalai-train`) currently ships from `physical-ai-studio`. The question is whether
to move it to the `physicalai` repo now, later, or never.

**Option A: Keep SDK in `physical-ai-studio`**

What the LeRobot community sees:

- `physicalai` repo: inference, cameras, robots, benchmarks. Clearly a deployment tool. No policy
  implementations. No training code. Nothing that overlaps with LeRobot.
- `physical-ai-studio` repo: a product — an application with a UI that includes a Python library.
  The policies and training code are nested inside `library/`. The README leads with the Studio
  application, not the SDK.

Competition signal: **low**. The policies and training code are inside a product repo. A LeRobot
maintainer browsing GitHub sees a deployment studio, not a competing ML framework. If they dig
into `library/src/physicalai/policies/`, they find the reimplementations — but context matters.
Finding them inside a product is very different from finding them as the headline feature of a
standalone SDK repo.

**Option B: Move SDK to `physicalai` repo**

What the LeRobot community sees:

- `physicalai` repo: contains `packages/physicalai/` (runtime) AND `packages/physicalai-train/`
  (full SDK with ACT, Pi0, GR00T, SmolVLA, LeRobot wrappers, Trainer, export). The repo README
  must describe both packages. The policies become top-level visible.

Competition signal: **high**. Even if the README says "complementary to LeRobot," the repo
structure says "we have our own ACT, our own Pi0, our own training pipeline." A LeRobot
maintainer sees a monorepo with policy reimplementations at the top level. That reads as
competition regardless of intent.

**Recommendation: Option A — keep SDK in Studio until the LeRobot relationship is secured.**

The engineering cost is small (two repos, slightly more CI complexity). The positioning benefit is
large: it keeps the `physicalai` repo clean and unambiguously a deployment tool, buries the policy
reimplementations inside a product, and makes the upstream export PR easier to land. It also gives
an honest answer when a LeRobot maintainer asks "are you building a competitor?" — "No, our repo
is a deployment runtime. The policies are part of our Studio product."

### Phase 1 — Now

```text
physicalai repo  (private → going public)
├── pyproject.toml              name = "physicalai"
└── src/physicalai/
    ├── inference/               InferenceModel  ← move here from studio now
    ├── capture/                 camera interfaces
    ├── robot/                   robot ABC
    └── benchmark/               NumPy-only benchmark runner

physical-ai-studio repo  (public, Apache 2.0)
├── application/                 FastAPI + React
└── library/
    ├── pyproject.toml           name = "physicalai-train"
    └── src/physicalai/
        ├── policies/            ACT, Pi0, Pi0.5, GR00T, SmolVLA + LeRobot wrappers
        ├── data/                DataModule, LeRobot integration
        ├── export/              ONNX, OpenVINO, Torch Export IR
        ├── train/               Lightning Trainer
        └── eval/ benchmark/ gyms/ cli/ config/
```

`InferenceModel` has no torch dependency in its core interface today. Moving it to `physicalai`
makes the runtime real rather than scaffolding, and directly unblocks user stories 1 and 3.

### Phase 2 — After LeRobot relationship is secured (pending approval)

All Python library code moves to `physical-ai`. Studio becomes a pure application. SDK is renamed
from `physicalai-train` to `physicalai-sdk`.

```text
physical-ai repo  (public)
├── packages/
│   ├── physicalai/              runtime
│   └── physicalai-sdk/          full SDK (formerly physicalai-train)
└── (all Python library code lives here)

physical-ai-studio repo  (public, Apache 2.0)
├── application/                 FastAPI + React only
└── library/                     frozen — retained for OSS users, no new releases

PyPI:
  physicalai-sdk    → the real package
  physicalai-train  → metapackage, depends on physicalai-sdk (backwards compat)
```

**Phase 2 gate conditions** — consolidate only after:

1. The LeRobot export PR is merged upstream
2. physicalai appears in LeRobot's docs or README as the deployment path
3. At least one LeRobot tutorial or notebook references physicalai for deployment
4. Enough time has passed that "physicalai = deployment for LeRobot" is the default community
   perception

Once the complementary positioning is cemented, consolidating code into one repo is an internal
logistics decision that will not change external perception. Before that point, it is a risk.

Phase 1 is a stable long-term fallback if Phase 2 approval is not granted or the gate conditions
are not yet met. The PEP 420 namespace split works correctly in both phases.

### Repo and org names

| Repo      | Current                                 | Issue                                                          | Recommendation                                    |
| --------- | --------------------------------------- | -------------------------------------------------------------- | ------------------------------------------------- |
| Runtime   | `openvinotoolkit/physicalai`            | Intel org signals lock-in, contradicts hardware-agnostic story | Ideally move to a neutral org before going public |
| SDK + App | `open-edge-platform/physical-ai-studio` | "Studio" misleads PyPI users until Phase 2                     | Keep — becomes accurate once Studio is app-only   |

---

## Decision Summary

| Decision                                     | Recommendation                                 | Urgency | Requires approval? |
| -------------------------------------------- | ---------------------------------------------- | ------- | ------------------ |
| `physicalai` = edge runtime, prime name      | Yes — optimizes for primary audience           | —       | No                 |
| PEP 420 namespace split over extras          | Yes — structural enforcement                   | —       | No                 |
| Keep SDK in `physical-ai-studio` for now     | Yes — reduces LeRobot competition signal       | —       | No                 |
| Move `InferenceModel` to `physicalai` now    | Yes — makes runtime real                       | High    | No                 |
| Fix `physicalai-train` pyproject description | Yes — one line                                 | High    | No                 |
| Get LeRobot export PR merged                 | Yes — highest strategic leverage               | High    | No (upstream)      |
| Keep name `physicalai-train` for now         | Yes — safest during LeRobot relationship phase | —       | No                 |
| Rename to `physicalai-sdk` at Phase 2        | Yes — accurate, clean transition via metapkg   | Low     | No                 |
| Consolidate libraries into `physicalai` repo | Yes — after LeRobot relationship is secured    | Medium  | Yes                |
| Move runtime repo out of `openvinotoolkit`   | Yes — before going public                      | Medium  | Yes                |

---

_2026-03-02_
