# PhysicalAI Studio Q3 Strategy

This document is to align the team on the execution items of `physicalai-train` and `physicalai` runtime for Q3.

- Sources: [PAI Feature Tracker for Q3](https://intel-my.sharepoint.com/:x:/p/samet_akcay/IQDD_l82x1_fQ6HCf-bRDq5cAbJDy_bhaqAv7wJQdnR7vUE?e=CaF6My)
- Deadlines: **ICRA 16 Sep, ICLR 25 Sep, CVPR 14 Nov** or ArXiV in case we miss these previous deadlines.

## Goal

Maximize high-value VLA/WAM model coverage. Each model must be trained, benchmarked, exported, parity-tested, runtime-loaded, and deployed when hardware is available.

Existing loop:

```text
dataset -> Observation -> train -> benchmark -> export -> manifest -> InferenceModel.load -> PolicyRuntime -> robot
```

## Scope

In:

- Build P1/P2 VLA/WAM policies.
- Per-model benchmark coverage.
- Torch, ONNX, OpenVINO, ExecuTorch export where supported.
- PyTorch/export parity.
- INT8/PTQ workflow.
- Stronger sim benchmarks.
- MuJoCo and Isaac Lab via `physicalai-train`.
- Robots: UR, Franka, Unitree R1D, OpenArm2 and Oversonic (optional).
- Cameras: UVC, GMSL, MIPI, GenICam.
- Real-hardware eval setup.
- Minimal dataset -> train -> export -> deploy flow.
- Agentic orchestration exploration for year-end pipeline design.

Out of Q3 delivery:

- Navigation/VLN.
- ROS2 / ZeroMQ.
- RL / DAgger, except generic data-contract decisions.

## Current State

### `physicalai-train`

Exists:

- Policies: ACT, Pi0, Pi0.5, SmolVLA, GR00T-N1.5.
- LeRobot wrappers.
- `Observation`, `LeRobotDataModule`, Lightning training.
- LIBERO/PushT via `Gym`.
- Export mixin: Torch, ONNX, OpenVINO, ExecuTorch.
- CLI: `fit`, `validate`, `test`, `predict`, `benchmark`, `export`.

Gaps:

- Model coverage: Build P1/P2 models are not end-to-end.
- E2E path lacks broad coverage: train -> benchmark -> export -> parity -> runtime load -> sim/real eval.
- Export matrix:
  - ACT: Torch, OpenVINO, ONNX, ExecuTorch.
  - Pi0.5: Torch, OpenVINO. Current VLA baseline.
  - SmolVLA: Torch, OpenVINO.
  - Pi0, GR00T: Torch-only unless extended.
- Missing: export-equivalence harness, INT8/PTQ, WAM policy, standardized latency/throughput, full backend/UI policy exposure.

### `physicalai` Runtime

Exists:

- `InferenceModel.load(export_dir)`.
- OpenVINO, ONNX Runtime.
- Studio Torch/ExecuTorch adapters.
- `PolicyRuntime`: Sync, Async, RTC.
- Manifest schema.
- Robots: SO-101, Trossen WidowX-AI, bimanual Trossen.
- Cameras: UVC/V4L2, RealSense, Basler.
- iceoryx2 camera transport.

Gaps:

- Missing robots: UR, Franka, Unitree R1D, Oversonic, OpenArm2.
- UR/Franka extras are placeholders.
- Missing/unvalidated cameras: GMSL, MIPI, GenICam.
- `PolicyRuntime.from_config()` incomplete.
- RTC might require refactoring
- Remote execution and telemetry are scaffolding.

## Priorities

1. Model coverage.
2. E2E validation per model.
3. Benchmark breadth.
4. Deployment breadth.
5. Year-end orchestration design.

### P0

| Item | Repo | Output | Owner |
|---|---|---|---|
| Export parity | train | PyTorch vs exported-backend report | TBD |
| Model plan | train | Owners and E2E criteria per Build P1/P2 model | TBD |
| Baseline export | train | Pi0.5 hardened; Pi0/GR00T gaps scoped | TBD |
| Benchmarks | train | First-party RoboCasa; `vla-eval` plugin | TBD |
| Hardware wave 1 | runtime/backend | First agenda robots; UVC validated; GMSL/MIPI/GenICam scoped | TBD |
| Real eval | runtime/train | Frozen task, metrics, result format per robot | TBD |
| Dataset freeze | train/backend | Collection cutoff and eval schedule | TBD |

### P1

| Item | Repo | Output | Owner |
|---|---|---|---|
| Policy builds | train | 7 Build P1/P2 models exportable or blocked | TBD |
| INT8/PTQ | train/runtime | Calibration, support matrix, latency/accuracy | TBD |
| Hardware wave 2 | runtime/backend | Remaining agenda robots | TBD |
| Simulation | train/backend | MuJoCo path; Isaac Lab design or first task | TBD |
| Policy I/O | train/runtime | Shared input/output schema | TBD |
| One-flow UX | backend/ui/cli | dataset -> train -> export -> deploy recipe | TBD |
| GR00T | train/backend | N1.7 plan or implementation; backend/UI wiring | TBD |
| Orchestration exploration | train/runtime/backend | Year-end Qwen-style design: manipulation, navigation, world, agent | TBD |

### P2

| Item | Repo | Output |
|---|---|---|
| Runtime config | runtime | YAML deploy path |
| Dataset quality | train/backend | Quality/version report |
| Data contract | train | Generic `Observation` contract for VLA/WAM |
| Isaac Lab | train/backend | One task through benchmark flow |

## Model Plan

Source: [PAI Feature Tracker for Q3](https://intel-my.sharepoint.com/:x:/p/samet_akcay/IQDD_l82x1_fQ6HCf-bRDq5cAbJDy_bhaqAv7wJQdnR7vUE?e=CaF6My). One engineer owns each Build P1/P2 model.

| Owner | Model | Priority | Class | Target backends |
|---|---|---|---|---|
| E1 | MolmoAct 2 | Build P1 | ARM/VLA | ONNX, OpenVINO, ExecuTorch |
| E2 | RLDX-1 | Build P1 | VLA + tactile/torque | ONNX, OpenVINO |
| E3 | DreamZero | Build P1 | WAM | Torch, ONNX; OpenVINO stretch |
| E4 | VLA-JEPA | Build P1 | VLA + latent WM | ONNX, OpenVINO, Torch |
| E5 | Spirit v1.5 | Build P2 | VLA | ONNX, OpenVINO |
| E6 | LingBot-VLA | Build P2 | VLA | ONNX, OpenVINO |
| E7 | Qwen-VLA | Build P2 | VLA | ONNX, OpenVINO |

Refactor:

| Owner | Model | Priority | Target |
|---|---|---|---|
| TBD | GR00T N1.7 | Refactor P2 | N1.7, backend/UI, swappable VLM, export gaps |

Evaluate:

| Model | Priority | Decision |
|---|---|---|
| mimic-video | Evaluate | Openness, weights, benchmark value |
| LeWorldModel | Evaluate P3 | Planner module or policy |
| InternVLA-M1 | Evaluate P3 | License, weights, LeRobot status |
| GigaBrain-0.5M | Evaluate P3 | Open weights/buildability |
| LingBot-VA | Evaluate P3 | Compare with DreamZero |
| TD-MPC2 | Evaluate P3 | Defer unless RL enters scope |
| DreamerV3 | Evaluate P3 | Defer unless RL enters scope |
| V-JEPA 2/2.1 | Evaluate P3 | Backbone only; track via VLA-JEPA |

Done per model:

1. Training/parity on known data.
2. Benchmark entry.
3. Export path.
4. Export-equivalence pass.
5. `InferenceModel.load(...)` pass.
6. Real-robot run when hardware exists.

## Hardware Plan

| Type | Target | Status | Q3 |
|---|---|---|---|
| Robot | SO-101 | Existing | Keep/evaluate |
| Robot | Trossen WidowX-AI | Existing | Keep/evaluate |
| Robot | Bimanual Trossen | Existing | Generalize bimanual |
| Robot | Unitree R1D | Missing | Add |
| Robot | UR | Placeholder/missing | Add |
| Robot | Franka | Placeholder/missing | Add |
| Robot | Oversonic | Missing | Add |
| Robot | OpenArm2 | Missing | Add |
| Robot | ABB | Placeholder | Remove unless owned |
| Camera | UVC/V4L2 | Existing | Validate |
| Camera | GMSL | Missing | Scope/add |
| Camera | MIPI | Missing | Scope/add |
| Camera | GenICam | Placeholder/missing | Add/validate |
| Camera | RealSense | Existing | Keep |
| Camera | Basler | Existing | Keep; align with GenICam |
| Camera | IP camera | Stub | Defer unless needed |

New robot acceptance:

- `Robot` driver.
- Backend setup/calibration.
- UI setup if needed.
- Smoke test.
- Real-eval task.
- Safety docs.

## Benchmark Plan

First-party:

- Existing: PushT, LIBERO.
- Add: RoboCasa.
- Build: export-equivalence harness, real-hardware eval harness.

Plugin:

- Add `vla-eval` plugin, like the LeRobot policy plugin.
- Goal: run supported `vla-eval` benchmarks off the shelf.
- Do not reimplement `vla-eval` benchmarks first-party.

Track: RoboArena, RoboChallenge/Table30, MolmoSpaces, DexBench.

## Simulation Plan

Library owns simulator semantics. Studio launches and displays jobs.

Path:

```text
Gym adapter -> Observation -> evaluate_policy -> Benchmark -> results/video
```

MuJoCo:

- `physicalai-train[mujoco]`.
- `MuJoCoGym` if `GymnasiumGym` is insufficient.
- Observation and success mapping.
- `MuJoCoBenchmark`.

Isaac Lab:

- `physicalai-train[isaac]`.
- `IsaacLabGym`.
- Batched GPU obs/action handling.
- Observation and success mapping.
- `IsaacLabBenchmark`.
- Install/launch docs.

Studio:

- List envs/tasks.
- Launch jobs.
- Pass configs.
- Store metrics, videos, datasets.
- Display results.

## Export and Quantization

Track: `policy x backend x precision x device x parity status`.

Backends: Torch, ONNX, OpenVINO, ExecuTorch.

Quantization:

- Keep FP16 OpenVINO.
- Add NNCF INT8/PTQ.
- Define calibration input.
- Report latency/accuracy.
- Do not default to INT8 until validated.

## One-Flow UX

```text
select/import dataset
-> choose policy/config
-> train
-> benchmark
-> export
-> validate export
-> select robot/cameras
-> deploy
```

Implement recipe/config first, UI second.

## Agentic Orchestration Exploration

Goal: design the year-end pipeline, not ship it in Q3.

Reference: Qwen Robot Suite pattern — manipulation, navigation, world model, agent layer.

Q3 output:

- Capability map: current manipulation/VLA, WAM, missing navigation/VLN.
- Tool interface sketch for exported policies.
- Agent/runtime boundary proposal.
- Data and observation requirements for orchestration.
- Year-end implementation plan.

## Paper Evidence

| Claim | Evidence |
|---|---|
| Export works | PyTorch/export parity |
| Runtime works | Latency/throughput by device/backend |
| Edge works | FP16/INT8 latency and quality |
| Policies generalize | First-party LIBERO/RoboCasa plus `vla-eval` plugin results |
| Simulation is reusable | MuJoCo/Isaac Lab via `Gym` -> `Benchmark` |
| Hardware works | Success/stability across 3-4 embodiments |
| Coverage scaled | Pi0.5 plus 7 Build P1/P2 models end-to-end or blocked |
| Year-end pipeline scoped | Manipulation/navigation/world/agent design document |

## Open Decisions

- Confirm E1-E7 owners/backups.
- Assign GR00T owner.
- Assign Evaluate queue owner.
- Robot order/owners: UR, Franka, Unitree R1D, Oversonic, OpenArm2.
- ABB: remove unless owned?
- Camera order/owners: GMSL, MIPI, GenICam.
- Simulation owners: MuJoCo, Isaac Lab.
- Orchestration exploration owner.
- Paper robot count: 3 or 4?
- Data contract: extend `Observation`/`LeRobotDataModule` or add a new generic format?
- P0 owners.

## Future Work

- Agentic orchestration implementation.
- Navigation/VLN implementation.
- World model as data generator/evaluator.
- RL / DAgger.
- Cross-embodiment canonical state/action.
- ROS2 / ZeroMQ.
- Remote inference.

Sequence: **scale train-to-deploy first; compose later.**
