# Inference Discrepancies: LeRobot vs PhysicalAI

**Scope:** Inference / deployment-time behavior of LeRobot (HuggingFace) vs PhysicalAI (`physicalai/src/physicalai/`).
**Goal:** Identify gaps in PhysicalAI's inference stack relative to LeRobot and describe their stability implications.

> **Note:** This document is a gap analysis only. The proposed design that closes these gaps lives in [`policy_runtime_design.md`](./policy_runtime_design.md), which includes the API surface, phased rollout, acceptance criteria, and gap-by-gap traceability back to this report.

---

## 0. TL;DR

1. **LeRobot ships an end-to-end deployment subsystem.** A single CLI (`lerobot-rollout`) wires policies, robots, teleop, datasets, processors, and inference engines through pluggable **strategies** (`base` / `sentry` / `highlight` / `dagger`) and pluggable **inference engines** (`sync` / `RTC`). RTC enables async chunk consumption for flow/diffusion policies under variable inference latency.
2. **PhysicalAI ships a deployment-time model wrapper, not a control loop.** `InferenceModel` loads exported policy packages (OpenVINO / ONNX / Torch / ExecuTorch) via a Pydantic `Manifest`, with pluggable runners, processors, and callbacks. Its scope ends at "produce one action for one observation." The control loop, FPS enforcement, async chunk production, and multi-strategy rollout are the caller's responsibility.
3. **Three concrete root causes of the reported inference instability:**
   - **PhysicalAI lacks a control-loop runner.** The framework provides no FPS-tracked execution loop, deterministic sleep helper, loop-overrun detection, shutdown handling, or rate decoupling between the policy and the robot. Loops written on top of `InferenceModel` slip silently when inference latency approaches the control period.
   - **PhysicalAI uses open-loop chunk consumption.** The current chunking runner pops one action per call and refills synchronously when its buffer empties. Without a background chunk producer, a delay-aware queue, or guided overlap (RTC), this produces a periodic latency spike every `chunk_size` ticks.
   - **PhysicalAI lacks a shared image preprocessing primitive.** Pi0.5 and SmolVLA each implement their own resize / pad / scale logic rather than composing a single `ImagePreprocessor` with explicit fields. The result is silent geometry drift between policies, and between training and inference.

---

## 1. Reference architecture — how LeRobot does inference today

### 1.1 Entry point

```
pyproject.toml → lerobot-rollout = lerobot.scripts.lerobot_rollout:main
```

`lerobot_rollout.py` is small and orchestrational:

```python
@parser.wrap()
def rollout(cfg: RolloutConfig):
    init_logging()
    if cfg.display_data: init_rerun(...)
    signal_handler = ProcessSignalHandler(use_threads=True)
    shutdown_event = signal_handler.shutdown_event
    ctx = build_rollout_context(cfg, shutdown_event)
    strategy = create_strategy(cfg.strategy)
    try:
        strategy.setup(ctx); strategy.run(ctx)
    finally:
        strategy.teardown(ctx)
```

### 1.2 Layered architecture

```
                ┌─────────────────────────────────────┐
                │            lerobot-rollout          │  CLI (draccus)
                └─────────────────────────────────────┘
                                  │
                ┌─────────────────────────────────────┐
                │           RolloutConfig             │  config dataclasses
                │   robot / teleop / policy / dataset │
                │   strategy + inference + fps + ...  │
                └─────────────────────────────────────┘
                                  │
                ┌─────────────────────────────────────┐
                │       build_rollout_context()       │  DI: load policy,
                │  → RolloutContext(runtime, hardware,│  connect robot,
                │     policy, processors, data)       │  build processors,
                └─────────────────────────────────────┘  pick engine
                              │           │
              ┌───────────────┘           └───────────────┐
              ▼                                            ▼
   ┌──────────────────────┐                   ┌─────────────────────────┐
   │     Strategy         │                   │    InferenceEngine      │
   │  base / sentry /     │ ← shared helper → │   sync   |    RTC       │
   │  highlight / dagger  │  send_next_action │  (inline)|  (bg thread) │
   └──────────────────────┘                   └─────────────────────────┘
              │                                            │
              ▼                                            ▼
   ┌──────────────────────┐                   ┌─────────────────────────┐
   │ ActionInterpolator   │                   │      ActionQueue        │
   │  + precise_sleep +   │                   │  merge / replace /      │
   │  control_interval    │                   │  delay-aware popping    │
   └──────────────────────┘                   └─────────────────────────┘
              │
              ▼
   robot.send_action(processed)
```

### 1.3 Components relevant to inference stability

| Component | Where | Behavior |
|---|---|---|
| Strategy / Engine split | `rollout/strategies/*` + `rollout/inference/*` | Decouples per-tick behavior (record, dagger…) from policy execution mode (sync vs RTC). RTC can be swapped in without modifying the loop. |
| `build_rollout_context` | `rollout/context.py` | Centralizes policy load, device/dtype selection, optional compile, PEFT wiring, processor build, and rejection of incompatible combinations (e.g. sync + relative-action). |
| `ActionInterpolator` | `rollout/strategies/core.py` | Control loop ticks at `fps × multiplier`; policy fires only when `needs_new_action()`. Decouples policy rate from robot rate. |
| `precise_sleep` + overrun warning | `strategies/base.py` | Computes `dt`, sleeps the remainder, logs when the loop overruns the period. |
| torch.compile warmup gate | `strategies/base.py` + `RTCInferenceEngine` | Skips the first N inferences so first-compile latency does not exceed the loop budget. |
| Robot wrapper for thread safety | `rollout/robot_wrapper.py` | Allows RTC's background thread to read observations without racing the main loop. |
| `ActionQueue` with delay-aware replace | `policies/rtc/action_queue.py` | Merges new chunks with leftovers from the previous chunk; drops the inference-delay prefix. |
| Relative-action reanchoring | `policies/rtc/relative.py` + `rollout/inference/rtc.py` | Re-anchors RTC leftovers against the snapshot state used for the previous chunk, for OpenPI-style relative-action policies. |

### 1.4 The RTC algorithm, in one paragraph

RTC ([Black, Galliker, Levine, *Real-Time Execution of Action Chunking Flow Policies*, arXiv:2506.07339](https://arxiv.org/abs/2506.07339)) treats the next action chunk as an inpainting problem during diffusion/flow denoising. With prediction horizon `H`, execution horizon `s`, and inference delay `d` (in timesteps), the first `d` actions of the new chunk are frozen to match what will already have executed, the overlap region `d..H-s-1` is soft-masked (e.g. exponential schedule), and the tail is freely generated. At each denoising step the algorithm computes the predicted clean chunk `Â^1_t = A^τ_t + (1-τ) v_π(A^τ_t, o_t, τ)`, takes a weighted error against the previous chunk's tail `e = (A_prev - Â^1_t) ⊙ W`, backprops to get a guidance correction `g`, and updates `A^{τ+1/n} = A^τ + (1/n)(v_π + clamp(β) · g)`. Overlapping chunks are made consistent without averaging, while a background thread predicts the next chunk so execution does not block. Feasibility requires `d ≤ H - s`. Physical Intelligence reports stability up to >300 ms inference delays.

LeRobot's implementation: `policies/rtc/modeling_rtc.py`. Reference Pi/Kinetix implementation: [`Physical-Intelligence/real-time-chunking-kinetix`](https://github.com/Physical-Intelligence/real-time-chunking-kinetix).

---

## 2. PhysicalAI inference today — what is present

`physicalai/src/physicalai/` contains three peer subsystems:

```
physicalai/src/physicalai/
├── inference/
│   ├── model.py                   # InferenceModel — select_action / __call__
│   ├── manifest.py                # Pydantic Manifest schema + loader
│   ├── component_factory.py       # DI / instantiate_component
│   ├── adapters/                  # OpenVINO, ONNX, Torch, ExecuTorch backends
│   ├── runners/                   # SinglePass, chunked-select adapter via ActionChunkCursor
│   ├── preprocessors/             # StatsNormalizer + Pi05, SmolVLA, tokenizers
│   ├── postprocessors/            # StatsDenormalizer, ActionNormalizer
│   └── callbacks/                 # Latency, Throughput
├── capture/                       # Camera abstraction
│   └── cameras/                   # Basler, RealSense, UVC + depth mixin
└── robot/
    ├── interface.py               # Robot Protocol (duck-typed)
    ├── so101/                     # SO-101 driver
    └── trossen/                   # Trossen WidowX-AI driver
```

### 2.1 What is implemented

- **Multi-backend adapters.** `RuntimeAdapter` interface with implementations for OpenVINO, ONNX, PyTorch, and ExecuTorch. LeRobot covers PyTorch only at deployment time.
- **Manifest-driven loading.** `manifest.json` (Pydantic-validated) declares artifacts, runner spec, processor pipelines, robot spec, and tensor shapes. Components resolve via `type` + flat params (LeRobot-compatible) or `class_path` + `init_args` (jsonargparse-style). Legacy `metadata.{yaml,json}` is loaded with a `DeprecationWarning`.
- **Runner abstraction.** `SinglePass` invokes the adapter once. Chunk-producing runners predict an action chunk; `select_action()` consumes it through an internal `ActionChunkCursor`. LeRobot keeps equivalent buffering inside each policy class.
- **Processor pipelines.** Preprocessor and postprocessor lists are instantiated from the manifest. `StatsNormalizer` / `StatsDenormalizer` load `safetensors` stats and support `mean_std`, `min_max`, `quantiles`, and `identity` modes — functional parity with LeRobot's `Normalize` / `Unnormalize`.
- **Callback surface.** `Callback` exposes `on_load`, `on_predict_start`, `on_predict_end`, `on_reset`. `Latency` and `Throughput` callbacks are included. LeRobot has no equivalent at this layer.
- **Robot Protocol.** `physicalai.robot.interface.Robot` is a `runtime_checkable` Protocol; concrete drivers exist for SO-101 and Trossen WidowX-AI.
- **Capture subsystem.** `physicalai/capture/` packages Basler, RealSense, and UVC backends with a depth mixin. LeRobot bundles cameras inside the main package.

### 2.2 What is not present

PhysicalAI currently lacks a number of capabilities that LeRobot's rollout subsystem provides. These fall into four groups.

**Control-loop and orchestration.** PhysicalAI does not include a control-loop or rollout runner equivalent to `rollout/strategies/base.py`; callers are expected to write the `while` loop around `InferenceModel.select_action` themselves. No CLI is registered in `pyproject.toml`, so there is no `physicalai-rollout` entry point. The framework also lacks a deterministic timing helper (`precise_sleep`) with loop-overrun detection — `examples/so101/move_joints.py` uses bare `time.sleep` — and provides no framework-level handling for safe shutdown or return-to-initial-position. There is no `ActionInterpolator` to decouple the control rate from the policy rate, and no equivalent of LeRobot's `Strategy` abstractions (`base` / `sentry` / `highlight` / `dagger`) for product-level workflows such as DAgger or HG-DAgger.

**Inference execution.** PhysicalAI does not expose an `InferenceEngine` abstraction: `InferenceModel` is synchronous, and there is no boundary for swapping in an asynchronous implementation without modifying call sites. There is no asynchronous or RTC inference path — no background producer thread, no delay-aware `ActionQueue.replace`, and no guided-denoising for flow/diffusion policies. Relative-action reanchoring (the bookkeeping required when an OpenPI-style policy is consumed under RTC) is also absent. Remote inference over gRPC (`PolicyServer` / `RobotClient`) is not implemented. A `torch.compile` warmup gate is also missing; this is only relevant when the Torch adapter is active, since the OpenVINO / ONNX / ExecuTorch adapters do not pay first-compile cost.

**Preprocessing parity.** PhysicalAI does not currently provide a shared `ImagePreprocessor` primitive. `preprocessors/pi05.py` and `preprocessors/smolvla.py` each implement resize, padding, and scaling independently, and there is no single object with explicit `scale`, `channel_order`, and `pad_mode` fields populated from the manifest. The framework also lacks a preflight assertion that the resolved processor pipeline matches what the model expects (dtype, scale range, NCHW vs NHWC); mismatches currently surface as adapter errors at first inference, without provenance.

**Telemetry.** PhysicalAI does not include Rerun telemetry hooks for live debugging of queue depth, inference latency, or observation freshness.

---

## 3. Discrepancy matrix

| Feature | LeRobot | PhysicalAI | Stability impact |
|---|---|---|---|
| Backend coverage | PyTorch only | OpenVINO, ONNX, Torch, ExecuTorch | PhysicalAI broader at deployment time |
| Single deployment CLI | `lerobot-rollout` | none registered | High — loops are hand-rolled |
| Manifest / artifact loader | implicit (multiple files) | Pydantic `Manifest` (single contract) | PhysicalAI stricter |
| Strategy abstraction (record / dagger / highlight / sentry) | yes | none | Medium |
| Sync inference engine | `rollout/inference/sync.py` | implicit in `InferenceModel.__call__` | Medium — no swappable engine boundary |
| RTC inference engine | `rollout/inference/rtc.py` + `policies/rtc/*` | absent | **High** — chunk-boundary discontinuities on flow/diffusion policies |
| Async gRPC server / client | `async_inference/policy_server.py` + `robot_client.py` | absent | Medium (relevant for remote GPU) |
| `ActionInterpolator` (control-rate decoupling) | yes | absent | Medium |
| `precise_sleep` + overrun warning | yes | bare `time.sleep` in examples, no warn | Medium (silent slippage) |
| torch.compile warmup gate | yes (`compile_warmup_inferences`) | absent | Low–Medium (Torch backend only) |
| Robot wrapper for thread safety | `rollout/robot_wrapper.py` | absent | High (precondition for RTC) |
| Action chunk queue mechanics | per-policy queue + RTC replace | direct `select_action()` uses an internal `ActionChunkCursor`; runtime path uses `ActionQueue` with background refill | High (stale actions under latency) — closed by `ActionQueue`; cursor is the small shared helper for pop-from-chunk mechanics |
| Relative-action reanchor in RTC | yes | absent | Medium (Pi0/Pi0.5 path) |
| Pre/post processor pipeline | `DataProcessorPipeline` from checkpoint | `Preprocessor` / `Postprocessor` lists from manifest | Functional parity |
| Stats normalization | `Normalize` / `Unnormalize` | `StatsNormalizer` / `StatsDenormalizer` (mean_std / min_max / quantiles) | Parity |
| Image scale / channel order / resize / pad | single `VanillaObservationProcessorStep` with explicit fields | per-policy preprocessors (`pi05.py`, `smolvla.py`) | **High** — geometry drift across policies; cf. LeRobot [#3158](https://github.com/huggingface/lerobot/pull/3158) (0% vs 80–90% success on aspect-ratio-preserving resize) |
| Preflight dtype/shape assertion | implicit | absent | Medium — mismatches surface inside the adapter |
| Robot abstraction | concrete classes + base | `Protocol` (duck-typed) | Both functional |
| Robot driver coverage | many (SO-100/101, Koch, ALOHA, Stretch, …) | SO-101, Trossen WidowX-AI | LeRobot wider |
| Camera subsystem | within `lerobot/cameras/` | dedicated `physicalai/capture/` (Basler, RealSense, UVC, depth mixin) | Different packaging |
| Episode termination + safe shutdown | strategy-level + return-to-initial-position | absent at framework level | Medium |
| Rerun telemetry | yes | absent | Low (debuggability) |
| Inference-time callback hooks | none at this layer | `on_load` / `on_predict_start/end` / `on_reset` | PhysicalAI exposes hooks |

---

## 4. Concrete causes of inference instability

1. **Configuration validation.** `RolloutConfig.__post_init__` rejects illegal combinations (sync + relative-action policy, missing teleop for DAgger, etc.). PhysicalAI does not currently perform equivalent validation, because there is no runtime layer above `InferenceModel` to host it.
2. **FPS enforcement.** `strategies/base.py` computes `dt`, sleeps the remainder via `precise_sleep`, and logs when the loop overruns. PhysicalAI does not yet provide a comparable timing helper, so loop slippage is undetected when inference latency rises.
3. **Rate decoupling.** `ActionInterpolator` lets the robot tick at `fps × multiplier` while the policy fires only when `needs_new_action()`. PhysicalAI does not provide an equivalent, so control output is bound 1:1 to policy output.
4. **Async chunk generation.** RTC's background thread and queue ensure the next chunk is populated before the current one is exhausted. The direct-call path re-runs the adapter synchronously when its `ActionChunkCursor` empties, producing a periodic latency spike every `chunk_size` ticks.
5. **Relative-action reanchoring.** OpenPI-style Pi0/Pi0.5 produce deltas from a snapshot state. LeRobot caches that state and re-anchors RTC leftovers to it. Without this bookkeeping, naive chunking on relative-action checkpoints drifts.
6. **Image preprocessing parity.** LeRobot has a single `VanillaObservationProcessorStep`, and policy-specific resize/pad lives next to the model definition. PhysicalAI's per-policy preprocessors handle dtype, scale, resize, and padding independently, so the same checkpoint can produce different geometry depending on which preprocessor wraps it.
7. **First-compile latency.** `compile_warmup_inferences=2` and a warmup branch in the strategy together contain the first-compile cost. PhysicalAI does not currently provide a warmup gate, though this primarily affects the Torch adapter.
8. **External validation.** PRs [#3444](https://github.com/huggingface/lerobot/pull/3444), [#3453](https://github.com/huggingface/lerobot/pull/3453), [#3469](https://github.com/huggingface/lerobot/pull/3469) and issues [#2356](https://github.com/huggingface/lerobot/issues/2356), [#2475](https://github.com/huggingface/lerobot/issues/2475), [#3158](https://github.com/huggingface/lerobot/pull/3158) document the failure modes a hand-rolled control loop encounters, and the fixes the LeRobot path now contains.

---

## 5. Bibliography

### Code references (LeRobot, current HEAD)

- `src/lerobot/scripts/lerobot_rollout.py` — CLI entry
- `src/lerobot/rollout/configs.py` — config dataclasses
- `src/lerobot/rollout/context.py` — DI / build_rollout_context
- `src/lerobot/rollout/strategies/{base,sentry,highlight,dagger,core}.py`
- `src/lerobot/rollout/inference/{factory,base,sync,rtc}.py`
- `src/lerobot/rollout/robot_wrapper.py`
- `src/lerobot/policies/rtc/{configuration_rtc,modeling_rtc,action_queue,relative}.py`
- `src/lerobot/async_inference/{policy_server,robot_client,configs,helpers}.py`
- `src/lerobot/processor/{normalize_processor,observation_processor,pipeline}.py`

### Code references (PhysicalAI)

- `physicalai/src/physicalai/inference/model.py` — `InferenceModel`
- `physicalai/src/physicalai/inference/manifest.py` — Pydantic manifest schema
- `physicalai/src/physicalai/inference/component_factory.py` — DI / instantiate_component
- `physicalai/src/physicalai/inference/adapters/{openvino,onnx,base,registry}.py`
- `physicalai/src/physicalai/inference/runners/{single_pass,action_chunking,factory,base}.py`
- `physicalai/src/physicalai/inference/preprocessors/{stats_normalizer,pi05,smolvla,hf_tokenizer,ov_tokenizer}.py`
- `physicalai/src/physicalai/inference/postprocessors/{stats_denormalizer,action_normalizer}.py`
- `physicalai/src/physicalai/inference/callbacks/{latency,throughput,base}.py`
- `physicalai/src/physicalai/robot/{interface,connect,verify}.py`
- `physicalai/src/physicalai/robot/{so101,trossen}/`
- `physicalai/src/physicalai/capture/cameras/{basler,realsense,uvc}/`

### Papers / blogs

- Black, Galliker, Levine. *Real-Time Execution of Action Chunking Flow Policies*. arXiv:2506.07339 (2025). https://arxiv.org/abs/2506.07339
- Physical Intelligence. *Real-Time Action Chunking with Large Models* (blog, 2025-06-09). https://physicalintelligence.company/research/real_time_chunking
- Physical Intelligence. *π0* (2024-10-31). https://physicalintelligence.company/blog/pi0
- Physical Intelligence. *π0.5* (2025-04-22). https://physicalintelligence.company/blog/pi05
- HuggingFace. *Asynchronous Robot Inference* (2025-07-10). https://huggingface.co/blog/async-robot-inference
- HuggingFace. *SmolVLA* (2025-06-03). https://huggingface.co/blog/smolvla
- LeRobot docs: https://huggingface.co/docs/lerobot/en/async, https://huggingface.co/docs/lerobot/main/en/smolvla

### Reference implementations

- `huggingface/lerobot` RTC: https://github.com/huggingface/lerobot/blob/main/src/lerobot/policies/rtc/modeling_rtc.py
- `Physical-Intelligence/real-time-chunking-kinetix`: https://github.com/Physical-Intelligence/real-time-chunking-kinetix
- `Physical-Intelligence/openpi`: https://github.com/Physical-Intelligence/openpi

### Key PRs / issues

- [#3413](https://github.com/huggingface/lerobot/pull/3413) — `lerobot-rollout` CLI
- [#1698](https://github.com/huggingface/lerobot/pull/1698) — RTC for Pi0/SmolVLA/Pi0.5
- [#2970](https://github.com/huggingface/lerobot/pull/2970) — relative actions for OpenPI parity
- [#3158](https://github.com/huggingface/lerobot/pull/3158) — aspect-ratio-preserving resize: 0% vs 80–90% async/sync success
- [#3453](https://github.com/huggingface/lerobot/pull/3453), [#3444](https://github.com/huggingface/lerobot/pull/3444) — relative-action RTC stabilization
- [#1196](https://github.com/huggingface/lerobot/pull/1196) — async inference / `PolicyServer` / `RobotClient`
- [#3047](https://github.com/huggingface/lerobot/issues/3047) — pickle RCE (avoid in PhysicalAI port)
- [#2475](https://github.com/huggingface/lerobot/issues/2475) — image-resize divergence between async and local inference
- [#2356](https://github.com/huggingface/lerobot/issues/2356) — async inference completes only one chunk
