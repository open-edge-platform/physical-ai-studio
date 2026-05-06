---
marp: true
theme: default
paginate: true
size: 16:9
style: |
  section { font-size: 26px; padding: 50px 70px; }
  h1 { font-size: 40px; }
  h2 { font-size: 32px; }
  pre { font-size: 19px; line-height: 1.35; }
  code { font-size: 0.9em; }
  table { font-size: 22px; }
  .small { font-size: 20px; color: #555; }
  .columns { display: grid; grid-template-columns: 1fr 1fr; gap: 30px; }
  .footnote { position: absolute; bottom: 20px; left: 70px; font-size: 16px; color: #888; }
---

# Policy Runtime Design

PhysicalAI inference runtime — design review

<br>

**Goal of this meeting**

- Walk through the proposed runtime architecture
- Get sign-off on Phase 0–2 scope
- Surface concerns on RTC, remote execution, and the Studio split

<br>

<span class="small">Full docs: `policy_runtime_design.md`, `policy_server_design.md` · 1-pager: `design_review_summary.md`</span>

---

## The problem in one sentence

PhysicalAI has policies (`InferenceModel`, runners, adapters) and Studio has a worker (`RobotControlWorker`).

**Nothing in `physicalai` owns the per-tick control loop.**

<br>

Today every consumer reinvents:

- the loop
- timing and overrun handling
- async dispatch and chunk queueing
- RTC-style guidance
- callbacks for recording / telemetry
- remote inference

…and the runtime contract is not part of the package's public API.

---

## The gap (selected rows from `inference_comparison_report.md`)

| Capability | LeRobot | PhysicalAI today |
|---|---|---|
| Control-loop runner | `Strategy` + `lerobot-rollout` | none |
| `InferenceEngine` abstraction | `sync` / `rtc` | none |
| Async + RTC | `RTCInferenceEngine` | none |
| Precise timing + overrun warning | yes | none |
| First-inference warmup | yes (Torch) | none |
| Chunk queue with cross-chunk smoothing | `ActionQueue` + `QueueMixer` | partial, in `application/` |
| CLI to run a policy | `lerobot-rollout` | none |
| Remote inference | `PolicyServer` + `RobotClient` | none |

This design closes every row.

---

## Three boxes

```
┌─────────────────┐    ┌──────────────────┐    ┌──────────────────────┐
│ InferenceModel  │ ←→ │    Execution     │ ←→ │    PolicyRuntime     │
│  (policy math)  │    │   (transport)    │    │  (loop, queue, hooks)│
└─────────────────┘    └──────────────────┘    └──────────────────────┘
                                                          │
                                                          ▼
                                                  ┌──────────────┐
                                                  │    Robot     │
                                                  └──────────────┘
```

- **`InferenceModel`** — the policy. Knows nothing about timing or transport.
- **`Execution`** — how the model is called: sync, thread, process, remote. Single interface.
- **`PolicyRuntime`** — owns the loop, the action queue, callbacks, episode lifecycle.

Each box is testable in isolation. None of the three knows about the others' internals.

---

## The standalone API (this is the contract)

```python
from physicalai.runtime import PolicyRuntime
from physicalai.runtime.execution import AsyncExecution
from physicalai.inference import InferenceModel
from physicalai.robot import SO101

robot = SO101(port="/dev/ttyUSB0")
model = InferenceModel.load("checkpoints/so101_act")

runtime = PolicyRuntime(
    robot=robot,
    model=model,
    execution=AsyncExecution(transport="thread"),
    fps=30,
)

runtime.run(duration_s=60)
```

No Studio dependency. No `application/`. This is what `physicalai run --config foo.yaml` does too.

---

## `Execution` — one concept, four transports

```
┌──────────────────────┐  ┌──────────────────────┐
│  SyncExecution       │  │  AsyncExecution      │
│  inline on loop      │  │  (transport=thread)  │
└──────────────────────┘  └──────────────────────┘
┌──────────────────────┐  ┌──────────────────────┐
│  AsyncExecution      │  │  RemoteExecution     │
│  (transport=process) │  │  gRPC to PolicyServer│
└──────────────────────┘  └──────────────────────┘
```

- All four satisfy the same `Execution` protocol: `start(queue, model)`, `maybe_request(obs)`, `warmup`, `estimated_delay_frames`, `stop`.
- `PolicyRuntime` does not know which transport it's using.
- `AsyncExecution(transport="process")` is the path that maps onto today's `ModelWorker` + `ModelWorkerRegistry` — single-flight semantics preserved.
- Remote is a sibling, not a flag, because network failure modes don't belong in the local class.

---

## RTC composes — it does not subclass

LeRobot's `RTCInferenceEngine` bundles **async dispatch + denoising guidance + queue mechanics + latency tracking + warmup** into one class.

In this design:

```python
runtime = PolicyRuntime(
    robot=robot,
    model=InferenceModel.load("pi05", runner=FlowMatching(guidance=RTC())),
    execution=AsyncExecution(transport="process"),
    fps=30,
    action_queue=ActionQueue(
        smoother=LerpChunkSmoother(duration_frames=30),
        merger=RTCQueueMerger(),
    ),
)
```

- **`FlowMatching(guidance=RTC())`** — denoising + RTC math.
- **`AsyncExecution(...)`** — transport + latency tracking.
- **`ActionQueue(merger=RTCQueueMerger())`** — overlap handling.

Async without RTC works. RTC without a custom engine class works. Each piece is replaceable.

---

## What lives where

<div class="columns">

**`physicalai.runtime`**
- `PolicyRuntime` (loop)
- `Execution` (transport)
- `ActionQueue` + smoother
- `Callback` base + hooks
- `RuntimeValidator`
- `precise_sleep`, timing
- `ThreadSafeRobotProxy`

**`physicalai.inference`**
- `InferenceModel`
- runners (`SinglePass`, `ActionChunking`, `FlowMatching`, `TemporalEnsemble`)
- `Guidance` (`RTC`, …)
- preprocessors

</div>

<br>

**Consumer (Studio `application/backend/`)**

Recording, telemetry, WebSocket transport, teleop arbitration, session lifecycle, strategy workflows. Constructed on top of `PolicyRuntime` via callbacks and a thin shell around `runtime.run()`.

---

## Studio strategies — composition, not subclassing

LeRobot ships `Strategy` subclasses (sentry, dagger, highlight, HIL). We don't.

Each strategy is ~30 lines of consumer code over `PolicyRuntime` + `Callback`:

```python
class HILStrategy:
    def __init__(self, runtime, teleop):
        self.runtime, self.teleop = runtime, teleop
        runtime.callbacks.append(HILCallback(teleop, self._mode))

    def _mode(self):
        return self.teleop.current_mode()  # "policy" | "teleop" | "blend"

# HILCallback.before_send_action substitutes or blends actions.
```

Same shape works for sentry (`on_observation` trigger), highlight (rolling buffer + trigger), DAgger (record both expert and policy actions, beta-mix).

The runtime stays workflow-agnostic. Product workflows compose around `PolicyRuntime` with callbacks and consumer-owned code.

---

## RemoteExecution + PolicyServer

```
┌────────── Robot host ──────────┐    ┌──────── Server host ────────┐
│ PolicyRuntime                  │    │ PolicyServer                │
│  ├─ Robot, cameras             │    │  ├─ InferenceModel          │
│  ├─ ActionQueue                │    │  ├─ runner (FlowMatching…)  │
│  └─ RemoteExecution ── gRPC ───┼────┼─→ Guidance (RTC, …)         │
│       (client)                 │    │                             │
└────────────────────────────────┘    └─────────────────────────────┘
```

- **Same `Execution` contract** — `PolicyRuntime` doesn't know remote from process.
- **gRPC bidirectional streaming**, typed proto, `prev_chunk_left_over` and `inference_delay` cross the wire.
- **One model per server.** Multi-tenancy is a follow-up.
- **Build target Phase 5.** Design fixed now in `policy_server_design.md`; `RemoteExecution` uses the same `Execution` contract as local async.

Adopts LeRobot's `PolicyServer` / `RobotClient` split, restructured around our `Execution` boundary.

---

## What we're explicitly NOT building

| Component | Why not |
|---|---|
| `ObservationAssembler` protocol | Callbacks cover all known cases |
| `ActionArbiter` protocol | Two-source case (policy/teleop) handled in consumer; promote when 3+ sources |
| `ActionFilter` / `SafetyGate` | Limits live in `Robot` drivers; promote on multi-driver safety req |
| `ActionInterpolator` | All current policies match control rate |
| `ShutdownPolicy` | `return_to_home: bool` covers today's needs |
| `TwoPhase` runner | No supported policy publishes encode/decode split |
| Strategy classes in `physicalai` | Workflows live in consumer |
| `torch.compile` warmup | We're ExecuTorch / OpenVINO / ONNX, not Torch |

Promotion needs a concrete consumer, not an anticipated workflow.

---

## Phases

| Phase | Scope | Independent of |
|---|---|---|
| **0** | Image preprocessing parity, processor state hard-fail | Everything else |
| **1** | `predict_action_chunk`, `close`, `Guidance`, `FlowMatching`, `TemporalEnsemble` | Runtime |
| **2** | `physicalai.runtime` core + `SyncExecution` + `AsyncExecution(thread)` + CLI | Async-process |
| **3** | `AsyncExecution(process)` + RTC + `RTCQueueMerger` | Hardening |
| **4** | Hardening (numerical equivalence, latency accuracy) | Remote |
| **5** | `RemoteExecution` + `PolicyServer` + `physicalai serve` | Deferred |
| **6** | Deferred components when consumer demand justifies | — |

Phase 0 + Phase 1 land independently; they're correctness and inference-API work, not runtime work. Phase 2 is when `physicalai.runtime` shows up.

---

## Open questions for the team

1. **`TemporalEnsemble` output shape** — one smoothed action per tick, or smoothed chunk?
2. **`AsyncExecution` cancellation** — keep no-cancel default, or add `cancel_inflight=True` opt-in?
3. **Process-transport reader** — dedicated reader thread, or poll inside `maybe_request`?
4. **`ObservationAssembler` protocol** — needed in Phase 2, or do callbacks suffice indefinitely?
5. **`RemoteExecution` transport** — gRPC streaming (recommended) vs HTTP/2+JSON; multi-tenancy timing; auth posture.

<br>

Plus: **does the Studio split land?** Strategies stay in the consumer — does that match where Studio wants to invest?

---

# Discussion

<br>

**Decisions needed today**

- Phase 0 + Phase 1 scope: ship as proposed?
- Phase 2 runtime API: `PolicyRuntime` constructor signature locked?
- Studio strategies: stay in consumer (this design) or pull into `physicalai`?

<br>

**Reading after the meeting**

- `design_review_summary.md` — 1-pager
- `policy_runtime_design.md` — full design (22 sections)
- `policy_server_design.md` — remote inference (13 sections)
- `inference_comparison_report.md` — gap analysis vs LeRobot

---

<!-- Backup slides below -->

# Backup slides

---

## Backup: `predict_action_chunk` shape

```python
chunk: Mapping[str, Any] = model.predict_action_chunk(observation)

chunk["actions"]    # np.ndarray, shape (H, D)
chunk["policy_dt"]  # float | None
chunk["t0"]         # float | None — observation timestamp

# Runtime-injected by Execution for RTC:
#   inference_delay: int (policy frames)
#   prev_chunk_left_over: dict
```

Dict, not a dataclass. Unknown keys pass through. Matches existing PhysicalAI conventions.

`ActionChunking(SinglePass())` is **kept**, not deprecated. It and `PolicyRuntime + predict_action_chunk` are two valid ways to consume chunks.

---

## Backup: per-tick loop pseudocode

```python
def run(self, ...):
    with connect(self.robot):
        self.model.reset()
        self.execution.start(self.action_queue, self.model)
        self.execution.warmup(self._sample_observation())

        while not self.should_stop():
            step = StepContext(t=now())
            obs = self._assemble_observation()

            self.callbacks.on_observation(obs, step)
            self.execution.maybe_request(obs)
            action = self.action_queue.pop_or_none() or self._hold_position()

            self.callbacks.before_send_action(action, step)
            self.robot.send_action(action)
            self.callbacks.on_action_sent(action, step)

            self.sleep_until_next_tick()
```

---

## Backup: `physicalai run` CLI integration

The runtime is shipped to edge / inference hosts that **cannot install Torch**. The existing `physicalai` console script in `library/` uses `LightningCLI`, which pulls Lightning → Torch at import time. So `run` cannot live there.

**New layering — one command, two distributions:**

- Runtime distribution owns the `physicalai` console script (`physicalai.cli.main:main`). Pure `jsonargparse`, no Torch.
- Subcommands (`run`, `serve`) registered from the runtime via `[project.entry-points."physicalai.cli.subcommands"]`.
- Training subcommands (`fit`, `validate`, `test`, `predict`, `benchmark`) plug in from the training distribution via the same entry-point group. Lightning / Torch are only imported when one of those subcommands is invoked.

Edge host: `physicalai run`, `physicalai serve` available. Dev box (both distributions): all of the above.

Same in-process API: `PolicyRuntime.from_config(path)`. No Torch dependency in the runtime CLI.

---

## Backup: gRPC proto sketch (PolicyServer)

```proto
service PolicyServer {
  rpc Handshake(HandshakeRequest) returns (HandshakeReply);
  rpc Warmup(WarmupRequest) returns (WarmupReply);
  rpc Predict(stream PredictRequest) returns (stream PredictReply);
  rpc Health(google.protobuf.Empty) returns (HealthReply);
}

message PredictRequest {
  string request_id = 1;
  double t0 = 2;
  map<string, Tensor> observation = 3;
  optional int32 inference_delay = 4;        // RTC
  optional ActionChunk prev_chunk_left_over = 5;  // RTC
}

message PredictReply {
  string request_id = 1;
  Tensor actions = 3;                         // (H, D)
  optional double policy_dt = 4;
  map<string, Tensor> extra = 5;
}
```

---

## Backup: file layout

```
physicalai/runtime/
  policy_runtime.py
  action_queue.py            # ActionQueue + smoothers (co-located)
  execution/
    base.py
    sync.py
    async_thread.py
    async_process.py
    remote/
      client.py              # RemoteExecution
      server.py              # PolicyServer
      proto/                 # gRPC stubs
      auth.py
      codecs.py
  context.py                 # StepContext, UserEvent
  timing.py                  # precise_sleep, overrun warning
  callbacks.py
  validator.py
  robot_proxy.py             # ThreadSafeRobotProxy
```

Studio strategies live in their own packages. Runtime CLI: `physicalai/src/physicalai/cli/` (jsonargparse, no Torch). Training subcommands plug in from the training distribution via entry points.

---

## Backup: differences from LeRobot

| LeRobot | This design | Why |
|---|---|---|
| `Strategy` (loop + workflow) | `PolicyRuntime` (loop only) + consumer strategies | loop is workflow-agnostic |
| `RTCInferenceEngine` (math + sched + queue + timing) | `FlowMatching` + `Guidance` + `Execution` + `ActionQueue` | each piece replaceable |
| `RobotClient` peer to engine | `RemoteExecution` is an `Execution` | one transport interface, four impls |
| `compile_warmup_inferences` (Torch) | `Execution.warmup()` (backend-agnostic) | ExecuTorch / OpenVINO / ONNX |
| `lerobot-rollout` standalone CLI | `physicalai run` subcommand on existing CLI | one entry point |

This produces several small types instead of one large rollout subsystem. Each capability tested in isolation.
