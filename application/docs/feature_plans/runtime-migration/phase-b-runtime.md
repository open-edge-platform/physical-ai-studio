# Phase B — Zenoh transport, inference and recording

Move the session into its own process addressed over Zenoh, put inference and dataset recording on it, and delete the old inference stack.

Read [design.md](design.md) first, and [phase-a-teleop.md](phase-a-teleop.md) for what already exists. This document owns the sequence.

> Delete this document in the pull request that completes the phase.

## Prerequisites

- Phase A merged. The session, the action source, the contract and the config builder all exist and work.
- Phase 0 merged for B3 onward: `PolicySource.reset()` and public `warmup()`. See [phase-0-upstream.md](phase-0-upstream.md).
- The exclusivity bug fixes merged before B4. Specifically: `api/camera.py` must pass `is_locked` to `CameraWorker`, so a preview stream attaches read-only during recording instead of being able to reconfigure the publisher. B4 adds preview subscribers during recording, which makes that latent bug more likely to fire.

## Sequence

Five pull requests. Each is independently reviewable, and each deletes the old code it makes dead rather than deferring to a sweep at the end.

| PR  | Title                                                          | Deletes                                                                                                                        |
| --- | -------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| B1  | `feat(runtime): carry session control over zenoh`              | —                                                                                                                              |
| B2  | `feat(runtime): make the session a discoverable runtime owner` | —                                                                                                                              |
| B3  | `feat(runtime): run policy inference on the session`           | `model_worker`, `model_worker_registry`, `inference_poller`, `queue_mixer`, `inference_result`, `sync_mixed_model_integration` |
| B4  | `feat(runtime): record datasets from runtime ticks`            | `robot_control_worker`, `environment_integration`                                                                              |
| B5  | `feat(runtime): export a runnable inference bundle`            | —                                                                                                                              |

B1 comes first deliberately. It changes transport and host with **no functional change**, against teleoperation's small surface, so the Zenoh work is proven before inference depends on it.

---

## B1 — Zenoh transport

Move the phase A teleoperation session from a thread with direct calls to a process with Zenoh, behaving identically.

### Topic scheme

```text
studio/rt/{session}/metadata      queryable  — session identity and status
studio/rt/{session}/command       pub/sub    — idempotent commands
studio/rt/{session}/request       queryable  — acked commands
studio/rt/{session}/tick          pub/sub    — observations
studio/rt/{session}/state         pub/sub    — session state
studio/rt/{session}/error         pub/sub    — errors
studio/rt/{session}/lifecycle     pub/sub    — run boundaries
```

`{session}` is **`rt-<follower uuid>`**, not the bare UUID. One session per controlled robot, so the follower is the natural key, and it makes the exclusivity property structural — see [design.md](design.md#exclusivity).

**The `rt-` prefix is load-bearing.** The bare UUID is already `SharedRobot`'s own name, and upstream's name lock and rendezvous port are both keyed on that string without per-caller namespacing. A session sharing it deadlocks its own robot's owner process at startup — see [design.md](design.md#zenoh-event-architecture) for the mechanism. This fails at B1's first step, the phase whose whole purpose is to prove the transport before inference depends on it, so the regression test in [Validation](#validation) is not optional.

### Quality of service

Follow physicalai's robot transport, which sets these explicitly for the same reasons:

| Channel     | Settings                                           | Rationale                                                                  |
| ----------- | -------------------------------------------------- | -------------------------------------------------------------------------- |
| commands    | `BEST_EFFORT`, `DROP`, subscriber `RingChannel(1)` | Idempotent absolute state; only the newest matters                         |
| requests    | queryable with reply                               | Edge-triggered and cannot be lost                                          |
| tick        | `BEST_EFFORT`, `DROP`, `express=True`              | Small, 30 Hz; a slow consumer degrades to fewer frames instead of stalling |
| state/error | `BEST_EFFORT`, `DROP`, `express=True`              | Low rate; latest wins                                                      |

`express=True` matters: small messages at tens of hertz otherwise sit in Zenoh's batching window and arrive late.

### Studio owns its session helper

Studio needs its own function to open a Zenoh session, for two reasons:

- **Port derivation.** Upstream's helper derives its port from `physicalai/robot/{name}`. Studio's sessions must derive theirs from their own prefix, or a session and a robot owner sharing a name segment collide.
- **Private module.** `open_session` lives in `physicalai.robot.transport._session`, which is private. Upstream states the convention against reaching into private modules across packages.

Studio's version must apply the same security configuration, because getting it wrong is a security bug rather than a malfunction:

| Setting                      | Value                                           |
| ---------------------------- | ----------------------------------------------- |
| mode                         | `peer`                                          |
| `scouting/multicast/enabled` | `false`                                         |
| `scouting/gossip/enabled`    | `false`                                         |
| listen endpoint (session)    | `tcp/127.0.0.1:<derived from studio/rt/{name}>` |
| connect endpoint (client)    | the same derived port on loopback               |

Do **not** copy `TelemetryEmitter`, which opens a bare `zenoh.Config()` — scouting enabled, nothing loopback-bound. That module is explicitly marked as unwired scaffolding and has not had a security pass. Studio is putting a _control plane_ on the wire here, so the stakes are higher than telemetry.

Phase 0 asks upstream to make its session helper public, which would let Studio pass its own key prefix and drop the duplicated configuration. Until then, duplicate it deliberately and cover it with the test below.

### Startup race

A command published before the session's subscriber is declared is dropped. An `mp.Queue` would have buffered it.

The session answers `/metadata` once its subscribers are declared. The client probes that queryable and does not publish commands until it answers, which is exactly how `SharedRobot.connect()` waits on `_query_metadata_with_retry`.

### Process host

New `application/backend/src/runtime/hosts/process_host.py`, a `BaseProcessWorker`. `setup()` already re-initialises loguru sinks in the child, and the spawn start method is already forced.

**Send plain data, never the factory.** `RobotClientFactory` cannot cross the boundary: `RobotCatalogRegistry` holds builder, probe and resolver callables, `create_model`-generated classes and a `TypeAdapter`, none of which pickle under spawn. Send the config document from phase A's builder plus the pydantic rows the API already resolved, and let the session build its own factory in `setup()`. See [design.md](design.md#robotclientfactory-cannot-cross-a-process-boundary).

Keep `Popen` parentage alongside Zenoh. Control over Zenoh buys reattach; being the parent buys termination of a wedged process regardless of its internal state. physicalai does both for robot owners.

### Camera frames do not cross the boundary

Each side subscribes to the same iceoryx2 publisher through its own `SharedCamera`. `CameraWorker` already proves the pattern. Nothing is transported.

### Validation

Teleoperation behaves exactly as at the end of phase A. Reuse phase A's tests against the process host; add:

- a command published before `/metadata` answers is not lost, because the client waits
- an acked command receives a reply carrying its `request_id`
- killing the session process surfaces as an error on the socket rather than a hang
- no Zenoh session is created with scouting enabled, and every endpoint is bound to loopback — assert on the config Studio's session helper builds
- **a session and its own robot's `SharedRobot` come up together.** The regression test for the naming collision: start a session for robot X, confirm the robot owner reaches a connected state rather than failing with `name_lock_contention`, and confirm the two listen on different ports

---

## B2 — Runtime owner semantics

Turn the parent-owned process into a discoverable, reattachable owner.

### Discover or spawn

Follow `SharedRobot.connect()`'s shape: probe `/metadata`; if nobody answers, acquire a host-local name lock on the session's own `rt-<uuid>` identity and spawn; on losing the race, re-probe and attach to the winner. Idempotent, and reattach after an API restart comes free.

The lock is Studio's own, keyed on the `rt-` identity. Do not acquire upstream's name lock: it is a private module, and the identity must differ from the robot's for the reason in [Topic scheme](#topic-scheme).

### Idle self-exit

Follow the subscriber-presence loop in `physicalai/robot/transport/_owner_worker.py`: read the telemetry publisher's matching status, track `idle_since`, exit past `idle_timeout`.

Two requirements, which are design constraints rather than tuning:

- **Finalize recording on the way out.** An idle exit during recording must run `RecordingMutation.teardown` so saved episodes survive. Otherwise self-exit is data loss. This lands here even though recording arrives in B4 — write the shutdown path so B4 only has to hook into it.
- **Default 45 seconds, exposed as a setting.** Not a constant: labs on flaky networks want it longer, unattended rigs want it shorter. Put the reasoning in the docstring next to the default — see [design.md](design.md#lifecycle) — because whoever tunes it will not have these documents open.

Leave follower torque enabled on shutdown. SO101 holds position rather than dropping under gravity.

### Camera claim registry

Generalise `recording_locked_camera_fingerprints` into a claim registry: `fingerprint → (holder, settings)`. First claimant pins the settings; a later session requesting different settings is rejected with the conflicting project named. Recording becomes one reason to hold a claim rather than a separate mechanism.

This subsumes the cross-project settings conflict, so coordinate the two.

### Deletion checks query discovery

Deleting a robot, camera or environment that a live session holds is rejected with HTTP 423 and the holder named. The check queries `/metadata`, not in-memory state — an in-memory registry forgets everything on API restart, and the delete would then succeed while a session is still driving the arm.

### Validation

- a second client attaches to an existing session instead of spawning a second one
- restarting the API reattaches to a live session, and telemetry resumes
- a session with no subscribers exits within `idle_timeout`
- a session with no subscribers **during recording** finalizes the dataset before exiting
- a second session for the same robot is rejected
- deleting a held robot returns 423

---

## B3 — Policy inference

Add the policy delegate to `StudioActionSource` and delete the old inference stack.

### Wiring

```python
PolicySource(
    model=InferenceModel(export_dir=..., policy_name=..., backend=..., device=...),
    execution=AsyncExecution(request_threshold=...),
    action_queue=ChunkedActionQueue(smoother=LerpSmoother(duration_frames=...)),
    task=...,
)
```

`AsyncExecution` plus `ChunkedActionQueue` plus `LerpSmoother` replace `InferencePoller` plus `QueueMixer` exactly, and improve on them: the chunk offset comes from the queue's exact pop count rather than an estimate of measured latency times fps.

### Loading

Model loading takes seconds and must not stall the loop. On `load_model`:

1. Hand the work to a loader thread inside the session process.
2. When it completes, construct the `PolicySource` and call `connect(bus=..., session_id=...)` with the values cached by `StudioActionSource.connect()`.
3. Call `warmup()` with the most recent tick's observation, still on the loader thread.
4. Publish `state` with `model_loaded: true`.

Step 3 is why phase 0 adds public `warmup()`. Without it the first armed tick pays a blocking inference on the control thread.

### Arm and disarm

`start_task` sets the task via `set_task()` and switches the mode to `policy`. `stop_task` switches to `hold`.

Re-arming calls `PolicySource.reset()` **before** switching the mode. Without it, the first tick pops an action computed from an observation taken before the operator moved the arm. See [phase-0-upstream.md](phase-0-upstream.md#1-policysourcereset) for the timeline.

Do not use `stop()` then `run()` as an arm boundary. It ends the session, closes the callbacks, and also ends teleoperation, which the same runtime is driving.

### Two silent bugs this fixes by construction

- **The task string now reaches the model.** Today `format_model_input_observation` accepts `task` and drops it, so Pi0, Pi0.5 and SmolVLA run without language conditioning. `PolicySource` forwards it.
- **Channel order becomes correct.** Today the inference path swaps RGB to BGR while the dataset is RGB. `PolicySource._to_model_input` performs no swap. Nothing to add — but assert it, so nobody reintroduces a swap. See [design.md](design.md#dataset-frames-are-rgb-end-to-end) for the evidence.

### Error handling

`AsyncExecution.start()` waits `_STRAGGLER_GRACE_S` (2 s) for a worker left over from the previous run, then raises `RuntimeError`. The message quotes 12 s because it sums in the `_JOIN_TIMEOUT_S` (10 s) join that the preceding `stop()` already spent — **`start()` itself blocks for two seconds, not twelve.** That is why re-arm can stay on the command path instead of being pushed to a background thread.

Catch the error on re-arm, report it as an `error` event, and leave the session alive.

`WorkerDiedError` propagates out of `run()`. Treat it as fatal for the session and report it.

### Camera keys must match the training dataset

With two or more cameras, `PolicySource` emits `images.<name>` per camera, and `InferenceModel._prepare_inputs` raises `KeyError` for any input the model expects and cannot find. The expected names come from the dataset the model was **trained** on, which is not necessarily the current environment's camera names — renaming a camera between recording and inference breaks it.

Validate at model load, not at the first tick: compare the environment's sanitized camera names against the model's expected image inputs and fail with a message naming both sets. A single-camera setup is insensitive to this, because the key collapses to bare `images`. See [design.md](design.md#camera-feature-keys).

### Deletions

`workers/model_worker.py`, `workers/model_worker_registry.py`, `control/inference_poller.py`, `control/queue_mixer.py`, `control/inference_result.py`, `control/sync_mixed_model_integration.py`, plus `ModelRegistryDep` and the registry construction in `core/lifecycle.py`, plus their tests.

`models/utils.py:load_inference_model` stays — the session uses it, and B5's builder mirrors its path construction.

### Accepted regression

A fresh session pays process spawn plus imports before its first tick, where `ModelWorkerRegistry` pre-spawns today. The pre-spawn only hides process creation; `load_inference_model` still runs on demand inside it, and that is the multi-second part.

**Measure, against a threshold decided now.** The metric is the median wall time from the `load_model` command to the first action the policy actually sends. Build the pool if that exceeds **10 seconds**; accept the regression if it does not.

Ten seconds is roughly where a progress indicator stops reading as slow and starts reading as broken. Fixing the number before anyone has built anything keeps the later conversation about a measurement rather than about taste.

If the pool is needed, it is a direct port of `ModelWorker`'s idle → configure → run → idle shape, and B2's reusable owner already lets one process serve consecutive sessions.

### Validation

Against `FakeInferenceModel` returning a fixed chunk:

- a model loading does not stall the loop; ticks continue at fps throughout
- `warmup()` on the loader thread means the first armed tick does not block
- re-arming after a mode excursion sends an action derived from the current observation, not a pre-excursion chunk
- a straggler `RuntimeError` on re-arm produces an `error` event and the session survives
- the model input carries the task string
- the model input's image channel order matches the dataset's
- **with two cameras, the model input keys match the training dataset's feature keys** — and a renamed camera surfaces as a clear error at model load, naming both sets, rather than a `KeyError` at the first tick

Hardware: run a known ACT checkpoint and confirm the rollout matches the old path's behaviour or beats it. The channel-order and task fixes mean a VLA policy should behave _better_, which is worth measuring rather than assuming.

---

## B4 — Recording

### Recording callback

New `application/backend/src/runtime/callbacks/recording.py`.

`on_tick` writes a frame when the session is recording. `TickEvent` carries the observation, the camera frames and the action that was actually sent, so recording needs no separate robot read — which removes a whole class of skew between what was recorded and what was executed.

Gate on `is_recording` **and** a mode other than `hold`. The runtime always sends an action, so hold ticks would otherwise be recorded as if the operator had commanded them.

Keep the callback synchronous. lerobot already runs its own image-writer threads, and ordering against `save_episode` and `discard_episode` must hold.

`build_lerobot_dataset_features` moves to `runtime/dataset_features.py`, taking joint names and camera specifications instead of a `RobotClient`. Move `sanitize_camera_name` with it — it defines the camera feature key, and the runtime's `cameras={...}` mapping must use the same keys or datasets recorded before the migration stop matching models trained on them. See [design.md](design.md#camera-feature-keys).

These two functions are the only parts of `EnvironmentIntegration` that survive.

### Stream callback gains camera frames

Extend the phase A stream callback with base64 JPEG frames keyed by camera UUID, and wrap the whole callback in `AsyncCallback` so encoding leaves the control thread. `AsyncCallback` already replaces borrowed iceoryx2 frames with owned copies before queueing, which is required — those buffers are invalidated by the next read.

If B4 also moves the record page onto the binary stream (below), the base64 path disappears instead and this step is smaller.

### Record page camera panels

Move `features/robots/robot-control/camera-cell.component.tsx` from base64-in-payload to the existing `<WebsocketCamera>` component, already used in three other places.

This requires the preview-stream lock fix to be merged first: `api/camera.py` must pass `is_locked=<fingerprint in locked set>` to `CameraWorker`, so the stream attaches with `overwrite_settings=False`. Without it, adding preview subscribers during recording makes an existing reconfiguration bug more likely to fire, not less.

### Record socket vocabulary

Converge the record socket on `hold | teleop | policy`, matching what phase A applied to the teleoperation socket. This is the right moment because the handler is being rewritten anyway.

Frontend: `features/robots/robot-control-provider.tsx` — the `FollowerSource` type, the `set_follower_source` payload, and the state matchers that settle command promises.

### Deletions

`workers/robot_control_worker.py`, `control/environment_integration.py`, and their tests.

### Validation

- frames are written only while recording and only when the mode is not `hold`
- saving an episode, then discarding one, then saving again produces exactly two episodes
- an idle exit mid-recording finalizes the dataset — the B2 requirement, now testable end to end
- a recorded episode's frames match the frames the model saw, channel order included
- the record page renders camera panels through the binary stream

Hardware: record a short dataset, then train on it, confirming the resulting model behaves. Recording is the input to everything downstream, so a schema or ordering regression here is expensive to find later.

---

## B5 — Inference export

### Builder extension

Add the `PolicySource` fragment to `build_runtime_config`:

```yaml
action_source:
  class_path: physicalai.runtime.PolicySource
  init_args:
    model:
      class_path: physicalai.inference.InferenceModel
      init_args:
        export_dir: ./exports/openvino # relative to the bundle root
        policy_name: act
        device: GPU
    execution:
      class_path: physicalai.runtime.AsyncExecution
      init_args: { request_threshold: 0.5 }
    action_queue:
      class_path: physicalai.runtime.ChunkedActionQueue
      init_args:
        smoother:
          class_path: physicalai.runtime.LerpSmoother
          init_args: { duration_frames: 3 }
    task: "pick up the red cube"
```

The execution and queue settings must be the same values the session uses. That identity is the whole point — see [design.md](design.md#export).

### Bundle

```text
studio-runtime-<name>-<timestamp>.zip
├── runtime.yaml
├── exports/<backend>/
└── README.md
```

`ModelDownloadService.create_backend_archive` and `services/staged_archive.py` already exist, so this extends an existing endpoint rather than adding plumbing.

The README carries the exact command and every value needing local attention:

```bash
physicalai run --config runtime.yaml --run.duration_s=60
```

### UI

The download button, on the model or inference page. This is the deliverable the whole export thread was aimed at: click, download, run the CLI, watch the robot.

### Validation

- the emitted document passes `validate_config` and instantiates through `RobotRuntime.from_config` with a stubbed `InferenceModel`
- the `execution` and `action_queue` fragments are byte-identical to the ones the session constructs — assert against the session's own document, not a literal
- `export_dir` is relative, and resolves correctly from the extracted bundle root
- the bundle extracts and runs on a machine that has never seen Studio

That last check is the real gate. Everything else is a proxy for it.

---

## After B5

The migration is complete when:

- one session type serves the robot controller page, the record page and the inference page
- teleoperation, policy execution and recording are modes and callbacks on one runtime
- roughly 1100 lines of loop, process and queue machinery are gone
- a downloaded bundle reproduces a Studio session through `physicalai run`

Then delete this document and [phase-a-teleop.md](phase-a-teleop.md). [design.md](design.md) stays; human-in-the-loop is the next thing to pick up.
