# Phase A — Teleoperation on the runtime

Replace `TeleoperateWorker` with a `RuntimeSession` built on `physicalai.runtime.RobotRuntime`, and add the config builder that both the session and the export endpoint consume.

Read [design.md](design.md) first. It owns the architecture; this document owns the steps. Where the two appear to disagree, design.md wins and this document is stale.

> Delete this document in the pull request that completes the phase.

## Scope

In scope:

- `RuntimeSession` with a thread host and direct command binding
- `StudioActionSource` with `hold` and `teleop` modes and an empty policy slot
- `build_runtime_config`, emitting a document that both the session and the export use
- An export endpoint returning a runnable `runtime.yaml`
- Mode vocabulary unified on the teleoperation websocket
- `TeleoperateWorker` and its tests removed

Out of scope, and why:

- **Zenoh.** A thread in the API process has no boundary to cross. The contract is defined here; phase B binds it to Zenoh.
- **The export button.** Phase A's export is a teleoperation config, useful primarily as this phase's hardware gate. The user-facing download needs a model, so it lands in B5.
- **Cameras in the session.** The teleoperation page does not display camera frames; previews come from `/api/cameras/ws` independently. `RobotRuntime` accepts an empty camera mapping. The builder still emits camera entries, because the exported config needs them.
- **`PolicySource`.** Phase B.

## Prerequisite

**Bump the `physicalai` pin first, in its own pull request.** Phase A needs `RobotRuntime.stop()`, merged in [#226](https://github.com/openvinotoolkit/physicalai/pull/226) as `cb934cd`. Both `application/backend/pyproject.toml` and `library/pyproject.toml` pin the same revision and move together.

This document is written against the **post-bump** API. Do not start step 5 before the bump lands.

The bump spans more than `stop()`. [#212](https://github.com/openvinotoolkit/physicalai/pull/212) renamed the whole `physicalai.config` public surface between the current pin (`3e163cfe`) and `cb934cd`:

| Before                       | After               |
| ---------------------------- | ------------------- |
| `ComponentConfig`            | `Config`            |
| `ComponentConfigError`       | `ConfigError`       |
| `ComponentImportError`       | `ConfigImportError` |
| `validate_component_config`  | `validate_config`   |
| `normalize_component_config` | `normalize_config`  |

Unchanged, and therefore safe to write against before the bump: `to_config`, `instantiate`, `to_yaml`, `save_yaml`, `load_yaml`.

Two consequences:

- **Only one Studio import breaks.** `application/backend/src/utils/camera_factory.py` imports `ComponentConfig`, inside `TYPE_CHECKING`. The rename also affects a module docstring there and the `_camera_component_config` function name, plus one docstring mention in `application/plugin/src/physicalai_studio_plugin/catalog.py`. Nothing else in the repo references the renamed symbols.
- **Step 5 calls `validate_config`, which exists only after the bump.** At the current pin it is `validate_component_config`. This is correct as written — the document targets the new API deliberately — but it means the builder cannot be written against the current pin.

## Steps

### 1. Define the command and event contract

New `application/backend/src/runtime/contract.py`.

Commands, as a discriminated union of pydantic models. Phase A implements the first two and defines the rest so phase B does not redesign them:

| Command               | Class      | Payload                     |
| --------------------- | ---------- | --------------------------- |
| `set_follower_source` | idempotent | `follower_source`           |
| `disconnect`          | idempotent | none                        |
| `load_model`          | idempotent | `model`, `inference_device` |
| `load_dataset`        | idempotent | `dataset`                   |
| `start_task`          | idempotent | `task`                      |
| `stop_task`           | idempotent | none                        |
| `start_recording`     | idempotent | `task`                      |
| `save_episode`        | **acked**  | none                        |
| `discard_episode`     | **acked**  | none                        |

Each command carries an optional `request_id`. Acked commands require one. See [design.md](design.md#command-classes) for why the split exists.

Events, also pydantic:

| Event         | Payload                                             |
| ------------- | --------------------------------------------------- |
| `observation` | `{feature_name: value}`                             |
| `state`       | `connected`, `follower_source`, plus phase B fields |
| `error`       | `message`, `error_code`                             |
| `lifecycle`   | `event`, `reason`, `metadata`                       |
| `ack`         | `request_id`, `ok`, `error`                         |

Keep the contract free of transport concerns: no queue types, no Zenoh key expressions, no websocket references. It is data plus a `CommandMailbox` protocol with `apply(command)` and an event sink protocol.

### 2. Split robot construction from adapter wrapping

`application/backend/src/robots/robot_client_factory.py` currently builds a driver, wraps it in `SharedRobot`, and then wraps that in `PhysicalAIRobotAdapter`. The runtime needs the middle layer.

Add:

```python
async def build_shared_robot(self, robot: Robot) -> tuple[SharedRobot, RobotCatalogDefinition]:
    """Build a SharedRobot and return it with its catalog definition.

    The runtime consumes SharedRobot directly — it satisfies the physicalai
    Robot protocol. The definition carries adapter options the runtime layer
    needs for feature naming and effort gain.
    """
```

`build()` keeps its current signature and delegates to it, so the calibration wizard, identify jog and hardware probes are untouched. Do not change `PhysicalAIRobotAdapter`.

### 3. Feature naming helpers

New `application/backend/src/runtime/features.py`, lifting the naming logic out of `PhysicalAIRobotAdapter`:

```python
def feature_names(joint_names: list[str], *, include_velocities: bool) -> list[str]:
    """Return ["<joint>.pos", ...] plus ["<joint>.vel", ...] when requested."""

def observation_to_dict(joint_names, observation, *, include_velocities) -> dict[str, float]:
    """Map a RobotObservation onto the feature-name dict the UI and dataset use."""
```

`include_velocities` comes from `definition.adapter_options`. WidowXAI sets it; SO101 does not.

Verified equivalence: `RobotObservation.state` already concatenates positions then velocities in the same order these helpers produce, so nothing about the state vector changes.

### 4. `StudioActionSource`

New `application/backend/src/runtime/action_source.py`.

```python
class StudioActionSource:
    """Selects the action to send, per Studio's current mode.

    Implements physicalai's ActionSource protocol structurally. Modes are
    selected between live delegates rather than switched between lifecycle
    branches, so a policy delegate added later keeps running across mode
    changes. See design.md.
    """
```

`update()` does, in order:

1. Drain the command mailbox. Mode changes take effect on the tick they arrive.
2. Compute `leader_action` when a leader is present.
3. Compute `policy_action` when a policy delegate is present — always `None` in phase A.
4. Arbitrate on the current mode and return one action.

Mode behaviour:

- **`hold`** — return the latched target. Latch it when the mode is entered, from the follower's current position, and return that same value every tick afterwards. Do not re-read. See [design.md](design.md#modes) for why.
- **`teleop`** — read the leader's observation, map it to an action, and push the follower's efforts back to the leader when `external_effort_gain` is set and the observation carries efforts. Replaces `read_forces`/`set_forces`.

`connect(bus, session_id)` caches both for phase B, connects the leader if the session has not, and nothing else. `disconnect()` clears per-run state and **does not disconnect the leader** — the session owns it.

Errors reading the leader must not end the session. Log, fall back to the last good leader action or the hold target, and continue. A leader hiccup is routine.

**But bound it.** Past `3 * fps` consecutive failures — matching upstream's own ceiling, `RobotRuntime._max_consecutive_error_ticks = int(3 * fps)` — drop to `hold`, emit one `error` event, and stop retrying. Log once at that transition, not once per tick.

Without the bound, a leader whose cable is pulled mid-teleoperation logs a traceback thirty times a second indefinitely, while the follower silently holds a stale target and the browser still shows a connected session. The idle timeout does not save this case, because the session still has a supervisor attached. It is the same hazard the [lifecycle section](design.md#lifecycle) treats as a safety property rather than cleanup — an arm under torque with nobody effectively watching — reached by a different route.

### 5. Config builder

New `application/backend/src/runtime/config_builder.py`.

```python
def build_runtime_config(
    *,
    follower: Robot,
    leader: Robot | None,
    cameras: list[Camera],
    fps: float,
    port_resolver: PortResolver,
) -> dict[str, Any]:
    """Assemble a physicalai runtime config document from Studio rows.

    Plain data, not to_config() on a live object: InferenceModel loads weights
    on construction and StudioActionSource is deliberately not exportable.
    """
```

Emits the CLI document shape:

```yaml
runtime:
  robot:
    class_path: physicalai.robot.SharedRobot
    init_args:
      name: <robot uuid>
      robot:
        class_path: physicalai.robot.SO101
        init_args:
          port: /dev/serial/by-id/...
          calibration: { ... } # inline
          role: follower
          unit: normalized
  action_source:
    class_path: physicalai.runtime.TeleopSource
    init_args:
      leader:
        class_path: physicalai.robot.SharedRobot
        init_args: { name: <leader uuid>, robot: { ... } }
  cameras:
    overhead:
      class_path: physicalai.capture.SharedCamera
      init_args:
        camera: { class_path: physicalai.capture.UVCCamera, init_args: { ... } }
  fps: 30.0
```

Requirements:

- Validate the emitted document with `physicalai.config.validate_config` before returning.
- Calibration is inline. `SO101Calibration.to_config_value()` produces the mapping and `SO101.__init__` accepts a dict, so no calibration file belongs in a bundle.
- Resolve `/dev/serial/by-id/...` from the stored serial number where possible; otherwise emit the raw port with a `CHANGE_ME` comment. Same for camera devices via `/dev/v4l/by-id/...`.
- `SharedRobot.name` is the robot UUID, matching what the session uses.
- Camera mapping keys are `sanitize_camera_name(camera.name)`, matching the dataset feature keys. See [design.md](design.md#camera-feature-keys). Phase A's session runs with no cameras, but the emitted document must already be correct, because phase B4's recording depends on these keys.

The `port_resolver` seam keeps the builder unit-testable without a udev tree.

### 6. `RuntimeSession` and the thread host

New `application/backend/src/runtime/session.py` and `application/backend/src/runtime/hosts/thread_host.py`.

`RuntimeSession` owns the devices and treats the runtime as a view:

```python
class RuntimeSession:
    def __init__(self, document, *, event_sink): ...
    async def setup(self) -> None: ...        # build + connect devices from the document
    def build_runtime(self) -> RobotRuntime: ...
    def apply(self, command) -> None: ...     # into the mailbox
    def run(self, stop_signal) -> None: ...   # runtime.connect(); runtime.run(stop_event=...)
    async def teardown(self) -> None: ...     # disconnect devices
```

Two rules, both easy to get wrong and both worth a comment in the code:

- **Never `with runtime:`.** `__exit__` disconnects devices, which breaks rebuilds. Call `connect()` explicitly.
- Devices are session-owned. Only `teardown()` disconnects them.

The stop signal adapter, satisfying `physicalai.runtime.StopSignal` structurally:

```python
class WorkerStopSignal:
    """Bridges Studio's stop semantics onto physicalai's StopSignal protocol."""

    def __init__(self, worker: StoppableMixin) -> None:
        self._worker = worker

    def is_set(self) -> bool:
        return self._worker.should_stop()
```

`should_stop()` already combines the scheduler's global stop event, the per-worker stop, and parent-process death, so application shutdown ends sessions with no extra wiring. Do not use `isinstance` against `StopSignal`; it is deliberately not runtime-checkable.

`RuntimeThreadHost(BaseThreadWorker)` runs `setup()`, then `run()`, then `teardown()`.

### 7. Stream callback

New `application/backend/src/runtime/callbacks/stream.py`.

- `on_tick` → an `observation` event built with `observation_to_dict`
- `on_lifecycle("start")` → set a ready flag and emit `state` with `connected: true`. This is the correct readiness signal: it fires after `action_source.connect()`, so the follower and the leader are both proven connected.
- `on_lifecycle("shutdown")` → emit `lifecycle` carrying `reason` from `last_run_reason`

Phase A does not need `AsyncCallback`, because the payload is a small float dict. Phase B adds it when camera frames arrive.

### 8. Rewire the endpoint

`application/backend/src/api/robot_control.py`.

The handshake is unchanged: the first client message carries `follower_id` and optionally `leader_id`. Then resolve both robots, call `build_runtime_config`, construct the session, start the host, and pump events to the socket.

Error surfacing is simpler than the worker it replaces, because exceptions propagate:

| Signal                        | Meaning                                          | Response                                         |
| ----------------------------- | ------------------------------------------------ | ------------------------------------------------ |
| `runtime.connect()` raises    | follower or camera setup failed                  | `translate_robot_error` → `error` event          |
| `run()` raises before `start` | action source setup failed, typically the leader | same                                             |
| `on_lifecycle("start")`       | ready                                            | `state` with `connected: true`                   |
| `on_lifecycle("shutdown")`    | run ended, with a reason                         | clean close, or `error` when `reason == "error"` |

`#226` moved `action_source.connect()` and the start event inside the try block, so a failure there still runs `_shutdown()` **and** propagates. No `setup_error` field and no `wait_until_loaded` polling — drop both patterns from the old worker.

### 9. Export endpoint

`GET /api/projects/{project_id}/environments/{environment_id}/runtime-config`

Returns `build_runtime_config(...)` as YAML via `physicalai.config.to_yaml`, with a `Content-Disposition` attachment header. No UI in this phase.

### 10. Mode vocabulary on the teleoperation socket

Backend accepts `{"event": "set_follower_source", "data": {"follower_source": "hold" | "teleop"}}` and reports the same string in `state`.

Frontend, roughly ten lines:

| File                                                          | Change                                                              |
| ------------------------------------------------------------- | ------------------------------------------------------------------- |
| `features/robots/use-joint-state.ts:39-42`                    | replace the `RobotActionReadState` numeric enum with a string union |
| `features/robots/use-joint-state.ts:46,53`                    | `follower_source` becomes that union                                |
| `features/robots/use-joint-state.ts:116-121`                  | send `{ follower_source: value }` instead of a bare number          |
| `features/robots/environment-form/cells/robot-cell.tsx:21,60` | compare against and send `'teleop'` / `'hold'`                      |

The record socket keeps its own vocabulary until phase B rewrites that path. The two are inconsistent in the meantime; that is deliberate, to avoid touching code scheduled for deletion.

### 11. Delete `TeleoperateWorker`

Remove `application/backend/src/workers/teleoperate_worker.py` and port `tests/workers/test_teleoperate_worker.py` to `tests/runtime/`.

The port is a simplification. The old tests fake `RobotClient`, a twelve-method abstract base class. The new ones fake `physicalai.robot.interface.Robot`: five methods and a property.

## Tests

New `application/backend/tests/runtime/`.

`fakes.py`:

- `FakeRobot` — `connect`, `disconnect`, `get_observation`, `send_action`, `is_connected`, `joint_names`. Records sent actions so tests can assert on them.
- `FakeObservation` — a dataclass satisfying `RobotObservation`, with optional `sensor_data` for velocity and effort cases.

Cases:

| Test                       | Asserts                                                                                                                          |
| -------------------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| hold latches               | with the follower drifting, every action sent equals the target latched on entry                                                 |
| teleop forwards            | actions sent match the leader's positions                                                                                        |
| haptics gated              | efforts reach the leader only when `external_effort_gain` is set and the observation has them                                    |
| leader read error survives | a raising leader logs and continues; the run does not end                                                                        |
| leader failure is bounded  | a permanently failing leader drops to `hold` after `3 * fps` ticks and emits exactly one `error`, rather than logging every tick |
| mode switch timing         | a command applied between ticks takes effect on the next tick                                                                    |
| stop ends the run          | setting the stop signal ends `run()` with `last_run_reason == "stop_requested"`                                                  |
| setup error propagates     | a failing follower connect raises out of `connect()` and maps to an `error` event                                                |
| lifecycle to state         | `on_lifecycle("start")` produces exactly one `state` with `connected: true`                                                      |
| builder shape              | the emitted document passes `validate_config` and round-trips through `RobotRuntime.from_config` with fakes                      |
| builder calibration inline | calibration appears as a dict in the document, with no filesystem path                                                           |
| port resolution            | a known serial number yields a by-id path; an unknown one falls back with a `CHANGE_ME` marker                                   |
| websocket contract         | the exact JSON shapes for `observation`, `state` and `error`                                                                     |

The contract test matters more than it looks. These payload shapes exist only in code — nothing in `openapi-spec.d.ts` describes the websocket messages — so a silent break is possible today.

## Validation

```bash
# backend
cd application/backend && uv run pytest
prek run --all-files application/backend/

# ui
cd application/ui && npm install && npm run test
prek run --all-files application/ui/
```

## Hardware gate

The phase is not done until both halves work on a real SO101 pair.

1. **Studio path.** Open the robot controller page. Joints stream. Toggling teleoperate moves the follower from the leader. Toggling it off leaves the arm holding position without drift. Closing the tab ends the session and logs `stop_requested`.
2. **Export path.** Download the config for the same environment, edit any `CHANGE_ME` values, and run:

   ```bash
   physicalai run --config runtime.yaml --run.duration_s=30
   ```

   The follower must track the leader the same way it does in Studio.

The second check is the point of the whole phase. If a config generated from Studio's own database drives the arm through `physicalai run`, then export fidelity is demonstrated rather than asserted, and phase B inherits that.

Watch for two behaviour changes that are expected:

- Connect order is follower, then cameras, then leader. It was leader first.
- `goal_time` becomes three ticks, where teleoperation used two. SO101 ignores `goal_time`, so this is invisible there; check it on WidowXAI if that hardware is available.

## Pull requests

| Order | Title                                                                                    |
| ----- | ---------------------------------------------------------------------------------------- |
| 1     | `chore(deps): bump physicalai to cb934cd` — owner-supplied, includes the `Config` rename |
| 2     | `refactor(robots): split shared-robot construction from adapter wrapping`                |
| 3     | `feat(runtime): run teleoperation on physicalai RobotRuntime`                            |
| 4     | `feat(runtime): export a runnable teleoperation config`                                  |

Pull requests 3 and 4 can merge together if the builder lands with the session that consumes it, which is the arrangement that makes export verified rather than merely tested.
