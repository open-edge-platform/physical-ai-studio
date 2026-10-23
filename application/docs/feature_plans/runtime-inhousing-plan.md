# Runtime Inhousing Plan

## Why?

The current physicalai runtime does not allow for the same flexibility that is needed in the studio. The studio needs to be able to switch models, datasets, run human-in-the-loop. Reimplementing a flexible version of the Runtime session into physicalai would be possible, but it might be too complicated for the use case in physicalai, where the main goal is to run a model on the robot.

Specifically the planned "Control" feature will be extremely handy for many of the issues that we are facing when implementing new stuff in the robot.

## Plan

Since this will be a rather big refactor we will work on it in steps using a feature branch, only merging said feature branch at the end.
The current implementation of runtime is first removed and the endpoints made into placeholders, or buttons disabled. After that reimplementation is gradually added.
This is done because a gradual change is very hard and the feature-branch approach allows for incomplete or broken states of the feature branch.
A detailed AI implementation plan exists in the second part of this document.

Now onto the architecture. A new RuntimeSession is created. This session is responsible for handling lifecycle of the session. This session can load an environment, dataset, model and control.

## Lifecycle handling

A lot of components in these acquire real hardware. This means that it needs to be acquired properly and released once done.
Same goes for Processes that are started which need to be terminated when the RuntimeSession stops.

In order to do this we will use contextmanagers. The goal is to have the lifecycle be handled by the thread or process itself:

```python

class EnvironmentContext:
    follower: RobotWorker
    leader: RobotWorker | None
    cameras: dict[str, CameraWorker]

class EnvironmentWorker(BaseThreadWorker[EnvironmentContext]):
   def __init__(self, environment: EnvironmentWithRelations, stop_event: EventClass, daemon: bool = False):
        super().__init__(stop_event=stop_event, daemon=daemon)
        self.environment_config = environment

   async def lifecycle(self):
        # one robot for now until refactor of environment
        try:
            robot = self.environment_config.robots[0]

            #logic to get robot from config:
            follower = self._robot_worker(robot)
            leader = None
            if robot.tele_operator:
                leader = await self._robot_worker(robot.tele_operator.robot)

            cameras = { camera.name, await self._camera_worker(camera):  for camera in self.environment_config.cameras}

            yield EnvironmentContext(
                follower=follower,
                leader=leader,
            )
        except Exception as e:
            logger.exception(f"Unhandled exception in {self.name}")
            raise e
        finally:
            follower.disconnect()
            if leader:
                leader.disconnect()

    async def run_loop(self, context: EnvironmentContext):
        # do things with context.

```

The BaseProcessWorker or BaseThreadWorker will inherit from thread or mp.process and critically the run function will do this:
```Python

def run(self) -> None:
    async with self.lifecycle() as context:
        self.run_loop(context)
```



## Communication and Data Store

Some of these components live in different processes for stability. This means that cross process communication needs to be established. Two will be added:

1. A Store

A store where the robot state, actions and observations, task (current joint state, next joint state, camera views respectively and task) are kept.
These are kept in SharedMemory so that all components can easily read and write to it.
The shape of the store depends on what was loaded first.
A dataset for example has a specific feature set (robot state and cameras), the actions are simply a mirror of the robot state.
A model has specific model inputs which decide the shape and an Environment has robots and cameras which decide the robot state, actions and observations.
The store initially only has state (array), action (array)  and observations (dict of arrays).
This distills down to state being a Sample.
The Sample contains the shared memory, the feature names (list of joints or just a name in case of observation (e.g. overhead, wrist)) and a timestamp of when it was set.

2. Events over Zenoh.

To be able to easily handle events components need to be able to listen for specific events over zenoh. The consumer of the session (e.g. a websocket connection from fastapi) needs to be able to adjust things in the subcomponents.

## Workers

Most components run on their own frequency. A dataset can be recorded at 50hz, robot at 100hz and model at 30hz for example. It is then also possible to run inference using a model of 30hz but record at 50hz. Yes, there will be some stepping, but that is fine.

### Robot

Every robot needs their own worker in a process. It connects to the robot and disconnects it after done. It runs on a specific hz and takes writes their features to Store state and reads the action from the Store and writes it to the robot using the timestamp to get the goal_time. Not every robot uses goal_time and that is fine for now. No complex interpolation

### Camera

CameraWorker is a thread worker that starts a shared_camera from physicalai library. It then writes to the Store's observation shared memory of that camera at the desired hz.

### Dataset

A dataset worker runs in a separate process. It listen for start, stop, discard recording events. Once recording it writes the store's values as a frame to the dataset. This is at a desired hz.

### Control

In order to allow for very flexible control of the entire system we need a control system.
1. TeleoperationControl reads from leader and writes to the store.action.
2. ModelControl takes a loaded model and reads the current action from the model giving the current store frame and writes it to the store.action.
3. MoveControl will gradually move the robot from the current position to the target position over a time by writing to the store.action.

However, it needs more flexibility than just writing to the storte.action. A future human-in-the-loop Control would have to deal with the model, leader and perhaps even set the leader position. To make the Control as powerful as possible it is ran in a seperate process or thread and it can get zenoh events. Allowing for seperate hz's for whatever purpose it needs. Only one control can be activate at the same time.

## RuntimeSession

RuntimeSession all ties this together. The RuntimeSession holds the current store. On load_environment it will do so. Detailed errors will be thrown when a load_dataset (or other components) is called with a config that does not match current feature spec of the store. All subworkers are added to the Context of the RuntimeSession so that it cleans up after itself.

---

# Implementation Plan

## Starting point

What we have today and what we keep from it:

- `src/runtime/` drives everything through `physicalai.runtime.RobotRuntime`: a single loop at one fps, with Studio plugging in an `ActionSource` and some callbacks. It runs in a detached subprocess found over zenoh (`transport/`, `owner.py`, `hosts/`). This is the code we replace.
- `physicalai.runtime` also gives us the policy execution (`PolicySource`, `ChunkedActionQueue`/`ActionChunk(t0, dt)`, `AsyncExecution`, `LerpSmoother`, RTC). Bringing this in-house is the real work behind `ModelControl`.
- We keep depending on `physicalai.robot`, `physicalai.capture` (`SharedCamera`), `physicalai.inference` (`InferenceModel`) and `physicalai.config`. Studio stops *running* on `physicalai.runtime`.
- **Runtime config export stays.** The environment and model "export runtime config" endpoints (`api/environments.py`, `api/models.py`) write a YAML that runs on `physicalai.runtime.RobotRuntime` *outside* Studio, which is the deployment path. They only build a Config document and never run it, so they move to `services/runtime_export.py` and keep their `physicalai.runtime` class paths.
  - **Why both exist:** `physicalai.runtime` stays the lean way to run a policy on a robot without Studio. Studio's own runtime has different needs (concurrent rates, swappable controls, recording, HIL), so the two are allowed to diverge internally.
  - **Export parity is a hard requirement:** whatever a user can run in Studio with `ModelControl` must stay exportable as a runnable `physicalai.runtime` config, as it is today.
    - `ModelControl`'s settings (fps, sync or async execution, chunk threshold, smoothing) are chosen so each one maps to a `physicalai.runtime` equivalent: `SyncExecution`/`AsyncExecution`, `ChunkedActionQueue`, `LerpSmoother`.
    - `services/runtime_export.py` builds the export from the same `ModelControlConfig` the session uses, not from a separate set of defaults.
    - A Studio-only setting with no runtime equivalent (e.g. HIL) is left out of the export with a `# NOT EXPORTED:` comment, the same way `CHANGE_ME` paths are flagged today.
  - **Test:** a round-trip test exports an environment + model and runs `Config.from_dict(...).instantiate()` against `physicalai.runtime` with fake devices. This catches drift when either side changes.
- The `rhecker/flexible-runtime` prototype (`runtimev2/`) already works through most of this design. We port from it instead of rewriting:
  - `shared_store.py` (shm layout, create and attach by name) becomes the base of the Store, minus its seqlock.
  - `workers/process.py` and `workers/base.py` (ready/failure propagation, kill a wedged child) become the base of the workers.
  - `environment.py`, `features.py` and `dataset_layout.py` become the feature spec, derived from an environment or a dataset.
  - `control/*` gives us the Control shape.
  - `workers/robot.py` gives us the robot loop. We drop its interpolation, since the plan says goal_time only.

## Package and branch

- Feature branch `feature/runtime-inhousing`. Each phase below is its own PR into that branch, and `main` gets one merge at the end.
- New code goes in `src/runtime_session/`.
- **The old runtime is removed first (Phase 0), not at the end.** Running the old and new runtimes side by side would mean keeping both working on one branch. Instead, features that depend on the runtime are disabled in the UI with a "work in progress" state and come back phase by phase. The feature branch may have fewer features than `main` while it is in progress; `main` only gets the result once everything is back.

## Proposed changes to the sketch

1. **The worker base is generic over its context.** `lifecycle()` is an `@asynccontextmanager` that runs inside the thread or child process. `run()` becomes:
   ```python
   def run(self) -> None:
       asyncio.run(self._main())

   async def _main(self) -> None:
       try:
           async with self.lifecycle() as context:
               self._ready.set()
               await self.run_loop(context)
       except BaseException as exc:
           self._report_failure(exc)  # text over an mp.Queue for processes
       finally:
           self._ready.set()
   ```
   The parent side uses the worker itself as an async context manager. `async with worker:` starts it, waits for `ready`, and re-raises a startup failure. Leaving the block sets `stop`, joins with a timeout, and kills the worker if it is still alive. `RuntimeSession` then holds one `AsyncExitStack` and pushes every worker onto it, which is what "added to the Context of the RuntimeSession" means.
2. **Processes build their hardware in the child.** The worker object gets pickled, so it carries config only (DB rows or Config dicts), never drivers. `lifecycle()` does the instantiation.
3. **Environment is a composite context manager, not a worker.** In the sketch, `EnvironmentWorker` has no loop of its own: the robot runs in its own process and the cameras in their own threads. `load_environment` is just an `AsyncExitStack` that enters the follower `RobotWorker` and the `CameraWorker`s and connects the leader driver, **in parallel**. That keeps today's parallel connect, and if anything fails to connect, everything already connected is released.
4. **The session owns what is loaded; controls borrow it.** The environment owns the follower `RobotWorker`, the `CameraWorker`s and the leader. A loaded model owns its policy. A control only gets references (`TeleopControl` borrows the leader, `ModelControl` the policy), so creating or stopping a control never connects, disconnects or reloads anything. That matters because control switches often (toggling teleop off sets control to `None`), and some robots reset on disconnect. The leader is loaded as a plain driver: the environment connects it on load and disconnects it on unload, with no worker, no loop and no store slot. A borrowed driver or policy cannot cross a process boundary, so borrowing controls run as `ThreadWorker`s in the session's process.

## Store design

```
FeatureSpec
  joint_names:  tuple[str, ...]          (state and action share them)
  observations: {camera_key: ImageSpec(h, w, c, dtype=uint8)}
  task:         str  (not in shm, see below)

Store (shm, one block per session, name = f"pai-{session_id}")
  state            -> Sample(values, names, timestamp)
  action           -> Sample(values, names, timestamp)
  observation[cam] -> Sample(frame,  name,  timestamp)
```

The joint names come namespaced the way the dataset stores them (e.g. `shoulder_pan.pos`). Then a dataset's `observation.state` and `action` names compare directly against the spec without a mapping step.

- **No locking.** Every feature (joint vector or camera frame) is a single buffer plus a timestamp, and readers copy it out without coordinating with the writer. A read that overlaps a write can be torn: some joints or rows from the previous sample, the rest from the next. We accept that because consecutive samples are only one tick apart and the world moves continuously, so a torn sample is still physically plausible.
  - **No garbage values:** aligned 8-byte stores are atomic on x86-64 and arm64. A torn joint vector mixes two real readings and never contains half-written floats.
  - **Write order:** the writer stores the values first and the timestamp last. A reader that sees a timestamp therefore never gets the zeros of a feature that has not been written yet; NaN still means "not written yet".
  - **Fallback:** if tearing turns out to matter somewhere (for example in recorded frames), add a seqlock or extra slots for that feature only. Readers don't need to change for that.
- **Timestamps:** `time.monotonic()`, which is system-wide on Linux and comparable across processes. NaN means "not written yet".
- **Task:** too awkward to keep in shm, since it is a variable-length string. Put it in session state and broadcast changes over zenoh; the dataset and model workers cache the latest value.
- **Lifetime:** `RuntimeSession` creates and unlinks the block, and children attach by name plus spec. At backend startup, sweep `/dev/shm/pai-*` blocks whose session no longer exists, because a SIGKILLed session leaves its block behind.
- **First loaded defines the shape:**
  - an environment supplies the robot and cameras;
  - a dataset supplies the state/action names and camera shapes;
  - a model supplies its input features.

  Every later load is checked against the spec with `spec.check_compatible(other) -> list[Mismatch]`. Compatible means everything the newcomer needs exists: joint names equal and in order, and each of its cameras present with the same shape. Extra cameras in the store are fine. A mismatch raises `FeatureSpecMismatchError` listing each difference, for example "camera `wrist`: dataset 480x640x3, environment 720x1280x3" or "joint names differ at index 3". Once every shape-defining component is unloaded, the store is dropped and the next load creates a fresh one.

## Event bus (zenoh)

- One zenoh session per process, peer mode, scouting off. The session process hosts it on a Unix socket (`<tmp>/pai-{session_id}.sock`) and its worker processes connect only to that socket, so nothing listens on the network and sessions cannot hear each other. `eclipse-zenoh` is already a direct dependency.
- Most traffic never touches zenoh: the websocket calls the session directly, and controls are threads in the session's process. Zenoh carries what crosses into worker processes, and events controls listen to.
- Key layout is `pai/{session_id}/…`:
  - `cmd/{component}`: commands into a component (pub/sub, pydantic payloads, the same discriminated-union style as `contract.py`).
  - `evt/{component}`: state and errors out of a component, which the session aggregates for the websocket.
  - `rpc/{component}/{op}`: a zenoh queryable for commands that need an ack or a result (save/discard episode, load model). It always has a timeout.
- `runtime_session/bus.py` provides `Bus.host(session_id)` / `Bus.join(session_id)` with `publish`, `subscribe` (a `Subscription` with `await next()` and a non-blocking `poll()` for fixed-rate loops), `request(key, message, response, timeout=)` and `serve(key, model, handler)`. Models may be discriminated unions. `subscribe` and `serve` return a `Declaration` to close or use as a context manager, so a control undeclares what it added on the borrowed session bus when it is switched out. Controls never open a bus of their own: they run in the session process and subscribe from their own thread's loop. A handler's exception is raised in the requester as `BusRequestError`; no reply in time raises `TimeoutError`.
- Pub/sub is fire-and-forget: a message published before a new subscription has reached the publisher's peer is lost. Anything that must arrive (start/stop recording, save, load) uses `request`.
- Telemetry (joints and frames) for the UI does **not** go over zenoh. The FastAPI process attaches to the Store and samples it at UI rate. That is cheaper and removes today's bounded event queue.

## Workers

| Worker | Kind | Rate | Reads | Writes |
|---|---|---|---|---|
| `RobotWorker` (follower) | process | robot hz (e.g. 100) | `action` | `state` |
| `CameraWorker` | thread (`SharedCamera` already runs its own publisher process) | camera fps | – | `observation[cam]` |
| `DatasetWorker` | process | dataset fps | full snapshot | LeRobot frames |
| `Control` (exactly one) | thread in the session process (borrows session-owned parts) | its own hz | whatever it needs | `action` |

- **RobotWorker:**
  - On connect it seeds `action` with the measured position, so with no control running the arm holds still.
  - Before driving, it checks the robot's joint order against the spec; the prototype already does this.
  - **Action timestamp:** the time the action *should have happened*, which is usually now. A teleop or move control writes `now`. A model writes `t0 + i*dt` for the step it samples.
  - **goal_time:** each tick, `goal_time = max(action.timestamp - previous_tick_timestamp, 1/robot_hz)`. The `1/hz` floor stops the robot from being asked to jump instantly, so in practice goal_time will mostly be `1/hz`. There is no interpolation, and robots without goal_time ignore it.
  - Any timeout or exception ends the worker. The session sees the child die and emits an error. There are no silent retries.
- **DatasetWorker:**
  - Commands are start, stop, save and discard; save and discard are RPCs and acked.
  - **Join timeout ≥ 60 s.** Finalizing copies the cache back to the dataset, and killing the process mid-copy loses the dataset. This is the main lesson to carry over from `session.py` today, along with "discard the open episode on teardown".
- **Controls:**
  - `MoveControl` interpolates from the current state to a target over T and then holds. It is also the right thing to run when switching between controls.
  - `TeleopControl` borrows the environment's leader driver, and each tick reads it and writes `action`. Robot-specific subclasses can drive the leader too (set its position, change torque). Stopping it leaves the leader connected.
  - `ReplayControl` plays a loaded dataset's episode into `action` at the dataset fps, writing each frame's timestamp as `start + frame_index / fps`. It needs a loaded dataset and an environment whose spec matches it.
  - `ModelControl` borrows the loaded policy, snapshots the store into an observation, runs inference with `t0 = observation timestamp`, and samples a clock-indexed chunk queue at control hz. This is where the time-based `ActionChunk` work is ported over. Order: sync first, then async with a separate inference thread, then smoothing, and RTC last. Open for Phase 8: inference then runs in the API process, so measure whether Python-side preprocessing holding the GIL makes the websocket stutter before settling on that.
- **Switching control:** stop the old control (join), re-seed `action` from the current state, then start the new one. That keeps the one-active-control rule structural instead of arbitrated.

## RuntimeSession API (asyncio, in the FastAPI process)

```python
class RuntimeSession:
    async def load_environment(self, env: EnvironmentWithRelations) -> None
    async def unload_environment(self) -> None
    async def load_dataset(self, dataset: Dataset) -> None
    async def unload_dataset(self) -> None
    async def load_model(self, model: Model) -> None
    async def set_control(self, control: ControlConfig | None) -> None  # discriminated: hold/teleop/model/move/replay
    async def set_task(self, task: str) -> None
    def store(self) -> StoreReader
    async def aclose(self) -> None  # unwinds the AsyncExitStack in reverse
```

- **At most one environment at a time.** Calling `load_environment` while another is loaded raises an error; the caller unloads first. An environment has exactly one robot for now (`robots[0]`), and loading one with more robots raises an error that says so.
- **Lifetime:** the session lives in the FastAPI process and **belongs to the websocket that opened it**. Closing the socket calls `aclose()`, and backend shutdown (`core/lifecycle.py`) closes whatever is left. Nothing is reattached: a page reload opens a new session. The `RuntimeSessionRegistry` only lists live sessions and enforces claims.
- **Claims:** the registry enforces exclusive claims on robots and cameras across sessions. It reuses `camera_claims` and reports `RuntimeSessionBusyError` to the setup and calibration workers.

## UI

### Today's edges

| Surface | Backend today | Problem |
|---|---|---|
| Camera preview (camera form, environment form, record and inference views) | `/api/cameras/ws`, one `CameraWorker` and one socket **per camera** | Record and inference pages open N camera sockets next to the session socket, so frames and joints come from different places at different rates. |
| Robot detail and robot form | `/robots/{id}/observations/ws`, a `SharedRobot` attach | Separate code path, and its own copy of the error mapping. |
| Environment preview (`robot-cell.tsx`) | `useJointState` → `/runtime/ws?follower_id=` (detached session keyed by follower) | A preview starts a full control session and relies on reattach and idle timeouts. |
| Record and inference | `RuntimeSessionProvider` → same `/runtime/ws`, plus camera sockets | Reattach and handshake logic in `runtime-session-provider.tsx` and `use-joint-state.ts`. |

### Target

Three levels, each a strict superset of the one below it. Every one uses the same workers and the same Store, so preview, recording and inference all run the same code.

1. **Single camera.** `/api/cameras/ws` attaches a `SharedCamera` and streams JPEG at the camera fps, with no worker or store. It's used by the camera form.
2. **Single robot.** `/robots/{id}/observations/ws` opens a `RobotWorker` in **observe-only mode** (it never sends an action, never changes torque) writing into a one-robot store, and streams its joints. It's used by the robot detail page and the robot form.
3. **Runtime session.** `/api/projects/{project_id}/runtime/ws` creates a `RuntimeSession` for the lifetime of the socket. The UI sends commands to load things into it:
   - the environment form preview loads **only the environment**;
   - the record page loads environment + dataset;
   - the inference page loads environment + model.

**Levels 1 and 2 are for the forms only.** Pages that have a session (environment preview, record, inference) **never** open a camera or robot socket. Their camera and joint views read from the `RuntimeSessionProvider`'s refs, so a robot is connected exactly once, by the session's `RobotWorker`.

**Device already held by a session:** a level 1 or 2 socket asking for it gets a clear `device_in_use` error naming the session, instead of a second connection. Reading from the session's store would let a form preview a held device, but it is not needed for the forms and we can add it later.

### Session websocket protocol

Client → server: JSON commands, each with a `request_id`. **Every** command gets an `ack` with `ok` and either an error or a result, which removes today's split between fire-and-forget and acked commands.

```
load_environment {environment_id}   unload_environment
load_dataset {dataset_id}           unload_dataset
load_model {model_id, device}       unload_model
set_control {type: hold|teleop|model|move|replay, ...}
set_task {task}
start_episode  stop_episode  save_episode  discard_episode
```

Server → client:

- `state` (JSON, sent on change): what is loaded, which control is active, recording status, episodes recorded, and the store spec, so the UI knows the joint and camera names.
- `error` (JSON): component, message, error_code, and whether the session survived it. A robot worker dying is fatal to the environment, not to the socket.
- `observation` (**~20 Hz**, configurable via `?hz=`): one binary message per tick holding the joints and the camera frames together, read from the store.
  - Layout: `[u32 big-endian header length][header JSON][jpeg bytes]`. The header is `{type, timestamp, state: {joint: value}, action: {...}, frames: [{key, timestamp, size}]}`, and the JPEGs follow in the order of `frames`.
  - `frames` lists only frames newer than the last ones sent, so a 15 fps camera isn't encoded 20 times a second.
  - Binary avoids the ~33% base64 overhead of JPEGs inside JSON.
  - `ack`, `state` and `error` stay JSON text messages.

JPEG encoding runs off the event loop (`asyncio.to_thread`, one call per new frame, in parallel). One sampler task per session reads the store and sends one observation per tick.

### Frontend

- **One `RuntimeSessionProvider`, rewritten.**
  - It opens the socket and sends the loads its props imply (environment, then optional dataset and model).
  - It exposes `state`, typed command mutations (one promise per ack), and refs for observation and frames.
  - It drops the handshake, reattach and idle logic in today's `runtime-session-provider.tsx` and `use-joint-state.ts`.
- **20 Hz data stays out of React state.** Observations and frames go into refs or a small external store (`useSyncExternalStore`). Each camera renders by drawing an `ImageBitmap` onto a canvas, so a tick does not re-render the page.
- **`CameraView` and `RobotJointsView` are dumb components** that take their data from a source:
  - inside a provider, `useSessionCamera(key)` and `useSessionJoints()` read the provider's observation and frame refs, and no extra socket is opened;
  - on the forms, `useCameraSocket(camera)` and `useRobotSocket(robot)` use the standalone sockets.

  That removes the duplicated `camera-cell.tsx` and `camera-cell.component.tsx`. The record and inference viewers drop their per-camera `WebsocketCamera` sockets.
- The environment preview becomes `<RuntimeSessionProvider environment={env}>` with the existing cells. The record page adds `dataset`, and the inference page adds `model`.
- The runtime-sessions footer and list come back on the registry in Phase 5. "Stop" closes the session, and the owning socket receives a final `error` with `session_closed`.
- Regenerate the OpenAPI types at each phase that changes a schema. The ws message types become pydantic models included in the OpenAPI schema, so the UI gets them typed.

## Phases (each one a PR into the feature branch)

0. **Remove the current runtime.**
   - Backend:
     - Delete `src/runtime/` (session, owner, hosts, transport, action source, callbacks, policy loader), `api/runtime_ws.py`, `api/runtime_sessions.py`, `services/runtime_session_service.py`, `schemas/runtime_session.py` and their routers in `main.py`.
     - Delete `tests/runtime/` and the matching `tests/api/` files: runtime ws, runtime sessions, runtime owner, robot control errors and request ack.
     - Delete the stale `tests/runtimev2/` and the leftover `src/runtimev2/__pycache__`.
   - **Moved, not deleted:**
     - `runtime/features.py` (`observation_to_dict`, `sanitize_camera_name`, …) moves to `robots/features.py`. It is used by `robots/physicalai_adapter.py`, `api/robot_observations.py` and the zero-calibration worker.
     - `ErrorEvent` and `ObservationEvent` move into a small `schemas/robot_stream.py` for `robot_observations`.
     - The runtime config export (`build_runtime_config`, `policy_source_fragment`, `runtime_export_readme`, `runtime_config_change_me`) moves to `services/runtime_export.py`. See the note below.
   - `api/robots.py`: drop the `runtime_session_holder` busy check for now. Registry claims replace it in Phase 5.
   - Untouched: `/api/cameras/ws` and `/robots/{id}/observations/ws` do not depend on the runtime and keep working until Phase 4 replaces them.
   - UI:
     - Delete `runtime-session-provider.tsx`, `use-joint-state.ts` and the runtime-sessions feature, and remove the footer badge.
     - Replace the environment form preview's robot cell with a "Live preview is being reworked" placeholder. Camera cells keep their standalone socket for now.
     - Record and inference pages: put a work-in-progress empty state behind a single `RUNTIME_WIP` flag; their buttons (record, run model, teleop) are disabled.
     - Regenerate the OpenAPI types.
   - The rest of the app has to keep working: robots, cameras, environments CRUD, datasets, training and export. `prek` and the backend and UI tests must pass.
   - **Done.** Deviations from the steps above:
     - `RuntimeSessionBusyError` (backend) and `isRuntimeSessionBusyError` (UI) were removed too; Phase 5 adds the claim errors back.
     - `RUNTIME_IDLE_TIMEOUT_S` was removed from settings.
     - `useSynchronizeModelJoints` moved out of `use-joint-state.ts` into `use-synchronize-model-joints.ts`.
     - The record and inference viewers (`recording-viewer.tsx`, `inference-viewer.tsx`) and `features/robots/robot-control/` were deleted rather than kept, because they only worked through the old provider. Phases 7 and 8 rebuild them on the new provider; the old layout is in git history.
     - `RUNTIME_WIP` lives in `components/runtime-wip/runtime-wip.tsx`, together with `RuntimeWipNotice`. It disables "Start recording", "Add Episode" and "Run model". The environment preview shows the robot model at rest with an info note.
1. **Worker foundations:** generic `ThreadWorker[C]` and `ProcessWorker[C]` in `src/workers/base.py`, replacing the old `BaseThreadWorker`/`BaseProcessWorker`, plus `Rate` in `src/workers/rate.py` replacing `run_at_frequency`. Covers readiness and failure propagation, stop, join, then kill. The existing workers (training, dataset import, training dispatcher, camera) and the `Scheduler` moved onto it. Tests use fake workers that crash during setup, crash mid-run, and hang.
2. **Store:** `FeatureSpec`, `Sample`, lock-free scalar and image buffers (values first, timestamp last), create and attach, compatibility checks with detailed mismatches, and the stale-block sweep. Includes a cross-process read/write test.
3. **Event bus:** a typed zenoh wrapper with pub/sub and an RPC with ack and timeout. Tests run against a local-only zenoh config.
   - Done: `runtime_session/bus.py` over a per-session Unix socket. Tests cover pub/sub with a union, request/reply with handler errors and timeouts, a `ProcessWorker` on the bus, and isolation between two sessions. Nothing uses it yet; workers adopt it in Phases 4 and 7.
4. **Robot and camera workers, plus the environment:** spec from the environment, ported from `runtimev2/environment.py`, and observe-only mode for `RobotWorker`. Move `/api/cameras/ws` and `/robots/{id}/observations/ws` onto the new workers (UI levels 1 and 2). The frontend gets the dumb `CameraView` and `RobotJointsView` plus the standalone hooks. Tested with a fake robot driver, then once on a real SO-101 through `scratch.py`.
   - Done (backend only):
     - `workers/robot_worker.py`: `RobotRecipe` (driver config dict, `<joint>.pos` names, device ids) and `RobotWorker`, a `ProcessWorker` that seeds `action` on connect, writes `state`, and sends `action` with the goal_time rule. `observe_only` never sends.
     - `claim_robot(recipe)` holds physicalai's flock device lock in the parent, so a robot held by another socket, worker or `SharedRobot` owner is refused with `robot_device_already_owned`, and a killed child cannot leave it locked. Shortcut: it imports physicalai's private `_lock` module.
     - `CameraWorker` writes into a store slot (`camera_image_spec(config)`) instead of its own buffer.
     - `runtime_session/environment.py`: `describe_environment(env, factory)` returns the spec, the follower recipe and the cameras by dataset key. It refuses anything but one robot and cameras whose keys collide. The leader is left to Phases 5 and 6.
     - `/robots/{id}/observations/ws` runs an observe-only `RobotWorker` with a one-robot store. `/api/cameras/ws` reads `SharedCamera` directly with no worker or store, because `SharedCamera` already captures in its own publisher process, so a preview only needs `read_latest` and a JPEG encode. It sends each frame once, by `frame.sequence`. Both wire protocols are unchanged, so the UI needed no change.
     - The worker base shields stop-and-join from cancellation, so a closed websocket still waits for its device to be released.
   - Moved to Phase 5: the dumb `CameraView` / `RobotJointsView` and the standalone hooks. They only pay off once a provider exists to feed them.
   - Not done: the real SO-101 check. The robot detail page and robot form exercise the observe-only path on hardware; the driving path has no UI until Phase 6.
5. **RuntimeSession, registry and session websocket:** load and unload the environment, store lifetime, claims (`device_in_use` for levels 1 and 2, the busy check in `api/robots.py`). Includes the new ws protocol at `/runtime/ws` (commands with acks, `state`, and a 20 Hz binary `observation` carrying joints and frames), the new `RuntimeSessionProvider`, and the runtime-sessions footer and list on the registry. The **environment form preview** comes back: its placeholder is removed and its cells read from the provider.
   - Done:
     - `runtime_session/session.py`: `RuntimeSession` with `load_environment` / `unload_environment` / `close`.
       - Loading claims the robots in the registry, pins the cameras in `CameraClaimRegistry` and takes the device locks. It then creates the store and connects the follower `RobotWorker`, the `CameraWorker`s and the leader driver in parallel.
       - If any step fails, everything already acquired is released. A worker that dies sends `worker_failed` and unloads the environment; the session survives.
     - `runtime_session/registry.py`: live sessions and their robots. Forms get `device_in_use` naming the session: the robot observations socket, the camera preview socket for a pinned camera, and `DELETE /robots/{id}` (423).
     - `/api/projects/{id}/runtime/ws?hz=20` implements the protocol above. `load_environment` takes an `EnvironmentDraft` (robots, teleoperator, cameras by id) rather than an `environment_id`, because the form preview loads unsaved environments. The message models are in the OpenAPI schema through the GET stub.
     - `/api/runtime/sessions/ws` pushes the session list (`RuntimeSessionsMessage`) on connect and whenever a session opens, closes, loads or unloads. There is no polling. `POST /api/runtime/sessions/{id}/stop` closes a session, and its socket gets `session_closed` and closes.
     - The setup and zero-calibration wizards take the same device lock (`claim_devices`) while they hold the port, so a session, a form or another wizard on that device is refused with `robot_device_already_owned`. They work on unsaved robots, so the registry cannot name the holder.
     - UI:
       - `features/runtime-session/runtime-session-provider.tsx`: the socket, state, acked commands, and observation and frame channels. `useSessionJoints` and `useSessionCamera` read from those channels, not React state.
       - The dumb `CameraView` (`features/cameras/camera-view.tsx`) takes a `FrameSource`; `useCameraSocket` feeds it on the camera form.
       - The environment preview runs on the provider again.
       - The footer shows the runtime sessions with Stop, and the robot list marks robots in use.
   - Not done:
     - `RobotJointsView` was not extracted: the robot cell feeds `RobotViewer` through `useSessionJoints`, and the robot form keeps `useRobotObservations`.
   - Known limitation: the form preview reloads the whole environment when its robots or cameras change, so adding a camera reconnects the robot.
6. **Controls:** base class, `MoveControl`, `TeleopControl`, and the switching protocol. Teleop buttons are re-enabled.
   - Done:
     - `runtime_session/controls.py`:
       - `Control` is a `ThreadWorker` whose `tick()` runs at its own `hz`.
       - `Control` calls `on_start()` once on its own thread inside `lifecycle` (so a failure there fails the start), then `on_tick()` at `hz`. Each control is its own module in `runtime_session/controls/`.
       - `TeleopControl` borrows the leader and writes its measured position with the leader's capture time. It refuses a leader whose joints differ from the follower's.
       - Every time teleop starts there is a grace period (`grace_s`, default 3 s; `0` turns it off). The command blends from where the follower was to the leader's live position, so the follower closes the gap instead of jumping.
       - `MoveControl` interpolates from the measured state to a target by joint name over `duration_s`, then holds. Joints left out stay put.
     - Configs are pydantic (`TeleopControlConfig`, `MoveControlConfig`, the `ControlConfig` union) and show up in `state.control`.
     - `set_control {control: ... | null}`: the session stops and joins the old control, re-seeds `action` from the measured state (hold), then starts the new one. `null` is hold. Unloading the environment stops the control first.
     - A control that dies (an unplugged leader, say) sends `control_failed`, and the session falls back to hold. The environment stays loaded.
     - UI: a Teleoperate switch in the environment preview's robot cell, shown when the environment has a leader.
     - The session and sessions websockets run their tasks in an anyio task group. With `asyncio.TaskGroup`, Starlette's anyio cancellation on close sometimes escaped the handler.
     - Session lifecycle:
       - `RuntimeSession` is an async context manager; the websocket holds it with `async with`. Leaving the block unloads everything in reverse and unregisters it.
       - `stop()` (the stop endpoint, backend shutdown) runs the same idempotent teardown, then sends `session_closed`.
     - Optional parts:
       - Each optional part lives in a `Slot[T]` (`runtime_session/slot.py`): empty, or holding what its async context manager yielded. `load` / `unload` / `require(message)` replace the hand-kept stack, value and state fields.
       - The environment slot holds the claims, the connection and the worker watcher. The control slot holds the running `Control`, which carries its own config.
       - Dependencies stay explicit: unloading the environment unloads the control first. Phase 7 adds dataset and model slots the same way.
   - Not done: robot-specific teleop subclasses (driving the leader, torque changes), and a UI for `MoveControl`. Phase 7 uses `MoveControl` for replay and resets.
7. **Dataset worker:** spec from the dataset (it can be the first load), recording commands, acked save and discard, the finalize path. The **record page** comes back on the new provider (lifting `RUNTIME_WIP` for it). `ReplayControl` follows in the same phase.
   - Split into three PRs: 7a backend, 7b the recording viewer UI, 7c `ReplayControl` (backend only; its UI is decided later).
   - Decided: the environment loads first and owns the store; dataset-first loading is deferred. `load_dataset` checks the dataset's joints and cameras against the environment with `require_compatible`. A dataset with nothing recorded yet takes the environment's shape and records at 30 fps; an existing one keeps its own fps.
   - 7a done:
     - `runtime_session/dataset.py` reads a dataset's spec and fps from `meta/info.json` and builds the lerobot features for a new one.
     - `workers/dataset_worker.py`: `DatasetWorker` is a `ProcessWorker`, so lerobot's writes and encoding stay off the API process's GIL.
       - It attaches to the store and opens the recording mutation, which copies the dataset into the cache.
       - It owns its commands and its state. It serves `start_episode`, `save_episode` and `discard_episode` on the bus (the bus's first real use), replies with its new status, and publishes every change on `evt/dataset`.
       - While an episode is open it samples the store at the dataset fps. A tick with any feature missing is skipped, never padded.
       - On exit it discards an open episode, then copies the cache back. `ready_timeout_s` and `stop_timeout_s` are 300 s, because killing it mid-copy loses the dataset.
     - The session is itself a part on the bus, following the same lifecycle pattern as a worker.
       - `async with session:` hosts the bus and serves the session's own commands: `load_environment`, `unload_environment`, `set_control`, `load_dataset {dataset_id}` and `unload_dataset`. It publishes its state and errors on `evt/session`.
       - Session commands map their failures to error codes themselves (`_command_error`), and the code travels in the bus's error reply.
       - The session subscribes to `evt/dataset` before starting the worker and folds what it publishes into `state.dataset` (`{dataset_id, name, fps, is_recording, episodes_recorded}`), so the worker is the single source of truth.
       - Unloading the environment unloads the control, then the dataset, then the environment. A dead dataset worker sends `worker_failed` and unloads the dataset; what it saved but had not copied back is lost.
     - Commands are routed by name: a client command `foo` is a request on `rpc/foo` (`command_key`), served by whichever part handles it. The socket (`api/runtime_ws.py`) only forwards.
       - It reads each command's `command` and `request_id`, puts the raw JSON on the bus, and acks with the reply. It forwards `evt/session`, and samples the store for observations.
       - It knows nothing about what any command does; the handler validates the payload against the command models.
       - A command nothing serves right now (unknown, or for a part that is not loaded) fails at once with `command_unavailable` (the bus's `NoResponderError`), rather than waiting out a timeout.
     - Database lookups for commands (a draft's rows, a dataset in the project) live in `runtime_session/resolve.py` and are handed to the session as `Resolvers`.
     - `Declaration.close()` takes its zenoh entity out before undeclaring it, so a declaration closed by its owner and by the bus at once is undeclared exactly once.
     - Ack error codes include `feature_mismatch`, `no_episode` / `episode_open` / `task_required` from the worker, `command_unavailable`, `invalid_command` and `command_timeout`.
   - 7b done (the recording viewer UI):
     - `RuntimeSessionProvider` takes an optional `dataset_id` and loads it once the environment is loaded; reloading the environment reloads both. It sends the recording commands, and `environmentDraft()` turns a saved environment into the draft a session loads.
     - `SessionViewer` (`features/runtime-session/session-viewer.tsx`) replaces the separate record and inference viewers.
       - It docks the session's follower and cameras with dockview, as `main` does, under a toolbar the page passes in.
       - Until the environment and every part the page names (a dataset, later a model) is loaded, it shows what it is waiting for, the session's error, and Cancel.
       - The record page passes a `RecordingToolbar` (task box, start / accept / discard with the hotkeys, the recording timer); Phase 8 passes the inference toolbar.
       - The session cells (`RobotCell` with its Teleoperate switch, `CameraCell`) moved to `features/runtime-session/cells/`, shared with the environment preview.
       - `state.environment.cameras` carries each camera's `name` for the panel titles.
     - The record page's session content sits inside its own `<Suspense>`, inside the provider. Otherwise a cell suspending while it loads took the page-level boundary, unmounted the provider, and ended the session; the page then reopened a session that found the robot still held by the one tearing down.
     - Teleop starts once the dataset is loaded when the environment has a leader, as before. `TELEOP_CONTROL` is the one place the UI's teleop rate and grace period live.
     - Leaving the page finishes the recording first: the back link sends `unload_dataset`, which copies the episodes into the dataset, waits for the ack, then refreshes the episodes and navigates. The episode total adds the session's `episodes_recorded` to what the dataset holds, because saved episodes reach the dataset only on that copy.
     - `RUNTIME_WIP` is off "Start recording" and "Add Episode"; it stays on the inference page until Phase 8.
   - Known gap: closing the tab or navigating away another way still saves the recording (the session's teardown copies it back), but the dataset page can then load before the copy finishes and briefly show the old episode count.
8. **ModelControl:** port the policy execution off `physicalai.runtime` (sync, async with the time-based chunk queue, smoothing). Introduce `ModelControlConfig` and switch `services/runtime_export.py` over to it, along with the export round-trip test. The **inference page** comes back on the new provider, and the `RUNTIME_WIP` flag is deleted. Measure with the oracle model: `actions[i] = f(t0 + i)`.
9. **Finish:**
   - Check that no `physicalai.runtime` import is left outside `services/runtime_export.py`.
   - Rewrite `docs/explanation/runtime-session-architecture.md`.
   - Do a full manual pass on real hardware: preview, teleop, record + save/discard, inference, replay.
10. **Merge** the feature branch into `main`.

## Decisions

1. **RuntimeSession lives in the FastAPI process** for now and belongs to its websocket. Closing the socket or restarting the backend ends the session, and there is no reattaching.
2. **One active environment per session, with one robot per environment** until the environment refactor. The robot may have one leader.
3. **The action timestamp is the time the action should have happened** (usually now). The robot worker sets `goal_time = max(action.timestamp - previous_tick_timestamp, 1/hz)`.
4. **A loaded dataset decides the store shape**, just as an environment or model would. Replaying episodes is a Control (`ReplayControl`).
5. **Single robot until the environment refactor.** The store holds one `state` and one `action`; the leader is a driver owned by the environment (decision 9).
6. **UI levels:** a camera form opens a `CameraWorker`, a robot form opens an observe-only `RobotWorker`, and the environment, record and inference pages open a `RuntimeSession`, loading environment only, + dataset, or + model respectively. All live updates come from the session store at ~20 Hz. Pages with a session read joints and frames only from the provider, never from per-device sockets.
7. **Keep the runtime export.** Studio exports a config that runs on `physicalai.runtime` outside Studio, built from the same `ModelControlConfig` the session runs. Settings without a runtime equivalent are flagged as not exported.
8. **Remove first, rebuild after.** The old runtime is deleted in Phase 0; unfinished features show as work in progress on the feature branch until their phase lands.
9. **Controls borrow; the session owns.** The environment connects the leader as a plain driver (no worker) and keeps it connected across control switches; `TeleopControl` and HIL borrow it, `ModelControl` borrows the loaded policy. Borrowing controls run as threads in the session process. The store has no leader slot.
