# Plan: Migrate TeleoperateWorker to a SharedRobot-compatible thread worker

## Root cause

`robot_websocket` (`application/backend/src/api/robot_control.py`) builds a `PhysicalAIRobotAdapter` through `RobotClientFactory.build()`, connects it, and passes the connected adapter to `TeleoperateWorker` (`application/backend/src/workers/teleoperate_worker.py`). `TeleoperateWorker` is a `BaseProcessWorker`, and the backend explicitly configures multiprocessing to use `spawn`. Starting the worker therefore pickles its constructor state and fails on the live Rust/pyo3 `zenoh.Session` stored by `SharedRobot` with `cannot pickle 'builtins.Session'`.

`RobotControlWorker` (`application/backend/src/workers/robot_control_worker.py`) already uses the appropriate ownership model. It is a `BaseThreadWorker`, receives an unbuilt `RobotClientFactory`, and builds and connects `SharedRobot`-backed clients from its worker thread. The library still provides hardware isolation through its internal `RobotOwner` subprocess, so an additional application process is unnecessary.

Confirmed behavior:

- `SharedRobot.connect()` opens the Zenoh resources held by the client.
- `SharedRobot.disconnect()` tears those resources down.
- `PhysicalAIRobotAdapter.features()` reads `SharedRobot.joint_names`, which requires the robot to be connected.
- `BaseThreadWorker.run()` invokes `setup()` synchronously and then awaits `run_loop()` on the thread's event loop. Async factory calls therefore cannot live in a `BaseThreadWorker.setup()` override.
- `RobotClientFactory.build()` may be called from the worker thread's event loop. Its `RobotConnectionManager` is shared mutable state, however, so this plan does not claim general thread safety or add concurrent discovery support.

## Decisions

- Convert `TeleoperateWorker` from `BaseProcessWorker` to `BaseThreadWorker`.
- Pass the `RobotClientFactory` and robot schema objects into the worker; build and connect clients in the worker thread.
- Publish readiness only after features, initial state, and initial actions are all available.
- Preserve setup exceptions for the websocket handler and make every readiness wait failure-aware.
- Fix connect-before-features ordering in `EnvironmentIntegration.setup()` and make its setup transactional.
- Update focused worker, environment, and websocket tests.
- Do not change unrelated process workers or `SharedRobot` composition.

## Phase 1: Rewrite TeleoperateWorker

File: `application/backend/src/workers/teleoperate_worker.py`

1. Change the base class to `BaseThreadWorker` and import it from `.base`.
2. Remove `ctypes` and `multiprocessing as mp`; add `threading`.
3. Keep `multiprocessing.synchronize.Event` only as the type of the scheduler-owned stop event, matching `RobotControlWorker`.
4. Import `RobotClientFactory` and the `Robot` schema.
5. Change the constructor to:

   ```python
   def __init__(
       self,
       robot_client_factory: RobotClientFactory,
       follower: Robot,
       leader: Robot | None,
       frequency: float,
       stop_event: EventClass,
   ) -> None:
   ```

6. Store the factory and schema objects as `self.robot_client_factory`, `self._follower_robot`, and `self._leader_robot`.
7. Initialize state shared with the API thread:

   ```python
   self.loaded_event = threading.Event()
   self._action_lock = threading.Lock()
   self._state_lock = threading.Lock()
   self._action_read_state = ActionReadState.NONE
   self._output_state: list[float] = []
   self.follower: RobotClient | None = None
   self.leader: RobotClient | None = None
   self.features: list[str] = []
   self.setup_error: Exception | None = None
   ```

8. Call `super().__init__(stop_event=stop_event)`.
9. Protect the observation snapshot and action-read state with thread locks. Return a copy from the observation getter and use slice assignment in its setter.
10. Make `set_action_read_state()` validate and normalize input with `ActionReadState(value)` before storing it under `_action_lock`. This preserves the constrained state domain after removing `mp.Value`.
11. Remove the async `setup()` override. Put async build and synchronous connection work at the beginning of `run_loop()`.
12. Structure `run_loop()` with an outer `try/finally` and a setup-only nested `try/except`:

```python
async def run_loop(self) -> None:
    try:
        try:
            self.follower = await self.robot_client_factory.build(self._follower_robot)
            if self._leader_robot is not None:
                self.leader = await self.robot_client_factory.build(self._leader_robot)

            if self.leader is not None:
                self.leader.connect()
            self.follower.connect()

            self.features = self.follower.features()
            state = self.follower.read_state()["state"]
            aligned_state = self._align_feature_values(state)
            self._set_state(aligned_state)
            self.loaded_event.set()
        except Exception as exc:
            self.setup_error = exc
            logger.exception("Failed to set up teleoperation worker")
            return

        # Existing frequency-controlled teleoperation loop.
    finally:
        logger.info("Teleoperating stopped, disconnecting robots.")
        if self.leader is not None:
            self.leader.disconnect()
        if self.follower is not None:
            self.follower.disconnect()
```

    Initializing the state buffer before `loaded_event` is essential. Otherwise the websocket can enumerate `features` against an empty state list and terminate its outgoing task with `IndexError`.

13. Keep the existing main loop behavior after setup:
    - Read and publish follower state.
    - In `TELEOPERATION`, align leader values to follower features and send them to the follower.
14. Remove the no-op `teardown()` override.
15. Remove or replace `wait_for_loading_to_complete()`. It must not retain an indefinite `Event.wait()` that cannot report setup failure. Prefer one failure-aware readiness method shared by the API and tests:

```python
async def wait_until_loaded(self, poll_interval: float = 0.05) -> None:
    while not self.loaded_event.is_set():
        if not self.is_alive():
            if self.setup_error is not None:
                raise self.setup_error
            raise RuntimeError("Teleoperation worker stopped before loading")
        await asyncio.sleep(poll_interval)
```

16. Update the class docstring and example to describe the thread-shared observation snapshot, the factory/schema constructor, and `stop_event`. Remove process and shared-memory terminology.

## Phase 2: Update robot_websocket

File: `application/backend/src/api/robot_control.py`

1. Keep resolving IDs and loading follower and optional leader schema objects through `robot_service` in the request event loop.
2. Remove direct client construction, connection, and feature reads from the websocket handler.
3. Construct and start the worker:

   ```python
   worker = TeleoperateWorker(
       robot_client_factory=robot_client_factory,
       follower=follower,
       leader=leader,
       frequency=fps,
       stop_event=scheduler.mp_stop_event,
   )
   worker.start()
   ```

4. Await the worker's failure-aware readiness method instead of waiting directly on `loaded_event`.
5. After readiness succeeds, copy `features = worker.features` and start the existing incoming and outgoing tasks.
6. Preserve the existing outer exception handler so factory, calibration, discovery, and connection errors result in logging and WebSocket close code 1011.
7. Keep `handle_incoming`, `handle_outgoing`, and `_build_robot_control_state` unchanged except for any type adjustment required by normalized `ActionReadState` handling.
8. Keep `worker.stop()` in `finally`. Do not add client cleanup to the request handler; the worker owns all clients it builds.

## Phase 3: Fix EnvironmentIntegration ordering and rollback

File: `application/backend/src/control/environment_integration.py`

1. Build the follower and optional leader as today.
2. Connect both clients before calling `self.follower.features()`.
3. Keep camera construction and the warmup sleep after robot connection and action-key resolution.
4. Make setup transactional. If robot connection, feature discovery, camera connection, or warmup fails, tear down every resource acquired so far and re-raise the original exception:

   ```python
   async def setup(self) -> None:
       try:
           # Build robots.
           # Connect follower and optional leader.
           # Resolve action_keys.
           # Connect cameras and warm them up.
       except Exception:
           await self.teardown()
           raise
   ```

5. Ensure `teardown()` remains safe for partially initialized resources and repeated calls. Clear or otherwise avoid re-disconnecting successfully released references if the underlying clients are not idempotent.
6. Remove the old connect calls from the bottom of `setup()`.

This rollback requirement prevents the ordering fix from introducing a robot leak when camera setup fails. It also closes the existing leak where an earlier camera remains connected if a later camera fails.

## Phase 4: Update focused tests

### TeleoperateWorker tests

File: `application/backend/tests/workers/test_teleoperate_worker.py`

Rewrite process-era fixtures and tests so they no longer pass built clients or call async `setup()` directly.

Cover these invariants:

- The constructor accepts a factory and robot schemas.
- The follower and optional leader are built in the worker execution path.
- Each client is connected before its connection-dependent methods are used.
- `features()` is read only after follower connection.
- Initial follower state populates both state and action buffers before `loaded_event` is set.
- State and action getters return locked copies.
- Action source values are normalized to `ActionReadState` and invalid values fail.
- Leader-less, leader teleoperation, missing-feature fallback, and external-action behavior remain unchanged.
- Successful and partial setup paths disconnect every client that was built.
- A factory, connection, or initial-read failure is stored in `setup_error`, leaves `loaded_event` clear, and is raised by `wait_until_loaded()`.
- A worker that exits without either readiness or `setup_error` produces the fallback `RuntimeError`.

Use mocked hardware and run the actual thread only where thread lifecycle is the behavior under test. Keep deterministic direct coroutine tests for loop behavior where practical.

### EnvironmentIntegration tests

File: `application/backend/tests/control/test_environment_integration.py`

Add assertions that:

- Follower connection occurs before `features()`.
- Optional leader connection occurs during setup.
- A camera setup failure disconnects robots and any cameras already connected.
- Teardown is safe after partial setup and does not produce duplicate-release failures.

### Websocket tests

Add or extend an API test module for `robot_websocket`:

- Successful readiness uses `worker.features` and starts streaming.
- Setup failure raises the original stored exception and closes the socket with code 1011.
- A worker that dies without a stored setup exception closes with the fallback error instead of hanging.
- The handler always stops a worker it successfully started.

## Verification

1. Run focused tests from `application/backend`:

   ```bash
   uv run pytest tests/workers/test_teleoperate_worker.py tests/control/test_environment_integration.py
   ```

   Include the websocket test module once added.

2. Run static diagnostics on:

   - `application/backend/src/workers/teleoperate_worker.py`
   - `application/backend/src/api/robot_control.py`
   - `application/backend/src/control/environment_integration.py`
   - Updated test files.

3. Run backend hooks from the repository root:

   ```bash
   prek run --all-files application/backend/
   ```

4. Manual success path:

   - Start the backend.
   - Open a robot-control WebSocket session against a real or mocked SO101 or WidowXAI follower.
   - Confirm no Zenoh pickling traceback.
   - Confirm the first observation is complete and subsequent observations stream.
   - Repeat with a leader and follower pair and exercise teleoperation.

5. Manual failure path:

   - Use a robot with missing calibration or an unavailable port.
   - Confirm the original setup error is logged and the socket closes with code 1011.
   - Confirm readiness does not hang and partially created clients are disconnected.

6. Recording regression:
   - Run a recording session with a leader and follower.
   - Confirm `EnvironmentIntegration` connects before feature discovery and recording behavior remains unchanged.

## Scope boundaries

- Modify only:
  - `application/backend/src/workers/teleoperate_worker.py`
  - `application/backend/src/api/robot_control.py`
  - `application/backend/src/control/environment_integration.py`
  - Their focused tests.
- Do not modify `TrainingWorker` or `DatasetImportWorker`; their process isolation is intentional.
- Do not modify `CameraWorker` or `RobotControlWorker`; they already use `BaseThreadWorker`.
- Do not change bimanual `SharedRobot` composition in `RobotClientFactory`.
- Do not add locking to `RobotConnectionManager` in this change. Concurrent robot discovery is a separate design concern; document it if simultaneous setup becomes a supported workflow.
