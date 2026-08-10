# Phase 0 — Upstream physicalai changes

Four changes in [openvinotoolkit/physicalai](https://github.com/openvinotoolkit/physicalai) that Studio's phase B depends on. Two are additions to `PolicySource`, one is a bug in the runtime's per-run shutdown path, and one exposes a transport helper Studio would otherwise have to duplicate.

**Required before phase B3**, not before phase A. Phase A needs only `RobotRuntime.stop()`, released in [#226](https://github.com/openvinotoolkit/physicalai/pull/226).

Sections 1, 2 and 4 are written to paste into an issue or pull request description. Section 3 is a bug report.

> Delete this document once the changes are merged.

## Why Studio needs these

Studio runs one long `RobotRuntime.run()` per session and switches between hold, teleoperation and policy modes inside its own `ActionSource`. The policy delegate is a real `PolicySource`, so the inner pipeline matches what `physicalai run` executes from an exported config — that shared path is what makes Studio's config export trustworthy.

Switching _away_ from policy and back is where the current API runs out. See [design.md](design.md#core-decisions) for the session model.

---

## 1. `PolicySource.reset()`

### Problem

Consider one Studio inference session:

```text
t=0    websocket opens, runtime.connect(), run() starts
t=2    user loads a model, PolicySource built and connected
t=3    "start task"  → mode = policy, the arm works
t=20   "stop task"   → mode = hold
t=21   user switches to teleoperation and drives the arm across the table
t=45   "start task"  → mode = policy
```

At t=45 the action queue may still hold the chunk computed at t=20, from an observation where the arm was somewhere else, and `_last` holds the action sent at t=20. Resuming without discarding them commands a 25-second-old target — a large, unexpected motion under torque.

There is no public way to discard that state.

### Why the existing operations do not fit

| Operation                                    | Effect                                                                                                                                                                                                                                                            |
| -------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `RobotRuntime.stop()` then `run()`           | Ends the session. `_shutdown()` calls `_bus.close()`, which closes every callback — including Studio's dataset writer, mid-recording. New `session_id`. Also ends teleoperation, which the same runtime is driving. Tears down and restarts the inference worker. |
| `PolicySource.disconnect()` then `connect()` | Leaves the session intact, but stops and restarts the inference worker for no reason. `stop()` joins for up to `_JOIN_TIMEOUT_S` (10 s), then `start()` waits `_STRAGGLER_GRACE_S` (2 s) and can raise `RuntimeError`. All to discard a few arrays.               |

`stop()` answers _should the loop keep running?_ The need here is _are the queued actions still valid?_ The first cannot express the second, because it ends the loop.

### The current public API makes things worse

`PolicySource.action_queue` is public, so a caller can reset the queue. `_last` and `_warmed_up` are private, so it cannot clear those. The result of a partial reset:

1. queue empty, `_warmed_up` still `True`
2. next `update()` skips warmup and submits asynchronously
3. `pop()` returns `None`
4. falls back to `_last` — the stale t=20 action

The arm jumps anyway. A partial reset is a footgun, which is the strongest argument for making the whole operation available.

### Proposal

```python
def reset(self, *, reset_model: bool = True) -> None:
    """Discard queued actions and per-episode state; the next update() re-seeds.

    Leaves the execution worker running and the source connected. Use when the
    world has moved on since the queued actions were computed — a new episode,
    or re-arming after an operator drove the robot elsewhere.

    Args:
        reset_model: Also reset the underlying model, clearing runner state for
            stateful policies. Pass False to keep model state across the reset.
    """
    self._action_queue.reset()
    self._last = None
    self._warmed_up = False
    if reset_model:
        self._model.reset()
```

Post-conditions:

- the runtime is untouched and still looping
- the execution worker is untouched and still alive
- no lifecycle events, no `session_id` change, no callbacks closed
- the next `update()` runs one synchronous warmup against the **current** observation, seeds the queue, and returns its first action

Queue invalidation itself is cheap. A concurrency-safe implementation may wait
for inference already inside the model before resetting model state, and model
or execution reset errors may propagate. It still leaves the worker running and
avoids the stop/join/start cycle.

### Scope of the ask

To be precise about what is new: `connect()` **already does most of this**, unconditionally, inside its `if not self._connected` guard — it resets the queue, clears `_last`, and sets `_warmed_up = False`. That behaviour is correct and already tested.

So the ask is mostly "expose what `connect()` already does as a method a caller can invoke on its own", not "add new reset semantics". The one genuinely new piece is `model.reset()`, which `connect()` does not currently call.

Refactoring `connect()` to delegate is therefore available:

```python
def connect(self, *, bus, session_id):
    if not self._connected:
        self._connected = True
        self._bus = bus
        self._session_id = session_id
        self._execution.set_bus(bus, session_id)
        self._execution.start(self._model, self._action_queue)
        self.reset()
```

**This is a behaviour change, not a pure simplification.** Delegating adds a `model.reset()` call to the connect path that is not there today. That looks desirable — a fresh connection should not inherit runner state from a previous one — but it is a change, and an upstream reviewer should be told rather than left to find it. Keep the `connect()` test #226 added, which guards the queue/`_last`/`_warmed_up` behaviour.

If that change is unwanted, leave `connect()` alone and let `reset()` stand on its own. The duplication is three lines.

### Tests

- after `reset()`, `action_queue.remaining == 0`
- after `reset()`, a subsequent `update()` on an empty queue calls `execution.warmup()` exactly once and returns an action derived from the observation passed to that call, not from any pre-reset chunk
- `reset()` does not call `execution.stop()` or `execution.start()`, and the worker thread identity is unchanged
- `reset(reset_model=False)` does not call `model.reset()`
- `reset()` emits no lifecycle events and does not close callbacks
- `connect()` after the refactor still resets queue, `_last` and `_warmed_up` — guard against the refactor dropping a behaviour #226 added

---

## 2. Public `PolicySource.warmup(observation)`

### Problem

The synchronous warmup is load-bearing, not incidental. After `connect()` or `reset()`, `_warmed_up` is `False` and the queue is empty. The next `update()` calls `execution.warmup()`, which for `AsyncExecution` explicitly "runs one inference in the main thread". That is what guarantees `pop()` has something to return.

Skip it — set `_warmed_up = True` with an empty queue and no `_last` — and `update()` raises `RuntimeError("No action available and none produced yet")`.

So the guarantee is sound, but the cost lands on the control thread: every arm and every episode reset pays one blocking inference. At 30 fps with a 200 ms model that is a visible tick overrun.

The caller cannot avoid it. Calling `execution.warmup()` directly leaves `PolicySource._warmed_up` at `False`, so `update()` warms up a second time.

### Proposal

```python
def warmup(self, observation: dict[str, np.ndarray]) -> None:
    """Seed the queue now, so the first update() does not block.

    Optional. update() still warms up lazily when a caller has not. Callers
    that can supply a recent observation off the control thread — for example
    while a mode switch is still pending — use this to avoid a stalled tick.
    """
```

It must set the same flag `update()` checks, so the lazy path does not repeat the work.

Studio's use: build the `PolicySource` on a loader thread once the model has loaded, call `warmup()` with the most recent tick's observation, and only then let the mode switch to `policy`. The arm becomes stall-free.

### Alternative, if the above is unwanted

Give the empty-queue failure its own exception type instead of bare `RuntimeError`:

```python
class ActionNotAvailableError(RuntimeError):
    """The queue is empty and no action has been produced yet."""
```

A caller can then catch precisely and hold position, rather than catching `RuntimeError` on a 30 Hz path and inferring what it meant. This does not remove the stall, so `warmup()` is preferred.

`is_ready` alone does not help: if the caller skips `update()` while not ready, nothing ever warms the source up.

### Tests

- `warmup(obs)` populates the queue and the following `update()` does not call `execution.warmup()` again
- `update()` without a prior `warmup()` still warms up lazily — the existing behaviour holds
- `reset()` re-arms warmup, so `warmup()` or the lazy path runs again on the next cycle

---

## 3. Bug: per-run shutdown closes callbacks, so runtimes are not reusable

### Report

#226 made `RobotRuntime` reusable across `run()` calls: `_stop` is cleared in `_shutdown()`, devices stay connected, and the documentation states that "one runtime can stop and run again".

But `_shutdown()` ends with `_emit_shutdown()`, whose `finally` calls `self._bus.close()`, and `_CallbackBus.close()` calls `close()` on every callback. Callbacks holding resources do not survive into a second run.

### Reproduction

```python
runtime = RobotRuntime(robot=robot, action_source=src, fps=30,
                       callbacks=[JsonlCallback("/tmp/ticks.jsonl")])
with runtime:
    runtime.run(duration_s=1)   # fine
    runtime.run(duration_s=1)   # every tick logs ValueError: I/O operation on closed file
```

`with runtime:` is the shortest reproduction and matches upstream's own examples. Studio itself never uses it — see [design.md](design.md#the-session-owns-devices-the-runtime-is-a-view), where `__exit__` disconnecting devices would break rig-change rebuilds — so the two documents differ here deliberately.

`JsonlCallback` opens its file in `__init__` and closes it in `close()`. There is no reopen path. Because `emit_tick` isolates callback exceptions, the second run logs a traceback per tick rather than failing, so the data loss is quiet.

`AsyncCallback` is worse. `close()` sets its stop event and joins its worker thread, with no restart. A second run enqueues into a deque nobody drains: **silent, total telemetry loss** for everything wrapped in it.

### Severity

Any stateful callback plus the reuse #226 newly permits. It will not appear in a single-run test, which is how the existing suite exercises callbacks.

### Options

1. **Move `close()` out of the per-run path.** Flush at shutdown, close when the runtime is disposed — `disconnect()` or `__exit__`. Matches the distinction #226 drew between stopping a run and shutting down a runtime.
2. **Give callbacks a reopen contract.** An `on_session_start` hook, or make `close()` idempotent and reversible. More surface area, and every third-party callback has to implement it.
3. **Document that reuse requires stateless callbacks.** Cheapest, but it makes the reuse feature much less useful, since telemetry is exactly what you want across runs.

Option 1 looks right: the bus is runtime-scoped, not run-scoped.

### Tests

- two consecutive `run()` calls with a `JsonlCallback` produce tick records from both runs in the file
- two consecutive `run()` calls with an `AsyncCallback` deliver events from both runs to the inner callback
- `disconnect()` or `__exit__` still closes callbacks exactly once

### Studio's workaround meanwhile

Do not treat `stop()` then `run()` as an episode boundary — which is independently the right design, and is what `reset()` in section 1 is for. Studio only rebuilds a runtime when the rig changes, and it constructs fresh callbacks when it does. See [design.md](design.md#the-session-owns-devices-the-runtime-is-a-view).

---

## 4. Make the Zenoh session helper public

### Problem

`open_session` lives in `physicalai.robot.transport._session` — a private module. It is the only correct way to open a Zenoh session for this transport, because it applies a security posture that is easy to get wrong and that its own docstring calls out:

> Secure by default: unless the caller explicitly opts into `allow_remote`, multicast/gossip scouting is disabled and the owner's listen endpoint is bound to loopback only, so the transport is unreachable off-host — not merely undiscoverable.

Studio's phase B needs a session for its own control plane, and has two problems reaching it:

1. **It is private.** Upstream states the convention against private cross-package imports — `config/__init__.py`: "Transport and other callers import public names from here (no private `physicalai.config._*` imports)." The same reasoning applies here.
2. **The port derivation is robot-specific.** `derive_endpoint_port` hashes `physicalai/robot/{name}`. Studio's sessions are not robots and must derive their rendezvous port from their own prefix, or a session and a robot owner sharing a name segment bind the same port.

So Studio has to duplicate the configuration. That is how security posture drifts. Upstream's own `TelemetryEmitter` is the existing example: it opens a bare `zenoh.Config()`, so scouting is enabled and nothing is loopback-bound, unlike every other session in the package.

### Proposal

Export a helper that takes the key prefix from the caller:

```python
def open_session(
    name: str | None = None,
    *,
    key_prefix: str = "physicalai/robot",
    listen: bool = False,
    allow_remote: bool = False,
) -> Session:
    """Open a Zenoh session pinned to peer mode, loopback-bound by default."""
```

The default keeps every current caller unchanged. Studio would pass `key_prefix="studio/rt"` and get the same security posture without copying it.

Alternatively, expose the security configuration alone — `zenoh_config(*, allow_remote=False)` returning a `Config` — and let callers handle their own endpoints. Less convenient, but it still puts the part that matters in one place.

### Why it is worth doing

Any downstream that builds on this transport hits the same wall, and the failure mode of getting it wrong is a robot control plane reachable from the network with no authentication. That belongs in the library, not in each consumer.

### Tests

- the default prefix keeps existing derived ports unchanged
- a custom prefix derives a different port for the same name
- `allow_remote=False` disables both scouting modes and binds loopback

---

## Not requested

For the record, so nobody builds them for Studio:

- **`RemoteExecution`.** The documentation lists it as planned. Studio puts the whole session — loop, execution, inference, dataset writing — in one process, so inference is already isolated from the API process. Studio does not need it.
- **A configurable `goal_time` horizon.** `_GOAL_TIME_TICKS = 3` is hardcoded. Studio uses two ticks for teleoperation and one for inference today, per robot type, and all the multipliers are currently 1.0. SO101 ignores `goal_time` entirely; only WidowXAI honours it. Deferred until hardware testing shows it matters.
- **An explicit "no-op action" contract.** `ActionSource.update()` must return an action every tick. Studio solves its hold mode with a latched target. Revisit only if a second consumer needs it.

## Minor observation

`TeleopSource._leader_owned` is never cleared in `disconnect()`. Harmless in the common case, but across the reuse #226 enables: run one connects the leader and sets the flag, shutdown disconnects it, and if anything else connects the leader before run two, `connect()` sees it already connected and skips — yet the stale flag means run two's `disconnect()` tears down a leader it did not own.

Studio writes its own teleoperation action source, so this does not block anything.
