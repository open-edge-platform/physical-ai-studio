# Policy Runtime Design Summary

We want a small runtime layer for running trained policies on robots.

Today, model inference, action chunking, queueing, async execution, remote serving, benchmarking, and product workflows are easy to mix together. The proposal separates those responsibilities.

## Proposed API

```python
from physicalai.inference import InferenceModel
from physicalai.runtime import PolicyRuntime, SyncExecution
from physicalai.robot.so101 import SO101

model = InferenceModel.load("./exports/act_policy")
robot = SO101(port="/dev/ttyACM0")

runtime = PolicyRuntime(
    robot=robot,
    model=model,
    execution=SyncExecution(mode="chunk"),
    fps=30,
)

runtime.run(duration_s=60)
```

CLI equivalent:

```bash
physicalai run --config so101_act.yaml --duration-s 60
```

## Responsibilities

| Component | Owns | Does not own |
|---|---|---|
| `InferenceModel` | loading, preprocessing, backend inference, `select_action()`, `predict_action_chunk()` | robot loop, FPS, callbacks, action dispatch |
| `Execution` | sync/thread/process/remote inference scheduling | action buffering, robot IO |
| `ActionQueue` | chunk storage, merge, smoothing, pop-one-action-per-tick | model execution, robot IO |
| `PolicyRuntime` | observation, timing, queue consumption, `robot.send_action()`, callbacks, shutdown | policy math |
| `Benchmark` / `LiberoBenchmark` | task-suite evaluation over gyms, episodes, success rate, reward, episode length, FPS, video/export | production robot-loop semantics |

## `select_action()` vs `predict_action_chunk()`

```python
action = model.select_action(obs)
```

`select_action()` returns one action. It is the simple direct-call API for scripts, tests, and existing users.

For chunk-producing policies, `select_action()` uses an internal `ActionChunkCursor`:

```python
if cursor.empty():
    cursor.push_chunk(model.predict_action_chunk(obs))

return cursor.pop()
```

`ActionChunkCursor` is a small internal helper. It tracks position inside one chunk. It is not the runtime `ActionQueue`.

```python
chunk = model.predict_action_chunk(obs)
actions = chunk["actions"]  # shape: (H, action_dim)
```

`predict_action_chunk()` returns a chunk-shaped `Mapping[str, Any]`. `PolicyRuntime` consumes it through `Execution` and stores the result in `ActionQueue`.

`ActionQueue` is the runtime buffer. It may use `ActionChunkCursor` internally, but it adds refill thresholds, async/process/remote support, smoothing, RTC merge behavior, and telemetry.

## Why This Shape

```text
PolicyRuntime tick:
  obs = robot.get_observation()
  execution.maybe_request(obs)
  action = action_queue.pop_or_none()
  robot.send_action(action)
```

This keeps runtime behavior out of `InferenceModel`. It also lets the same loop use sync, thread, process, or remote inference.

## Remote Inference

```python
runtime = PolicyRuntime(
    robot=robot,
    model=RemoteInferenceModel(endpoint="grpc://gpu-host:50051"),
    execution=RemoteExecution(endpoint="grpc://gpu-host:50051"),
    fps=30,
)
```

Server side:

```bash
physicalai serve --config policy_server.yaml
```

The robot host owns the robot loop and `ActionQueue`. The server host owns the real `InferenceModel`.

## Current Recommendation

Build the runtime in this order:

1. Add `predict_action_chunk()` and `close()` to `InferenceModel`.
2. Add `PolicyRuntime`, `Execution`, and `ActionQueue`.
3. Add async thread/process execution.
4. Add RTC-specific merging and delay tracking.
5. Add `RemoteExecution` and `PolicyServer`.

Open discussion points:

1. Should `select_action()` stay as direct-call convenience only?
2. Should runtime-owned chunking be the recommended path for all robot loops?
3. Should benchmark rollouts continue to call `select_action()` directly, or should some benchmark modes evaluate `PolicyRuntime` against gym-like robot adapters?
