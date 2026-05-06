# Policy Runtime Design

This is the concise design for running a trained PhysicalAI policy on a robot.

The main boundary:

```text
InferenceModel   computes actions
PolicyRuntime    runs the robot control loop that uses those actions
```

## 1. Goal

Create a small `physicalai.runtime` package that can run a policy on robot hardware from Python or the CLI.

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

CLI:

```bash
physicalai run --config so101_act.yaml --duration-s 60
```

## 2. Component Ownership

| Component | Owns | Does not own |
|---|---|---|
| `InferenceModel` | load model, preprocess input, run backend, return actions | robot timing, action queue, callbacks, shutdown |
| `InferenceRunner` | policy computation strategy, e.g. single pass, flow matching, temporal ensemble | robot loop, async scheduling, runtime queueing |
| `ActionChunkCursor` | internal helper: hold one chunk, pop one action at a time | refill policy, smoothing, async, RTC |
| `Execution` | when and where inference runs: sync, thread, process, remote | queueing policy, robot IO |
| `ActionQueue` | store chunks, merge chunks, smooth boundaries, pop one action per tick | model inference, robot IO |
| `PolicyRuntime` | observe robot, call `Execution`, pop action, send action, callbacks, timing | policy math |
| `Benchmark` / `LiberoBenchmark` | evaluate policies across gyms/tasks, episodes, success rate, reward, episode length, FPS, videos, JSON/CSV export | production robot-loop semantics |

## 3. `select_action()` vs `predict_action_chunk()`

### `select_action()`

```python
action = model.select_action(observation)
```

`select_action()` returns one action now.

Use it for:

- simple scripts
- tests
- demos
- existing code that directly calls the model

For chunk-producing policies, `select_action()` uses an internal `ActionChunkCursor`:

```python
if cursor.empty():
    cursor.push_chunk(model.predict_action_chunk(obs))

return cursor.pop()
```

The cursor only tracks position inside one chunk. It is not `ActionQueue`. It is hidden, synchronous, and only adapts chunk-producing policies to the one-action API.

But `select_action()` should not own:

- robot FPS
- async inference
- remote inference
- runtime callbacks
- production action queue behavior

### `predict_action_chunk()`

```python
chunk = model.predict_action_chunk(observation)

actions = chunk["actions"]      # np.ndarray, shape (H, action_dim)
policy_dt = chunk.get("policy_dt")
t0 = chunk.get("t0")
```

`predict_action_chunk()` returns a chunk without consuming it.

Use it when a runtime loop owns timing and queueing:

```text
PolicyRuntime
  -> Execution.maybe_request(obs)
  -> InferenceModel.predict_action_chunk(obs)
  -> ActionQueue.push_chunk(chunk)
  -> ActionQueue.pop_or_none()
  -> robot.send_action(action)
```

### Contract: shape-stable across runners

Both methods must work for any runner. `InferenceModel` adapts; callers do not branch on runner type.

| Runner | `select_action()` | `predict_action_chunk()` |
|---|---|---|
| single-pass | runner output | wrap as `(1, D)` chunk |
| chunk-producing | pop one via `ActionChunkCursor` | runner output |

This is deliberate. If `predict_action_chunk()` raised on single-pass runners, `PolicyRuntime`, `ActionQueue`, `Benchmark`, and `PolicyServer` would each have to branch on runner type. The cursor and the 1-step wrap exist to keep that branching in one place.

## 4. Chunking and Queueing

Three layers, one shared low-level helper.

```text
action chunk        Mapping[str, Any] returned by predict_action_chunk()
ActionChunkCursor   internal helper: pop one action from a chunk
ActionQueue         runtime queue used by PolicyRuntime
```

### Direct-call path

```python
action = model.select_action(obs)
```

Internally:

```python
if cursor.empty():
    cursor.push_chunk(model.predict_action_chunk(obs))

return cursor.pop()
```

For code that does not use `PolicyRuntime`.

### Runtime path

```python
chunk = model.predict_action_chunk(obs)
action_queue.push_chunk(chunk)
action = action_queue.pop_or_none()
```

Recommended for robot control loops. `ActionQueue` may use `ActionChunkCursor` internally, but it owns runtime concerns: background inference, process workers, remote inference, refill thresholds, cross-chunk smoothing, late results, RTC overlap state.

`InferenceModel` / `InferenceRunner` must not import `ActionQueue`. If both layers need pop-from-chunk mechanics, they share `ActionChunkCursor`, not `ActionQueue`.

| Thing | Scope | Owns |
|---|---|---|
| `ActionChunkCursor` | internal helper | current chunk position |
| `select_action()` | model convenience API | sync refill when cursor empty |
| `ActionQueue` | runtime | refill policy, smoothing, RTC, telemetry |

## 5. Core Runtime API

```python
class PolicyRuntime:
    def __init__(
        self,
        robot: Robot,
        model: InferenceModel,
        execution: Execution,
        fps: float,
        cameras: Mapping[str, Camera] | None = None,
        action_queue: ActionQueue | None = None,
        callbacks: Sequence[Callback] = (),
        return_to_home: bool = False,
    ): ...

    def run(self, *, duration_s: float | None = None) -> None: ...
    def stop(self) -> None: ...

    @classmethod
    def from_config(cls, path: str | Path) -> "PolicyRuntime": ...
```

Loop shape:

```python
while running:
    obs = robot.get_observation()
    obs.update(read_cameras(cameras))

    execution.maybe_request(obs)
    action = action_queue.pop_or_none()

    if action is None:
        action = hold_position()

    robot.send_action(action)
    sleep_until_next_tick()
```

## 6. Execution Modes

`Execution` decides when and where inference happens.

```python
class Execution:
    def start(self, action_queue: ActionQueue, model: InferenceModel) -> None: ...
    def maybe_request(self, observation: Mapping[str, Any]) -> None: ...
    def warmup(self, sample_observation: Mapping[str, Any], n: int = 2) -> None: ...
    def stop(self) -> None: ...
```

| Implementation | Where inference runs | Use case |
|---|---|---|
| `SyncExecution(mode="single_action")` | runtime thread | simple policies |
| `SyncExecution(mode="chunk")` | runtime thread | chunk policies, no background worker |
| `AsyncExecution(transport="thread")` | worker thread | avoid blocking the control loop |
| `AsyncExecution(transport="process")` | worker process | Studio-style model worker |
| `RemoteExecution` | remote server | robot host without policy weights/GPU |

## 7. ActionQueue

```python
queue = ActionQueue(
    smoother=LerpChunkSmoother(duration_frames=10),
    merger=ReplaceMerger(),
)
```

`ActionQueue` owns runtime action buffering.

```python
queue.push_chunk(chunk)
action = queue.pop_or_none()
```

RTC can use a specialized merger:

```python
queue = ActionQueue(merger=RTCQueueMerger())
```

## 8. Config Example

```yaml
runtime:
  class_path: physicalai.runtime.PolicyRuntime
  init_args:
    fps: 30
    robot:
      class_path: physicalai.robot.so101.SO101
      init_args:
        port: /dev/ttyACM0
    model:
      class_path: physicalai.inference.InferenceModel
      init_args:
        path: ./exports/act_policy
    execution:
      class_path: physicalai.runtime.SyncExecution
      init_args:
        mode: chunk
```

## 9. CLI

Runtime CLI lives in the `physicalai` distribution, not in the training distribution.

```toml
[project.scripts]
physicalai = "physicalai.cli.main:main"

[project.entry-points."physicalai.cli.subcommands"]
run = "physicalai.cli.run:register"
serve = "physicalai.cli.serve:register"
```

Training commands plug in from the training distribution:

```toml
[project.entry-points."physicalai.cli.subcommands"]
fit = "physicalai.train.cli:register_fit"
validate = "physicalai.train.cli:register_validate"
test = "physicalai.train.cli:register_test"
predict = "physicalai.train.cli:register_predict"
benchmark = "physicalai.benchmark.cli:register"
```

This keeps `physicalai run` usable on inference hosts without Torch or Lightning.

## 10. Benchmarking vs Runtime

`library/src/physicalai/benchmark/` is not just a latency/throughput microbenchmark package.

Current shape:

```python
from physicalai.benchmark import LiberoBenchmark
from physicalai.inference import InferenceModel

benchmark = LiberoBenchmark(task_suite="libero_10", num_episodes=20)
model = InferenceModel.load("./exports/act_policy")

results = benchmark.evaluate(model)
print(results.summary())
```

`Benchmark` / `LiberoBenchmark` own evaluation orchestration:

- gyms and task suites
- episodes and max steps
- policy comparison
- success rate
- reward
- episode length
- average FPS
- optional video recording
- JSON/CSV result export

Today, benchmark rollouts evaluate `Policy` or `InferenceModel` by calling `select_action()`. That is correct for direct policy evaluation.

`PolicyRuntime` has a different job: production robot-loop semantics.

| Concern | `Benchmark` | `PolicyRuntime` |
|---|---|---|
| Task suite / gym orchestration | yes | no |
| Episode aggregation and success metrics | yes | no |
| Video/result export | yes | no |
| Robot/camera connection lifecycle | no | yes |
| FPS-controlled robot loop | no, except measuring rollout FPS | yes |
| Runtime-owned action queue | only if explicitly benchmarking runtime behavior | yes |
| Async/process/remote inference scheduling | only if explicitly configured through runtime components | yes |

Recommended split:

1. Keep `Benchmark.evaluate(model)` using `select_action()` for standard policy/task evaluation.
2. Add optional runtime benchmarks later only when we want to evaluate `PolicyRuntime` itself, e.g. mock robot loop FPS, queue behavior, async execution, or remote execution.
3. Do not make `Benchmark` a second implementation of the production robot loop.

## 11. Phases

1. Add `InferenceModel.predict_action_chunk()` and `close()`.
2. Add `PolicyRuntime`, `Execution`, `ActionQueue`, callbacks, and validation.
3. Add async thread/process execution.
4. Add RTC delay tracking and `RTCQueueMerger`.
5. Add `RemoteExecution`, `PolicyServer`, and `physicalai serve`.
6. Add deferred extension points only when a concrete consumer needs them.

## 12. Deferred Until Needed

Do not add these in the initial API unless a concrete consumer needs them:

- `ObservationAssembler`
- `ActionArbiter`
- `ActionFilter` / `SafetyGate`
- `ActionInterpolator`
- strategy classes such as sentry, HIL, highlight, DAgger

For now, product workflows should compose around `PolicyRuntime` with callbacks and consumer-owned code.
