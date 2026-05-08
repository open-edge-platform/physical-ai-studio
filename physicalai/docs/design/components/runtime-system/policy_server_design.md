# PolicyServer and RemoteExecution

This is the remote inference design for `PolicyRuntime`.

## Goal

Run the robot loop on one host and the policy model on another host.

```text
Robot host                              Server host
----------                              -----------
PolicyRuntime                           PolicyServer
  Robot                                   InferenceModel
  cameras                                 runner / backend
  ActionQueue                             predict_action_chunk()
  RemoteExecution  <------ gRPC ------>   warmup / health
```

The robot host owns timing and action dispatch. The server host owns model inference.

## Robot Host

```python
from physicalai.runtime import PolicyRuntime, RemoteExecution
from physicalai.inference.remote import RemoteInferenceModel

runtime = PolicyRuntime(
    robot=robot,
    model=RemoteInferenceModel(endpoint="grpc://gpu-host:50051"),
    execution=RemoteExecution(endpoint="grpc://gpu-host:50051"),
    fps=30,
)

runtime.run()
```

`RemoteExecution` implements the same interface as sync and async execution:

```python
class RemoteExecution(Execution):
    def start(self, action_queue, model): ...
    def maybe_request(self, observation): ...
    def warmup(self, sample_observation, n=2): ...
    def stop(self): ...
```

It sends observation snapshots to the server and pushes returned chunks into the local `ActionQueue`.

## Server Host

```bash
physicalai serve --config policy_server.yaml
```

Server config:

```yaml
server:
  host: 0.0.0.0
  port: 50051

model:
  class_path: physicalai.inference.InferenceModel
  init_args:
    path: ./exports/pi0_policy
```

`PolicyServer` loads the real `InferenceModel` and exposes:

- handshake
- warmup
- health
- `predict_action_chunk`
- optionally `select_action`

## Data Flow

```text
PolicyRuntime tick
  obs = robot.get_observation()
  execution.maybe_request(obs)

RemoteExecution
  serialize obs
  send PredictRequest

PolicyServer
  chunk = model.predict_action_chunk(obs)
  send PredictReply

RemoteExecution
  action_queue.push_chunk(chunk)

PolicyRuntime tick
  action = action_queue.pop_or_none()
  robot.send_action(action)
```

## Transport

Recommended default: gRPC bidirectional streaming.

Sketch:

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
  optional int32 inference_delay = 4;
  optional ActionChunk prev_chunk_left_over = 5;
}

message PredictReply {
  string request_id = 1;
  double t0 = 2;
  Tensor actions = 3;
  optional double policy_dt = 4;
  map<string, Tensor> extra = 5;
}
```

## RTC

RTC works the same way as local async execution.

```text
RemoteExecution computes delay
RemoteExecution sends inference_delay + prev_chunk_left_over
PolicyServer calls model.predict_action_chunk(...)
server-side runner uses FlowMatching(guidance=RTC())
```

The client still owns `ActionQueue` and `RTCQueueMerger`. The server returns chunks; it does not smooth or dispatch actions.

## Failure Policy

| Failure | Behavior |
|---|---|
| Cannot connect at startup | `start()` raises; runtime exits cleanly |
| Connection lost with no request in flight | reconnect until `reconnect_budget_s` is exhausted |
| Connection lost with request in flight | drop that request; next observation creates a new request |
| Server error | surface error on next runtime call |
| Deadline exceeded | drop request; continue with next tick/request |
| Schema mismatch | fail during handshake |

Do not retry stale observations. In control loops, a late retry is usually worse than dropping the request.

## CLI

`physicalai serve` is registered by the runtime distribution:

```toml
[project.entry-points."physicalai.cli.subcommands"]
serve = "physicalai.cli.serve:register"
```

It uses the same runtime-side CLI as `physicalai run`, without Torch or Lightning imports.

## Out of Scope

- multi-robot single-server
- server-side action smoothing
- server-side robot state
- model hot-swap semantics beyond rejecting in-flight requests
- low-level transport optimization beyond gRPC defaults

## Build Target

Build after local `PolicyRuntime`, `AsyncExecution`, and `ActionQueue` are stable.

Acceptance test:

```text
RemoteExecution + PolicyServer produces actions equivalent to
AsyncExecution(transport="process") for the same model and observations.
```
