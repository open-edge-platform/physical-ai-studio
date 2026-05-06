# Runtime System Design

This directory describes how PhysicalAI should run a trained policy on a robot.

The short version:

```text
InferenceModel   computes actions
Execution        decides when/where inference runs
ActionQueue      buffers chunks and emits one action per tick
PolicyRuntime    runs the robot loop
```

## Recommended Reading Order

For a design review:

1. [design_review_summary.md](./design_review_summary.md)
2. [design_review_deck.md](./design_review_deck.md)
3. [policy_runtime_design.md](./policy_runtime_design.md), only for details
4. [policy_server_design.md](./policy_server_design.md), only for remote inference

For implementation:

1. [policy_runtime_design.md](./policy_runtime_design.md)
2. [policy_server_design.md](./policy_server_design.md)
3. [inference_comparison_report.md](./inference_comparison_report.md), only for background

## Main Example

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

Same shape from the CLI:

```bash
physicalai run --config so101_act.yaml --duration-s 60
```

## Documents

| File | Use it for |
|---|---|
| [design_review_summary.md](./design_review_summary.md) | One-page summary for colleagues |
| [design_review_deck.md](./design_review_deck.md) | Presentation deck |
| [policy_runtime_design.md](./policy_runtime_design.md) | API shape, code examples, ownership rules |
| [policy_server_design.md](./policy_server_design.md) | Remote inference with `PolicyServer` and `RemoteExecution` |
| [inference_comparison_report.md](./inference_comparison_report.md) | Background gap analysis |

## Key Decisions

1. Keep `InferenceModel` as the object that loads and runs the policy.
2. Add `PolicyRuntime` as the object that owns the robot control loop.
3. Keep `select_action()` as the simple one-action API.
4. Add `predict_action_chunk()` as the chunk-producing API used by the runtime.
5. Keep runtime action buffering in `ActionQueue`, not inside `PolicyRuntime` or `InferenceModel`.
6. Keep benchmarking as a measurement harness, not a second runtime.
7. Put `physicalai run` and `physicalai serve` in the runtime distribution, without Torch or Lightning.

## Related Docs

- Broader stack vision: `../../architecture/robot_stack_vision.md`
- Top-level design index: `../../README.md`
