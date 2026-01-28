# Benchmark Design

Policy evaluation and benchmarking infrastructure.

## Documents

- **[Overview](overview.md)** - Architecture and components

## Components

| Component          | Purpose                                   |
| ------------------ | ----------------------------------------- |
| `Benchmark`        | Abstract base class for evaluation suites |
| `LiberoBenchmark`  | LIBERO task suite implementation          |
| `BenchmarkResults` | Results aggregation and export            |
| `VideoRecorder`    | Episode video capture                     |

## Related

- [Gyms](../gyms/overview.md) - Environment wrappers
- [Eval](../eval/rollout_metric.md) - Rollout metrics
- [CLI](../cli/overview.md) - Command-line interface
