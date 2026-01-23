# Explanation

Understanding-oriented documentation that explains why things work the way they do.

## Architecture

High-level explanations of Geti Action's design:

- [Why Action Chunking?](why-action-chunking.md) - The key insight behind modern imitation learning
- [Architecture Overview](architecture.md) - How the components fit together

## Design Documentation

Detailed design documentation for each module (for contributors and advanced users):

| Module                             | Description                                        |
| ---------------------------------- | -------------------------------------------------- |
| [Benchmark](benchmark/overview.md) | Policy evaluation and benchmarking                 |
| [CLI](cli/overview.md)             | Command-line interface using PyTorch Lightning CLI |
| [Config](config/overview.md)       | Configuration system (dataclasses, Pydantic, YAML) |
| [Data](data/overview.md)           | Dataset management and data loading                |
| [Gyms](gyms/overview.md)           | Simulation environments for training               |
| [Policy](policy/overview.md)       | Policy implementations and base classes            |
| [Trainer](trainer/overview.md)     | Training infrastructure and metrics                |
| [Export](export/overview.md)       | Model export (OpenVINO, ONNX, Torch Export)        |
| [Inference](inference/overview.md) | Production deployment                              |

## See Also

- [Getting Started](../getting-started/) - Learn by doing
- [How-To Guides](../how-to/) - Solve specific problems
