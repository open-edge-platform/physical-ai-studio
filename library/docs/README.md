# Library Documentation

Documentation for the Geti Action Python library.

**→ [Start Here](index.md)** - Documentation home page

## Quick Navigation

| Section                             | Description                               |
| ----------------------------------- | ----------------------------------------- |
| [Getting Started](getting-started/) | Installation, quickstart, and first steps |
| [How-To Guides](how-to/)            | Goal-oriented guides for specific tasks   |
| [Explanation](explanation/)         | Architecture and design documentation     |

## Quick Start

```bash
# Install
pip install getiaction

# Train
getiaction fit --config configs/getiaction/act.yaml

# Benchmark
getiaction benchmark \
    --benchmark getiaction.benchmark.LiberoBenchmark \
    --policy getiaction.policies.ACT \
    --ckpt_path ./checkpoints/model.ckpt

# Export
policy.export("./exports", backend="openvino")
```

## See Also

- **[Library README](../README.md)** - Installation and overview
- **[Main Repository](../../README.md)** - Project overview
