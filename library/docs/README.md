# Library Documentation

Documentation for the PhysicalAI Python library.

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
pip install physicalai-train

# Train
physicalai fit --config configs/physicalai/act.yaml

# Benchmark
physicalai benchmark \
    --benchmark physicalai.benchmark.LiberoBenchmark \
    --policy physicalai.policies.ACT \
    --ckpt_path ./checkpoints/model.ckpt

# Export
policy.export("./exports", backend="openvino")
```

## See Also

- **[Library README](../README.md)** - Installation and overview
- **[Main Repository](../../README.md)** - Project overview

## Pyrefly baseline

The library uses a Pyrefly baseline file (`pyrefly-baseline.json`) to track known
existing type-check errors while still failing on newly introduced issues.

Common commands from the `library/` directory:

```bash
# Run Pyrefly with the configured baseline
uv run pyrefly check -c pyproject.toml

# Regenerate/update the baseline after intentional typing changes
uv run pyrefly check -c pyproject.toml --baseline="pyrefly-baseline.json" --update-baseline
```

When updating the baseline, include a short note in your PR explaining why the
baseline changed and which error categories were added or removed.
