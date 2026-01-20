# User Guides

Practical guides for training and deploying policies.

## Guides

| Guide                                         | Description                                    |
| --------------------------------------------- | ---------------------------------------------- |
| **[CLI Guide](cli.md)**                       | Train policies from the command line           |
| **[Benchmark Guide](benchmark.md)**           | Evaluate policies on standardized environments |
| **[Export & Inference](export_inference.md)** | Deploy models to production                    |

## Quick Start

```bash
# 1. Install
cd library && uv sync --all-extras && source .venv/bin/activate

# 2. Train
getiaction fit --config configs/getiaction/act.yaml

# 3. Benchmark
getiaction benchmark --config configs/benchmark/libero.yaml

# 4. Export
# See Export & Inference guide
```

## See Also

- **[Design Documentation](../design/)** - Architecture details (for contributors)
- **[Library README](../../README.md)** - Installation and overview
