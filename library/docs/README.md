# Library Documentation

Documentation for the Geti Action Python library.

## User Guides

Practical guides for common workflows:

- **[CLI Guide](guides/cli.md)** - Train policies from the command line
- **[Benchmark Guide](guides/benchmark.md)** - Evaluate policies on standardized environments
- **[Export & Inference](guides/export_inference.md)** - Deploy models to production

## Design Documentation

Architecture and implementation details (for contributors):

- **[Architecture Overview](design/)** - Module structure
- **[CLI Design](design/cli/)** - Command-line interface
- **[Config System](design/config/)** - Configuration patterns
- **[Data Module](design/data/)** - Dataset management
- **[Policies](design/policy/)** - Policy implementations
- **[Gyms](design/gyms/)** - Simulation environments
- **[Export](design/export/)** - Model export backends
- **[Inference](design/inference/)** - Production deployment

## Quick Reference

```bash
# Train
getiaction fit --config configs/getiaction/act.yaml

# Benchmark
getiaction benchmark --config configs/benchmark/libero.yaml

# Export
policy.export("./exports", backend="openvino")
```

## See Also

- **[Library README](../README.md)** - Installation and quick start
- **[Main Repository](../../README.md)** - Project overview
