# GetiAction Documentation

This directory contains both user guides and design documentation.

## Documentation Structure

```text
docs/
├── guides/             # User-facing documentation
│   ├── README.md
│   └── cli.md          # CLI usage guide
│
└── design/             # Developer documentation
    ├── intro.md        # Architecture overview
    ├── cli/            # CLI design docs
    ├── config/         # Config system design docs
    ├── data/           # Data module design docs
    ├── gyms/           # Gym environment design docs
    ├── policy/         # Policy design docs
    └── trainer/        # Training system design docs
```

## Quick Navigation

### 👤 For Users

Start here if you want to **use GetiAction** to train policies:

- **[User Guides](guides/)** - Practical how-to documentation
  - [CLI Guide](guides/cli.md) - Using the command-line interface
  - [LeRobot Guide](guides/lerobot.md) - Using LeRobot policies

### 👨‍💻 For Developers

Start here if you want to **contribute to GetiAction** or understand the implementation:

- **[Design Documentation](design/)** - Architecture and implementation
  - [Architecture Overview](design/intro.md) - System overview
  - [CLI Design](design/cli/) - CLI implementation
  - [Config Design](design/config/) - Configuration system
  - [Policy Design](design/policy/) - Policy architecture
    - [LeRobot Integration](design/policy/lerobot.md) - LeRobot policy wrappers
  - [Data Design](design/data/) - Data module architecture

## Getting Started

### New Users

1. Read the [main README](../../README.md) for project overview
2. Check the [CLI Guide](guides/cli.md) to learn basic commands
3. Start training!

### New Developers

1. Read the [Architecture Overview](design/intro.md) to understand the system
2. Review relevant design docs for the area you're working on
3. Check existing implementations for patterns
4. Start contributing!

## Related Resources

- **[Main README](../../README.md)** - Project overview
- **[Contributing Guide](../../CONTRIBUTING.md)** - How to contribute
- **[Configuration Examples](../../configs/)** - Example YAML files
- **[Tests](../tests/)** - Test suite

## Need Help?

- **Issues:** See existing documentation or [open an issue](https://github.com/samet-akcay/geti-action/issues)
- **Questions:** Check the guides first, then ask in discussions
- **Contributions:** Follow the contributing guide

---
