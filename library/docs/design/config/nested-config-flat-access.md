# Nested Config with Flat Access

## Problem

Our policy configs are flat today (~100 keys with prefixes like `optimizer_lr`). The aim of this document is to explore logical grouping (`optimizer.lr`) without losing quick access.

## Proposal

If we are to nest config groups as child dataclasses. The `Config` base class can achieve this by two new features:

1. **`__getattr__` fallback** — `cfg.lr` resolves to `cfg.optimizer.lr` when `lr` isn't a direct field.
2. **`update()` helper** — returns a new frozen config with changes applied to the right child.

```python
@dataclass(frozen=True)
class OptimizerConfig(Config):
    lr: float = 1e-5
    weight_decay: float = 1e-4

@dataclass(frozen=True)
class ModelConfig(Config):
    backbone: str = "resnet18"
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
```

## Access patterns

| Operation | Syntax | Works? |
|---|---|---|
| Read (typed) | `cfg.optimizer.lr` | Yes |
| Read (flat) | `cfg.lr` | Yes (`__getattr__` fallback) |
| Write (flat) | `cfg.lr = 999` | **Blocked** (frozen) |
| Write (nested) | `cfg.optimizer.lr = 999` | **Blocked** (frozen) |
| Update (flat) | `cfg.update(lr=1e-3)` | Yes (returns new config) |
| Update (nested) | `cfg.optimizer.update(lr=1e-3)` | Yes |

## Why frozen + `update()`?

- `frozen=True` catches accidental mutation at write time to avoid shadowing.
- `update()` returns a new copy; original is untouched. Safe for shared/threaded configs.
- Unchanged children are shared by reference (no deep copy cost).

```python
cfg2 = cfg.update(backbone="resnet50", lr=1e-3)
# cfg is unchanged; cfg2.optimizer.lr == 1e-3, cfg2.backbone == "resnet50"
```

## Collision rule

If two children share a field name, first child in declaration is updated. Explicit child update could disambiguate this:

```python
cfg = Model3(
    optimization=cfg.optimization.update(lr=1e-3),
    other=cfg.other.update(lr=0.9),
)
```

## Typing caveat

`cfg.lr` (flat) is a runtime convenience — IDE/mypy won't autocomplete or type-check it. `cfg.optimizer.lr` should be used for full typing. This is a Python limitation of `__getattr__`; no clean workaround without per-field properties (not maintainable for 100+ keys).

## Backwards compatibility

Existing flat YAMLs (`optimizer_lr: 1e-5`) won't auto-migrate to nested shape (`optimizer: { lr: 1e-5 }`). Options:

- One-time migration script for old checkpoints.
- Accept both shapes in `from_dict` (future enhancement).

New configs should use the nested shape.

## Branch

`feat/nested-config-flat-access` — adds `__getattr__` + `update()` to `Config` base, 5 parametrized tests.

