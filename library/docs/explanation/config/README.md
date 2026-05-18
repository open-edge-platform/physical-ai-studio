# Config System

Reference docs for `physicalai.config`.

## Pages

- [Overview](overview.md): what the config system does and when to use each entry point
- [Instantiation](instantiate.md): `instantiate_obj(...)` and related helpers
- [FromConfig](mixin.md): `FromConfig` mixin and `@from_config` decorator

## Core Idea

There is one shared backend:

```python
instantiate_obj_from_dict(config, target_cls=Cls)
```

Behavior:

- If `config` contains `class_path`, import and instantiate that class.
- Otherwise, if `target_cls` is provided, call `target_cls(**config)`.
- Recursively instantiate nested `class_path` objects inside dicts, lists, and tuples.

Everything else is sugar on top of that:

- `instantiate_obj(...)` dispatches by input type.
- `FromConfig` adds `from_yaml`, `from_dict`, `from_pydantic`, `from_dataclass`, and `from_config` classmethods.
- `@from_config` adds the same methods without changing inheritance.

## Related Files

- Code: `library/src/physicalai/config/`
- Tests: `library/tests/unit/config/`
- Walkthrough script: `library/docs/explanation/config/from_config_walkthrough.py`
