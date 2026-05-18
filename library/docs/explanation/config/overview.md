# Configuration System

`physicalai.config` provides a small set of generic building blocks for turning config-like inputs into Python objects.

## Supported Inputs

- `dict`
- YAML or JSON file path
- dataclass instance
- Pydantic `BaseModel`

## Supported Shapes

### Direct kwargs

```python
MyClass.from_dict({"hidden_size": 256, "num_layers": 3})
```

Equivalent to:

```python
MyClass(hidden_size=256, num_layers=3)
```

### `class_path` / `init_args`

```yaml
class_path: mypkg.models.MyClass
init_args:
  hidden_size: 256
  num_layers: 3
```

Equivalent to:

```python
from mypkg.models import MyClass
MyClass(hidden_size=256, num_layers=3)
```

This is the same config shape used by `jsonargparse` and Lightning-style CLIs.

## Nested Instantiation

Nested `class_path` blocks are instantiated recursively inside:

- dicts
- lists
- tuples

Example:

```python
config = {
    "model": {
        "class_path": "mypkg.models.Backbone",
        "init_args": {"hidden_size": 256},
    },
    "optimizer": {
        "class_path": "torch.optim.Adam",
        "init_args": {"lr": 1e-3},
    },
}

policy = MyPolicy.from_dict(config)
```

## Entry Points

### `instantiate_obj(...)`

Use this when you want the generic backend directly.

```python
obj = instantiate_obj(config)
obj = instantiate_obj(config, target_cls=MyClass)
```

### `FromConfig`

Use this when you want class-level sugar:

```python
class MyClass(FromConfig):
    ...

obj = MyClass.from_yaml("config.yaml")
obj = MyClass.from_dataclass(cfg)
obj = MyClass.from_config(any_supported_input)
```

### `@from_config`

Use this when you want the same API without changing the class MRO.

```python
@from_config
class MyClass:
    ...
```

## Dataclass and Pydantic Conversion

`from_dataclass(..., recursive=False)` and `from_pydantic(..., recursive=False)` preserve nested instances by default.

Use `recursive=True` when you want nested objects flattened to plain dicts before instantiation.

This matters when:

- your constructor expects nested dataclass / Pydantic instances: use `recursive=False`
- your constructor expects plain dictionaries: use `recursive=True`

## Validation

Validation happens in the source object or target class, not in the instantiator itself:

- Pydantic validates before instantiation.
- Dataclasses can validate in `__post_init__`.
- Constructors can validate in `__init__`.

`instantiate_obj(...)` does not perform its own schema validation beyond dispatch and import errors.
