# Instantiation

`instantiate.py` contains the generic backend used by the rest of the config system.

## Main API

```python
instantiate_obj(config, *, key=None, target_cls=None) -> object
```

`config` may be:

- `dict`
- YAML / JSON file path
- dataclass instance
- Pydantic `BaseModel`

## Core Rule

All roads lead to:

```python
instantiate_obj_from_dict(config, *, key=None, target_cls=None)
```

Behavior:

1. If `key` is provided, extract that sub-config first.
2. If the selected config has `class_path`, import that class and instantiate it.
3. Otherwise, if `target_cls` is provided, call `target_cls(**config)`.
4. Recursively instantiate nested `class_path` configs inside dicts, lists, and tuples.

If both `class_path` and `target_cls` are present, `class_path` wins.

## Examples

### Import from `class_path`

```python
optimizer = instantiate_obj({
    "class_path": "torch.optim.Adam",
    "init_args": {"lr": 1e-3},
})
```

### Direct kwargs via `target_cls`

```python
model = instantiate_obj(
    {"hidden_size": 256, "num_layers": 3},
    target_cls=MyModel,
)
```

### YAML or JSON file

```python
policy = instantiate_obj("config.yaml")
model = instantiate_obj("train.json", key="model")
```

### Dataclass or Pydantic source

```python
model = instantiate_obj(dataclass_cfg, target_cls=MyModel)
model = instantiate_obj(pydantic_cfg, target_cls=MyModel)
```

## Nested Objects

Nested `class_path` blocks are instantiated automatically:

```python
policy = instantiate_obj({
    "class_path": "mypkg.Policy",
    "init_args": {
        "model": {
            "class_path": "mypkg.Model",
            "init_args": {"hidden_size": 256},
        },
    },
})
```

The same recursion works inside lists and tuples.

## Errors

Common failures:

- missing `key`
- missing `class_path` when no `target_cls` was provided
- invalid import path in `class_path`
- constructor errors from the target class itself

## When To Use It

Use `instantiate_obj(...)` when:

- the config decides the concrete class
- you do not want to mix `FromConfig` into the class
- you are writing infrastructure code that accepts arbitrary config-like inputs
