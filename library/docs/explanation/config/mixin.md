# FromConfig

`FromConfig` is a thin class-level wrapper over `instantiate_obj(...)`.

It adds these classmethods:

- `from_yaml(path, *, key=None)`
- `from_dict(config, *, key=None)`
- `from_pydantic(config, *, key=None, recursive=False)`
- `from_dataclass(config, *, key=None, recursive=False)`
- `from_config(config, *, key=None, recursive=False)`

There is also a decorator with identical behavior:

```python
@from_config
class MyClass:
    ...
```

Use the decorator when you want the API but cannot change the class inheritance.

## Direct kwargs vs `class_path`

`from_dict(...)` supports both shapes.

### Direct kwargs

```python
class MyModel(FromConfig):
    def __init__(self, hidden_size: int, num_layers: int = 3) -> None:
        self.hidden_size = hidden_size
        self.num_layers = num_layers

model = MyModel.from_dict({"hidden_size": 256})
```

### `class_path` / `init_args`

```python
model = MyModel.from_dict({
    "class_path": "mypkg.models.BigModel",
    "init_args": {"hidden_size": 512},
})
```

That second form is useful for subclass dispatch driven by config.

## Nested Instantiation

Nested `class_path` objects are instantiated before calling the constructor.

```python
class Policy(FromConfig):
    def __init__(self, model: object) -> None:
        self.model = model

policy = Policy.from_dict({
    "model": {
        "class_path": "mypkg.models.Backbone",
        "init_args": {"hidden_size": 256},
    },
})
```

## Dataclass and Pydantic Sources

### Dataclass

```python
@dataclass
class ModelConfig:
    hidden_size: int = 256

model = MyModel.from_dataclass(ModelConfig())
```

### Pydantic

```python
class ModelConfig(BaseModel):
    hidden_size: int = 256

model = MyModel.from_pydantic(ModelConfig())
```

## `recursive` Flag

`from_dataclass(..., recursive=False)` and `from_pydantic(..., recursive=False)` preserve nested instances by default.

Example:

```python
@dataclass
class Inner:
    size: int = 128

@dataclass
class Outer:
    inner: Inner = field(default_factory=Inner)

class UsesInner(FromConfig):
    def __init__(self, inner: Inner) -> None:
        self.inner = inner

obj = UsesInner.from_dataclass(Outer(), recursive=False)
assert isinstance(obj.inner, Inner)
```

Set `recursive=True` when the target constructor expects nested dictionaries instead.

## `from_config(...)`

`from_config(...)` dispatches by input type:

- path -> `from_yaml`
- Pydantic model -> `from_pydantic`
- dataclass instance -> `from_dataclass`
- dict -> `from_dict`

## Notes

- `FromConfig` does not need to be on every base class. You can mix it into specific classes or use `@from_config`.
- Constructor validation still belongs in Pydantic, dataclass `__post_init__`, or the class `__init__`.
- The full runnable example is in `docs/explanation/config/from_config_walkthrough.py`.
