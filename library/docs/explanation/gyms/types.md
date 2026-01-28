# Types

Shared type definitions used across gym environments.

```python
SingleOrBatch[T] = T | list[T] | np.ndarray
```

Represents scalar or batched values.
Commonly returned in rewards, termination flags, and env metadata.
