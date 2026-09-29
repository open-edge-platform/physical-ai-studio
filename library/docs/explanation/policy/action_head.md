# Action Heads

Action heads are shared components that turn a conditioning context into a
chunk of actions `(B, chunk_size, action_dim)`.

```python
from physicalai.policies.components import ActionHead, IterativeActionHead
```

- `ActionHead` defines `compute_loss(actions, context)` and `sample(context)`.
- `IterativeActionHead` adds the denoising loop shared by flow matching and
  diffusion. Subclasses implement `denoise`, `timesteps` and `step`.
- Neither class holds weights, so existing policies can adopt them without
  changing their checkpoint keys.

See the [action heads README](../../../src/physicalai/policies/components/action_heads/README.md)
for the full design, the time conventions and an example.
