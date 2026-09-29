# Action Heads

An action head turns a conditioning context into a chunk of actions of shape
`(B, chunk_size, action_dim)`. Policies build the context (for example image or
VLM tokens) and the head does the rest: the training loss and the sampler.

```python
from physicalai.policies.components.action_heads import ActionHead, IterativeActionHead
```

## Context

The context is a `dict[str, Tensor]`. Heads document which keys they read.
A common layout is:

| Key      | Shape       | Meaning                             |
| -------- | ----------- | ----------------------------------- |
| `tokens` | `(B, S, D)` | Conditioning tokens                 |
| `mask`   | `(B, S)`    | Optional bool mask, `True` = attend |

A flat dict of tensors keeps the head independent of the policy, and lets the
sampling loop be captured in a CUDA or XPU graph.

## `ActionHead`

The minimal interface. Use it directly for one-shot heads such as regression.

| Method                                            | Purpose                                |
| ------------------------------------------------- | -------------------------------------- |
| `compute_loss(actions, context)`                  | Unreduced per-element loss `(B, T, A)` |
| `sample(context, *, noise=None, num_steps=None)`  | Action chunk `(B, T, A)`               |
| `forward(context, *, noise=None, num_steps=None)` | Same as `sample`                       |

The loss is not reduced, so policies can ignore padded action steps with
`physicalai.policies.utils.loss.reduce_losses`:

```python
losses = head.compute_loss(actions, context)
loss = reduce_losses(losses, in_episode_bound(batch))
```

## `IterativeActionHead`

The shared base for flow matching and diffusion. Sampling is one loop:

```python
x = noise
for t, t_next in zip(timesteps[:-1], timesteps[1:]):
    x = step(x, denoise(x, t, context), t, t_next)
```

A subclass implements four things:

| Method                           | Flow matching example         | Diffusion example  |
| -------------------------------- | ----------------------------- | ------------------ |
| `denoise(x_t, t, context)`       | Predict the velocity          | Predict the noise  |
| `timesteps(num_steps, ...)`      | `linspace(0, 1, num_steps+1)` | `[99, 89, ..., 0]` |
| `step(x_t, output, t, t_next)`   | Euler: `x + (t_next - t) * v` | DDPM / DDIM update |
| `compute_loss(actions, context)` | MSE to the velocity target    | MSE to the noise   |

Optionally override `prepare_context(context)` for work that does not change
between steps, such as projecting cross-attention keys and values once.

`integrate(x, context, timesteps)` runs the loop with tensor operations only.
Keep `denoise` and `step` free of Python-side randomness, `.item()` calls and
shape changes, so the loop stays graph-capturable.

### Time conventions

Existing policies use two flow-matching conventions. Both fit the same loop,
only `timesteps` and `step` differ:

| Policies              | Noise at | Data at | Velocity target   |
| --------------------- | -------- | ------- | ----------------- |
| XR0, RLDX1, MolmoAct2 | `t = 0`  | `t = 1` | `actions - noise` |
| Pi05, SmolVLA         | `t = 1`  | `t = 0` | `noise - actions` |

## Pretrained weights

The base classes register no parameters, buffers or submodules. A module's
`state_dict` keys only come from its own attribute names, so an existing head
can inherit from `ActionHead` without changing its checkpoint keys.

When adopting the base classes in an existing policy:

- Keep module attribute names unchanged (for example `action_in_proj`).
- Check the policy's LoRA target regexes and Hugging Face key remapping, which
  match these names literally.
- Store constant tensors, such as noise schedules, as non-persistent buffers
  (`register_buffer(..., persistent=False)`) so they never enter checkpoints.

## Example

```python
import torch
from torch import nn

from physicalai.policies.components.action_heads import Context, IterativeActionHead


class EulerFlowHead(IterativeActionHead):
    def __init__(self, context_dim: int, chunk_size: int, action_dim: int) -> None:
        super().__init__(chunk_size, action_dim, num_inference_steps=10)
        self.velocity = nn.Linear(action_dim + context_dim + 1, action_dim)

    def denoise(self, x_t: torch.Tensor, t: torch.Tensor, context: Context) -> torch.Tensor:
        pooled = context["tokens"].mean(dim=1, keepdim=True).expand(-1, x_t.shape[1], -1)
        time = t[:, None, None].expand(-1, x_t.shape[1], 1)
        return self.velocity(torch.cat([x_t, pooled, time], dim=-1))

    def timesteps(self, num_steps: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return torch.linspace(0, 1, num_steps + 1, device=device, dtype=dtype)

    def step(self, x_t, model_output, t, t_next):
        return x_t + (t_next - t) * model_output

    def compute_loss(self, actions: torch.Tensor, context: Context) -> torch.Tensor:
        noise = torch.randn_like(actions)
        t = torch.rand(actions.shape[0], device=actions.device)[:, None, None]
        x_t = (1 - t) * noise + t * actions
        return (self.denoise(x_t, t[:, 0, 0], context) - (actions - noise)) ** 2


head = EulerFlowHead(context_dim=384, chunk_size=16, action_dim=7)
context = {"tokens": torch.randn(2, 256, 384)}
actions = head.sample(context)  # (2, 16, 7)
```
