# Policy Components

Reusable building blocks for policies. Prefer these over policy-specific copies.

| Component                                  | What it is                                                                                    |
| ------------------------------------------ | --------------------------------------------------------------------------------------------- |
| [`action_heads/`](./action_heads/)         | `ActionHead`, `IterativeActionHead`, `DiffusionActionHead`: turn a context into action chunks |
| [`vision_encoder.py`](./vision_encoder.py) | `VisionEncoder`: pooled, multi-layer features from timm/torchvision backbones                 |
| [`nn.py`](./nn.py)                         | Small layers: timestep encoders, category-specific MLPs, `swish`                              |

```python
from physicalai.policies.components import ActionHead, IterativeActionHead, SinusoidalPositionalEncoding, VisionEncoder
```

Components must not change the `state_dict` keys of the policies that use them,
so pretrained checkpoints keep loading. See the
[action heads README](./action_heads/README.md#pretrained-weights) for details.
