# Feature Extractor

`FeatureExtractor` is a shared vision encoder for policies. It wraps a timm model,
a torchvision model or any `nn.Module` and returns features from chosen layers.

```python
from physicalai.policies.components import FeatureExtractor
```

## Inputs

- `forward(images)` takes a batch of shape `(B, C, H, W)` with values in `[0, 1]`.
  Inputs are normalized internally (timm pretrained statistics, else ImageNet).
- `encode_image(image)` is a convenience helper for a PIL image, numpy array or tensor,
  or a list of them. Numpy arrays and PIL images are assumed `(H, W, C)` and tensors
  `(C, H, W)`; override with `channels_last=True/False`. Use `size=` to resize.

## Options

| Argument    | Values                      | Meaning                                            |
| ----------- | --------------------------- | -------------------------------------------------- |
| `backbone`  | timm name or `nn.Module`    | A string is created with `timm.create_model`       |
| `layers`    | module names or `None`      | e.g. `["layer4"]`, `["blocks.11"]`; `None` = final |
| `pooling`   | `none`, `avg`, `max`, `cls` | Per-layer reduction (see below)                    |
| `aggregate` | `none`, `concat`, `mean`    | How multiple layers are combined                   |
| `frozen`    | `True` / `False`            | Freeze weights and keep the backbone in eval mode  |

Pooling works on both architecture families:

| `pooling`     | Convolutional `(B, C, H, W)` | Transformer `(B, N, D)`             |
| ------------- | ---------------------------- | ----------------------------------- |
| `none`        | `(B, C, H, W)`               | patch tokens, prefix tokens removed |
| `avg` / `max` | `(B, C)`                     | `(B, D)` over patch tokens          |
| `cls`         | error                        | `(B, D)` CLS token                  |

With multiple layers, `aggregate="none"` returns a `dict` keyed by layer name,
`"concat"` concatenates on the channel dimension (resizing feature maps to the
largest one) and `"mean"` averages features of equal shape. `feature_dims`
gives the output dimension of each layer.

## Examples

```python
import torchvision

# DINOv2 CLS token -> (B, 384)
FeatureExtractor("vit_small_patch14_dinov2.lvd142m", pooling="cls", dynamic_img_size=True)

# DINOv2 patch tokens -> (B, N, 384), or averaged -> (B, 384)
FeatureExtractor("vit_small_patch14_dinov2.lvd142m", pooling="none", dynamic_img_size=True)
FeatureExtractor("vit_small_patch14_dinov2.lvd142m", pooling="avg", dynamic_img_size=True)

# torchvision ResNet18, final stage average pooled -> (B, 512)
resnet = torchvision.models.resnet18(weights="DEFAULT")
FeatureExtractor(resnet, layers=["layer4"], pooling="avg")

# All four ResNet stages, average pooled and concatenated -> (B, 960)
FeatureExtractor(resnet, layers=["layer1", "layer2", "layer3", "layer4"], pooling="avg", aggregate="concat")
```

For models whose transformer tokens do not expose `num_prefix_tokens` (e.g. a
`torch.hub` DINOv2 with registers), pass `num_prefix_tokens=` explicitly.
