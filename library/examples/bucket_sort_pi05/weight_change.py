"""Fraction of weights that changed between two Lightning checkpoints, per component and dtype."""
import sys
from collections import defaultdict
import torch

a = torch.load(sys.argv[1], map_location="cpu", weights_only=False, mmap=True)["state_dict"]
b = torch.load(sys.argv[2], map_location="cpu", weights_only=False, mmap=True)["state_dict"]
acc = defaultdict(lambda: [0, 0, 0.0, 0.0])  # n, changed, sum|d|, sum|w|
for k, x in a.items():
    y = b[k]
    if not x.is_floating_point():
        continue
    g = "vision" if "vision_tower" in k else "expert" if "gemma_expert" in k else "llm" if "language_model" in k else "other"
    key = f"{g:7s} {str(x.dtype).removeprefix('torch.')}"
    x, y = x.float(), y.float()
    s = acc[key]
    s[0] += x.numel(); s[1] += (x != y).sum().item(); s[2] += (x - y).abs().sum().item(); s[3] += x.abs().sum().item()
for key, (n, c, d, w) in sorted(acc.items()):
    print(f"{key:18s} params {n/1e9:5.2f}B  changed {100*c/n:5.1f}%  sum|d|/sum|w| {d/w:.2e}")
