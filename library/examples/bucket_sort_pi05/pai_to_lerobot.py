"""Write a physicalai pi05 Lightning checkpoint as a LeRobot pretrained_model dir (processors copied from a LeRobot run)."""
import shutil
import sys
from pathlib import Path
import torch
from safetensors import safe_open
from safetensors.torch import save_file

src, template, out = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
sd = torch.load(src, map_location="cpu", weights_only=False, mmap=True)["state_dict"]
with safe_open(template / "model.safetensors", "pt") as f:
    lr_keys = set(f.keys())
mapped, miss = {}, []
for k, v in sd.items():
    for cand in (k, k.removeprefix("model."), "model." + k.removeprefix("model.model.")):
        if cand in lr_keys:
            mapped[cand] = v.float().contiguous()
            break
    else:
        miss.append(k)
print("pai keys", len(sd), "mapped", len(mapped), "lr keys", len(lr_keys))
print("unmapped pai (first 5)", miss[:5])
print("lr keys not filled (first 5)", sorted(lr_keys - set(mapped))[:5], len(lr_keys - set(mapped)))
if len(mapped) == len(lr_keys):
    out.mkdir(parents=True, exist_ok=True)
    for p in template.iterdir():
        if p.name != "model.safetensors":
            shutil.copy(p, out / p.name)
    save_file(mapped, out / "model.safetensors")
    print("wrote", out)
