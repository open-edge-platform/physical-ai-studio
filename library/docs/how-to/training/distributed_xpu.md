# Multi-GPU training on Intel XPU

## Why XPU needs its own DDP setup

On PCIe-only Intel Arc Pro / BMG (B-series) cards there is no XeLink bridge
between GPUs, so standard Level-Zero peer-to-peer (P2P) transfers between
devices are unsupported. Running collectives (`all_reduce`, `broadcast`, ...)
without accounting for that can silently corrupt results or hang the process
indefinitely. `physicalai.devices.xpu` provides the accelerator and strategy
that configure this correctly:

- [`XPUAccelerator`](../../../src/physicalai/devices/xpu/accelerator.py) (registered as `"xpu"`)
- [`XPUDDPStrategy`](../../../src/physicalai/devices/xpu/strategy.py) (registered as `"xpu_ddp"`), which wraps
  Lightning's `DDPStrategy` and uses Intel's `xccl` backend (oneCCL) for the collective communication.

Both are registered as soon as `physicalai.devices.xpu` is imported (the
`physicalai.train` package does this for you).

## Required environment variables

Source these before launching `torchrun` (or `accelerate`) on Intel XPU:

```bash
# Force oneCCL to handle rank synchronization via torchrun's rendezvous env-vars
export CCL_PROCESS_LAUNCHER=torchrun

# Force host-staging (USM) instead of Level-Zero direct P2P on non-XeLink cards
export CCL_TOPO_P2P_ACCESS=0

# Select TCP as the libfabric provider (recommended for stability)
export FI_PROVIDER=tcp

# Route Level-Zero IPC handles via pidfd_getfd(2) (Linux kernel >= 5.6)
export CCL_ZE_IPC_EXCHANGE=pidfd

# Use the libfabric transport layer instead of MPI transport
export CCL_ATL_TRANSPORT=ofi

# Optional: share the Xeon + Xe memory pool
export CCL_ZE_SHARED_DEV_POOL=1
```

These are collected in [`library/scripts/intel_env_combo.sh`](../../../scripts/intel_env_combo.sh); source it directly
instead of retyping them:

```bash
source library/scripts/intel_env_combo.sh
```

## Configuring the trainer

### Python

```python
import lightning.pytorch as pl
# Importing registers "xpu" and "xpu_ddp" with Lightning
from physicalai.devices.xpu import XPUAccelerator, XPUDDPStrategy

trainer = pl.Trainer(
    accelerator="xpu",
    strategy="xpu_ddp",
    devices=2,  # use the first 2 XPU indices
    precision="bf16-mixed",
)
```

### YAML

```yaml
trainer:
  max_epochs: 30
  accelerator: xpu
  strategy: xpu_ddp
  devices: 2 # an integer count resolves to the first N XPU indices, e.g. [0, 1]
  precision: bf16-mixed
```

## Running

```bash
source library/.venv/bin/activate
source library/scripts/intel_env_combo.sh

torchrun --nproc_per_node=2 path/to/your_training_script.py \
    --device xpu \
    --devices 2 \
    [any other arguments...]
```

## Known limitation: cap multi-XPU training at 2 devices

More than 2 ranks over `xccl` currently hangs or fails to initialize on
PCIe-only B-series cards. This traces to unresolved bugs in oneCCL /
`torch-xpu-ops`, not to anything in this repo:

- [intel/torch-xpu-ops#2700](https://github.com/intel/torch-xpu-ops/issues/2700) - root-caused to a oneCCL hang in
  `batch_isend_irecv` when send/recv order is reversed. A fix landed in oneCCL master in July 2026, but a
  follow-up report from the same month shows a new failure (`onecclCommInitRank failed with code 11`), and a
  separate report reproduces the original 4-rank timeout on a 4-card BMG system specifically.
- [intel/torch-xpu-ops#2701](https://github.com/intel/torch-xpu-ops/issues/2701) and
  [#2702](https://github.com/intel/torch-xpu-ops/issues/2702) - barrier/monitored-barrier timeouts, worked around
  upstream with `TORCH_DISTRIBUTED_DEBUG=DETAIL`. The real fix depends on XCCL changes that were staged in a fork
  (`daisyden/pytorch`) since deprecated; re-upstreaming is tracked in
  [pytorch/pytorch#183620](https://github.com/pytorch/pytorch/pull/183620).

Separately, our `uv.lock` currently pins `oneccl==2021.17.2` (via the `torch==2.11.0+xpu` wheel), predating all of
the above fixes; PyPI has since moved to the `2022.x` series. Bumping torch/oneCCL is a prerequisite for revisiting
this, not just a driver update.

**Until upstream and our pin catch up, limit multi-XPU training to `devices: 2`.** If you hit hangs even at 2
devices, try adding `TORCH_DISTRIBUTED_DEBUG=DETAIL` to your environment before filing an issue.

## Memory limits for large policies

Training larger policy networks (e.g. **Pi0.5**, 4.14B parameters) on 32 GB B70 cards requires
`train_expert_only=true`. Standard Adam over the full 4.14B trainable parameters needs multiple fp32 buffers
(master weights + moments) per parameter, which overruns 32 GB VRAM and crashes the driver
(`UR_RESULT_ERROR_DEVICE_LOST` / `UR_RESULT_ERROR_OUT_OF_DEVICE_MEMORY`). Freeze the vision-language backbone and
train only the expert heads (693M params) instead:

```python
model = Pi05(
    pretrained_name_or_path="lerobot/pi05_base",
    dtype="bfloat16",
    train_expert_only=True,  # keep this True to stay under 32 GB VRAM
)
```

</content>
</invoke>
