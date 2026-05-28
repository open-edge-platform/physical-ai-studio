# Intel XPU Multi-Device (DDP) Training Guide

This document describes how to configure and run multi-device Distributed Data Parallel (DDP) training on Intel XPUs (such as PCIe-based Intel Arc Pro B70 / BMG / B-series cards) using the physical-ai-studio framework.

---

## 1. Required Hardware & Environment Settings

On PCIe-only B-series or Arc Pro GPUs (no dedicated XeLink hardware bridges), standard Level-Zero peer-to-peer (P2P) transfers are unsupported. Executing collectives without specific topologies can cause silent corruption, wrong results, or indefinite process hangs.

You **must** source the following 5 environment variables before executing any distributed training via `torchrun`:

```bash
# Force oneCCL to handle rank synchronization via torchrun's rendezvous env-vars
export CCL_PROCESS_LAUNCHER=torchrun

# Force host-staging (USM) instead of Level-Zero direct P2P on non-XeLink cards
export CCL_TOPO_P2P_ACCESS=0

# Select TCP as the libfabric provider (officially recommended/used for stability)
export FI_PROVIDER=tcp

# Route Level-Zero IPC handles using pidfd_getfd(2) syscalls for stability (Linux kernel >= 5.6)
export CCL_ZE_IPC_EXCHANGE=pidfd

# Set libfabric transport layer instead of MPI transport
export CCL_ATL_TRANSPORT=ofi

# (Optional) Optimize Xeon + Xe memory pool sharing
export CCL_ZE_SHARED_DEV_POOL=1
```

For convenience, these can be saved and sourced from a shell script, e.g.:
```bash
source library/scripts/intel_env_combo.sh
```

---

## 2. Using PyTorch Lightning Multi-XPU Strategy

Inside physical-ai-studio, we provide the custom strategy [XPUDDPStrategy](library/src/physicalai/devices/xpu/strategy.py), which registers as `"xpu_ddp"`. This wraps PyTorch Lightning's standard `DDPStrategy` but overrides the backend communication layer to use Intel's `xccl` collective communication library.

### Configuration (`yaml`)
To utilize multiple XPUs in your training configuration files, configure the `trainer` block as follows:

```yaml
trainer:
  max_epochs: 30
  accelerator: xpu
  strategy: xpu_ddp
  devices: 2           # Integer count automatically resolves to the first N XPU indices (e.g., 0, 1)
  precision: bf16-mixed
```

*Note: Our custom `XPUAccelerator.parse_devices` parser has been updated to automatically map an integer number like `devices: 2` into explicit device indexes (e.g. `[0, 1]`), avoiding legacy "device index out of range" issues.*

---

## 3. High-Value Memory Limitations & Optimizer Policies

When training larger policy networks (like **Pi0.5** with 4.14B parameters), memory overhead must be managed stringently:

1. **`train_expert_only=true` is strictly required on B70 (32 GB) cards**:
   Full 4.14B trainable parameter configurations require storing multiple fp32 buffers (master weights + moments) per parameter for standard Adam. This overruns 32 GB of card VRAM and causes a driver crash (`UR_RESULT_ERROR_DEVICE_LOST` / `UR_RESULT_ERROR_OUT_OF_DEVICE_MEMORY`). Freeze the core vision-language backbones and train only the expert heads (693M params total) to remain under memory caps:
   
   ```python
   # Pi0.5 parameter adjustments
   model = Pi05(
       pretrained_name_or_path="lerobot/pi05_base",
       dtype="bfloat16",
       train_expert_only=True,      # Keep this True to stay under 32 GB VRAM!
       use_relative_actions=True,
   )
   ```

2. **DDP Hangs on 4-Rank configurations**:
   Currently, an upstream Intel `torch-xpu-ops` compilation issue (specifically `#2700`, `#2701`, and `#2702`) blocks 4-rank DDP collectives on B-series / PCIe-only cards. Until upstream drivers address the XCCL broadcast-dispatch compile bug, **limit multi-XPU training to 2 discrete cards (`devices: 2`)**.

---

## 4. Run commands

Launch multi-XPU training using `torchrun`. You must specify the number of processes (matching `--devices` in the script) and run it with the oneCCL environment setup:

```bash
# 1. Activate your UV environment
source library/.venv/bin/activate

# 2. Source the optimized oneCCL environment variables
export CCL_PROCESS_LAUNCHER=torchrun
export CCL_TOPO_P2P_ACCESS=0
export FI_PROVIDER=tcp
export CCL_ZE_IPC_EXCHANGE=pidfd
export CCL_ATL_TRANSPORT=ofi
export CCL_ZE_SHARED_DEV_POOL=1

# 3. Launch the fine-tuning script on 2 GPUs
torchrun --nproc_per_node=2 tmp_scripts/pi05_so101_train.py \
    --repo-id "eugene123tw/lerobot-so101-pick-and-place-potato" \
    --device xpu \
    --max-epochs 30 \
    --dataset-name so101-potato
```
