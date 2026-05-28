# Intel oneCCL env vars required for multi-GPU xccl on PCIe-only B-series.
# Source this file before launching torchrun / accelerate on Intel XPU.
#
# Usage:
#     source intel_env_combo.sh
#     torchrun --nproc_per_node=2 ...

export CCL_PROCESS_LAUNCHER=torchrun
export CCL_TOPO_P2P_ACCESS=0
export FI_PROVIDER=tcp
export CCL_ZE_IPC_EXCHANGE=pidfd
export CCL_ATL_TRANSPORT=ofi
export CCL_ZE_SHARED_DEV_POOL=1
