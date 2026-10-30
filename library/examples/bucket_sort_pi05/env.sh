# Sourced by every launcher here. Datasets, runs and logs live under $WORK, on a big data disk.
export S=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export WORK=${WORK:-/mnt/data/$USER/bucket_sort_pi05}
export PY=${PY:-python}
export HF_HOME=${HF_HOME:-$WORK/hf}
export MENAGERIE_CACHE_DIR=${MENAGERIE_CACHE_DIR:-$WORK/menagerie_cache}
export MUJOCO_GL=egl
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
mkdir -p "$WORK/data" "$WORK/runs" "$WORK/logs"
cd "$WORK"
