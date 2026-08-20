#!/usr/bin/env bash
set -euo pipefail

NUM_SHARDS="${1:-4}"
CONFIG="configs/benchmarks/libero/10.yaml"
EVAL_ID="${EVAL_ID:-$(python -c 'import uuid; print(uuid.uuid4())')}"

cd "$(dirname "$0")"

echo "Launching ${NUM_SHARDS} LIBERO-10 shards with eval ID ${EVAL_ID}"

pids=()
for SHARD_ID in $(seq 0 $((NUM_SHARDS - 1))); do
  uv run --no-sync vla-eval run \
    --config "$CONFIG" \
    --eval-id "$EVAL_ID" \
    --shard-id "$SHARD_ID" \
    --num-shards "$NUM_SHARDS" &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    failed=$((failed + 1))
  fi
done

if ((failed > 0)); then
  echo "${failed} of ${NUM_SHARDS} shards failed" >&2
  exit 1
fi

uv run --no-sync vla-eval merge \
  --config "$CONFIG" \
  --eval-id "$EVAL_ID"
