#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<EOF
Usage: $(basename "$0") --config <path> [options]

Run a vla-eval benchmark across parallel episode shards, then merge results.

Options:
  -c, --config <path>       Benchmark config YAML (required)
  -n, --shards <count>      Number of shards (default: 4)
  -e, --eval-id <id>        Shared evaluation ID (default: generated UUID)
  -o, --output-dir <path>   Override the config output directory
  -h, --help                Show this help
EOF
}

CONFIG=""
NUM_SHARDS=4
EVAL_ID="${EVAL_ID:-}"
OUTPUT_DIR=""

while (($# > 0)); do
  case "$1" in
    -c|--config)
      CONFIG="${2:-}"
      shift 2
      ;;
    -n|--shards)
      NUM_SHARDS="${2:-}"
      shift 2
      ;;
    -e|--eval-id)
      EVAL_ID="${2:-}"
      shift 2
      ;;
    -o|--output-dir)
      OUTPUT_DIR="${2:-}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ -z "$CONFIG" ]]; then
  echo "--config is required" >&2
  usage >&2
  exit 1
fi
if [[ ! -f "$CONFIG" ]]; then
  echo "Config not found: $CONFIG" >&2
  exit 1
fi
if [[ ! "$NUM_SHARDS" =~ ^[1-9][0-9]*$ ]]; then
  echo "--shards must be a positive integer" >&2
  exit 1
fi
if [[ -z "$EVAL_ID" ]]; then
  EVAL_ID="$(python -c 'import uuid; print(uuid.uuid4())')"
fi

RUN_ARGS=(--config "$CONFIG" --eval-id "$EVAL_ID")
MERGE_ARGS=(--config "$CONFIG" --eval-id "$EVAL_ID")
if [[ -n "$OUTPUT_DIR" ]]; then
  RUN_ARGS+=(--output-dir "$OUTPUT_DIR")
  MERGE_ARGS+=(--output-dir "$OUTPUT_DIR")
fi

echo "Launching $NUM_SHARDS shards with eval ID $EVAL_ID"

pids=()
for SHARD_ID in $(seq 0 $((NUM_SHARDS - 1))); do
  uv run --no-sync vla-eval run \
    "${RUN_ARGS[@]}" \
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
  echo "$failed of $NUM_SHARDS shards failed" >&2
  exit 1
fi

uv run --no-sync vla-eval merge "${MERGE_ARGS[@]}"
