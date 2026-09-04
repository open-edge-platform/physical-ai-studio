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
  -g, --gpus <spec>         GPU devices for containers, e.g. '0,1' (round-robin across shards)
  -l, --log-dir <path>      Tee each shard's output to <path>/shard-<id>.log
                            (default: results/logs)
  -y, --yes                 Skip confirmation prompts (e.g. docker pull)
  -h, --help                Show this help
EOF
}

require_value() {
  local option="$1"
  local value="${2:-}"
  if (($# < 2)) || [[ -z "$value" || "$value" == -* ]]; then
    echo "$option requires a value" >&2
    usage >&2
    exit 1
  fi
}

CONFIG=""
NUM_SHARDS=4
EVAL_ID="${EVAL_ID:-}"
OUTPUT_DIR=""
GPUS=""
LOG_DIR=""
ASSUME_YES=0
pids=()

cleanup() {
  local exit_code="$1"
  trap - INT TERM
  if ((${#pids[@]} > 0)); then
    echo "Stopping ${#pids[@]} shard process(es)..." >&2
    kill "${pids[@]}" 2>/dev/null || true
    for pid in "${pids[@]}"; do
      wait "$pid" 2>/dev/null || true
    done
  fi
  exit "$exit_code"
}

trap 'cleanup 130' INT
trap 'cleanup 143' TERM

while (($# > 0)); do
  case "$1" in
    -c|--config)
      require_value "$@"
      CONFIG="$2"
      shift 2
      ;;
    -n|--shards)
      require_value "$@"
      NUM_SHARDS="$2"
      shift 2
      ;;
    -e|--eval-id)
      require_value "$@"
      EVAL_ID="$2"
      shift 2
      ;;
    -o|--output-dir)
      require_value "$@"
      OUTPUT_DIR="$2"
      shift 2
      ;;
    -g|--gpus)
      require_value "$@"
      GPUS="$2"
      shift 2
      ;;
    -l|--log-dir)
      require_value "$@"
      LOG_DIR="$2"
      shift 2
      ;;
    -y|--yes)
      ASSUME_YES=1
      shift
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
if ((ASSUME_YES)); then
  RUN_ARGS+=(--yes)
fi
if [[ -n "$GPUS" ]]; then
  RUN_ARGS+=(--gpus "$GPUS")
fi
if [[ -z "$LOG_DIR" ]]; then
  LOG_DIR="results/logs/$EVAL_ID"
fi
mkdir -p "$LOG_DIR"

echo "Launching $NUM_SHARDS shards with eval ID $EVAL_ID"
echo "Logs: $LOG_DIR"

for SHARD_ID in $(seq 0 $((NUM_SHARDS - 1))); do
  # Process substitution (not a pipe) so $! stays the shard's own PID.
  uv run --no-sync vla-eval run \
    "${RUN_ARGS[@]}" \
    --shard-id "$SHARD_ID" \
    --num-shards "$NUM_SHARDS" \
    > >(tee "$LOG_DIR/shard-$SHARD_ID.log") 2>&1 &
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

uv run --no-sync vla-eval merge "${MERGE_ARGS[@]}" > >(tee "$LOG_DIR/merge.log") 2>&1
