#!/usr/bin/env bash
# Runs on the administrator machine: copies a bundle produced by provision-robot.sh
# to a robot over SSH and runs install-greengrass-snap.sh there.
set -euo pipefail

usage() {
  cat <<EOF
Usage: $(basename "$0") --bundle DIR --ssh TARGET

Options:
  --bundle DIR   Bundle directory from provision-robot.sh
  --ssh TARGET   Robot SSH target (user@host); the user needs sudo
  -h, --help     Show this help
EOF
}

BUNDLE_DIR=""
SSH_TARGET=""
INSTALLER="$(cd "$(dirname "$0")" && pwd)/install-greengrass-snap.sh"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --bundle) BUNDLE_DIR="$2"; shift 2 ;;
    --ssh) SSH_TARGET="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 1 ;;
  esac
done

if [[ -z "$BUNDLE_DIR" || -z "$SSH_TARGET" ]]; then
  usage >&2
  exit 1
fi
for f in device.pem.crt private.pem.key AmazonRootCA1.pem config.yaml; do
  [[ -f "$BUNDLE_DIR/$f" ]] || { echo "Missing $BUNDLE_DIR/$f" >&2; exit 1; }
done
[[ -f "$INSTALLER" ]] || { echo "Missing $INSTALLER" >&2; exit 1; }

FILES=("$BUNDLE_DIR/device.pem.crt" "$BUNDLE_DIR/private.pem.key" "$BUNDLE_DIR/AmazonRootCA1.pem" "$BUNDLE_DIR/config.yaml" "$INSTALLER")

REMOTE_TMP="$(ssh "$SSH_TARGET" 'mktemp -d')"
if [[ ! "$REMOTE_TMP" =~ ^/[A-Za-z0-9._/-]+$ ]]; then
  echo "Unexpected remote temporary directory: $REMOTE_TMP" >&2
  exit 1
fi
cleanup() { ssh "$SSH_TARGET" "rm -rf '$REMOTE_TMP'" || echo "Remove $SSH_TARGET:$REMOTE_TMP manually." >&2; }
trap cleanup EXIT

echo "Copying bundle to $SSH_TARGET:$REMOTE_TMP..."
scp -q "${FILES[@]}" "$SSH_TARGET:$REMOTE_TMP/"
# shellcheck disable=SC2029
ssh -t "$SSH_TARGET" "sudo bash '$REMOTE_TMP/install-greengrass-snap.sh' '$REMOTE_TMP'"
echo "Greengrass snap installed on $SSH_TARGET; the bundle copy on the robot was removed."
