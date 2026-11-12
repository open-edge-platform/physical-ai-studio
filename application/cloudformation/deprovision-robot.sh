#!/usr/bin/env bash
# Removes a robot provisioned by provision-robot.sh: the Greengrass snap on the robot (with --ssh),
# certificates, Greengrass core device, and thing.
set -euo pipefail

usage() {
  cat <<EOF
Usage: $(basename "$0") --thing-name NAME --stack-identifier ID [options]

Options:
  --thing-name NAME        IoT thing name of the robot (required)
  --stack-identifier ID    StackIdentifier output of aws-batch-trainer.yaml
                           (<account-id>-<region>-<stack-uuid>; required)
  --ssh TARGET             Robot SSH target (user@host); removes the aws-iot-greengrass snap there
  --yes                    Do not ask for confirmation
  -h, --help               Show this help
EOF
}

THING_NAME=""
STACK_IDENTIFIER=""
SSH_TARGET=""
ASSUME_YES=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --thing-name) THING_NAME="$2"; shift 2 ;;
    --stack-identifier) STACK_IDENTIFIER="$2"; shift 2 ;;
    --ssh) SSH_TARGET="$2"; shift 2 ;;
    --yes) ASSUME_YES=true; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 1 ;;
  esac
done

if [[ ! "$THING_NAME" =~ ^[a-zA-Z0-9:_-]{1,128}$ || -z "$STACK_IDENTIFIER" ]]; then
  usage >&2
  exit 1
fi

if [[ ! "$STACK_IDENTIFIER" =~ ^([0-9]{12})-([a-z]{2}(-[a-z]+)+-[0-9]+)-([0-9a-f]{8})$ ]]; then
  echo "Invalid StackIdentifier: $STACK_IDENTIFIER" >&2
  exit 1
fi
ACCOUNT_ID="${BASH_REMATCH[1]}"
REGION="${BASH_REMATCH[2]}"
AWS=(aws --region "$REGION" --output text)

if [[ "$("${AWS[@]}" sts get-caller-identity --query Account)" != "$ACCOUNT_ID" ]]; then
  echo "Current AWS credentials are not for account $ACCOUNT_ID." >&2
  exit 1
fi

if ! "${AWS[@]}" iot describe-thing --thing-name "$THING_NAME" >/dev/null 2>&1; then
  echo "Thing $THING_NAME not found in $REGION." >&2
  exit 1
fi

read -r -a CERT_ARNS <<< "$("${AWS[@]}" iot list-thing-principals --thing-name "$THING_NAME" --query 'principals[]')"

echo "Robot $THING_NAME in account $ACCOUNT_ID ($REGION):"
echo "  Certificates to deactivate and delete: ${#CERT_ARNS[@]}"
if [[ -n "$SSH_TARGET" ]]; then
  echo "  Greengrass snap to remove from: $SSH_TARGET"
fi
if [[ "$ASSUME_YES" != true ]]; then
  read -r -p "Delete this robot? Type the thing name to confirm: " answer
  if [[ "$answer" != "$THING_NAME" ]]; then
    echo "Aborted." >&2
    exit 1
  fi
fi

if [[ -n "$SSH_TARGET" ]]; then
  # Refuses to touch a robot whose Greengrass config belongs to another thing.
  # shellcheck disable=SC2029
  ssh -t "$SSH_TARGET" "set -e
    cfg=/var/snap/aws-iot-greengrass/common/greengrass/v2/config.yaml
    if sudo test -f \$cfg && ! sudo grep -q 'thingName: \"$THING_NAME\"' \$cfg; then
      echo 'Greengrass on this robot is configured for a different thing; nothing removed.' >&2
      exit 1
    fi
    if snap list aws-iot-greengrass >/dev/null 2>&1; then
      sudo snap remove --purge aws-iot-greengrass
    else
      echo 'aws-iot-greengrass snap is not installed.'
    fi
    sudo rmdir /etc/ssl/certs/java 2>/dev/null || true"
  echo "Removed Greengrass from $SSH_TARGET."
fi

for cert_arn in "${CERT_ARNS[@]}"; do
  cert_id="${cert_arn##*/}"
  "${AWS[@]}" iot detach-thing-principal --thing-name "$THING_NAME" --principal "$cert_arn"
  if [[ -n "$("${AWS[@]}" iot list-principal-things --principal "$cert_arn" --query 'things[]')" ]]; then
    echo "Certificate $cert_id is still used by other things; detached from $THING_NAME only."
    continue
  fi
  "${AWS[@]}" iot update-certificate --certificate-id "$cert_id" --new-status INACTIVE
  for policy in $("${AWS[@]}" iot list-attached-policies --target "$cert_arn" --query 'policies[].policyName'); do
    "${AWS[@]}" iot detach-policy --policy-name "$policy" --target "$cert_arn"
  done
  "${AWS[@]}" iot delete-certificate --certificate-id "$cert_id"
  echo "Deleted certificate $cert_id."
done

if "${AWS[@]}" greengrassv2 get-core-device --core-device-thing-name "$THING_NAME" >/dev/null 2>&1; then
  "${AWS[@]}" greengrassv2 delete-core-device --core-device-thing-name "$THING_NAME"
  echo "Deleted Greengrass core device $THING_NAME."
fi

# Thing principal detachment is eventually consistent.
for attempt in 1 2 3 4 5; do
  if "${AWS[@]}" iot delete-thing --thing-name "$THING_NAME" 2>/dev/null; then
    echo "Deleted thing $THING_NAME."
    break
  fi
  if [[ $attempt -eq 5 ]]; then
    echo "Could not delete thing $THING_NAME; retry later." >&2
    exit 1
  fi
  sleep 3
done

cat <<EOF

Deprovisioned $THING_NAME. Existing temporary credentials stay valid until they expire.
EOF
if [[ -z "$SSH_TARGET" ]]; then
  echo "On the robot, run 'sudo snap remove --purge aws-iot-greengrass'."
fi
echo "Delete any local bundle copy of $THING_NAME."
