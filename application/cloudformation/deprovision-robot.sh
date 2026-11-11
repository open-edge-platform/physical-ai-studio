#!/usr/bin/env bash
# Removes a robot provisioned by provision-robot.sh: certificates, Greengrass core device, and thing.
set -euo pipefail

usage() {
  cat <<EOF
Usage: $(basename "$0") --thing-name NAME [options]

Options:
  --thing-name NAME        IoT thing name of the robot (required)
  --stack NAME             aws-batch-trainer.yaml stack name to read StackIdentifier from,
                           in the AWS CLI configured region (default: physical-ai-studio-batch-trainer)
  --stack-identifier ID    StackIdentifier output (<account-id>-<region>-<stack-uuid>);
                           skips the CloudFormation lookup
  --yes                    Do not ask for confirmation
  -h, --help               Show this help
EOF
}

THING_NAME=""
STACK_NAME="physical-ai-studio-batch-trainer"
STACK_IDENTIFIER=""
ASSUME_YES=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --thing-name) THING_NAME="$2"; shift 2 ;;
    --stack) STACK_NAME="$2"; shift 2 ;;
    --stack-identifier) STACK_IDENTIFIER="$2"; shift 2 ;;
    --yes) ASSUME_YES=true; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 1 ;;
  esac
done

if [[ ! "$THING_NAME" =~ ^[a-zA-Z0-9:_-]{1,128}$ ]]; then
  usage >&2
  exit 1
fi

if [[ -z "$STACK_IDENTIFIER" ]]; then
  echo "Reading StackIdentifier from stack $STACK_NAME..."
  STACK_IDENTIFIER="$(aws cloudformation describe-stacks --output text \
    --stack-name "$STACK_NAME" --query "Stacks[0].Outputs[?OutputKey=='StackIdentifier'].OutputValue | [0]")"
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
if [[ "$ASSUME_YES" != true ]]; then
  read -r -p "Delete this robot? Type the thing name to confirm: " answer
  if [[ "$answer" != "$THING_NAME" ]]; then
    echo "Aborted." >&2
    exit 1
  fi
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

Deprovisioned $THING_NAME. Existing temporary credentials on the robot stay valid until they expire.
On the robot, remove /greengrass/v2 and the Greengrass service; delete any local bundle copy.
EOF
