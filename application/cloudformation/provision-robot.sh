#!/usr/bin/env bash
# Provisions a single robot as an AWS IoT Greengrass V2 core device running the aws-iot-greengrass snap.
# Requires the aws-batch-trainer.yaml stack to be deployed first.
set -euo pipefail

usage() {
  cat <<EOF
Usage: $(basename "$0") --thing-name NAME --stack-identifier ID [options]

Options:
  --thing-name NAME        IoT thing name for the robot (required, e.g. robot-01)
  --stack-identifier ID    StackIdentifier output of aws-batch-trainer.yaml
                           (<account-id>-<region>-<stack-uuid>; required)
  --thing-group NAME       Thing group to add the robot to (created if missing)
  --output-dir DIR         Local bundle directory (default: ./robot-bundles/<thing-name>)
  --ssh TARGET             Deliver the bundle to TARGET (user@host) and install Greengrass there
  -h, --help               Show this help
EOF
}

THING_NAME=""
THING_GROUP=""
STACK_IDENTIFIER=""
OUTPUT_DIR=""
SSH_TARGET=""
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
GG_ROOT="/var/snap/aws-iot-greengrass/common/greengrass/v2"
ROOT_CA_URL="https://www.amazontrust.com/repository/AmazonRootCA1.pem"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --thing-name) THING_NAME="$2"; shift 2 ;;
    --stack-identifier) STACK_IDENTIFIER="$2"; shift 2 ;;
    --thing-group) THING_GROUP="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --ssh) SSH_TARGET="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 1 ;;
  esac
done

if [[ -z "$THING_NAME" || -z "$STACK_IDENTIFIER" ]]; then
  usage >&2
  exit 1
fi
if [[ ! "$THING_NAME" =~ ^[a-zA-Z0-9:_-]{1,128}$ ]]; then
  echo "Invalid thing name: $THING_NAME" >&2
  exit 1
fi
[[ -f "$SCRIPT_DIR/install-greengrass-snap.sh" ]] || { echo "Missing $SCRIPT_DIR/install-greengrass-snap.sh" >&2; exit 1; }

for cmd in aws curl; do
  command -v "$cmd" >/dev/null || { echo "Missing required command: $cmd" >&2; exit 1; }
done

if [[ ! "$STACK_IDENTIFIER" =~ ^([0-9]{12})-([a-z]{2}(-[a-z]+)+-[0-9]+)-([0-9a-f]{8})$ ]]; then
  echo "Invalid StackIdentifier: $STACK_IDENTIFIER" >&2
  exit 1
fi
ACCOUNT_ID="${BASH_REMATCH[1]}"
REGION="${BASH_REMATCH[2]}"
STACK_UUID="${BASH_REMATCH[4]}"
AWS=(aws --region "$REGION" --output text)

if [[ "$("${AWS[@]}" sts get-caller-identity --query Account)" != "$ACCOUNT_ID" ]]; then
  echo "Current AWS credentials are not for account $ACCOUNT_ID." >&2
  exit 1
fi

OUTPUT_DIR="${OUTPUT_DIR:-./robot-bundles/$THING_NAME}"
if [[ -e "$OUTPUT_DIR/device.pem.crt" ]]; then
  echo "Bundle already exists in $OUTPUT_DIR; remove it or choose another --output-dir." >&2
  exit 1
fi
umask 077
mkdir -p "$OUTPUT_DIR"

# Names follow the aws-batch-trainer.yaml convention.
ROLE_ALIAS="pai-studio-$STACK_UUID"
IOT_POLICY="pai-robot-$STACK_UUID"
"${AWS[@]}" iot describe-role-alias --role-alias "$ROLE_ALIAS" >/dev/null
"${AWS[@]}" iot get-policy --policy-name "$IOT_POLICY" >/dev/null
DATA_ENDPOINT="$("${AWS[@]}" iot describe-endpoint --endpoint-type iot:Data-ATS --query endpointAddress)"
CRED_ENDPOINT="$("${AWS[@]}" iot describe-endpoint --endpoint-type iot:CredentialProvider --query endpointAddress)"
echo "Using stack $STACK_IDENTIFIER (role alias $ROLE_ALIAS, IoT policy $IOT_POLICY)."

# 1. Robot identity
if THING_ARN="$("${AWS[@]}" iot describe-thing --thing-name "$THING_NAME" --query thingArn 2>/dev/null)"; then
  echo "Thing $THING_NAME already exists."
else
  THING_ARN="$("${AWS[@]}" iot create-thing --thing-name "$THING_NAME" --query thingArn)"
  echo "Created thing $THING_NAME."
fi

if [[ -n "$THING_GROUP" ]]; then
  if ! "${AWS[@]}" iot describe-thing-group --thing-group-name "$THING_GROUP" >/dev/null 2>&1; then
    "${AWS[@]}" iot create-thing-group --thing-group-name "$THING_GROUP" >/dev/null
    echo "Created thing group $THING_GROUP."
  fi
  "${AWS[@]}" iot add-thing-to-thing-group --thing-name "$THING_NAME" --thing-group-name "$THING_GROUP"
  echo "Added $THING_NAME to $THING_GROUP."
fi

# 2. Certificate
read -r CERT_ARN CERT_ID < <("${AWS[@]}" iot create-keys-and-certificate --set-as-active \
  --certificate-pem-outfile "$OUTPUT_DIR/device.pem.crt" \
  --private-key-outfile "$OUTPUT_DIR/private.pem.key" \
  --query "[certificateArn, certificateId]")
if [[ -z "${CERT_ID:-}" ]]; then
  echo "Certificate creation failed." >&2
  exit 1
fi
echo "Created ACTIVE certificate $CERT_ID."

curl --fail --silent --show-error --proto '=https' -o "$OUTPUT_DIR/AmazonRootCA1.pem" "$ROOT_CA_URL"

# 3. Bind certificate to policy and thing
"${AWS[@]}" iot attach-policy --policy-name "$IOT_POLICY" --target "$CERT_ARN"
"${AWS[@]}" iot attach-thing-principal --thing-name "$THING_NAME" --principal "$CERT_ARN"
echo "Attached policy $IOT_POLICY and thing $THING_NAME to certificate."

# 4. Render nucleus config and deliver bundle
# install-greengrass-snap.sh replaces the version with the snap's nucleus version.
cat > "$OUTPUT_DIR/config.yaml" <<EOF
---
system:
  certificateFilePath: "$GG_ROOT/device.pem.crt"
  privateKeyPath: "$GG_ROOT/private.pem.key"
  rootCaPath: "$GG_ROOT/AmazonRootCA1.pem"
  rootpath: "$GG_ROOT"
  thingName: "$THING_NAME"
services:
  aws.greengrass.Nucleus:
    componentType: "NUCLEUS"
    version: "snap"
    configuration:
      awsRegion: "$REGION"
      iotRoleAlias: "$ROLE_ALIAS"
      iotDataEndpoint: "$DATA_ENDPOINT"
      iotCredEndpoint: "$CRED_ENDPOINT"
EOF
install -m 755 "$SCRIPT_DIR/install-greengrass-snap.sh" "$OUTPUT_DIR/"
BUNDLE_FILES=(device.pem.crt private.pem.key AmazonRootCA1.pem config.yaml install-greengrass-snap.sh)

if [[ -n "$SSH_TARGET" ]]; then
  "$SCRIPT_DIR/deliver-robot-bundle.sh" --bundle "$OUTPUT_DIR" --ssh "$SSH_TARGET"
fi

cat <<EOF

Provisioned $THING_NAME
  Thing ARN:       $THING_ARN
  Certificate ID:  $CERT_ID
  Certificate ARN: $CERT_ARN
  Bundle:          $OUTPUT_DIR (${BUNDLE_FILES[*]})

Kill switch: aws iot update-certificate --region $REGION --certificate-id $CERT_ID --new-status INACTIVE
EOF
