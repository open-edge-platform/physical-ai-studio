#!/usr/bin/env bash
# Provisions a single robot as an AWS IoT Greengrass V2 core device.
# Requires the aws-batch-trainer.yaml stack to be deployed first.
set -euo pipefail

usage() {
  cat <<EOF
Usage: $(basename "$0") --thing-name NAME [options]

Options:
  --thing-name NAME        IoT thing name for the robot (required, e.g. robot-01)
  --thing-group NAME       Thing group to add the robot to (created if missing)
  --stack NAME             aws-batch-trainer.yaml stack name to read StackIdentifier from,
                           in the AWS CLI configured region (default: physical-ai-studio-batch-trainer)
  --stack-identifier ID    StackIdentifier output (<account-id>-<region>-<stack-uuid>);
                           skips the CloudFormation lookup
  --csr FILE               Sign this CSR instead of generating a key pair; the
                           private key stays on the robot
  --nucleus-version VER    Greengrass nucleus version installed on the robot (default: 2.15.0)
  --output-dir DIR         Local bundle directory (default: ./robot-bundles/<thing-name>)
  --ssh TARGET             Copy the bundle to TARGET (user@host) under /greengrass/v2
  -h, --help               Show this help
EOF
}

THING_NAME=""
THING_GROUP=""
STACK_NAME="physical-ai-studio-batch-trainer"
STACK_IDENTIFIER=""
CSR_FILE=""
NUCLEUS_VERSION="2.15.0"
OUTPUT_DIR=""
SSH_TARGET=""
GG_ROOT="/greengrass/v2"
ROOT_CA_URL="https://www.amazontrust.com/repository/AmazonRootCA1.pem"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --thing-name) THING_NAME="$2"; shift 2 ;;
    --thing-group) THING_GROUP="$2"; shift 2 ;;
    --stack) STACK_NAME="$2"; shift 2 ;;
    --stack-identifier) STACK_IDENTIFIER="$2"; shift 2 ;;
    --csr) CSR_FILE="$2"; shift 2 ;;
    --nucleus-version) NUCLEUS_VERSION="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --ssh) SSH_TARGET="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 1 ;;
  esac
done

if [[ -z "$THING_NAME" ]]; then
  usage >&2
  exit 1
fi
if [[ ! "$THING_NAME" =~ ^[a-zA-Z0-9:_-]{1,128}$ ]]; then
  echo "Invalid thing name: $THING_NAME" >&2
  exit 1
fi
if [[ -n "$CSR_FILE" && ! -f "$CSR_FILE" ]]; then
  echo "CSR file not found: $CSR_FILE" >&2
  exit 1
fi

for cmd in aws curl; do
  command -v "$cmd" >/dev/null || { echo "Missing required command: $cmd" >&2; exit 1; }
done

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
if [[ -n "$CSR_FILE" ]]; then
  read -r CERT_ARN CERT_ID < <("${AWS[@]}" iot create-certificate-from-csr \
    --certificate-signing-request "file://$CSR_FILE" --set-as-active \
    --query "[certificateArn, certificateId]")
  "${AWS[@]}" iot describe-certificate --certificate-id "$CERT_ID" \
    --query certificateDescription.certificatePem > "$OUTPUT_DIR/device.pem.crt"
else
  read -r CERT_ARN CERT_ID < <("${AWS[@]}" iot create-keys-and-certificate --set-as-active \
    --certificate-pem-outfile "$OUTPUT_DIR/device.pem.crt" \
    --private-key-outfile "$OUTPUT_DIR/private.pem.key" \
    --query "[certificateArn, certificateId]")
fi
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
    version: "$NUCLEUS_VERSION"
    configuration:
      awsRegion: "$REGION"
      iotRoleAlias: "$ROLE_ALIAS"
      iotDataEndpoint: "$DATA_ENDPOINT"
      iotCredEndpoint: "$CRED_ENDPOINT"
EOF

BUNDLE_FILES=(device.pem.crt AmazonRootCA1.pem config.yaml)
if [[ -z "$CSR_FILE" ]]; then
  BUNDLE_FILES+=(private.pem.key)
fi

if [[ -n "$SSH_TARGET" ]]; then
  REMOTE_TMP="$(ssh "$SSH_TARGET" 'mktemp -d')"
  scp "${BUNDLE_FILES[@]/#/$OUTPUT_DIR/}" "$SSH_TARGET:$REMOTE_TMP/"
  # shellcheck disable=SC2029
  ssh -t "$SSH_TARGET" "set -e
    sudo install -d -m 755 '$GG_ROOT'
    for f in device.pem.crt AmazonRootCA1.pem config.yaml; do
      sudo install -m 644 '$REMOTE_TMP'/\$f '$GG_ROOT'/\$f
    done
    if [ -f '$REMOTE_TMP/private.pem.key' ]; then
      sudo install -m 600 '$REMOTE_TMP/private.pem.key' '$GG_ROOT/private.pem.key'
    fi
    rm -rf '$REMOTE_TMP'"
  echo "Delivered bundle to $SSH_TARGET:$GG_ROOT."
fi

cat <<EOF

Provisioned $THING_NAME
  Thing ARN:       $THING_ARN
  Certificate ID:  $CERT_ID
  Certificate ARN: $CERT_ARN
  Bundle:          $OUTPUT_DIR (${BUNDLE_FILES[*]})

Kill switch: aws iot update-certificate --region $REGION --certificate-id $CERT_ID --new-status INACTIVE
EOF
