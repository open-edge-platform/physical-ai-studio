#!/usr/bin/env bash
# Runs on the robot: installs the aws-iot-greengrass snap and starts Greengrass V2
# from a bundle produced by provision-robot.sh.
set -euo pipefail

SNAP_NAME="aws-iot-greengrass"
GG_ROOT="/var/snap/$SNAP_NAME/common/greengrass/v2"
BUNDLE_DIR="${1:-$(cd "$(dirname "$0")" && pwd)}"

if [[ $EUID -ne 0 ]]; then
  echo "Run as root: sudo $0 [bundle-dir]" >&2
  exit 1
fi
for f in device.pem.crt private.pem.key AmazonRootCA1.pem config.yaml; do
  [[ -f "$BUNDLE_DIR/$f" ]] || { echo "Missing $BUNDLE_DIR/$f" >&2; exit 1; }
done
if ! grep -q "rootpath: \"$GG_ROOT\"" "$BUNDLE_DIR/config.yaml"; then
  echo "config.yaml is not for the snap; provision the robot again." >&2
  exit 1
fi

if ! snap list "$SNAP_NAME" >/dev/null 2>&1; then
  snap install "$SNAP_NAME"
fi
SNAP_VERSION="$(snap list "$SNAP_NAME" | awk 'NR==2 {print $2}')"
if [[ ! "$SNAP_VERSION" =~ ^2\.[0-9.]+$ || ! -f "/snap/$SNAP_NAME/current/opt/greengrass/greengrass-nucleus.zip" ]]; then
  echo "Snap $SNAP_NAME $SNAP_VERSION does not provide Greengrass V2." >&2
  exit 1
fi
if [[ -e "$GG_ROOT/alts/current" ]]; then
  echo "Greengrass is already installed in $GG_ROOT; run 'snap remove --purge $SNAP_NAME' to start over." >&2
  exit 1
fi

# Interfaces whose providers are absent (e.g. docker) stay disconnected.
snap connections "$SNAP_NAME" | awk -v s="$SNAP_NAME:" 'NR>1 && index($2, s)==1 && $3=="-" {print $2}' |
  while read -r plug; do
    snap connect "$plug" 2>/dev/null && echo "Connected $plug." || echo "Could not connect $plug; skipped."
  done

install -d -m 755 "$GG_ROOT"
install -m 644 "$BUNDLE_DIR/device.pem.crt" "$BUNDLE_DIR/AmazonRootCA1.pem" "$GG_ROOT/"
install -m 600 "$BUNDLE_DIR/private.pem.key" "$GG_ROOT/"
sed -E "s/^(    version: ).*/\1\"$SNAP_VERSION\"/" "$BUNDLE_DIR/config.yaml" > "$GG_ROOT/config.yaml"
chmod 644 "$GG_ROOT/config.yaml"
echo "Installed bundle into $GG_ROOT (nucleus version $SNAP_VERSION)."

# Run inside the snap's confinement so its Java, Python, and paths are used.
snap run --shell "$SNAP_NAME.configure" -c '
  set -e
  cd "$SNAP_COMMON/greengrass/v2"
  PY="$SNAP/bin/python3"; [ -x "$PY" ] || PY=python3
  "$PY" -m zipfile -e "$SNAP/opt/greengrass/greengrass-nucleus.zip" .
  "$JAVA_HOME/bin/java" -Droot="$PWD" -Dlog.store=FILE \
    -Djavax.net.ssl.trustStore="$SNAP/etc/ssl/certs/java/cacerts" -Djavax.net.ssl.trustStoreType=JKS \
    -jar lib/Greengrass.jar --init-config config.yaml \
    --component-default-user root:root --setup-system-service false --start false
'
echo "Installed Greengrass nucleus $SNAP_VERSION."

snap restart "$SNAP_NAME.greengrass-daemon"
cat <<EOF

Greengrass is starting (the service waits about 30 seconds before launching the nucleus).
  Service logs:    sudo snap logs -f $SNAP_NAME.greengrass-daemon
  Greengrass logs: sudo tail -f $GG_ROOT/logs/greengrass.log
Delete the bundle copy at $BUNDLE_DIR once Greengrass is running.
EOF
