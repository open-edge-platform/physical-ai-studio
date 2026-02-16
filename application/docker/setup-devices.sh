#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# setup-devices.sh — Detect host device-group GIDs for Docker
#
# Docker's group_add resolves group names against the container's
# /etc/group, not the host's. When the host uses different GIDs
# (e.g. Arch Linux vs Debian), the container process won't have
# the right supplementary groups to access /dev/video* or /dev/ttyACM*.
#
# This script detects the correct numeric GIDs on the host and writes
# them to the .env file used by docker-compose.yaml. It also checks
# that the current user is a member of each group.
#
# Usage:
#   ./application/docker/setup-devices.sh          # auto-detect & write
#   ./application/docker/setup-devices.sh --check   # check only, no writes
# ---------------------------------------------------------------------------
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${SCRIPT_DIR}/.env"

# Colors (disabled if stdout is not a terminal)
if [ -t 1 ]; then
    RED='\033[0;31m'
    GREEN='\033[0;32m'
    YELLOW='\033[0;33m'
    BOLD='\033[1m'
    RESET='\033[0m'
else
    RED='' GREEN='' YELLOW='' BOLD='' RESET=''
fi

info()  { printf "${GREEN}[ok]${RESET}    %s\n" "$*"; }
warn()  { printf "${YELLOW}[warn]${RESET}  %s\n" "$*"; }
error() { printf "${RED}[error]${RESET} %s\n" "$*"; }

CHECK_ONLY=false
if [[ "${1:-}" == "--check" ]]; then
    CHECK_ONLY=true
fi

# ---------------------------------------------------------------------------
# Resolve a device group GID from one or more candidate group names.
# Different distros use different names for the same purpose:
#   - Serial ports: "dialout" (Debian/Ubuntu/Fedora) vs "uucp" (Arch)
#   - Video:        "video"   (universal)
#   - Plugdev:      "plugdev" (Debian/Ubuntu) — may not exist elsewhere
# ---------------------------------------------------------------------------
resolve_gid() {
    local var_name="$1"
    shift
    local candidates=("$@")

    for name in "${candidates[@]}"; do
        local entry
        entry=$(getent group "$name" 2>/dev/null || true)
        if [[ -n "$entry" ]]; then
            local gid
            gid=$(echo "$entry" | cut -d: -f3)
            local members
            members=$(echo "$entry" | cut -d: -f4)

            # Check if the current user is a member (either via primary
            # group or supplementary groups listed in /etc/group).
            local current_user
            current_user=$(id -un)
            local user_primary_gid
            user_primary_gid=$(id -g)

            if [[ "$user_primary_gid" == "$gid" ]] || echo ",$members," | grep -q ",$current_user,"; then
                info "$var_name=$gid (host group: $name) — user '$current_user' is a member"
            else
                warn "$var_name=$gid (host group: $name) — user '$current_user' is NOT a member"
                warn "  -> Run: sudo usermod -aG $name $current_user  (then re-login)"
            fi

            # Export for later use
            eval "$var_name=$gid"
            eval "${var_name}_NAME=$name"
            return 0
        fi
    done

    warn "$var_name: none of the candidate groups (${candidates[*]}) exist on this host"
    eval "$var_name="
    eval "${var_name}_NAME="
    return 1
}

# ---------------------------------------------------------------------------
# Detect GIDs
# ---------------------------------------------------------------------------
printf "${BOLD}Detecting host device-group GIDs...${RESET}\n\n"

DIALOUT_GID=""
VIDEO_GID=""
PLUGDEV_GID=""
DIALOUT_GID_NAME=""
VIDEO_GID_NAME=""
PLUGDEV_GID_NAME=""
has_errors=false

resolve_gid DIALOUT_GID dialout uucp   || has_errors=true
resolve_gid VIDEO_GID   video          || has_errors=true
resolve_gid PLUGDEV_GID plugdev        || has_errors=true

echo ""

if [[ "$has_errors" == true ]]; then
    warn "Some groups were not found. The container may not be able to access all devices."
    echo ""
fi

# ---------------------------------------------------------------------------
# Summarise what was detected
# ---------------------------------------------------------------------------
printf "${BOLD}Detected configuration:${RESET}\n"
[[ -n "$VIDEO_GID" ]]   && echo "  VIDEO_GID=$VIDEO_GID        # $VIDEO_GID_NAME"
[[ -n "$DIALOUT_GID" ]] && echo "  DIALOUT_GID=$DIALOUT_GID      # $DIALOUT_GID_NAME"
[[ -n "$PLUGDEV_GID" ]] && echo "  PLUGDEV_GID=$PLUGDEV_GID       # $PLUGDEV_GID_NAME"
echo ""

# Also detect APP_UID/APP_GID for convenience
APP_UID=$(id -u)
APP_GID=$(id -g)
printf "${BOLD}User/group IDs:${RESET}\n"
echo "  APP_UID=$APP_UID"
echo "  APP_GID=$APP_GID"
echo ""

# ---------------------------------------------------------------------------
# Compare with Debian defaults and advise
# ---------------------------------------------------------------------------
DEBIAN_VIDEO=44
DEBIAN_DIALOUT=20
DEBIAN_PLUGDEV=46

needs_update=false
if [[ -n "$VIDEO_GID"   && "$VIDEO_GID"   != "$DEBIAN_VIDEO" ]]   ||
   [[ -n "$DIALOUT_GID" && "$DIALOUT_GID" != "$DEBIAN_DIALOUT" ]] ||
   [[ -n "$PLUGDEV_GID" && "$PLUGDEV_GID" != "$DEBIAN_PLUGDEV" ]]; then
    warn "Your host GIDs differ from the Debian defaults used by docker-compose.yaml."
    warn "You need to set these in your .env file for device access to work."
    needs_update=true
else
    info "Your host GIDs match the Debian defaults — no .env changes needed for device access."
fi

echo ""

# ---------------------------------------------------------------------------
# Write to .env (unless --check)
# ---------------------------------------------------------------------------
if [[ "$CHECK_ONLY" == true ]]; then
    if [[ "$needs_update" == true ]]; then
        printf "${BOLD}Run without --check to write these values to %s${RESET}\n" "$ENV_FILE"
    fi
    exit 0
fi

# Helper: update or append a key=value in the .env file
set_env_var() {
    local key="$1" value="$2"
    if [[ -z "$value" ]]; then
        return
    fi
    if [[ -f "$ENV_FILE" ]] && grep -q "^${key}=" "$ENV_FILE" 2>/dev/null; then
        # Update existing (uncommented) line
        sed -i "s|^${key}=.*|${key}=${value}|" "$ENV_FILE"
    elif [[ -f "$ENV_FILE" ]] && grep -q "^# *${key}=" "$ENV_FILE" 2>/dev/null; then
        # Uncomment and update
        sed -i "s|^# *${key}=.*|${key}=${value}|" "$ENV_FILE"
    else
        # Append
        echo "${key}=${value}" >> "$ENV_FILE"
    fi
}

# Create .env from template if it doesn't exist
if [[ ! -f "$ENV_FILE" ]] && [[ -f "${SCRIPT_DIR}/.env.example" ]]; then
    cp "${SCRIPT_DIR}/.env.example" "$ENV_FILE"
    info "Created $ENV_FILE from .env.example"
fi

set_env_var APP_UID "$APP_UID"
set_env_var APP_GID "$APP_GID"
set_env_var VIDEO_GID "$VIDEO_GID"
set_env_var DIALOUT_GID "$DIALOUT_GID"
set_env_var PLUGDEV_GID "$PLUGDEV_GID"

info "Updated $ENV_FILE"
echo ""
printf "${BOLD}You can now run:${RESET}\n"
echo "  docker compose -f application/docker/docker-compose.yaml up"
