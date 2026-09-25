#!/usr/bin/env bash
# Shared host prerequisite check and opt-in installer for managed trainers.
set -euo pipefail

if [[ $# -ne 1 || ( $1 != --check && $1 != --install ) ]]; then
  echo 'Usage: host-prerequisites.sh --check|--install' >&2
  exit 2
fi
mode=$1

# shellcheck source=/dev/null
. /etc/os-release

installed() {
  [[ $(dpkg-query -W -f='${Status}' "$1" 2>/dev/null) == *' ok installed' ]]
}

update_apt() {
  local source=${1:-}
  if [[ -z $source && -f /etc/apt/sources.list.d/ubuntu.sources ]]; then
    source=/etc/apt/sources.list.d/ubuntu.sources
  elif [[ -z $source && -f /etc/apt/sources.list ]]; then
    source=/etc/apt/sources.list
  fi
  # An unrelated third-party source must not block Ubuntu security packages.
  if [[ -n $source ]]; then
    "${privileged[@]}" apt-get update -qq -o "Dir::Etc::sourcelist=$source" -o Dir::Etc::sourceparts=-
  else
    "${privileged[@]}" apt-get update -qq
  fi
}
case "$ID:$VERSION_ID" in
  ubuntu:24.04|amzn:2023) ;;
  *) echo "UNSUPPORTED_OS: $ID $VERSION_ID" >&2; exit 2 ;;
esac

nvidia=0
intel=0
for device in /sys/bus/pci/devices/*; do
  read -r class < "$device/class"
  [[ $class == 0x03* ]] || continue
  read -r vendor < "$device/vendor"
  case "$vendor" in
    0x10de) nvidia=1 ;;
    0x8086) intel=1 ;;
  esac
done

if (( nvidia + intel != 1 )); then
  echo 'GPU_AMBIGUOUS: expected exactly one NVIDIA or Intel display controller' >&2
  exit 2
fi
if (( intel )) && [[ $ID == amzn ]]; then
  echo 'UNSUPPORTED_GPU: Intel on Amazon Linux 2023' >&2
  exit 2
fi
# ponytail: leave kernel changes to the host admin; add an HWE upgrade path after clean-host testing.
if (( intel )); then
  render_device_found=0
  for render in /dev/dri/renderD*; do
    if [[ -e $render ]]; then render_device_found=1; fi
  done
  if (( !render_device_found )); then
    echo 'INTEL_KERNEL_UNAVAILABLE: no GPU render device; verify the Ubuntu HWE kernel and GPU firmware' >&2
    exit 1
  fi
fi
# A ready host is a true no-op, including when the user cannot use sudo.
if [[ $mode == --install ]] && bash "$0" --check >/dev/null 2>&1; then
  bash "$0" --check
  exit 0
fi

if [[ $mode == --install ]]; then
  if (( EUID == 0 )); then
    privileged=()
  else
    privileged=(sudo -n)
    if ! sudo -n true 2>/dev/null; then
      echo 'SUDO_REQUIRED: passwordless sudo is required for installation' >&2
      exit 1
    fi
  fi
  if [[ $ID == ubuntu ]]; then
    audit=$("${privileged[@]}" dpkg --audit 2>/dev/null) || {
      echo 'PACKAGE_MANAGER_BROKEN: could not check dpkg state' >&2; exit 1;
    }
    if [[ -n $audit ]]; then
      echo 'PACKAGE_MANAGER_BROKEN: repair incomplete dpkg transactions before installing prerequisites' >&2
      exit 1
    fi
  fi
  if [[ -n ${TRAINER_SSH_USER:-} ]] && ! id -nG "$TRAINER_SSH_USER" | grep -qw docker; then
    "${privileged[@]}" usermod -aG docker "$TRAINER_SSH_USER" || { echo 'DOCKER_USER_ACCESS_MISSING' >&2; exit 1; }
    if bash "$0" --check; then exit 0; fi
  fi
  # Check as root: an SSH user without Docker group access must not hide active workloads.
  if command -v docker >/dev/null; then
    containers=$("${privileged[@]}" docker ps -q 2>/dev/null) || {
      echo 'DOCKER_UNAVAILABLE: cannot inspect running containers before installation' >&2; exit 1;
    }
    if [[ -n $containers ]]; then
      echo 'ACTIVE_CONTAINERS: stop running containers before installing prerequisites' >&2
      exit 1
    fi
  fi
  export DEBIAN_FRONTEND=noninteractive
  if [[ $ID == ubuntu ]]; then
    update_apt || { echo 'APT_UPDATE_FAILED: Ubuntu package source could not be refreshed' >&2; exit 1; }
    if ! command -v docker >/dev/null; then
      "${privileged[@]}" apt-get install -y docker.io=29.1.3-0ubuntu3~24.04.2 || {
        echo 'DOCKER_INSTALL_FAILED: pinned package unavailable or package manager failed' >&2; exit 1;
      }
    fi
    "${privileged[@]}" systemctl enable --now docker || { echo 'DOCKER_RESTART_FAILED' >&2; exit 1; }
    if (( EUID != 0 )) && ! id -nG "$(id -un)" | grep -qw docker; then
      "${privileged[@]}" usermod -aG docker "$(id -un)" || { echo 'DOCKER_USER_ACCESS_MISSING' >&2; exit 1; }
    fi
    if (( nvidia )); then
      if ! nvidia-smi -L >/dev/null 2>&1; then
        if installed nvidia-driver-580; then
          echo 'NVIDIA_DRIVER_UNAVAILABLE: driver is installed; reboot or diagnose the host' >&2
          exit 1
        fi
        "${privileged[@]}" apt-get install -y nvidia-driver-580 || {
          echo 'NVIDIA_DRIVER_INSTALL_FAILED: Ubuntu NVIDIA driver branch 580 unavailable' >&2; exit 1;
        }
        if ! nvidia-smi -L >/dev/null 2>&1; then
          echo 'REBOOT_REQUIRED: NVIDIA driver installed; reboot before continuing' >&2
          exit 10
        fi
      fi
      installed_toolkit=0
      if ! command -v nvidia-ctk >/dev/null || ! command -v nvidia-container-runtime >/dev/null; then
        "${privileged[@]}" apt-get install -y ca-certificates curl gnupg || {
          echo 'NVIDIA_TOOLKIT_REPO_FAILED: missing repository tools' >&2; exit 1;
        }
        key_dir=$(mktemp -d)
        trap 'rm -rf "$key_dir"' EXIT
        curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey -o "$key_dir/key" || {
          echo 'NVIDIA_TOOLKIT_REPO_FAILED: could not download the signing key' >&2; exit 1;
        }
        fingerprint=$(gpg --show-keys --with-colons "$key_dir/key" 2>/dev/null | awk -F: '$1 == "fpr" { print $10; exit }') || {
          echo 'NVIDIA_TOOLKIT_REPO_FAILED: invalid signing key' >&2; exit 1;
        }
        if [[ $fingerprint != C95B321B61E88C1809C4F759DDCAE044F796ECB0 ]]; then
          echo 'NVIDIA_TOOLKIT_REPO_FAILED: unexpected signing key' >&2
          exit 1
        fi
        "${privileged[@]}" gpg --yes --dearmor --output /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg "$key_dir/key" || {
          echo 'NVIDIA_TOOLKIT_REPO_FAILED: could not install the signing key' >&2; exit 1;
        }
        rm -rf "$key_dir"
        trap - EXIT
        # shellcheck disable=SC2016  # apt substitutes $(ARCH), not the shell.
        printf 'deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://nvidia.github.io/libnvidia-container/stable/deb/$(ARCH) /\n' | \
          "${privileged[@]}" tee /etc/apt/sources.list.d/nvidia-container-toolkit.list >/dev/null || {
            echo 'NVIDIA_TOOLKIT_REPO_FAILED: could not configure apt' >&2; exit 1;
          }
        update_apt /etc/apt/sources.list.d/nvidia-container-toolkit.list || {
          echo 'NVIDIA_TOOLKIT_REPO_FAILED: apt update failed' >&2; exit 1;
        }
        "${privileged[@]}" apt-get install -y --no-install-recommends \
          nvidia-container-toolkit=1.17.8-1 nvidia-container-toolkit-base=1.17.8-1 \
          libnvidia-container-tools=1.17.8-1 libnvidia-container1=1.17.8-1 || {
          echo 'NVIDIA_TOOLKIT_INSTALL_FAILED: pinned packages unavailable or package manager failed' >&2; exit 1;
        }
        installed_toolkit=1
      fi
      if (( installed_toolkit )) || ! docker info --format '{{json .Runtimes}}' | grep -q '"nvidia"'; then
        "${privileged[@]}" nvidia-ctk runtime configure --runtime=docker || {
          echo 'NVIDIA_RUNTIME_CONFIG_FAILED' >&2; exit 1;
        }
        "${privileged[@]}" systemctl restart docker || { echo 'DOCKER_RESTART_FAILED' >&2; exit 1; }
      fi
    else
      installed_intel=0
      if ! installed intel-opencl-icd || ! installed libze-intel-gpu1; then
        "${privileged[@]}" apt-get install -y ca-certificates curl ocl-icd-libopencl1 || {
          echo 'INTEL_INSTALL_FAILED: missing system dependencies' >&2; exit 1;
        }
        intel_dir=$(mktemp -d)
        trap 'rm -rf "$intel_dir"' EXIT
        for package in intel-igc-core-2_2.32.7+21184_amd64.deb intel-igc-opencl-2_2.32.7+21184_amd64.deb; do
          curl -fsSL --max-time 180 "https://github.com/intel/intel-graphics-compiler/releases/download/v2.32.7/$package" \
            -o "$intel_dir/$package" || { echo "INTEL_DOWNLOAD_FAILED: $package" >&2; exit 1; }
        done
        for package in intel-opencl-icd_26.14.37833.4-0_amd64.deb libigdgmm12_22.9.0_amd64.deb libze-intel-gpu1_26.14.37833.4-0_amd64.deb; do
          curl -fsSL --max-time 180 "https://github.com/intel/compute-runtime/releases/download/26.14.37833.4/$package" \
            -o "$intel_dir/$package" || { echo "INTEL_DOWNLOAD_FAILED: $package" >&2; exit 1; }
        done
        printf '64e5230788e3a31e611e8d815a141b1facb91e5f0ef239233ef3f0614bfe3fd6  %s/intel-igc-core-2_2.32.7+21184_amd64.deb\n3c9bddbfe558279402bbeaabcf9c63b8de46b956b0ad9625415fd35dda53ad52  %s/intel-igc-opencl-2_2.32.7+21184_amd64.deb\n2e15eeb4fe9c1bba467a655967373eec6a20dd04cc7159de53c359f17ab53e41  %s/intel-opencl-icd_26.14.37833.4-0_amd64.deb\n9d712f71c18baee076de9961dda71e8089291e1bd0deb5d649ab5ba5de114f97  %s/libigdgmm12_22.9.0_amd64.deb\n34ce5791160d87ce6d54edb558a4030858ee1dad2afb067b9c5c58d4cde774c6  %s/libze-intel-gpu1_26.14.37833.4-0_amd64.deb\n' \
          "$intel_dir" "$intel_dir" "$intel_dir" "$intel_dir" "$intel_dir" | sha256sum --strict --check || {
            echo 'INTEL_CHECKSUM_FAILED: unexpected GPU package checksum' >&2; exit 1;
          }
        "${privileged[@]}" apt-get install -y "$intel_dir"/*.deb || {
          echo 'INTEL_INSTALL_FAILED: pinned GPU packages or dependencies unavailable' >&2; exit 1;
        }
        rm -rf "$intel_dir"
        trap - EXIT
        installed_intel=1
      fi
      if ! installed clinfo; then
        "${privileged[@]}" apt-get install -y clinfo || { echo 'INTEL_INSTALL_FAILED: clinfo unavailable' >&2; exit 1; }
        installed_intel=1
      fi
      if (( installed_intel )) && ! "${privileged[@]}" clinfo -l 2>/dev/null | grep -E 'Device #[0-9]+:.*Intel' | grep -qv CPU; then
        echo 'REBOOT_REQUIRED: Intel GPU packages installed; reboot before continuing' >&2
        exit 10
      fi
      render_access=0
      for render in /dev/dri/renderD*; do
        if [[ -r $render && -w $render ]]; then render_access=1; fi
      done
      if (( !render_access && EUID != 0 )); then
        if ! id -nG "$(id -un)" | grep -qw render; then
          "${privileged[@]}" usermod -aG render "$(id -un)" || { echo 'INTEL_RENDER_DEVICE_UNAVAILABLE' >&2; exit 1; }
        fi
        echo 'RELOGIN_REQUIRED: reconnect the SSH user to activate render group access' >&2
        exit 11
      fi
    fi
  elif ! command -v docker >/dev/null || ! nvidia-smi -L >/dev/null 2>&1; then
    echo 'AL2023_PREREQUISITES_MISSING: use the ECS GPU AMI with Docker and NVIDIA drivers' >&2
    exit 1
  else
    "${privileged[@]}" systemctl enable --now docker || { echo 'DOCKER_RESTART_FAILED' >&2; exit 1; }
  fi
  # Docker group access is root-equivalent; this only happens on explicit installation.
  if ! docker version --format '{{.Server.Version}}' >/dev/null 2>&1; then
    if (( EUID != 0 )) && ! id -nG "$(id -un)" | grep -qw docker; then
      "${privileged[@]}" usermod -aG docker "$(id -un)" || { echo 'DOCKER_USER_ACCESS_MISSING' >&2; exit 1; }
    fi
    echo 'RELOGIN_REQUIRED: reconnect the SSH user to activate Docker access' >&2
    exit 11
  fi
fi

if [[ -n ${TRAINER_SSH_USER:-} ]] && ! id -nG "$TRAINER_SSH_USER" | grep -qw docker; then
  echo 'DOCKER_USER_ACCESS_MISSING: SSH account lacks Docker access' >&2
  exit 1
fi
if ! command -v docker >/dev/null; then
  echo 'DOCKER_MISSING' >&2
  exit 1
fi
if ! docker version --format '{{.Server.Version}}' >/dev/null 2>&1; then
  echo 'DOCKER_UNAVAILABLE: start Docker or grant this SSH user Docker access' >&2
  exit 1
fi
if (( nvidia )); then
  if ! command -v nvidia-smi >/dev/null || ! nvidia-smi -L >/dev/null 2>&1; then
    echo 'NVIDIA_DRIVER_UNAVAILABLE' >&2
    exit 1
  fi
  if ! command -v nvidia-ctk >/dev/null || ! command -v nvidia-container-runtime >/dev/null || \
    ! docker info --format '{{json .Runtimes}}' | grep -q '"nvidia"'; then
    echo 'NVIDIA_CONTAINER_RUNTIME_UNAVAILABLE' >&2
    exit 1
  fi
  echo 'READY:nvidia'
else
  if ! command -v clinfo >/dev/null || ! clinfo -l 2>/dev/null | grep -E 'Device #[0-9]+:.*Intel' | grep -qv CPU; then
    echo 'INTEL_COMPUTE_RUNTIME_UNAVAILABLE' >&2
    exit 1
  fi
  render_access=0
  for render in /dev/dri/renderD*; do
    if [[ -r $render && -w $render ]]; then render_access=1; fi
  done
  if (( !render_access )); then
    echo 'INTEL_RENDER_DEVICE_UNAVAILABLE: check render group membership' >&2
    exit 1
  fi
  echo 'READY:intel'
fi
