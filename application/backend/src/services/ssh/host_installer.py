"""Transfer and run the bundled host installer over a verified SSH connection."""

import re
from pathlib import Path

from services.ssh.transport import CommandFailure, SshTransport

_SCRIPT = Path(__file__).with_name("host-prerequisites.sh")
_INSTALL_TIMEOUT_S = 1800
_ERROR_MARKERS = {
    "SUDO_REQUIRED",
    "ACTIVE_CONTAINERS",
    "APT_UPDATE_FAILED",
    "PACKAGE_MANAGER_BROKEN",
    "DOCKER_INSTALL_FAILED",
    "NVIDIA_DRIVER_INSTALL_FAILED",
    "NVIDIA_DRIVER_UNAVAILABLE",
    "NVIDIA_TOOLKIT_INSTALL_FAILED",
    "NVIDIA_TOOLKIT_REPO_FAILED",
    "NVIDIA_RUNTIME_CONFIG_FAILED",
    "DOCKER_RESTART_FAILED",
    "DOCKER_USER_ACCESS_MISSING",
    "INTEL_INSTALL_FAILED",
    "INTEL_DOWNLOAD_FAILED",
    "INTEL_CHECKSUM_FAILED",
    "AL2023_PREREQUISITES_MISSING",
    "DOCKER_MISSING",
    "DOCKER_UNAVAILABLE",
    "NVIDIA_CONTAINER_RUNTIME_UNAVAILABLE",
    "INTEL_COMPUTE_RUNTIME_UNAVAILABLE",
    "INTEL_RENDER_DEVICE_UNAVAILABLE",
    "INTEL_KERNEL_UNAVAILABLE",
    "GPU_AMBIGUOUS",
    "UNSUPPORTED_GPU",
    "UNSUPPORTED_OS",
}


async def install(transport: SshTransport) -> str:
    """Install host dependencies, returning ``ready``, ``reboot_required`` or a safe failure marker."""
    temporary = await transport.run_command(["mktemp", "-d", "/tmp/physicalai-installer.XXXXXXXX"])  # noqa: S108 - mktemp creates the private directory.
    directory = temporary.first_line()
    if not temporary.ok or not re.fullmatch(
        r"/tmp/physicalai-installer\.[A-Za-z0-9]{8}",  # noqa: S108 - private remote mktemp path.
        directory,
    ):
        return "transfer_failed"
    script = f"{directory}/host-prerequisites.sh"
    log = f"{directory}/install.log"
    try:
        await transport.upload_file(_SCRIPT, script)
        # Package managers emit unbounded logs: keep them on the host and return only the tail.
        command = 'bash "$1" --install >"$2" 2>&1; status=$?; tail -n 20 "$2"; exit "$status"'
        result = await transport.run_command(["bash", "-c", command, "_", script, log], timeout=_INSTALL_TIMEOUT_S)
        if result.ok:
            return "ready"
        if result.exit_status == 10:
            return "reboot_required"
        if result.exit_status == 11:
            return "relogin_required"
        for line in reversed(result.stdout.splitlines()):
            marker = line.split(":", 1)[0]
            if marker in _ERROR_MARKERS:
                return marker.lower()
        return "installation_timeout" if result.failure is CommandFailure.TIMEOUT else "installation_failed"
    finally:
        await transport.run_command(["rm", "-f", script, log])
        await transport.run_command(["rmdir", directory])
