"""Remote installer transfers a bundled script and returns bounded, safe results."""

import subprocess
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from services.ssh.host_installer import install
from services.ssh.transport import CommandFailure, CommandResult


def test_ubuntu_26_uses_its_own_docker_and_intel_packages() -> None:
    source = (Path(__file__).resolve().parents[3] / "src/services/ssh/host-prerequisites.sh").read_text()
    assert "ubuntu:24.04|ubuntu:26.04) ;;" in source
    assert "amzn:2023" not in source
    assert "if [[ $VERSION_ID == 24.04 ]]; then docker_package=docker.io=29.1.3-0ubuntu3~24.04.2; fi" in source
    assert "if [[ $VERSION_ID == 26.04 ]]; then\n        if ! installed intel-opencl-icd" in source
    assert "apt-get install -y intel-opencl-icd libze-intel-gpu1 libze1" in source


def test_buildx_is_installed_without_replacing_docker_and_required_for_readiness() -> None:
    source = (Path(__file__).resolve().parents[3] / "src/services/ssh/host-prerequisites.sh").read_text()
    assert "curl -fsSL --max-time 300" in source
    assert "sha256sum --strict --check" in source
    assert '"$HOME/.docker/cli-plugins/docker-buildx"' in source
    assert source.count("docker buildx version >/dev/null 2>&1") == 2
    assert source.index("BUILDX_UNAVAILABLE") < source.index("READY:nvidia")


def test_daemon_proxy_escapes_systemd_values_and_is_private() -> None:
    source = (Path(__file__).resolve().parents[3] / "src/services/ssh/host-prerequisites.sh").read_text()
    function = "proxy_environment() {" + source.split("proxy_environment() {", 1)[1].split("\n}", 1)[0] + "\n}"
    result = subprocess.run(
        ["bash", "-c", function + '\nproxy_environment HTTPS_PROXY "$1"', "_", 'http://user:p%25\\"@proxy:8080'],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout == 'Environment="HTTPS_PROXY=http://user:p%%25\\\\\\"@proxy:8080"\n'
    injected = subprocess.run(
        ["bash", "-c", function + '\nproxy_environment HTTPS_PROXY "$1"', "_", "http://proxy\nEnvironment=EVIL"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert injected.returncode != 0 and not injected.stdout
    assert 'install -D -m 600 "$proxy_dir/proxy.conf"' in source
    assert source.index("DOCKER_PROXY_UNAVAILABLE") < source.index("READY:nvidia")
    assert source.index('"${privileged[@]}" docker ps -q', source.index("physicalai-proxy.conf")) < source.index(
        '"${privileged[@]}" systemctl restart docker', source.index("physicalai-proxy.conf")
    )


def test_docker_access_distinguishes_daemon_failure_from_relogin() -> None:
    source = (Path(__file__).resolve().parents[3] / "src/services/ssh/host-prerequisites.sh").read_text()
    assert "TRAINER_SSH_USER" not in source
    access_check = source.split("# A new Docker group membership needs a fresh SSH session", 1)[1]
    assert access_check.index('"${privileged[@]}" docker version') < access_check.index("! id -nG | grep -qw docker")
    assert "DOCKER_UNAVAILABLE: Docker is not responding on the SSH host" in access_check


def test_installer_rechecks_running_containers_before_docker_restart() -> None:
    source = (Path(__file__).resolve().parents[3] / "src/services/ssh/host-prerequisites.sh").read_text()
    configure = source.index('"${privileged[@]}" nvidia-ctk runtime configure')
    restart = source.index('"${privileged[@]}" systemctl restart docker')
    assert configure < source.rindex('"${privileged[@]}" docker ps -q', 0, restart) < restart


def test_intel_reboot_is_reported_before_render_group_relogin() -> None:
    script = Path(__file__).resolve().parents[3] / "src/services/ssh/host-prerequisites.sh"
    source = script.read_text()
    assert source.index('"${privileged[@]}" clinfo -l') < source.index(
        "RELOGIN_REQUIRED: reconnect the SSH user to activate render group access"
    )
    assert source.index("systemctl enable --now docker") < source.index("docker ps -q")


async def test_install_does_not_upload_when_private_temp_directory_fails() -> None:
    transport = AsyncMock()
    transport.run_command.return_value = CommandResult(argv=("mktemp",), command="mktemp", exit_status=1)
    assert await install(transport) == "transfer_failed"
    transport.upload_file.assert_not_awaited()


@pytest.mark.parametrize(
    ("code", "output", "expected"),
    [
        (0, "READY:nvidia", "ready"),
        (10, "REBOOT_REQUIRED: driver installed", "reboot_required"),
        (11, "RELOGIN_REQUIRED: Docker group changed", "relogin_required"),
        (1, "secret remote apt output\nNVIDIA_DRIVER_INSTALL_FAILED: details", "nvidia_driver_install_failed"),
        (1, "PACKAGE_MANAGER_BROKEN: incomplete kernel packages", "package_manager_broken"),
        (1, "APT_UPDATE_FAILED: Ubuntu package source could not be refreshed", "apt_update_failed"),
        (1, "BUILDX_INSTALL_FAILED: checksum mismatch", "buildx_install_failed"),
        (1, "BUILDX_UNAVAILABLE: plugin not found", "buildx_unavailable"),
        (1, "DOCKER_PROXY_CONFIG_FAILED: daemon reload failed", "docker_proxy_config_failed"),
        (1, "DOCKER_PROXY_UNAVAILABLE: daemon has no proxy", "docker_proxy_unavailable"),
        (1, "INTEL_DOWNLOAD_FAILED: package unavailable", "intel_download_failed"),
        (1, "INTEL_CHECKSUM_FAILED: unexpected package checksum", "intel_checksum_failed"),
        (1, "secret remote apt output", "installation_failed"),
    ],
)
async def test_install_reports_only_known_outcomes_and_cleans_up(code: int, output: str, expected: str) -> None:
    transport = AsyncMock()
    transport.run_command.side_effect = [
        CommandResult(argv=("mktemp",), command="mktemp", exit_status=0, stdout="/tmp/physicalai-installer.ABC123xy\n"),
        CommandResult(argv=("bash",), command="bash", exit_status=code, stdout=output),
        CommandResult(argv=("rm",), command="rm", exit_status=0),
        CommandResult(argv=("rmdir",), command="rmdir", exit_status=0),
    ]
    assert await install(transport) == expected
    transport.upload_file.assert_awaited_once()
    assert transport.run_command.await_count == 4


@pytest.mark.parametrize(
    ("failure", "expected"),
    [
        (CommandFailure.TIMEOUT, "installation_timeout"),
        (CommandFailure.CHANNEL_REFUSED, "installation_failed"),
        (CommandFailure.SIGNALED, "installation_failed"),
    ],
)
async def test_install_distinguishes_timeout_from_other_command_failures(
    failure: CommandFailure, expected: str
) -> None:
    transport = AsyncMock()
    transport.run_command.side_effect = [
        CommandResult(argv=("mktemp",), command="mktemp", exit_status=0, stdout="/tmp/physicalai-installer.ABC123xy\n"),
        CommandResult(argv=("bash",), command="bash", exit_status=124, failure=failure),
        CommandResult(argv=("rm",), command="rm", exit_status=0),
        CommandResult(argv=("rmdir",), command="rmdir", exit_status=0),
    ]
    assert await install(transport) == expected
