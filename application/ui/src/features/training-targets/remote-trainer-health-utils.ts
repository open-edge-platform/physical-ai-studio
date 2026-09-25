import { SchemaRemoteTrainerHealth } from '../../api/openapi-spec';

export type CheckState = 'positive' | 'yellow' | 'negative' | 'neutral';

/**
 * A re-check in flight keeps showing the last-known status rather than
 * flipping to "Checking…"/neutral — only the very first check (no health
 * reported yet) reads as checking.
 */
export const healthLabel = (health?: SchemaRemoteTrainerHealth, isChecking = false) => {
    if (health === undefined) return isChecking ? 'Checking…' : 'Not checked';
    if (health.status === 'starting') return 'Starting…';
    if (health.reason_code === 'check_failed') return 'Check failed';
    return health.status === 'healthy' ? 'Healthy' : health.status === 'degraded' ? 'Degraded' : 'Unreachable';
};

export const healthVariant = (health?: SchemaRemoteTrainerHealth, _isChecking = false) => {
    if (health === undefined) return 'neutral' as const;
    if (health.status === 'starting') return 'neutral' as const;
    return health.status === 'healthy'
        ? ('positive' as const)
        : health.status === 'degraded'
          ? ('yellow' as const)
          : ('negative' as const);
};

export const healthDescription = (health?: SchemaRemoteTrainerHealth) => {
    if (health === undefined) return 'Connection status has not been checked.';
    if (health.status === 'starting') {
        return health.reason_code ?? 'The trainer container is still starting.';
    }
    if (health.status === 'healthy') return 'The trainer health endpoint and device report are available.';
    switch (health.reason_code) {
        case 'timeout':
            return 'The trainer did not respond within five seconds.';
        case 'connection_failed':
            return 'Studio could not connect to the configured trainer URL.';
        case 'http_error':
            return 'The trainer returned an error response.';
        case 'unhealthy':
            return 'The trainer health endpoint did not report a healthy status.';
        case 'check_failed':
            return 'Studio could not complete the health check. Try again.';
        case 'docker_unavailable':
            return 'Studio-managed training requires Docker to be installed and running on the SSH host.';
        case 'accelerator_unavailable':
            return 'Studio-managed training requires a working CUDA or XPU driver on the SSH host.';
        case 'container_accelerator_unavailable':
            return 'Studio started the trainer, but its container cannot access a CUDA or XPU device.';
        case 'reboot_required':
            return 'GPU prerequisites were installed. Confirm a host reboot to finish setup.';
        case 'relogin_required':
            return 'Reconnect SSH to activate Docker or GPU access, then retry installation.';
        case 'active_containers':
            return 'Host setup cannot modify Docker while containers are running.';
        case 'reboot_blocked_active_containers':
            return 'Stop running containers on the host before confirming the reboot again.';
        case 'sudo_required':
            return 'Host installation requires non-interactive sudo for the SSH user.';
        case 'nvidia_driver_install_failed':
            return 'NVIDIA driver branch 580 could not be installed. Check Ubuntu apt sources.';
        case 'nvidia_toolkit_install_failed':
            return 'NVIDIA Container Toolkit installation failed. Check the pinned package repository.';
        case 'nvidia_toolkit_repo_failed':
            return 'Could not configure the signed NVIDIA Container Toolkit repository.';
        case 'intel_install_failed':
            return 'Intel GPU dependency installation failed. Check the host package sources.';
        case 'intel_download_failed':
            return 'Could not download the pinned Intel GPU packages.';
        case 'intel_checksum_failed':
            return 'An Intel GPU package failed checksum verification; installation was stopped.';
        case 'docker_install_failed':
            return 'Docker installation failed: the package is unavailable or apt failed.';
        case 'gpu_ambiguous':
            return 'Exactly one NVIDIA or Intel GPU vendor must be present on the SSH host.';
        case 'unsupported_gpu':
            return 'Intel GPU installation is not supported on Amazon Linux 2023.';
        case 'unsupported_os':
            return 'Host installation supports Ubuntu 24.04, Ubuntu 26.04, or Amazon Linux 2023 only.';
        case 'reboot_failed':
            return 'The host did not reboot or reconnect. Check it manually before retrying.';
        case 'apt_update_failed':
            return 'Ubuntu package update failed. Run sudo apt-get update on the SSH host to diagnose it.';
        case 'package_manager_broken':
            return 'The SSH host has incomplete dpkg transactions. Repair them before installing prerequisites.';
        case 'installation_timeout':
            return 'Host installation timed out. Check the host before retrying.';
        case 'installation_failed':
            return 'Host installation failed. Check Studio backend logs for details.';
        case 'transfer_failed':
            return 'Could not transfer the installation script to the SSH host.';
        case 'ssh_install_connection_failed':
            return 'The SSH connection or script transfer failed. Check host access, then retry.';
        case 'ssh_install_auth_failed':
            return 'The SSH host rejected authentication. Check your SSH key or agent.';
        case 'ssh_install_host_key_failed':
            return 'The SSH host key could not be verified. Check known_hosts before retrying.';
        case 'nvidia_driver_unavailable':
            return 'NVIDIA driver is installed but not working; reboot or diagnose the host.';
        case 'nvidia_runtime_config_failed':
            return 'Could not configure the NVIDIA Docker runtime.';
        case 'docker_restart_failed':
            return 'Docker could not start or restart after configuring the GPU runtime.';
        case 'docker_user_access_missing':
            return 'Could not grant the SSH user access to Docker.';
        case 'al2023_prerequisites_missing':
            return 'Use an Amazon Linux 2023 ECS GPU AMI with Docker and NVIDIA drivers.';
        case 'nvidia_container_runtime_unavailable':
            return 'The NVIDIA Docker runtime is not available.';
        case 'intel_compute_runtime_unavailable':
            return 'The Intel GPU compute runtime is not available.';
        case 'intel_render_device_unavailable':
            return 'The SSH user cannot access the Intel GPU render device.';
        case 'intel_kernel_unavailable':
            return 'No Intel GPU render device is present. Check the Ubuntu HWE kernel and GPU firmware.';
        default:
            return 'The trainer returned an invalid device report.';
    }
};

export const deviceTypes = (health?: SchemaRemoteTrainerHealth) => [
    ...new Set((health?.devices ?? []).map((device) => device.type.toUpperCase())),
];

export const formatBytes = (bytes: number): string => {
    if (bytes <= 0) return '0 GB';
    const gib = bytes / 1024 ** 3;
    return gib >= 1024 ? `${(gib / 1024).toFixed(1)} TB` : `${gib.toFixed(1)} GB`;
};

export const formatStorage = (storage: SchemaRemoteTrainerHealth['storage']) =>
    storage ? `${formatBytes(storage.free_bytes)} free of ${formatBytes(storage.total_bytes)}` : undefined;

export const getCapabilityState = (health: SchemaRemoteTrainerHealth | undefined, isChecking: boolean): CheckState => {
    if (isChecking || health === undefined || health.status === 'unreachable' || health.status === 'starting')
        return 'neutral';
    return (health.devices?.length ?? 0) > 0 ? 'positive' : 'yellow';
};

export const getStorageState = (health: SchemaRemoteTrainerHealth | undefined, isChecking: boolean): CheckState => {
    if (isChecking || health === undefined || health.status === 'unreachable' || health.status === 'starting')
        return 'neutral';
    return health.storage ? 'positive' : 'yellow';
};

export const trainerHealthDetail = (
    health: SchemaRemoteTrainerHealth | undefined,
    isChecking: boolean,
    deviceReportIsInvalid: boolean
) => {
    if (isChecking) return 'connection check in progress';
    if (health?.status === 'starting') return healthDescription(health);
    if (health?.status === 'healthy' || deviceReportIsInvalid) {
        return health?.latency_ms != null
            ? `responded in ${health.latency_ms} ms and is ready for training requests`
            : 'ready for training requests';
    }
    if (health?.status === 'degraded') return healthDescription(health);
    return healthDescription(health);
};

export const capabilityDetail = (health: SchemaRemoteTrainerHealth | undefined, isChecking: boolean) => {
    const devices = health?.devices ?? [];
    if (devices.length > 0) {
        return devices.map((device) => `${device.type.toUpperCase()} · ${device.name}`).join(', ');
    }
    if (health?.status === 'starting') return 'awaiting device report';
    return health === undefined || isChecking ? 'awaiting device report' : 'no compute device reported';
};

export const storageDetail = (health: SchemaRemoteTrainerHealth | undefined, isChecking: boolean) => {
    const isAwaiting = health === undefined || isChecking || health.status === 'starting';
    return formatStorage(health?.storage) ?? (isAwaiting ? 'awaiting storage report' : 'no storage reported');
};

export const getDisplayHealth = (
    remoteTrainerId: string,
    health: SchemaRemoteTrainerHealth | undefined,
    hasError: boolean
) =>
    health ??
    (hasError
        ? {
              remote_trainer_id: remoteTrainerId,
              status: 'unreachable' as const,
              checked_at: new Date().toISOString(),
              latency_ms: null,
              devices: [],
              reason_code: 'check_failed' as const,
          }
        : undefined);
