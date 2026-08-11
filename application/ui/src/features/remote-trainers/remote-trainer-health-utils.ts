import { SchemaRemoteTrainerHealth } from '../../api/openapi-spec';

export type CheckState = 'positive' | 'yellow' | 'negative' | 'neutral';

export const healthLabel = (health?: SchemaRemoteTrainerHealth, isChecking = false) => {
    if (isChecking) return 'Checking…';
    if (health === undefined) return 'Not checked';
    if (health.reason_code === 'check_failed') return 'Check failed';
    return health.status === 'healthy' ? 'Healthy' : health.status === 'degraded' ? 'Degraded' : 'Unreachable';
};

export const healthVariant = (health?: SchemaRemoteTrainerHealth, isChecking = false) => {
    if (isChecking || health === undefined) return 'neutral' as const;
    return health.status === 'healthy'
        ? ('positive' as const)
        : health.status === 'degraded'
          ? ('yellow' as const)
          : ('negative' as const);
};

export const healthDescription = (health?: SchemaRemoteTrainerHealth) => {
    if (health === undefined) return 'Connection status has not been checked.';
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
    if (isChecking || health === undefined || health.status === 'unreachable') return 'neutral';
    return (health.devices?.length ?? 0) > 0 ? 'positive' : 'yellow';
};

export const getStorageState = (health: SchemaRemoteTrainerHealth | undefined, isChecking: boolean): CheckState => {
    if (isChecking || health === undefined || health.status === 'unreachable') return 'neutral';
    return health.storage ? 'positive' : 'yellow';
};

export const trainerHealthDetail = (
    health: SchemaRemoteTrainerHealth | undefined,
    isChecking: boolean,
    deviceReportIsInvalid: boolean
) => {
    if (isChecking) return 'connection check in progress';
    if (health?.status === 'healthy' || deviceReportIsInvalid) {
        return health?.latency_ms != null
            ? `responded in ${health.latency_ms} ms and is ready for training requests`
            : 'ready for training requests';
    }
    if (health?.status === 'degraded') return 'responded with a degraded status';
    return healthDescription(health);
};

export const capabilityDetail = (health: SchemaRemoteTrainerHealth | undefined, isChecking: boolean) => {
    const devices = health?.devices ?? [];
    if (devices.length > 0) {
        return devices.map((device) => `${device.type.toUpperCase()} · ${device.name}`).join(', ');
    }
    return health === undefined || isChecking ? 'awaiting device report' : 'no compute device reported';
};

export const storageDetail = (health: SchemaRemoteTrainerHealth | undefined, isChecking: boolean) =>
    formatStorage(health?.storage) ??
    (health === undefined || isChecking ? 'awaiting storage report' : 'no storage reported');

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
