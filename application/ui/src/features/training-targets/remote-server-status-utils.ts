import {
    SchemaCheckOutcome,
    SchemaPreflightCheck,
    SchemaPreflightTier,
    SchemaRemoteServerStatus,
} from '../../api/openapi-spec';
import { CheckState } from './remote-trainer-health-utils';

export type RemoteServerStatusVariant = 'positive' | 'notice' | 'negative' | 'neutral';

const outcomeCheckState = (outcome: SchemaCheckOutcome): CheckState => {
    switch (outcome) {
        case 'passed':
            return 'positive';
        case 'warning':
            return 'yellow';
        case 'failed':
            return 'negative';
        case 'skipped':
            return 'neutral';
    }
};

export const checkStateForCheck = (check: SchemaPreflightCheck): CheckState => outcomeCheckState(check.outcome);

export const checksForTier = (
    checks: SchemaPreflightCheck[] | undefined,
    tier: SchemaPreflightTier
): SchemaPreflightCheck[] => (checks ?? []).filter((check) => check.tier === tier);

/**
 * Rolls a status result up into one badge variant, distinguishing a transiently
 * busy GPU (reported via a WARNING on gpu_free, never blocking) from an actual
 * failure so a busy target still reads as "notice", not "negative".
 */
export const remoteServerStatusVariant = (
    status: Pick<SchemaRemoteServerStatus, 'status' | 'checks'> | undefined,
    isChecking: boolean
): RemoteServerStatusVariant => {
    if (isChecking || status === undefined) return 'neutral';
    if (status.status === 'healthy') {
        const isBusy = (status.checks ?? []).some((check) => check.key === 'gpu_free' && check.outcome === 'warning');
        return isBusy ? 'notice' : 'positive';
    }
    if (status.status === 'degraded') return 'notice';
    return 'negative';
};

export const remoteServerStatusLabel = (
    status: Pick<SchemaRemoteServerStatus, 'status' | 'checks'> | undefined,
    isChecking: boolean
): string => {
    if (isChecking) return 'Checking…';
    if (status === undefined) return 'Not checked';
    const variant = remoteServerStatusVariant(status, isChecking);
    if (variant === 'notice' && status.status === 'healthy') return 'Busy';
    if (status.status === 'healthy') return 'Healthy';
    if (status.status === 'degraded') return 'Degraded';
    return 'Unreachable';
};

export const checkLabel: Record<string, string> = {
    alias_resolved: 'SSH host alias resolves',
    reachable: 'Reachable',
    authenticated: 'Authenticated',
    host_key_verified: 'Host key verified',
    docker_usable: 'Docker available',
    disk_space: 'Storage available',
    driver_present: 'GPU driver present',
    registry_reachable: 'Registry reachable',
    gpu_free: 'GPU free',
    image_resolved: 'Image resolved & pulled',
    image_signature: 'Image signature verified',
    container_device_probe: 'Container compute probe',
    protocol_compatible: 'Trainer protocol compatible',
};

/**
 * The GPU/XPU name reported by the driver_present check (e.g. "Intel(R) Data
 * Center GPU Max 1100"), for the table's Compute column - falls back to
 * undefined when the check hasn't run or reported no detail yet.
 */
export const remoteServerComputeDetail = (
    status: Pick<SchemaRemoteServerStatus, 'checks'> | undefined
): string | undefined => (status?.checks ?? []).find((check) => check.key === 'driver_present')?.detail ?? undefined;

export const checkStatusLabel = (outcome: SchemaCheckOutcome): string => {
    switch (outcome) {
        case 'passed':
            return 'Healthy';
        case 'warning':
            return 'Busy';
        case 'failed':
            return 'Failed';
        case 'skipped':
            return 'Skipped';
    }
};
