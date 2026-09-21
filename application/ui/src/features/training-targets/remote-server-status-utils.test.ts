import { SchemaPreflightCheck } from '../../api/openapi-spec';
import { remoteServerStatusLabel, remoteServerStatusVariant } from './remote-server-status-utils';

const check = (overrides: Partial<SchemaPreflightCheck>): SchemaPreflightCheck => ({
    key: 'reachable',
    tier: 1,
    outcome: 'passed',
    blocking: true,
    checked_at: '2026-08-07T12:00:00Z',
    ...overrides,
});

describe('remoteServerStatusVariant/Label', () => {
    it('reads a degraded status with a failed connectivity check as unreachable', () => {
        const status = { status: 'degraded', checks: [check({ key: 'reachable', outcome: 'failed' })] };

        expect(remoteServerStatusVariant(status, false)).toBe('negative');
        expect(remoteServerStatusLabel(status, false)).toBe('Unreachable');
    });

    it('reads a degraded status with only a non-connectivity check failure as degraded', () => {
        const status = { status: 'degraded', checks: [check({ key: 'disk_space', outcome: 'failed' })] };

        expect(remoteServerStatusVariant(status, false)).toBe('notice');
        expect(remoteServerStatusLabel(status, false)).toBe('Degraded');
    });

    it('reads a healthy status with a busy GPU as notice/Busy', () => {
        const status = { status: 'healthy', checks: [check({ key: 'gpu_free', outcome: 'warning' })] };

        expect(remoteServerStatusVariant(status, false)).toBe('notice');
        expect(remoteServerStatusLabel(status, false)).toBe('Busy');
    });

    it('reads a healthy status with no warnings as positive/Healthy', () => {
        const status = { status: 'healthy', checks: [] };

        expect(remoteServerStatusVariant(status, false)).toBe('positive');
        expect(remoteServerStatusLabel(status, false)).toBe('Healthy');
    });
});
