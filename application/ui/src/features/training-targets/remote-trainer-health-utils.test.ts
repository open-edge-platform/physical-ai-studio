import { SchemaRemoteTrainerHealth } from '../../api/openapi-spec';
import { healthDescription, healthLabel, healthVariant } from './remote-trainer-health-utils';

const startingHealth: SchemaRemoteTrainerHealth = {
    remote_trainer_id: 'trainer-1',
    status: 'starting',
    checked_at: '2026-09-22T12:00:00Z',
    latency_ms: null,
    devices: [],
    storage: null,
    reason_code: 'Pulling trainer image',
};

describe('remote-trainer-health-utils starting status', () => {
    it('shows the current startup phase instead of a generic status', () => {
        expect(healthLabel(startingHealth)).toBe('Starting: Pulling trainer image');
        expect(healthLabel({ ...startingHealth, reason_code: null })).toBe('Starting');
        expect(healthLabel(undefined, true)).toBe('Checking');
    });

    it('uses a neutral (not negative) status light while starting', () => {
        expect(healthVariant(startingHealth)).toBe('neutral');
    });

    it('surfaces the in-progress launch phase as the description', () => {
        expect(healthDescription(startingHealth)).toBe('Pulling trainer image');
    });

    it('falls back to a generic starting message when no phase is reported', () => {
        expect(healthDescription({ ...startingHealth, reason_code: null })).toBe(
            'The trainer container is still starting.'
        );
    });

    it('explains an installation failure and the required reboot', () => {
        expect(
            healthDescription({ ...startingHealth, status: 'degraded', reason_code: 'intel_install_failed' })
        ).toContain('Intel GPU dependency installation failed');
        expect(healthDescription({ ...startingHealth, status: 'degraded', reason_code: 'reboot_required' })).toContain(
            'Confirm a host reboot'
        );
        expect(
            healthDescription({ ...startingHealth, status: 'degraded', reason_code: 'package_manager_broken' })
        ).toContain('incomplete dpkg transactions');
        expect(healthDescription({ ...startingHealth, status: 'degraded', reason_code: 'unsupported_os' })).toBe(
            'Host installation supports Ubuntu 24.04 or Ubuntu 26.04 only.'
        );
    });

    it('explains a remote port collision and how to resolve it', () => {
        expect(healthDescription({ ...startingHealth, status: 'degraded', reason_code: 'remote_port_in_use' })).toBe(
            'The remote port is already in use on the SSH host. Go to Settings → Training Targets, edit this trainer, and choose another remote port under Advanced settings.'
        );
    });

    it('explains missing Docker for a Studio-managed trainer', () => {
        expect(healthDescription({ ...startingHealth, status: 'degraded', reason_code: 'docker_unavailable' })).toBe(
            'Docker must be running and accessible to the SSH user. If Docker was just installed, log out of the SSH host and back in, then retry setup.'
        );
    });

    it('explains a missing accelerator driver for a Studio-managed trainer', () => {
        expect(
            healthDescription({ ...startingHealth, status: 'degraded', reason_code: 'accelerator_unavailable' })
        ).toBe('Studio-managed training requires a working CUDA or XPU driver on the SSH host.');
    });

    it('explains when the running managed container cannot access an accelerator', () => {
        expect(
            healthDescription({
                ...startingHealth,
                status: 'degraded',
                reason_code: 'container_accelerator_unavailable',
            })
        ).toBe('Studio started the trainer, but its container cannot access a CUDA or XPU device.');
    });
});
