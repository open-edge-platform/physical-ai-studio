import { SchemaRemoteTrainerHealth } from '../../api/openapi-spec';
import { healthDescription, healthLabel, healthVariant } from './remote-trainer-health-utils';

const startingHealth: SchemaRemoteTrainerHealth = {
    remote_trainer_id: 'trainer-1',
    status: 'starting',
    checked_at: '2026-09-22T12:00:00Z',
    latency_ms: null,
    devices: [],
    storage: null,
    reason_code: 'Pulling trainer image…',
};

describe('remote-trainer-health-utils starting status', () => {
    it("labels a launching trainer 'Starting…' rather than 'Unreachable'", () => {
        expect(healthLabel(startingHealth)).toBe('Starting…');
    });

    it('uses a neutral (not negative) status light while starting', () => {
        expect(healthVariant(startingHealth)).toBe('neutral');
    });

    it('surfaces the in-progress launch phase as the description', () => {
        expect(healthDescription(startingHealth)).toBe('Pulling trainer image…');
    });

    it('falls back to a generic starting message when no phase is reported', () => {
        expect(healthDescription({ ...startingHealth, reason_code: null })).toBe(
            'The trainer container is still starting.'
        );
    });

    it('explains missing Docker for a Studio-managed trainer', () => {
        expect(healthDescription({ ...startingHealth, status: 'degraded', reason_code: 'docker_unavailable' })).toBe(
            'Studio-managed training requires Docker to be installed and running on the SSH host.'
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
