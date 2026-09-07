import { getMockedTrainJobPayload } from '../../../test-utils/mocks/mock-train-job-payload';
import { getTrainerLabel } from './trainer';

describe('getTrainerLabel', () => {
    it('returns undefined when there is no payload', () => {
        expect(getTrainerLabel(undefined)).toBeUndefined();
    });

    it("returns 'Local' for a local training target", () => {
        expect(getTrainerLabel(getMockedTrainJobPayload({ training_target: 'local' }))).toBe('Local');
    });

    it("returns 'SSH' for an ssh training target", () => {
        expect(
            getTrainerLabel(getMockedTrainJobPayload({ training_target: 'ssh', remote_server_id: 'server-1' }))
        ).toBe('SSH');
    });

    it('returns the remote_trainer_name for a remote training target', () => {
        expect(
            getTrainerLabel(
                getMockedTrainJobPayload({
                    training_target: 'remote',
                    remote_trainer_id: 'trainer-1',
                    remote_trainer_name: 'managed-trainer',
                })
            )
        ).toBe('managed-trainer');
    });
});
