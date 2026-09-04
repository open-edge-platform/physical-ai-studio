import { SchemaTrainJob } from '../../../api/openapi-spec';

type TrainJobPayload = SchemaTrainJob['payload'];

export const getTrainerLabel = (payload: TrainJobPayload | undefined): string | undefined => {
    if (payload === undefined) {
        return undefined;
    }

    if (payload.training_target === 'remote') {
        return payload.remote_trainer_name ?? 'Remote';
    }

    if (payload.training_target === 'local') {
        return 'Local';
    }

    if (payload.training_target === 'ssh') {
        return 'SSH';
    }

    console.error('Unhandled training_target', payload satisfies never);
};
