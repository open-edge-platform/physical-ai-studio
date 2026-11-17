import { SchemaOptimizeModelJob } from '../../api/openapi-spec';
import { getMockedTrainJob } from './mock-train-job';

export const getMockedOptimizeModelJob = (overrides: Partial<SchemaOptimizeModelJob> = {}): SchemaOptimizeModelJob => ({
    ...getMockedTrainJob(),
    type: 'optimize_model',
    status: 'pending',
    message: 'Creating INT8 export',
    payload: { model_id: 'model-1', cancellation_requested: false },
    ...overrides,
});
