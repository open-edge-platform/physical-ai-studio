import { SchemaModelExportJob } from '../../api/openapi-spec';
import { getMockedTrainJob } from './mock-train-job';

export const getMockedOptimizeModelJob = (overrides: Partial<SchemaModelExportJob> = {}): SchemaModelExportJob => ({
    ...getMockedTrainJob(),
    type: 'model_export',
    status: 'pending',
    message: 'Creating INT8 export',
    payload: { model_id: 'model-1', cancellation_requested: false, backend: 'openvino', operation: 'int8_sym' },
    ...overrides,
});
