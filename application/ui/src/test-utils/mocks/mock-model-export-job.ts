import { SchemaModelExportJob } from '../../api/openapi-spec';
import { getMockedTrainJob } from './mock-train-job';

export const getMockedModelExportJob = (overrides: Partial<SchemaModelExportJob> = {}): SchemaModelExportJob => ({
    ...getMockedTrainJob(),
    id: 'export-job-1',
    type: 'model_export',
    status: 'pending',
    progress: 0,
    payload: { model_id: 'model-1', backend: 'torch', operation: 'export', cancellation_requested: false },
    ...overrides,
});
