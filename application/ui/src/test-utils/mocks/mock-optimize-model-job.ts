import { SchemaOptimizeModelJob } from '../../api/openapi-spec';

export const getMockedOptimizeModelJob = (overrides: Partial<SchemaOptimizeModelJob> = {}): SchemaOptimizeModelJob => ({
    id: 'optimization-1',
    project_id: 'project-1',
    type: 'optimize_model',
    status: 'pending',
    progress: 0,
    message: 'Queued for optimization',
    created_at: '2026-07-14T13:00:00Z',
    payload: { model_id: 'model-1', mode: 'int8_sym' },
    ...overrides,
});
