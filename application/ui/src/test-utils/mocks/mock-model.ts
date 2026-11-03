import { SchemaModel } from '../../api/openapi-spec';

export const getMockedModel = (overrides: Partial<SchemaModel> = {}): SchemaModel => ({
    id: 'model-1',
    name: 'pick-and-place',
    path: '/models/pick-and-place',
    policy: 'act',
    properties: {},
    project_id: 'project-1',
    dataset_id: 'dataset-1',
    snapshot_id: null,
    train_job_id: null,
    parent_model_id: null,
    version: 1,
    created_at: '2026-07-14T12:00:00Z',
    available_backends: [],
    lora_enabled: false,
    lora_use_dora: false,
    snapflow_enabled: false,
    ...overrides,
});
