import { SchemaModel } from '../../api/openapi-spec';

export const getMockedModel = (overrides: Partial<SchemaModel> = {}): SchemaModel => ({
    id: 'model-1',
    name: 'pick-and-place',
    path: '/models/model-1',
    policy: 'act',
    properties: {},
    project_id: 'project-1',
    dataset_id: null,
    snapshot_id: null,
    train_job_id: 'job-1',
    parent_model_id: null,
    version: 1,
    created_at: '2026-07-14T12:00:00Z',
    available_backends: ['openvino'],
    optimization: null,
    lora_enabled: false,
    lora_use_dora: false,
    snapflow_enabled: false,
    ...overrides,
});
