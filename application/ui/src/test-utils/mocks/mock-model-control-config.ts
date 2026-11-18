import { SchemaModelControlConfig } from '../../api/openapi-spec';

export const getMockedModelControlConfig = (
    overrides: Partial<SchemaModelControlConfig> = {}
): SchemaModelControlConfig => ({
    type: 'model',
    task: '',
    hz: 30,
    execution: { mode: 'sync', request_threshold: 0.5 },
    grace_s: 3,
    ...overrides,
});
