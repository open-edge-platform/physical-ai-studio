import { SchemaStreamDeckTarget } from '../../api/openapi-spec';
import { getMockedModelControlConfig } from './mock-model-control-config';

export const getMockedStreamDeckTarget = (overrides: Partial<SchemaStreamDeckTarget> = {}): SchemaStreamDeckTarget => ({
    session_id: 'one',
    connection_id: 'connection',
    page: 'record',
    task: 'Old task',
    inference: getMockedModelControlConfig(),
    dictation: null,
    ...overrides,
});
