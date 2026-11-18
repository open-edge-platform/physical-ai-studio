import { SchemaStreamDeckStatus } from '../../api/openapi-spec';

export const getMockedStreamDeckStatus = (overrides: Partial<SchemaStreamDeckStatus> = {}): SchemaStreamDeckStatus => ({
    helper_online: true,
    can_start_native: false,
    helper_starting: false,
    connected: false,
    device: null,
    error: null,
    target: null,
    dictation_status: 'idle',
    ...overrides,
});
