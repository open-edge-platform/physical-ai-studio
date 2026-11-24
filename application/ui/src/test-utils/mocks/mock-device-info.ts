import { SchemaDeviceInfo } from '../../api/openapi-spec';

export const getMockedDeviceInfo = (overrides: Partial<SchemaDeviceInfo> = {}): SchemaDeviceInfo => ({
    type: 'cuda',
    index: 0,
    name: 'Test GPU',
    memory: 8 * 1024 ** 3,
    busy: null,
    ...overrides,
});
