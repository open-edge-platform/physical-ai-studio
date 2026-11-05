import { SchemaAwsAuthCapabilities } from '../../api/openapi-spec';

export const getMockedAwsAuthCapabilities = (
    overrides: Partial<SchemaAwsAuthCapabilities> = {}
): SchemaAwsAuthCapabilities => ({
    profiles: ['studio'],
    ...overrides,
});
