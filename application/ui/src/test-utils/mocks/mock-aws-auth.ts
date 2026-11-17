import { SchemaAwsAuthCapabilities } from '../../api/openapi-spec';

export const getMockedAwsAuthCapabilities = (
    overrides: Partial<SchemaAwsAuthCapabilities> = {}
): SchemaAwsAuthCapabilities => ({
    profiles: ['studio'],
    use_device_certificate: false,
    ...overrides,
});
