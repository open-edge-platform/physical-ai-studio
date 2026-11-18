import { SchemaExportVariantDetail } from '../../api/openapi-spec';

export const getMockedExportVariant = (
    overrides: Partial<SchemaExportVariantDetail> = {}
): SchemaExportVariantDetail => ({
    id: 'original',
    schema_version: 1,
    weight_compression: null,
    size_bytes: 4096,
    file_count: 3,
    ...overrides,
});
