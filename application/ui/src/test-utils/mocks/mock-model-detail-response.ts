import { SchemaModelDetailResponse } from '../../api/openapi-spec';
import { getMockedExportVariant } from './mock-export-variant';
import { getMockedModel } from './mock-model';

export const getMockedModelDetailResponse = (
    overrides: Partial<SchemaModelDetailResponse> = {}
): SchemaModelDetailResponse => ({
    model: getMockedModel({ available_backends: ['openvino'] }),
    exports: [{ type: 'openvino', size_bytes: 4096, file_count: 3, variants: [getMockedExportVariant()] }],
    ...overrides,
});
