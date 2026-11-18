import { SchemaBackendExportDetail, SchemaModelDetailResponse } from '../../api/openapi-spec';
import { getMockedExportVariant } from './mock-export-variant';
import { getMockedModel } from './mock-model';

export const getMockedBackendExportDetail = (
    overrides: Partial<SchemaBackendExportDetail> = {}
): SchemaBackendExportDetail => {
    const defaults = { type: 'torch', size_bytes: 1024, file_count: 1, variants: [getMockedExportVariant()] };
    return { ...defaults, ...overrides };
};

export const getMockedModelDetail = (
    overrides: Partial<SchemaModelDetailResponse> = {}
): SchemaModelDetailResponse => ({
    model: getMockedModel(),
    exports: [],
    available_actions: [
        { backend: 'torch', operation: 'export' },
        { backend: 'onnx', operation: 'export' },
        { backend: 'openvino', operation: 'export' },
    ],
    ...overrides,
});
