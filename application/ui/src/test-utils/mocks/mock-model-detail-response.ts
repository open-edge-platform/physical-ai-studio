import { SchemaModelDetailResponse } from '../../api/openapi-spec';
import { getMockedModel } from './mock-model';

export const getMockedModelDetailResponse = (
    overrides: Partial<SchemaModelDetailResponse> = {}
): SchemaModelDetailResponse => ({
    model: getMockedModel(),
    exports: [{ type: 'openvino', size_bytes: 1048576, file_count: 2 }],
    training_summary: null,
    hparams: null,
    ...overrides,
});
