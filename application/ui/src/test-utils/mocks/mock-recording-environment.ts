import { SchemaRecordingEnvironment } from '../../api/openapi-spec';

export const getMockedRecordingEnvironment = (
    overrides: Partial<SchemaRecordingEnvironment> = {}
): SchemaRecordingEnvironment => ({
    id: 'recording-environment-1',
    dataset_id: 'dataset-1',
    environment_id: 'environment-1',
    camera_mappings: [],
    ...overrides,
});
