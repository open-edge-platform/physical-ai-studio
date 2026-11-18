import { SchemaStreamDeckTranscript } from '../../api/openapi-spec';

export const getMockedStreamDeckTranscript = (
    overrides: Partial<SchemaStreamDeckTranscript> = {}
): SchemaStreamDeckTranscript => ({
    id: 'dictation-one',
    task: 'Spoken task',
    ...overrides,
});
