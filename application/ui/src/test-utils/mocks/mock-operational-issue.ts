import { SchemaOperationalIssue } from '../../api/openapi-spec';

export const getMockedOperationalIssue = (overrides: Partial<SchemaOperationalIssue> = {}): SchemaOperationalIssue => ({
    id: 'issue-1',
    title: 'Action required',
    detail: 'Restore access on the backend host.',
    ...overrides,
});
