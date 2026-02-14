import { getClient } from './client';

describe('fetchClient.PATH', () => {
    const fetchClient = getClient({ baseUrl: 'https://geti.ai' });

    describe('runtime behavior', () => {
        it('returns the base URL joined with the path when no params are needed', () => {
            expect(fetchClient.PATH('/api/projects')).toBe('https://geti.ai/api/projects');
        });

        it('substitutes path parameters into the URL', () => {
            expect(
                fetchClient.PATH('/api/projects/{project_id}/robots/{robot_id}', {
                    params: { path: { project_id: 'xxx', robot_id: 'yyy' } },
                })
            ).toBe('https://geti.ai/api/projects/xxx/robots/yyy');
        });

        it('appends query parameters to the URL', () => {
            expect(
                fetchClient.PATH('/api/cameras/supported_formats/{driver}', {
                    params: { path: { driver: 'usb' }, query: { fingerprint: 'abc123' } },
                })
            ).toBe('https://geti.ai/api/cameras/supported_formats/usb?fingerprint=abc123');
        });

        it('throws when path parameters are missing', () => {
            expect(() =>
                fetchClient.PATH('/api/projects/{project_id}/robots/{robot_id}', {
                    // @ts-expect-error missing required project_id
                    params: { path: { robot_id: 'yyy' } },
                })
            ).toThrow('Unresolved path parameters in "/api/projects/{project_id}/robots/{robot_id}": {project_id}');
        });

        it('throws when params object is empty but path has required params', () => {
            expect(() =>
                fetchClient.PATH('/api/projects/{project_id}/robots/{robot_id}', {
                    // @ts-expect-error missing required path
                    params: {},
                })
            ).toThrow('Unresolved path parameters');
        });
    });

    describe('type safety', () => {
        it('does not require options for paths without required parameters', () => {
            fetchClient.PATH('/api/projects');
        });

        it('requires options with path params for paths that have required parameters', () => {
            expect(() =>
                // @ts-expect-error missing required options argument
                fetchClient.PATH('/api/projects/{project_id}/robots/{robot_id}')
            ).toThrow('Unresolved path parameters');
        });

        it('requires params when options are provided for paths with required parameters', () => {
            expect(() =>
                // @ts-expect-error empty options object — missing required params
                fetchClient.PATH('/api/projects/{project_id}/robots/{robot_id}', {})
            ).toThrow('Unresolved path parameters');
        });

        it('rejects invalid path parameter types (compile-time only, coerced at runtime)', () => {
            const result = fetchClient.PATH('/api/projects/{project_id}/robots/{robot_id}', {
                // @ts-expect-error project_id must be a string, not a number
                params: { path: { project_id: 123, robot_id: 'yyy' } },
            });
            // At runtime the number is coerced to a string by the path serializer
            expect(result).toBe('https://geti.ai/api/projects/123/robots/yyy');
        });

        it('rejects unknown path keys (compile-time only)', () => {
            // At runtime the path has no placeholders so it succeeds
            // @ts-expect-error path does not exist in the spec
            const result = fetchClient.PATH('/api/this/does/not/exist');
            expect(result).toBe('https://geti.ai/api/this/does/not/exist');
        });
    });
});
