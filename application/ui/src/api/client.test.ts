import { API_BASE_URL, getClient } from './client';

describe('fetchClient.PATH', () => {
    describe('with absolute base URL', () => {
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

    describe('with default base URL (same host)', () => {
        const fetchClient = getClient({ baseUrl: API_BASE_URL });

        it('returns a relative path when no params are needed', () => {
            expect(fetchClient.PATH('/api/projects')).toBe('/api/projects');
        });

        it('substitutes path parameters into the URL', () => {
            expect(
                fetchClient.PATH('/api/projects/{project_id}/robots/{robot_id}', {
                    params: { path: { project_id: 'xxx', robot_id: 'yyy' } },
                })
            ).toBe('/api/projects/xxx/robots/yyy');
        });

        it('appends query parameters to the URL', () => {
            expect(
                fetchClient.PATH('/api/cameras/supported_formats/{driver}', {
                    params: { path: { driver: 'usb' }, query: { fingerprint: 'abc123' } },
                })
            ).toBe('/api/cameras/supported_formats/usb?fingerprint=abc123');
        });
    });

    describe('WebSocket paths', () => {
        const fetchClient = getClient({ baseUrl: API_BASE_URL });

        it('resolves /api/cameras/ws without params (query params passed separately to useWebSocket)', () => {
            expect(fetchClient.PATH('/api/cameras/ws')).toBe('/api/cameras/ws');
        });

        it('also accepts optional camera query param inline', () => {
            expect(
                fetchClient.PATH('/api/cameras/ws', {
                    params: { query: { camera: '{"driver":"usb_camera"}' } },
                })
            ).toBe('/api/cameras/ws?camera=%7B%22driver%22%3A%22usb_camera%22%7D');
        });

        it('resolves /api/jobs/ws without params', () => {
            expect(fetchClient.PATH('/api/jobs/ws')).toBe('/api/jobs/ws');
        });

        it('resolves /api/record/teleoperate/ws without params', () => {
            expect(fetchClient.PATH('/api/record/teleoperate/ws')).toBe('/api/record/teleoperate/ws');
        });

        it('resolves /api/record/inference/ws without params', () => {
            expect(fetchClient.PATH('/api/record/inference/ws')).toBe('/api/record/inference/ws');
        });

        it('resolves /api/projects/{project_id}/robots/{robot_id}/ws with params', () => {
            expect(
                fetchClient.PATH('/api/projects/{project_id}/robots/{robot_id}/ws', {
                    params: { path: { project_id: 'p1', robot_id: 'r1' } },
                })
            ).toBe('/api/projects/p1/robots/r1/ws');
        });

        it('throws when robot WebSocket path params are missing', () => {
            expect(() =>
                // @ts-expect-error missing required options argument
                fetchClient.PATH('/api/projects/{project_id}/robots/{robot_id}/ws')
            ).toThrow('Unresolved path parameters');
        });

        it('constructs absolute WebSocket-ready URLs with a base URL', () => {
            const absoluteClient = getClient({ baseUrl: 'https://geti.ai' });
            const url = absoluteClient.PATH('/api/cameras/ws');
            expect(url).toBe('https://geti.ai/api/cameras/ws');
            expect(url.replace(/^http/, 'ws')).toBe('wss://geti.ai/api/cameras/ws');
        });
    });
});
