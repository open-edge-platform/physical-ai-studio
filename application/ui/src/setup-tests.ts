import '@testing-library/jest-dom';

import { afterAll, afterEach } from 'vitest';

import { server } from './msw-node-setup';

// Start MSW at module-evaluation time so that globalThis.fetch is patched before
// any test-file import (e.g. src/api/client.ts) captures it via
//   `fetch: baseFetch = globalThis.fetch`
// in openapi-fetch. If we defer to beforeAll, client.ts is imported first and
// keeps a stale reference to the pre-patch fetch.
server.listen({ onUnhandledRequest: 'bypass' });

afterEach(() => {
    server.resetHandlers();
});

afterAll(() => {
    server.close();
});
