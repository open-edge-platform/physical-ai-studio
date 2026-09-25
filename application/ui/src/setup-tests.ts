import '@testing-library/jest-dom';

import { afterAll, afterEach } from 'vitest';

import { server } from './msw-node-setup';

// jsdom has no ResizeObserver; stub it globally for components that construct
// one directly (e.g. log-content.tsx's auto-scroll-on-resize effect) instead
// of relying on a virtualized-list library to no-op it.
class ResizeObserverStub {
    observe() {}
    unobserve() {}
    disconnect() {}
}
globalThis.ResizeObserver ??= ResizeObserverStub as unknown as typeof ResizeObserver;

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
