import { afterAll, afterEach, expect, rs } from '@rstest/core';
import * as jestDomMatchers from '@testing-library/jest-dom/matchers';

import { server } from './msw-node-setup';

expect.extend(jestDomMatchers);

// jsdom has no ResizeObserver; stub it globally for components that construct
// one directly (e.g. log-content.tsx's auto-scroll-on-resize effect) instead
// of relying on a virtualized-list library to no-op it.
class ResizeObserverStub {
    observe() {}
    unobserve() {}
    disconnect() {}
}
globalThis.ResizeObserver ??= ResizeObserverStub as unknown as typeof ResizeObserver;

// jsdom has no canvas 2D context, which uPlot needs to draw; stub the charts out.
rs.mock('uplot', () => {
    class UPlotStub {
        static sync = () => ({ sub() {}, unsub() {}, pub() {}, plots: [] });
        setData() {}
        setSize() {}
        redraw() {}
        destroy() {}
    }

    return { default: UPlotStub };
});

// Start MSW at module-evaluation time so that globalThis.fetch is patched before
// any test-file import (e.g. src/api/client.ts) captures it via
//   `fetch: baseFetch = globalThis.fetch`
// in openapi-fetch. If we defer to beforeAll, client.ts is imported first and
// keeps a stale reference to the pre-patch fetch.
server.listen({ onUnhandledFrame: 'bypass' });

afterEach(() => {
    server.resetHandlers();
});

afterAll(() => {
    server.close();
});
