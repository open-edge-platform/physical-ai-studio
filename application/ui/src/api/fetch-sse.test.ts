import { rs } from '@rstest/core';

import { getMockedOperationalIssue } from '../test-utils/mocks/mock-operational-issue';
import { fetchSSE } from './fetch-sse';

class TestEventSource extends EventTarget {
    static current: TestEventSource;
    onmessage: ((event: { data: string }) => void) | null = null;
    onerror: (() => void) | null = null;
    close = rs.fn();

    constructor() {
        super();
        TestEventSource.current = this;
    }
}

describe('SSE issues', () => {
    beforeEach(() => {
        rs.stubGlobal('EventSource', TestEventSource);
    });
    afterEach(() => {
        rs.unstubAllGlobals();
    });

    it('delivers issue snapshots separately from metric rows, including recovery after reconnect', async () => {
        const onIssues = rs.fn();
        for (const issues of [[getMockedOperationalIssue()], []]) {
            const stream = fetchSSE('/metrics', { onIssues })[Symbol.asyncIterator]();
            const next = stream.next();
            const source = TestEventSource.current;
            source.dispatchEvent(new MessageEvent('issues', { data: JSON.stringify(issues) }));
            expect(onIssues).toHaveBeenLastCalledWith(JSON.stringify(issues));
            source.onmessage?.({ data: '{"step":1}' });
            expect(await next).toEqual({ value: { step: 1 }, done: false });
            const done = stream.next();
            source.onmessage?.({ data: 'DONE' });
            await done;
            expect(source.close).toHaveBeenCalled();
        }
    });

    it('delivers every row in a burst, including rows received while the consumer is paused', async () => {
        const stream = fetchSSE('/metrics')[Symbol.asyncIterator]();
        const first = stream.next();
        const source = TestEventSource.current;
        source.onmessage?.({ data: '{"step":1}' });
        source.onmessage?.({ data: '{"step":2}' });
        expect(await first).toEqual({ value: { step: 1 }, done: false });
        source.onmessage?.({ data: 'invalid JSON' });
        source.onmessage?.({ data: '{"step":3}' });
        source.onmessage?.({ data: 'DONE' });
        expect(await stream.next()).toEqual({ value: { step: 2 }, done: false });
        expect(await stream.next()).toEqual({ value: { step: 3 }, done: false });
        expect(await stream.next()).toEqual({ value: undefined, done: true });
        expect(source.close).toHaveBeenCalled();
    });

    it('flushes buffered rows before reporting a connection error', async () => {
        const stream = fetchSSE('/metrics')[Symbol.asyncIterator]();
        const first = stream.next();
        const source = TestEventSource.current;
        source.onmessage?.({ data: '{"step":1}' });
        source.onmessage?.({ data: '{"step":2}' });
        source.onerror?.();
        expect(await first).toEqual({ value: { step: 1 }, done: false });
        expect(await stream.next()).toEqual({ value: { step: 2 }, done: false });
        await expect(stream.next()).rejects.toThrow('EventSource connection failed');
        expect(source.close).toHaveBeenCalled();
    });

    it('closes a waiting stream when canceled', async () => {
        const controller = new AbortController();
        const stream = fetchSSE('/metrics', { signal: controller.signal })[Symbol.asyncIterator]();
        const next = stream.next();
        controller.abort();
        expect(await next).toEqual({ value: undefined, done: true });
        expect(TestEventSource.current.close).toHaveBeenCalled();
    });

    it('does not emit a false recovery when a stream disconnects', async () => {
        const onIssues = rs.fn();
        const stream = fetchSSE('/metrics', { onIssues })[Symbol.asyncIterator]();
        const next = stream.next();
        TestEventSource.current.onerror?.();
        await expect(next).rejects.toThrow('EventSource connection failed');
        expect(onIssues).not.toHaveBeenCalled();
    });
});
