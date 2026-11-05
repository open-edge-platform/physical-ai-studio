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

    it('does not emit a false recovery when a stream disconnects', async () => {
        const onIssues = rs.fn();
        const stream = fetchSSE('/metrics', { onIssues })[Symbol.asyncIterator]();
        const next = stream.next();
        TestEventSource.current.onerror?.();
        await expect(next).rejects.toThrow('EventSource connection failed');
        expect(onIssues).not.toHaveBeenCalled();
    });
});
